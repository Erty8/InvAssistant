"""Classify a filer's recent 8-K events from SEC submissions metadata.

Every 8-K a company files is tagged by SEC with one or more structured
"item numbers" (e.g. ``5.02`` for an officer/director departure, ``4.02``
for a restatement of previously issued financial statements). Those item
codes travel in the same ``submissions`` document this project already
fetches and caches for the next-earnings estimate
(:func:`sec_analyzer.fetch.filings.estimate_next_earnings`), in the
parallel array ``filings.recent.items``.

That means a genuinely useful *qualitative* signal -- "did anything material
happen to this company recently?" -- is available with **zero** extra network
requests, zero document parsing, and no LLM: we only classify the item codes
SEC has already assigned. This module is the event-oriented sibling of
:mod:`sec_analyzer.normalize.red_flags`: each recognized item code maps to a
Turkish category label and a severity, and :func:`detect_events` returns the
recent filings that carry them.

Like ``detect_red_flags``, this module is fully defensive: malformed or
missing submissions data simply yields no events, and :func:`detect_events`
never raises.

Item-code reference: SEC Form 8-K, current item list
(https://www.sec.gov/fast-answers/answersform8khtm.html). Codes not in the
map below are still surfaced (so a newly-added item is never silently
dropped) but classified as ``"info"`` with a generic label.
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Date format used throughout SEC submissions data for filing dates.
_DATE_FMT = "%Y-%m-%d"

#: Severity ranking, high to low. Used both to compute an event's overall
#: severity (the max over its item codes) and to filter by ``min_severity``.
_SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}

#: Default lookback window: only 8-Ks filed within this many days of today
#: are considered "recent" enough to surface.
_DEFAULT_LOOKBACK_DAYS = 180

#: SEC Form 8-K item code -> (Turkish label, severity). Severity buckets:
#:   critical -- bankruptcy, delisting, restatement, control change, debt
#:               acceleration: an event that can invalidate the numbers or
#:               the going-concern assumption the valuation rests on.
#:   warning  -- material agreements, acquisitions/disposals, auditor change,
#:               officer/board departures, impairments, restructuring: worth
#:               a human's attention before trusting the model.
#:   info     -- routine disclosures (earnings releases, votes, Reg FD, other
#:               events, exhibits) that carry little standalone signal.
_ITEM_MAP: Dict[str, Tuple[str, str]] = {
    # 1.xx -- Registrant's business and operations
    "1.01": ("Materyal sözleşme imzalandı", "warning"),
    "1.02": ("Materyal sözleşme sona erdi", "warning"),
    "1.03": ("İflas veya kayyum atanması", "critical"),
    "1.04": ("Maden güvenliği bildirimi", "info"),
    "1.05": ("Materyal siber güvenlik olayı", "critical"),
    # 2.xx -- Financial information
    "2.01": ("Varlık alımı/satışı tamamlandı", "warning"),
    "2.02": ("Kazanç/finansal sonuç açıklaması", "info"),
    "2.03": ("Yeni doğrudan finansal yükümlülük", "warning"),
    "2.04": ("Borç hızlandırma/covenant tetikleyici", "critical"),
    "2.05": ("Yeniden yapılanma / çıkış maliyetleri", "warning"),
    "2.06": ("Materyal değer düşüklüğü (impairment)", "warning"),
    # 3.xx -- Securities and trading markets
    "3.01": ("Borsadan çıkarma / kotasyon uyarısı", "critical"),
    "3.02": ("Kayıtsız hisse satışı", "info"),
    "3.03": ("Menkul kıymet haklarında materyal değişiklik", "warning"),
    # 4.xx -- Matters related to accountants and financial statements
    "4.01": ("Bağımsız denetçi değişikliği", "warning"),
    "4.02": ("Önceki finansal tablolara güvenilemez (restatement)", "critical"),
    # 5.xx -- Corporate governance and management
    "5.01": ("Şirket kontrolünde değişiklik", "critical"),
    "5.02": ("Üst düzey yönetici/kurul değişikliği", "warning"),
    "5.03": ("Ana sözleşme/tüzük veya mali yıl değişikliği", "info"),
    "5.04": ("Çalışan planlarında geçici işlem durdurma", "info"),
    "5.05": ("Etik kuralları değişikliği", "info"),
    "5.06": ("Kabuk şirket statüsünde değişiklik", "warning"),
    "5.07": ("Genel kurul oylama sonuçları", "info"),
    "5.08": ("Hissedar yönetim adaylıkları", "info"),
    # 6.xx -- Asset-backed securities (niche)
    "6.01": ("ABS bilgilendirme", "info"),
    "6.02": ("Servis sağlayıcı/varlık değişikliği (ABS)", "info"),
    "6.03": ("Kredi geliştirme değişikliği (ABS)", "info"),
    "6.04": ("Menkul kıymet yükümlülüklerinde başarısızlık (ABS)", "warning"),
    "6.05": ("Menkul kıymetleştirme derecelendirme değişikliği (ABS)", "info"),
    # 7.xx / 8.xx -- Regulation FD and other events
    "7.01": ("Regülasyon FD açıklaması", "info"),
    "8.01": ("Diğer olaylar", "info"),
    # 9.xx -- Financial statements and exhibits
    "9.01": ("Finansal tablolar ve ekler", "info"),
}

#: Fallback classification for an item code SEC assigns that isn't in
#: :data:`_ITEM_MAP` (e.g. a future item number). Surfaced, never dropped.
_UNKNOWN_SEVERITY = "info"

#: 8-K form types that carry item codes (the base form and its amendment).
_EIGHTK_FORMS = ("8-K", "8-K/A")


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse a ``YYYY-MM-DD`` date string, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _DATE_FMT).date()
    except (ValueError, TypeError):
        return None


def _split_items(raw: Optional[str]) -> List[str]:
    """Split a submissions ``items`` cell (e.g. ``"1.01,5.02,9.01"``) into a
    list of trimmed item codes, dropping blanks. Non-string input yields an
    empty list."""
    if not isinstance(raw, str):
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _classify_item(code: str) -> Tuple[str, str]:
    """Return ``(label, severity)`` for one item code, falling back to a
    generic label at :data:`_UNKNOWN_SEVERITY` for an unrecognized code."""
    mapped = _ITEM_MAP.get(code)
    if mapped is not None:
        return mapped
    return (f"8-K madde {code}", _UNKNOWN_SEVERITY)


def _max_severity(severities: List[str]) -> str:
    """Return the highest-ranked severity in ``severities`` (``"info"`` if
    empty or all-unknown)."""
    best = "info"
    best_rank = _SEVERITY_RANK["info"]
    for sev in severities:
        rank = _SEVERITY_RANK.get(sev, 0)
        if rank > best_rank:
            best, best_rank = sev, rank
    return best


def detect_events(
    submissions: Optional[dict],
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    min_severity: str = "info",
    max_events: Optional[int] = None,
    today: Optional[date] = None,
) -> List[dict]:
    """Classify a filer's recent 8-K events from its submissions history.

    Args:
        submissions: The dict returned by
            :func:`sec_analyzer.fetch.companyfacts.get_submissions`, or any
            dict with the same ``filings.recent.{form,filingDate,items,
            accessionNumber,primaryDocument}`` parallel-array shape.
        lookback_days: Only 8-Ks filed within this many days of ``today``
            are considered. Non-positive values disable the window (all
            available 8-Ks qualify).
        min_severity: Drop events whose overall severity ranks below this
            (``"info"`` keeps everything; ``"warning"`` keeps warning +
            critical; ``"critical"`` keeps only critical).
        max_events: If set, keep at most this many events (most recent
            first) *after* severity filtering.
        today: Reference date for the lookback window; defaults to
            :meth:`date.today`. Exposed for deterministic testing.

    Returns:
        A list of event dicts, most recent first::

            {
              "date": "2026-07-01",        # filing date, YYYY-MM-DD
              "form": "8-K",
              "items": ["5.02"],           # SEC item codes on this filing
              "categories": ["Üst düzey yönetici/kurul değişikliği"],
              "severity": "warning",       # max over the filing's items
              "accession": "0000002488-26-000115",
              "primary_doc": "amd-20260626.htm",
            }

        Empty list if ``submissions`` is missing/malformed or nothing
        qualifies. Never raises.
    """
    try:
        return _detect_events(
            submissions or {},
            lookback_days=lookback_days,
            min_severity=min_severity,
            max_events=max_events,
            today=today or date.today(),
        )
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("detect_events() failed unexpectedly; returning no events.")
        return []


def _detect_events(
    submissions: dict,
    lookback_days: int,
    min_severity: str,
    max_events: Optional[int],
    today: date,
) -> List[dict]:
    recent = ((submissions.get("filings") or {}).get("recent")) or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    items_col = recent.get("items") or []
    accessions = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []

    min_rank = _SEVERITY_RANK.get(min_severity, _SEVERITY_RANK["info"])
    events: List[dict] = []

    for idx, form in enumerate(forms):
        if form not in _EIGHTK_FORMS:
            continue

        filed = _parse_date(filing_dates[idx] if idx < len(filing_dates) else None)
        if filed is None:
            continue
        if lookback_days > 0 and (today - filed).days > lookback_days:
            continue

        raw_items = items_col[idx] if idx < len(items_col) else None
        codes = _split_items(raw_items)
        if not codes:
            # An 8-K with no item codes carries no classifiable signal.
            continue

        categories: List[str] = []
        severities: List[str] = []
        for code in codes:
            label, severity = _classify_item(code)
            if label not in categories:
                categories.append(label)
            severities.append(severity)

        severity = _max_severity(severities)
        if _SEVERITY_RANK.get(severity, 0) < min_rank:
            continue

        events.append(
            {
                "date": filed.strftime(_DATE_FMT),
                "form": form,
                "items": codes,
                "categories": categories,
                "severity": severity,
                "accession": accessions[idx] if idx < len(accessions) else None,
                "primary_doc": primary_docs[idx] if idx < len(primary_docs) else None,
            }
        )

    # Most recent first. Filing dates in submissions are already newest-first,
    # but sort explicitly so we don't depend on that ordering.
    events.sort(key=lambda e: e["date"], reverse=True)

    if max_events is not None and max_events >= 0:
        events = events[:max_events]

    return events


def summarize_events(events: List[dict], max_shown: int = 3) -> str:
    """Render a compact one-line Turkish summary of the most material events,
    suitable for the verdict card's future "Olaylar:" line.

    Leads with a severity tally (e.g. ``"1 kritik, 2 uyarı"``) and then lists
    up to ``max_shown`` events (most severe first, then most recent) as
    ``"<ilk kategori> <gün Ay>"``. Returns ``"yok"`` for an empty list.

    This helper is pure and does not touch any other module; it exists so the
    eventual CLI/report integration has a single, tested formatter to call.
    """
    if not events:
        return "yok"

    counts = {"critical": 0, "warning": 0, "info": 0}
    for event in events:
        counts[event.get("severity", "info")] = counts.get(event.get("severity", "info"), 0) + 1

    tally_parts = []
    for sev, tr in (("critical", "kritik"), ("warning", "uyarı"), ("info", "bilgi")):
        if counts.get(sev):
            tally_parts.append(f"{counts[sev]} {tr}")
    tally = ", ".join(tally_parts)

    ranked = sorted(
        events,
        key=lambda e: (_SEVERITY_RANK.get(e.get("severity", "info"), 0), e.get("date", "")),
        reverse=True,
    )
    shown = []
    for event in ranked[:max_shown]:
        category = (event.get("categories") or ["olay"])[0]
        shown.append(f"{category} ({event.get('date', '')})")

    return f"{tally} — " + " · ".join(shown) if shown else tally
