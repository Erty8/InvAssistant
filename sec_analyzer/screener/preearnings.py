"""Pre-earnings briefing screener.

For a watchlist of tickers, finds the names whose next earnings release
(per :func:`sec_analyzer.fetch.filings.estimate_next_earnings`) falls within
a lookahead window and assembles a one-screen briefing per name: the stored
verdict (fair value vs. price), how stale that opinion is, the historical
beat/miss record against consensus, and any recent material 8-K events.

This is a **read/assemble** layer only: it runs no valuation and computes no
new fair value. Every numeric field either comes straight from the most
recent stored *live* verdict (:func:`sec_analyzer.store.database.load_verdicts`)
or is simple arithmetic over it. The structural precedent for this module is
:mod:`sec_analyzer.screener.swing_scan` -- same result-dict shape, same
``generated_at`` provenance stamp, same "one bad ticker never fails the
whole scan" posture.

Determinism: the only wall-clock value anywhere in the result is
``generated_at`` (UTC, for provenance/display only). Every date-dependent
computation -- the earnings-window filter, verdict staleness, "just
reported" -- is driven by the ``today`` parameter, which defaults to
:meth:`date.today` but can be pinned for tests and as-of runs.
"""

import logging
import math
from datetime import date, datetime, timezone
from typing import Callable, List, Optional

from sec_analyzer.fetch.companyfacts import get_submissions
from sec_analyzer.fetch.earnings import get_earnings_history
from sec_analyzer.fetch.filings import estimate_next_earnings
from sec_analyzer.fetch.tickers import resolve_cik
from sec_analyzer.http_client import SecHttpClient
from sec_analyzer.signals.events import detect_events
from sec_analyzer.store.database import load_verdicts

logger = logging.getLogger(__name__)

#: Default lookahead window (days) for "reports soon" (SPEC per the feature
#: brief). Callers may widen/narrow via ``scan_preearnings(within_days=...)``.
DEFAULT_WITHIN_DAYS = 14

#: A stored verdict older than this many days is flagged ``stale_verdict``:
#: for a name about to report, an opinion this old was formed on data/price
#: levels that may no longer hold.
DEFAULT_STALE_DAYS = 45

#: Material 8-K lookback window shown in the briefing (independent of the
#: earnings-window filter above -- this looks BACKWARD for recent filings,
#: not forward for the next release).
_EVENTS_LOOKBACK_DAYS = 120

#: Most recent quarters of beat/miss history kept per name.
_MAX_SURPRISE_QUARTERS = 4

#: Momentum-label -> sign, used by the value x momentum note rules below.
#: These are the exact composite momentum labels this codebase produces
#: (see ``sec_analyzer.signals.momentum``); anything else is unrecognized
#: and contributes no note.
_MOMENTUM_SIGN = {
    "STRONG+": 1,
    "POSITIVE": 1,
    "NEUTRAL": 0,
    "NEGATIVE": -1,
}

#: Marker filings.py's label always embeds right after the last-report date
#: when ``recently_reported`` is True (see ``fetch/filings.py::_build_result``)
#: -- used to pull that date back out for the "just reported" note without
#: duplicating the date-formatting logic here. NOTE: ``fetch/filings.py`` is
#: outside this module's ownership and still renders this marker (and the
#: date before it) in Turkish; left untranslated here so the substring match
#: keeps working until that module is translated separately.
_JUST_REPORTED_MARKER = " tarihinde açıklandı"


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_surprise_pct(value: Optional[float]) -> str:
    """Render a signed EPS-surprise percentage.

    Args:
        value: A surprise percentage (e.g. ``8.4`` for a beat, ``-2.1`` for
            a miss), or ``None``.

    Returns:
        ``"+8.4%"`` / ``"-2.1%"`` for a usable number, or ``"—"`` (em-dash)
        for ``None``/non-finite input -- never the strings "None" or "nan".
    """
    if not _is_num(value):
        return "—"
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return "—"
    sign = "+" if v >= 0 else "-"
    formatted = f"{abs(v):.1f}"
    return f"{sign}{formatted}%"


def _parse_analyzed_at(value: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of a stored ``analyzed_at`` ISO timestamp."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _verdict_age_days(analyzed_at: Optional[str], today: date) -> Optional[int]:
    """Days between a stored verdict's ``analyzed_at`` and ``today``, or
    ``None`` if ``analyzed_at`` is missing/unparseable."""
    parsed = _parse_analyzed_at(analyzed_at)
    if parsed is None:
        return None
    return (today - parsed.date()).days


def _extract_just_reported_date(catalyst_label: Optional[str]) -> str:
    """Pull the already-formatted report date back out of a
    ``recently_reported`` catalyst label (``"29 Tem tarihinde açıklandı ·
    sonraki: ..."``) for reuse in the "just reported" note, without
    duplicating filings.py's date-formatting logic here. The date text
    itself may still render in filings.py's Turkish locale (see
    ``_JUST_REPORTED_MARKER``'s note) until that module is translated
    separately."""
    label = catalyst_label or ""
    idx = label.find(_JUST_REPORTED_MARKER)
    if idx == -1:
        return "Earnings"
    return label[:idx]


def _build_notes(row: dict) -> List[str]:
    """Assemble the deterministic observation list for one row.

    Each rule below contributes at most one sentence, in the fixed order
    given in the feature spec. Pure function of ``row`` -- no I/O, no
    wall-clock reads -- so each rule is independently unit-testable with a
    minimal fixture dict.
    """
    notes: List[str] = []

    verdict_date = row.get("verdict_date")
    verdict_age_days = row.get("verdict_age_days")
    verdict = row.get("verdict")
    momentum = row.get("momentum_verdict")
    surprises = row.get("surprises") or []
    events = row.get("events") or []

    # Rule 1: no stored opinion at all -- ``verdict_date`` is only ever set
    # when a live verdict row exists, so its absence is the signal.
    if verdict_date is None:
        notes.append("No analysis on file for this name — run one before earnings.")

    # Rule 2: a stored opinion exists but is old enough that price/data may
    # have moved past it. Only meaningful when a verdict exists at all (rule
    # 1 already covers the no-verdict case), so this is naturally exclusive
    # of it since verdict_age_days is None with no verdict.
    if _is_num(verdict_age_days) and verdict_age_days > DEFAULT_STALE_DAYS:
        notes.append(f"Stored analysis is {verdict_age_days} days old — refresh before earnings.")

    # Rule 3: the quarter already came out -- the upcoming catalyst is the
    # NEXT one, not the report just published.
    if row.get("just_reported"):
        reported_date = _extract_just_reported_date(row.get("catalyst_label"))
        notes.append(f"Reported on {reported_date}; the next catalyst is the following quarter.")

    # Rule 4: earnings is imminent (and hasn't just happened) -- a timing
    # risk worth calling out for anyone considering a new position now.
    days_until = row.get("days_until")
    if not row.get("just_reported") and _is_num(days_until) and days_until <= 3:
        notes.append(f"Earnings in {days_until} days — timing risk is high for opening a new position.")

    # Rule 5: value x momentum cross -- only meaningful when both the
    # fundamental verdict and the momentum label are known and recognized.
    momentum_sign = _MOMENTUM_SIGN.get(momentum)
    if verdict is not None and momentum_sign is not None:
        if verdict == "CHEAP" and momentum_sign < 0:
            notes.append("The model says CHEAP but momentum is negative; earnings could be the catalyst that resolves this divergence.")
        elif verdict == "CHEAP" and momentum_sign > 0:
            notes.append("A CHEAP valuation lines up with positive momentum — earnings could be confirmatory.")
        elif verdict == "EXPENSIVE" and momentum_sign > 0:
            notes.append("An EXPENSIVE valuation is being carried by momentum; earnings is a potential inflection point.")

    # Rule 6: beat/miss record, only shown with a reasonably sized sample.
    if len(surprises) >= 3:
        beat_count = row.get("beat_count") or 0
        if beat_count == 0:
            notes.append(f"Missed consensus in all of the last {len(surprises)} quarters.")
        else:
            avg_txt = format_surprise_pct(row.get("avg_surprise_pct"))
            notes.append(f"Beat consensus {beat_count} times in the last {len(surprises)} quarters (avg. {avg_txt}).")

    # Rule 7: material recent filing activity.
    if events:
        first_category = (events[0].get("categories") or ["event"])[0]
        notes.append(f"Material filing event in the last {_EVENTS_LOOKBACK_DAYS} days: {first_category}.")

    # Rule 8: low confidence in the stored valuation -- a band that may move
    # once the print lands.
    if row.get("confidence") == "LOW":
        notes.append("Valuation confidence is LOW; the band may shift after earnings.")

    return notes


def _build_row(
    ticker: str,
    cik: str,
    name: Optional[str],
    catalyst: dict,
    verdict: Optional[dict],
    quarters: List[dict],
    events: List[dict],
    today: date,
) -> dict:
    """Assemble one briefing row from its already-fetched ingredients."""
    just_reported = bool(catalyst.get("recently_reported"))
    days_until = catalyst.get("days_until")

    verdict_date = None
    verdict_age_days = None
    fundamental_verdict = None
    technical_verdict = None
    momentum_verdict = None
    confidence = None
    sector_type = None
    horizon = None
    price = None
    fv_base_lo = None
    fv_base_hi = None

    if verdict is not None:
        verdict_date = verdict.get("analyzed_at")
        verdict_age_days = _verdict_age_days(verdict_date, today)
        fundamental_verdict = verdict.get("fundamental_verdict")
        technical_verdict = verdict.get("technical_verdict")
        momentum_verdict = verdict.get("momentum_verdict")
        confidence = verdict.get("confidence")
        sector_type = verdict.get("sector_type")
        horizon = verdict.get("horizon")
        price = verdict.get("price")
        fv_base_lo = verdict.get("fv_base_lo")
        fv_base_hi = verdict.get("fv_base_hi")

    # Stale when there's no opinion at all, or the one on file predates the
    # freshness window.
    stale_verdict = verdict is None or (_is_num(verdict_age_days) and verdict_age_days > DEFAULT_STALE_DAYS)

    fv_base_mid = None
    if _is_num(fv_base_lo) and _is_num(fv_base_hi):
        fv_base_mid = (float(fv_base_lo) + float(fv_base_hi)) / 2.0

    fv_vs_price_pct = None
    if _is_num(fv_base_mid) and _is_num(price) and price > 0:
        fv_vs_price_pct = round((fv_base_mid / float(price) - 1.0) * 100.0, 1)

    # Beat/miss tally over the usable (non-None surprise_pct) quarters kept.
    usable_surprises = [q.get("surprise_pct") for q in quarters if _is_num(q.get("surprise_pct"))]
    beat_count = sum(1 for s in usable_surprises if s > 0)
    miss_count = sum(1 for s in usable_surprises if s <= 0)

    beat_streak = 0
    for q in quarters:  # newest first
        sp = q.get("surprise_pct")
        if _is_num(sp) and sp > 0:
            beat_streak += 1
        else:
            break

    avg_surprise_pct = round(sum(usable_surprises) / len(usable_surprises), 1) if usable_surprises else None

    row = {
        "ticker": ticker,
        "name": name,
        "cik": cik,
        "earnings_date": catalyst.get("estimate_date"),
        "days_until": days_until,
        "catalyst_label": catalyst.get("label"),
        "catalyst_source": catalyst.get("source"),
        "catalyst_basis": catalyst.get("based_on"),
        "just_reported": just_reported,
        "verdict": fundamental_verdict,
        "technical_verdict": technical_verdict,
        "momentum_verdict": momentum_verdict,
        "confidence": confidence,
        "sector_type": sector_type,
        "horizon": horizon,
        "verdict_date": verdict_date,
        "verdict_age_days": verdict_age_days,
        "stale_verdict": stale_verdict,
        "price": price,
        "fv_base_lo": fv_base_lo,
        "fv_base_hi": fv_base_hi,
        "fv_base_mid": fv_base_mid,
        "fv_vs_price_pct": fv_vs_price_pct,
        "surprises": quarters,
        "beat_count": beat_count,
        "miss_count": miss_count,
        "beat_streak": beat_streak,
        "avg_surprise_pct": avg_surprise_pct,
        "events": events,
        "notes": [],
    }
    row["notes"] = _build_notes(row)
    return row


def _process_ticker(
    ticker: str,
    within_days: int,
    no_cache: bool,
    today: date,
    db_path: Optional[str],
    include_all: bool,
    client: SecHttpClient,
) -> "tuple[str, object]":
    """Run the full per-ticker flow.

    Returns:
        ``("row", row_dict)`` on success, or ``("skip", reason_str)`` when
        the ticker is deliberately excluded (no usable earnings estimate, or
        outside the window and ``include_all`` is False).

    Raises:
        Exception: Any failure fetching CIK/submissions is left to propagate
            -- the caller converts it into a ``skipped`` entry.
    """
    cik, name = resolve_cik(ticker, client, no_cache=no_cache)
    submissions = get_submissions(cik, client, no_cache=no_cache)

    catalyst = estimate_next_earnings(submissions, today=today)
    if catalyst is None:
        return "skip", "could not estimate earnings date"

    days_until = catalyst.get("days_until")
    just_reported = bool(catalyst.get("recently_reported"))

    if not include_all:
        in_window = just_reported or (_is_num(days_until) and 0 <= days_until <= within_days)
        if not in_window:
            return "skip", f"earnings in {days_until} days (window: {within_days})"

    verdict_rows = load_verdicts(ticker, db_path=db_path, limit=1, live_only=True)
    verdict = verdict_rows[0] if verdict_rows else None

    history = get_earnings_history(ticker, no_cache=no_cache)
    quarters = list((history.get("quarters") or [])[:_MAX_SURPRISE_QUARTERS]) if history else []

    events = detect_events(
        submissions, lookback_days=_EVENTS_LOOKBACK_DAYS, min_severity="warning", max_events=3, today=today
    )

    row = _build_row(ticker, cik, name, catalyst, verdict, quarters, events, today)
    return "row", row


def _emit_progress(progress_cb, done: int, total: int, ticker) -> None:
    """Call ``progress_cb(done, total, ticker)``, swallowing any exception it
    raises -- a broken callback must never break the scan. No-op when
    ``progress_cb`` is ``None``."""
    if progress_cb is None:
        return
    try:
        progress_cb(done, total, ticker)
    except Exception:  # noqa: BLE001 - a broken progress callback must not break the scan
        logger.warning("preearnings scan progress_cb failed", exc_info=True)


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sort_key(row: dict):
    days_until = row.get("days_until")
    return (
        0 if row.get("just_reported") else 1,
        days_until if days_until is not None else 10**6,
        row.get("ticker") or "",
    )


def _run_scan(
    tickers: List[str],
    within_days: int,
    no_cache: bool,
    today: date,
    db_path: Optional[str],
    progress_cb: Optional[Callable[[int, int, str], None]],
    include_all: bool,
) -> dict:
    client = SecHttpClient()
    total = len(tickers)

    rows: List[dict] = []
    skipped: List[dict] = []

    for done, raw in enumerate(tickers, start=1):
        ticker = str(raw).strip().upper() if raw else str(raw)
        try:
            status, payload = _process_ticker(ticker, within_days, no_cache, today, db_path, include_all, client)
            if status == "row":
                rows.append(payload)
            else:
                skipped.append({"ticker": ticker, "reason": payload})
        except Exception as exc:  # noqa: BLE001 - one bad ticker must never fail the scan
            logger.warning("Pre-earnings scan failed for %s", ticker, exc_info=True)
            skipped.append({"ticker": ticker, "reason": str(exc)})
        _emit_progress(progress_cb, done, total, ticker)

    rows.sort(key=_sort_key)

    return {
        "generated_at": _generated_at(),
        "today": today.isoformat(),
        "within_days": within_days,
        "count": len(rows),
        "requested": total,
        "rows": rows,
        "skipped": skipped,
    }


def scan_preearnings(
    tickers: List[str],
    within_days: int = DEFAULT_WITHIN_DAYS,
    no_cache: bool = False,
    today: Optional[date] = None,
    db_path: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    include_all: bool = False,
) -> dict:
    """Scan a watchlist for names reporting earnings soon and brief each one.

    For every ticker, resolves its CIK, fetches its SEC submissions history,
    and projects its next earnings release
    (:func:`sec_analyzer.fetch.filings.estimate_next_earnings`). Names whose
    release falls inside ``within_days`` (or that just reported -- see
    ``recently_reported`` in that function's docstring) get a full briefing
    row: the latest stored *live* verdict (fair value vs. price, confidence,
    staleness), recent beat/miss history, and any material 8-K events from
    the last :data:`_EVENTS_LOOKBACK_DAYS` days.

    This is a read/assemble layer only -- it runs no valuation and computes
    no new fair value; every numeric field either comes straight from the
    stored verdict or is arithmetic over it.

    Args:
        tickers: Watchlist of ticker symbols to check.
        within_days: Lookahead window, in days, for "reports soon"
            (default :data:`DEFAULT_WITHIN_DAYS`).
        no_cache: Bypass on-disk caches for CIK resolution, submissions, and
            earnings history, re-fetching everything.
        today: Reference date for every date-dependent computation (window
            filter, staleness, verdict age). Defaults to :meth:`date.today`.
            The sole wall-clock value in the result besides ``generated_at``
            is this parameter, held fixed throughout the scan.
        db_path: Path to the SQLite verdicts store. Defaults to
            ``Config.DB_PATH`` (via
            :func:`sec_analyzer.store.database.load_verdicts`).
        progress_cb: Optional ``progress_cb(done, total, ticker)``, called
            after each ticker completes (row or skip). A raising callback is
            swallowed -- it cannot break the scan.
        include_all: When True, keep every ticker in ``rows`` regardless of
            how far out its earnings date is (a full watchlist calendar);
            the default keeps only names inside the window (or just
            reported).

    Returns:
        ``{"generated_at", "today", "within_days", "count", "requested",
        "rows", "skipped"}``. ``rows`` is sorted by ``days_until`` ascending
        (ticker ascending as a tiebreak), with every ``just_reported`` name
        sorted first regardless of its (large, next-quarter) ``days_until``.
        ``skipped`` holds ``{"ticker", "reason"}`` for every ticker that
        could not be resolved, had no usable earnings estimate, or fell
        outside the window with ``include_all=False``.

    This function never raises: any unexpected failure is logged and turns
    into a well-formed, empty-``rows`` result rather than propagating, and
    (independently) a single bad ticker within an otherwise-successful scan
    never aborts the rest of it -- it is recorded in ``skipped`` instead.
    """
    ref_today = today or date.today()
    ticker_list = list(tickers or [])
    try:
        return _run_scan(ticker_list, within_days, no_cache, ref_today, db_path, progress_cb, include_all)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("scan_preearnings() failed unexpectedly; returning an empty result.")
        return {
            "generated_at": _generated_at(),
            "today": ref_today.isoformat(),
            "within_days": within_days,
            "count": 0,
            "requested": len(ticker_list),
            "rows": [],
            "skipped": [{"ticker": None, "reason": "unexpected error during scan"}],
        }
