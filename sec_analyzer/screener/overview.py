"""Portfolio-wide sector/route overview over already-analyzed tickers.

Every other screener in this package (swing scan, pre-earnings, insider) fans
out over a bundled *universe* and computes something fresh. This module does
the opposite: it takes the verdicts this project has *already* produced --
one row per ticker, its most recent live (non-backtest) analysis, as loaded
by the orchestrator's DB helper -- and turns them into a single dashboard:
per-ticker fair-value-vs-price gaps, a sector heat map, a valuation-route
(``sector_type``) breakdown, a staleness list, and an upcoming-earnings
watchlist.

This module is pure and network-free. Its only I/O is reading the bundled
index-constituent CSVs through :mod:`sec_analyzer.screener.universe` (for
GICS sector + display name); it never touches the database, the filing
fetch layer, or price data itself -- all of that already happened upstream,
in whatever produced the input rows.

Like :mod:`sec_analyzer.signals.events`, this module is fully defensive:
:func:`build_overview` never raises, every row key is read with ``.get()``
(older stored verdicts may lack newer columns entirely), and every
date-dependent computation is threaded through an explicit ``today``
parameter so results are deterministic and testable.
"""

import logging
import statistics
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sec_analyzer.screener.universe import UNIVERSES, load_universe

try:
    from sec_analyzer.technical.momentum import sector_etf_for_sic
except Exception:  # noqa: BLE001 - this module must stay import-safe even if momentum.py changes
    sector_etf_for_sic = None

logger = logging.getLogger(__name__)

if sector_etf_for_sic is None:
    logger.warning(
        "overview: could not import sector_etf_for_sic from sec_analyzer.technical.momentum; "
        "SIC-based sector fallback disabled."
    )

#: Valuation-route code -> Turkish display label.
SECTOR_TYPE_LABELS = {
    "mature": "Olgun",
    "cyclical": "Döngüsel",
    "financial": "Finansal",
    "reit": "GYO",
    "growth_unprofitable": "Büyüme (zarar eden)",
}

#: A stored verdict older than this many days is flagged stale: the price has
#: moved and the filing behind it may have been superseded by a new quarter.
DEFAULT_STALE_DAYS = 90

#: Earnings within this many days is surfaced in the overview's catalyst list.
DEFAULT_EARNINGS_WINDOW_DAYS = 21

#: Buckets for fair-value-vs-price, in percent. A name is only called cheap or
#: expensive once it clears a margin wide enough that band arithmetic noise
#: cannot explain it; between them is "fair".
CHEAP_THRESHOLD_PCT = 15.0
EXPENSIVE_THRESHOLD_PCT = -15.0

#: Display-only saturation point for the heat-map cell value (see ``heat`` in
#: :func:`_enrich_row`): a fair-value gap this wide (or wider) maps to +/-1.0.
#: This is purely a rendering scale for the sector heat map -- it never feeds
#: `value_bucket`, `verdict_drift`, or any other decision.
_HEAT_SATURATION_PCT = 40.0

#: Stored ``fundamental_verdict`` -> the ``value_bucket`` it corresponds to,
#: used only to detect drift between the two (see :func:`_enrich_row`).
_VERDICT_TO_BUCKET = {"UCUZ": "ucuz", "MAKUL": "makul", "PAHALI": "pahali"}

#: Known values for each count dict in the top-level result (see
#: :func:`build_overview`). Every key here is present in the corresponding
#: count dict with an explicit ``0`` even when no row has that value, plus a
#: ``"bilinmiyor"`` catch-all for ``None``/unrecognized values.
_VERDICT_VALUES = ["UCUZ", "MAKUL", "PAHALI"]
_BUCKET_VALUES = ["ucuz", "makul", "pahali"]
_CONFIDENCE_VALUES = ["YÜKSEK", "ORTA", "DÜŞÜK"]
_MOMENTUM_VALUES = ["GÜÇLÜ+", "POZİTİF", "NÖTR", "NEGATİF"]
_INSIDER_VALUES = ["GÜÇLÜ ALIM", "ALIM", "NÖTR", "SATIŞ", "YOĞUN SATIŞ"]

#: Turkish label for the sector/route bucket holding rows with no known value.
_UNCLASSIFIED_SECTOR_LABEL = "Sınıflandırılmamış"

#: The bundled index-constituent CSVs do not share a sector vocabulary
#: (nasdaq100.csv says "Technology"/"Basic Materials"/"Telecommunications"
#: where sp500.csv says "Information Technology"/"Materials"/"Communication
#: Services"), so the same sector would otherwise render as two tiles.
#: Everything is normalized onto the GICS-11 spelling. Keyed by the
#: stripped, lower-cased alias; an unrecognized sector passes through
#: unchanged (never dropped, never silently remapped to something it isn't).
_SECTOR_ALIASES = {
    "technology": "Information Technology",
    "basic materials": "Materials",
    "telecommunications": "Communication Services",
}

#: Sector-ETF symbol -> GICS-11 sector display name, for the SIC-derived
#: sector fallback (see :func:`_sector_from_sic`). Mirrors the sector each
#: ETF targets in :mod:`sec_analyzer.technical.momentum`'s
#: ``_SIC_ETF_RANGES``/``_SIC_ETF_SINGLES`` comments.
_ETF_SECTOR_NAMES = {
    "XLK": "Information Technology",
    "SMH": "Information Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


def _canonicalize_sector(sector: Optional[str]) -> Optional[str]:
    """Normalize a sector string onto the GICS-11 vocabulary via
    :data:`_SECTOR_ALIASES`, matching case-insensitively on the stripped
    value. ``None``/blank input passes through unchanged; an unrecognized
    sector is returned exactly as given (never dropped, never guessed at)."""
    if not sector:
        return sector
    return _SECTOR_ALIASES.get(sector.strip().lower(), sector)


def _sector_from_sic(sic: Optional[str]) -> Optional[str]:
    """Best-effort GICS-11 sector name derived from a stored SIC code, via
    :func:`sec_analyzer.technical.momentum.sector_etf_for_sic`. Returns
    ``None`` when the helper couldn't be imported, the SIC is missing or
    unparseable, or the resulting ETF has no entry in
    :data:`_ETF_SECTOR_NAMES`. Never raises -- this module must stay
    never-fatal regardless of what ``momentum.py`` does."""
    if sector_etf_for_sic is None:
        return None
    try:
        etf = sector_etf_for_sic(sic)
    except Exception:  # noqa: BLE001 - a broken SIC mapping must not break the overview
        logger.warning("overview: sector_etf_for_sic failed for sic=%r", sic, exc_info=True)
        return None
    if not etf:
        return None
    return _ETF_SECTOR_NAMES.get(etf)


def _is_num(value: Any) -> bool:
    """Return ``True`` for a real ``int``/``float`` (excluding ``bool``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_date_only(value: Optional[str]) -> Optional[date]:
    """Parse the date part of ``value``, accepting both ``"YYYY-MM-DD"`` and
    full-ISO ``"YYYY-MM-DDTHH:MM:SS"`` timestamps. Returns ``None`` for
    missing, non-string, or unparseable input -- never raises."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def sector_type_label(sector_type: Optional[str]) -> str:
    """Turkish display label for a valuation-route code.

    Args:
        sector_type: A route code (key of :data:`SECTOR_TYPE_LABELS`), or
            ``None``/an unrecognized string.

    Returns:
        The Turkish label, or ``"Bilinmiyor"`` when ``sector_type`` is
        ``None`` or not a known route code.
    """
    return SECTOR_TYPE_LABELS.get(sector_type, "Bilinmiyor")


def _build_universe_lookup() -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Build a ``TICKER -> (name, sector)`` map from every bundled universe.

    Iterates :data:`sec_analyzer.screener.universe.UNIVERSES` in its declared
    order; for a given ticker, the first universe that supplies a non-empty
    ``name`` wins for ``name``, and independently the first that supplies a
    non-empty ``sector`` wins for ``sector`` -- so a ticker missing its
    sector in one bundled CSV can still pick it up from another. Every
    sector is passed through :func:`_canonicalize_sector` before being
    stored, since sp500.csv and nasdaq100.csv do not share one vocabulary
    (e.g. "Technology" vs. "Information Technology") and would otherwise
    produce two heat-map tiles for the same sector.

    Returns:
        The lookup dict, keyed by upper-cased ticker. An unreadable CSV for
        one universe is logged and skipped (that universe simply contributes
        nothing); a totally broken universe layer degrades to an empty dict
        rather than raising, so every row's ``name``/``sector`` end up
        ``None``.
    """
    lookup: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    try:
        for code in UNIVERSES:
            try:
                rows = load_universe(index=code)
            except Exception:  # noqa: BLE001 - one bad universe must not break the lookup
                logger.warning("overview: could not load universe %r; skipping", code, exc_info=True)
                continue
            for row in rows:
                ticker = str(row.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                name = row.get("name")
                sector = _canonicalize_sector(row.get("sector"))
                cur_name, cur_sector = lookup.get(ticker, (None, None))
                lookup[ticker] = (cur_name or name, cur_sector or sector)
    except Exception:  # noqa: BLE001 - the lookup must never raise
        logger.warning("overview: universe lookup failed entirely; sector/name will be unknown.", exc_info=True)
        return {}
    return lookup


def _enrich_row(
    row: dict,
    lookup: Dict[str, Tuple[Optional[str], Optional[str]]],
    today: date,
    stale_days: int,
) -> dict:
    """Return a copy of ``row`` with every derived overview field added.

    See the module-level field list in :func:`build_overview`'s docstring
    for the meaning of each derived key. Every input key is read with
    ``.get()`` -- an older stored verdict may lack ``insider_verdict`` or
    ``catalyst_date`` entirely.
    """
    ticker_key = str(row.get("ticker") or "").strip().upper()
    name, index_sector = lookup.get(ticker_key, (None, None))

    # Sector fallback chain: the bundled index CSVs only cover current
    # S&P 500 / Nasdaq 100 members, so a large share of analyzed tickers have
    # no entry there at all. Rather than leaving those in the catch-all
    # "unclassified" bucket, fall back to a sector derived from the stored
    # SIC code before giving up. `sector_source` records which source won
    # (or `None` when neither did), so the provenance is inspectable rather
    # than guessed.
    if index_sector:
        sector = index_sector
        sector_source = "index"
    else:
        sic_sector = _sector_from_sic(row.get("sic"))
        if sic_sector:
            sector = sic_sector
            sector_source = "sic"
        else:
            sector = None
            sector_source = None

    sector_type = row.get("sector_type")

    fv_base_lo = row.get("fv_base_lo")
    fv_base_hi = row.get("fv_base_hi")
    fv_base_mid = None
    if _is_num(fv_base_lo) and _is_num(fv_base_hi):
        fv_base_mid = (fv_base_lo + fv_base_hi) / 2.0

    price = row.get("price")
    fv_vs_price_pct = None
    if fv_base_mid is not None and _is_num(price) and price > 0:
        fv_vs_price_pct = round((fv_base_mid / price - 1.0) * 100.0, 1)

    # `value_bucket` is derived from the fair-value gap, NOT copied from the
    # stored `fundamental_verdict` -- the two measures ask different
    # questions about the very same price/band, and can legitimately
    # disagree: `fundamental_verdict` asks WHERE IN THE BAND the price sits
    # (inside the base band means MAKUL, regardless of band width), while
    # `value_bucket` asks how far the band's MIDPOINT sits from the price.
    # On a wide, low-confidence band both can be correct simultaneously --
    # price inside the band, midpoint far above/below it. The disagreement
    # is surfaced explicitly via `verdict_drift` below rather than being
    # silently papered over; see that variable's definition for what it
    # actually signals.
    if fv_vs_price_pct is None:
        value_bucket = None
    elif fv_vs_price_pct >= CHEAP_THRESHOLD_PCT:
        value_bucket = "ucuz"
    elif fv_vs_price_pct <= EXPENSIVE_THRESHOLD_PCT:
        value_bucket = "pahali"
    else:
        value_bucket = "makul"

    # `verdict_drift` is NOT an inconsistency or an error: both measures use
    # the same stored price, so price movement since analysis is never the
    # cause. In practice it isolates wide-band, low-conviction names (price
    # sits inside a wide base band, so the engine committed to MAKUL; the
    # band's midpoint nonetheless leans hard toward one side) -- a signal
    # about band width and confidence, not about which measure is "right".
    mapped_bucket = _VERDICT_TO_BUCKET.get(row.get("fundamental_verdict"))
    verdict_drift = bool(mapped_bucket is not None and value_bucket is not None and mapped_bucket != value_bucket)

    analyzed_date = _parse_date_only(row.get("analyzed_at"))
    age_days = (today - analyzed_date).days if analyzed_date is not None else None
    stale = age_days is not None and age_days > stale_days

    catalyst_date = _parse_date_only(row.get("catalyst_date"))
    days_until_earnings = (catalyst_date - today).days if catalyst_date is not None else None

    heat = None
    if fv_vs_price_pct is not None:
        heat = round(max(-1.0, min(1.0, fv_vs_price_pct / _HEAT_SATURATION_PCT)), 3)

    enriched = dict(row)
    enriched.update(
        {
            "name": name,
            "sector": sector,
            "sector_source": sector_source,
            "sector_type_label": SECTOR_TYPE_LABELS.get(sector_type),
            "fv_base_mid": fv_base_mid,
            "fv_vs_price_pct": fv_vs_price_pct,
            "value_bucket": value_bucket,
            "verdict_drift": verdict_drift,
            "age_days": age_days,
            "stale": stale,
            "days_until_earnings": days_until_earnings,
            "heat": heat,
        }
    )
    return enriched


def _pct_sort_key(row: dict) -> Tuple[int, float, str]:
    """Total sort key for ``rows``: ``fv_vs_price_pct`` descending, ``None``
    last, ties broken by ticker ascending. Never compares ``None`` to a
    number."""
    pct = row.get("fv_vs_price_pct")
    ticker = row.get("ticker") or ""
    if pct is None:
        return (1, 0.0, ticker)
    return (0, -pct, ticker)


def _aggregate(
    rows: List[dict],
    key_fn: Callable[[dict], Any],
    label_fn: Callable[[Any], str],
    id_field: str,
) -> List[dict]:
    """Group ``rows`` by ``key_fn`` into one summary dict per group.

    Shared by the ``sectors`` and ``routes`` groupings in
    :func:`build_overview` so the two aggregations can never drift apart.

    Args:
        rows: Enriched rows (see :func:`_enrich_row`).
        key_fn: Returns the grouping key for a row (e.g. ``sector`` or
            ``sector_type``); may return ``None`` for an "unknown" bucket.
        label_fn: Turkish display label for a grouping key.
        id_field: Name of the key holding the raw grouping value in each
            summary dict (``"sector"`` or ``"sector_type"``).

    Returns:
        One dict per distinct key, sorted by ``n`` descending then ``label``
        ascending.
    """
    groups: Dict[Any, List[dict]] = {}
    for row in rows:
        key = key_fn(row)
        groups.setdefault(key, []).append(row)

    result: List[dict] = []
    for key, group_rows in groups.items():
        pct_values = [r["fv_vs_price_pct"] for r in group_rows if r.get("fv_vs_price_pct") is not None]
        heat_values = [r["heat"] for r in group_rows if r.get("heat") is not None]
        median_pct = round(statistics.median(pct_values), 1) if pct_values else None
        mean_heat = round(statistics.mean(heat_values), 3) if heat_values else None

        tickers = sorted({r.get("ticker") for r in group_rows if r.get("ticker")})

        result.append(
            {
                id_field: key,
                "label": label_fn(key),
                "n": len(group_rows),
                "median_fv_vs_price_pct": median_pct,
                "mean_heat": mean_heat,
                "cheap_n": sum(1 for r in group_rows if r.get("value_bucket") == "ucuz"),
                "fair_n": sum(1 for r in group_rows if r.get("value_bucket") == "makul"),
                "expensive_n": sum(1 for r in group_rows if r.get("value_bucket") == "pahali"),
                "unknown_n": sum(1 for r in group_rows if r.get("value_bucket") is None),
                "stale_n": sum(1 for r in group_rows if r.get("stale")),
                "tickers": tickers,
            }
        )

    result.sort(key=lambda entry: (-entry["n"], entry["label"]))
    return result


def _empty_counts(values: List[str]) -> Dict[str, int]:
    """A count dict with every known value at ``0`` plus the ``"bilinmiyor"``
    catch-all -- the baseline both :func:`_count_by` and the empty-payload
    fallback build from, so the two can never disagree on which keys exist."""
    counts = {v: 0 for v in values}
    counts["bilinmiyor"] = 0
    return counts


def _count_by(rows: List[dict], field: str, values: List[str]) -> Dict[str, int]:
    """Tally ``rows`` by ``row.get(field)`` into a dict with every value in
    ``values`` present (even at ``0``), plus a ``"bilinmiyor"`` catch-all for
    ``None``/unrecognized values."""
    counts = _empty_counts(values)
    for row in rows:
        value = row.get(field)
        if value in counts:
            counts[value] += 1
        else:
            counts["bilinmiyor"] += 1
    return counts


def _empty_overview(today: date, stale_days: int, earnings_window_days: int) -> dict:
    """A well-formed, empty overview payload -- used both for ``rows=[]`` and
    as the last-resort fallback on catastrophic failure."""
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "today": today.strftime("%Y-%m-%d"),
        "count": 0,
        "rows": [],
        "sectors": [],
        "routes": [],
        "verdict_counts": _empty_counts(_VERDICT_VALUES),
        "bucket_counts": _empty_counts(_BUCKET_VALUES),
        "confidence_counts": _empty_counts(_CONFIDENCE_VALUES),
        "momentum_counts": _empty_counts(_MOMENTUM_VALUES),
        "insider_counts": _empty_counts(_INSIDER_VALUES),
        "median_fv_vs_price_pct": None,
        "stale": [],
        "stale_count": 0,
        "upcoming_earnings": [],
        "drift": [],
        "stale_days": stale_days,
        "earnings_window_days": earnings_window_days,
    }


def build_overview(
    rows: List[dict],
    today: Optional[date] = None,
    stale_days: int = DEFAULT_STALE_DAYS,
    earnings_window_days: int = DEFAULT_EARNINGS_WINDOW_DAYS,
) -> dict:
    """Build the portfolio-wide sector/route overview from stored verdicts.

    Args:
        rows: One dict per ticker -- that ticker's most recent live verdict,
            as loaded by the orchestrator's DB helper. Every key is optional;
            see the module docstring for the shape. ``None`` entries in the
            list are skipped.
        today: Reference date for age/earnings-window computations. Defaults
            to :meth:`date.today`; pass explicitly for deterministic tests.
        stale_days: A verdict older than this many days is flagged ``stale``.
        earnings_window_days: Earnings within this many days of ``today``
            land in ``upcoming_earnings``.

    Returns:
        A dict with keys ``generated_at``, ``today``, ``count``, ``rows``,
        ``sectors``, ``routes``, ``verdict_counts``, ``bucket_counts``,
        ``confidence_counts``, ``momentum_counts``, ``insider_counts``,
        ``median_fv_vs_price_pct``, ``stale``, ``stale_count``,
        ``upcoming_earnings``, ``drift``, ``stale_days``,
        ``earnings_window_days``. Each element of ``rows`` (and of the
        derived lists ``stale``/``upcoming_earnings``/``drift``, which are
        references into the same enriched dicts) carries every original key
        plus ``name``, ``sector``, ``sector_source``, ``sector_type_label``,
        ``fv_base_mid``, ``fv_vs_price_pct``, ``value_bucket``,
        ``verdict_drift``, ``age_days``, ``stale``, ``days_until_earnings``,
        ``heat``.

        ``generated_at`` (UTC, ``%Y-%m-%dT%H:%M:%SZ``) is the only
        wall-clock value in the result; every other field is a pure function
        of ``rows`` and ``today``. Never raises -- degrades to a well-formed
        empty payload (``count == 0``, empty lists, ``None`` medians) on any
        unexpected failure.
    """
    resolved_today = today or date.today()
    try:
        return _build_overview(
            [r for r in (rows or []) if r is not None],
            today=resolved_today,
            stale_days=stale_days,
            earnings_window_days=earnings_window_days,
        )
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("build_overview() failed unexpectedly; returning an empty overview.")
        return _empty_overview(resolved_today, stale_days, earnings_window_days)


def _build_overview(
    rows: List[dict],
    today: date,
    stale_days: int,
    earnings_window_days: int,
) -> dict:
    lookup = _build_universe_lookup()
    enriched = [_enrich_row(row, lookup, today, stale_days) for row in rows]
    enriched.sort(key=_pct_sort_key)

    sectors = _aggregate(
        enriched,
        key_fn=lambda r: r.get("sector"),
        label_fn=lambda s: s if s else _UNCLASSIFIED_SECTOR_LABEL,
        id_field="sector",
    )
    routes = _aggregate(
        enriched,
        key_fn=lambda r: r.get("sector_type"),
        label_fn=sector_type_label,
        id_field="sector_type",
    )

    pct_values = [r["fv_vs_price_pct"] for r in enriched if r.get("fv_vs_price_pct") is not None]
    median_pct = round(statistics.median(pct_values), 1) if pct_values else None

    stale_rows = sorted(
        (r for r in enriched if r.get("stale")),
        key=lambda r: (-r["age_days"], r.get("ticker") or ""),
    )
    upcoming = sorted(
        (
            r
            for r in enriched
            if r.get("days_until_earnings") is not None and 0 <= r["days_until_earnings"] <= earnings_window_days
        ),
        key=lambda r: (r["days_until_earnings"], r.get("ticker") or ""),
    )
    drift_rows = sorted(
        (r for r in enriched if r.get("verdict_drift")),
        key=lambda r: (-abs(r["fv_vs_price_pct"]), r.get("ticker") or ""),
    )

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "today": today.strftime("%Y-%m-%d"),
        "count": len(enriched),
        "rows": enriched,
        "sectors": sectors,
        "routes": routes,
        "verdict_counts": _count_by(enriched, "fundamental_verdict", _VERDICT_VALUES),
        "bucket_counts": _count_by(enriched, "value_bucket", _BUCKET_VALUES),
        "confidence_counts": _count_by(enriched, "confidence", _CONFIDENCE_VALUES),
        "momentum_counts": _count_by(enriched, "momentum_verdict", _MOMENTUM_VALUES),
        "insider_counts": _count_by(enriched, "insider_verdict", _INSIDER_VALUES),
        "median_fv_vs_price_pct": median_pct,
        "stale": stale_rows,
        "stale_count": len(stale_rows),
        "upcoming_earnings": upcoming,
        "drift": drift_rows,
        "stale_days": stale_days,
        "earnings_window_days": earnings_window_days,
    }
