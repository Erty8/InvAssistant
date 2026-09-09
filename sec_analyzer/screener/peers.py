"""Cross-sectional peer comparison built on SEC's XBRL Frames API.

Every other reference this package has for "is this multiple cheap or
expensive" is the filer's OWN history (``valuation.multiples``) or a
static, hand-refreshed, annual CSV (``data/damodaran/multiples.csv``) matched
by a SIC-description substring. Neither compares a filer against other real
filers in its sector, right now. This module adds that missing axis: for a
bundled index universe (~550 large caps, see
``sec_analyzer.screener.universe``), it pulls each company's most recent
filed fundamentals via :mod:`sec_analyzer.fetch.frames`, derives a small set
of price-free ratios, and buckets them by canonical GICS sector so a single
filer's own ratios can be percentile-ranked against its peers.

Scope boundary (non-negotiable): this is a **context layer**. Nothing here
feeds the fair-value computation, the triangulation signals, or
``sector_ratio``. ``sec_analyzer/valuation/SPEC.md`` remains the sole binding
contract for those and is not amended by this module -- the numbers below
are reported ALONGSIDE a valuation, never inside one.

Metric set -- deliberately price-free. A price-based cross-section (P/E,
EV/EBITDA across ~550 names) would need a market cap for every peer, i.e.
hundreds of price fetches, and would go stale the moment the market moved.
Margins, returns and leverage are reported facts: they change only when
someone files, which is exactly the cadence this snapshot refreshes on.

Known approximation, stated honestly: SEC's Frames API buckets each filer's
fact into the calendar period that best fits its own fiscal calendar, so a
company with a non-December fiscal year end is compared on a slightly
different 12-month window than a December filer. That is acceptable for a
sector median -- it is not exact, and it is one more reason this layer is
context, not a valuation input.

Three public entry points:

* :func:`build_peer_snapshot` -- I/O + aggregation, builds one sector-bucketed
  snapshot for a calendar year.
* :func:`rank_against_peers` -- pure, no I/O, ranks one company's own metrics
  against a snapshot's sector distribution.
* :func:`resolve_sector` -- pure (aside from a bundled-CSV read), resolves a
  filer's canonical GICS sector so a caller has a ready-made key into
  ``snapshot["sectors"]`` without reaching into another module's privates.

Plus :func:`metrics_from_normalized`, a small pure helper that assembles the
same nine metric keys for the package's OWN analyzed filer (from its
already-computed ``normalized``/``ratios``/``metrics`` structures) so the
comparison side of :func:`rank_against_peers` is apples-to-apples with the
Frames-derived peer side.

Sector-vocabulary ownership: :data:`SECTOR_ALIASES`, :data:`ETF_SECTOR_NAMES`,
:func:`canonicalize_sector`, and :func:`sector_from_sic` are the CANONICAL
copies of what used to be private, independently-drifting helpers of the
same name/shape in :mod:`sec_analyzer.screener.overview`
(``_SECTOR_ALIASES``, ``_ETF_SECTOR_NAMES``, ``_canonicalize_sector``,
``_sector_from_sic``). This module owns them now; ``overview.py`` is
expected to import these public names instead of keeping its own copies, so
the GICS-11 vocabulary can never fragment into two versions again.
"""

import logging
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sec_analyzer.fetch.frames import annual_period, get_frame_with_aliases, instant_period
from sec_analyzer.normalize.concepts import CONCEPTS
from sec_analyzer.normalize.normalizer import to_annual_series
from sec_analyzer.screener.universe import UNIVERSES, load_universe

logger = logging.getLogger(__name__)

try:
    from sec_analyzer.normalize.metrics import resolve_fundamental_fy
except Exception:  # noqa: BLE001 - this module must stay import-safe even if metrics.py changes
    resolve_fundamental_fy = None
    logger.warning("peers: could not import resolve_fundamental_fy from normalize.metrics.")

try:
    from sec_analyzer.technical.momentum import sector_etf_for_sic
except Exception:  # noqa: BLE001 - this module must stay import-safe even if momentum.py changes
    sector_etf_for_sic = None
    logger.warning("peers: could not import sector_etf_for_sic from technical.momentum; SIC fallback disabled.")

#: Canonical GICS-11 sector-alias table. The bundled index CSVs do not share
#: one sector vocabulary (nasdaq100.csv says "Technology"/"Basic
#: Materials"/"Telecommunications" where sp500.csv says "Information
#: Technology"/"Materials"/"Communication Services"); without
#: canonicalizing, the same sector fragments into two heat-map/peer-bucket
#: tiles. Keyed by the stripped, lower-cased alias; an unrecognized sector
#: passes through :func:`canonicalize_sector` unchanged (never dropped,
#: never silently remapped to something it isn't).
#:
#: This is the CANONICAL copy -- formerly duplicated as a private
#: ``_SECTOR_ALIASES`` in ``screener.overview``, which should import this
#: dict instead of keeping its own (see module docstring).
SECTOR_ALIASES: Dict[str, str] = {
    "technology": "Information Technology",
    "basic materials": "Materials",
    "telecommunications": "Communication Services",
}

#: Sector-ETF symbol -> GICS-11 sector display name, for the SIC-derived
#: sector fallback (see :func:`sector_from_sic`). Mirrors the sector each
#: ETF targets in :mod:`sec_analyzer.technical.momentum`'s
#: ``_SIC_ETF_RANGES``/``_SIC_ETF_SINGLES`` comments.
#:
#: This is the CANONICAL copy -- formerly duplicated as a private
#: ``_ETF_SECTOR_NAMES`` in ``screener.overview``, which should import this
#: dict instead of keeping its own (see module docstring).
ETF_SECTOR_NAMES: Dict[str, str] = {
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


def canonicalize_sector(sector: Optional[str]) -> Optional[str]:
    """Normalize a sector string onto the GICS-11 vocabulary via
    :data:`SECTOR_ALIASES`, matching case-insensitively on the stripped
    value. ``None``/blank input passes through unchanged; an unrecognized
    sector is returned exactly as given (never dropped, never guessed at).

    This is the CANONICAL implementation -- see the module docstring's
    "Sector-vocabulary ownership" note.
    """
    if not sector:
        return sector
    return SECTOR_ALIASES.get(sector.strip().lower(), sector)


def sector_from_sic(sic: Optional[str]) -> Optional[str]:
    """Best-effort GICS-11 sector name derived from a filer's SIC code, via
    :func:`sec_analyzer.technical.momentum.sector_etf_for_sic` and
    :data:`ETF_SECTOR_NAMES`.

    Returns ``None`` when the helper couldn't be imported, ``sic`` is
    missing/unparseable, or the resulting ETF has no entry in
    :data:`ETF_SECTOR_NAMES`. Never raises.

    This is the CANONICAL implementation -- see the module docstring's
    "Sector-vocabulary ownership" note.
    """
    if sector_etf_for_sic is None:
        return None
    try:
        etf = sector_etf_for_sic(sic)
    except Exception:  # noqa: BLE001 - a broken SIC mapping must not break the caller
        logger.warning("peers: sector_etf_for_sic failed for sic=%r", sic, exc_info=True)
        return None
    if not etf:
        return None
    return ETF_SECTOR_NAMES.get(etf)


#: The nine metric keys reported per company, in the fixed order used for
#: sector aggregation and for :func:`metrics_from_normalized`'s output.
METRIC_KEYS = (
    "revenue", "net_margin", "operating_margin", "gross_margin", "fcf_margin",
    "roe", "roa", "debt_to_equity", "revenue_growth",
)

#: Metrics for which a HIGH peer percentile is a WEAKNESS rather than a
#: strength (more leverage vs. peers is bad, not good) -- see
#: :func:`rank_against_peers`. Every other metric in :data:`METRIC_KEYS`
#: reads "higher percentile = stronger" directly.
_LOWER_IS_BETTER = {"debt_to_equity"}

#: Minimum peer sample size before a percentile is reported at all --
#: mirrors ``valuation.multiples._MIN_PERCENTILE_SAMPLE`` exactly (see
#: :func:`_percentile_rank`'s docstring for why the formula is reimplemented
#: here rather than imported).
_MIN_PEER_SAMPLE = 5

#: Percentile bands for the strengths/weaknesses split in
#: :func:`rank_against_peers`.
_STRENGTH_PCT = 75.0
_WEAKNESS_PCT = 25.0

#: Display labels for the note rendered by :func:`rank_against_peers`.
_METRIC_TR_LABELS = {
    "revenue": "revenue",
    "net_margin": "net margin",
    "operating_margin": "operating margin",
    "gross_margin": "gross margin",
    "fcf_margin": "FCF margin",
    "roe": "return on equity (ROE)",
    "roa": "return on assets (ROA)",
    "debt_to_equity": "debt-to-equity ratio",
    "revenue_growth": "revenue growth",
}


def _now_iso() -> str:
    """UTC timestamp in the exact format used by
    ``screener.swing_scan.scan_swing``'s ``generated_at`` -- the only
    wall-clock value in this module's output."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# build_peer_snapshot
# ---------------------------------------------------------------------------


def _coerce_cik(raw_cik) -> Optional[int]:
    """Best-effort int CIK from a universe-CSV cell. ``None`` for
    missing/blank/unparseable input -- never raises."""
    if raw_cik is None:
        return None
    text = str(raw_cik).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _resolve_indexes(indexes: Optional[List[str]]) -> List[str]:
    if indexes:
        return list(indexes)
    return list(UNIVERSES)


def load_combined_universe(indexes: Optional[List[str]] = None) -> List[dict]:
    """Union of every bundled index's constituent rows.

    Mirrors ``screener.overview._build_universe_lookup``'s per-field "first
    non-empty wins" merge, extended to also carry ``cik`` through. Sector
    values are canonicalized before being stored so downstream grouping
    never has to canonicalize twice. Rows are returned sorted by ticker for
    a fully deterministic downstream iteration order.

    Public (unlike most of this module's I/O helpers) because
    :func:`build_universe_sector_lookup` and :func:`resolve_sector`'s rung 2
    both need it, and because ``screener.overview`` may want it too instead
    of maintaining its own near-identical merge.

    Args:
        indexes: Which bundled universes to union. Defaults to every key of
            :data:`sec_analyzer.screener.universe.UNIVERSES`.
    """
    combined: Dict[str, dict] = {}
    for code in _resolve_indexes(indexes):
        try:
            rows = load_universe(index=code)
        except Exception:  # noqa: BLE001 - one bad universe must not break the build
            logger.warning("load_combined_universe: could not load universe %r; skipping.", code, exc_info=True)
            continue
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            sector = canonicalize_sector(row.get("sector"))
            existing = combined.get(ticker)
            if existing is None:
                combined[ticker] = {
                    "ticker": ticker,
                    "name": row.get("name"),
                    "sector": sector,
                    "cik": row.get("cik"),
                }
            else:
                if not existing.get("name"):
                    existing["name"] = row.get("name")
                if not existing.get("sector"):
                    existing["sector"] = sector
                if not existing.get("cik"):
                    existing["cik"] = row.get("cik")
    return [combined[t] for t in sorted(combined)]


def build_universe_sector_lookup(indexes: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """Ticker (upper-cased) -> canonical GICS sector, across the union of
    bundled indexes. A thin, sector-only view of :func:`load_combined_universe`,
    used by :func:`resolve_sector`'s rung 2 (a filer that is a constituent of
    a bundled index but was not counted in a particular
    :func:`build_peer_snapshot` result).
    """
    return {row["ticker"]: row.get("sector") for row in load_combined_universe(indexes)}


def _frac(numerator: Optional[float], denominator: Optional[float], require_positive: bool = True) -> Optional[float]:
    """Divide two optional numbers, guarding against ``None`` and a
    zero (or, when ``require_positive``, non-positive) denominator."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    if require_positive and denominator <= 0:
        return None
    return numerator / denominator


def _derive_company_metrics(
    revenue: Optional[float],
    revenue_prior: Optional[float],
    net_income: Optional[float],
    operating_income: Optional[float],
    gross_profit: Optional[float],
    ocf: Optional[float],
    capex: Optional[float],
    equity: Optional[float],
    assets: Optional[float],
    liabilities: Optional[float],
) -> Dict[str, Optional[float]]:
    """Derive the nine peer metrics for one company from raw Frames values.

    Every derivation is ``None`` unless every required input is present and
    the denominator is non-zero / strictly positive where a negative
    denominator would make the ratio meaningless (SPEC for this module):
    margins require ``revenue > 0``; ``roe``/``debt_to_equity`` require
    ``equity > 0`` (a negative-equity ROE, or a debt/equity ratio over a
    negative base, is not a huge number -- it is not a number); ``roa``
    requires ``assets > 0``; ``revenue_growth`` requires the prior year's
    revenue to be strictly positive.
    """
    net_margin = _frac(net_income, revenue)
    operating_margin = _frac(operating_income, revenue)
    gross_margin = _frac(gross_profit, revenue)

    fcf = None if ocf is None or capex is None else ocf - capex
    fcf_margin = _frac(fcf, revenue)

    roe = _frac(net_income, equity)
    roa = _frac(net_income, assets)
    debt_to_equity = _frac(liabilities, equity)

    revenue_growth = None
    if revenue is not None and revenue_prior is not None and revenue_prior > 0:
        revenue_growth = round(revenue / revenue_prior - 1.0, 6)

    return {
        "revenue": revenue,
        "net_margin": _round_or_none(net_margin),
        "operating_margin": _round_or_none(operating_margin),
        "gross_margin": _round_or_none(gross_margin),
        "fcf_margin": _round_or_none(fcf_margin),
        "roe": _round_or_none(roe),
        "roa": _round_or_none(roa),
        "debt_to_equity": _round_or_none(debt_to_equity),
        "revenue_growth": revenue_growth,
    }


def _round_or_none(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    return None if value is None else round(value, ndigits)


def _stats(values: List[float]) -> Optional[dict]:
    """Median/p25/p75 (linear interpolation, i.e. the same convention as
    ``numpy.percentile``'s default / ``statistics.quantiles(..., method="inclusive")``)
    plus the sorted value list itself, so a caller can look up an exact
    percentile later (see :func:`_percentile_rank`) without refetching.

    Returns ``None`` for an empty ``values`` list.
    """
    if not values:
        return None
    values_sorted = sorted(values)
    n = len(values_sorted)
    median = statistics.median(values_sorted)
    if n >= 2:
        q1, _q2, q3 = statistics.quantiles(values_sorted, n=4, method="inclusive")
    else:
        q1 = q3 = values_sorted[0]
    return {
        "n": n,
        "median": round(median, 6),
        "p25": round(q1, 6),
        "p75": round(q3, 6),
        "values": values_sorted,
    }


def _empty_snapshot(year: int, indexes: Optional[List[str]], skipped: Optional[List[dict]] = None) -> dict:
    """A well-formed, empty snapshot -- used both as the catastrophic-failure
    fallback and as a stable shape for callers to depend on."""
    return {
        "generated_at": _now_iso(),
        "year": year,
        "indexes": _resolve_indexes(indexes),
        "universe_size": 0,
        "covered": 0,
        "sectors": {},
        "companies": {},
        "frames_fetched": 0,
        "frames_missing": [],
        "skipped": skipped or [],
    }


def build_peer_snapshot(
    year: int,
    client,
    no_cache: bool = False,
    indexes: Optional[List[str]] = None,
) -> dict:
    """Build a sector-bucketed cross-sectional peer snapshot for one year.

    Args:
        year: Calendar year whose annual (duration) and year-end (instant)
            frames are pulled, e.g. ``2024`` for FY2024 filings.
        client: :class:`sec_analyzer.http_client.SecHttpClient` (or anything
            with a compatible ``get_json``) used by the underlying
            :mod:`sec_analyzer.fetch.frames` calls.
        no_cache: Forwarded to every underlying frame fetch.
        indexes: Which bundled universes (keys of
            :data:`sec_analyzer.screener.universe.UNIVERSES`) to union into
            the peer set. Defaults to every bundled universe.

    Returns:
        A dict (see the module's SPEC for the exact shape) with keys
        ``generated_at`` (the only wall-clock value), ``year``, ``indexes``,
        ``universe_size``, ``covered``, ``sectors``, ``companies``,
        ``frames_fetched``, ``frames_missing``, ``skipped``. Never raises --
        on catastrophic failure, returns :func:`_empty_snapshot` with the
        error recorded in ``skipped``.
    """
    try:
        return _build_peer_snapshot(year, client, no_cache=no_cache, indexes=indexes)
    except Exception:  # noqa: BLE001 - build_peer_snapshot() must never raise
        logger.exception("build_peer_snapshot() failed unexpectedly; returning an empty snapshot.")
        return _empty_snapshot(
            year, indexes,
            skipped=[{"ticker": None, "reason": "build_peer_snapshot() encountered an unexpected error."}],
        )


def _build_peer_snapshot(year: int, client, no_cache: bool, indexes: Optional[List[str]]) -> dict:
    resolved_indexes = _resolve_indexes(indexes)
    universe_rows = load_combined_universe(resolved_indexes)
    universe_size = len(universe_rows)

    skipped: List[dict] = []
    entries: List[tuple] = []
    for row in universe_rows:
        cik = _coerce_cik(row.get("cik"))
        if cik is None:
            skipped.append({"ticker": row.get("ticker"), "reason": "No usable CIK"})
            continue
        entries.append((cik, row))

    period = annual_period(year)
    prior_period = annual_period(year - 1)
    instant = instant_period(year, 4)

    frames_fetched = 0
    frames_missing: List[str] = []

    def _fetch(concept: str, period_str: str) -> Dict[int, float]:
        nonlocal frames_fetched
        frames_fetched += 1
        tags = CONCEPTS.get(concept) or [concept]
        values = get_frame_with_aliases(tags, period_str, client, no_cache=no_cache)
        if not values:
            frames_missing.append(f"{concept}@{period_str}")
        return values

    revenue_cur = _fetch("Revenue", period)
    revenue_prior = _fetch("Revenue", prior_period)
    net_income = _fetch("NetIncome", period)
    operating_income = _fetch("OperatingIncome", period)
    gross_profit = _fetch("GrossProfit", period)
    ocf = _fetch("OperatingCashFlow", period)
    capex = _fetch("CapEx", period)
    equity = _fetch("StockholdersEquity", instant)
    assets = _fetch("TotalAssets", instant)
    liabilities = _fetch("TotalLiabilities", instant)

    companies: Dict[str, dict] = {}
    sector_values: Dict[str, Dict[str, List[float]]] = {}
    sector_counts: Dict[str, int] = {}
    covered = 0

    for cik, row in entries:
        metrics = _derive_company_metrics(
            revenue=revenue_cur.get(cik),
            revenue_prior=revenue_prior.get(cik),
            net_income=net_income.get(cik),
            operating_income=operating_income.get(cik),
            gross_profit=gross_profit.get(cik),
            ocf=ocf.get(cik),
            capex=capex.get(cik),
            equity=equity.get(cik),
            assets=assets.get(cik),
            liabilities=liabilities.get(cik),
        )
        if any(v is not None for v in metrics.values()):
            covered += 1

        sector = row.get("sector")
        companies[str(cik)] = {
            "ticker": row.get("ticker"),
            "sector": sector,
            "metrics": metrics,
        }

        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            bucket = sector_values.setdefault(sector, {k: [] for k in METRIC_KEYS})
            for key in METRIC_KEYS:
                v = metrics.get(key)
                if v is not None:
                    bucket[key].append(v)

    sectors: Dict[str, dict] = {}
    for sector, metric_lists in sector_values.items():
        metric_stats = {}
        for key in METRIC_KEYS:
            stats = _stats(metric_lists.get(key) or [])
            if stats is not None:
                metric_stats[key] = stats
        sectors[sector] = {"n": sector_counts.get(sector, 0), "metrics": metric_stats}

    return {
        "generated_at": _now_iso(),
        "year": year,
        "indexes": resolved_indexes,
        "universe_size": universe_size,
        "covered": covered,
        "sectors": sectors,
        "companies": companies,
        "frames_fetched": frames_fetched,
        "frames_missing": frames_missing,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# rank_against_peers
# ---------------------------------------------------------------------------


def _percentile_rank(history_values: List[Optional[float]], current: Optional[float]) -> Optional[float]:
    """Midrank percentile of ``current`` within ``history_values``.

    Deliberately reimplements
    ``sec_analyzer.valuation.multiples.percentile_position`` byte-for-byte
    (percentage strictly less + half the tied percentage, min sample 5)
    rather than importing it: that module imports ``pandas`` at module
    scope for its own (unrelated) price-history handling, and this package
    -- ``fetch/frames.py`` and ``screener/peers.py`` -- is required to stay
    pandas-free. Any change to the tie-break rule in ``multiples.py`` must be
    mirrored here by hand.
    """
    if current is None:
        return None
    valid = [v for v in (history_values or []) if v is not None]
    if len(valid) < _MIN_PEER_SAMPLE:
        return None
    less_count = sum(1 for v in valid if v < current)
    equal_count = sum(1 for v in valid if v == current)
    pct = (less_count + 0.5 * equal_count) / len(valid) * 100.0
    return round(pct, 1)


def _ordinal_suffix(n: int) -> str:
    """English ordinal suffix for an integer (``1`` -> ``"st"``, ``12`` ->
    ``"th"``, ``23`` -> ``"rd"``, ...)."""
    if 10 <= abs(n) % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(abs(n) % 10, "th")


def _format_pct(pct: float) -> str:
    """Format a percentile for the note with its English ordinal suffix: an
    integer value renders as e.g. ``"88th"``, a fractional one keeps one
    decimal place with the suffix on the integer part (``"81.3rd"``)."""
    if pct == int(pct):
        n = int(pct)
        return f"{n}{_ordinal_suffix(n)}"
    return f"{pct:.1f}{_ordinal_suffix(int(pct))}"


def _join_and(phrases: List[str]) -> str:
    """Join phrases English-style: ``"a"``, ``"a and b"``, ``"a, b, and c"``."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _metric_phrase(key: str, pct: float) -> str:
    label = _METRIC_TR_LABELS.get(key, key)
    return f"{label} ({_format_pct(pct)} percentile)"


def _build_note(sector: str, peer_n: int, strengths: List[str], weaknesses: List[str], metrics: Dict[str, dict]) -> str:
    if not strengths and not weaknesses:
        return f"No notable divergence versus {peer_n} peers."

    clauses = []
    if strengths:
        phrases = [_metric_phrase(k, metrics[k]["percentile"]) for k in strengths]
        clauses.append(f"strong in {_join_and(phrases)}")
    if weaknesses:
        phrases = [_metric_phrase(k, metrics[k]["percentile"]) for k in weaknesses]
        clauses.append(f"weak in {_join_and(phrases)}")
    return f"Versus {peer_n} peers in the {sector} sector, " + "; ".join(clauses) + "."


def rank_against_peers(metrics: Optional[Dict[str, Optional[float]]], sector: Optional[str], snapshot: Optional[dict]) -> Optional[dict]:
    """Rank one company's own metrics against a sector's peer distribution.

    Args:
        metrics: A dict of the same metric names in :data:`METRIC_KEYS` for
            ONE company -- typically the output of
            :func:`metrics_from_normalized` for the package's own analyzed
            filer, though it need not come from that helper.
        sector: The canonical GICS sector to rank against (must match a key
            of ``snapshot["sectors"]``).
        snapshot: The dict returned by :func:`build_peer_snapshot`.

    Returns:
        ``None`` when ``snapshot`` has no such sector, or when ``metrics``/
        ``sector``/``snapshot`` are missing entirely. Otherwise a dict with
        keys ``sector``, ``peer_n``, ``year`` (the peer SNAPSHOT's fixed
        calendar year), ``company_fy`` (the filer's own latest fiscal year,
        read from ``metrics["fy"]`` -- see :func:`metrics_from_normalized`),
        ``fy_mismatch`` (``True`` when both years are known and differ),
        ``metrics`` (only metrics present in BOTH inputs, each carrying
        ``value``, ``percentile``, ``median``, ``p25``, ``p75``, ``n``,
        ``delta_vs_median`` -- and omitted rather than ``None``-valued when
        the peer sample for that metric has fewer than 5 usable values),
        ``strengths`` (percentile >= 75, sorted descending by percentile),
        ``weaknesses`` (percentile <= 25, sorted ascending by percentile),
        and one plain-language sentence ``note``.

        ``debt_to_equity`` is read with the sense inverted for the
        strengths/weaknesses classification only (see :data:`_LOWER_IS_BETTER`):
        a HIGH percentile there means MORE leverage than peers, which is a
        weakness, not a strength -- the reported ``percentile`` value itself
        is never inverted, only which bucket it lands in.

        A percentile computed against a peer snapshot built for a DIFFERENT
        fiscal year than the one the filer's own metrics were drawn from
        overstates its own precision (a sector's margins move year to
        year) -- this is disclosed, not corrected: when ``fy_mismatch`` is
        ``True``, ``note`` gets a trailing caveat naming both years
        rather than silently presenting the percentile as if the periods
        matched. This module stays a context layer either way; it does not
        re-derive the filer's metrics for the snapshot's year.

        Pure, no I/O, never raises.
    """
    try:
        return _rank_against_peers(metrics or {}, sector, snapshot or {})
    except Exception:  # noqa: BLE001 - rank_against_peers() must never raise
        logger.warning("rank_against_peers() failed unexpectedly.", exc_info=True)
        return None


def _rank_against_peers(metrics: Dict[str, Optional[float]], sector: Optional[str], snapshot: dict) -> Optional[dict]:
    if not sector:
        return None
    sectors = snapshot.get("sectors") or {}
    sector_data = sectors.get(sector)
    if not sector_data:
        return None

    peer_metrics = sector_data.get("metrics") or {}
    peer_n = sector_data.get("n") or 0

    result_metrics: Dict[str, dict] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        peer_entry = peer_metrics.get(key)
        if not peer_entry:
            continue
        values = peer_entry.get("values") or []
        pct = _percentile_rank(values, value)
        if pct is None:
            continue
        median = peer_entry.get("median")
        result_metrics[key] = {
            "value": value,
            "percentile": pct,
            "median": median,
            "p25": peer_entry.get("p25"),
            "p75": peer_entry.get("p75"),
            "n": peer_entry.get("n", len(values)),
            "delta_vs_median": None if median is None else round(value - median, 6),
        }

    strengths_set = []
    weaknesses_set = []
    for key, info in result_metrics.items():
        pct = info["percentile"]
        lower_is_better = key in _LOWER_IS_BETTER
        # debt_to_equity reads inverted: a LOW percentile (little leverage
        # vs. peers) is the strength, a HIGH one is the weakness. Every
        # other metric reads directly.
        is_strength = (pct <= _WEAKNESS_PCT) if lower_is_better else (pct >= _STRENGTH_PCT)
        is_weakness = (pct >= _STRENGTH_PCT) if lower_is_better else (pct <= _WEAKNESS_PCT)
        if is_strength:
            strengths_set.append(key)
        elif is_weakness:
            weaknesses_set.append(key)

    strengths = sorted(strengths_set, key=lambda k: (-result_metrics[k]["percentile"], k))
    weaknesses = sorted(weaknesses_set, key=lambda k: (result_metrics[k]["percentile"], k))

    note = _build_note(sector, peer_n, strengths, weaknesses, result_metrics)

    # Disclosure, not correction (see rank_against_peers's docstring): the
    # peer snapshot is built for one FIXED calendar year, but the filer's
    # own metrics (metrics_from_normalized's "fy" metadata key) may be
    # newer -- a real apples-to-oranges gap a percentile alone hides. Only
    # flagged when BOTH years are actually known and differ; either being
    # unresolvable is "unknown", not "mismatched".
    peer_year = snapshot.get("year")
    company_fy = metrics.get("fy")
    fy_mismatch = bool(company_fy is not None and peer_year is not None and company_fy != peer_year)
    if fy_mismatch:
        note = f"{note} (company FY{company_fy}, peer snapshot FY{peer_year} — different years)."

    return {
        "sector": sector,
        "peer_n": peer_n,
        "year": peer_year,
        "company_fy": company_fy,
        "fy_mismatch": fy_mismatch,
        "metrics": result_metrics,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "note": note,
    }


# ---------------------------------------------------------------------------
# resolve_sector
# ---------------------------------------------------------------------------


def resolve_sector(
    ticker: Optional[str] = None,
    cik=None,
    sic: Optional[str] = None,
    snapshot: Optional[dict] = None,
) -> Optional[str]:
    """Resolve a filer's canonical GICS sector, ready to use as a key into
    ``snapshot["sectors"]`` for :func:`rank_against_peers`.

    Resolution order, first hit wins -- most specific evidence first:

    1. ``snapshot["companies"][str(cik)]["sector"]`` -- when a snapshot AND
       a resolvable ``cik`` are given, and that CIK is itself one of the
       peer rows the snapshot was built from, use the EXACT sector its own
       row was bucketed under. This guarantees a filer is never ranked
       against a sector it was not counted in, even if the bundled index
       CSV and this snapshot happen to disagree.
    2. The bundled-index lookup by ``ticker`` (canonicalized via
       :func:`build_universe_sector_lookup`) -- for a filer that is a
       constituent of a bundled index but was not counted in THIS
       particular snapshot (a narrower ``indexes`` selection, a snapshot
       built before the filer was added to the index, ...).
    3. A SIC-derived sector (:func:`sector_from_sic`) -- the last-resort
       fallback, shared with what used to be ``screener.overview``'s own
       private SIC fallback.
    4. ``None``.

    Args:
        ticker: The filer's ticker, if known.
        cik: The filer's CIK (int, or a numeric string in either padded or
            unpadded form), if known.
        sic: The filer's SIC code, if known.
        snapshot: A :func:`build_peer_snapshot` result, if available.

    Returns:
        The canonical GICS-11 sector name, or ``None`` if no rung resolves.
        Never raises.
    """
    try:
        return _resolve_sector(ticker, cik, sic, snapshot)
    except Exception:  # noqa: BLE001 - resolve_sector() must never raise
        logger.warning("resolve_sector() failed unexpectedly.", exc_info=True)
        return None


def _resolve_sector(ticker: Optional[str], cik, sic: Optional[str], snapshot: Optional[dict]) -> Optional[str]:
    cik_int = _coerce_cik(cik) if cik is not None else None

    if snapshot and cik_int is not None:
        companies = snapshot.get("companies") or {}
        company = companies.get(str(cik_int))
        if company and company.get("sector"):
            return company["sector"]

    if ticker:
        lookup = build_universe_sector_lookup()
        sector = lookup.get(str(ticker).strip().upper())
        if sector:
            return sector

    if sic:
        sector = sector_from_sic(sic)
        if sector:
            return sector

    return None


# ---------------------------------------------------------------------------
# metrics_from_normalized
# ---------------------------------------------------------------------------


def _resolve_latest_fy(ratios: List[dict], metrics: dict) -> Optional[int]:
    if resolve_fundamental_fy is not None:
        fy = resolve_fundamental_fy(metrics)
        if fy is not None:
            return fy
    if ratios:
        # compute_ratios() sorts descending by fy, so the first row is the
        # latest fiscal year even if resolve_fundamental_fy is unavailable.
        return ratios[0].get("fy")
    return None


def metrics_from_normalized(
    normalized: Optional[dict], ratios: Optional[List[dict]], metrics: Optional[dict]
) -> Dict[str, Optional[float]]:
    """Build the same nine :data:`METRIC_KEYS` for the package's own
    analyzed filer, for an apples-to-apples comparison against
    :func:`build_peer_snapshot`'s Frames-derived peers.

    Prefers reusing the package's own already-computed fields over
    recomputing them from raw facts:

    * ``net_margin``, ``operating_margin``, ``gross_margin``, ``fcf_margin``,
      ``roa`` come straight from the matching
      :func:`sec_analyzer.normalize.ratios.compute_ratios` row.
    * ``roe``/``debt_to_equity`` also come from that row, but are re-masked
      to ``None`` when ``StockholdersEquity`` for that fiscal year is
      missing or non-positive -- ``ratios.compute_ratios`` does not apply
      that guard (it only checks for a non-zero denominator), while
      :mod:`sec_analyzer.screener.peers`'s Frames-side derivation does (a
      negative-equity ROE is not a huge number, it is not a number). Without
      this re-mask the two sides of a comparison would not be reading the
      same rule.
    * ``revenue`` (a raw figure Frames carries but ``ratios``/``metrics``
      does not) and ``revenue_growth`` (``ratios``'s ``yoy_revenue_growth``)
      are read from ``normalized``'s own annual ``Revenue`` series.

    Args:
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.
        ratios: The list returned by
            ``sec_analyzer.normalize.ratios.compute_ratios``.
        metrics: The dict returned by
            ``sec_analyzer.normalize.metrics.compute_metrics`` (used only to
            resolve which fiscal year is "latest fundamental").

    Returns:
        A dict with exactly :data:`METRIC_KEYS` plus one extra metadata key,
        ``"fy"`` -- the fiscal year the figures were drawn from (see
        :func:`_resolve_latest_fy`), NOT one of :data:`METRIC_KEYS` itself,
        so any caller that loops over ``METRIC_KEYS`` to read this dict is
        unaffected. ``"fy"`` matters downstream because
        :func:`build_peer_snapshot` is built for a FIXED calendar year while
        this filer's own latest reported fiscal year may be newer -- see
        :func:`rank_against_peers`'s ``fy_mismatch``/``company_fy`` handling,
        which is what surfaces that gap to the reader rather than silently
        comparing two different periods. Every :data:`METRIC_KEYS` entry is
        ``None``-safe -- empty or malformed input (e.g. ``normalized`` has no
        annual data at all) degrades to every key ``None`` (``"fy"``
        included) rather than an exception. ``{}`` is returned only on a
        genuine internal failure. Pure, no I/O, never raises.
    """
    try:
        return _metrics_from_normalized(normalized or {}, ratios or [], metrics or {})
    except Exception:  # noqa: BLE001 - metrics_from_normalized() must never raise
        logger.warning("metrics_from_normalized() failed unexpectedly.", exc_info=True)
        return {}


def _metrics_from_normalized(normalized: dict, ratios: List[dict], metrics: dict) -> Dict[str, Optional[float]]:
    fy = _resolve_latest_fy(ratios, metrics)

    row = next((r for r in ratios if r.get("fy") == fy), None)
    if row is None and ratios:
        row = ratios[0]
    row = row or {}

    revenue = None
    equity = None
    if fy is not None:
        try:
            revenue = to_annual_series(normalized, "Revenue").get(fy)
        except Exception:  # noqa: BLE001 - a broken normalized dict must not crash this helper
            revenue = None
        try:
            equity = to_annual_series(normalized, "StockholdersEquity").get(fy)
        except Exception:  # noqa: BLE001 - a broken normalized dict must not crash this helper
            equity = None

    equity_positive = equity is not None and equity > 0
    roe = row.get("roe") if equity_positive else None
    debt_to_equity = row.get("debt_to_equity") if equity_positive else None

    return {
        # Metadata, not a metric -- deliberately outside METRIC_KEYS (see
        # metrics_from_normalized's docstring). Used by rank_against_peers
        # to detect/disclose a fiscal-year mismatch against the peer
        # snapshot's fixed year.
        "fy": fy,
        "revenue": revenue,
        "net_margin": row.get("net_margin"),
        "operating_margin": row.get("operating_margin"),
        "gross_margin": row.get("gross_margin"),
        "fcf_margin": row.get("fcf_margin"),
        "roe": roe,
        "roa": row.get("roa"),
        "debt_to_equity": debt_to_equity,
        "revenue_growth": row.get("yoy_revenue_growth"),
    }
