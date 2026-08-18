"""Rates/macro access layer: FRED primary, Treasury.gov fallback.

Originally this module only fetched FRED's 10-Year Treasury constant-
maturity rate (DGS10) for point-in-time ("as-of") backtest mode. It has
since been generalized twice:

1. To serve a small panel of yield-curve and credit-spread series (see
   :data:`MACRO_PANEL`) and to work in *live* mode: ``as_of=None`` means
   "the latest available observation", which is what lets a live
   (non-backtest) run pull a real current risk-free rate instead of always
   falling back to the archived Damodaran ERP/risk-free CSV.
2. To fall back to :mod:`sec_analyzer.fetch.treasury` (home.treasury.gov's
   own key-free daily par-yield-curve CSV) whenever FRED itself does not
   answer. This is not a throttling workaround -- on some networks
   ``fred.stlouisfed.org`` is simply unreachable while ``home.treasury.gov``
   and ``sec.gov`` are fine -- so a bare "FRED down -> None" would silently
   make this whole module inert on those networks.

The fallback is *faithful* for the three yield-curve series and the
breakeven-inflation series: ``DGS10``/``DGS2``/``DGS30`` are the exact same
H.15 constant-maturity-Treasury series Treasury itself publishes, and
``T10YIE`` is the exact construction FRED uses (nominal 10y minus real/TIPS
10y). It is deliberately **not** faithful for the two credit-spread series
(``BAA10Y``, ``BAMLH0A0HYM2``): Treasury publishes no Moody's Baa or ICE
BofA HY index data, so there is no substitute to fall back to, and this
module returns ``None`` for them rather than inventing a proxy that would
silently corrupt anything computed from a credit spread (e.g.
:func:`sec_analyzer.signals.macro.build_macro_context`'s ``credit_regime``).

Every returned dict carries a ``"provider"`` key (``"FRED"`` or
``"Treasury"``) and a ``"source"`` string that names the *actual* origin
(e.g. ``"FRED DGS10"`` vs. ``"Treasury CMT 10 Yr"``), because this value
flows into :mod:`sec_analyzer.valuation.damodaran`'s
``risk_free_source``, which is printed in the valuation notes -- a number
whose origin is unstated is a number nobody can check.

FRED and Treasury are both third-party hosts, not SEC EDGAR, so they are
fetched with plain ``requests`` calls and a normal browser-style User-Agent
rather than through :class:`sec_analyzer.http_client.SecHttpClient` (whose
throttling/UA policy exists to satisfy EDGAR's fair-access rules and would
be misleading here).

This module never raises: any failure (offline, HTTP error, unparseable
body, no observation on/before the requested date, on both FRED and its
Treasury fallback) is logged and returns ``None`` so the caller can fall
back further, to the archived ERP/risk-free values. When fetching a panel
of several series, one series failing never prevents the others from being
returned.
"""

import csv
import io
import logging
import os
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

import requests

from sec_analyzer.config import Config
from sec_analyzer.fetch import treasury

logger = logging.getLogger(__name__)

#: FRED's free CSV download endpoint. No API key required.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

#: 10-Year Treasury constant maturity -- the risk-free rate the valuation
#: engine's CAPM and terminal-growth anchor read.
SERIES_DGS10 = "DGS10"
#: 2-Year Treasury constant maturity -- short end of the curve.
SERIES_DGS2 = "DGS2"
#: 30-Year Treasury constant maturity -- long end of the curve.
SERIES_DGS30 = "DGS30"
#: 10-Year breakeven inflation: the inflation rate the TIPS market prices in.
SERIES_T10YIE = "T10YIE"
#: Moody's Baa corporate yield MINUS the 10y Treasury -- investment-grade
#: credit spread, already published as a spread (not a yield).
SERIES_BAA10Y = "BAA10Y"
#: ICE BofA US High Yield index option-adjusted spread.
SERIES_HY_OAS = "BAMLH0A0HYM2"

#: The macro panel fetched by :func:`get_macro_panel` by default.
MACRO_PANEL: Tuple[str, ...] = (
    SERIES_DGS10,
    SERIES_DGS2,
    SERIES_DGS30,
    SERIES_T10YIE,
    SERIES_BAA10Y,
    SERIES_HY_OAS,
)

#: Kept for backward compatibility with any code still referencing the
#: single-series constant this module used to expose.
_SERIES = SERIES_DGS10

#: FRED series -> Treasury nominal-curve maturity label, for the three
#: series that are the exact same H.15 CMT data under a different publisher.
#: ``T10YIE`` is handled separately (:func:`_treasury_breakeven_asof`) since
#: it is a derived spread, not a single maturity column. ``BAA10Y`` and
#: :data:`SERIES_HY_OAS` are deliberately absent -- see module docstring.
_TREASURY_MATURITY_FALLBACK: Dict[str, str] = {
    SERIES_DGS10: treasury.MATURITY_10Y,
    SERIES_DGS2: treasury.MATURITY_2Y,
    SERIES_DGS30: treasury.MATURITY_30Y,
}

#: A normal browser-style User-Agent (see module docstring).
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: Cache freshness window, in seconds (24 hours). The series is append-only
#: history, so a stale cache is harmless for historical dates; the window just
#: bounds how often a same-day run re-fetches the tail.
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

#: Sentinel cutoff used internally to mean "no as_of given -- take the latest
#: available observation". No real FRED observation will ever be dated this
#: far out, so reusing the on/before-cutoff walk-back logic with this sentinel
#: is equivalent to "the newest row with a usable value".
_LATEST_SENTINEL_ISO = "9999-12-31"

#: A percentile computed against fewer than this many historical observations
#: is not reported (too small a window to mean anything); the field is
#: ``None`` rather than a number derived from a handful of points.
_MIN_PERCENTILE_WINDOW_N = 30


def _cache_path(series: str) -> str:
    """Return the on-disk cache path for a FRED series CSV."""
    return os.path.join(Config.RAW_DIR, f"fred_{series}.csv")


def _is_cache_fresh(path: str) -> bool:
    """Return True if ``path`` exists and was modified within the last 24h."""
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < _CACHE_MAX_AGE_SECONDS


def _fetch_csv(series: str) -> Optional[str]:
    """Download the raw CSV text for ``series`` from FRED, or ``None`` on failure."""
    url = FRED_CSV_URL.format(series=series)
    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("fred: request failed for series %s", series, exc_info=True)
        return None
    text = response.text or ""
    if "," not in text:
        logger.warning("fred: unusable response for series %s (no CSV body).", series)
        return None
    return text


def _load_series_text(series: str, no_cache: bool) -> Optional[str]:
    """Return the raw CSV text for ``series``, via cache or a fresh fetch.

    Reads a fresh on-disk cache if present (unless ``no_cache``), otherwise
    fetches from FRED and writes the cache; if the fetch fails, falls back to
    a stale cache rather than failing outright. Returns ``None`` only when
    neither a usable cache nor a fetch is available.
    """
    path = _cache_path(series)

    text: Optional[str] = None
    if not no_cache and _is_cache_fresh(path):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            logger.warning("fred: failed to read cache %s", path, exc_info=True)
            text = None

    if text is None:
        text = _fetch_csv(series)
        if text is None:
            # Fall back to a stale cache if one exists (better than nothing).
            if not no_cache and os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        text = fh.read()
                except OSError:
                    return None
            else:
                return None
        else:
            try:
                os.makedirs(Config.RAW_DIR, exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(text)
            except OSError:
                logger.warning("fred: failed to write cache %s", path, exc_info=True)

    return text


def _parse_asof(text: str, series: str, as_of_iso: str) -> Optional[dict]:
    """Pick the last observation dated on/before ``as_of_iso`` from CSV text.

    Tolerates both ``DATE,DGS10`` (legacy) and ``observation_date,DGS10``
    (current) header names, and skips FRED's ``"."`` missing-value markers and
    weekends/holidays by walking backward from the cutoff. Returns ``None`` if
    nothing on/before the cutoff is parseable.

    ``as_of_iso`` may be :data:`_LATEST_SENTINEL_ISO` to mean "no real
    cutoff": since no FRED observation is ever dated that far in the future,
    the walk-back naturally lands on the newest available observation.
    """
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except csv.Error:
        logger.warning("fred: could not parse CSV for series %s", series, exc_info=True)
        return None
    if len(rows) < 2:
        return None

    best_date: Optional[str] = None
    best_value: Optional[float] = None
    for row in rows[1:]:  # skip header
        if len(row) < 2:
            continue
        date_str = (row[0] or "").strip()
        value_str = (row[1] or "").strip()
        if not date_str or value_str in ("", "."):
            continue
        if date_str > as_of_iso:
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        # Rows are chronological; keep the latest date <= cutoff.
        if best_date is None or date_str > best_date:
            best_date = date_str
            best_value = value

    if best_date is None or best_value is None:
        return None
    return {
        "value_pct": best_value,
        "date": best_date,
        "series": series,
        "source": f"FRED {series}",
    }


def _valid_observations(text: str, series: str) -> List[Tuple[str, float]]:
    """Return every ``(date, value)`` pair in ``text`` that parses cleanly.

    Skips FRED's ``"."`` missing-value markers and any malformed row. Order
    is not assumed or guaranteed; callers that need chronological order or a
    bounded window should filter/sort the result themselves. Returns an empty
    list (never raises) on unparseable CSV.
    """
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except csv.Error:
        logger.warning("fred: could not parse CSV for series %s", series, exc_info=True)
        return []
    if len(rows) < 2:
        return []

    observations: List[Tuple[str, float]] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        date_str = (row[0] or "").strip()
        value_str = (row[1] or "").strip()
        if not date_str or value_str in ("", "."):
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        observations.append((date_str, value))
    return observations


def _window_start_iso(reference_date_iso: str, years: int) -> Optional[str]:
    """Return the ISO date ``years`` before ``reference_date_iso``, or ``None``."""
    try:
        reference = date.fromisoformat(reference_date_iso)
    except ValueError:
        return None
    try:
        start = reference.replace(year=reference.year - years)
    except ValueError:
        # reference is Feb 29 and (year - years) is not a leap year.
        start = reference.replace(year=reference.year - years, day=28)
    return start.isoformat()


def _percentile_from_observations(
    observations: List[Tuple[str, float]],
    current_value: float,
    reference_date_iso: str,
    percentile_years: int,
) -> dict:
    """Rank ``current_value`` within its own trailing history.

    Why a self-anchored percentile rather than a fixed threshold: "is this
    yield/spread high?" has no universal constant, and inventing one would be
    exactly the kind of unjustified parameter this project's ROADMAP forbids.
    Ranking the value within its own recent history is deterministic, needs
    no judgment call, and is honest about the window it was measured against
    (hence this function also returns the window bounds, not just the
    number).

    Percentile definition mirrors
    :func:`sec_analyzer.valuation.multiples.percentile_position`: percentage
    of window observations strictly less than ``current_value``, plus half
    the percentage of values tied with it (midrank treatment of ties).

    Args:
        observations: ``(date_iso, value)`` pairs, any order, from any
            provider -- FRED's own CSV or Treasury's per-year CSVs.

    Returns:
        ``{"percentile", "percentile_years", "window_start", "window_n",
        "min_pct", "max_pct"}``. ``percentile`` is ``None`` when the window
        holds fewer than :data:`_MIN_PERCENTILE_WINDOW_N` observations;
        ``min_pct``/``max_pct`` are ``None`` only when the window is empty.
    """
    window_start = _window_start_iso(reference_date_iso, percentile_years)
    result = {
        "percentile": None,
        "percentile_years": percentile_years,
        "window_start": window_start,
        "window_n": 0,
        "min_pct": None,
        "max_pct": None,
    }
    if window_start is None:
        return result

    window_values = [
        value
        for date_str, value in observations
        if window_start <= date_str <= reference_date_iso
    ]
    result["window_n"] = len(window_values)
    if not window_values:
        return result

    result["min_pct"] = min(window_values)
    result["max_pct"] = max(window_values)

    if len(window_values) < _MIN_PERCENTILE_WINDOW_N:
        return result

    less_count = sum(1 for v in window_values if v < current_value)
    equal_count = sum(1 for v in window_values if v == current_value)
    pct = (less_count + 0.5 * equal_count) / len(window_values) * 100.0
    result["percentile"] = round(pct, 1)
    return result


def _compute_percentile(
    text: str, series: str, current_value: float, reference_date_iso: str, percentile_years: int
) -> dict:
    """Rank ``current_value`` within its own trailing history in FRED CSV ``text``.

    Thin wrapper over :func:`_percentile_from_observations` for the FRED
    path, kept under its original name/signature (existing tests call it
    directly with raw CSV text).
    """
    observations = _valid_observations(text, series)
    return _percentile_from_observations(observations, current_value, reference_date_iso, percentile_years)


def get_series_asof(
    series: str,
    as_of=None,
    no_cache: bool = False,
    percentile_years: int = 10,
) -> Optional[dict]:
    """Return one FRED series observation as of ``as_of``, or the latest.

    Args:
        series: A FRED series id, e.g. :data:`SERIES_DGS10`.
        as_of: The point-in-time date (``datetime.date`` or ISO
            ``"YYYY-MM-DD"`` string). The last observation dated on/before
            this date is returned (handles weekends/holidays). ``None``
            (the default) means "no cutoff": return the latest available
            observation -- this is the live-mode path.
        no_cache: If True, bypass the on-disk cache and re-fetch.
        percentile_years: Width, in years, of the trailing window the
            returned percentile is measured against.

    Returns:
        ``{"value_pct": float, "date": "YYYY-MM-DD", "series": series,
        "source": <Turkish-free source string naming the actual origin>,
        "provider": "FRED" or "Treasury", "percentile": float or None,
        "percentile_years": int, "window_start": "YYYY-MM-DD" or None,
        "window_n": int, "min_pct": float or None, "max_pct": float or None}``
        where ``value_pct`` is a percentage number (e.g. ``2.98`` for 2.98%)
        and ``percentile`` is the observation's 0-100 midrank position within
        its own trailing ``percentile_years``-year window (``None`` if fewer
        than 30 observations fall in that window). If FRED does not answer,
        transparently falls back to :mod:`sec_analyzer.fetch.treasury` for
        ``DGS10``/``DGS2``/``DGS30``/``T10YIE`` (``"provider"`` becomes
        ``"Treasury"`` and ``"source"`` names the Treasury series used).
        ``BAA10Y``/:data:`SERIES_HY_OAS` have no Treasury substitute, so a
        FRED failure for either returns ``None`` -- see module docstring.
        Returns ``None`` if no provider has data on/before ``as_of``. Never
        raises.
    """
    try:
        return _get_series_asof(series, as_of, no_cache, percentile_years)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("get_series_asof(%r) failed unexpectedly; returning None.", series)
        return None


def _get_series_asof(
    series: str, as_of, no_cache: bool, percentile_years: int
) -> Optional[dict]:
    fred_result = _get_series_asof_fred(series, as_of, no_cache, percentile_years)
    if fred_result is not None:
        fred_result["provider"] = "FRED"
        return fred_result
    return _get_series_asof_treasury_fallback(series, as_of, no_cache, percentile_years)


def _get_series_asof_fred(
    series: str, as_of, no_cache: bool, percentile_years: int
) -> Optional[dict]:
    if as_of is None:
        cutoff_iso = _LATEST_SENTINEL_ISO
    else:
        cutoff_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)

    text = _load_series_text(series, no_cache)
    if text is None:
        return None

    base = _parse_asof(text, series, cutoff_iso)
    if base is None:
        return None

    percentile_info = _compute_percentile(
        text, series, base["value_pct"], base["date"], percentile_years
    )
    base.update(percentile_info)
    return base


def _cutoff_iso_and_year(as_of):
    """Return ``(cutoff_iso, cutoff_year)`` for the Treasury fallback path."""
    if as_of is None:
        return _LATEST_SENTINEL_ISO, date.today().year
    cutoff_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
    try:
        cutoff_year = int(str(cutoff_iso)[:4])
    except ValueError:
        cutoff_year = date.today().year
    return cutoff_iso, cutoff_year


def _get_series_asof_treasury_fallback(
    series: str, as_of, no_cache: bool, percentile_years: int
) -> Optional[dict]:
    """Try Treasury for the series that have a faithful substitute.

    Returns ``None`` immediately (without ever calling into
    :mod:`sec_analyzer.fetch.treasury`) for a series with no substitute --
    this is what lets a test assert the Treasury module was never invoked
    for ``BAA10Y``/``BAMLH0A0HYM2``.
    """
    if series == SERIES_T10YIE:
        return _treasury_breakeven_asof(as_of, no_cache, percentile_years)

    maturity = _TREASURY_MATURITY_FALLBACK.get(series)
    if maturity is None:
        return None
    return _treasury_maturity_asof(
        series, maturity, as_of, no_cache, percentile_years, source=f"Treasury CMT {maturity}"
    )


def _treasury_maturity_asof(
    series: str, maturity: str, as_of, no_cache: bool, percentile_years: int, source: str
) -> Optional[dict]:
    cutoff_iso, cutoff_year = _cutoff_iso_and_year(as_of)
    start_year = cutoff_year - percentile_years - 1

    observations = treasury.get_maturity_series(
        maturity, start_year, cutoff_year, real=False, no_cache=no_cache
    )
    eligible = [pair for pair in observations if pair[0] <= cutoff_iso]
    if not eligible:
        return None
    best_date, best_value = max(eligible, key=lambda pair: pair[0])

    result = {
        "value_pct": best_value,
        "date": best_date,
        "series": series,
        "source": source,
        "provider": "Treasury",
    }
    result.update(
        _percentile_from_observations(observations, best_value, best_date, percentile_years)
    )
    return result


def _treasury_breakeven_asof(as_of, no_cache: bool, percentile_years: int) -> Optional[dict]:
    """Fall back for ``T10YIE`` via nominal-10y minus real(TIPS)-10y.

    This is the exact construction FRED itself uses for breakeven inflation
    -- not an approximation -- so the ``"source"`` string says "breakeven"
    rather than naming a single Treasury column.
    """
    cutoff_iso, cutoff_year = _cutoff_iso_and_year(as_of)
    start_year = cutoff_year - percentile_years - 1

    nominal = dict(
        treasury.get_maturity_series(
            treasury.MATURITY_10Y, start_year, cutoff_year, real=False, no_cache=no_cache
        )
    )
    real = dict(
        treasury.get_maturity_series(
            treasury.MATURITY_10Y, start_year, cutoff_year, real=True, no_cache=no_cache
        )
    )
    common_dates = set(nominal) & set(real)
    if not common_dates:
        return None
    breakeven_obs = [(d, round(nominal[d] - real[d], 2)) for d in common_dates]

    eligible = [pair for pair in breakeven_obs if pair[0] <= cutoff_iso]
    if not eligible:
        return None
    best_date, best_value = max(eligible, key=lambda pair: pair[0])

    result = {
        "value_pct": best_value,
        "date": best_date,
        "series": SERIES_T10YIE,
        "source": "Treasury 10Y breakeven (nominal - TIPS)",
        "provider": "Treasury",
    }
    result.update(
        _percentile_from_observations(breakeven_obs, best_value, best_date, percentile_years)
    )
    return result


def get_risk_free_asof(as_of, no_cache: bool = False) -> Optional[dict]:
    """Return the DGS10 risk-free rate as of ``as_of``, or the latest.

    Thin wrapper over :func:`get_series_asof` for :data:`SERIES_DGS10`, kept
    under its original name/signature since it is called from ``cli.py``,
    ``backtest/runner.py``, ``calibrate.py`` and ``web/app.py``.

    Args:
        as_of: The point-in-time date (``datetime.date`` or ISO
            ``"YYYY-MM-DD"`` string), or ``None`` for the latest available
            observation (the live-mode path).
        no_cache: If True, bypass the on-disk cache and re-fetch.

    Returns:
        Same shape as :func:`get_series_asof`, i.e. at least
        ``{"value_pct": float, "date": "YYYY-MM-DD", "series": "DGS10",
        "source": ..., "provider": "FRED" or "Treasury"}`` plus percentile
        fields, or ``None`` if neither provider has data on/before ``as_of``.
        Falls back to Treasury's nominal 10y CMT yield when FRED is
        unreachable -- the exact same underlying series under a different
        publisher (see module docstring). Never raises.
    """
    return get_series_asof(SERIES_DGS10, as_of=as_of, no_cache=no_cache)


def get_macro_panel(
    as_of=None, no_cache: bool = False, series: Tuple[str, ...] = MACRO_PANEL
) -> Dict[str, Optional[dict]]:
    """Fetch a panel of FRED series, each independently.

    Args:
        as_of: Forwarded to :func:`get_series_asof` for every series in the
            panel; ``None`` means "latest available observation".
        no_cache: Forwarded to :func:`get_series_asof`.
        series: The series ids to fetch. Defaults to :data:`MACRO_PANEL`.

    Returns:
        ``{series_id: result_dict_or_None}`` -- one entry per requested
        series id. A single series failing (network error, no observation
        on/before ``as_of``, etc.) never prevents the others from being
        returned: that series' value is simply ``None``. Each entry's
        ``"provider"`` key is independent -- e.g. when FRED is unreachable
        the panel can come back with ``DGS10``/``DGS2``/``DGS30``/``T10YIE``
        sourced from Treasury while ``BAA10Y``/``BAMLH0A0HYM2`` are ``None``
        (no Treasury substitute exists for either). Always returns a dict
        (possibly with every value ``None``), never ``None``, never raises.
    """
    panel: Dict[str, Optional[dict]] = {}
    for series_id in series or ():
        try:
            panel[series_id] = get_series_asof(series_id, as_of=as_of, no_cache=no_cache)
        except Exception:  # noqa: BLE001 - one series must never sink the rest
            logger.exception(
                "get_macro_panel: series %s failed unexpectedly; recording None.", series_id
            )
            panel[series_id] = None
    return panel
