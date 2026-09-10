"""Fetch and cache daily Treasury par-yield-curve CSVs (home.treasury.gov).

Treasury's own data center publishes the exact same underlying series FRED's
``DGS10``/``DGS2``/``DGS30`` mirror (the H.15 constant-maturity Treasury
rates) and the nominal/TIPS pair FRED's ``T10YIE`` breakeven inflation is
built from, as a plain key-free CSV -- one file per calendar year, one row
per trading day, one column per maturity. This module exists purely as the
data-access layer :mod:`sec_analyzer.fetch.fred` falls back to when FRED
itself is unreachable (it is a real host outage on some networks, not a
throttling issue): see that module's docstring for the fallback chain and
why the two credit-spread series (``BAA10Y``, ``BAMLH0A0HYM2``) have no
Treasury substitute and must stay ``None`` rather than get a fabricated
proxy.

Two curve types are served, selected by the ``real`` flag:

- Nominal par yield curve (``real=False``): columns like ``"1 Mo"``,
  ``"2 Yr"``, ``"10 Yr"``, ``"30 Yr"``.
- Real (TIPS) par yield curve (``real=True``): columns like ``"5 YR"``,
  ``"10 YR"``, ``"30 YR"`` (Treasury capitalizes "YR" differently between
  the two endpoints -- maturity matching in this module is case/whitespace
  insensitive, see :func:`_normalize_maturity`).

Treasury is a third-party host, not SEC EDGAR, so it is fetched with a plain
``requests`` call and a normal browser-style User-Agent, exactly like
:mod:`sec_analyzer.fetch.fred`.

This module never raises: any failure (offline, HTTP error, unparseable
body, missing maturity column, no observation on/before the requested date)
is logged and returns ``None``/an empty list so the caller can fall back
further (to the archived ERP/risk-free CSV, ultimately).
"""

import csv
import io
import logging
import os
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import requests

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Treasury's daily par yield curve CSV endpoint. ``{year}`` covers exactly
#: one calendar year per request -- there is no multi-year query parameter.
TREASURY_NOMINAL_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
#: Treasury's daily real (TIPS) par yield curve CSV endpoint. Used to derive
#: breakeven inflation (nominal - real) as a fallback for FRED's ``T10YIE``.
TREASURY_REAL_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)

#: Canonical maturity labels this module's callers ask for. Matching against
#: a CSV's actual header is done case/whitespace-insensitively (see
#: :func:`_normalize_maturity`), so ``"10 Yr"`` also matches the real curve's
#: ``"10 YR"`` header spelling.
MATURITY_2Y = "2 Yr"
MATURITY_10Y = "10 Yr"
MATURITY_30Y = "30 Yr"

#: A normal browser-style User-Agent (see module docstring).
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: A calendar year that has already ended never gets a new row -- Treasury
#: does not revise published daily rates -- so a cached closed-year file can
#: be kept effectively forever. The current year's file gains a new row each
#: trading day, so it gets the same 24h freshness window `fetch/fred.py`
#: uses.
_CURRENT_YEAR_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60
#: Effectively permanent (50 years) -- a closed year's published rates are
#: never revised, so there is no real staleness window to bound.
_CLOSED_YEAR_CACHE_MAX_AGE_SECONDS = 50 * 365 * 24 * 60 * 60

#: Sentinel cutoff meaning "no real as_of cutoff -- take the latest
#: available observation" (mirrors ``fetch.fred._LATEST_SENTINEL_ISO``).
_LATEST_SENTINEL_ISO = "9999-12-31"

#: Values Treasury's CSV uses to mark "no observation" for a cell, beyond a
#: simple blank string.
_MISSING_MARKERS = ("", "N/A", "n/a", "NA")


def _cache_path(year: int, real: bool) -> str:
    """Return the on-disk cache path for one calendar year's curve CSV."""
    kind = "real" if real else "nominal"
    return os.path.join(Config.RAW_DIR, f"treasury_{kind}_{year}.csv")


def _is_closed_year(year: int, today: Optional[date] = None) -> bool:
    """Return True if ``year`` has fully ended relative to ``today``."""
    return year < (today or date.today()).year


def _is_cache_fresh(path: str, year: int) -> bool:
    """Return True if ``path`` exists and is within its freshness window.

    A closed year's file gets a long (effectively permanent) TTL; the
    current year's file gets the same 24h window as ``fetch/fred.py``,
    since it gains a new row on every trading day.
    """
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    max_age = (
        _CLOSED_YEAR_CACHE_MAX_AGE_SECONDS
        if _is_closed_year(year)
        else _CURRENT_YEAR_CACHE_MAX_AGE_SECONDS
    )
    return age < max_age


def _fetch_csv(year: int, real: bool) -> Optional[str]:
    """Download one calendar year's curve CSV text, or ``None`` on failure."""
    url_template = TREASURY_REAL_URL if real else TREASURY_NOMINAL_URL
    url = url_template.format(year=year)
    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        logger.warning(
            "treasury: request failed for year %s (real=%s)", year, real, exc_info=True
        )
        return None
    text = response.text or ""
    if "," not in text:
        logger.warning(
            "treasury: unusable response for year %s (real=%s, no CSV body).", year, real
        )
        return None
    return text


def _load_year_text(year: int, real: bool, no_cache: bool) -> Optional[str]:
    """Return the raw CSV text for one calendar year, via cache or fetch.

    Mirrors :func:`sec_analyzer.fetch.fred._load_series_text`'s cache/fetch/
    stale-fallback pattern, parameterized by (year, real) instead of series.
    """
    path = _cache_path(year, real)

    text: Optional[str] = None
    if not no_cache and _is_cache_fresh(path, year):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            logger.warning("treasury: failed to read cache %s", path, exc_info=True)
            text = None

    if text is None:
        text = _fetch_csv(year, real)
        if text is None:
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
                logger.warning("treasury: failed to write cache %s", path, exc_info=True)

    return text


def _normalize_maturity(label: Optional[str]) -> str:
    """Normalize a maturity column label for case/whitespace-insensitive
    matching (``"10 Yr"``, ``"10 YR"``, ``"10YR"`` all normalize the same)."""
    if not label:
        return ""
    return "".join(ch for ch in label.upper() if ch.isalnum())


def _mmddyyyy_to_iso(value: Optional[str]) -> Optional[str]:
    """Parse Treasury's ``MM/DD/YYYY`` date format to ISO ``YYYY-MM-DD``."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%m/%d/%Y").date().isoformat()
    except (ValueError, TypeError):
        return None


def _parse_rows(text: str) -> List[Tuple[str, Dict[str, float]]]:
    """Parse one year's curve CSV into ``(iso_date, {label: value})`` rows.

    Tolerates quoted headers, blank cells, ``"N/A"`` markers, and a missing
    or malformed maturity column (that maturity is simply absent from the
    row's dict rather than raising). ``label`` keys are the raw header text
    (stripped, original casing) -- callers matching a specific maturity
    should compare via :func:`_normalize_maturity`. Returns an empty list
    (never raises) for unparseable or too-short CSV text.
    """
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except csv.Error:
        logger.warning("treasury: could not parse CSV text", exc_info=True)
        return []
    if len(rows) < 2:
        return []

    header = rows[0]
    if len(header) < 2:
        return []
    labels = [(cell or "").strip() for cell in header[1:]]

    result: List[Tuple[str, Dict[str, float]]] = []
    for row in rows[1:]:
        if not row:
            continue
        date_iso = _mmddyyyy_to_iso(row[0])
        if date_iso is None:
            continue
        values: Dict[str, float] = {}
        for idx, label in enumerate(labels, start=1):
            if idx >= len(row) or not label:
                continue
            cell = (row[idx] or "").strip()
            if cell in _MISSING_MARKERS:
                continue
            try:
                values[label] = float(cell)
            except ValueError:
                continue
        result.append((date_iso, values))
    return result


def get_maturity_series(
    maturity: str, start_year: int, end_year: int, real: bool = False, no_cache: bool = False
) -> List[Tuple[str, float]]:
    """Return every observation for one maturity across a range of years.

    Args:
        maturity: A maturity label, e.g. :data:`MATURITY_10Y`. Matched
            case/whitespace-insensitively against each year's CSV header.
        start_year: First calendar year to include (inclusive).
        end_year: Last calendar year to include (inclusive). Order of
            ``start_year``/``end_year`` does not matter.
        real: If True, read the real (TIPS) curve instead of the nominal one.
        no_cache: If True, bypass the on-disk per-year cache and re-fetch.

    Returns:
        A list of ``(date_iso, value_pct)`` tuples, sorted chronologically
        ascending, one entry per trading day that has that maturity column
        populated. Empty list if the maturity is never found or every year's
        fetch fails. Never raises.
    """
    try:
        return _get_maturity_series(maturity, start_year, end_year, real, no_cache)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("get_maturity_series(%r) failed unexpectedly; returning [].", maturity)
        return []


def _get_maturity_series(
    maturity: str, start_year: int, end_year: int, real: bool, no_cache: bool
) -> List[Tuple[str, float]]:
    target = _normalize_maturity(maturity)
    lo, hi = min(start_year, end_year), max(start_year, end_year)

    out: List[Tuple[str, float]] = []
    for year in range(lo, hi + 1):
        text = _load_year_text(year, real, no_cache)
        if text is None:
            continue
        for date_iso, values in _parse_rows(text):
            for label, value in values.items():
                if _normalize_maturity(label) == target:
                    out.append((date_iso, value))
                    break

    out.sort(key=lambda pair: pair[0])
    return out


def get_curve_asof(as_of=None, real: bool = False, no_cache: bool = False) -> Optional[dict]:
    """Return one trading day's full par yield curve, as of ``as_of``.

    Args:
        as_of: The point-in-time date (``datetime.date`` or ISO
            ``"YYYY-MM-DD"`` string), or ``None`` for the latest available
            trading day.
        real: If True, read the real (TIPS) curve instead of the nominal one.
        no_cache: If True, bypass the on-disk per-year cache and re-fetch.

    Returns:
        ``{"date": "YYYY-MM-DD", "maturities": {"10 Yr": 4.69, ...}, "real":
        bool, "source": "Treasury daily nominal/real yield curve"}``, or
        ``None`` if no trading day on/before ``as_of`` is available. Never
        raises.
    """
    try:
        return _get_curve_asof(as_of, real, no_cache)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("get_curve_asof() failed unexpectedly; returning None.")
        return None


def _get_curve_asof(as_of, real: bool, no_cache: bool) -> Optional[dict]:
    if as_of is None:
        cutoff_iso = _LATEST_SENTINEL_ISO
        cutoff_year = date.today().year
    else:
        cutoff_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
        try:
            cutoff_year = int(str(cutoff_iso)[:4])
        except ValueError:
            cutoff_year = date.today().year

    # Fetch the cutoff year and the one before it, so an as_of early in
    # January (before that year's first trading day is published) still
    # walks back correctly into December of the prior year.
    rows_by_date: Dict[str, Dict[str, float]] = {}
    for year in (cutoff_year - 1, cutoff_year):
        text = _load_year_text(year, real, no_cache)
        if text is None:
            continue
        for date_iso, values in _parse_rows(text):
            if date_iso > cutoff_iso:
                continue
            rows_by_date.setdefault(date_iso, {}).update(values)

    if not rows_by_date:
        return None

    best_date = max(rows_by_date)
    kind = "real" if real else "nominal"
    return {
        "date": best_date,
        "maturities": rows_by_date[best_date],
        "real": real,
        "source": f"Treasury daily {kind} yield curve",
    }
