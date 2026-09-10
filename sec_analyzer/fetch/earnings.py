"""Fetch and cache recent quarterly earnings-surprise ("beat/miss") history.

This is a **display-only cross-check**, not a valuation input -- exactly like
:mod:`sec_analyzer.fetch.analyst`. The numbers here (consensus EPS estimate vs.
the actual reported EPS, and the resulting surprise percentage) are shown in
the web UI's balance-sheet tab so a reader can see whether a company has been
beating or missing Wall Street expectations. They never feed the deterministic
valuation engine, triangulation, or any other computed output, so this module
is not bound by ``sec_analyzer/valuation/SPEC.md``.

The only source is the optional ``yfinance`` package's
``Ticker.get_earnings_history()`` (backed by Yahoo's ``quoteSummary`` JSON
API -- no HTML scraping / ``lxml`` needed). yfinance being uninstalled or the
call failing simply means no beat/miss history is shown; it is never fatal.

The surprise percentage is (re)computed here from ``epsActual``/``epsEstimate``
rather than trusting yfinance's own ``surprisePercent`` column, whose scaling
(fraction vs. percent) is inconsistent across responses -- computing it
ourselves keeps the output deterministic with respect to the two EPS figures.
"""

import json
import logging
import math
import os
import time
from typing import List, Optional

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Cache freshness window, in seconds (24 hours) -- mirrors
#: :mod:`sec_analyzer.fetch.analyst`'s ``_CACHE_MAX_AGE_SECONDS``.
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

#: Number of most-recent quarters to retain (Yahoo typically returns 4).
_MAX_QUARTERS = 8


def _cache_path(ticker: str) -> str:
    """Return the on-disk cache path for ``ticker``'s earnings history."""
    return os.path.join(Config.RAW_DIR, f"earnings_{ticker.upper()}.json")


def _is_cache_fresh(path: str) -> bool:
    """Return True if ``path`` exists and was modified within the last 24h."""
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < _CACHE_MAX_AGE_SECONDS


def _load_cache(path: str) -> Optional[dict]:
    """Load a previously cached earnings-history JSON file.

    Returns ``None`` (rather than raising) if the file is missing, unreadable,
    or not valid JSON -- a corrupt cache must not be fatal, it should just
    trigger a re-fetch.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - a corrupt cache file must not be fatal
        logger.warning("Failed to load earnings-history cache at %s; will re-fetch.", path, exc_info=True)
        return None


def _write_cache(path: str, data: dict) -> None:
    """Serialize ``data`` to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _coerce_float(value) -> Optional[float]:
    """Best-effort ``float(value)``; ``None`` for anything unparseable or a
    non-finite (``NaN``/``inf``) result. Guarding against ``NaN`` matters here
    (unlike for analyst targets): a missing estimate/actual arrives as pandas
    ``NaN``, and a raw ``NaN`` would serialize to invalid JSON that the
    browser's ``JSON.parse`` rejects."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _period_label(index_value) -> str:
    """Render a DataFrame index value (a quarter date) as ``YYYY-MM-DD``.

    yfinance indexes ``earnings_history`` by a ``pandas.Timestamp``; fall back
    to a trimmed string for any other index type so a non-date label (e.g. a
    quarter code like ``"1Q2024"``) still renders rather than crashing.
    """
    if hasattr(index_value, "strftime"):
        try:
            return index_value.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001 - fall back to str() for odd index types
            pass
    return str(index_value)[:10]


def _surprise_pct(actual: Optional[float], estimate: Optional[float]) -> Optional[float]:
    """Percentage EPS surprise: ``(actual - estimate) / |estimate| * 100``.

    ``None`` when either figure is missing or the estimate is zero (a division
    that would be undefined or meaningless). Uses ``abs(estimate)`` in the
    denominator so the sign of the surprise reflects beat (positive) vs. miss
    (negative) even when the estimate itself is negative (an expected loss).
    """
    if actual is None or estimate is None or estimate == 0:
        return None
    return (actual - estimate) / abs(estimate) * 100.0


def _build_quarters(df) -> List[dict]:
    """Convert a yfinance ``earnings_history`` DataFrame into a JSON-friendly
    list of per-quarter beat/miss records, newest first.

    Expected columns (yfinance): ``epsEstimate``, ``epsActual``. Any row with
    neither a usable estimate nor a usable actual is dropped. Never raises: a
    malformed frame just yields fewer (or zero) rows.
    """
    quarters: List[dict] = []
    try:
        rows = list(df.iterrows())
    except Exception:  # noqa: BLE001 - a non-iterable / malformed frame -> no data
        return quarters

    for index_value, row in rows:
        try:
            estimate = _coerce_float(row.get("epsEstimate"))
            actual = _coerce_float(row.get("epsActual"))
        except Exception:  # noqa: BLE001 - a malformed row must not abort the whole parse
            continue
        if estimate is None and actual is None:
            continue
        quarters.append(
            {
                "period": _period_label(index_value),
                "eps_estimate": estimate,
                "eps_actual": actual,
                "surprise_pct": _surprise_pct(actual, estimate),
            }
        )

    # Newest first, capped -- Yahoo usually returns oldest-first and only 4
    # rows, but sort defensively so the contract doesn't depend on that.
    quarters.sort(key=lambda q: q.get("period") or "", reverse=True)
    return quarters[:_MAX_QUARTERS]


def get_earnings_history(ticker: str, no_cache: bool = False) -> Optional[dict]:
    """Fetch (or load from cache) recent quarterly EPS beat/miss history.

    Display-only: the returned dict is meant to be shown alongside the
    financials in the web UI's balance-sheet tab as a "has this company been
    beating expectations?" cross-check, never consumed by the valuation engine
    or any other computed output.

    Never raises: yfinance being uninstalled, the network call failing, the
    response being empty/malformed, or any other error all result in a logged
    warning/info and a ``None`` return -- the caller renders that as "no
    earnings-surprise data available", never a crash.

    Args:
        ticker: Stock ticker symbol, e.g. ``"AAPL"``.
        no_cache: When True, bypass any existing cache and re-fetch,
            overwriting the cache file on success.

    Returns:
        A dict ``{"quarters": [{"period", "eps_estimate", "eps_actual",
        "surprise_pct"}, ...], "source": "yfinance"}`` with quarters newest
        first, or ``None`` if no usable history could be obtained.
    """
    Config.ensure_dirs()
    ticker = ticker.strip().upper()
    path = _cache_path(ticker)

    if not no_cache and _is_cache_fresh(path):
        cached = _load_cache(path)
        if cached is not None:
            logger.info("Earnings-history cache hit for %s: %s", ticker, path)
            return cached
        logger.warning("Cached earnings-history file for %s failed to load; re-fetching.", ticker)

    try:
        import yfinance as yf
    except ImportError:
        logger.info("yfinance is not installed; no earnings-history data is available for %s.", ticker)
        return None

    try:
        df = yf.Ticker(ticker).get_earnings_history()
    except Exception:  # noqa: BLE001 - any yfinance failure just means no beat/miss data
        logger.warning("yfinance earnings-history request failed for %s", ticker, exc_info=True)
        return None

    if df is None:
        logger.info("No earnings-history data available for %s.", ticker)
        return None

    quarters = _build_quarters(df)
    if not quarters:
        logger.info("Earnings-history response for %s had no usable quarters.", ticker)
        return None

    result = {"quarters": quarters, "source": "yfinance"}

    try:
        _write_cache(path, result)
        logger.info("Fetched earnings history for %s from yfinance; cached to %s", ticker, path)
    except Exception:  # noqa: BLE001 - a cache-write failure must not lose the fetched data
        logger.warning("Failed to write earnings-history cache for %s at %s", ticker, path, exc_info=True)

    return result
