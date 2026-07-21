"""Fetch and cache Wall Street consensus analyst price targets.

This is a **display-only cross-check**, not a valuation input: the numbers
returned here are shown next to the deterministic fair-value range so a
reader can see how the engine's own math compares to what sell-side analysts
publish, but they never feed into the valuation engine, triangulation, or any
other computed output (see ``sec_analyzer/valuation/SPEC.md``, which this
module does not touch and is not bound by).

The only source is the optional ``yfinance`` package's ``Ticker.info`` dict.
Stooq (the primary source for :mod:`sec_analyzer.fetch.prices`) has no
analyst-target data at all, so there is no first-choice/fallback pair here
like there is for price history -- yfinance is the sole source, and its
absence (or failure) simply means no consensus figure is shown.
"""

import json
import logging
import os
import time
from typing import Optional

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Cache freshness window, in seconds (24 hours) -- mirrors
#: :mod:`sec_analyzer.fetch.prices`'s ``_CACHE_MAX_AGE_SECONDS``.
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60


def _cache_path(ticker: str) -> str:
    """Return the on-disk cache path for ``ticker``'s analyst targets."""
    return os.path.join(Config.RAW_DIR, f"analyst_{ticker.upper()}.json")


def _is_cache_fresh(path: str) -> bool:
    """Return True if ``path`` exists and was modified within the last 24h."""
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < _CACHE_MAX_AGE_SECONDS


def _load_cache(path: str) -> Optional[dict]:
    """Load a previously cached analyst-target JSON file.

    Returns ``None`` (rather than raising) if the file is missing, unreadable,
    or not valid JSON -- a corrupt cache must not be fatal, it should just
    trigger a re-fetch.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - a corrupt cache file must not be fatal
        logger.warning("Failed to load analyst-target cache at %s; will re-fetch.", path, exc_info=True)
        return None


def _write_cache(path: str, data: dict) -> None:
    """Serialize ``data`` to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _coerce_float(value) -> Optional[float]:
    """Best-effort ``float(value)``, or ``None`` if it can't be coerced."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value) -> Optional[int]:
    """Best-effort ``int(value)``, or ``None`` if it can't be coerced."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_analyst_dict(info: dict) -> Optional[dict]:
    """Extract and coerce the analyst-target fields out of a yfinance ``info`` dict.

    Returns ``None`` if ``target_mean`` is missing or non-positive -- a
    consensus target with no usable mean is worthless as a cross-check.
    """
    target_mean = _coerce_float(info.get("targetMeanPrice"))
    if target_mean is None or target_mean <= 0:
        return None

    return {
        "target_mean": target_mean,
        "target_high": _coerce_float(info.get("targetHighPrice")),
        "target_low": _coerce_float(info.get("targetLowPrice")),
        "target_median": _coerce_float(info.get("targetMedianPrice")),
        "num_analysts": _coerce_int(info.get("numberOfAnalystOpinions")),
        "currency": info.get("currency") if isinstance(info.get("currency"), str) else None,
        "recommendation": (
            info.get("recommendationKey") if isinstance(info.get("recommendationKey"), str) else None
        ),
        "source": "yfinance",
    }


def get_analyst_targets(ticker: str, no_cache: bool = False) -> Optional[dict]:
    """Fetch (or load from cache) consensus analyst price targets for ``ticker``.

    Display-only: the returned dict is meant to be shown alongside the
    deterministic fair-value range as a sanity cross-check, never consumed by
    the valuation engine, triangulation, or any other computed output. No
    date/timestamp is stamped into the dict, so it stays deterministic with
    respect to its own inputs (the caller stamps a display date elsewhere,
    e.g. the verdict card's ``date.today()`` header).

    Never raises: yfinance being uninstalled, the network call failing, the
    response having no usable target, or any other error all result in a
    logged warning/info and a ``None`` return -- the caller renders that as
    "no analyst data available", never a crash.

    Args:
        ticker: Stock ticker symbol, e.g. ``"AAPL"``.
        no_cache: When True, bypass any existing cache and re-fetch,
            overwriting the cache file on success.

    Returns:
        A dict with keys ``target_mean``, ``target_high``, ``target_low``,
        ``target_median`` (floats or ``None``), ``num_analysts`` (int or
        ``None``), ``currency`` (str or ``None``), ``recommendation`` (str or
        ``None``), and ``source`` (always ``"yfinance"``) -- or ``None`` if
        no usable consensus target could be obtained.
    """
    Config.ensure_dirs()
    ticker = ticker.strip().upper()
    path = _cache_path(ticker)

    if not no_cache and _is_cache_fresh(path):
        cached = _load_cache(path)
        if cached is not None:
            logger.info("Analyst-target cache hit for %s: %s", ticker, path)
            return cached
        logger.warning("Cached analyst-target file for %s failed to load; re-fetching.", ticker)

    try:
        import yfinance as yf
    except ImportError:
        logger.info("yfinance is not installed; no analyst-target data is available for %s.", ticker)
        return None

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001 - any yfinance failure just means no consensus data
        logger.warning("yfinance analyst-target request failed for %s", ticker, exc_info=True)
        return None

    analyst = _build_analyst_dict(info)
    if analyst is None:
        logger.info("No usable analyst target (targetMeanPrice) available for %s.", ticker)
        return None

    try:
        _write_cache(path, analyst)
        logger.info("Fetched analyst targets for %s from yfinance; cached to %s", ticker, path)
    except Exception:  # noqa: BLE001 - a cache-write failure must not lose the fetched data
        logger.warning("Failed to write analyst-target cache for %s at %s", ticker, path, exc_info=True)

    return analyst
