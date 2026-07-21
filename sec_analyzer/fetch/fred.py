"""Fetch and cache the FRED 10-Year Treasury constant-maturity rate (DGS10).

Used by point-in-time ("as-of") mode to supply a historical risk-free rate:
the CAPM intercept as it stood on a past date. FRED's free CSV download
endpoint needs no API key.

Like :mod:`sec_analyzer.fetch.prices` (Stooq), FRED is a third-party host, not
SEC EDGAR, so it is fetched with a plain ``requests`` call and a normal
browser-style User-Agent rather than through
:class:`sec_analyzer.http_client.SecHttpClient` (whose throttling/UA policy
exists to satisfy EDGAR's fair-access rules and would be misleading here).

This module never raises: any failure (offline, HTTP error, unparseable body,
no observation on/before the requested date) is logged and returns ``None`` so
the caller can fall back to the archived ERP/risk-free values.
"""

import csv
import io
import logging
import os
import time
from typing import Optional

import requests

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: FRED's free CSV download endpoint. No API key required.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

#: 10-Year Treasury constant-maturity rate, in percent (e.g. 2.98 = 2.98%).
_SERIES = "DGS10"

#: A normal browser-style User-Agent (see module docstring).
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: Cache freshness window, in seconds (24 hours). The series is append-only
#: history, so a stale cache is harmless for historical dates; the window just
#: bounds how often a same-day run re-fetches the tail.
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60


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


def _parse_asof(text: str, series: str, as_of_iso: str) -> Optional[dict]:
    """Pick the last observation dated on/before ``as_of_iso`` from CSV text.

    Tolerates both ``DATE,DGS10`` (legacy) and ``observation_date,DGS10``
    (current) header names, and skips FRED's ``"."`` missing-value markers and
    weekends/holidays by walking backward from the cutoff. Returns ``None`` if
    nothing on/before the cutoff is parseable.
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


def get_risk_free_asof(as_of, no_cache: bool = False) -> Optional[dict]:
    """Return the DGS10 risk-free rate as of ``as_of``, or ``None``.

    Args:
        as_of: The point-in-time date (``datetime.date`` or ISO
            ``"YYYY-MM-DD"`` string). The last observation dated on/before
            this date is returned (handles weekends/holidays).
        no_cache: If True, bypass the on-disk cache and re-fetch.

    Returns:
        ``{"value_pct": float, "date": "YYYY-MM-DD", "series": "DGS10",
        "source": "FRED DGS10"}`` where ``value_pct`` is a percentage number
        (e.g. ``2.98`` for 2.98%), or ``None`` if the data is unavailable or
        no observation exists on/before ``as_of``. Never raises.
    """
    try:
        as_of_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
        path = _cache_path(_SERIES)

        text: Optional[str] = None
        if not no_cache and _is_cache_fresh(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                logger.warning("fred: failed to read cache %s", path, exc_info=True)
                text = None

        if text is None:
            text = _fetch_csv(_SERIES)
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

        return _parse_asof(text, _SERIES, as_of_iso)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("get_risk_free_asof() failed unexpectedly; returning None.")
        return None
