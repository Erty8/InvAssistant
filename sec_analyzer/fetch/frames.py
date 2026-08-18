"""Download and cache SEC XBRL "frames" -- one concept across every filer.

Unlike :mod:`sec_analyzer.fetch.companyfacts` (one filer, every concept),
SEC's Frames API inverts the axis: it returns **one us-gaap concept for
every filer that reported it** in a single calendar period, e.g. every
``Revenues`` figure any filer tagged for ``CY2024`` in one JSON document.
That is exactly the shape a cross-sectional peer comparison needs, and it is
the only SEC endpoint that provides it without fetching every peer's full
companyfacts document individually.

Scope boundary (non-negotiable, see ``sec_analyzer/screener/peers.py`` for
the consumer): this module -- and everything built on it -- is a **context
layer**. Its output is reported alongside a filer's valuation and never
enters the fair-value computation, the triangulation signals, or
``sector_ratio``. ``sec_analyzer/valuation/SPEC.md`` remains the sole
binding contract for those; it is not amended by this module.

Endpoint: ``https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json``.
Response shape (trimmed to what matters here)::

    {"taxonomy": "us-gaap", "tag": "Revenues", "ccp": "CY2024", "uom": "USD",
     "pts": 4321,
     "data": [{"accn": "...", "cik": 320193, "entityName": "Apple Inc.",
                "loc": "US-CA", "start": "...", "end": "...",
                "val": 391035000000}, ...]}

Every public function here follows the same never-fatal posture as the rest
of the ``fetch`` package: a 404 (tag not reported for that period -- a
routine outcome, not every tag exists for every period) is logged at debug
and returns ``None``/``{}``; any other failure is logged as a warning and
also degrades to ``None``/``{}`` rather than raising.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests

from sec_analyzer.config import Config
from sec_analyzer.http_client import SecHttpClient

logger = logging.getLogger(__name__)

#: One us-gaap (or other taxonomy) concept, one unit, one period, every filer.
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"

#: A closed calendar period's frame still drifts as filers amend, but slowly
#: -- a month is long enough to avoid re-pulling a multi-MB document on every
#: run and short enough that restatements land within a quarter.
FRAMES_CACHE_TTL_HOURS = 720.0


def annual_period(year: int) -> str:
    """Return the Frames API period code for a full calendar year.

    Args:
        year: Calendar year, e.g. ``2024``.

    Returns:
        ``"CY2024"`` -- the duration period used for income-statement and
        cash-flow-statement concepts (SPEC.md "annual" bucket equivalent).
    """
    return f"CY{int(year)}"


def instant_period(year: int, quarter: int = 4) -> str:
    """Return the Frames API period code for a point-in-time snapshot.

    Args:
        year: Calendar year, e.g. ``2024``.
        quarter: Calendar quarter (1-4) whose quarter-end the snapshot is
            taken at. Defaults to 4 (year-end), the natural anchor for a
            filer's most recently completed fiscal year balance sheet.

    Returns:
        ``"CY2024Q4I"`` -- the instant period used for balance-sheet
        concepts (``StockholdersEquity``, ``Assets``, ``Liabilities``, ...).
    """
    return f"CY{int(year)}Q{int(quarter)}I"


def _frames_dir() -> str:
    """Return the on-disk cache directory for reduced frame documents,
    creating it if missing."""
    path = os.path.join(Config.RAW_DIR, "frames")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(taxonomy: str, tag: str, unit: str, period: str) -> str:
    """Cache path for one ``(taxonomy, tag, unit, period)`` frame."""
    return os.path.join(_frames_dir(), f"{taxonomy}_{tag}_{unit}_{period}.json")


def _cache_is_fresh(path: str, ttl_hours: float) -> bool:
    """Whether ``path`` was written within ``ttl_hours`` (mirrors
    ``fetch.companyfacts._cache_is_fresh``).

    A non-positive ``ttl_hours`` means "never expires". An unreadable mtime
    is treated as stale, so a broken cache re-fetches rather than trusting a
    file it cannot even stat.
    """
    if ttl_hours <= 0:
        return True
    try:
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
    except OSError:
        return False
    return age_hours < ttl_hours


def _read_cache(path: str) -> Optional[dict]:
    """Read a cached reduced-frame JSON file. ``None`` on any failure
    (missing file, corrupt JSON) -- a broken cache must trigger a re-fetch,
    never a raised exception."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - a corrupt cache file must not be fatal
        logger.warning("Frame cache at %s is unreadable; will re-fetch.", path, exc_info=True)
        return None


def _write_cache(path: str, data: dict) -> None:
    """Serialize ``data`` (the reduced ``{"pts", "values"}`` shape) to
    ``path``. Failures are logged, not raised -- losing a cache write must
    not lose the fetched data itself."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:  # noqa: BLE001 - a cache-write failure must not be fatal
        logger.warning("Failed to write frame cache at %s", path, exc_info=True)


def _reduce(raw: dict) -> dict:
    """Reduce SEC's full frame response to ``{"pts": int, "values": {cik_str: val}}``.

    SEC's ``data`` array carries ``accn``/``entityName``/``loc``/``start``/
    ``end`` per row -- none of which this project uses -- across potentially
    thousands of filers, making the full document multi-MB. Only ``cik`` and
    ``val`` matter downstream, so only those survive the cache.

    Duplicate CIKs within one frame (a filer reporting the same concept
    twice for the same period) keep the FIRST occurrence encountered, so the
    result is deterministic regardless of how SEC orders ``data``. The count
    of dropped duplicates is logged at debug.
    """
    data = raw.get("data") or []
    values: Dict[str, float] = {}
    duplicates = 0
    for item in data:
        cik = item.get("cik")
        val = item.get("val")
        if cik is None or val is None:
            continue
        key = str(int(cik))
        if key in values:
            duplicates += 1
            continue
        values[key] = val

    if duplicates:
        logger.debug("get_frame: dropped %d duplicate CIK entr%s (kept first occurrence).",
                     duplicates, "y" if duplicates == 1 else "ies")

    return {"pts": raw.get("pts"), "values": values}


def _finalize(reduced: dict, tag: str, period: str, unit: str, taxonomy: str) -> dict:
    """Build the public :func:`get_frame` return shape from a reduced
    (cached-or-fresh) ``{"pts", "values"}`` dict, converting the JSON-only
    string CIK keys back to ints."""
    raw_values = reduced.get("values") or {}
    values: Dict[int, float] = {}
    for key, val in raw_values.items():
        try:
            values[int(key)] = val
        except (TypeError, ValueError):
            continue
    return {
        "tag": tag,
        "period": period,
        "unit": unit,
        "taxonomy": taxonomy,
        "pts": reduced.get("pts"),
        "values": values,
    }


def _get_frame(
    tag: str, period: str, client: SecHttpClient, unit: str, taxonomy: str, no_cache: bool,
) -> Optional[dict]:
    Config.ensure_dirs()
    path = _cache_path(taxonomy, tag, unit, period)

    cached = None
    if os.path.exists(path) and not no_cache:
        cached = _read_cache(path)
        if cached is not None and _cache_is_fresh(path, FRAMES_CACHE_TTL_HOURS):
            logger.debug("Frame cache hit for %s/%s/%s/%s", taxonomy, tag, unit, period)
            return _finalize(cached, tag, period, unit, taxonomy)
        if cached is not None:
            logger.info(
                "Frame cache for %s/%s/%s/%s is older than %.0fh; re-fetching from SEC.",
                taxonomy, tag, unit, period, FRAMES_CACHE_TTL_HOURS,
            )

    url = FRAMES_URL.format(taxonomy=taxonomy, tag=tag, unit=unit, period=period)
    logger.info("Fetching frame %s/%s/%s/%s from %s", taxonomy, tag, unit, period, url)

    try:
        raw = client.get_json(url)
    except requests.HTTPError as err:
        if err.response is not None and err.response.status_code == 404:
            # A normal outcome: not every tag exists for every period. No
            # retry storm -- SecHttpClient already does not retry a 404, and
            # neither do we.
            logger.debug("Frame not reported: %s/%s/%s/%s (404).", taxonomy, tag, unit, period)
        else:
            logger.warning(
                "Frame fetch failed for %s/%s/%s/%s", taxonomy, tag, unit, period, exc_info=True,
            )
        if cached is not None:
            logger.warning(
                "Frame re-fetch for %s/%s/%s/%s failed; falling back to the stale cache.",
                taxonomy, tag, unit, period,
            )
            return _finalize(cached, tag, period, unit, taxonomy)
        return None
    except Exception:  # noqa: BLE001 - get_frame() must never raise
        logger.warning(
            "Unexpected failure fetching frame %s/%s/%s/%s", taxonomy, tag, unit, period, exc_info=True,
        )
        if cached is not None:
            logger.warning(
                "Frame re-fetch for %s/%s/%s/%s failed; falling back to the stale cache.",
                taxonomy, tag, unit, period,
            )
            return _finalize(cached, tag, period, unit, taxonomy)
        return None

    reduced = _reduce(raw)
    _write_cache(path, reduced)
    return _finalize(reduced, tag, period, unit, taxonomy)


def get_frame(
    tag: str,
    period: str,
    client: SecHttpClient,
    unit: str = "USD",
    taxonomy: str = "us-gaap",
    no_cache: bool = False,
) -> Optional[dict]:
    """Fetch (or load from cache) one us-gaap concept for every filer in one period.

    Args:
        tag: A single us-gaap (or other taxonomy) tag, e.g. ``"Revenues"``.
            For the alias-aware version that merges several candidate tags
            for one canonical concept, see :func:`get_frame_with_aliases`.
        period: A Frames period code -- see :func:`annual_period` /
            :func:`instant_period`.
        client: HTTP client used to fetch the document when not cached.
        unit: XBRL unit key, e.g. ``"USD"`` (the default) or ``"USD/shares"``.
        taxonomy: XBRL taxonomy, ``"us-gaap"`` by default.
        no_cache: When True, bypass any existing cache and re-fetch from SEC.

    Returns:
        ``{"tag", "period", "unit", "taxonomy", "pts", "values"}`` where
        ``values`` is ``{cik_int: val_float}`` (reduced from SEC's much
        larger per-filer response -- entity names/accession numbers are
        dropped, they are not used downstream). ``None`` when the tag is not
        reported for that period (404) or on any other failure -- this
        function never raises. The on-disk cache
        (``Config.RAW_DIR/frames/{taxonomy}_{tag}_{unit}_{period}.json``)
        stores only the reduced ``{"pts", "values"}`` shape, refreshed every
        :data:`FRAMES_CACHE_TTL_HOURS`; a re-fetch that fails falls back to
        the stale cache (mirrors ``fetch.companyfacts``'s "stale beats
        nothing" behavior) and a corrupt cache file is treated as a cache
        miss rather than raised.
    """
    try:
        return _get_frame(tag, period, client, unit=unit, taxonomy=taxonomy, no_cache=no_cache)
    except Exception:  # noqa: BLE001 - get_frame() must never raise
        logger.warning(
            "get_frame() failed unexpectedly for %s/%s/%s/%s", taxonomy, tag, unit, period, exc_info=True,
        )
        return None


def get_frame_with_aliases(
    tags: List[str],
    period: str,
    client: SecHttpClient,
    unit: str = "USD",
    taxonomy: str = "us-gaap",
    no_cache: bool = False,
) -> Dict[int, float]:
    """Merge an ordered alias list (see ``normalize.concepts.CONCEPTS``) into
    one cross-sectional frame.

    Revenue in particular is split across several us-gaap tags depending on
    the filer's reporting style (``RevenueFromContractWithCustomerExcludingAssessedTax``,
    ``Revenues``, ``SalesRevenueNet``, ...); a single-tag frame silently
    omits whichever slice of the market tags it differently. Merging every
    alias -- earliest tag in the list wins per CIK, matching how
    ``normalize.normalizer`` resolves the same alias lists for one filer --
    is what makes the cross-section actually representative.

    Args:
        tags: Ordered alias list, most preferred first (as in
            ``normalize.concepts.CONCEPTS[...]``).
        period: A Frames period code -- see :func:`annual_period` /
            :func:`instant_period`.
        client: HTTP client used to fetch each alias's frame.
        unit: XBRL unit key.
        taxonomy: XBRL taxonomy.
        no_cache: Forwarded to each underlying :func:`get_frame` call.

    Returns:
        ``{cik_int: val_float}``, merged across every alias in ``tags`` (a
        CIK present under more than one alias keeps the value from the
        earliest alias in the list). ``{}`` if ``tags`` is empty or every
        alias misses (404 or failure). Never raises.
    """
    try:
        return _get_frame_with_aliases(tags, period, client, unit=unit, taxonomy=taxonomy, no_cache=no_cache)
    except Exception:  # noqa: BLE001 - get_frame_with_aliases() must never raise
        logger.warning("get_frame_with_aliases() failed unexpectedly for tags=%r/%s", tags, period, exc_info=True)
        return {}


def _get_frame_with_aliases(
    tags: List[str], period: str, client: SecHttpClient, unit: str, taxonomy: str, no_cache: bool,
) -> Dict[int, float]:
    merged: Dict[int, float] = {}
    contributions = []

    for tag in tags or []:
        frame = get_frame(tag, period, client, unit=unit, taxonomy=taxonomy, no_cache=no_cache)
        if frame is None:
            continue
        values = frame.get("values") or {}
        added = 0
        for cik, val in values.items():
            if cik not in merged:
                merged[cik] = val
                added += 1
        contributions.append((tag, added))

    if contributions:
        logger.debug(
            "get_frame_with_aliases: %s/%s resolved %d CIK(s) total (%s).",
            taxonomy, period, len(merged),
            ", ".join(f"{t}={n}" for t, n in contributions),
        )
    else:
        logger.debug(
            "get_frame_with_aliases: no alias resolved any data for %s/%s (tags=%r).",
            taxonomy, period, tags,
        )

    return merged
