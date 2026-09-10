"""Download and cache SEC XBRL company facts, submissions, and concept data.

This module wraps three SEC EDGAR JSON endpoints:

* ``companyfacts`` -- every XBRL fact SEC has extracted for a filer, keyed by
  taxonomy and tag.
* ``submissions`` -- a filer's metadata and filing history.
* ``companyconcept`` -- the time series for a single us-gaap tag for a
  filer (a narrower, cheaper alternative to pulling the full companyfacts
  document).

All fetches go through a shared on-disk JSON cache under
``Config.RAW_DIR`` so repeated runs against the same CIK avoid hitting SEC's
servers again.
"""

import json
import logging
import os
import time

import requests

from sec_analyzer.config import Config
from sec_analyzer.http_client import SecHttpClient

logger = logging.getLogger(__name__)

#: Full XBRL "company facts" document for a filer (all tags, all periods).
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

#: Filer metadata and filing history.
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

#: Single us-gaap tag time series for a filer.
COMPANYCONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
)


def _read_cache(path: str) -> dict:
    """Read and parse a cached JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_cache(path: str, data: dict) -> None:
    """Serialize ``data`` to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _cache_is_fresh(path: str, ttl_hours: float) -> bool:
    """Whether ``path`` was written within ``ttl_hours`` (SPEC.md Sec.21c).

    A non-positive ``ttl_hours`` means "never expires" -- the behavior every
    cache here had before this check existed, kept as an escape hatch. An
    unreadable mtime is treated as stale, so a broken cache re-fetches.
    """
    if ttl_hours <= 0:
        return True
    try:
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
    except OSError:
        return False
    return age_hours < ttl_hours


def _load_or_fetch(
    cache_path: str, url: str, client: SecHttpClient, no_cache: bool,
    ttl_hours: float, what: str, cik: str,
) -> dict:
    """Return a cached document if it exists and is fresh, else re-fetch.

    When a STALE cache exists but the re-fetch fails (network down, SEC 5xx),
    the stale copy is returned with a warning rather than propagating the
    error: a month-old document still supports an analysis, whereas a raised
    exception kills the whole run. That preserves the offline/degraded
    operation the previous exists-only check gave for free.
    """
    cached = os.path.exists(cache_path) and not no_cache
    if cached and _cache_is_fresh(cache_path, ttl_hours):
        logger.info("%s cache hit for CIK %s: %s", what, cik, cache_path)
        return _read_cache(cache_path)

    if cached:
        logger.info(
            "%s cache for CIK %s is older than %.0fh; re-fetching from SEC.",
            what, cik, ttl_hours,
        )

    logger.info("Fetching %s for CIK %s from %s", what.lower(), cik, url)
    try:
        data = client.get_json(url)
    except Exception:  # noqa: BLE001 - a stale document beats no analysis
        if cached:
            logger.warning(
                "%s re-fetch for CIK %s failed; falling back to the stale "
                "cache at %s.", what, cik, cache_path, exc_info=True,
            )
            return _read_cache(cache_path)
        raise

    _write_cache(cache_path, data)
    logger.debug("Wrote %s cache: %s", what.lower(), cache_path)
    return data


def get_company_facts(
    cik: str, client: SecHttpClient, no_cache: bool = False
) -> dict:
    """Fetch (or load from cache) the full XBRL companyfacts document.

    Args:
        cik: 10-digit, zero-padded CIK string, e.g. ``"0000320193"``.
        client: HTTP client used to fetch the document when not cached.
        no_cache: When True, bypass any existing cache and re-fetch from SEC,
            overwriting the cache file.

    Returns:
        The parsed companyfacts JSON document. A cache older than
        ``Config.COMPANYFACTS_CACHE_TTL_HOURS`` is re-fetched (SPEC.md
        Sec.21c); if that re-fetch fails, the stale copy is returned.
    """
    Config.ensure_dirs()
    return _load_or_fetch(
        os.path.join(Config.RAW_DIR, f"CIK{cik}.json"),
        COMPANYFACTS_URL.format(cik=cik),
        client, no_cache, Config.COMPANYFACTS_CACHE_TTL_HOURS,
        what="Company facts", cik=cik,
    )


def get_submissions(
    cik: str, client: SecHttpClient, no_cache: bool = False
) -> dict:
    """Fetch (or load from cache) a filer's submissions/filing history.

    Args:
        cik: 10-digit, zero-padded CIK string, e.g. ``"0000320193"``.
        client: HTTP client used to fetch the document when not cached.
        no_cache: When True, bypass any existing cache and re-fetch from SEC,
            overwriting the cache file.

    Returns:
        The parsed submissions JSON document. A cache older than
        ``Config.SUBMISSIONS_CACHE_TTL_HOURS`` is re-fetched (SPEC.md
        Sec.21c) -- the earnings catalyst is derived from this document, so a
        stale copy silently hides a just-published earnings 8-K. If the
        re-fetch fails, the stale copy is returned.
    """
    Config.ensure_dirs()
    return _load_or_fetch(
        os.path.join(Config.RAW_DIR, f"submissions_CIK{cik}.json"),
        SUBMISSIONS_URL.format(cik=cik),
        client, no_cache, Config.SUBMISSIONS_CACHE_TTL_HOURS,
        what="Submissions", cik=cik,
    )


def get_company_concept(cik: str, tag: str, client: SecHttpClient) -> dict | None:
    """Fetch a single us-gaap concept's time series for a filer.

    Unlike :func:`get_company_facts` and :func:`get_submissions`, this call
    is not cached to disk, since it targets one narrow tag rather than a
    filer's whole document.

    Args:
        cik: 10-digit, zero-padded CIK string, e.g. ``"0000320193"``.
        tag: us-gaap XBRL tag name, e.g. ``"Assets"``.
        client: HTTP client used to fetch the concept document.

    Returns:
        The parsed companyconcept JSON document, or ``None`` if SEC has no
        data for that tag for this filer (HTTP 404).
    """
    url = COMPANYCONCEPT_URL.format(cik=cik, tag=tag)
    logger.info("Fetching concept %r for CIK %s from %s", tag, cik, url)

    try:
        return client.get_json(url)
    except requests.HTTPError as err:
        if err.response is not None and err.response.status_code == 404:
            logger.debug("Concept %r not reported for CIK %s (404)", tag, cik)
            return None
        raise
