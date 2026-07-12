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
        The parsed companyfacts JSON document.
    """
    Config.ensure_dirs()
    cache_path = os.path.join(Config.RAW_DIR, f"CIK{cik}.json")

    if os.path.exists(cache_path) and not no_cache:
        logger.info("Company facts cache hit for CIK %s: %s", cik, cache_path)
        return _read_cache(cache_path)

    url = COMPANYFACTS_URL.format(cik=cik)
    logger.info("Fetching company facts for CIK %s from %s", cik, url)
    data = client.get_json(url)

    _write_cache(cache_path, data)
    logger.debug("Wrote company facts cache: %s", cache_path)

    return data


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
        The parsed submissions JSON document.
    """
    Config.ensure_dirs()
    cache_path = os.path.join(Config.RAW_DIR, f"submissions_CIK{cik}.json")

    if os.path.exists(cache_path) and not no_cache:
        logger.info("Submissions cache hit for CIK %s: %s", cik, cache_path)
        return _read_cache(cache_path)

    url = SUBMISSIONS_URL.format(cik=cik)
    logger.info("Fetching submissions for CIK %s from %s", cik, url)
    data = client.get_json(url)

    _write_cache(cache_path, data)
    logger.debug("Wrote submissions cache: %s", cache_path)

    return data


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
