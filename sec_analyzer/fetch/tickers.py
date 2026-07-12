"""Ticker-to-CIK resolution against SEC's company_tickers.json index.

The SEC publishes a single JSON file mapping every registered ticker symbol
to its CIK (Central Index Key) and company title. This module downloads that
file (with on-disk caching), builds a ticker lookup, and exposes a single
public helper, :func:`resolve_cik`, for turning a ticker symbol into the
10-digit, zero-padded CIK string that the rest of the SEC EDGAR APIs expect.
"""

import json
import logging
import os

from sec_analyzer.config import Config
from sec_analyzer.http_client import SecHttpClient

logger = logging.getLogger(__name__)

#: SEC's canonical ticker -> CIK/title index.
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

#: On-disk cache path for the downloaded ticker index.
TICKERS_CACHE = os.path.join(Config.RAW_DIR, "company_tickers.json")


def _load_ticker_index(client: SecHttpClient, no_cache: bool = False) -> dict:
    """Return the raw ``company_tickers.json`` payload, using the disk cache.

    The upstream file is a JSON object keyed by stringified integer indices
    (``"0"``, ``"1"``, ...), each value being a dict with ``cik_str``,
    ``ticker``, and ``title`` keys.

    Args:
        client: HTTP client used to fetch the index when the cache is
            absent or bypassed.
        no_cache: When True, always fetch fresh data from SEC and overwrite
            the cache file.

    Returns:
        The parsed JSON object as a dict.
    """
    Config.ensure_dirs()

    if os.path.exists(TICKERS_CACHE) and not no_cache:
        logger.debug("Loading ticker index from cache: %s", TICKERS_CACHE)
        with open(TICKERS_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Fetching ticker index from %s", COMPANY_TICKERS_URL)
    data = client.get_json(COMPANY_TICKERS_URL)

    with open(TICKERS_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.debug("Wrote ticker index cache: %s", TICKERS_CACHE)

    return data


def resolve_cik(
    ticker: str, client: SecHttpClient, no_cache: bool = False
) -> tuple[str, str]:
    """Resolve a stock ticker symbol to its SEC CIK and company title.

    Args:
        ticker: Stock ticker symbol, e.g. ``"AAPL"``. Matching is
            case-insensitive and surrounding whitespace is stripped.
        client: HTTP client used to fetch the ticker index if it is not
            already cached on disk.
        no_cache: When True, bypass any existing on-disk cache and re-fetch
            the ticker index from SEC.

    Returns:
        A ``(cik, title)`` tuple where ``cik`` is the 10-digit, zero-padded
        CIK string (e.g. ``"0000320193"``) and ``title`` is the company's
        registered name (e.g. ``"Apple Inc."``).

    Raises:
        ValueError: If ``ticker`` does not appear in the SEC ticker index.
    """
    index = _load_ticker_index(client, no_cache=no_cache)

    lookup = {
        entry["ticker"].strip().upper(): entry for entry in index.values()
    }

    key = ticker.strip().upper()
    entry = lookup.get(key)
    if entry is None:
        raise ValueError(
            f"Ticker {ticker!r} not found in SEC company_tickers.json"
        )

    cik = str(entry["cik_str"]).zfill(10)
    title = entry["title"]
    logger.debug("Resolved ticker %r -> CIK %s (%s)", ticker, cik, title)

    return cik, title
