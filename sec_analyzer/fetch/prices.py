"""Download and cache daily OHLCV price history for a ticker.

Primary source is Stooq's free, no-key CSV endpoint. Stooq is intentionally
not fetched through :class:`sec_analyzer.http_client.SecHttpClient`: that
client's throttling and User-Agent policy exist specifically to satisfy SEC
EDGAR's fair-access rules and are not relevant (and would be misleading) for
a third-party market-data host. A plain ``requests`` call with a normal
browser-style User-Agent is used instead.

If Stooq is unavailable or returns something unusable (an HTML error page,
an empty body, or too few rows), this module falls back to the optional
``yfinance`` package when it is installed. If neither source yields usable
data, :class:`PriceDataError` is raised with a message intended to be shown
directly to a user.
"""

import io
import logging
import os
import time

import pandas as pd
import requests

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Stooq's free daily-history CSV endpoint. ``symbol`` must be the ticker in
#: lowercase with a market suffix, e.g. ``"aapl.us"``.
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

#: A normal browser-style User-Agent. Stooq has no fair-access identity
#: requirement like SEC EDGAR does; this just avoids looking like a bare
#: script to generic bot filters.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: Columns expected in a successful Stooq CSV response.
_EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

#: Minimum number of rows required for a fetched price history to be
#: considered usable, from either source.
_MIN_ROWS = 30

#: Cache freshness window, in seconds (24 hours).
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

#: yfinance ``period`` string used as the default lookback for the fallback
#: path. ``"max"`` returns the *entire* available daily history for the
#: ticker -- mirroring the Stooq path (see :data:`STOOQ_URL`), which already
#: returns full history because no ``d1``/``d2`` range params are sent.
#: Valuation's historical-multiples percentiles (see
#: ``sec_analyzer.valuation.multiples``) want 10-15 years of year-end
#: prices, so both sources default to "as much as is available" rather than
#: a short fixed window.
_YFINANCE_DEFAULT_PERIOD = "max"

#: Cache filename suffix. Bumped from the unsuffixed name used before wide
#: history was fetched, so any pre-existing cache file written under the old
#: ~2-year window (yfinance's old hardcoded ``period="2y"``) is simply never
#: read again -- it's a different filename, so it's silently superseded by
#: a fresh full-history fetch on next use instead of being (incorrectly)
#: trusted as-is or requiring manual cache-busting.
_CACHE_SUFFIX = "_full"


class PriceDataError(Exception):
    """Raised when no usable price history could be obtained for a ticker."""


def _cache_path(ticker: str) -> str:
    """Return the on-disk cache path for ``ticker``'s price history."""
    return os.path.join(Config.RAW_DIR, f"prices_{ticker.upper()}{_CACHE_SUFFIX}.csv")


def _load_cache(path: str) -> pd.DataFrame:
    """Load a previously cached price-history CSV, indexed by Date."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def _write_cache(path: str, df: pd.DataFrame) -> None:
    """Write a price-history DataFrame (Date as index) to ``path`` as CSV."""
    df.to_csv(path, index_label="Date")


def _is_cache_fresh(path: str) -> bool:
    """Return True if ``path`` exists and was modified within the last 24h."""
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < _CACHE_MAX_AGE_SECONDS


def _validate_frame(df: pd.DataFrame) -> bool:
    """Return True if ``df`` looks like a usable OHLCV price history."""
    if df is None or df.empty:
        return False
    missing_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in df.columns]
    if missing_cols:
        return False
    return len(df) >= _MIN_ROWS


def _fetch_stooq(ticker: str) -> pd.DataFrame:
    """Fetch and parse daily price history from Stooq.

    Args:
        ticker: Stock ticker symbol, e.g. ``"AAPL"``.

    Returns:
        A DataFrame indexed by ``Date`` (ascending), with ``Open``/``High``/
        ``Low``/``Close``/``Volume`` columns.

    Raises:
        PriceDataError: If the request fails, or the response is not a
            usable CSV (HTML error page, empty body, or too few rows).
    """
    symbol = f"{ticker.strip().lower()}.us"
    url = STOOQ_URL.format(symbol=symbol)
    logger.info("Fetching price history for %s from Stooq: %s", ticker, url)

    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PriceDataError(f"Stooq request failed for {ticker}: {exc}") from exc

    text = response.text
    if not text or not text.startswith("Date,"):
        raise PriceDataError(
            f"Stooq returned an unusable response for {ticker} "
            "(not a CSV -- likely an unknown symbol or an HTML error page)."
        )

    try:
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
    except Exception as exc:  # noqa: BLE001 - surface any parse failure uniformly
        raise PriceDataError(f"Stooq CSV for {ticker} could not be parsed: {exc}") from exc

    if not _validate_frame(df):
        raise PriceDataError(
            f"Stooq returned too little data for {ticker} "
            f"({len(df)} row(s); need at least {_MIN_ROWS})."
        )

    df = df.set_index("Date").sort_index()
    return df


def _fetch_yfinance(ticker: str, period: str = _YFINANCE_DEFAULT_PERIOD) -> pd.DataFrame:
    """Fetch daily price history from the optional ``yfinance`` fallback.

    Args:
        ticker: Stock ticker symbol, e.g. ``"AAPL"``.
        period: yfinance ``period`` string (e.g. ``"max"``, ``"15y"``,
            ``"2y"``). Defaults to :data:`_YFINANCE_DEFAULT_PERIOD` ("max"),
            which returns the full available daily history rather than a
            short fixed window.

    Returns:
        A DataFrame indexed by ``Date`` (ascending), with ``Open``/``High``/
        ``Low``/``Close``/``Volume`` columns.

    Raises:
        PriceDataError: If ``yfinance`` is not installed, the fetch fails,
            or the result has too few rows.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise PriceDataError(
            "yfinance is not installed; no fallback price source is available."
        ) from exc

    logger.info("Fetching price history for %s from yfinance (period=%s)", ticker, period)
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
    except Exception as exc:  # noqa: BLE001 - any yfinance failure is a data-source failure
        raise PriceDataError(f"yfinance request failed for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise PriceDataError(f"yfinance returned no data for {ticker}.")

    # Newer yfinance versions return MultiIndex columns (field, ticker) even
    # for a single symbol. Flatten to just the field level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"

    if not _validate_frame(df):
        raise PriceDataError(
            f"yfinance returned too little data for {ticker} "
            f"({len(df)} row(s); need at least {_MIN_ROWS})."
        )

    df = df.sort_index()
    return df[[c for c in _EXPECTED_COLUMNS[1:] if c in df.columns]]


def get_price_history(
    ticker: str, no_cache: bool = False, yfinance_period: str = _YFINANCE_DEFAULT_PERIOD
) -> "tuple[pd.DataFrame, str]":
    """Fetch (or load from cache) daily OHLCV price history for ``ticker``.

    Tries the on-disk cache first (unless ``no_cache``), then Stooq, then
    the optional ``yfinance`` fallback if Stooq fails or is unusable.

    Stooq's endpoint (see :data:`STOOQ_URL`) is called with no ``d1``/``d2``
    range params, so it always returns the full daily history it has --
    there is no lookback window to widen on that path. ``yfinance_period``
    only affects the fallback path.

    Args:
        ticker: Stock ticker symbol, e.g. ``"AAPL"``.
        no_cache: When True, bypass any existing cache and re-fetch,
            overwriting the cache file on success.
        yfinance_period: yfinance ``period`` string used only if the
            Stooq fetch fails and the ``yfinance`` fallback is used.
            Defaults to :data:`_YFINANCE_DEFAULT_PERIOD` ("max" -- full
            available history). Valuation's historical-multiples
            percentiles want 10-15 years of year-end prices; pass e.g.
            ``"15y"`` to cap the fallback fetch instead.

    Returns:
        A ``(df, source)`` tuple. ``df`` is indexed by ``Date`` (ascending)
        with ``Open``/``High``/``Low``/``Close``/``Volume`` columns.
        ``source`` identifies where the data came from: ``"stooq"``,
        ``"yfinance"``, ``"cache(stooq)"``, or ``"cache(yfinance)"``.

    Raises:
        PriceDataError: If neither the cache, Stooq, nor yfinance yield a
            usable price history.
    """
    Config.ensure_dirs()
    ticker = ticker.strip().upper()
    path = _cache_path(ticker)

    if not no_cache and _is_cache_fresh(path):
        try:
            df = _load_cache(path)
            if _validate_frame(df):
                logger.info(
                    "price cache hit for %s: %s (%d rows)", ticker, path, len(df)
                )
                # The cache doesn't record which upstream source produced it,
                # so report it generically as a Stooq-origin cache hit (the
                # only source this module writes to cache).
                return df, "cache(stooq)"
            logger.warning("Cached price file for %s failed validation; re-fetching.", ticker)
        except Exception:  # noqa: BLE001 - a corrupt cache file must not be fatal
            logger.warning("Failed to load price cache for %s at %s; re-fetching.", ticker, path, exc_info=True)

    stooq_error = None
    try:
        df = _fetch_stooq(ticker)
    except PriceDataError as exc:
        stooq_error = exc
        df = None

    if df is not None:
        _write_cache(path, df)
        logger.info("Fetched %d price rows for %s from stooq; cached to %s", len(df), ticker, path)
        return df, "stooq"

    logger.warning("Stooq failed (%s); using yfinance fallback", stooq_error)

    try:
        df = _fetch_yfinance(ticker, period=yfinance_period)
    except PriceDataError as exc:
        logger.error("Both Stooq and yfinance failed for %s: stooq=%s yfinance=%s", ticker, stooq_error, exc)
        raise PriceDataError(
            f"Could not obtain price history for {ticker!r} from Stooq or yfinance. "
            f"Stooq error: {stooq_error}. yfinance error: {exc}"
        ) from exc

    _write_cache(path, df)
    logger.info("Fetched %d price rows for %s from yfinance; cached to %s", len(df), ticker, path)
    return df, "yfinance"


def latest_price(df: pd.DataFrame) -> "tuple[float, str]":
    """Return the most recent Close price and its date.

    Args:
        df: A price-history DataFrame as returned by :func:`get_price_history`
            (Date index, ascending, with a ``Close`` column).

    Returns:
        A ``(price, as_of)`` tuple, where ``price`` is the last Close as a
        float and ``as_of`` is that row's date formatted ``"YYYY-MM-DD"``.
    """
    last = df.iloc[-1]
    as_of = df.index[-1]
    return float(last["Close"]), as_of.strftime("%Y-%m-%d")
