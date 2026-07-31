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

Both Stooq and yfinance can hand back a trailing (or, rarely, interior) bar
for an in-progress/unsettled session that carries a ``Volume`` estimate but
``NaN`` for ``Open``/``High``/``Low``/``Close``. Every frame this module
returns -- from cache, Stooq, or yfinance -- has such rows dropped before
being handed back (see :func:`_drop_unusable_bars`), so a caller can always
rely on ``Close`` being usable on every row of the returned frame.
"""

import io
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

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

#: Hour (UTC) after which the current weekday's US equity session is treated
#: as closed AND published. The cash close is 16:00 ET = 20:00 UTC in summer
#: (EDT) / 21:00 UTC in winter (EST); 22:00 covers both without needing a
#: timezone database, and leaves the provider an hour to publish the bar.
#: Erring late only costs one extra re-fetch attempt, never a wrong price.
_SESSION_PUBLISHED_HOUR_UTC = 22

#: Hard ceiling on how old a cache may be before it is refused even as a
#: last-resort fallback when every upstream source is failing.
_CACHE_ABSOLUTE_MAX_AGE_DAYS = 30

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


def _source_path(path: str) -> str:
    """Sidecar file recording which upstream produced a cached CSV."""
    return f"{path}.source"


def _write_cache(path: str, df: pd.DataFrame, source: Optional[str] = None) -> None:
    """Write a price-history DataFrame (Date as index) to ``path`` as CSV.

    ``source`` is recorded in a sidecar file so a later cache hit can report
    where the data actually came from. Without it the module reported every
    cache hit as ``"cache(stooq)"`` even when the bytes had come from the
    yfinance fallback -- which then surfaced in the HTML report's provenance
    line as a flat, sometimes false, "Stooq".
    """
    df.to_csv(path, index_label="Date")
    if source:
        try:
            with open(_source_path(path), "w", encoding="utf-8") as handle:
                handle.write(source)
        except OSError:  # provenance is nice-to-have, never fatal
            logger.debug("Could not write price-cache source marker for %s", path, exc_info=True)


def _read_cache_source(path: str) -> str:
    """Which upstream produced a cached CSV; ``"stooq"`` when unrecorded.

    Caches written before the sidecar existed have no marker, and Stooq is
    the primary source, so that is the honest default for them.
    """
    try:
        with open(_source_path(path), "r", encoding="utf-8") as handle:
            recorded = handle.read().strip()
        return recorded or "stooq"
    except OSError:
        return "stooq"


def last_completed_session(now_utc: Optional[datetime] = None) -> date:
    """The most recent weekday whose US equity session has closed and published.

    Deliberately calendar-light: weekends are skipped, market holidays are
    not known here. Treating a holiday as a session only costs one wasted
    re-fetch (the returned frame simply has no bar for it), whereas missing a
    real session hands back a stale price -- so the asymmetry is chosen on
    purpose.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    day = now_utc.date()
    if now_utc.hour < _SESSION_PUBLISHED_HOUR_UTC:
        day -= timedelta(days=1)
    while day.weekday() >= 5:  # Saturday / Sunday
        day -= timedelta(days=1)
    return day


def _cache_covers_last_session(df: pd.DataFrame, now_utc: Optional[datetime] = None) -> bool:
    """Whether a cached frame includes the last completed trading session.

    Replaces a fixed 24-hour age window, which was misaligned with the daily
    market cycle: a cache written at, say, 12:09 UTC -- before the US session
    had even opened -- stayed "fresh" for another 24 hours, straight past
    that day's close, so any run in the following morning silently served a
    price two sessions old. Observed on MU 2026-07-31: the report showed
    ``$739`` (the 07-29 close) while 07-30 had closed at ``$874.66``, an 18%
    error. Freshness is now a property of the DATA (does it contain the last
    closed session?), not of the file's mtime.
    """
    if df is None or df.empty:
        return False
    try:
        newest = df.index.max().date()
    except (AttributeError, ValueError):
        return False
    return newest >= last_completed_session(now_utc)


def _stale_cache_is_usable(path: str) -> bool:
    """Whether a stale cache is recent enough to serve as a last resort.

    Past :data:`_CACHE_ABSOLUTE_MAX_AGE_DAYS` the file describes a different
    market, so failing loudly beats quietly valuing a company off month-old
    prices.
    """
    try:
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
    except OSError:
        return False
    return age_days <= _CACHE_ABSOLUTE_MAX_AGE_DAYS


def _validate_frame(df: pd.DataFrame) -> bool:
    """Return True if ``df`` looks like a usable OHLCV price history."""
    if df is None or df.empty:
        return False
    missing_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in df.columns]
    if missing_cols:
        return False
    return len(df) >= _MIN_ROWS


def _drop_unusable_bars(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Drop rows with no usable ``Close`` price from a price-history frame.

    Stooq (and occasionally yfinance) can include a trailing bar for an
    in-progress/unsettled session that carries a ``Volume`` estimate but
    ``NaN`` for ``Open``/``High``/``Low``/``Close`` -- and, more rarely, an
    interior gap where a session's data is equally incomplete. Either shape
    poisons every rolling-window indicator downstream: a single NaN
    anywhere in a rolling window makes that whole window's result NaN (see
    ``sec_analyzer.technical.indicators.compute_indicators``, which reads
    ``close.iloc[-1]`` and rolling ``sma50``/``sma200`` windows). So no path
    in this module may return a frame with an unusable ``Close`` on any row.

    This must be called on every frame this module produces (cache load,
    Stooq, yfinance) *before* :func:`_validate_frame`, so a frame that only
    clears the minimum-row bar because of junk rows is correctly rejected
    as too short rather than silently passed through.

    Dropping a lone trailing row is an expected, routine occurrence during
    market hours (not an error), so it is logged at debug level; dropping
    more than one row is logged at info level since it is more likely to
    indicate an actual upstream data problem worth noticing.

    Args:
        df: A candidate price-history DataFrame, or ``None``.
        ticker: Ticker symbol, used only to make the log message more
            useful. Optional.

    Returns:
        ``df`` with every row whose ``Close`` is NaN removed (order and
        remaining columns unchanged). ``None``, an empty frame, or a frame
        with no ``Close`` column at all is returned unchanged -- there is
        nothing to drop, and the missing-column case is already rejected
        by :func:`_validate_frame` downstream. Never raises.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return df

    close = pd.to_numeric(df["Close"], errors="coerce")
    bad = close.isna()
    n_bad = int(bad.sum())
    if n_bad == 0:
        return df

    label = f" for {ticker}" if ticker else ""
    log = logger.debug if n_bad == 1 else logger.info
    log("Dropping %d row(s) with unusable (NaN) Close price%s.", n_bad, label)

    df = df.copy()
    df["Close"] = close
    return df.loc[~bad]


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

    df = _drop_unusable_bars(df, ticker)

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

    df = _drop_unusable_bars(df, ticker)

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

    Every row of the returned frame has a usable (non-NaN) ``Close``: see
    :func:`_drop_unusable_bars`, which is applied on the cache-load path
    and on both the Stooq and yfinance fetch paths, before each frame is
    checked by :func:`_validate_frame`. On the Stooq/yfinance paths the
    cleaned frame is what gets both cached to disk and returned -- there is
    no separate "dirty" version written to cache, since cleaning once and
    reusing the same frame for both is simpler than keeping the raw and
    cleaned frames separate. On the cache-load path the cleaning happens
    only in memory (the on-disk file is left as-is); a pre-existing cache
    file written before this cleaning existed will simply have its junk
    rows dropped again on every load until its next 24h refresh replaces it
    with an already-clean fetch.

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

    # A usable-but-stale cache is kept aside: if every upstream then fails,
    # a price two sessions old still beats aborting the whole analysis.
    stale_df = None
    if not no_cache and os.path.exists(path):
        try:
            cached = _drop_unusable_bars(_load_cache(path), ticker)
            if _validate_frame(cached):
                if _cache_covers_last_session(cached):
                    source = _read_cache_source(path)
                    logger.info(
                        "price cache hit for %s: %s (%d rows, source=%s)",
                        ticker, path, len(cached), source,
                    )
                    return cached, f"cache({source})"
                stale_df = cached
                logger.info(
                    "Price cache for %s is missing the last completed session (%s); re-fetching.",
                    ticker, last_completed_session().isoformat(),
                )
            else:
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
        _write_cache(path, df, source="stooq")
        logger.info("Fetched %d price rows for %s from stooq; cached to %s", len(df), ticker, path)
        return df, "stooq"

    logger.warning("Stooq failed (%s); using yfinance fallback", stooq_error)

    try:
        df = _fetch_yfinance(ticker, period=yfinance_period)
    except PriceDataError as exc:
        logger.error("Both Stooq and yfinance failed for %s: stooq=%s yfinance=%s", ticker, stooq_error, exc)
        if stale_df is not None and _stale_cache_is_usable(path):
            logger.warning(
                "Both price sources failed for %s; falling back to the stale cache "
                "(newest bar %s). Price-derived figures will be out of date.",
                ticker, stale_df.index.max().date().isoformat(),
            )
            return stale_df, f"stale-cache({_read_cache_source(path)})"
        raise PriceDataError(
            f"Could not obtain price history for {ticker!r} from Stooq or yfinance. "
            f"Stooq error: {stooq_error}. yfinance error: {exc}"
        ) from exc

    _write_cache(path, df, source="yfinance")
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


def slice_asof(df: pd.DataFrame, as_of) -> pd.DataFrame:
    """Return the rows of a price-history frame dated on/before ``as_of``.

    Args:
        df: A price-history DataFrame (Date index, ascending).
        as_of: Point-in-time cutoff (``datetime.date`` or ISO
            ``"YYYY-MM-DD"`` string). ``None`` returns ``df`` unchanged.

    Returns:
        A view/copy of ``df`` with all rows whose index is ``<= as_of``.
        Never mutates the input. May be empty (e.g. the ticker had not
        started trading yet), which callers handle gracefully.
    """
    if as_of is None:
        return df
    cutoff = pd.Timestamp(as_of.isoformat() if hasattr(as_of, "isoformat") else as_of)
    return df.loc[df.index <= cutoff]
