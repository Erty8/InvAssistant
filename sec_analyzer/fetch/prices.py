"""Download and cache daily OHLCV price history for a ticker.

Sources are tried in the order given by :data:`_FETCH_CHAIN`: the ``yfinance``
package, then Yahoo's chart endpoint over plain HTTP, then Nasdaq's public
endpoint -- after which a recent-enough on-disk cache is the last resort (see
:func:`_stale_cache_is_usable`) before :class:`PriceDataError` is raised with
a message intended to be shown directly to a user.

Stooq's free CSV endpoint was the primary source until 2026-08-03, when Stooq
put every endpoint (``stooq.com`` and ``stooq.pl`` alike, the CSV endpoint and
ordinary quote pages) behind a JavaScript browser-verification challenge --
HTTP 200 with a ~796-byte HTML page instead of CSV, which no plain
``requests`` call can get past. It was removed rather than kept as a
guaranteed-failing request plus a warning line per ticker per run.

**Not every source shares a price basis**, and that is the sharpest edge in
this module. The two Yahoo paths return split- AND dividend-adjusted closes;
Nasdaq returns split-adjusted-only closes, which drift further from a
total-return series the further back you look. :func:`is_total_return_basis`
is how a caller must decide whether a frame may reach the valuation layer.
See :data:`_TOTAL_RETURN_SOURCES` for the measured numbers.

Any source can hand back a trailing (or, rarely, interior) bar for an
in-progress/unsettled session that carries a ``Volume`` estimate but ``NaN``
for ``Open``/``High``/``Low``/``Close``. Every frame this module returns --
from cache or from any upstream -- has such rows dropped before being handed
back (see :func:`_drop_unusable_bars`), so a caller can always rely on
``Close`` being usable on every row of the returned frame.

A frame returned to a caller may still end in a bar for the session currently
in progress -- during market hours that IS the current price, and it is dated,
so it is shown -- but only to a caller that asked for it via
``prefer_live`` (see :func:`get_price_history`). What must never happen is
such a bar being *persisted*: on any later day its date makes the cache look
fresh while its ``Close`` is a mid-day snapshot, so the real close is never
fetched. Two functions hold that line: :func:`drop_unsettled_bars` on every
write, and :func:`_drop_partial_trailing_bar` on every read, for caches
written before the fix existed.
"""

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Columns a usable price-history frame is expected to carry.
_EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

#: Yahoo's chart endpoint, used by :func:`_fetch_yahoo_chart`. ``period1``/
#: ``period2`` epoch bounds are MANDATORY for daily data: ``range=max`` with
#: ``interval=1d`` silently returns ~quarterly bars (measured 2026-08-03:
#: 163 bars with a 92-day median gap, versus 10,176 true daily bars for the
#: same ticker via period1/period2).
_YAHOO_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1=0&period2={period2}&interval=1d"
)

#: Nasdaq's public quote/historical endpoint, used by :func:`_fetch_nasdaq`.
#: It only serves a window ending today -- an interior range (e.g. a two-week
#: window in 2024) comes back with ``totalRecords: 0`` -- and caps history at
#: roughly 10 years regardless of ``fromdate``.
_NASDAQ_URL = (
    "https://api.nasdaq.com/api/quote/{symbol}/historical"
    "?assetclass=stocks&fromdate={fromdate}&todate={todate}&limit=99999"
)

#: Years of history to ask Nasdaq for. Its own ceiling is near 10; asking for
#: more is harmless but does not yield more.
_NASDAQ_HISTORY_YEARS = 10

#: A normal browser-style User-Agent for the two plain-HTTP fallbacks. Neither
#: host has a fair-access identity requirement like SEC EDGAR does (see
#: :class:`sec_analyzer.http_client.SecHttpClient`, deliberately not used
#: here); this just avoids looking like a bare script to generic bot filters.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: Sources whose closes are adjusted for BOTH splits and dividends, i.e. a
#: total-return series -- the basis every price-derived valuation figure in
#: this project assumes.
#:
#: ``nasdaq`` is deliberately absent. Its closes are split-adjusted but NOT
#: dividend-adjusted, and the gap compounds backwards: measured on ORCL
#: 2026-08-03, Nasdaq vs yfinance was 0% on the newest bar, -5% four years
#: back and -14% ten years back. Feeding that series to
#: ``valuation.multiples``, which builds historical-multiple percentiles from
#: 10-15 years of year-end prices, would quietly bias every "cheap or
#: expensive versus its own history" verdict for any dividend payer. See
#: :func:`is_total_return_basis`.
_TOTAL_RETURN_SOURCES = frozenset({"yfinance", "yahoo-chart"})

#: Sources good enough to serve technicals but not to keep: a cache written
#: from one of these is never taken as a plain hit, so the better sources get
#: retried on the next run instead of the degraded data sticking forever.
_DEGRADED_SOURCES = frozenset({"nasdaq"})

#: Minimum number of rows required for a fetched price history to be
#: considered usable, from either source.
_MIN_ROWS = 30

#: Hour (UTC) after which the current weekday's US equity session is treated
#: as closed AND published. The cash close is 16:00 ET = 20:00 UTC in summer
#: (EDT) / 21:00 UTC in winter (EST); 22:00 covers both without needing a
#: timezone database, and leaves the provider an hour to publish the bar.
#: Erring late only costs one extra re-fetch attempt, never a wrong price.
_SESSION_PUBLISHED_HOUR_UTC = 22

#: Hour (UTC) from which a US equity session may be open. The cash open is
#: 09:30 ET = 13:30 UTC in summer (EDT) / 14:30 UTC in winter (EST); 13
#: covers both. Erring early only means a ``prefer_live`` re-fetch that finds
#: no new bar yet -- the same cheap, safe direction as
#: :data:`_SESSION_PUBLISHED_HOUR_UTC` erring late.
_SESSION_OPEN_HOUR_UTC = 13

#: Hard ceiling on how old a cache may be before it is refused even as a
#: last-resort fallback when every upstream source is failing.
_CACHE_ABSOLUTE_MAX_AGE_DAYS = 30

#: yfinance ``period`` string used as the default lookback. ``"max"`` returns
#: the *entire* available daily history for the ticker. Valuation's
#: historical-multiples percentiles (see ``sec_analyzer.valuation.multiples``)
#: want 10-15 years of year-end prices, so the default is "as much as is
#: available" rather than a short fixed window.
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
    where the data actually came from, rather than assuming a default and
    surfacing a possibly-false source in the HTML report's provenance line.

    The frame is passed through :func:`drop_unsettled_bars` first, so no code
    path can persist a bar for a session that has not closed and published
    yet. This is the single choke point for that invariant -- the in-memory
    frame the caller returns keeps its live intraday bar (a run during market
    hours should show the current price, dated today), but the on-disk cache
    only ever holds settled sessions.
    """
    drop_unsettled_bars(df).to_csv(path, index_label="Date")
    if source:
        try:
            with open(_source_path(path), "w", encoding="utf-8") as handle:
                handle.write(source)
        except OSError:  # provenance is nice-to-have, never fatal
            logger.debug("Could not write price-cache source marker for %s", path, exc_info=True)


def upstream_of(source: Optional[str]) -> str:
    """The bare upstream name inside a ``source`` string.

    ``get_price_history`` returns wrapped forms -- ``"cache(nasdaq)"``,
    ``"stale-cache(yfinance)"`` -- so callers that need to reason about the
    upstream itself would otherwise each write the same unwrapping. Returns
    ``""`` for ``None``.
    """
    if not source:
        return ""
    start, end = source.find("("), source.rfind(")")
    if 0 <= start < end:
        return source[start + 1:end].strip()
    return source.strip()


def is_total_return_basis(source: Optional[str]) -> bool:
    """Whether ``source``'s closes are both split- AND dividend-adjusted.

    Every price-derived valuation figure in this project assumes a
    total-return series, so a caller must not hand a frame to the valuation
    layer when this is False -- see :data:`_TOTAL_RETURN_SOURCES` for the
    measured size of the distortion. Accepts wrapped source strings.

    An unrecognised or unrecorded source (including the ``"unknown"`` that
    pre-marker caches report) returns False: guessing "probably fine" here
    fails silently and in the expensive direction.
    """
    return upstream_of(source) in _TOTAL_RETURN_SOURCES


def _read_cache_source(path: str) -> str:
    """Which upstream produced a cached CSV; ``"unknown"`` when unrecorded.

    Caches written before the sidecar existed carry no marker. They used to
    default to ``"stooq"``, the primary source at the time; now that the Stooq
    path is gone that default would name a source this module can no longer
    even reach, so such files report ``"unknown"`` instead. Every write records
    a real marker, so a markerless file resolves itself on its next re-fetch.
    """
    try:
        with open(_source_path(path), "r", encoding="utf-8") as handle:
            recorded = handle.read().strip()
        return recorded or "unknown"
    except OSError:
        return "unknown"


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


def session_in_progress(now_utc: Optional[datetime] = None) -> bool:
    """Whether a US equity session may be open right now.

    Calendar-light in the same way as :func:`last_completed_session`: weekends
    are excluded, market holidays are not known here. A holiday reads as "in
    progress", which costs a ``prefer_live`` re-fetch that finds no new bar --
    never a wrong price.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:  # Saturday / Sunday
        return False
    return _SESSION_OPEN_HOUR_UTC <= now_utc.hour < _SESSION_PUBLISHED_HOUR_UTC


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


def drop_unsettled_bars(
    df: pd.DataFrame, ticker: str = "", now_utc: Optional[datetime] = None
) -> pd.DataFrame:
    """Drop bars for sessions that have not closed and published yet.

    The companion to :func:`_cache_covers_last_session`, and the other half of
    the same defect. That function asks only whether the newest cached bar's
    DATE reaches the last completed session -- not whether that bar was
    captured *after* the close. So a run during market hours cached an
    in-progress bar stamped with today's date, and from the next day onward
    that partial bar made the cache look permanently fresh: the real close was
    never fetched.

    Observed on ORCL 2026-08-03: the cache had been written Friday 2026-07-31
    at 14:48 UTC, mid-session, so it held a partial 07-31 bar closing at
    ``$127.89`` on 13.2M shares. The true 07-31 close was ``$129.87`` on 34.4M
    shares, but because the partial bar was dated 07-31 -- the last completed
    session as of Monday -- every Monday run served ``$127.89`` as that close.
    An in-progress bar has a perfectly valid ``Close``, so
    :func:`_drop_unusable_bars` (NaN-only) cannot catch it.

    Dropping such a bar before it is written keeps the newest cached bar
    always settled, which in turn makes the freshness check mean what it says.

    Args:
        df: A price-history DataFrame indexed by Date, or ``None``.
        ticker: Ticker symbol, used only to make the log message more useful.
        now_utc: Injectable current time, for tests.

    Returns:
        ``df`` without any bar dated after
        :func:`last_completed_session`. ``None`` and empty frames pass through
        unchanged. If the filter would empty the frame entirely -- only
        possible for a frame consisting of nothing but unsettled bars, which
        :func:`_validate_frame` would reject anyway -- ``df`` is returned
        unchanged rather than writing an empty cache. Never raises.
    """
    if df is None or df.empty:
        return df

    try:
        cutoff = last_completed_session(now_utc)
        settled = df.loc[[d.date() <= cutoff for d in df.index]]
    except (AttributeError, TypeError, ValueError):
        # A non-datetime index is not something to fail a fetch over; the
        # frame is handed back as-is and rejected downstream if unusable.
        logger.debug("Could not filter unsettled bars; index is not dates.", exc_info=True)
        return df

    n_dropped = len(df) - len(settled)
    if n_dropped == 0 or settled.empty:
        return df

    label = f" for {ticker}" if ticker else ""
    logger.debug(
        "Not caching %d unsettled bar(s)%s (session on/after %s has not published).",
        n_dropped, label, cutoff.isoformat(),
    )
    return settled


def _drop_partial_trailing_bar(
    path: str, df: pd.DataFrame, ticker: str = ""
) -> pd.DataFrame:
    """Drop a trailing bar that a cache file's mtime proves was partial.

    :func:`drop_unsettled_bars` stops *new* partial bars from being written,
    but it cannot heal the caches already on disk -- and those never heal
    themselves: a partial bar carries the date of a real session, so
    :func:`_cache_covers_last_session` keeps reporting such a file as fresh
    and the true close is never fetched.

    The file's mtime settles it. A cache whose newest bar is dated ``D`` but
    which was written before ``D``'s close had published (see
    :data:`_SESSION_PUBLISHED_HOUR_UTC`) can only have captured ``D``
    mid-session, so that bar is dropped and the frame falls back to its last
    settled session -- which then reads as stale and triggers a re-fetch.
    Only the trailing bar can be affected: at write time exactly one session
    was in progress.

    Returns ``df`` unchanged when the mtime is unreadable, when the trailing
    bar was written after its session published, or when dropping it would
    empty the frame. Never raises.
    """
    if df is None or df.empty:
        return df

    try:
        newest = df.index.max().date()
        written = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    except (AttributeError, OSError, ValueError):
        return df

    published = datetime(
        newest.year, newest.month, newest.day,
        _SESSION_PUBLISHED_HOUR_UTC, tzinfo=timezone.utc,
    )
    if written >= published:
        return df

    settled = df.loc[[d.date() < newest for d in df.index]]
    if settled.empty:
        return df

    label = f" for {ticker}" if ticker else ""
    logger.info(
        "Cached %s bar%s was captured mid-session (file written %s, before that "
        "session published); discarding it and re-fetching.",
        newest.isoformat(), label, written.isoformat(timespec="minutes"),
    )
    return settled


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

    yfinance can include a trailing bar for an
    in-progress/unsettled session that carries a ``Volume`` estimate but
    ``NaN`` for ``Open``/``High``/``Low``/``Close`` -- and, more rarely, an
    interior gap where a session's data is equally incomplete. Either shape
    poisons every rolling-window indicator downstream: a single NaN
    anywhere in a rolling window makes that whole window's result NaN (see
    ``sec_analyzer.technical.indicators.compute_indicators``, which reads
    ``close.iloc[-1]`` and rolling ``sma50``/``sma200`` windows). So no path
    in this module may return a frame with an unusable ``Close`` on any row.

    This must be called on every frame this module produces (cache load,
    yfinance) *before* :func:`_validate_frame`, so a frame that only
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


def _fetch_yfinance(ticker: str, period: str = _YFINANCE_DEFAULT_PERIOD) -> pd.DataFrame:
    """Fetch daily price history from the ``yfinance`` package.

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
            "yfinance is not installed; it is the only price source available."
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


def _fetch_yahoo_chart(ticker: str) -> pd.DataFrame:
    """Fetch daily history from Yahoo's chart endpoint over plain HTTP.

    First fallback, and basis-identical to the ``yfinance`` package: verified
    on 2026-08-03 that this endpoint's ``adjclose`` equals what
    ``yf.download(auto_adjust=True)`` returns for ``Close`` to the cent, that
    scaling OHLC by ``adjclose/close`` reproduces yfinance's OHLC, that
    ``Volume`` is untouched by the adjustment, and that both return the same
    10,176 bars for ORCL.

    That equality is the point of this fallback: it covers the failure mode
    that actually happens -- the yfinance package breaking when Yahoo changes
    something behind it -- without introducing a second price basis. It does
    NOT cover Yahoo itself being down or blocking; :func:`_fetch_nasdaq` is
    the independent-provider tier.

    Raises:
        PriceDataError: On any request, shape, or sufficiency failure.
    """
    symbol = ticker.strip().upper()
    # A day past "now" so the current session's bar is never cut off by clock
    # skew between here and Yahoo.
    url = _YAHOO_CHART_URL.format(symbol=symbol, period2=int(time.time()) + 86400)
    logger.info("Fetching price history for %s from the Yahoo chart endpoint", symbol)

    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise PriceDataError(f"Yahoo chart request failed for {ticker}: {exc}") from exc
    except ValueError as exc:  # not JSON at all
        raise PriceDataError(f"Yahoo chart returned a non-JSON body for {ticker}: {exc}") from exc

    try:
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            error = (payload.get("chart") or {}).get("error")
            raise PriceDataError(
                f"Yahoo chart returned no result for {ticker} (error: {error!r})."
            )
        block = result[0]
        stamps = block.get("timestamp") or []
        quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
        adj_block = ((block.get("indicators") or {}).get("adjclose") or [{}])[0]
        adjclose = adj_block.get("adjclose") or []
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise PriceDataError(
            f"Yahoo chart payload for {ticker} had an unexpected shape: {exc}"
        ) from exc

    if not stamps or not adjclose:
        raise PriceDataError(f"Yahoo chart returned no daily bars for {ticker}.")

    df = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
            "_Adj": adjclose,
        },
        index=pd.to_datetime(pd.Series(stamps), unit="s", utc=True).dt.tz_localize(None).dt.normalize(),
    )
    df.index.name = "Date"

    # Reproduce yfinance's auto_adjust: Close becomes adjclose and OHLC are
    # scaled by the same ratio, so the intraday relationships survive.
    # Volume is deliberately left alone -- that is what yfinance does too.
    close = pd.to_numeric(df["Close"], errors="coerce")
    adj = pd.to_numeric(df["_Adj"], errors="coerce")
    ratio = (adj / close).where(close != 0)
    for column in ("Open", "High", "Low"):
        df[column] = pd.to_numeric(df[column], errors="coerce") * ratio
    df["Close"] = adj
    df = df.drop(columns=["_Adj"])

    df = _drop_unusable_bars(df, ticker)

    if not _validate_frame(df):
        raise PriceDataError(
            f"Yahoo chart returned too little data for {ticker} "
            f"({len(df)} row(s); need at least {_MIN_ROWS})."
        )

    return df.sort_index()


def _parse_nasdaq_number(raw) -> float:
    """Parse Nasdaq's display-formatted numbers (``"$1,234.56"``, ``"N/A"``).

    Returns NaN for anything unparseable, so the row survives to be dropped
    by :func:`_drop_unusable_bars` rather than aborting the whole fetch.
    """
    if raw is None:
        return float("nan")
    text = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _fetch_nasdaq(ticker: str) -> pd.DataFrame:
    """Fetch daily history from Nasdaq's public endpoint. Last-resort tier.

    The only genuinely provider-independent source found that needs no API key
    and is not behind a bot challenge. Two properties make it a last resort
    rather than a peer of the Yahoo paths:

    * **Split-adjusted only, not dividend-adjusted.** See
      :data:`_TOTAL_RETURN_SOURCES`. Callers must gate the valuation layer on
      :func:`is_total_return_basis`; the technical layer is unaffected, since
      RSI/SMA/52-week levels are conventionally read off a
      split-adjusted-only series anyway.
    * **~10 years of history**, against the 10-15 years
      ``valuation.multiples`` wants -- a second reason the valuation layer
      must not consume it.

    Raises:
        PriceDataError: On any request, shape, or sufficiency failure.
    """
    symbol = ticker.strip().upper()
    today = datetime.now(timezone.utc).date()
    url = _NASDAQ_URL.format(
        symbol=symbol,
        fromdate=(today - timedelta(days=365 * _NASDAQ_HISTORY_YEARS + 3)).isoformat(),
        todate=today.isoformat(),
    )
    logger.info("Fetching price history for %s from the Nasdaq endpoint", symbol)

    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise PriceDataError(f"Nasdaq request failed for {ticker}: {exc}") from exc
    except ValueError as exc:
        raise PriceDataError(f"Nasdaq returned a non-JSON body for {ticker}: {exc}") from exc

    rows = (((payload.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    if not rows:
        # Nasdaq answers HTTP 200 with rows=None / totalRecords=0 for an
        # unknown symbol or a window it will not serve.
        raise PriceDataError(
            f"Nasdaq returned no rows for {ticker} "
            f"(status {(payload.get('status') or {}).get('rCode')!r})."
        )

    records = []
    for row in rows:
        try:
            stamp = pd.to_datetime(row.get("date"), format="%m/%d/%Y")
        except (TypeError, ValueError):
            continue  # a row without a parseable date carries no usable bar
        records.append(
            {
                "Date": stamp,
                "Open": _parse_nasdaq_number(row.get("open")),
                "High": _parse_nasdaq_number(row.get("high")),
                "Low": _parse_nasdaq_number(row.get("low")),
                "Close": _parse_nasdaq_number(row.get("close")),
                "Volume": _parse_nasdaq_number(row.get("volume")),
            }
        )

    if not records:
        raise PriceDataError(f"No Nasdaq row for {ticker} carried a parseable date.")

    df = pd.DataFrame(records).set_index("Date").sort_index()
    df = _drop_unusable_bars(df, ticker)

    if not _validate_frame(df):
        raise PriceDataError(
            f"Nasdaq returned too little data for {ticker} "
            f"({len(df)} row(s); need at least {_MIN_ROWS})."
        )

    return df


#: The fetch chain, in order. Each entry is ``(source_name, fetcher)``.
#: Order encodes preference: the package first, then the same data over plain
#: HTTP (identical basis), then the independent provider (different basis, so
#: :func:`is_total_return_basis` gates what may consume it).
_FETCH_CHAIN = (
    ("yfinance", _fetch_yfinance),
    ("yahoo-chart", _fetch_yahoo_chart),
    ("nasdaq", _fetch_nasdaq),
)


def _fetch_with_fallbacks(
    ticker: str, yfinance_period: str
) -> "tuple[pd.DataFrame, str, list[str]]":
    """Walk :data:`_FETCH_CHAIN` until one source yields a usable frame.

    Returns ``(df, source_name, errors)``; ``errors`` holds one message per
    source that failed, so the caller can report why the chain degraded (or
    raise with all of them when nothing worked).
    """
    errors = []
    for source, fetcher in _FETCH_CHAIN:
        try:
            if source == "yfinance":
                df = fetcher(ticker, period=yfinance_period)
            else:
                df = fetcher(ticker)
        except PriceDataError as exc:
            errors.append(f"{source}: {exc}")
            logger.warning("Price source %s failed for %s: %s", source, ticker, exc)
            continue
        if errors:
            logger.warning(
                "Price data for %s came from the fallback source %s after %d failure(s).",
                ticker, source, len(errors),
            )
        return df, source, errors
    return None, "", errors


def get_price_history(
    ticker: str,
    no_cache: bool = False,
    yfinance_period: str = _YFINANCE_DEFAULT_PERIOD,
    prefer_live: bool = False,
) -> "tuple[pd.DataFrame, str]":
    """Fetch (or load from cache) daily OHLCV price history for ``ticker``.

    Tries the on-disk cache first (unless ``no_cache``), then yfinance.

    Every row of the returned frame has a usable (non-NaN) ``Close``: see
    :func:`_drop_unusable_bars`, which is applied on the cache-load path and
    on the yfinance fetch path, before each frame is checked by
    :func:`_validate_frame`. On the fetch path the cleaned frame is what gets
    both cached to disk and returned -- there is no separate "dirty" version
    written to cache, since cleaning once and reusing the same frame for both
    is simpler than keeping the raw and cleaned frames separate. On the
    cache-load path the cleaning happens only in memory (the on-disk file is
    left as-is); a pre-existing cache file written before this cleaning
    existed will simply have its junk rows dropped again on every load until
    its next refresh replaces it with an already-clean fetch.

    Args:
        ticker: Stock ticker symbol, e.g. ``"AAPL"``.
        no_cache: When True, bypass any existing cache and re-fetch,
            overwriting the cache file on success.
        yfinance_period: yfinance ``period`` string. Defaults to
            :data:`_YFINANCE_DEFAULT_PERIOD` ("max" -- full available
            history). Valuation's historical-multiples percentiles want 10-15
            years of year-end prices; pass e.g. ``"15y"`` to cap the fetch.
        prefer_live: When True and a session is currently in progress (see
            :func:`session_in_progress`), a cache that only reaches the last
            *settled* session is re-fetched so the frame ends in today's live
            bar. Off by default, and that default matters: a cache covering
            the last completed session is "fresh" by design, so without this
            flag a caller running at midday gets the previous close -- which
            is the right answer for anything ranking on daily bars (the
            screener, backtests) and the wrong one for a report that prints a
            current price. Never affects what is written to disk: the cache
            stays settled-only either way.

    Returns:
        A ``(df, source)`` tuple. ``df`` is indexed by ``Date`` (ascending)
        with ``Open``/``High``/``Low``/``Close``/``Volume`` columns.
        ``source`` identifies where the data came from: ``"yfinance"``,
        ``"cache(yfinance)"``, or ``"stale-cache(...)"``. Caches written
        before source markers existed report ``cache(unknown)``.

    Raises:
        PriceDataError: If neither the cache nor yfinance yield a usable
            price history.
    """
    Config.ensure_dirs()
    ticker = ticker.strip().upper()
    path = _cache_path(ticker)

    # A usable-but-stale cache is kept aside: if the upstream then fails,
    # a price two sessions old still beats aborting the whole analysis.
    stale_df = None
    if not no_cache and os.path.exists(path):
        try:
            loaded = _drop_unusable_bars(_load_cache(path), ticker)
            cached = _drop_partial_trailing_bar(path, loaded, ticker)
            if _validate_frame(cached):
                if _cache_covers_last_session(cached):
                    source = _read_cache_source(path)
                    if len(cached) < len(loaded):
                        # A healed file must be rewritten, or the partial bar
                        # sits there being re-discarded on every single read.
                        # Safe to restamp the mtime: every remaining bar is
                        # settled, so the evidence _drop_partial_trailing_bar
                        # relies on still reads correctly afterwards.
                        _write_cache(path, cached, source=source)
                    if upstream_of(source) in _DEGRADED_SOURCES:
                        # Written by a last-resort source. Its data is fine for
                        # technicals but carries a different price basis and a
                        # shorter history, so it must not become permanent just
                        # because it covers the last session -- retry the better
                        # sources, keeping this as the fallback if they fail.
                        stale_df = cached
                        logger.info(
                            "Price cache for %s came from the degraded source %s; "
                            "retrying the preferred sources.",
                            ticker, source,
                        )
                    elif prefer_live and session_in_progress():
                        # The cache can never hold the live bar (it is
                        # settled-only on purpose), so serving it here would
                        # print this morning's report with yesterday's close.
                        stale_df = cached
                        logger.info(
                            "Price cache for %s is settled through %s but a session is "
                            "in progress and a live price was requested; re-fetching.",
                            ticker, cached.index.max().date().isoformat(),
                        )
                    else:
                        logger.info(
                            "price cache hit for %s: %s (%d rows, source=%s)",
                            ticker, path, len(cached), source,
                        )
                        return cached, f"cache({source})"
                else:
                    stale_df = cached
                    logger.info(
                        "Price cache for %s is missing the last completed session (%s); re-fetching.",
                        ticker, last_completed_session().isoformat(),
                    )
            else:
                logger.warning("Cached price file for %s failed validation; re-fetching.", ticker)
        except Exception:  # noqa: BLE001 - a corrupt cache file must not be fatal
            logger.warning("Failed to load price cache for %s at %s; re-fetching.", ticker, path, exc_info=True)

    df, source, errors = _fetch_with_fallbacks(ticker, yfinance_period)

    if df is None:
        logger.error("Every price source failed for %s: %s", ticker, "; ".join(errors))
        if stale_df is not None and _stale_cache_is_usable(path):
            logger.warning(
                "Every price source failed for %s; falling back to the cache "
                "(newest bar %s). Price-derived figures may be out of date.",
                ticker, stale_df.index.max().date().isoformat(),
            )
            return stale_df, f"stale-cache({_read_cache_source(path)})"
        raise PriceDataError(
            f"Could not obtain price history for {ticker!r} from any price source. "
            + " | ".join(errors)
        )

    _write_cache(path, df, source=source)
    logger.info("Fetched %d price rows for %s from %s; cached to %s", len(df), ticker, source, path)
    return df, source


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
