"""Compute technical-analysis indicators from a daily OHLCV price history.

All indicators are derived purely from the price DataFrame produced by
:func:`sec_analyzer.fetch.prices.get_price_history` (a ``Date``-indexed,
ascending-order DataFrame with at least ``Close``/``High``/``Low``
columns). Every computation guards against insufficient history and
returns ``None`` for a given key rather than raising, so callers never have
to special-case a young or short-history ticker.
"""

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

#: Wilder's RSI smoothing period.
_RSI_PERIOD = 14

#: Simple moving average windows.
_SMA_SHORT_WINDOW = 50
_SMA_LONG_WINDOW = 200

#: Trading days considered "52 weeks" for high/low/range calculations.
_WEEKS_52_TRADING_DAYS = 252

#: Window (trading days) for the annualized volatility calculation.
_VOLATILITY_WINDOW = 20

#: Trading days used to detect a golden/death cross "recently".
_CROSS_LOOKBACK_DAYS = 60

#: Trading days per year, used to annualize daily return volatility.
_TRADING_DAYS_PER_YEAR = 252


def _round_or_none(value, digits: int):
    """Round ``value`` to ``digits`` decimals, or return ``None`` if it's
    missing/NaN."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _rsi14(close: pd.Series) -> "float | None":
    """Compute Wilder's 14-period RSI on a Close series.

    Uses the standard Wilder smoothing approximation via an exponentially
    weighted moving average with ``alpha = 1/14`` (equivalent to Wilder's
    original recursive smoothing), rather than a simple rolling mean of
    gains/losses.

    Returns:
        The RSI at the last observation, rounded to 1 decimal, or ``None``
        if there isn't enough history (fewer than ``_RSI_PERIOD`` + 1 daily
        changes) to produce a value.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / _RSI_PERIOD, min_periods=_RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / _RSI_PERIOD, min_periods=_RSI_PERIOD).mean()

    # avg_loss == 0 (all recent moves were gains) legitimately yields
    # rs == inf, which correctly resolves to rsi == 100 below -- no special
    # casing needed. avg_gain == avg_loss == 0 (a perfectly flat series)
    # yields rs == NaN, which is handled by the pd.isna check further down.
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    value = rsi.iloc[-1]
    if pd.isna(value):
        return None
    return round(float(value), 1)


def _sma(close: pd.Series, window: int) -> pd.Series:
    """Return the simple rolling mean of ``close`` over ``window`` periods."""
    return close.rolling(window=window, min_periods=window).mean()


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute the full technical-indicator set from a price history.

    Args:
        df: A price-history DataFrame indexed by Date (ascending order),
            with at least a ``Close`` column (``High``/``Low`` are not
            currently used but may be present).

    Returns:
        A flat dict with the following keys (any of them may be ``None``
        when there isn't enough history to compute it):

        * ``price``: last Close, 2dp float.
        * ``as_of``: last date, ``"YYYY-MM-DD"``.
        * ``rsi14``: Wilder's 14-period RSI, 1dp.
        * ``sma50`` / ``sma200``: simple moving averages of Close.
        * ``dist_sma50_pct`` / ``dist_sma200_pct``: ``(price/sma - 1) * 100``,
          1dp.
        * ``high_52w`` / ``low_52w``: max/min Close over the last 252
          trading days (or over all available rows if fewer than 252 are
          present -- there simply isn't a full 52-week window yet).
        * ``range_position_pct``: ``(price - low) / (high - low) * 100``,
          1dp; ``None`` if ``high_52w == low_52w`` (would divide by zero).
        * ``volatility_20d``: standard deviation of daily percentage
          returns over the last 20 trading days, annualized by multiplying
          by ``sqrt(252)`` and expressed as a decimal fraction (e.g.
          ``0.32`` means roughly 32% annualized volatility), 4dp.
        * ``golden_cross`` / ``death_cross``: booleans, ``True`` if SMA50
          crossed above/below SMA200 within the last 60 trading days
          (comparing the sign of ``sma50 - sma200`` now vs. 60 trading days
          ago). ``None`` if there isn't enough history (needs both a valid
          SMA200 now and 60 trading days ago).
        * ``sma50_above_sma200``: current state (``sma50 > sma200``) as a
          bool, or ``None`` if either SMA is unavailable.
    """
    close = df["Close"]

    price = _round_or_none(close.iloc[-1], 2)
    as_of = df.index[-1]
    as_of_str = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)

    rsi14 = _rsi14(close)

    sma50_series = _sma(close, _SMA_SHORT_WINDOW)
    sma200_series = _sma(close, _SMA_LONG_WINDOW)
    sma50 = _round_or_none(sma50_series.iloc[-1], 2)
    sma200 = _round_or_none(sma200_series.iloc[-1], 2)

    dist_sma50_pct = None
    if sma50 is not None and sma50 != 0 and price is not None:
        dist_sma50_pct = round((price / sma50 - 1) * 100, 1)

    dist_sma200_pct = None
    if sma200 is not None and sma200 != 0 and price is not None:
        dist_sma200_pct = round((price / sma200 - 1) * 100, 1)

    window_52w = close.tail(min(_WEEKS_52_TRADING_DAYS, len(close)))
    high_52w = _round_or_none(window_52w.max(), 2)
    low_52w = _round_or_none(window_52w.min(), 2)

    range_position_pct = None
    if high_52w is not None and low_52w is not None and high_52w != low_52w and price is not None:
        range_position_pct = round((price - low_52w) / (high_52w - low_52w) * 100, 1)

    returns = close.pct_change().dropna()
    volatility_20d = None
    if len(returns) >= _VOLATILITY_WINDOW:
        last_returns = returns.tail(_VOLATILITY_WINDOW)
        stdev = last_returns.std()
        if not pd.isna(stdev):
            volatility_20d = round(float(stdev) * math.sqrt(_TRADING_DAYS_PER_YEAR), 4)

    sma50_above_sma200 = None
    if sma50 is not None and sma200 is not None:
        sma50_above_sma200 = bool(sma50 > sma200)

    golden_cross = None
    death_cross = None
    if len(close) > _SMA_LONG_WINDOW + _CROSS_LOOKBACK_DAYS:
        now_sma50 = sma50_series.iloc[-1]
        now_sma200 = sma200_series.iloc[-1]
        past_sma50 = sma50_series.iloc[-(_CROSS_LOOKBACK_DAYS + 1)]
        past_sma200 = sma200_series.iloc[-(_CROSS_LOOKBACK_DAYS + 1)]
        if not (pd.isna(now_sma50) or pd.isna(now_sma200) or pd.isna(past_sma50) or pd.isna(past_sma200)):
            now_diff = now_sma50 - now_sma200
            past_diff = past_sma50 - past_sma200
            golden_cross = bool(past_diff <= 0 and now_diff > 0)
            death_cross = bool(past_diff >= 0 and now_diff < 0)

    indicators = {
        "price": price,
        "as_of": as_of_str,
        "rsi14": rsi14,
        "sma50": sma50,
        "sma200": sma200,
        "dist_sma50_pct": dist_sma50_pct,
        "dist_sma200_pct": dist_sma200_pct,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "range_position_pct": range_position_pct,
        "volatility_20d": volatility_20d,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "sma50_above_sma200": sma50_above_sma200,
    }

    logger.debug("Computed technical indicators: %s", indicators)
    return indicators
