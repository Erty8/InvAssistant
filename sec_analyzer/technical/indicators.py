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

#: Half-width of the price fractal used to locate swing highs/lows for RSI
#: divergence detection.
_RSI_DIVERGENCE_PIVOT = 5

#: A divergence is only reported when its most recent swing is within this
#: many bars of the end -- a stale divergence isn't actionable.
_RSI_DIVERGENCE_RECENCY = 30

#: RSI oversold/overbought lines for the swing "reversal reclaim" signal, and
#: the recent-bar window over which a reclaim still counts as fresh.
_RSI_OVERSOLD = 30.0
_RSI_OVERBOUGHT = 70.0
_RSI_RECLAIM_LOOKBACK = 10

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

#: Momentum-return lookback windows (trading days), roughly 1/3/6 months.
_RETURN_WINDOWS = {"return_1m_pct": 21, "return_3m_pct": 63, "return_6m_pct": 126}

#: ATR (Average True Range) smoothing period (Wilder), for volatility-scaled
#: swing stop distance.
_ATR_PERIOD = 14

#: Bollinger-band period + std multiple, and the lookback over which the
#: current band width's percentile is ranked. A "squeeze" = band width in the
#: bottom _BB_SQUEEZE_PCT percentile of its recent range (volatility
#: compression that often precedes an expansion move) -- direction-agnostic.
_BB_PERIOD = 20
_BB_STD = 2.0
_BB_SQUEEZE_LOOKBACK = 126  # ~6 months
_BB_SQUEEZE_MIN_SAMPLE = 30
_BB_SQUEEZE_PCT = 20.0

#: Volume-climax / capitulation scan: over the last _CLIMAX_LOOKBACK bars, a
#: bar whose volume is at least _CLIMAX_VOLUME_MULT x its trailing average AND
#: whose high-low range is at least _CLIMAX_RANGE_PCT of price is a climax bar
#: (seller/buyer exhaustion -- a reversal tell when it lands in a trend).
_CLIMAX_LOOKBACK = 10
_CLIMAX_VOLUME_MULT = 2.0
_CLIMAX_RANGE_PCT = 0.04

#: Half-width (bars on each side) of the swing-pivot detector: a bar is a
#: swing high/low when its High/Low is the max/min over the +/-k-bar window.
#: k=5 -> an 11-bar fractal, which surfaces meaningful multi-week swings on
#: daily data without flagging every minor wiggle.
_PIVOT_WINDOW = 5

#: Relative tolerance for clustering nearby swing levels into a single
#: support/resistance zone (1.5% of the level). Also the minimum gap from the
#: current price for a zone to count as "above"/"below" rather than "here".
_SR_CLUSTER_TOL = 0.015

#: Fibonacci retracement ratios drawn across the dominant swing (0.5 is not a
#: true Fibonacci ratio but is conventionally included).
_FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)

#: Zone-strength scoring weights. A 52-week extreme outranks everything else
#: when it's still near price (it is more important than a Fibonacci level),
#: touches accumulate, and a Fibonacci confluence adds a smaller boost -- so
#: ranking prefers 52w > heavily-tested swing > fib, and combining sources
#: ("farklı değerler") strengthens a zone. The 52w bonus is not unconditional,
#: though -- see :data:`_SR_52W_FAR_THRESHOLD_PCT`.
_SR_SCORE_52W = 100
_SR_SCORE_PER_TOUCH = 10
_SR_SCORE_FIB = 5

#: A major moving average (SMA50/SMA200) is dynamic support/resistance and a
#: heavily-watched level in its own right, so a zone that coincides with one
#: gets a solid score bonus -- roughly on par with a well-tested (~5-touch)
#: swing shelf: stronger than a lone Fibonacci confluence, weaker than a live
#: 52-week extreme. This is what surfaces "price is sitting on the 200-day"
#: as support on a stock that ran vertically past every horizontal shelf.
_SR_SCORE_MA = 50

#: Beyond this distance from price (as a percent), a 52-week extreme's score
#: bonus starts decaying -- a 52w level 300%+ away shouldn't auto-outrank a
#: closer, well-tested zone just because it's the all-time extreme.
_SR_52W_FAR_THRESHOLD_PCT = 60.0

#: Hard cap on how far a support/resistance level may sit from the current
#: price and still be reported. A level beyond this distance (e.g. a 52-week
#: low ~73% below a stock that has run up hard) is not an actionable trigger,
#: so it is dropped from selection entirely -- the "far" slot on that side is
#: then filled by the nearest strong in-range level (a tested shelf / Fib /
#: MA) instead of the extreme, or left empty if none qualifies.
_SR_MAX_DIST_PCT = 60.0

#: Corroboration window: when finalizing a chosen level, all evidence within
#: this fraction of its price is merged into it (summed touches, unioned fib
#: ratios, 52w flags) and defines its price *range*.
_SR_CORROBORATE_PCT = 0.03

#: Minimum half-width of a reported level's range (as a fraction of price),
#: so a single-price level still reads as a band rather than a point.
_SR_BAND_MIN_PCT = 0.005

#: Price-chart series exposed for the report. The report's chart supports a
#: client-side range switch (3m / 1y / 5y), so the payload must carry enough
#: history to cover the widest range while staying small: the most recent
#: ``_PRICE_SERIES_DAILY`` trading days are kept at daily resolution (for the
#: 3m/1y views), and everything older -- up to ``_PRICE_SERIES_MAX_ROWS`` (~5
#: trading years) -- is down-sampled to roughly weekly. The chart slices this
#: single ascending series by *date*, so the mixed resolution is transparent
#: to the 3m/1y views (which fall entirely inside the daily tail).
_PRICE_SERIES_DAILY = 320
_PRICE_SERIES_MAX_ROWS = 1300
_PRICE_SERIES_WEEKLY_STEP = 5

#: Only the most recent ~2 years of pivots feed support/resistance, so a
#: decade-old level doesn't outrank a fresh, actively-tested one.
_SR_LOOKBACK_DAYS = 504

#: MACD EMA spans (fast/slow) and signal-line span -- the classic 12/26/9.
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9

#: Bars over which a MACD/signal crossover is still reported as "recent".
_MACD_CROSS_LOOKBACK = 5

#: Window for relative volume (last session vs. its N-day average) and for
#: the OBV-trend slope comparison.
_REL_VOLUME_WINDOW = 20
_OBV_TREND_WINDOW = 20

#: OBV is called trending only when its net change over ``_OBV_TREND_WINDOW``
#: exceeds this many average-daily-volume units (filters out noise/flat drift).
_OBV_TREND_MIN_AVG_DAYS = 2.0


def _round_or_none(value, digits: int):
    """Round ``value`` to ``digits`` decimals, or return ``None`` if it's
    missing/NaN."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _rsi_series(close: pd.Series) -> pd.Series:
    """Wilder's 14-period RSI as a full series (one value per bar).

    Uses the standard Wilder smoothing approximation via an EWMA with
    ``alpha = 1/14``. ``avg_loss == 0`` (all recent moves were gains) yields
    ``rs == inf`` -> ``rsi == 100``; a perfectly flat window yields ``NaN``
    (handled by callers). Shared by the scalar :func:`_rsi14` and the
    divergence scan :func:`_rsi_divergence`.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / _RSI_PERIOD, min_periods=_RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / _RSI_PERIOD, min_periods=_RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _rsi14(close: pd.Series) -> "float | None":
    """Wilder's 14-period RSI at the last observation, rounded to 1 decimal,
    or ``None`` if there isn't enough history (fewer than ``_RSI_PERIOD`` + 1
    daily changes)."""
    value = _rsi_series(close).iloc[-1]
    if pd.isna(value):
        return None
    return round(float(value), 1)


def _rsi_reclaim(close: pd.Series) -> "str | None":
    """Detect a fresh RSI reclaim of an extreme line -- the swing "momentum is
    turning" tell.

    * ``"bullish"``: within the last :data:`_RSI_RECLAIM_LOOKBACK` bars RSI dipped
      below :data:`_RSI_OVERSOLD` (oversold) AND the latest value is back at/above
      it -- an oversold bounce being reclaimed.
    * ``"bearish"``: within that window RSI ran above :data:`_RSI_OVERBOUGHT`
      (overbought) AND the latest value is back at/below it -- an overbought
      rollover.
    * ``None``: neither (still oversold/overbought, or never near either line, or
      not enough history).

    Pure and NaN-safe: a too-short/flat window (all-``NaN`` RSI tail) returns
    ``None`` rather than raising.
    """
    rsi = _rsi_series(close).dropna()
    if len(rsi) < 2:
        return None
    window = rsi.iloc[-(_RSI_RECLAIM_LOOKBACK + 1):]
    last = float(window.iloc[-1])
    prior = window.iloc[:-1]
    if prior.empty:
        return None
    if (prior < _RSI_OVERSOLD).any() and last >= _RSI_OVERSOLD:
        return "bullish"
    if (prior > _RSI_OVERBOUGHT).any() and last <= _RSI_OVERBOUGHT:
        return "bearish"
    return None


def _rsi_divergence(df: pd.DataFrame) -> "dict | None":
    """Detect a recent regular RSI divergence between price and momentum.

    Compares the last two price swing highs/lows (``±_RSI_DIVERGENCE_PIVOT``
    fractals on Close) against the RSI at those same bars:

    * **Bearish** -- price makes a *higher* high but RSI makes a *lower*
      high (momentum failing to confirm the new price high -> possible top).
    * **Bullish** -- price makes a *lower* low but RSI makes a *higher* low
      (selling pressure fading -> possible bottom).

    Only fires when the most recent of the two swings is within the last
    :data:`_RSI_DIVERGENCE_RECENCY` bars (a stale divergence isn't
    actionable). If both a bullish and a bearish divergence qualify, the one
    whose latest swing is more recent wins.

    Returns ``{"type": "bullish"|"bearish", "price_prev", "price_last",
    "rsi_prev", "rsi_last", "last_date"}`` (the evidence, so callers can show
    *why*), or ``None`` when there's no qualifying divergence / not enough
    history.
    """
    close = df["Close"]
    k = _RSI_DIVERGENCE_PIVOT
    if len(close) < _RSI_PERIOD + 2 * k + 2:
        return None

    rsi = _rsi_series(close)
    n = len(close)
    highs: "list[tuple]" = []
    lows: "list[tuple]" = []
    for i in range(k, n - k):
        rsi_i = rsi.iloc[i]
        if pd.isna(rsi_i):
            continue
        close_i = close.iloc[i]
        window = close.iloc[i - k:i + k + 1]
        if close_i >= window.max():
            highs.append((i, float(close_i), float(rsi_i), _date_str(close.index[i])))
        if close_i <= window.min():
            lows.append((i, float(close_i), float(rsi_i), _date_str(close.index[i])))

    candidates: "list[tuple]" = []
    if len(highs) >= 2:
        (_, p1, r1, _), (i2, p2, r2, d2) = highs[-2], highs[-1]
        if i2 >= n - _RSI_DIVERGENCE_RECENCY and p2 > p1 and r2 < r1:
            candidates.append((i2, {
                "type": "bearish", "price_prev": round(p1, 2), "price_last": round(p2, 2),
                "rsi_prev": round(r1, 1), "rsi_last": round(r2, 1), "last_date": d2,
            }))
    if len(lows) >= 2:
        (_, p1, r1, _), (i2, p2, r2, d2) = lows[-2], lows[-1]
        if i2 >= n - _RSI_DIVERGENCE_RECENCY and p2 < p1 and r2 > r1:
            candidates.append((i2, {
                "type": "bullish", "price_prev": round(p1, 2), "price_last": round(p2, 2),
                "rsi_prev": round(r1, 1), "rsi_last": round(r2, 1), "last_date": d2,
            }))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])   # most recent swing wins
    return candidates[-1][1]


def _sma(close: pd.Series, window: int) -> pd.Series:
    """Return the simple rolling mean of ``close`` over ``window`` periods."""
    return close.rolling(window=window, min_periods=window).mean()


def _macd(close: pd.Series) -> "tuple[float | None, float | None, float | None, str | None]":
    """Classic 12/26/9 MACD on the Close series.

    ``macd = EMA12 - EMA26``; ``signal = EMA9(macd)``; ``hist = macd -
    signal`` -- all using the recursive (``adjust=False``) EMA convention
    standard for MACD. A crossover is flagged ``"bullish"``/``"bearish"``
    when the histogram's sign now differs from its sign
    :data:`_MACD_CROSS_LOOKBACK` bars ago (a fresh MACD/signal cross).

    Returns ``(macd, signal, hist, cross)`` -- the three values rounded to 3
    decimals -- or ``(None, None, None, None)`` when there isn't enough
    history (fewer than ``_MACD_SLOW + _MACD_SIGNAL`` bars) for a stable
    signal line. This is derived purely from Close (no new data source),
    exactly like RSI/SMA.
    """
    if len(close) < _MACD_SLOW + _MACD_SIGNAL:
        return None, None, None, None

    ema_fast = close.ewm(span=_MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=_MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=_MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal_line

    macd_now, signal_now, hist_now = macd_line.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]
    if pd.isna(macd_now) or pd.isna(signal_now) or pd.isna(hist_now):
        return None, None, None, None

    cross = None
    if len(hist) > _MACD_CROSS_LOOKBACK:
        hist_prev = hist.iloc[-1 - _MACD_CROSS_LOOKBACK]
        if not pd.isna(hist_prev):
            if hist_now > 0 and hist_prev <= 0:
                cross = "bullish"
            elif hist_now < 0 and hist_prev >= 0:
                cross = "bearish"

    return round(float(macd_now), 3), round(float(signal_now), 3), round(float(hist_now), 3), cross


def _volume_signals(df: pd.DataFrame) -> "tuple[float | None, str | None]":
    """Relative volume and OBV trend from the ``Volume`` column.

    * ``rel_volume`` -- the latest session's volume divided by its
      :data:`_REL_VOLUME_WINDOW`-day average (``1.3`` == 30% above average),
      2dp. Confirms whether a move is happening on conviction.
    * ``obv_trend`` -- ``"up"``/``"down"``/``"flat"`` from On-Balance Volume
      (cumulative signed volume: ``+vol`` on up-closes, ``-vol`` on
      down-closes, ``0`` on flat). Called trending only when OBV's net
      change over :data:`_OBV_TREND_WINDOW` exceeds
      :data:`_OBV_TREND_MIN_AVG_DAYS` average-daily-volume units, so noise
      reads as ``"flat"``.

    Returns ``(None, None)`` (each independently) when ``Volume`` is absent,
    all-zero, or history is too short -- volume can be missing/sparse for
    some tickers/ADRs from the price source, so this never assumes it's
    present.
    """
    if "Volume" not in df.columns:
        return None, None
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    if volume.dropna().empty or float(volume.fillna(0).sum()) <= 0:
        return None, None

    rel_volume = None
    if len(volume) >= _REL_VOLUME_WINDOW:
        avg_volume = volume.tail(_REL_VOLUME_WINDOW).mean()
        last_volume = volume.iloc[-1]
        if not pd.isna(avg_volume) and avg_volume > 0 and not pd.isna(last_volume):
            rel_volume = round(float(last_volume) / float(avg_volume), 2)

    obv_trend = None
    close = df["Close"]
    if len(close) > _OBV_TREND_WINDOW:
        direction = close.diff().fillna(0.0)
        signed = volume.fillna(0.0).copy()
        signed[direction < 0] *= -1
        signed[direction == 0] = 0.0
        obv = signed.cumsum()
        obv_now, obv_past = obv.iloc[-1], obv.iloc[-1 - _OBV_TREND_WINDOW]
        avg_volume = volume.tail(_OBV_TREND_WINDOW).mean()
        if not pd.isna(obv_now) and not pd.isna(obv_past) and not pd.isna(avg_volume):
            threshold = avg_volume * _OBV_TREND_MIN_AVG_DAYS
            delta = obv_now - obv_past
            if delta > threshold:
                obv_trend = "up"
            elif delta < -threshold:
                obv_trend = "down"
            else:
                obv_trend = "flat"

    return rel_volume, obv_trend


def _atr14(df: pd.DataFrame) -> "float | None":
    """Wilder's 14-period Average True Range in price units, or ``None`` if
    history is too short.

    True range = ``max(high-low, |high-prev_close|, |low-prev_close|)``,
    smoothed with the same ``alpha = 1/14`` EWMA convention as RSI. When
    ``High``/``Low`` are absent (some ADRs/price sources give Close only), TR
    degrades cleanly to ``|close - prev_close|`` -- a close-only volatility
    proxy -- rather than failing.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    if len(close) < _ATR_PERIOD + 1:
        return None
    prev_close = close.shift(1)
    high = pd.to_numeric(df["High"], errors="coerce") if "High" in df.columns else close
    low = pd.to_numeric(df["Low"], errors="coerce") if "Low" in df.columns else close
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / _ATR_PERIOD, min_periods=_ATR_PERIOD).mean()
    value = atr.iloc[-1]
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _bollinger_squeeze(close: pd.Series) -> "dict | None":
    """Bollinger-band "squeeze" state, or ``None`` if history is too short.

    Band width = ``(upper - lower) / mid = 2 * _BB_STD * stddev / SMA``. The
    current width is ranked as a percentile against its own last
    :data:`_BB_SQUEEZE_LOOKBACK` bars; ``active`` is ``True`` when that
    percentile is at/below :data:`_BB_SQUEEZE_PCT` (width sits in the tightest
    fifth of its recent range -- compression that often precedes a move).
    Direction-agnostic: a squeeze says "expansion likely soon", not which way.

    Returns ``{"active": bool, "bandwidth_pct": float, "percentile": float}``.
    """
    if len(close) < _BB_PERIOD + _BB_SQUEEZE_MIN_SAMPLE:
        return None
    mid = close.rolling(_BB_PERIOD).mean()
    std = close.rolling(_BB_PERIOD).std(ddof=0)
    bandwidth = (2.0 * _BB_STD * std) / mid
    bw = bandwidth.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(bw) < _BB_SQUEEZE_MIN_SAMPLE:
        return None
    window = bw.tail(_BB_SQUEEZE_LOOKBACK)
    current = float(bw.iloc[-1])
    percentile = float((window < current).sum()) / float(len(window)) * 100.0
    return {
        "active": percentile <= _BB_SQUEEZE_PCT,
        "bandwidth_pct": round(current * 100.0, 2),
        "percentile": round(percentile, 1),
    }


def _volume_climax(df: pd.DataFrame) -> "dict | None":
    """Most recent volume-climax / capitulation bar within the last
    :data:`_CLIMAX_LOOKBACK` bars, or ``None`` if none (or no usable volume).

    A climax bar has volume >= :data:`_CLIMAX_VOLUME_MULT` x its trailing
    :data:`_REL_VOLUME_WINDOW`-day average AND a high-low range >=
    :data:`_CLIMAX_RANGE_PCT` of its close -- the wide-range, huge-volume bar
    that marks buyer/seller exhaustion. Scans newest-first and returns the
    first hit, so ``bars_ago == 0`` means it printed on the latest bar.

    Returns ``{"detected": True, "bars_ago": int, "rel_volume": float,
    "direction": "up"|"down", "date": str}``. ``direction`` is the sign of the
    climax bar's close vs. the prior close.
    """
    if "Volume" not in df.columns:
        return None
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    n = len(close)
    if n < _REL_VOLUME_WINDOW + _CLIMAX_LOOKBACK:
        return None
    if volume.dropna().empty or float(volume.fillna(0).sum()) <= 0:
        return None
    high = pd.to_numeric(df["High"], errors="coerce") if "High" in df.columns else close
    low = pd.to_numeric(df["Low"], errors="coerce") if "Low" in df.columns else close
    avg = volume.rolling(_REL_VOLUME_WINDOW).mean()
    for offset in range(_CLIMAX_LOOKBACK):
        i = n - 1 - offset
        v, a, px = volume.iloc[i], avg.iloc[i], close.iloc[i]
        if pd.isna(v) or pd.isna(a) or a <= 0 or pd.isna(px) or px <= 0:
            continue
        rng = float(high.iloc[i] - low.iloc[i]) / float(px)
        rv = float(v) / float(a)
        if rv >= _CLIMAX_VOLUME_MULT and rng >= _CLIMAX_RANGE_PCT:
            prev = close.iloc[i - 1] if i > 0 else px
            direction = "down" if float(px) < float(prev) else "up"
            return {
                "detected": True,
                "bars_ago": offset,
                "rel_volume": round(rv, 1),
                "direction": direction,
                "date": _date_str(close.index[i]),
            }
    return None


def relative_strength(
    close: pd.Series,
    benchmark_close: "pd.Series | None",
    benchmark: str = "SPY",
) -> "dict | None":
    """Price relative strength of ``close`` vs. a benchmark close series.

    Aligns the two series on their common dates, then computes the *relative*
    return over the 1- and 3-month windows: ``stock_return - benchmark_return``
    (in percentage points). Positive means the stock outperformed the
    benchmark over that window; ``leading`` is the 3-month sign. Answers "is
    this weak on its own, or just moving with the market?".

    Pure and defensive: ``None`` benchmark, misaligned/too-short overlap, or a
    non-positive endpoint yields ``None`` (RS simply won't render), never a
    raise.

    Returns ``{"benchmark": str, "rs_1m_pct": float|None,
    "rs_3m_pct": float|None, "leading": bool|None}``.
    """
    if close is None or benchmark_close is None:
        return None
    joined = pd.concat(
        [pd.to_numeric(close, errors="coerce").rename("s"),
         pd.to_numeric(benchmark_close, errors="coerce").rename("b")],
        axis=1,
    ).dropna()
    if len(joined) <= _RETURN_WINDOWS["return_1m_pct"]:
        return None

    def _rel(win: int) -> "float | None":
        if len(joined) <= win:
            return None
        s0, s1 = float(joined["s"].iloc[-1 - win]), float(joined["s"].iloc[-1])
        b0, b1 = float(joined["b"].iloc[-1 - win]), float(joined["b"].iloc[-1])
        if s0 <= 0 or b0 <= 0:
            return None
        # round() of a native float returns a native float -> JSON-serializable
        # (numpy scalars are NOT, and would crash the report payload dump).
        return round(((s1 / s0 - 1.0) - (b1 / b0 - 1.0)) * 100.0, 1)

    rs_1m = _rel(_RETURN_WINDOWS["return_1m_pct"])
    rs_3m = _rel(_RETURN_WINDOWS["return_3m_pct"])
    return {
        "benchmark": benchmark,
        "rs_1m_pct": rs_1m,
        "rs_3m_pct": rs_3m,
        "leading": bool(rs_3m > 0) if rs_3m is not None else None,
    }


def _price_series(df: pd.DataFrame) -> "list[dict]":
    """A compact, multi-resolution close series as ``[{"t": date, "c": close}]``
    (ascending) for the report's range-switchable price chart.

    The most recent ``_PRICE_SERIES_DAILY`` trading days are kept daily; older
    history (up to ``_PRICE_SERIES_MAX_ROWS`` ~ 5 trading years) is
    down-sampled to roughly weekly. The chart slices by date, so the 3m/1y
    views land entirely in the daily tail while the 5y view spans the whole
    (mixed-resolution) series. Kept small so the embedded report payload stays
    light (~500 points for a 5y history)."""
    close = df["Close"].dropna()
    if close.empty:
        return []
    close = close.tail(_PRICE_SERIES_MAX_ROWS)
    if len(close) > _PRICE_SERIES_DAILY:
        older = close.iloc[:-_PRICE_SERIES_DAILY].iloc[::_PRICE_SERIES_WEEKLY_STEP]
        daily = close.iloc[-_PRICE_SERIES_DAILY:]
        close = pd.concat([older, daily])
    series: "list[dict]" = []
    for timestamp, value in close.items():
        if pd.isna(value):
            continue
        series.append({"t": _date_str(timestamp), "c": round(float(value), 2)})
    return series


def _return_pct(close: pd.Series, window: int) -> "float | None":
    """Percentage change of Close over the last ``window`` trading days, 1dp,
    or ``None`` if there isn't enough history (or the reference price is
    missing/zero)."""
    if len(close) <= window:
        return None
    past = close.iloc[-1 - window]
    last = close.iloc[-1]
    if pd.isna(past) or pd.isna(last) or past == 0:
        return None
    return round((float(last) / float(past) - 1) * 100, 1)


def _date_str(timestamp) -> "str | None":
    """``Timestamp`` -> ``"YYYY-MM-DD"`` (or ``None``)."""
    if timestamp is None:
        return None
    return timestamp.strftime("%Y-%m-%d") if hasattr(timestamp, "strftime") else str(timestamp)


def _cluster_pivots(pivots: "list[dict]", tol_frac: float) -> "list[dict]":
    """Greedily merge nearby swing pivots into price zones, preserving the
    evidence behind each zone.

    Sorts ``pivots`` (each ``{"price", "date"}``) ascending by price and
    walks them, growing the current cluster while the next pivot is within
    ``tol_frac`` of the running mean price; otherwise it closes the cluster.
    Each returned zone is ``{"price": mean_price, "touches": member_count,
    "last_touch": most_recent_date}`` -- ``touches`` is how many swing points
    landed in the zone (its strength / why it matters) and ``last_touch`` is
    the newest date the zone was tested (its recency).
    """
    def _finalize(members: "list[dict]") -> dict:
        prices = [m["price"] for m in members]
        dates = [m["date"] for m in members if m.get("date")]
        return {
            "price": sum(prices) / len(prices),
            "touches": len(members),
            "last_touch": max(dates) if dates else None,   # ISO dates sort chronologically
        }

    clusters: "list[dict]" = []
    current: "list[dict]" = []
    for pivot in sorted(pivots, key=lambda p: p["price"]):
        if not current:
            current = [pivot]
            continue
        current_mean = sum(m["price"] for m in current) / len(current)
        if current_mean > 0 and abs(pivot["price"] - current_mean) <= current_mean * tol_frac:
            current.append(pivot)
        else:
            clusters.append(_finalize(current))
            current = [pivot]
    if current:
        clusters.append(_finalize(current))
    return clusters


def _cluster_levels(levels: "list[float]", tol_frac: float) -> "list[dict]":
    """Cluster bare price levels into zones (thin wrapper over
    :func:`_cluster_pivots` for date-less inputs). Each zone is
    ``{"price": mean, "strength": member_count}``."""
    clusters = _cluster_pivots([{"price": v, "date": None} for v in levels], tol_frac)
    return [{"price": c["price"], "strength": c["touches"]} for c in clusters]


def _support_resistance(
    df: pd.DataFrame,
    price: "float | None",
    sma50: "float | None" = None,
    sma200: "float | None" = None,
) -> "tuple[list, list, float | None, float | None, dict | None]":
    """Derive support/resistance zones from swing pivots in the price history.

    Detects swing highs and swing lows (bars whose High/Low is the extreme
    over a +/-:data:`_PIVOT_WINDOW`-bar window) over the most recent
    :data:`_SR_LOOKBACK_DAYS` sessions, plus the absolute high/low as hard
    endpoints, then clusters them into zones (:func:`_cluster_levels`). A
    prior swing of either kind is a potential level regardless of type (a
    broken resistance becomes support), so highs and lows are pooled before
    clustering. Zones are split relative to the current price: those below
    are supports (nearest-first), those above are resistances (nearest-first).

    Falls back to ``Close`` when ``High``/``Low`` columns are absent. Returns
    ``([], [], None, None)`` when there isn't enough history or no usable
    price.

    Each zone carries the evidence behind it (so callers can show *why* it's
    a level and *how strong*): ``touches`` (swing points in the zone -- its
    strength), ``last_touch`` (newest date it was tested -- its recency), and
    ``is_52w_high``/``is_52w_low`` (whether it coincides with the 52-week
    extreme). ``strength`` is kept as an alias of ``touches`` for
    backward-compatibility.

    Fibonacci retracement levels of the dominant swing (highest high <->
    lowest low over the lookback) are folded in as additional evidence.
    Direction (which extreme came first) decides whether ratios are measured
    down from the high or up from the low.

    Major moving averages (``sma50``/``sma200``, when supplied) are folded in
    the same way: a MA that coincides with a swing cluster tags it as a
    confluence zone, and a MA with no nearby shelf becomes a standalone
    candidate (``touches == 0``) -- so a stock resting on its 200-day after a
    vertical run still reports that MA as support/resistance instead of falling
    back to a stale pre-breakout shelf or the 52-week extreme.

    Selection (per side): zones beyond :data:`_SR_MAX_DIST_PCT` from price are
    first dropped (a shelf ~70% away is not an actionable trigger); every
    remaining zone is scored -- a 52-week extreme dominates, a moving-average
    confluence is a strong boost (~a well-tested shelf), swing touches
    accumulate, a Fibonacci confluence adds a smaller boost -- then exactly
    **one near and one far** level are chosen, each the *strongest* in its distance half (so
    both are well-corroborated, not the weakest nearby). Each chosen level is
    then **strengthened by merging every nearby evidence source** within
    :data:`_SR_CORROBORATE_PCT` (summed touches, unioned fib ratios, 52w
    flags) and expressed as a price **range** (``low``/``high``).

    Falls back to ``Close`` when ``High``/``Low`` columns are absent. Returns
    all-empty when there isn't enough history or no usable price.

    Returns:
        ``(supports, resistances, nearest_support, nearest_resistance,
        fibonacci)`` where ``supports``/``resistances`` hold up to two levels
        (near + far) as ``{"low", "high", "price", "dist_pct", "strength",
        "touches", "last_touch", "fib", "ma", "is_52w_high", "is_52w_low"}``
        (``price`` is the range midpoint), the two scalars are the near
        level's midpoint on each side (or ``None``), and ``fibonacci`` is
        ``{"high", "low", "direction", "levels": [{"ratio", "price",
        "dist_pct"}]}`` (or ``None``).
    """
    if price is None or price <= 0 or len(df) < 2 * _PIVOT_WINDOW + 1:
        return [], [], None, None, None

    high = (df["High"] if "High" in df.columns else df["Close"]).tail(_SR_LOOKBACK_DAYS)
    low = (df["Low"] if "Low" in df.columns else df["Close"]).tail(_SR_LOOKBACK_DAYS)

    k = _PIVOT_WINDOW
    n = len(high)
    pivots: "list[dict]" = []
    for i in range(k, n - k):
        hv = high.iloc[i]
        if not pd.isna(hv) and hv >= high.iloc[i - k:i + k + 1].max():
            pivots.append({"price": float(hv), "date": _date_str(high.index[i])})
        lv = low.iloc[i]
        if not pd.isna(lv) and lv <= low.iloc[i - k:i + k + 1].min():
            pivots.append({"price": float(lv), "date": _date_str(low.index[i])})

    # Always represent the absolute extremes (they may sit inside the last k
    # bars and thus escape the pivot scan, yet they're the hardest levels).
    hi_val = lo_val = None
    if not high.dropna().empty:
        hi_val = float(high.max())
        pivots.append({"price": hi_val, "date": _date_str(high.idxmax())})
    if not low.dropna().empty:
        lo_val = float(low.min())
        pivots.append({"price": lo_val, "date": _date_str(low.idxmin())})

    if not pivots:
        return [], [], None, None, None

    clusters = _cluster_pivots(pivots, _SR_CLUSTER_TOL)
    gap = _SR_CLUSTER_TOL

    # --- Fibonacci retracement of the dominant swing (high<->low over the
    # lookback). Direction (which extreme came first) sets whether ratios are
    # measured down from the high (uptrend pullback) or up from the low
    # (downtrend bounce). Fib levels that land on a swing cluster tag it as a
    # confluence zone; the rest become fib-only zones (touches == 0).
    fib_context = None
    if hi_val is not None and lo_val is not None and hi_val > lo_val:
        span = hi_val - lo_val
        high_pos = high.reset_index(drop=True).idxmax()
        low_pos = low.reset_index(drop=True).idxmin()
        direction = "up" if low_pos <= high_pos else "down"
        fib_levels = []
        for ratio in _FIB_RATIOS:
            level_price = (hi_val - ratio * span) if direction == "up" else (lo_val + ratio * span)
            label = f"{ratio * 100:g}%"
            fib_levels.append({"price": level_price, "label": label})

            # Merge onto the nearest swing cluster within tolerance (confluence).
            best, best_diff = None, None
            for cluster in clusters:
                diff = abs(cluster["price"] - level_price)
                if diff <= level_price * gap and (best_diff is None or diff < best_diff):
                    best, best_diff = cluster, diff
            if best is not None:
                if best.get("fib") is None or best_diff < best.get("_fib_diff", float("inf")):
                    best["fib"], best["_fib_diff"] = label, best_diff
            else:
                clusters.append({"price": level_price, "touches": 0, "last_touch": None, "fib": label})

        fib_context = {
            "high": round(hi_val, 2), "low": round(lo_val, 2), "direction": direction,
            "levels": [
                {"ratio": lvl["label"], "price": round(lvl["price"], 2),
                 "dist_pct": round((lvl["price"] / price - 1) * 100, 1)}
                for lvl in fib_levels
            ],
        }

    # --- Major moving averages (SMA50/SMA200) as dynamic support/resistance.
    # Handled exactly like Fibonacci: a MA that lands on a swing cluster tags it
    # as a confluence zone (a shelf reinforced by the 200-day is a strong,
    # widely-watched level); a MA with no nearby shelf becomes a standalone
    # candidate (touches == 0) -- this is what catches a stock that ran
    # vertically past every horizontal pivot and is now resting on its MA.
    for ma_value, ma_label in ((sma50, "SMA50"), (sma200, "SMA200")):
        if ma_value is None or ma_value <= 0:
            continue
        best, best_diff = None, None
        for cluster in clusters:
            diff = abs(cluster["price"] - ma_value)
            if diff <= ma_value * gap and (best_diff is None or diff < best_diff):
                best, best_diff = cluster, diff
        if best is not None:
            existing = best.get("ma")
            best["ma"] = f"{existing}/{ma_label}" if existing else ma_label
        else:
            clusters.append({"price": float(ma_value), "touches": 0,
                             "last_touch": None, "ma": ma_label})

    # Flag 52-week extremes and score every zone (52w > tested swing > fib).
    for zone in clusters:
        zone["is_52w_high"] = hi_val is not None and abs(zone["price"] - hi_val) <= hi_val * gap
        zone["is_52w_low"] = lo_val is not None and abs(zone["price"] - lo_val) <= lo_val * gap
        is_52w = zone["is_52w_high"] or zone["is_52w_low"]
        score_52w = 0
        if is_52w:
            dist_pct = abs(zone["price"] / price - 1) * 100
            if dist_pct <= _SR_52W_FAR_THRESHOLD_PCT:
                score_52w = _SR_SCORE_52W
            else:
                score_52w = max(_SR_SCORE_PER_TOUCH, _SR_SCORE_52W * (_SR_52W_FAR_THRESHOLD_PCT / dist_pct))
        zone["score"] = (
            score_52w
            + zone.get("touches", 0) * _SR_SCORE_PER_TOUCH
            + (_SR_SCORE_FIB if zone.get("fib") else 0)
            + (_SR_SCORE_MA if zone.get("ma") else 0)
        )

    def _corroborate(pick: dict) -> dict:
        """Merge all evidence within ``_SR_CORROBORATE_PCT`` of ``pick`` into a
        single strong zone and express it as a price range."""
        window = pick["price"] * _SR_CORROBORATE_PCT
        members = [z for z in clusters if abs(z["price"] - pick["price"]) <= window]
        prices = [z["price"] for z in members]
        fibs = [z["fib"] for z in members if z.get("fib")]
        mas = [z["ma"] for z in members if z.get("ma")]
        last_touches = [z["last_touch"] for z in members if z.get("last_touch")]
        lo_p, hi_p = min(prices), max(prices)
        if hi_p - lo_p < pick["price"] * _SR_BAND_MIN_PCT:
            half = pick["price"] * _SR_BAND_MIN_PCT / 2.0
            lo_p, hi_p = pick["price"] - half, pick["price"] + half
        mid = (lo_p + hi_p) / 2.0
        return {
            "low": round(lo_p, 2), "high": round(hi_p, 2), "price": round(mid, 2),
            "dist_pct": round((mid / price - 1) * 100, 1),
            "touches": sum(z.get("touches", 0) for z in members),
            "strength": sum(z.get("touches", 0) for z in members),   # back-compat alias
            "fib": "/".join(dict.fromkeys(fibs)) if fibs else None,
            "ma": "/".join(dict.fromkeys(m for entry in mas for m in entry.split("/"))) if mas else None,
            "last_touch": max(last_touches) if last_touches else None,
            "is_52w_high": any(z.get("is_52w_high") for z in members),
            "is_52w_low": any(z.get("is_52w_low") for z in members),
        }

    def _pick_near_far(side_zones: "list[dict]") -> "list[dict]":
        """Pick one near + one far level, each the strongest in its distance
        half (so both are well-corroborated, not the weakest nearby)."""
        if not side_zones:
            return []
        by_distance = sorted(side_zones, key=lambda z: abs(z["price"] - price))
        if len(by_distance) == 1:
            return [_corroborate(by_distance[0])]
        split = (len(by_distance) + 1) // 2
        near = max(by_distance[:split], key=lambda z: (z["score"], -abs(z["price"] - price)))
        far = max(by_distance[split:], key=lambda z: (z["score"], -abs(z["price"] - price)))
        near_fmt, far_fmt = _corroborate(near), _corroborate(far)
        # If corroboration collapsed them onto the same band, keep just the near.
        if far_fmt["low"] <= near_fmt["high"] and far_fmt["high"] >= near_fmt["low"]:
            return [near_fmt]
        return [near_fmt, far_fmt]

    # Exclude levels beyond _SR_MAX_DIST_PCT from price: a shelf ~70% away is
    # not an actionable trigger, so the "far" slot is filled by the nearest
    # strong in-range level instead of the 52-week extreme (or left empty).
    max_dist = _SR_MAX_DIST_PCT / 100.0
    lo_bound, hi_bound = price * (1 - max_dist), price * (1 + max_dist)
    supports = _pick_near_far(
        [c for c in clusters if lo_bound <= c["price"] < price * (1 - gap)]
    )
    resistances = _pick_near_far(
        [c for c in clusters if price * (1 + gap) < c["price"] <= hi_bound]
    )
    supports.sort(key=lambda z: z["price"], reverse=True)   # nearest below first
    resistances.sort(key=lambda z: z["price"])              # nearest above first

    nearest_support = supports[0]["price"] if supports else None
    nearest_resistance = resistances[0]["price"] if resistances else None
    return supports, resistances, nearest_support, nearest_resistance, fib_context


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
        * ``return_1m_pct`` / ``return_3m_pct`` / ``return_6m_pct``: Close
          percentage change over the last 21 / 63 / 126 trading days, 1dp
          (momentum), or ``None`` if history is shorter than the window.
        * ``macd`` / ``macd_signal`` / ``macd_hist``: classic 12/26/9 MACD
          line, signal line, and histogram (3dp), or ``None`` if history is
          too short (see :func:`_macd`).
        * ``macd_cross``: ``"bullish"``/``"bearish"`` on a fresh MACD/signal
          crossover within the last 5 bars, else ``None``.
        * ``rel_volume``: latest volume / 20-day average volume (2dp), or
          ``None``; ``obv_trend``: ``"up"``/``"down"``/``"flat"`` from
          On-Balance Volume, or ``None`` (see :func:`_volume_signals`).
        * ``rsi_divergence``: ``"bullish"``/``"bearish"``/``None`` -- a
          recent price-vs-RSI divergence; ``rsi_divergence_detail`` carries
          the swing prices/RSI values behind it (see
          :func:`_rsi_divergence`).
        * ``rsi_reclaim``: ``"bullish"``/``"bearish"``/``None`` -- a fresh RSI
          reclaim of the oversold/overbought line within the last
          :data:`_RSI_RECLAIM_LOOKBACK` bars (see :func:`_rsi_reclaim`); a
          swing "momentum turning" input for the report's reversal-confirmation
          checklist.
        * ``atr14``: Wilder's 14-period Average True Range in price units, or
          ``None`` (see :func:`_atr14`) -- volatility-scaled swing stop distance.
        * ``bb_squeeze``: ``{"active", "bandwidth_pct", "percentile"}`` or
          ``None`` -- Bollinger band-width compression state (see
          :func:`_bollinger_squeeze`).
        * ``volume_climax``: ``{"detected", "bars_ago", "rel_volume",
          "direction", "date"}`` or ``None`` -- a recent capitulation/climax
          bar (see :func:`_volume_climax`).
        * ``support_levels`` / ``resistance_levels``: one near + one far zone
          below / above the current price (nearest-first), each a price
          **range** ``{"low", "high", "price", "dist_pct", "strength",
          "touches", "last_touch", "fib", "ma", "is_52w_high", "is_52w_low"}`` (see
          :func:`_support_resistance`) -- each is the strongest in its
          distance half, corroborated by merging nearby touches/fib/52w
          evidence. Levels beyond :data:`_SR_MAX_DIST_PCT` from price are
          excluded (not actionable triggers). Empty lists when history is too
          short.
        * ``fibonacci``: ``{"high", "low", "direction", "levels"}`` for the
          dominant swing's retracement grid, or ``None``.
        * ``price_series``: a compact multi-resolution ``{"t": date, "c":
          close}`` series (ascending, ~500 points spanning up to ~5y: daily
          tail + weekly older) for the report's range-switchable price chart
          (see :func:`_price_series`).
        * ``nearest_support`` / ``nearest_resistance``: the closest zone
          price on each side, or ``None``.
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

    returns_pct = {key: _return_pct(close, window) for key, window in _RETURN_WINDOWS.items()}

    macd, macd_signal, macd_hist, macd_cross = _macd(close)
    rel_volume, obv_trend = _volume_signals(df)
    rsi_divergence_detail = _rsi_divergence(df)

    supports, resistances, nearest_support, nearest_resistance, fibonacci = _support_resistance(
        df, price, sma50, sma200
    )

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
        "return_1m_pct": returns_pct["return_1m_pct"],
        "return_3m_pct": returns_pct["return_3m_pct"],
        "return_6m_pct": returns_pct["return_6m_pct"],
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "macd_cross": macd_cross,
        "rel_volume": rel_volume,
        "obv_trend": obv_trend,
        "rsi_divergence": (rsi_divergence_detail or {}).get("type"),
        "rsi_divergence_detail": rsi_divergence_detail,
        "rsi_reclaim": _rsi_reclaim(close),
        "atr14": _atr14(df),
        "bb_squeeze": _bollinger_squeeze(close),
        "volume_climax": _volume_climax(df),
        "support_levels": supports,
        "resistance_levels": resistances,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "fibonacci": fibonacci,
        "price_series": _price_series(df),
    }

    logger.debug("Computed technical indicators: %s", indicators)
    return indicators
