"""Unit tests for sec_analyzer.technical (indicators + verdict) and the
Stooq/yfinance price-fetching layer in sec_analyzer.fetch.prices.

No real network access is used anywhere in this module: the price-history
tests monkeypatch ``requests.get`` (and, for the fallback path, force
``import yfinance`` to fail via ``sys.modules``) rather than hitting Stooq
or Yahoo Finance.
"""

import io
import sys
from datetime import date

import pandas as pd
import pytest

from sec_analyzer.config import Config
from sec_analyzer.fetch import prices
from sec_analyzer.technical.indicators import compute_indicators
from sec_analyzer.technical.verdict import technical_verdict

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_df(closes, start="2023-01-02"):
    """Build a minimal OHLCV price-history DataFrame matching the shape
    ``get_price_history`` returns (Date index, ascending, Open/High/Low/
    Close/Volume columns). Open/High/Low are just set equal to Close since
    none of the code under test here reads them."""
    idx = pd.bdate_range(start=start, periods=len(closes))
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------


def test_compute_indicators_rising_series_is_strong_and_near_high():
    """A steadily rising Close series: RSI pegs high (all gains, no
    losses), price sits above SMA50, and the price is at (or essentially
    at) the top of its own 52-week range."""
    closes = [100 + i for i in range(100)]
    df = _make_df(closes)

    ind = compute_indicators(df)

    assert ind["rsi14"] is not None and ind["rsi14"] > 60
    assert ind["sma50"] is not None and ind["sma50"] < ind["price"]
    assert ind["dist_sma50_pct"] is not None and ind["dist_sma50_pct"] > 0
    assert ind["range_position_pct"] is not None and ind["range_position_pct"] >= 95
    assert ind["price"] == 199.0
    assert ind["as_of"] == df.index[-1].strftime("%Y-%m-%d")


def test_compute_indicators_falling_series_is_oversold():
    """A steadily falling Close series: RSI bottoms out (all losses, no
    gains), which the verdict rule must read as oversold regardless of the
    SMA50 relationship."""
    closes = [300 - i for i in range(100)]
    df = _make_df(closes)

    ind = compute_indicators(df)
    assert ind["rsi14"] is not None and ind["rsi14"] < 30

    verdict = technical_verdict(ind)
    assert verdict["verdict"] == "AŞIRI SATIM"


def test_technical_verdict_overbought_when_rsi_high_and_price_above_sma50():
    """RSI > 70 and price > SMA50 together must trigger the overbought
    verdict."""
    closes = [100 + i for i in range(100)]
    df = _make_df(closes)
    ind = compute_indicators(df)

    verdict = technical_verdict(ind)

    assert verdict["verdict"] == "AŞIRI ALIM"
    assert "RSI" in verdict["verdict_detail"]


def test_compute_indicators_short_flat_series_has_no_exceptions():
    """A short (20-row), perfectly flat series must not raise anywhere in
    the pipeline: SMA200/SMA50 need more history than is available, RSI is
    undefined for a zero-volatility series (0/0), and the verdict must
    fall back cleanly to neutral/insufficient-data."""
    closes = [100.0] * 20
    df = _make_df(closes)

    ind = compute_indicators(df)

    assert ind["sma50"] is None
    assert ind["sma200"] is None
    assert ind["rsi14"] is None
    assert ind["golden_cross"] is None
    assert ind["death_cross"] is None

    verdict = technical_verdict(ind)
    assert verdict["verdict"] == "NÖTR"
    assert verdict["verdict_detail"] == "yetersiz veri"


def test_compute_indicators_golden_cross_detected():
    """Engineer a long decline followed by a strong, sustained rise so
    that SMA50 is clearly below SMA200 sixty trading days ago and clearly
    above it now -- the textbook golden-cross setup.

    The values below are plain arithmetic (linear) sequences chosen so the
    SMA50/SMA200 values at both the "now" and "60 days ago" points can be
    (and were) verified by hand via the arithmetic-series average formula
    ``(first + last) / 2``:

    * Now (last row): SMA50 ~= 577, SMA200 ~= 374.5 -> SMA50 above SMA200.
    * 60 trading days earlier: SMA50 ~= 226, SMA200 ~= 403 -> SMA50 below
      SMA200.
    """
    decline = [1000 - 4 * i for i in range(220)]
    rise = [130 + 6 * j for j in range(100)]
    closes = decline + rise
    df = _make_df(closes)

    ind = compute_indicators(df)

    assert ind["sma50_above_sma200"] is True
    assert ind["golden_cross"] is True
    assert ind["death_cross"] is False


# ---------------------------------------------------------------------------
# momentum returns + support/resistance
# ---------------------------------------------------------------------------


def test_return_windows_computed_from_close():
    # 130 rising bars: last=229. return over 21d = 229/208 - 1; 63d = 229/166-1.
    closes = [100 + i for i in range(130)]
    ind = compute_indicators(_make_df(closes))
    assert ind["return_1m_pct"] == pytest.approx((229 / 208 - 1) * 100, abs=0.05)
    assert ind["return_3m_pct"] == pytest.approx((229 / 166 - 1) * 100, abs=0.05)
    assert ind["return_6m_pct"] == pytest.approx((229 / 103 - 1) * 100, abs=0.05)


def test_return_windows_none_when_history_too_short():
    ind = compute_indicators(_make_df([100.0] * 30))
    assert ind["return_3m_pct"] is None   # needs > 63 bars
    assert ind["return_6m_pct"] is None   # needs > 126 bars


def test_cluster_levels_merges_within_tolerance_and_counts_strength():
    from sec_analyzer.technical.indicators import _cluster_levels

    # [100, 100.5] cluster (0.5 <= 100.25*1.5% = 1.50); 110 starts a new
    # cluster; 111 joins it (1 <= 110*1.5% = 1.65). Two zones, strengths 2/2.
    clusters = _cluster_levels([111.0, 100.0, 110.0, 100.5], 0.015)
    assert clusters == [
        {"price": pytest.approx(100.25), "strength": 2},
        {"price": pytest.approx(110.5), "strength": 2},
    ]


def _zigzag_df():
    """A Close that oscillates between ~90 and ~110 several times, then ends
    at 100 -- so ~110 is a tested resistance above and ~90 a tested support
    below. High/Low straddle Close by +/-0.5%."""
    anchors = [100, 110, 90, 111, 89, 109, 91, 110, 90, 100]
    per = 14
    closes = []
    for a, b in zip(anchors, anchors[1:]):
        for t in range(per):
            closes.append(a + (b - a) * t / per)
    closes.append(100.0)
    idx = pd.bdate_range(start="2022-01-03", periods=len(closes))
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.005 for c in closes],
            "Low": [c * 0.995 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def test_support_resistance_brackets_current_price():
    ind = compute_indicators(_zigzag_df())
    price = ind["price"]

    ns, nr = ind["nearest_support"], ind["nearest_resistance"]
    assert ns is not None and nr is not None
    assert ns < price < nr
    # Every returned support is below price and every resistance above.
    assert all(s["price"] < price for s in ind["support_levels"])
    assert all(r["price"] > price for r in ind["resistance_levels"])
    # dist_pct sign matches side.
    assert all(s["dist_pct"] < 0 for s in ind["support_levels"])
    assert all(r["dist_pct"] > 0 for r in ind["resistance_levels"])


def test_support_resistance_carries_why_and_strength_evidence():
    ind = compute_indicators(_zigzag_df())
    all_levels = ind["support_levels"] + ind["resistance_levels"]
    assert all_levels  # non-empty

    for lvl in all_levels:
        # Every zone is justified by either a swing touch or a Fibonacci level.
        assert lvl["touches"] >= 1 or lvl.get("fib")
        assert lvl["strength"] == lvl["touches"]        # back-compat alias
        assert "last_touch" in lvl                       # date string or None
        assert isinstance(lvl["is_52w_high"], bool)
        assert isinstance(lvl["is_52w_low"], bool)
        if lvl["last_touch"] is not None:
            # ISO-ish date string.
            assert len(lvl["last_touch"]) == 10 and lvl["last_touch"][4] == "-"


def test_support_resistance_empty_on_short_history():
    ind = compute_indicators(_make_df([100.0] * 8))
    assert ind["support_levels"] == []
    assert ind["resistance_levels"] == []
    assert ind["nearest_support"] is None
    assert ind["nearest_resistance"] is None
    assert ind["fibonacci"] is None


def test_fibonacci_levels_and_confluence_tagging():
    # Uptrend swing from ~100 to ~200, then a pullback toward ~150 (the 50%
    # retracement). Fib grid is drawn DOWN from the high (uptrend).
    def _ramp(a, b, n):
        return [a + (b - a) * t / n for t in range(n)]

    ind = compute_indicators(_make_df(_ramp(100, 200, 120) + _ramp(200, 150, 30)))

    fib = ind["fibonacci"]
    assert fib is not None and fib["direction"] == "up"
    assert fib["low"] < fib["high"]
    prices = {lvl["ratio"]: lvl["price"] for lvl in fib["levels"]}
    # 50% retracement of ~100..200 sits at ~150.
    assert abs(prices["50%"] - (fib["low"] + 0.5 * (fib["high"] - fib["low"]))) < 0.5

    # Fibonacci levels are folded into the S/R zones (tagged via "fib").
    tagged = [z for z in ind["support_levels"] + ind["resistance_levels"] if z.get("fib")]
    assert tagged
    # Fib-only zones (no swing touches) still surface as levels.
    assert any(z["touches"] == 0 and z.get("fib") for z in tagged)
    # The swing's own extremes are flagged where they surface (the ~200 high
    # is a resistance, the ~100 low a support).
    assert any(r["is_52w_high"] for r in ind["resistance_levels"])
    assert any(s["is_52w_low"] for s in ind["support_levels"])


def test_price_series_is_ascending_capped_and_ends_at_latest_close():
    closes = [100 + i * 0.1 for i in range(400)]
    ind = compute_indicators(_make_df(closes))
    series = ind["price_series"]

    # Multi-resolution contract: the recent tail is kept daily while older
    # history is down-sampled to ~weekly, so a 400-row input is compressed
    # (older thinned) yet stays well under the ~5y cap (~520 points max).
    assert 2 <= len(series) <= 520
    assert 252 < len(series) < 400                      # older thinned, recent daily
    assert all({"t", "c"} <= set(p) for p in series)
    assert [p["t"] for p in series] == sorted(p["t"] for p in series)   # ascending
    assert series[-1]["c"] == ind["price"]              # last point == current price

    # The most recent points are at daily (business-day) resolution.
    def _d(s):
        y, m, day = (int(x) for x in s.split("-"))
        return date(y, m, day)
    assert (_d(series[-1]["t"]) - _d(series[-2]["t"])).days <= 4


def _multi_swing_df():
    def ramp(a, b, n):
        return [a + (b - a) * t / n for t in range(n)]

    closes = (
        ramp(100, 130, 20) + ramp(130, 105, 15) + ramp(105, 135, 20) + ramp(135, 108, 15)
        + ramp(108, 150, 25) + ramp(150, 120, 15) + ramp(120, 145, 20) + ramp(145, 125, 10)
    )
    return _make_df(closes)


def test_support_resistance_picks_one_near_one_far_as_ranges():
    ind = compute_indicators(_multi_swing_df())
    price = ind["price"]

    for side in ("support_levels", "resistance_levels"):
        levels = ind[side]
        assert 1 <= len(levels) <= 2, (side, len(levels))
        for lvl in levels:
            # Every level is a range (low <= midpoint <= high).
            assert lvl["low"] <= lvl["price"] <= lvl["high"]
        if len(levels) == 2:
            # One is nearer, one is farther (distinct distances to price).
            d0 = abs(levels[0]["price"] - price)
            d1 = abs(levels[1]["price"] - price)
            assert d0 != d1


def test_support_resistance_prefers_52w_over_fib_in_its_bucket():
    # The 52-week extremes outscore fib-only levels, so where a 52w extreme and
    # fib levels compete in the same (far) distance bucket, the 52w wins.
    ind = compute_indicators(_multi_swing_df())
    far_resistance = max(ind["resistance_levels"], key=lambda z: z["price"])
    far_support = min(ind["support_levels"], key=lambda z: z["price"])
    assert far_resistance["is_52w_high"] is True
    assert far_support["is_52w_low"] is True


def test_support_resistance_excludes_levels_beyond_max_distance():
    # A stock that ran vertically from a ~50 base to ~190: the 52-week low sits
    # ~73% below price -- too far to be an actionable trigger. It must NOT
    # surface as a support level; the reported levels stay within the distance
    # cap, and the "far" slot is filled by a nearer in-range level instead.
    from sec_analyzer.technical.indicators import _SR_MAX_DIST_PCT

    base = [50.0] * 30
    ramp = [50.0 + (i + 1) * 2.33 for i in range(60)]     # monotonic run to ~190
    ind = compute_indicators(_make_df(base + ramp))

    all_levels = ind["support_levels"] + ind["resistance_levels"]
    assert all_levels, "expected at least one in-range level"
    # Every reported level is within the distance cap...
    assert all(abs(lvl["dist_pct"]) <= _SR_MAX_DIST_PCT for lvl in all_levels)
    # ...and the far ~50 base (a 52-week low ~73% below) is gone.
    assert not any(s.get("is_52w_low") for s in ind["support_levels"])


def test_support_resistance_surfaces_moving_average_after_vertical_run():
    # A stock that ran vertically past every horizontal shelf: the only swing
    # pivots sit far below (the pre-run base / 52-week low), but the 200-day is
    # resting just under price. The MA must surface as the near support instead
    # of falling back to the distant base -- this is the MRVL case.
    from sec_analyzer.technical.indicators import _support_resistance

    base = [50.0] * 30                                    # one pre-run shelf ~50
    ramp = [50.0 + (i + 1) * 2.33 for i in range(60)]     # monotonic run to ~190
    df = _make_df(base + ramp)
    price = float(df["Close"].iloc[-1])

    supports, _resistances, nearest_support, _nr, _fib = _support_resistance(
        df, price, sma50=None, sma200=150.0
    )

    # The 200-day is folded in as a support and tagged with its MA.
    ma_supports = [s for s in supports if s.get("ma") and "SMA200" in s["ma"]]
    assert ma_supports, supports
    # It is the *near* support -- closer than the distant pre-run base / 52w low.
    near = min(supports, key=lambda z: abs(z["price"] - price))
    assert near.get("ma") and "SMA200" in near["ma"]
    assert near["price"] == nearest_support
    far = min(supports, key=lambda z: z["price"])
    assert abs(near["price"] - price) < abs(far["price"] - price)


def test_support_resistance_ignores_absent_moving_averages():
    # When no SMAs are supplied the behaviour is unchanged (no MA-tagged zones),
    # so short-history tickers without a valid SMA200 are unaffected.
    from sec_analyzer.technical.indicators import _support_resistance

    df = _multi_swing_df()
    price = float(df["Close"].iloc[-1])
    supports, resistances, _ns, _nr, _fib = _support_resistance(df, price)   # sma50/sma200 default None
    tagged = [z for z in supports + resistances if z.get("ma")]
    assert tagged == []


# ---------------------------------------------------------------------------
# MACD + volume signals
# ---------------------------------------------------------------------------


def test_macd_bullish_on_sustained_rise():
    from sec_analyzer.technical.indicators import _macd

    macd, signal, hist, cross = _macd(pd.Series([100 + i * 0.8 for i in range(80)]))
    # A steady uptrend keeps the fast EMA above the slow EMA and MACD above
    # its signal line -> positive histogram (bullish regime).
    assert macd > signal and hist > 0


def test_macd_bearish_on_sustained_decline():
    from sec_analyzer.technical.indicators import _macd

    macd, signal, hist, cross = _macd(pd.Series([200 - i * 0.8 for i in range(80)]))
    assert macd < signal and hist < 0


def test_macd_none_when_history_too_short():
    from sec_analyzer.technical.indicators import _macd

    assert _macd(pd.Series([100.0] * 20)) == (None, None, None, None)


def test_macd_flags_fresh_bullish_crossover():
    from sec_analyzer.technical.indicators import _macd

    # Long decline then a sharp rally. Evaluate the series exactly at the bar
    # where the histogram first flips positive, so the cross sits inside the
    # 5-bar "recent" window -> "bullish".
    series = pd.Series([200 - i for i in range(50)] + [155 + i * 4 for i in range(15)])
    ef = series.ewm(span=12, adjust=False).mean()
    es = series.ewm(span=26, adjust=False).mean()
    hist = (ef - es) - (ef - es).ewm(span=9, adjust=False).mean()
    flip = next(i for i in range(1, len(hist)) if hist.iloc[i] > 0 and hist.iloc[i - 1] <= 0)

    assert _macd(series.iloc[: flip + 1])[3] == "bullish"
    # Many bars past the flip, the cross is no longer "recent".
    assert _macd(series)[3] is None


def _volume_df(closes, volumes):
    idx = pd.bdate_range(start="2022-01-03", periods=len(closes))
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
            "Close": closes, "Volume": volumes,
        },
        index=idx,
    )


def test_relative_volume_and_obv_trend_on_rising_price_with_volume_spike():
    from sec_analyzer.technical.indicators import _volume_signals

    closes = [100 + i * 0.5 for i in range(60)]
    volumes = [1_000_000] * 59 + [3_000_000]   # a volume spike on the last session
    rel, obv = _volume_signals(_volume_df(closes, volumes))
    # The 20-day average INCLUDES the spike day: (19*1M + 3M)/20 = 1.1M,
    # so rel = 3M / 1.1M = 2.727... -> 2.73.
    assert rel == pytest.approx(2.73, abs=0.01)
    # Rising closes with positive volume accumulate OBV upward.
    assert obv == "up"


def test_volume_signals_none_when_volume_absent_or_zero():
    from sec_analyzer.technical.indicators import _volume_signals

    closes = [100 + i * 0.5 for i in range(60)]
    rel, obv = _volume_signals(_volume_df(closes, [0] * 60))
    assert rel is None and obv is None


# ---------------------------------------------------------------------------
# RSI divergence
# ---------------------------------------------------------------------------


def _ramp(a, b, n):
    return [a + (b - a) * t / n for t in range(n)]


def test_rsi_bearish_divergence_higher_price_high_lower_rsi():
    from sec_analyzer.technical.indicators import _rsi_divergence

    # Sharp rally to peak1, pullback, then a SLOW grind to a higher peak2 --
    # the slower advance leaves RSI lower at the higher price high.
    closes = [100.0] * 20 + _ramp(100, 140, 10) + _ramp(140, 110, 10) + _ramp(110, 142, 40) + _ramp(142, 130, 8)
    d = _rsi_divergence(_make_df(closes))
    assert d is not None and d["type"] == "bearish"
    assert d["price_last"] > d["price_prev"] and d["rsi_last"] < d["rsi_prev"]


def test_rsi_bullish_divergence_lower_price_low_higher_rsi():
    from sec_analyzer.technical.indicators import _rsi_divergence

    closes = [100.0] * 20 + _ramp(100, 60, 10) + _ramp(60, 95, 10) + _ramp(95, 58, 40) + _ramp(58, 72, 8)
    d = _rsi_divergence(_make_df(closes))
    assert d is not None and d["type"] == "bullish"
    assert d["price_last"] < d["price_prev"] and d["rsi_last"] > d["rsi_prev"]


def test_rsi_divergence_none_on_clean_uptrend_and_short_history():
    from sec_analyzer.technical.indicators import _rsi_divergence

    # Price and RSI both make higher highs -> no divergence.
    assert _rsi_divergence(_make_df([100 + i * 0.7 for i in range(120)])) is None
    # Not enough history.
    assert _rsi_divergence(_make_df([100.0] * 20)) is None


def test_compute_indicators_exposes_rsi_divergence_fields():
    closes = [100.0] * 20 + _ramp(100, 140, 10) + _ramp(140, 110, 10) + _ramp(110, 142, 40) + _ramp(142, 130, 8)
    ind = compute_indicators(_make_df(closes))
    assert ind["rsi_divergence"] == "bearish"
    assert ind["rsi_divergence_detail"]["type"] == "bearish"


# ---------------------------------------------------------------------------
# technical_verdict
# ---------------------------------------------------------------------------


def _sample_indicators(**overrides):
    base = {
        "rsi14": 74.0,
        "price": 112.0,
        "sma50": 100.0,
        "sma200": 90.0,
        "dist_sma50_pct": 12.0,
        "dist_sma200_pct": 24.4,
        "high_52w": 120.0,
        "low_52w": 80.0,
        "range_position_pct": 80.0,
        "volatility_20d": 0.32,
        "golden_cross": False,
        "death_cross": False,
        "sma50_above_sma200": True,
        "as_of": "2024-01-01",
    }
    base.update(overrides)
    return base


def test_verdict_detail_format_matches_spec_example():
    ind = _sample_indicators()
    result = technical_verdict(ind)
    assert result["verdict_detail"] == "RSI 74, SMA50 +%12"


def test_verdict_neutral_when_rsi_missing():
    ind = _sample_indicators(rsi14=None)
    result = technical_verdict(ind)
    assert result["verdict"] == "NÖTR"
    assert result["verdict_detail"] == "yetersiz veri"


def test_horizon_summary_3m_mentions_rsi():
    ind = _sample_indicators()
    result = technical_verdict(ind, horizon="3m")
    assert result["horizon"] == "3m"
    assert "RSI" in result["horizon_summary"]


def test_horizon_summary_1y_is_balanced():
    ind = _sample_indicators()
    result = technical_verdict(ind, horizon="1y")
    assert result["horizon"] == "1y"
    assert "RSI" in result["horizon_summary"]
    assert "SMA" in result["horizon_summary"]


def test_horizon_summary_5y_notes_rsi_not_relevant_and_frames_sma200_as_entry_timing():
    ind = _sample_indicators()
    result = technical_verdict(ind, horizon="5y")
    assert result["horizon"] == "5y"
    assert "RSI" in result["horizon_summary"]
    assert "SMA200" in result["horizon_summary"]
    assert "giriş" in result["horizon_summary"].lower()


# ---------------------------------------------------------------------------
# get_price_history / latest_price (network mocked out)
# ---------------------------------------------------------------------------


def _fake_stooq_csv(n=40, start_price=100.0):
    """Build a small, valid Stooq-shaped CSV payload as plain text."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    lines = ["Date,Open,High,Low,Close,Volume"]
    price = start_price
    for d in dates:
        price += 0.5
        lines.append(
            f"{d.date()},{price - 0.3:.2f},{price + 0.5:.2f},{price - 0.5:.2f},{price:.2f},1000000"
        )
    return "\n".join(lines) + "\n"


class _FakeResponse:
    """Minimal stand-in for requests.Response used by the fetch tests."""

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_get_price_history_stooq_success_writes_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    csv_text = _fake_stooq_csv(40)
    captured_url = {}

    def _fake_get(url, *a, **k):
        captured_url["url"] = url
        return _FakeResponse(csv_text)

    monkeypatch.setattr(prices.requests, "get", _fake_get)

    df, source = prices.get_price_history("FAKE", no_cache=True)

    assert source == "stooq"
    assert len(df) == 40
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    # The Stooq request must not carry any d1/d2 date-range params -- those
    # would narrow the response to a fixed lookback window instead of
    # Stooq's default behavior of returning the full available history.
    assert "d1=" not in captured_url["url"]
    assert "d2=" not in captured_url["url"]

    cache_file = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    assert cache_file.exists()


def test_get_price_history_cache_hit_skips_network(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    csv_text = _fake_stooq_csv(40)
    cached_df = (
        pd.read_csv(io.StringIO(csv_text), parse_dates=["Date"]).set_index("Date").sort_index()
    )
    cache_path = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    cached_df.to_csv(cache_path, index_label="Date")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("network should not be hit on a fresh cache hit")

    monkeypatch.setattr(prices.requests, "get", _fail_if_called)

    df, source = prices.get_price_history("FAKE")

    assert source == "cache(stooq)"
    assert len(df) == 40


def test_get_price_history_ignores_stale_narrow_cache_from_old_filename(monkeypatch, tmp_path):
    """A cache file written under the pre-fix filename (unsuffixed,
    ``prices_FAKE.csv``) -- e.g. one narrowed to ~2 years by yfinance's old
    hardcoded ``period="2y"`` -- must simply be ignored (different
    filename), not read as if it were the new full-history cache."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    stale_csv = _fake_stooq_csv(40)
    stale_df = (
        pd.read_csv(io.StringIO(stale_csv), parse_dates=["Date"]).set_index("Date").sort_index()
    )
    old_style_path = tmp_path / "prices_FAKE.csv"
    stale_df.to_csv(old_style_path, index_label="Date")

    fresh_csv = _fake_stooq_csv(80)
    monkeypatch.setattr(prices.requests, "get", lambda *a, **k: _FakeResponse(fresh_csv))

    df, source = prices.get_price_history("FAKE")

    assert source == "stooq"
    assert len(df) == 80


def test_fetch_yfinance_defaults_to_max_period(monkeypatch):
    """The yfinance fallback must default to the full available history
    (``period="max"``), not the old hardcoded ``"2y"`` window."""
    captured = {}

    class _FakeYf:
        @staticmethod
        def download(ticker, period=None, interval=None, progress=None):
            captured["period"] = period
            idx = pd.bdate_range("2023-01-02", periods=40)
            return pd.DataFrame(
                {
                    "Open": [1.0] * 40,
                    "High": [1.0] * 40,
                    "Low": [1.0] * 40,
                    "Close": [1.0] * 40,
                    "Volume": [1.0] * 40,
                },
                index=idx,
            )

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYf)

    df = prices._fetch_yfinance("FAKE")

    assert captured["period"] == "max"
    assert len(df) == 40


def test_get_price_history_raises_when_stooq_and_yfinance_both_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    # An HTML error page instead of CSV -- Stooq's "no data" / blocked shape.
    monkeypatch.setattr(prices.requests, "get", lambda *a, **k: _FakeResponse("<html>error</html>"))
    # Force `import yfinance` inside _fetch_yfinance to raise ImportError:
    # setting a sys.modules entry to None makes any subsequent `import` of
    # that name raise ImportError immediately, without needing the package
    # to be absent from the environment.
    monkeypatch.setitem(sys.modules, "yfinance", None)

    with pytest.raises(prices.PriceDataError):
        prices.get_price_history("FAKE", no_cache=True)


def test_latest_price_returns_last_close_and_its_date():
    idx = pd.bdate_range("2023-01-02", periods=5)
    df = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0, 14.5]}, index=idx)

    price, as_of = prices.latest_price(df)

    assert price == 14.5
    assert as_of == idx[-1].strftime("%Y-%m-%d")
