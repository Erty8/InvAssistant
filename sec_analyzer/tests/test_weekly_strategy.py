"""Tests for sec_analyzer.technical.weekly_strategy (WEEKLY_STRATEGY_SPEC.md)."""

import numpy as np
import pandas as pd
import pytest

from sec_analyzer.technical import weekly_strategy as ws


def _frame_from_closes(closes, start="2024-01-01", spread=0.01):
    """Business-day OHLC frame whose open/high/low sit around each close."""
    idx = pd.bdate_range(start=start, periods=len(closes))
    c = pd.Series(np.asarray(closes, dtype=float), index=idx)
    return pd.DataFrame(
        {"Open": c.shift(1).fillna(c), "High": c * (1 + spread), "Low": c * (1 - spread), "Close": c},
        index=idx,
    )


def _trim_to_weekday(df, weekday):
    """Drop trailing bars until the last bar falls on ``weekday`` (0=Mon)."""
    while df.index[-1].weekday() != weekday:
        df = df.iloc[:-1]
    return df


def test_returns_none_on_short_history():
    df = _frame_from_closes(np.linspace(100, 120, 150))
    assert ws.compute_weekly_strategy(df) is None
    assert ws.compute_weekly_strategy(None) is None


def test_never_raises_on_bad_input():
    df = pd.DataFrame({"Close": [1.0] * 300})  # missing OHLC columns
    assert ws.compute_weekly_strategy(df) is None


def test_tier_stop_uses_sma50_only_in_a_full_uptrend():
    up = {"close": 120.0, "sma50": 110.0, "sma200": 100.0}
    assert ws._tier_stop(up) == pytest.approx(110.0 * 0.95)
    below_sma50 = {"close": 105.0, "sma50": 110.0, "sma200": 100.0}
    assert ws._tier_stop(below_sma50) == pytest.approx(100.0 * 0.95)
    sma50_under_sma200 = {"close": 120.0, "sma50": 95.0, "sma200": 100.0}
    assert ws._tier_stop(sma50_under_sma200) == pytest.approx(100.0 * 0.95)


def test_x2_day1_needs_zone_and_approach():
    base = {"sma50": 100.0, "sma200": 80.0}
    assert ws._x2_day1({**base, "close": 96.0, "high": 99.0})
    # High never came within 2% of SMA50.
    assert not ws._x2_day1({**base, "close": 96.0, "high": 97.0})
    # Close closed above SMA50.
    assert not ws._x2_day1({**base, "close": 101.0, "high": 102.0})
    # Not more than 10% above SMA200.
    assert not ws._x2_day1({"sma50": 100.0, "sma200": 92.0, "close": 96.0, "high": 99.0})


def test_steady_decline_waits():
    df = _frame_from_closes(np.linspace(200, 80, 320))
    res = ws.compute_weekly_strategy(df)
    assert res["signal"] == ws.SIGNAL_WAIT
    assert res["entry_stop"] is None
    assert res["position"] is None
    assert res["above_sma200"] is False


def test_steady_uptrend_is_in_position_with_tiered_stop():
    df = _trim_to_weekday(_frame_from_closes(np.linspace(50, 200, 330)), 2)
    res = ws.compute_weekly_strategy(df)
    assert res["signal"] == ws.SIGNAL_IN_POSITION
    pos = res["position"]
    assert pos["kind"] == ws.KIND_ABOVE
    # In a clean uptrend the stop sits 5% under SMA50 and below the price.
    assert pos["stop"] < res["levels"]["close"]
    assert pos["next_stop"] >= pos["stop"]
    assert pos["next_stop"] == pytest.approx(res["levels"]["sma50"] * 0.95, rel=1e-3)


def test_first_close_above_sma200_mid_week_signals_next_open():
    closes = list(np.linspace(200, 100, 300))
    df = _frame_from_closes(closes)
    df = _trim_to_weekday(df, 2)  # last bar a Wednesday
    # Jump above SMA200 only on Tuesday and Wednesday of the last week.
    df.iloc[-2, df.columns.get_loc("Close")] = 250.0
    df.iloc[-1, df.columns.get_loc("Close")] = 255.0
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    res = ws.compute_weekly_strategy(df)
    assert res["signal"] == ws.SIGNAL_BUY_NEXT_OPEN
    assert res["position"] is None
    sma200 = res["levels"]["sma200"]
    assert res["entry_stop"] == pytest.approx(sma200 * 0.95, abs=0.02)
    assert res["entry_stop_pct"] < 0


def test_stop_exit_is_recorded_after_a_crash():
    up = list(np.linspace(50, 200, 300))
    crash = list(np.linspace(195, 100, 15))
    df = _frame_from_closes(up + crash)
    res = ws.compute_weekly_strategy(df, use_x2=False)
    assert res["recent_trades"], "a position opened in the uptrend should have been stopped out"
    last_trade = res["recent_trades"][-1]
    assert last_trade["exit_reason"] == ws.EXIT_STOP
    crash_start = df.index[300].date().isoformat()
    assert last_trade["exit_date"] >= crash_start
    # The fill is the stop or worse (a gap), never above the day's open.
    exit_bar = df.loc[last_trade["exit_date"]]
    assert last_trade["exit_price"] <= round(float(exit_bar["Open"]), 2) + 0.01
    assert res["position"] is None or res["position"]["entry_date"] > last_trade["exit_date"]


def test_two_friday_confirmation_below_sma200():
    # Long decline keeps SMA200 far above price; then a slow rebound that keeps
    # the close under SMA200 but lifts EMA8 above EMA21.
    decline = list(np.linspace(300, 100, 280))
    rebound = list(np.linspace(100, 125, 30))
    df = _trim_to_weekday(_frame_from_closes(decline + rebound), 2)
    res = ws.compute_weekly_strategy(df, use_x2=False)
    lv = res["levels"]
    assert lv["close"] < lv["sma200"] and lv["ema8"] > lv["ema21"]
    # Several Fridays of the rebound already met the condition -> the model bought
    # at a Friday close (rule 2) and holds a below-SMA200 position.
    assert res["signal"] == ws.SIGNAL_IN_POSITION
    pos = res["position"]
    assert pos["kind"] == ws.KIND_BELOW
    # Rule 2 buys at a Friday close.
    entry = pd.Timestamp(pos["entry_date"])
    assert entry.weekday() == 4
    assert pos["entry_price"] == pytest.approx(float(df.loc[entry, "Close"]), abs=0.01)


def _decline_then_rebound(rebound_days, last_weekday=2):
    """Decline long enough that SMA200 stays far above price, then a sharp
    rebound over the last ``rebound_days`` bars, ending on ``last_weekday``."""
    df = _trim_to_weekday(_frame_from_closes(np.linspace(300, 100, 330)), last_weekday)
    closes = df["Close"].to_numpy().copy()
    n = len(closes)
    closes[n - rebound_days:] = np.linspace(closes[n - rebound_days - 1] * 1.04, 130, rebound_days)
    df["Close"] = closes
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    return df


def test_first_friday_of_ema_uptrend_is_watch_only():
    df = _decline_then_rebound(3)  # Mon-Wed of the last week
    res = ws.compute_weekly_strategy(df)
    lv = res["levels"]
    assert lv["close"] < lv["sma200"] and lv["ema8"] > lv["ema21"] and lv["close"] > lv["ema8"]
    assert res["prev_week_condition"] is False
    assert res["signal"] == ws.SIGNAL_FIRST_FRIDAY
    assert res["entry_stop"] == pytest.approx(lv["ema21"] * 0.95, abs=0.02)


def test_second_friday_of_ema_uptrend_buys_at_the_close():
    df = _decline_then_rebound(8)  # previous Mon-Fri plus Mon-Wed of the last week
    res = ws.compute_weekly_strategy(df)
    assert res["prev_week_condition"] is True
    assert res["position"] is None
    assert res["signal"] == ws.SIGNAL_BUY_FRIDAY_CLOSE


def test_signal_is_deterministic():
    df = _frame_from_closes(np.concatenate([np.linspace(80, 160, 250), np.linspace(160, 140, 40)]))
    assert ws.compute_weekly_strategy(df) == ws.compute_weekly_strategy(df)


def test_cli_card_lines():
    from sec_analyzer.cli import _weekly_strategy_lines

    assert _weekly_strategy_lines({}) == []
    assert _weekly_strategy_lines({"weekly_strategy": None}) == []
    buy = {
        "signal": ws.SIGNAL_BUY_NEXT_OPEN,
        "label": ws.SIGNAL_LABELS[ws.SIGNAL_BUY_NEXT_OPEN],
        "entry_stop": 716.04,
        "entry_stop_pct": -10.0,
        "levels": {"sma50": 972.2},
        "x2": {"enabled": True, "day1": False},
    }
    lines = _weekly_strategy_lines({"weekly_strategy": buy})
    assert len(lines) == 1
    assert lines[0].startswith("Strategy:")
    assert "first stop $716.04 (-10.0%)" in lines[0]
    held = {
        "signal": ws.SIGNAL_IN_POSITION,
        "label": ws.SIGNAL_LABELS[ws.SIGNAL_IN_POSITION],
        "position": {"entry_date": "2026-08-10", "entry_price": 863.53, "stop": 709.29,
                     "stop_pct": -12.3, "next_stop": 709.29},
        "levels": {"sma50": 972.2},
        "x2": {"enabled": True, "day1": True},
    }
    lines = _weekly_strategy_lines({"weekly_strategy": held})
    assert "IN POSITION (model) since 2026-08-10" in lines[0]
    assert "stop $709.29 (-12.3%)" in lines[0]
    assert len(lines) == 2 and "X2 day 1" in lines[1]
    assert "None" not in " ".join(lines)


def test_output_is_json_serialisable():
    import json

    df = _frame_from_closes(np.linspace(50, 200, 330))
    json.dumps(ws.compute_weekly_strategy(df))
