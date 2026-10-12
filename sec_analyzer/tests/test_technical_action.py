"""Tests for sec_analyzer.technical.action.compute_technical_action -- the
single buy/hold/sell technical read. Pure and network-free."""

from sec_analyzer.technical.action import compute_technical_action


def _momentum(direction="up", score=70, accel="steady", label="UPWARD MOMENTUM"):
    return {"direction": direction, "score": score, "accel": accel, "label": label}


def _uptrend(**overrides):
    """Healthy uptrend: above both SMAs, modest extension, no exhaustion."""
    ind = {
        "price": 110.0, "sma50": 105.0, "sma200": 95.0, "atr14": 2.5,
        "dist_sma50_pct": 4.8, "rsi14": 60.0, "rsi_divergence": None,
        "obv_trend": "up", "volume_climax": None, "macd_hist": 0.4,
        "nearest_support": 104.0, "nearest_resistance": 120.0,
        "momentum": _momentum(),
    }
    ind.update(overrides)
    return ind


def _downtrend(**overrides):
    ind = {
        "price": 80.0, "sma50": 90.0, "sma200": 100.0, "atr14": 2.0,
        "rsi14": 35.0, "rsi_reclaim": None, "macd_cross": None, "macd_hist": -0.5,
        "obv_trend": "down", "rel_volume": 1.0, "volume_climax": None, "rsi_divergence": None,
        "nearest_support": 76.0, "nearest_resistance": 85.0,
        "momentum": _momentum("down", 25, "steady", "DOWNWARD MOMENTUM"),
    }
    ind.update(overrides)
    return ind


def test_healthy_uptrend_is_buy_with_room():
    a = compute_technical_action(_uptrend())
    assert a["action"] == "BUY"
    assert a["tone"] == "green"
    assert a["regime"] == "up"
    assert a["fuel"]["level"] == "strong"
    assert a["fuel"]["fired"] == 0
    assert a["reversal"] is None
    assert a["for_new"] and a["for_holder"]


def test_stretched_uptrend_waits_for_pullback():
    # 7x ATR above SMA50, RSI 78, bearish divergence -> 3 exhaustion signs.
    a = compute_technical_action(_uptrend(price=122.5, rsi14=78.0, rsi_divergence="bearish"))
    assert a["action"] == "WAIT FOR PULLBACK"
    assert a["fuel"]["level"] == "low"
    assert a["fuel"]["fired"] >= 2
    fired = {c["key"] for c in a["fuel"]["checks"] if c["fired"]}
    assert {"extended", "rsi_overbought", "bearish_divergence"} <= fired


def test_single_exhaustion_sign_still_buy_moderate_fuel():
    a = compute_technical_action(_uptrend(rsi14=74.0))
    assert a["action"] == "BUY"
    assert a["fuel"]["level"] == "moderate"


def test_uptrend_with_fading_momentum_is_hold():
    a = compute_technical_action(_uptrend(momentum=_momentum("flat", 52, "slowing", "FLAT MOMENTUM")))
    assert a["action"] == "HOLD"


def test_downtrend_without_confirmation_is_avoid():
    a = compute_technical_action(_downtrend())
    assert a["action"] == "AVOID"
    assert a["tone"] == "red"
    assert a["reversal"]["confirmed"] == 0
    assert a["trigger"] == 85.0


def test_downtrend_early_signal_is_wait():
    a = compute_technical_action(_downtrend(macd_cross="bullish"))
    assert a["action"] == "WAIT"
    assert a["reversal"]["confirmed"] == 1


def test_downtrend_confirmed_reversal_is_buy():
    a = compute_technical_action(_downtrend(macd_cross="bullish", rsi_reclaim="bullish", obv_trend="up"))
    assert a["action"] == "BUY"
    assert a["reversal"]["confirmed"] >= a["reversal"]["needed"]


def test_volume_only_confirmations_are_not_a_buy():
    # High volume + capitulation = exhaustion, but no RSI/MACD/divergence turn.
    ind = _downtrend(rel_volume=2.2, volume_climax={"detected": True, "direction": "down", "bars_ago": 2})
    a = compute_technical_action(ind)
    assert a["reversal"]["confirmed"] >= a["reversal"]["needed"]
    assert a["reversal"]["price_turn"] is False
    assert a["action"] == "WAIT"
    assert "hasn't turned" in a["headline"]


def test_horizon_changes_required_confirmations():
    ind = _downtrend(macd_cross="bullish")
    assert compute_technical_action(ind, "3m")["action"] == "BUY"      # 1 needed
    assert compute_technical_action(ind, "1y")["action"] == "WAIT"     # 2 needed
    five = compute_technical_action(ind, "5y")
    assert five["reversal"]["needed"] == 3
    assert five["horizon_note"]


def test_uptrend_breakdown_is_reduce():
    ind = _downtrend(price=97.0, sma50=100.0, sma200=90.0, macd_hist=-0.3)
    a = compute_technical_action(ind)
    assert a["regime"] == "transition"
    assert a["action"] == "REDUCE"


def test_pullback_without_bounce_is_wait():
    ind = _downtrend(price=97.0, sma50=100.0, sma200=90.0, macd_hist=0.1,
                     momentum=_momentum("flat", 50, "steady", "FLAT MOMENTUM"))
    a = compute_technical_action(ind)
    assert a["action"] == "WAIT"
    assert a["trigger"] == 100.0


def test_recovery_attempt_is_wait_with_sma200_trigger():
    ind = _downtrend(price=95.0, sma50=90.0, sma200=100.0)
    a = compute_technical_action(ind)
    assert a["action"] == "WAIT"
    assert a["trigger"] == 100.0


def test_missing_data_returns_none_and_never_raises():
    assert compute_technical_action(None) is None
    assert compute_technical_action({}) is None
    assert compute_technical_action({"price": 10.0}) is None
    # Garbage values must not raise.
    assert compute_technical_action({"price": 10.0, "sma50": "x", "sma200": None}) is None


def test_sparse_but_usable_input():
    a = compute_technical_action({"price": 10.0, "sma50": 9.0})
    assert a is not None
    assert a["regime"] == "transition"
    assert "None" not in a["for_new"] and "None" not in a["for_holder"]


def test_display_label_describes_state_and_carries_evidence_note():
    cases = [
        (_uptrend(), "UPTREND — ROOM TO RUN", "green"),
        (_uptrend(price=122.5, rsi14=78.0, rsi_divergence="bearish"), "UPTREND — STRETCHED", "amber"),
        (_downtrend(), "DOWNTREND — NO TURN", "red"),
        # A turn inside a downtrend is not painted green.
        (_downtrend(macd_cross="bullish", rsi_reclaim="bullish", obv_trend="up"), "DOWNTREND — TURN CONFIRMING", "amber"),
    ]
    for ind, label, tone in cases:
        a = compute_technical_action(ind)
        assert a["label"] == label
        assert a["tone"] == tone
        assert "no predictive edge" in a["evidence_note"]
        # Display text never tells the reader to buy or sell.
        for text in (a["label"], a["headline"], a["for_new"], a["for_holder"]):
            assert not any(w in text.lower() for w in ("buy", "sell", "stay out", "don't chase"))


def test_deterministic():
    ind = _uptrend(rsi14=74.0)
    assert compute_technical_action(ind) == compute_technical_action(ind)
