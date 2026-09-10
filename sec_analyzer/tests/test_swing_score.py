"""Unit tests for ``sec_analyzer.technical.swing.compute_swing_score`` and its
private helpers (``_classify_setup``, ``_badges``, ``_trade_levels``).

Binding contract: ``sec_analyzer/screener/SWING_SPEC.md`` -- Sec.3 for the
scoring math, Sec.9 for the specific invariants this module targets. All
inputs here are small, hand-built indicator dicts (no price history, no
network) -- exactly the flat dict shape ``compute_indicators`` +
``relative_strength`` would produce, per Sec.2.

Private helpers are imported and tested directly, mirroring
``test_technical.py``'s convention of unit-testing internals like ``_macd``,
``_rsi_divergence``, and ``_cluster_levels`` in isolation.
"""

import json

import pytest

from sec_analyzer.technical.swing import (
    _badges,
    _classify_setup,
    _trade_levels,
    compute_swing_score,
)


def _assert_json_native(value):
    """Recursively assert every scalar in ``value`` is an exact native Python
    type (str/int/float/bool/None) -- NOT a numpy scalar subclass, which
    would break ``json.dumps`` for the web API / store layer (SWING_SPEC.md
    Sec.3.12, invariant 5). Uses ``type(x) is ...`` (not ``isinstance``)
    because e.g. ``numpy.float64`` subclasses ``float`` and would slip past
    an ``isinstance`` check."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert type(k) is str
            _assert_json_native(v)
    elif isinstance(value, list):
        for v in value:
            _assert_json_native(v)
    else:
        assert type(value) in (int, float, str, bool, type(None)), (
            f"non-native type leaked into swing-score output: {value!r} ({type(value)})"
        )


# ---------------------------------------------------------------------------
# Invariant 1: None-safety
# ---------------------------------------------------------------------------


def test_compute_swing_score_none_input_returns_none():
    assert compute_swing_score(None) is None


def test_compute_swing_score_empty_dict_returns_none():
    assert compute_swing_score({}) is None


def test_compute_swing_score_non_dict_input_returns_none():
    # Not a single component sub-score can be computed from a non-dict input
    # -- the function must reject it outright rather than raising.
    assert compute_swing_score("AAPL") is None
    assert compute_swing_score(123) is None
    assert compute_swing_score([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# Invariant 11 / Sec.3.0: coverage floor
# ---------------------------------------------------------------------------


def test_coverage_floor_squeeze_only_dict_returns_none():
    """A dict carrying only a Bollinger squeeze (no price, no trend inputs
    at all) must be rejected: ``trend`` is not computable, so the coverage
    floor's ``"trend" not in raw`` condition fires regardless of weight."""
    ind = {"bb_squeeze": {"active": True, "percentile": 2.0}}
    assert compute_swing_score(ind) is None


def test_coverage_floor_trend_and_setup_computable_but_no_price_returns_none():
    """trend, setup, trigger, rel_strength and volume are ALL computable
    here (summed raw weight 0.25+0.25+0.20+0.15+0.10 = 0.95, comfortably
    above the 0.60 floor) -- the only thing missing is ``price`` itself, so
    this isolates the price-specific coverage-floor condition from the
    weight-sum condition."""
    ind = {
        "sma50_slope_pct": 5.0,
        "sma200_slope_pct": 3.0,
        "sma50_above_sma200": True,
        "dist_sma50_pct": 0.0,
        "bb_squeeze": {"active": True},
        "dist_52w_high_pct": -5.0,
        "rsi_reclaim": "bullish",
        "macd_cross": "bullish",
        "relative_strength": {"rs_3m_pct": 10.0, "rs_1m_pct": 5.0},
        "updown_volume_ratio": 2.0,
        "obv_trend": "up",
        # deliberately no "price" key
    }
    assert compute_swing_score(ind) is None


def test_coverage_floor_weight_just_under_returns_none_just_over_scores():
    """trend (0.25) + setup (0.25) alone sum to 0.50 raw weight -- strictly
    below the 0.60 floor -- so this dict is rejected even though `price`,
    `trend` and `setup` are all individually fine. Adding one more
    computable component (`volume`, weight 0.10, via `obv_trend`) pushes the
    summed weight to exactly 0.60, which the spec's `>= 0.60` floor accepts
    -- an otherwise-identical dict then scores instead of returning None."""
    under = {
        "price": 100.0,
        "sma50_slope_pct": 5.0,   # trend: computable (trend weight 0.25)
        "dist_sma50_pct": 0.0,    # feeds both trend and setup
        "bb_squeeze": {"active": True},  # setup: computable (setup weight 0.25)
    }
    assert compute_swing_score(under) is None

    over = dict(under)
    over["obv_trend"] = "up"  # adds the volume component (weight 0.10) -> total 0.60
    result = compute_swing_score(over)
    assert result is not None
    assert isinstance(result["score"], int)


# ---------------------------------------------------------------------------
# Invariant 2: monotonicity (raising one component's input never lowers score)
# ---------------------------------------------------------------------------


def _monotonic_base(**overrides):
    """A dict with all six components computable, values chosen to sit in
    the middle of each sub-score's linear (non-clamped) range so a small
    perturbation to one field visibly moves that component's sub-score
    without hitting a +-1 ceiling/floor."""
    base = {
        "price": 100.0,
        "sma50_slope_pct": 0.0,
        "sma200_slope_pct": 0.0,
        "sma50_above_sma200": True,
        "dist_sma50_pct": 0.0,
        "bb_squeeze": {"active": False, "percentile": 50.0},
        "dist_52w_high_pct": -20.0,
        "rsi_reclaim": None,
        "macd_cross": None,
        "macd_hist": 0.0,
        "rsi_divergence": None,
        "relative_strength": {"rs_3m_pct": 0.0, "rs_1m_pct": 0.0},
        "updown_volume_ratio": 1.0,
        "obv_trend": "flat",
        "nearest_support": 90.0,
        "nearest_resistance": 110.0,
        "atr14": 2.0,
    }
    base.update(overrides)
    return base


def test_monotonic_trend_component_sma200_slope():
    lower = compute_swing_score(_monotonic_base(sma200_slope_pct=0.0))
    higher = compute_swing_score(_monotonic_base(sma200_slope_pct=2.0))
    assert lower is not None and higher is not None
    assert higher["score"] >= lower["score"]


def test_monotonic_setup_component_dist_52w_high():
    lower = compute_swing_score(_monotonic_base(dist_52w_high_pct=-20.0))
    higher = compute_swing_score(_monotonic_base(dist_52w_high_pct=-5.0))
    assert lower is not None and higher is not None
    assert higher["score"] >= lower["score"]


def test_monotonic_rel_strength_component_rs_3m():
    lower = compute_swing_score(
        _monotonic_base(relative_strength={"rs_3m_pct": 0.0, "rs_1m_pct": 0.0})
    )
    higher = compute_swing_score(
        _monotonic_base(relative_strength={"rs_3m_pct": 10.0, "rs_1m_pct": 0.0})
    )
    assert lower is not None and higher is not None
    assert higher["score"] >= lower["score"]


# ---------------------------------------------------------------------------
# Invariant 3: hand-verified band cases
# ---------------------------------------------------------------------------


def _textbook_pullback():
    return {
        "price": 100.0,
        "sma50_slope_pct": 5.0,
        "sma200_slope_pct": 3.0,
        "sma50_above_sma200": True,
        "dist_sma50_pct": 0.0,
        "bb_squeeze": {"active": True, "percentile": 5.0},
        "dist_52w_high_pct": -5.0,
        "rsi_reclaim": "bullish",
        "macd_cross": "bullish",
        "relative_strength": {"rs_3m_pct": 10.0, "rs_1m_pct": 5.0},
        "updown_volume_ratio": 2.0,
        "obv_trend": "up",
        "nearest_support": 95.0,
        "nearest_resistance": 115.0,
        "atr14": 2.0,
    }


def test_textbook_pullback_scores_at_least_70():
    # Hand derivation (SWING_SPEC.md Sec.3.1-3.7), all 6 components
    # computable so total_w = 1.00 (no renormalization needed):
    #
    # trend  = avg(clamp(5/5), clamp(3/3), +1 [above], clamp(0/10))
    #        = avg(1, 1, 1, 0) = 0.75
    # setup  = avg(ext(0)=+1 [in buy zone -6..2],
    #              squeeze(active=True)=+1,
    #              high_prox(d=-5) = clamp(1 + (-5)/15) = 2/3)
    #        = (1 + 1 + 2/3) / 3 = 8/9 = 0.888889
    # trigger = avg(rsi_reclaim=bullish -> +1, macd_cross=bullish -> +1) = 1.0
    #   (rsi_divergence absent -> skipped)
    # rel_strength = avg(clamp(10/20), clamp(5/10)) = avg(0.5, 0.5) = 0.5
    # volume = avg(clamp(log(2)/log(2))=1, obv_trend=up -> +1) = 1.0
    # risk_reward: risk=100-95=5, reward=115-100=15, rr=3
    #   -> clamp(log(3)/log(3)) = 1.0
    #
    # s = .25*.75 + .25*(8/9) + .20*1 + .15*.5 + .10*1 + .05*1
    #   = .1875 + .222222 + .2 + .075 + .1 + .05 = 0.834722...
    # score = round(50 + 50*0.834722) = round(91.7361) = 92
    result = compute_swing_score(_textbook_pullback())

    assert result is not None
    assert result["score"] == 92
    assert result["score"] >= 70
    assert result["s"] == pytest.approx(0.835, abs=0.001)
    assert result["label"] == "GÜÇLÜ FIRSAT"
    assert result["setup"] == "TRENDDE GERİ ÇEKİLME"
    assert result["badges"] == ["SIKIŞMA"]
    assert [c["key"] for c in result["components"]] == [
        "trend", "setup", "trigger", "rel_strength", "volume", "risk_reward",
    ]
    # Everything positive -> a "strongest" driver is named, no "weakest".
    assert "en güçlü:" in result["summary"]
    assert "en zayıf" not in result["summary"]


def _extended_deteriorating_downtrend():
    return {
        "price": 100.0,
        "sma50_slope_pct": -5.0,
        "sma200_slope_pct": -3.0,
        "sma50_above_sma200": False,
        "dist_sma50_pct": -10.0,
        "bb_squeeze": {"active": False, "percentile": 90.0},
        "dist_52w_high_pct": -40.0,
        "rsi_reclaim": "bearish",
        "macd_cross": "bearish",
        "rsi_divergence": "bearish",
        "relative_strength": {"rs_3m_pct": -20.0, "rs_1m_pct": -10.0},
        "updown_volume_ratio": 0.5,
        "obv_trend": "down",
        "nearest_support": 90.0,
        "nearest_resistance": 103.0,
        "atr14": 2.0,
    }


def test_extended_deteriorating_downtrend_scores_at_most_30():
    # Hand derivation:
    #
    # trend = avg(clamp(-5/5), clamp(-3/3), -1 [below], clamp(-10/10))
    #        = avg(-1, -1, -1, -1) = -1.0
    # setup: ext(-10) is in the "broken down" piece [-30,-6):
    #   span = -6 - (-30) = 24; value = -1 + (-10-(-30))*(2/24)
    #        = -1 + 20*(1/12) = -1 + 1.66667 = 0.66667  (not yet fully broken)
    #   squeeze(inactive, percentile=90) = clamp((50-90)/50) = clamp(-0.8) = -0.8
    #   high_prox(d=-40, beyond the -35 floor) = -1.0
    #   setup = (0.66667 - 0.8 - 1.0) / 3 = -1.13333/3 = -0.37778
    #
    # NOTE: to make this genuinely "extended" (badly broken down, not just a
    # mild pullback) the extension leg alone isn't very negative at ext=-10;
    # the setup sub-score is still clearly negative once squeeze + 52w-high
    # distance are folded in, and every OTHER component below is at its
    # full -1.0 floor, which is what drives the final score to near 0.
    #
    # trigger = avg(rsi_reclaim=bearish -> -1, macd_cross=bearish -> -1,
    #               rsi_divergence=bearish -> -1) = -1.0
    # rel_strength = avg(clamp(-20/20), clamp(-10/10)) = avg(-1, -1) = -1.0
    # volume = avg(clamp(log(0.5)/log(2))=-1, obv_trend=down -> -1) = -1.0
    # risk_reward: risk=100-90=10, reward=103-100=3, rr=0.3
    #   -> clamp(log(0.3)/log(3)) = clamp(-1.0959) = -1.0
    #
    # s = .25*(-1.0) + .25*(-0.37778) + .20*(-1.0) + .15*(-1.0)
    #     + .10*(-1.0) + .05*(-1.0)
    #   = -0.25 - 0.094444 - 0.2 - 0.15 - 0.1 - 0.05 = -0.844444
    # score = round(50 + 50*(-0.844444)) = round(50 - 42.2222) = round(7.778) = 8
    result = compute_swing_score(_extended_deteriorating_downtrend())

    assert result is not None
    assert result["score"] == 8
    assert result["score"] <= 30
    assert result["label"] == "KAÇIN"
    assert result["setup"] == "KURULUM YOK"
    assert result["badges"] == []
    # Everything negative -> a "weakest" driver is named, no "strongest".
    assert "en zayıf" in result["summary"]
    assert "en güçlü:" not in result["summary"]


# ---------------------------------------------------------------------------
# Invariant 4: determinism
# ---------------------------------------------------------------------------


def test_determinism_same_input_same_output():
    ind = _textbook_pullback()
    first = compute_swing_score(ind)
    second = compute_swing_score(ind)
    assert first == second
    # And calling it doesn't mutate the input dict for next time either.
    third = compute_swing_score(_textbook_pullback())
    assert first == third


# ---------------------------------------------------------------------------
# Invariant 5: JSON-serializability, no numpy leakage
# ---------------------------------------------------------------------------


def test_result_is_json_round_trippable_and_has_no_numpy_scalars():
    for ind in (_textbook_pullback(), _extended_deteriorating_downtrend()):
        result = compute_swing_score(ind)
        _assert_json_native(result)
        round_tripped = json.loads(json.dumps(result, ensure_ascii=False))
        assert round_tripped == result


# ---------------------------------------------------------------------------
# Sec.3.8: setup classification, first-match-wins, one test per branch
# ---------------------------------------------------------------------------


def test_classify_setup_breakout():
    ind = {"dist_52w_high_pct": -1.0}
    assert _classify_setup(ind, trend_sub=0.5) == "BREAKOUT"


def test_classify_setup_trend_pullback():
    ind = {"dist_52w_high_pct": -10.0, "dist_sma50_pct": 0.0}
    assert _classify_setup(ind, trend_sub=0.5) == "TRENDDE GERİ ÇEKİLME"


def test_classify_setup_squeeze():
    ind = {
        "dist_52w_high_pct": -10.0,   # fails BREAKOUT
        "dist_sma50_pct": 10.0,       # fails pullback range
        "bb_squeeze": {"active": True},
    }
    assert _classify_setup(ind, trend_sub=0.1) == "SIKIŞMA"


def test_classify_setup_oversold_reaction_via_rsi_reclaim():
    ind = {
        "dist_52w_high_pct": -10.0,
        "dist_sma50_pct": 10.0,
        "bb_squeeze": {"active": False},
        "rsi_reclaim": "bullish",
    }
    assert _classify_setup(ind, trend_sub=-0.5) == "AŞIRI SATIM TEPKİSİ"


def test_classify_setup_oversold_reaction_via_rsi14_and_divergence():
    ind = {
        "dist_52w_high_pct": -10.0,
        "dist_sma50_pct": 10.0,
        "bb_squeeze": {"active": False},
        "rsi_reclaim": None,
        "rsi14": 35.0,
        "rsi_divergence": "bullish",
    }
    assert _classify_setup(ind, trend_sub=-0.5) == "AŞIRI SATIM TEPKİSİ"


def test_classify_setup_momentum_continuation():
    ind = {
        "dist_52w_high_pct": -10.0,
        "dist_sma50_pct": 10.0,
        "bb_squeeze": {"active": False},
        "rsi_reclaim": None,
        "rsi14": None,
        "rsi_divergence": None,
    }
    assert _classify_setup(ind, trend_sub=0.5) == "MOMENTUM DEVAM"


def test_classify_setup_no_setup_fallback():
    ind = {
        "dist_52w_high_pct": -10.0,
        "dist_sma50_pct": 10.0,
        "bb_squeeze": {"active": False},
        "rsi_reclaim": None,
        "rsi14": None,
        "rsi_divergence": None,
    }
    assert _classify_setup(ind, trend_sub=0.1) == "KURULUM YOK"


def test_classify_setup_first_match_wins_breakout_beats_pullback():
    # Both the BREAKOUT and TRENDDE GERİ ÇEKİLME conditions hold here --
    # BREAKOUT must win because it's evaluated first.
    ind = {"dist_52w_high_pct": -1.0, "dist_sma50_pct": 0.0}
    assert _classify_setup(ind, trend_sub=0.5) == "BREAKOUT"


def test_classify_setup_none_fields_and_none_trend_sub_fail_safely():
    # A totally empty indicators dict and a None trend_sub (treated as 0.0,
    # per spec) must fall through every rule without raising.
    assert _classify_setup({}, trend_sub=None) == "KURULUM YOK"


# ---------------------------------------------------------------------------
# Sec.3.9: badges -- fixed emission order, per-condition boundaries
# ---------------------------------------------------------------------------


def test_badges_all_conditions_emit_in_fixed_order():
    ind = {
        "rel_volume": 2.0,
        "volume_climax": {"detected": True, "direction": "down", "bars_ago": 3},
        "bb_squeeze": {"active": True},
        "golden_cross": True,
        "rsi_divergence": "bullish",
    }
    assert _badges(ind) == ["HACİM", "KAPİTÜLASYON", "SIKIŞMA", "GOLDEN CROSS", "RSI UYUMSUZLUK"]


def test_badges_empty_when_nothing_qualifies():
    assert _badges({}) == []


def test_badge_hacim_threshold_boundary():
    assert _badges({"rel_volume": 1.5}) == ["HACİM"]
    assert _badges({"rel_volume": 1.49}) == []


def test_badge_kapitulasyon_requires_down_direction_and_recency():
    assert _badges({"volume_climax": {"detected": True, "direction": "down", "bars_ago": 5}}) == ["KAPİTÜLASYON"]
    # One bar too late.
    assert _badges({"volume_climax": {"detected": True, "direction": "down", "bars_ago": 6}}) == []
    # Wrong direction.
    assert _badges({"volume_climax": {"detected": True, "direction": "up", "bars_ago": 1}}) == []
    # Not actually detected.
    assert _badges({"volume_climax": {"detected": False, "direction": "down", "bars_ago": 1}}) == []


def test_badge_golden_cross_and_rsi_divergence_require_exact_values():
    assert _badges({"golden_cross": True}) == ["GOLDEN CROSS"]
    assert _badges({"golden_cross": False}) == []
    assert _badges({"rsi_divergence": "bullish"}) == ["RSI UYUMSUZLUK"]
    assert _badges({"rsi_divergence": "bearish"}) == []


# ---------------------------------------------------------------------------
# Sec.3.11: trade levels
# ---------------------------------------------------------------------------


def test_trade_levels_none_when_price_missing():
    assert _trade_levels({}) == {
        "entry": None, "stop": None, "stop_pct": None,
        "target": None, "target_pct": None, "rr": None,
    }


def test_trade_levels_stop_support_clamp_closer_than_1atr_falls_back_to_atr_stop():
    # base_stop = 100 - 2*10 = 80; support_stop = 95*0.99 = 94.05.
    # max(80, 94.05) = 94.05 -> the support clamp wins the raw selection...
    # ...but the volatility floor (SWING_SPEC.md Sec.3.11) then kicks in:
    # price - 94.05 = 5.95, which is < 1.0*atr (10) -- a sub-ATR stop is
    # noise-range, so it's discarded in favour of base_stop = 80.
    # (Previously, before the floor existed, this test asserted the support
    # clamp of 94.05 stood; that expectation is now spec-incorrect and has
    # been updated to the floored value per the amended Sec.3.11.)
    levels = _trade_levels({"price": 100.0, "atr14": 10.0, "nearest_support": 95.0})
    assert levels["entry"] == 100.0
    assert levels["stop"] == 80.0
    assert levels["stop_pct"] == -20.0
    # No resistance given -> target falls back to price + 3*atr = 130.0.
    assert levels["target"] == 130.0
    assert levels["target_pct"] == 30.0
    # rr = (130-100) / (100-80) = 30 / 20 = 1.5
    assert levels["rr"] == pytest.approx(1.5, abs=1e-9)


def test_trade_levels_stop_support_clamp_exactly_1atr_is_kept():
    # Boundary case: the floor only fires on strict "<", so a support stop
    # exactly 1.0*atr away from price is kept, not floored.
    # base_stop = 100 - 2*5 = 90.
    # support = 95.0/0.99 (chosen so support*0.99 recovers exactly 95.0).
    # support_stop = 95.0 -> max(90, 95.0) = 95.0 (support wins the raw pick).
    # gap = 100 - 95.0 = 5.0 == 1.0*atr (5.0) -> not "< atr", so the floor
    # does NOT replace it; the support-based stop of 95.0 stands.
    support = 95.0 / 0.99
    levels = _trade_levels({"price": 100.0, "atr14": 5.0, "nearest_support": support})
    assert levels["stop"] == 95.0
    assert levels["stop_pct"] == -5.0
    assert levels["target"] == 115.0   # 100 + 3*5
    assert levels["target_pct"] == 15.0
    assert levels["rr"] == pytest.approx(3.0, abs=1e-9)   # (115-100)/(100-95) = 15/5


def test_trade_levels_stop_support_clamp_comfortably_beyond_1atr_is_kept():
    # base_stop = 100 - 2*4 = 92.
    # support = 94.0/0.99 (chosen so support*0.99 recovers exactly 94.0).
    # support_stop = 94.0 -> max(92, 94.0) = 94.0 (support wins the raw pick).
    # gap = 100 - 94.0 = 6.0, comfortably >= 1.0*atr (4.0) -> floor does not
    # fire; the support-based stop of 94.0 stands.
    support = 94.0 / 0.99
    levels = _trade_levels({"price": 100.0, "atr14": 4.0, "nearest_support": support})
    assert levels["stop"] == 94.0
    assert levels["stop_pct"] == -6.0
    assert levels["target"] == 112.0   # 100 + 3*4
    assert levels["target_pct"] == 12.0
    assert levels["rr"] == pytest.approx(2.0, abs=1e-9)   # (112-100)/(100-94) = 12/6


def test_trade_levels_stop_no_atr_very_close_support_stands_unfloored():
    # atr14 missing -> base_stop is None, so there's nothing to floor
    # against (Sec.3.11: "with atr14 unavailable ... the structural stop
    # stands"), even though the support is very close to price.
    # support_stop = 99 * 0.99 = 98.01 -> stop = 98.01 (no base_stop to max
    # against), kept as-is.
    levels = _trade_levels({"price": 100.0, "nearest_support": 99.0})
    assert levels["entry"] == 100.0
    assert levels["stop"] == 98.01
    assert levels["stop_pct"] == -2.0
    # No atr and no resistance -> target can't be computed at all.
    assert levels["target"] is None
    assert levels["target_pct"] is None
    assert levels["rr"] is None


def test_trade_levels_stop_picks_atr_stop_over_distant_support_clamp():
    # base_stop = 100 - 2*2 = 96; support_stop = 80*0.99 = 79.2.
    # max(96, 79.2) = 96 -> the ATR-based stop wins (support is too far below).
    levels = _trade_levels({"price": 100.0, "atr14": 2.0, "nearest_support": 80.0})
    assert levels["stop"] == 96.0
    assert levels["stop_pct"] == -4.0
    assert levels["target"] == 106.0   # 100 + 3*2
    assert levels["target_pct"] == 6.0
    assert levels["rr"] == 1.5         # (106-100)/(100-96) = 6/4 = 1.5


def test_trade_levels_target_uses_resistance_when_far_enough_above_price():
    # resistance 110 >= price*1.02 (102) -> target = resistance.
    levels = _trade_levels({"price": 100.0, "atr14": 5.0, "nearest_resistance": 110.0})
    assert levels["target"] == 110.0
    assert levels["stop"] == 90.0   # no support given -> base_stop = 100-10
    assert levels["rr"] == 1.0      # (110-100)/(100-90) = 10/10


def test_trade_levels_target_falls_back_to_atr_when_resistance_too_close():
    # resistance 101 < price*1.02 (102) -> too close to count, fall back to
    # price + 3*atr = 100 + 15 = 115.
    levels = _trade_levels({"price": 100.0, "atr14": 5.0, "nearest_resistance": 101.0})
    assert levels["target"] == 115.0
    assert levels["rr"] == 1.5   # (115-100)/(100-90) = 15/10


def test_trade_levels_invalid_stop_at_or_below_zero_yields_none():
    # base_stop = 5 - 2*10 = -15 (<= 0) -> stop must be rejected outright.
    levels = _trade_levels({"price": 5.0, "atr14": 10.0})
    assert levels["entry"] == 5.0
    assert levels["stop"] is None
    assert levels["stop_pct"] is None
    assert levels["rr"] is None   # rr needs a valid stop too
    # target is computed independently of the (invalid) stop.
    assert levels["target"] == 35.0   # 5 + 3*10
