"""Hand-verified unit tests for sec_analyzer.interpret.planning -- the four
pure, deterministic phase-2 post-processing functions (METODOLOJI.md Sec.1
items 4-7: scenario returns, entry plan, stop-adding signals, thesis metric).

Every numeric assertion below is derived by hand in a comment above the
assertion, following the methodology used in test_valuation_dcf.py /
test_valuation_engine.py. None of these functions touch the network or an
LLM, so no mocking is needed -- these are plain function-in, value-out tests.
"""

import copy

import pytest

from sec_analyzer.interpret import planning

# ---------------------------------------------------------------------------
# compute_scenario_returns
# ---------------------------------------------------------------------------


def _fvr(bear_lo=72, bear_hi=88, base_lo=90, base_hi=110, bull_lo=108, bull_hi=132):
    return {
        "bear": {"lo": bear_lo, "hi": bear_hi},
        "base": {"lo": base_lo, "hi": base_hi},
        "bull": {"lo": bull_lo, "hi": bull_hi},
    }


def test_compute_scenario_returns_hand_verified_values():
    # price = 100
    # bear: lo=72 -> (72/100-1)*100 = -28.0 ; hi=88 -> (88/100-1)*100 = -12.0
    # base: lo=90 -> -10.0 ; hi=110 -> 10.0
    # bull: lo=108 -> 8.0 ; hi=132 -> 32.0
    result = planning.compute_scenario_returns(_fvr(), 100)

    assert result == {
        "bear": {"ret_lo_pct": -28.0, "ret_hi_pct": -12.0},
        "base": {"ret_lo_pct": -10.0, "ret_hi_pct": 10.0},
        "bull": {"ret_lo_pct": 8.0, "ret_hi_pct": 32.0},
    }


def test_compute_scenario_returns_simple_round_number_case():
    # price=100, lo=90 -> -10.0 ; hi=130 -> +30.0 (the exact example from the spec)
    fvr = {"bear": {"lo": None, "hi": None}, "base": {"lo": 90, "hi": 130}, "bull": {"lo": None, "hi": None}}
    result = planning.compute_scenario_returns(fvr, 100)
    assert result["base"] == {"ret_lo_pct": -10.0, "ret_hi_pct": 30.0}


@pytest.mark.parametrize("price", [None, 0, -5])
def test_compute_scenario_returns_all_none_when_price_unusable(price):
    result = planning.compute_scenario_returns(_fvr(), price)
    for key in ("bear", "base", "bull"):
        assert result[key] == {"ret_lo_pct": None, "ret_hi_pct": None}


def test_compute_scenario_returns_all_none_when_fair_value_range_missing():
    result = planning.compute_scenario_returns(None, 100)
    assert set(result.keys()) == {"bear", "base", "bull"}
    for key in ("bear", "base", "bull"):
        assert result[key] == {"ret_lo_pct": None, "ret_hi_pct": None}


def test_compute_scenario_returns_all_three_keys_always_present_even_with_partial_bands():
    fvr = {"base": {"lo": 90, "hi": 110}}  # bear/bull entirely missing
    result = planning.compute_scenario_returns(fvr, 100)
    assert set(result.keys()) == {"bear", "base", "bull"}
    assert result["bear"] == {"ret_lo_pct": None, "ret_hi_pct": None}
    assert result["bull"] == {"ret_lo_pct": None, "ret_hi_pct": None}
    assert result["base"] == {"ret_lo_pct": -10.0, "ret_hi_pct": 10.0}


def test_compute_scenario_returns_does_not_mutate_input_dict():
    fvr = _fvr()
    original = copy.deepcopy(fvr)
    planning.compute_scenario_returns(fvr, 100)
    assert fvr == original


# ---------------------------------------------------------------------------
# compute_entry_plan
# ---------------------------------------------------------------------------


def _valuation_for_entry_plan(bear_lo=40, base_lo=60, base_hi=80, bull_hi=200):
    return {
        "fair_value_range": {
            "bear": {"lo": bear_lo},
            "base": {"lo": base_lo, "hi": base_hi},
            "bull": {"hi": bull_hi},
        }
    }


def test_compute_entry_plan_hand_verified_three_tranche_case():
    # valuation candidates: bear_lo=40, base_lo=60, base_hi=80, bull_hi=200
    # price=110. Candidates <= price: bear_lo=40, base_lo=60, base_hi=80
    #   (bull_hi=200 > 110 is excluded from tranche candidates, but is still
    #   the *target* anchor via _resolve_target, independent of price filter)
    # descending order, no two within 2% of each other -> kept = [80, 60, 40]
    # n = 3
    # target = bull.hi = 200 (bull.hi takes priority over base.hi)
    # invalidation sources = [bear_lo=40] (no technical/low_52w given) -> min=40
    #   invalidation = round(40 * (1 - 0.05), 2) = round(38.0, 2) = 38.0
    # weights = [1, 2, 3], sum = 6
    #   size_pct = [round(1/6*100,1), round(2/6*100,1), round(3/6*100,1)]
    #            = [16.7, 33.3, 50.0]  (sums to 100.0)
    # price zones (band = +/-1.5%, symmetric so entry == level exactly):
    #   level=80: lo=80*0.985=78.8, hi=80*1.015=81.2, entry=80.0
    #   level=60: lo=60*0.985=59.1, hi=60*1.015=60.9, entry=60.0
    #   level=40: lo=40*0.985=39.4, hi=40*1.015=40.6, entry=40.0
    # R:R = round((target*0.998 - entry*1.002) / (entry*1.002 - invalidation), 1)
    #   tranche1 entry=80: reward=200*0.998-80*1.002=199.6-80.16=119.44
    #                      risk=80.16-38=42.16 -> rr=round(119.44/42.16,1)=2.8
    #   tranche2 entry=60: reward=199.6-60*1.002=199.6-60.12=139.48
    #                      risk=60.12-38=22.12 -> rr=round(139.48/22.12,1)=6.3
    #   tranche3 entry=40: reward=199.6-40*1.002=199.6-40.08=159.52
    #                      risk=40.08-38=2.08 -> rr=round(159.52/2.08,1)=76.7
    valuation = _valuation_for_entry_plan()
    plan = planning.compute_entry_plan(valuation, None, 110)

    assert len(plan) == 3
    # descending price order
    levels = [t["price_zone"]["hi"] + t["price_zone"]["lo"] for t in plan]
    assert levels == sorted(levels, reverse=True)

    t1, t2, t3 = plan
    assert t1["n"] == 1 and t2["n"] == 2 and t3["n"] == 3

    assert t1["price_zone"] == {"lo": 78.8, "hi": 81.2}
    assert t2["price_zone"] == {"lo": 59.1, "hi": 60.9}
    assert t3["price_zone"] == {"lo": 39.4, "hi": 40.6}

    assert t1["size_pct"] == pytest.approx(16.7)
    assert t2["size_pct"] == pytest.approx(33.3)
    assert t3["size_pct"] == pytest.approx(50.0)
    assert sum(t["size_pct"] for t in plan) == pytest.approx(100.0)
    # ascending toward cheaper (last tranche gets the largest allocation)
    assert t1["size_pct"] < t2["size_pct"] < t3["size_pct"]

    # single shared invalidation and target across all tranches
    for t in plan:
        assert t["invalidation"] == pytest.approx(38.0)
        assert t["target"] == pytest.approx(200.0)

    assert t1["rr"] == pytest.approx(2.8)
    assert t2["rr"] == pytest.approx(6.3)
    assert t3["rr"] == pytest.approx(76.7)

    # R:R non-decreasing as price falls
    rrs = [t["rr"] for t in plan]
    assert rrs == sorted(rrs)


def test_compute_entry_plan_invalidation_uses_min_of_bear_lo_and_low_52w():
    # bear_lo=50, low_52w=45 -> min=45 -> invalidation = round(45*0.95, 2) = 42.75
    valuation = {"fair_value_range": {"bear": {"lo": 50}, "base": {"lo": 30, "hi": 60}, "bull": {"hi": 90}}}
    technical = {"low_52w": 45}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    assert plan  # sanity: some tranches produced
    assert plan[0]["invalidation"] == pytest.approx(42.75)


def test_compute_entry_plan_uses_technical_levels_as_candidates():
    valuation = {"fair_value_range": {"bear": {}, "base": {}, "bull": {}}}
    technical = {"low_52w": 40, "sma50": 60, "sma200": 50}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    levels = sorted(round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2, 2) for t in plan)
    assert levels == [40.0, 50.0, 60.0]


def test_compute_entry_plan_dedupe_collapses_near_equal_levels():
    # base_lo=99, sma50=100.5 -> relative distance |99-100.5|/100.5 = 0.0149 < 0.02
    # -> collapsed to a single tranche (the higher level, 100.5, is kept).
    valuation = {"fair_value_range": {"bear": {}, "base": {"lo": 99}, "bull": {}}}
    technical = {"sma50": 100.5}
    plan = planning.compute_entry_plan(valuation, technical, 101)
    assert len(plan) == 1
    entry = (plan[0]["price_zone"]["lo"] + plan[0]["price_zone"]["hi"]) / 2
    assert entry == pytest.approx(100.5, abs=0.01)


@pytest.mark.parametrize("price", [None, 0, -1])
def test_compute_entry_plan_empty_when_price_missing_or_non_positive(price):
    assert planning.compute_entry_plan(_valuation_for_entry_plan(), None, price) == []


def test_compute_entry_plan_empty_when_no_candidates_at_all():
    valuation = {"fair_value_range": {"bear": {}, "base": {}, "bull": {}}}
    assert planning.compute_entry_plan(valuation, None, 100) == []


def test_compute_entry_plan_empty_when_no_candidate_at_or_below_price():
    valuation = {"fair_value_range": {"bear": {"lo": 500}, "base": {"lo": 600, "hi": 700}, "bull": {"hi": 800}}}
    assert planning.compute_entry_plan(valuation, None, 100) == []


def test_compute_entry_plan_none_target_when_no_bull_or_base_hi_available():
    valuation = {"fair_value_range": {"bear": {"lo": 40}, "base": {}, "bull": {}}}
    plan = planning.compute_entry_plan(valuation, None, 100)
    assert plan
    assert all(t["target"] is None and t["rr"] is None for t in plan)


def test_compute_entry_plan_never_raises_on_garbage_input():
    assert planning.compute_entry_plan(None, None, None) == []
    assert planning.compute_entry_plan({}, {}, 100) == []
    assert planning.compute_entry_plan({"fair_value_range": "not a dict"}, None, 100) == []


# ---------------------------------------------------------------------------
# compute_stop_adding
# ---------------------------------------------------------------------------


def test_compute_stop_adding_below_bear_floor_fires():
    valuation = {"fair_value_range": {"bear": {"lo": 60.0}}}
    technical = {"price": 50.0}
    signals = planning.compute_stop_adding(valuation, technical, None, None, None)
    assert [s["code"] for s in signals] == ["BELOW_BEAR_FLOOR"]


def test_compute_stop_adding_near_invalidation_fires_at_and_within_buffer():
    # invalidation=100.0, buffer=3% -> threshold = 103.0 ; price <= threshold fires.
    entry_plan = [{"invalidation": 100.0}]
    technical = {"price": 103.0}
    signals = planning.compute_stop_adding({}, technical, None, entry_plan, None)
    assert [s["code"] for s in signals] == ["NEAR_INVALIDATION"]


def test_compute_stop_adding_near_invalidation_does_not_fire_just_above_buffer():
    entry_plan = [{"invalidation": 100.0}]
    technical = {"price": 103.01}
    signals = planning.compute_stop_adding({}, technical, None, entry_plan, None)
    assert signals == []


def test_compute_stop_adding_high_uncertainty_fires():
    valuation = {"sensitivity": {"high_uncertainty": True}}
    signals = planning.compute_stop_adding(valuation, None, None, None, None)
    assert [s["code"] for s in signals] == ["HIGH_UNCERTAINTY"]


def test_compute_stop_adding_active_red_flag_fires_and_summarizes_all_flags():
    red_flags = [{"message": "flag a"}, {"message": "flag b"}]
    signals = planning.compute_stop_adding({}, None, red_flags, None, None)
    assert len(signals) == 1
    assert signals[0]["code"] == "ACTIVE_RED_FLAG"
    assert "flag a" in signals[0]["message"] and "flag b" in signals[0]["message"]


def test_compute_stop_adding_binary_catalyst_near_fires():
    catalyst = {"label": "Q2 earnings ~27 Ağu"}
    signals = planning.compute_stop_adding({}, None, None, None, catalyst)
    assert [s["code"] for s in signals] == ["BINARY_CATALYST_NEAR"]


def test_compute_stop_adding_fixed_order_when_all_signals_apply():
    valuation = {
        "fair_value_range": {"bear": {"lo": 60.0}},
        "sensitivity": {"high_uncertainty": True},
    }
    technical = {"price": 50.0}
    entry_plan = [{"invalidation": 49.0}]  # threshold = 49*1.03 = 50.47 >= price 50 -> fires
    red_flags = [{"message": "a flag"}]
    catalyst = {"label": "Earnings"}

    signals = planning.compute_stop_adding(valuation, technical, red_flags, entry_plan, catalyst)

    assert [s["code"] for s in signals] == [
        "BELOW_BEAR_FLOOR",
        "NEAR_INVALIDATION",
        "HIGH_UNCERTAINTY",
        "ACTIVE_RED_FLAG",
        "BINARY_CATALYST_NEAR",
    ]


def test_compute_stop_adding_empty_when_nothing_applies():
    valuation = {"fair_value_range": {"bear": {"lo": 10.0}}, "sensitivity": {"high_uncertainty": False}}
    technical = {"price": 100.0}
    assert planning.compute_stop_adding(valuation, technical, [], [], None) == []
    assert planning.compute_stop_adding(valuation, technical, None, None, None) == []


def test_compute_stop_adding_never_raises_on_all_none_inputs():
    assert planning.compute_stop_adding(None, None, None, None, None) == []


def test_compute_stop_adding_never_raises_on_garbage_input():
    assert planning.compute_stop_adding("not a dict", "also not a dict", "nope", "nope", "nope") == []


# ---------------------------------------------------------------------------
# select_thesis_metric
# ---------------------------------------------------------------------------


def test_select_thesis_metric_mature_sector_uses_net_margin():
    # latest fy=2023 net_margin=0.234 -> "%23.4"; prior fy=2022 net_margin=0.20
    # diff = 0.234 - 0.20 = 0.034 >= 0.01 -> "iyileşiyor"
    ratios = [
        {"fy": 2023, "net_margin": 0.234},
        {"fy": 2022, "net_margin": 0.20},
    ]
    result = planning.select_thesis_metric("mature", ratios, {})
    assert result["name"] == "Net Kâr Marjı"
    assert result["latest_value"] == "%23.4"
    assert result["trend"] == "iyileşiyor"
    assert planning._THESIS_INVALIDATION_RULE_TR in result["rationale"]


def test_select_thesis_metric_trend_bozuluyor_when_metric_worsens():
    # diff = 0.10 - 0.20 = -0.10 <= -0.01 -> "bozuluyor"
    ratios = [{"fy": 2023, "net_margin": 0.10}, {"fy": 2022, "net_margin": 0.20}]
    result = planning.select_thesis_metric("mature", ratios, {})
    assert result["trend"] == "bozuluyor"


def test_select_thesis_metric_trend_yatay_within_flat_threshold():
    # diff = 0.205 - 0.20 = 0.005 ; abs(0.005) < 0.01 -> "yatay"
    ratios = [{"fy": 2023, "net_margin": 0.205}, {"fy": 2022, "net_margin": 0.20}]
    result = planning.select_thesis_metric("mature", ratios, {})
    assert result["trend"] == "yatay"


def test_select_thesis_metric_growth_unprofitable_uses_yoy_revenue_growth():
    ratios = [{"fy": 2023, "yoy_revenue_growth": 0.55}, {"fy": 2022, "yoy_revenue_growth": 0.40}]
    result = planning.select_thesis_metric("growth_unprofitable", ratios, {})
    assert result["name"] == "Yıllık Gelir Büyümesi (YoY)"
    assert result["latest_value"] == "%55.0"
    # diff = 0.55 - 0.40 = 0.15 -> iyileşiyor
    assert result["trend"] == "iyileşiyor"


def test_select_thesis_metric_growth_unprofitable_falls_back_to_metrics_revenue_cagr():
    # No yoy_revenue_growth in ratios at all -> fall back to metrics'
    # revenue_cagr_5y (preferred over revenue_cagr_3y); no derivable trend.
    result = planning.select_thesis_metric(
        "growth_unprofitable", [], {"revenue_cagr_5y": 0.25, "revenue_cagr_3y": 0.40}
    )
    assert result["name"] == "Yıllık Gelir Büyümesi (YoY)"
    assert result["latest_value"] == "%25.0"
    assert result["trend"] is None


def test_select_thesis_metric_growth_unprofitable_fallback_uses_3y_when_5y_missing():
    result = planning.select_thesis_metric("growth_unprofitable", [], {"revenue_cagr_3y": 0.40})
    assert result["latest_value"] == "%40.0"
    assert result["trend"] is None


def test_select_thesis_metric_financial_uses_roe():
    ratios = [{"fy": 2023, "roe": 0.18}]
    result = planning.select_thesis_metric("financial", ratios, {})
    assert result["name"] == "Özkaynak Getirisi (ROE, NIM proxy)"
    assert result["latest_value"] == "%18.0"
    assert result["trend"] is None  # no prior fiscal year to compare


def test_select_thesis_metric_reit_uses_fcf_margin():
    ratios = [{"fy": 2023, "fcf_margin": 0.30}]
    result = planning.select_thesis_metric("reit", ratios, {})
    assert result["name"] == "FCF Marjı (FFO proxy)"
    assert result["latest_value"] == "%30.0"


def test_select_thesis_metric_cyclical_uses_gross_margin():
    ratios = [{"fy": 2023, "gross_margin": 0.45}]
    result = planning.select_thesis_metric("cyclical", ratios, {})
    assert result["name"] == "Brüt Kâr Marjı"
    assert result["latest_value"] == "%45.0"


def test_select_thesis_metric_cyclical_falls_back_to_net_margin_when_gross_margin_missing():
    ratios = [{"fy": 2023, "net_margin": 0.12}]
    result = planning.select_thesis_metric("cyclical", ratios, {})
    assert result["name"] == "Net Kâr Marjı"
    assert result["latest_value"] == "%12.0"


def test_select_thesis_metric_latest_value_none_when_not_computable():
    result = planning.select_thesis_metric("mature", [], {})
    assert result["latest_value"] is None
    assert result["trend"] is None
    assert "hesaplanamadı" in result["rationale"]
    assert planning._THESIS_INVALIDATION_RULE_TR in result["rationale"]


def test_select_thesis_metric_unknown_sector_falls_back_to_default_net_margin():
    ratios = [{"fy": 2023, "net_margin": 0.12}]
    result = planning.select_thesis_metric("not-a-real-sector", ratios, {})
    assert result["name"] == "Net Kâr Marjı"
    assert result["latest_value"] == "%12.0"


def test_select_thesis_metric_none_sector_falls_back_to_default_net_margin():
    result = planning.select_thesis_metric(None, [], {})
    assert result["name"] == "Net Kâr Marjı"
    assert planning._THESIS_INVALIDATION_RULE_TR in result["rationale"]


def test_select_thesis_metric_rationale_always_contains_invalidation_rule():
    for sector_type in ("mature", "growth_unprofitable", "financial", "reit", "cyclical", None, "unknown"):
        result = planning.select_thesis_metric(sector_type, [], {})
        assert planning._THESIS_INVALIDATION_RULE_TR in result["rationale"]


def test_select_thesis_metric_never_raises_on_garbage_input():
    result = planning.select_thesis_metric("mature", None, None)
    assert result["latest_value"] is None
    result2 = planning.select_thesis_metric(123, "not a list", "not a dict")
    assert result2["latest_value"] is None
