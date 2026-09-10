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


def test_compute_entry_plan_invalidation_uses_min_of_bear_lo_low_52w_and_lowest_kept_dip():
    # bear_lo=50, low_52w=45, base_lo=30 -- base_lo=30 is itself a kept dip
    # candidate (<=price=100) and sits BELOW min(bear_lo, low_52w)=45, so per
    # fix #2 the floor now includes it:
    #   invalidation = round(min(50, 45, 30) * (1 - 0.05), 2)
    #                = round(30 * 0.95, 2) = round(28.5, 2) = 28.5
    # (Old behavior would have ignored base_lo entirely and returned 42.75 --
    # this is the corrected, spec-mandated value.)
    valuation = {"fair_value_range": {"bear": {"lo": 50}, "base": {"lo": 30, "hi": 60}, "bull": {"hi": 90}}}
    technical = {"low_52w": 45}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    assert plan  # sanity: some tranches produced
    assert plan[0]["invalidation"] == pytest.approx(28.5)


def test_compute_entry_plan_invalidation_unchanged_when_no_dip_level_below_bear_lo_and_low_52w():
    # bear_lo=50, low_52w=45 -- and every other dip candidate (base_lo=48,
    # base_hi=70, bull_hi=90) sits AT OR ABOVE min(bear_lo, low_52w)=45, so
    # the lowest kept dip level is low_52w=45 itself (nothing pulls the floor
    # lower). This proves the fix #2 floor only ever lowers the invalidation
    # when a dip level actually sits below the old bear_lo/low_52w floor --
    # here it does not, so the value equals the pre-fix formula:
    #   invalidation = round(min(50, 45, 45) * (1 - 0.05), 2)
    #                = round(45 * 0.95, 2) = round(42.75, 2) = 42.75
    valuation = {"fair_value_range": {"bear": {"lo": 50}, "base": {"lo": 48, "hi": 70}, "bull": {"hi": 90}}}
    technical = {"low_52w": 45}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    assert plan  # sanity: some tranches produced
    assert plan[0]["invalidation"] == pytest.approx(42.75)


def test_compute_entry_plan_invalidation_floor_strictly_below_lowest_dip_price_zone():
    # Same fixture as the "lowest kept dip pulls the floor lower" case above:
    # bear_lo=50, low_52w=45, base_lo=30 -> invalidation = 28.5 (hand-verified
    # above). The lowest dip tranche's own price_zone["lo"] is
    # round(30 * (1 - 0.015), 2) = round(29.55, 2) = 29.55, which is strictly
    # above the invalidation (28.5 < 29.55) -- i.e. the shared floor sits
    # strictly below every dip tranche's zone by construction (fix #2's
    # guarantee), not merely below the trigger level.
    valuation = {"fair_value_range": {"bear": {"lo": 50}, "base": {"lo": 30, "hi": 60}, "bull": {"hi": 90}}}
    technical = {"low_52w": 45}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    dip_tranches = [t for t in plan if t["kind"] == "dip"]
    assert dip_tranches
    lowest_dip = min(dip_tranches, key=lambda t: t["price_zone"]["lo"])
    assert lowest_dip["price_zone"]["lo"] == pytest.approx(29.55)
    assert lowest_dip["invalidation"] == pytest.approx(28.5)
    assert lowest_dip["invalidation"] < lowest_dip["price_zone"]["lo"]


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


# ---------------------------------------------------------------------------
# compute_entry_plan -- two-directional (dip + breakout) staged entry plan
# ---------------------------------------------------------------------------


def test_compute_entry_plan_breakout_candidates_appear_and_are_ordered_above_dip():
    # price=100. Dip candidates (<=100): bear_lo=60, base_lo=70, base_hi=80.
    # Breakout candidates (>100): sma50=105, sma200=118, resistance 125 & 145,
    # high_52w=130 -- all above price, so every one is a breakout candidate.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 60},
            "base": {"lo": 70, "hi": 80},
            "bull": {"hi": 150},
        }
    }
    technical = {
        "sma50": 105,
        "sma200": 118,
        "resistance_levels": [{"price": 125}, {"price": 145}],
        "high_52w": 130,
    }
    plan = planning.compute_entry_plan(valuation, technical, 100)

    assert plan
    breakout_tranches = [t for t in plan if t["kind"] == "breakout"]
    dip_tranches = [t for t in plan if t["kind"] == "dip"]
    assert breakout_tranches  # at least one breakout candidate survived selection
    assert dip_tranches  # at least one dip candidate survived selection
    assert len(plan) <= 5

    for t in breakout_tranches:
        assert "seviyesinin üzerine çıkarsa (yükseliş teyidi" in t["trigger"]
    for t in dip_tranches:
        assert "seviyesinin altına inerse" in t["trigger"]

    # Since the final list is sorted by descending price_zone level and every
    # breakout candidate's level is > price > every dip candidate's level,
    # all breakout tranches must be numbered ahead of all dip tranches.
    max_breakout_n = max(t["n"] for t in breakout_tranches)
    min_dip_n = min(t["n"] for t in dip_tranches)
    assert max_breakout_n < min_dip_n


def test_compute_entry_plan_dip_only_unchanged_when_technical_has_no_above_price_levels():
    # Legacy dip-only shape: no technical input at all -> every candidate
    # (bear_lo=40, base_lo=60, base_hi=80) is <= price=110, so every tranche
    # is kind="dip" with one shared structural invalidation.
    valuation = _valuation_for_entry_plan()
    plan = planning.compute_entry_plan(valuation, None, 110)

    assert plan
    assert all(t["kind"] == "dip" for t in plan)
    assert all("seviyesinin altına inerse" in t["trigger"] for t in plan)
    invalidations = {t["invalidation"] for t in plan}
    assert len(invalidations) == 1  # one shared structural invalidation


def test_compute_entry_plan_breakout_only_when_price_below_all_structural_levels():
    # price=10, every structural level (bands + SMAs + resistance + 52w-high)
    # sits above price -> no dip candidates at all, only breakout ones.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 20},
            "base": {"lo": 25, "hi": 30},
            "bull": {"hi": 50},
        }
    }
    technical = {
        "sma50": 15,
        "sma200": 18,
        "resistance_levels": [{"price": 22}],
        "high_52w": 40,
    }
    plan = planning.compute_entry_plan(valuation, technical, 10)

    assert plan
    assert all(t["kind"] == "breakout" for t in plan)
    # Hand-verify 2 of the 4 per-tranche failed-breakout invalidations:
    #   level=40 (high_52w) -> invalidation = round(40 * (1 - 0.05), 2) = 38.0
    #   level=15 (sma50)    -> invalidation = round(15 * (1 - 0.05), 2) = 14.25
    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}
    assert by_level[40]["invalidation"] == pytest.approx(38.0)
    assert by_level[15]["invalidation"] == pytest.approx(14.25)


def test_compute_entry_plan_selection_balances_both_sides_when_over_cap():
    # price=1000. Dip raw sources (all <=1000, spaced far apart so none
    # dedupe): bear_lo=100, base_lo=200, base_hi=300, bull_hi=400,
    # low_52w=500, sma50=600, sma200=700 -> 7 dip candidates (>5).
    # Breakout raw sources (all >1000): 5 resistance zones (1100..1900,
    # step 200) + high_52w=2100 -> 6 breakout candidates (>5).
    # Both sides exceed the cap (5), so selection must guarantee >=1
    # tranche per side and cap the total at 5.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 100},
            "base": {"lo": 200, "hi": 300},
            "bull": {"hi": 400},
        }
    }
    technical = {
        "low_52w": 500,
        "sma50": 600,
        "sma200": 700,
        "resistance_levels": [
            {"price": 1100},
            {"price": 1300},
            {"price": 1500},
            {"price": 1700},
            {"price": 1900},
        ],
        "high_52w": 2100,
    }
    plan = planning.compute_entry_plan(valuation, technical, 1000)

    assert len(plan) == 5
    dip_count = sum(1 for t in plan if t["kind"] == "dip")
    breakout_count = sum(1 for t in plan if t["kind"] == "breakout")
    assert dip_count >= 1
    assert breakout_count >= 1
    assert dip_count + breakout_count == 5

    # Sizing: weights = [1,2,3,4,5], sum=15 -> size_pct =
    #   [round(1/15*100,1), ..., round(5/15*100,1)]
    #   = [6.7, 13.3, 20.0, 26.7, 33.3], summing to 100.0.
    sizes = [t["size_pct"] for t in plan]
    assert sizes == pytest.approx([6.7, 13.3, 20.0, 26.7, 33.3])
    assert sum(sizes) == pytest.approx(100.0)
    # Cheapest (last / lowest-priced) tranche gets the largest allocation.
    ordered_by_price_desc = sorted(plan, key=lambda t: -(t["price_zone"]["lo"] + t["price_zone"]["hi"]))
    assert ordered_by_price_desc[-1]["size_pct"] == max(sizes)


def _mixed_plan_for_rr_and_note_checks():
    # price=100. Dip candidates (<=100): bear_lo=50, base_lo=70 -> 2 dip
    # tranches sharing invalidation = round(min(bear_lo=50) * 0.95, 2) = 47.5
    # (no low_52w given, so bear_lo is the only invalidation source).
    # Breakout candidates (>100): sma50=110, sma200=130, resistance=250 -> 3
    # breakout tranches, each with its own invalidation = level * 0.95.
    # Total = 5 <= cap, both sides present -> every candidate is kept.
    # target = bull.hi = 200.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 50},
            "base": {"lo": 70, "hi": 120},
            "bull": {"hi": 200},
        }
    }
    technical = {
        "sma50": 110,
        "sma200": 130,
        "resistance_levels": [{"price": 250}],
    }
    return planning.compute_entry_plan(valuation, technical, 100)


def test_compute_entry_plan_per_tranche_invalidation_dip_shared_breakout_own():
    plan = _mixed_plan_for_rr_and_note_checks()
    assert len(plan) == 5
    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}

    # Dip tranches (levels 70, 50) share one invalidation.
    assert by_level[70]["kind"] == "dip"
    assert by_level[50]["kind"] == "dip"
    assert by_level[70]["invalidation"] == pytest.approx(47.5)
    assert by_level[50]["invalidation"] == pytest.approx(47.5)

    # Breakout tranches each carry their own failed-breakout invalidation:
    #   level=130 -> round(130 * 0.95, 2) = 123.5
    #   level=110 -> round(110 * 0.95, 2) = 104.5
    #   level=250 -> round(250 * 0.95, 2) = 237.5
    assert by_level[130]["kind"] == "breakout"
    assert by_level[130]["invalidation"] == pytest.approx(123.5)
    assert by_level[110]["invalidation"] == pytest.approx(104.5)
    assert by_level[250]["invalidation"] == pytest.approx(237.5)


def test_compute_entry_plan_rr_per_own_invalidation_hand_verified():
    # Reuses the mixed fixture above. target=200, cost=0.002 (round-trip).
    # rr = round(reward/risk, 1) where entry == level (symmetric zone band),
    # reward = target*(1-cost) - entry*(1+cost), risk = entry*(1+cost) - invalidation.
    #
    # Breakout level=130, invalidation=123.5:
    #   reward = 200*0.998 - 130*1.002 = 199.6 - 130.26 = 69.34
    #   risk   = 130.26 - 123.5 = 6.76
    #   rr = round(69.34/6.76, 1) = 10.3
    #
    # Dip level=70, invalidation=47.5:
    #   reward = 199.6 - 70*1.002 = 199.6 - 70.14 = 129.46
    #   risk   = 70.14 - 47.5 = 22.64
    #   rr = round(129.46/22.64, 1) = 5.7
    #
    # Breakout level=250 (>= target=200), invalidation=237.5:
    #   reward = 199.6 - 250*1.002 = 199.6 - 250.5 = -50.9 (<=0) -> rr is None
    plan = _mixed_plan_for_rr_and_note_checks()
    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}

    assert by_level[130]["rr"] == pytest.approx(10.3)
    assert by_level[70]["rr"] == pytest.approx(5.7)
    assert by_level[250]["rr"] is None  # entry at/above target -> no positive reward


def test_compute_entry_plan_rr_ters_note_scoped_to_adjacent_dip_pairs_only():
    # Reuses the mixed fixture: hand-verified rr sequence (descending price
    # order) is [None, 10.3, 15.6, 5.7, 57.5] for
    # [breakout250, breakout130, breakout110, dip70, dip50].
    # The breakout110->dip70 transition (15.6 -> 5.7) is a REAL decrease, but
    # since the pair is not dip-dip, the monotonicity note must NOT fire.
    # The dip70->dip50 transition (5.7 -> 57.5) increases, consistent with
    # the "guaranteed monotonic among dip tranches" design -- also no note.
    #
    # Per fix #1, breakout250's entry (250) is >= target (200), so it now
    # carries the "Model üstü" note instead of None -- that note is unrelated
    # to R:R monotonicity, so the real point of this test (no *R:R ters*
    # monotonicity note ever appears on a breakout tranche, or across a
    # mixed/breakout pair) is re-asserted by checking for the absence of the
    # "R:R sırası ters" string specifically, not by asserting note is None.
    plan = _mixed_plan_for_rr_and_note_checks()
    ordered = sorted(plan, key=lambda t: -(t["price_zone"]["lo"] + t["price_zone"]["hi"]))
    rrs = [t["rr"] for t in ordered]
    kinds = [t["kind"] for t in ordered]
    assert kinds == ["breakout", "breakout", "breakout", "dip", "dip"]
    assert rrs == [None, pytest.approx(10.3), pytest.approx(15.6), pytest.approx(5.7), pytest.approx(57.5)]

    # No tranche of kind "breakout" ever carries the "R:R ters" monotonicity
    # note. breakout250 (entry=250 >= target=200) instead carries the
    # "Model üstü" note (fix #1); the other two breakout tranches (130, 110,
    # both below target) carry no note at all.
    for t in ordered:
        if t["kind"] == "breakout":
            assert "R:R sırası ters" not in (t["note"] or "")
    above_target_breakout = ordered[0]  # breakout, level=250, entry >= target=200
    assert above_target_breakout["note"] == (
        "Model üstü: tetik seviyesi model bull hedefinin (200.00 USD) üzerinde; "
        "değer-çapalı R:R tanımsız -- yalnızca trend-takip girişi."
    )
    for t in ordered[1:3]:  # breakout130, breakout110 -- both below target
        assert t["kind"] == "breakout"
        assert t["note"] is None

    # The breakout(15.6) -> dip(5.7) mixed pair is a genuine decrease, yet
    # must not be flagged (only consecutive dip-dip pairs are checked).
    breakout_to_dip = ordered[2]  # breakout, level=110, rr=15.6
    first_dip = ordered[3]  # dip, level=70, rr=5.7
    assert breakout_to_dip["kind"] == "breakout" and first_dip["kind"] == "dip"
    assert first_dip["rr"] < breakout_to_dip["rr"]  # genuine decrease across the pair
    assert first_dip["note"] is None  # not flagged: not a dip-dip pair

    # The dip-dip pair (70 -> 50) is naturally non-decreasing (5.7 -> 57.5),
    # so it is not flagged either -- consistent with the "guaranteed by
    # construction" design; a genuine dip-dip inversion could not be
    # constructed through the public compute_entry_plan() API (see this
    # file's module docstring note / the final report for details).
    second_dip = ordered[4]  # dip, level=50, rr=57.5
    assert second_dip["note"] is None


def test_compute_entry_plan_model_ustu_note_on_above_target_breakout_only():
    # price=100, target=bull.hi=150.
    # Dip candidates (<=100, unused for this assertion): bear_lo=60,
    # base_lo=70, base_hi=90 -> 3 dip tranches.
    # Breakout candidates (>100): sma50=120 (< target=150 -> normal tranche,
    # real rr, no "Model üstü" note), resistance=160 (>= target=150 ->
    # "Model üstü" note, rr=None). Total = 3 + 2 = 5 <= cap, all kept.
    #
    # Hand-verify rr for the below-target breakout (level=120, entry=120):
    #   invalidation = round(120 * (1 - 0.05), 2) = 114.0
    #   reward = target*(1-0.002) - entry*(1+0.002)
    #          = 150*0.998 - 120*1.002 = 149.7 - 120.24 = 29.46
    #   risk   = entry*1.002 - invalidation = 120.24 - 114.0 = 6.24
    #   rr = round(29.46 / 6.24, 1) = round(4.7211..., 1) = 4.7
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 60},
            "base": {"lo": 70, "hi": 90},
            "bull": {"hi": 150},
        }
    }
    technical = {"sma50": 120, "resistance_levels": [{"price": 160}]}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}

    above = by_level[160]
    below = by_level[120]

    assert above["kind"] == "breakout" and below["kind"] == "breakout"

    assert above["rr"] is None
    assert above["note"] == (
        "Model üstü: tetik seviyesi model bull hedefinin (150.00 USD) üzerinde; "
        "değer-çapalı R:R tanımsız -- yalnızca trend-takip girişi."
    )

    assert below["note"] is None
    assert below["rr"] == pytest.approx(4.7)


def test_compute_entry_plan_high_52w_skipped_when_resistance_zone_is_52w_high():
    # price=100. Resistance zone at 130 is flagged as the 52w high itself,
    # and high_52w=140 sits >2% away from it ((140-130)/130 = 7.7%, well
    # above the 2% dedupe threshold) -- so a naive dedupe pass would NOT
    # collapse them into one candidate. Per fix #3, high_52w must still be
    # omitted as a *separate* breakout candidate because an above-price
    # resistance zone already represents that "new highs" event -- so only
    # one breakout tranche (the resistance one, at 130) should result.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 60},
            "base": {"lo": 70, "hi": 90},
            "bull": {"hi": 150},
        }
    }
    technical = {"resistance_levels": [{"price": 130, "is_52w_high": True}], "high_52w": 140}
    plan = planning.compute_entry_plan(valuation, technical, 100)

    breakout_tranches = [t for t in plan if t["kind"] == "breakout"]
    assert len(breakout_tranches) == 1
    entry = (breakout_tranches[0]["price_zone"]["lo"] + breakout_tranches[0]["price_zone"]["hi"]) / 2
    assert entry == pytest.approx(130.0)


def test_compute_entry_plan_high_52w_added_when_no_resistance_zone_is_52w_high():
    # Control case for the test above: identical resistance/high_52w levels,
    # but the resistance zone is NOT flagged as the 52w high -> nothing
    # suppresses high_52w, so both the resistance breakout (130) and the
    # separate high_52w breakout (140) survive as distinct candidates.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 60},
            "base": {"lo": 70, "hi": 90},
            "bull": {"hi": 150},
        }
    }
    technical = {"resistance_levels": [{"price": 130}], "high_52w": 140}
    plan = planning.compute_entry_plan(valuation, technical, 100)

    breakout_tranches = [t for t in plan if t["kind"] == "breakout"]
    assert len(breakout_tranches) == 2
    entries = sorted((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2 for t in breakout_tranches)
    assert entries == pytest.approx([130.0, 140.0])


def test_compute_entry_plan_dip_trigger_names_its_source_level():
    # Every dip trigger sentence must name the Turkish source label of the
    # level it came from (same transparency the breakout sentence already
    # had), so the reader can tell a valuation-band level from a technical
    # one. Fixture: bear_lo=40, base_lo=60, base_hi=80 (all <= price=110),
    # plus sma200=95 and low_52w=35 from the technical read. bull_hi=200 is
    # above price (target anchor only). 5 dip candidates, no two within 2%,
    # all kept (cap=5).
    valuation = _valuation_for_entry_plan()
    technical = {"sma200": 95, "low_52w": 35}
    plan = planning.compute_entry_plan(valuation, technical, 110)

    assert len(plan) == 5
    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}
    assert "— SMA200 desteği" in by_level[95]["trigger"]
    assert "— baz senaryo üst bandı" in by_level[80]["trigger"]
    assert "— baz senaryo alt bandı" in by_level[60]["trigger"]
    assert "— ayı senaryosu alt bandı" in by_level[40]["trigger"]
    assert "— 52 hafta dibi" in by_level[35]["trigger"]
    # daily-close-only wording is retained
    assert all("seviyesinin altına inerse" in t["trigger"] for t in plan)


def test_compute_entry_plan_support_confluence_note_on_overlapping_dip_only():
    # support_levels zones are NOT dip candidates, but a dip level landing
    # inside a zone's low/high band (widened by the 2% dedupe tolerance) gets
    # an informational confluence note. Fixture: base_lo=60 falls inside the
    # 59-61 support zone -> note; bear_lo=40 is far from any zone -> no note.
    # base_hi=80 sits just outside the 82-84 zone even with the 2% widening
    # (82 * 0.98 = 80.36 > 80) -> no note.
    valuation = _valuation_for_entry_plan(bear_lo=40, base_lo=60, base_hi=80, bull_hi=200)
    technical = {
        "support_levels": [
            {"low": 59.0, "high": 61.0, "price": 60.0},
            {"low": 82.0, "high": 84.0, "price": 83.0},
        ]
    }
    plan = planning.compute_entry_plan(valuation, technical, 110)

    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}
    assert by_level[60]["note"] == "Teknik destek bölgesiyle örtüşüyor (59.00-61.00 USD)."
    assert by_level[80]["note"] is None
    assert by_level[40]["note"] is None


def test_compute_entry_plan_support_confluence_tolerance_edge():
    # A dip level just inside the widened band matches: zone 59-61, level
    # 58 -> 59 * 0.98 = 57.82 <= 58, so the note fires; level 50 is far
    # below the widened band, so it does not. (The two dip levels are >2%
    # apart so dedupe keeps both.) Uses sma50/sma200 as the dip levels to
    # keep the fixture to exactly two candidates.
    valuation = {"fair_value_range": {"bear": {}, "base": {}, "bull": {}}}
    technical = {
        "sma50": 58.0,
        "sma200": 50.0,
        "support_levels": [{"low": 59.0, "high": 61.0, "price": 60.0}],
    }
    plan = planning.compute_entry_plan(valuation, technical, 110)

    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2, 1): t for t in plan}
    assert by_level[58.0]["note"] == "Teknik destek bölgesiyle örtüşüyor (59.00-61.00 USD)."
    assert by_level[50.0]["note"] is None


def test_compute_entry_plan_support_confluence_never_touches_breakout_notes():
    # Breakout tranches are untouched by the support-confluence pass: a
    # support zone overlapping a breakout level (via garbage inputs) must not
    # attach a note, and the "Model üstü" note logic is unchanged. Reuses the
    # model-üstü fixture with a support zone straddling the sma50 breakout.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 60},
            "base": {"lo": 70, "hi": 90},
            "bull": {"hi": 150},
        }
    }
    technical = {
        "sma50": 120,
        "resistance_levels": [{"price": 160}],
        "support_levels": [{"low": 119.0, "high": 121.0, "price": 120.0}],
    }
    plan = planning.compute_entry_plan(valuation, technical, 100)
    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}

    assert by_level[120]["kind"] == "breakout"
    assert by_level[120]["note"] is None  # confluence pass is dip-only
    assert by_level[160]["note"].startswith("Model üstü")


def test_apply_stabilization_condition_appends_after_confluence_note():
    # When a dip tranche already carries a confluence note, the falling-knife
    # stabilization precondition appends after it rather than replacing it.
    valuation = _valuation_for_entry_plan(bear_lo=40, base_lo=60, base_hi=80, bull_hi=200)
    technical = {"support_levels": [{"low": 59.0, "high": 61.0, "price": 60.0}]}
    plan = planning.compute_entry_plan(valuation, technical, 110)
    plan = planning.apply_stabilization_condition(plan, True)

    by_level = {round((t["price_zone"]["lo"] + t["price_zone"]["hi"]) / 2): t for t in plan}
    note = by_level[60]["note"]
    assert note.startswith("Teknik destek bölgesiyle örtüşüyor (59.00-61.00 USD).")
    assert "Stabilizasyon koşulu" in note


def test_compute_entry_plan_kind_key_present_and_legacy_keys_retained():
    plan = _mixed_plan_for_rr_and_note_checks()
    assert plan  # sanity
    expected_keys = {
        "n", "trigger", "price_zone", "size_pct", "invalidation", "target", "rr", "note", "kind",
        "trigger_reason", "invalidation_reason", "target_reason", "size_reason",
    }
    for t in plan:
        assert set(t.keys()) == expected_keys
        assert t["kind"] in ("dip", "breakout")


# ---------------------------------------------------------------------------
# compute_entry_plan -- educational "why" annotations (trigger_reason,
# invalidation_reason, target_reason, size_reason)
# ---------------------------------------------------------------------------


def test_trigger_reason_lookup_covers_every_source_label():
    # Nobody may add a new source label to _collect_entry_candidates without
    # also adding its lookup entry -- this must fail loudly here, not
    # silently (as an empty trigger_reason) in the report.
    all_labels = set(planning._DIP_SOURCE_LABELS) | set(planning._BREAKOUT_SOURCE_LABELS) | {
        planning._BREAKOUT_RESISTANCE_SOURCE_LABEL
    }
    assert len(all_labels) == 7 + 3 + 1  # sanity: no accidental overlap collapsing the set
    for label in all_labels:
        reason = planning._TRIGGER_REASON_BY_SOURCE.get(label)
        assert reason, f"missing trigger_reason lookup entry for source label {label!r}"


def test_compute_entry_plan_every_tranche_has_nonempty_trigger_reason():
    plan = _mixed_plan_for_rr_and_note_checks()
    assert plan
    for t in plan:
        assert isinstance(t["trigger_reason"], str) and t["trigger_reason"]


def test_compute_entry_plan_invalidation_reason_names_lowest_kept_level_when_it_binds():
    # Same fixture as test_compute_entry_plan_invalidation_uses_min_of_bear_lo_low_52w_and_lowest_kept_dip:
    # bear_lo=50, low_52w=45, base_lo=30 -- base_lo=30 is the lowest kept dip
    # level and sits below min(bear_lo, low_52w)=45, so it is the binding floor.
    valuation = {"fair_value_range": {"bear": {"lo": 50}, "base": {"lo": 30, "hi": 60}, "bull": {"hi": 90}}}
    technical = {"low_52w": 45}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    dip_tranches = [t for t in plan if t["kind"] == "dip"]
    assert dip_tranches
    for t in dip_tranches:
        assert "en düşük birikim kademesi" in t["invalidation_reason"]
        assert "%5" in t["invalidation_reason"]
        assert "PAYLAŞILAN" in t["invalidation_reason"]


def test_compute_entry_plan_invalidation_reason_names_bear_lo_when_it_binds():
    # bear_lo=20 is now the lowest of (bear_lo=20, low_52w=45, lowest kept
    # dip level=20 -- bear_lo itself is a dip candidate) -- bear_lo binds.
    valuation = {"fair_value_range": {"bear": {"lo": 20}, "base": {"lo": 30, "hi": 60}, "bull": {"hi": 90}}}
    technical = {"low_52w": 45}
    plan = planning.compute_entry_plan(valuation, technical, 100)
    dip_tranches = [t for t in plan if t["kind"] == "dip"]
    assert dip_tranches
    for t in dip_tranches:
        assert "ayı senaryosu alt bandı" in t["invalidation_reason"]
        assert "%5" in t["invalidation_reason"]


def test_compute_entry_plan_invalidation_reason_differs_between_dip_and_breakout():
    plan = _mixed_plan_for_rr_and_note_checks()
    dip_reasons = {t["invalidation_reason"] for t in plan if t["kind"] == "dip"}
    breakout_reasons = {t["invalidation_reason"] for t in plan if t["kind"] == "breakout"}
    assert dip_reasons and breakout_reasons
    assert dip_reasons.isdisjoint(breakout_reasons)
    # Dip reason names the SHARED structural floor; breakout reason explains
    # its own per-tranche failed-breakout stop -- the two must read differently.
    assert all("PAYLAŞILAN" in r for r in dip_reasons)
    assert all("kendine özel" in r for r in breakout_reasons)
    assert all("%5" in r for r in dip_reasons | breakout_reasons)


def test_compute_entry_plan_target_reason_names_bull_hi_when_available():
    valuation = _valuation_for_entry_plan()  # bull_hi=200 present
    plan = planning.compute_entry_plan(valuation, None, 110)
    assert plan
    for t in plan:
        assert t["target_reason"] == planning._TARGET_REASON_BULL
        assert "boğa senaryosunun üst bandına" in t["target_reason"]


def test_compute_entry_plan_target_reason_names_base_hi_fallback_when_bull_hi_missing():
    valuation = {"fair_value_range": {"bear": {"lo": 40}, "base": {"lo": 60, "hi": 80}, "bull": {}}}
    plan = planning.compute_entry_plan(valuation, None, 110)
    assert plan
    for t in plan:
        assert t["target_reason"] == planning._TARGET_REASON_BASE_FALLBACK
        assert "baz senaryonun üst bandına" in t["target_reason"]


def test_compute_entry_plan_target_reason_unavailable_when_no_target():
    valuation = {"fair_value_range": {"bear": {"lo": 40}, "base": {}, "bull": {}}}
    plan = planning.compute_entry_plan(valuation, None, 100)
    assert plan
    for t in plan:
        assert t["target"] is None
        assert t["target_reason"] == planning._TARGET_REASON_UNAVAILABLE


def test_compute_entry_plan_size_reason_present_and_identical_across_tranches():
    plan = _mixed_plan_for_rr_and_note_checks()
    assert plan
    reasons = {t["size_reason"] for t in plan}
    assert reasons == {planning._SIZE_REASON_TR}


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


def test_compute_stop_adding_near_invalidation_uses_dip_structural_floor_not_breakout():
    # entry_plan mixes one breakout tranche (invalidation=30.0, much lower)
    # with two dip tranches (invalidation=90.0 each). The dip-only floor is
    # min(90.0, 90.0) = 90.0 -> threshold = 90.0 * 1.03 = 92.7.
    # price=91.0 <= 92.7 -> fires off the DIP floor.
    # A buggy "min across all tranches" implementation would instead use
    # 30.0 -> threshold = 30.0 * 1.03 = 30.9, and 91.0 > 30.9 would NOT fire.
    # Asserting the signal fires (and its message cites 90.00, not 30.00)
    # proves the dip-only floor is what's actually used.
    entry_plan = [
        {"kind": "breakout", "invalidation": 30.0},
        {"kind": "dip", "invalidation": 90.0},
        {"kind": "dip", "invalidation": 90.0},
    ]
    technical = {"price": 91.0}
    signals = planning.compute_stop_adding({}, technical, None, entry_plan, None)
    assert [s["code"] for s in signals] == ["NEAR_INVALIDATION"]
    assert "90.00" in signals[0]["message"]
    assert "30.00" not in signals[0]["message"]


def test_compute_stop_adding_near_invalidation_skipped_when_no_dip_tranches():
    # Per fix #4, the "fall back to min across all tranches" behavior was
    # removed: a breakout-only plan (no dip-kind tranche at all) has no
    # structural dip floor, so NEAR_INVALIDATION must be skipped entirely --
    # even though price (30.5) sits well within the old fallback's would-be
    # threshold (30.0 * 1.03 = 30.9).
    entry_plan = [{"kind": "breakout", "invalidation": 30.0}]
    technical = {"price": 30.5}
    signals = planning.compute_stop_adding({}, technical, None, entry_plan, None)
    assert [s["code"] for s in signals] == []


def test_compute_stop_adding_near_invalidation_integration_dip_present_fires_on_dip_floor():
    # End-to-end: feed a real compute_entry_plan() output (mixed dip +
    # breakout, from the shared fixture) into compute_stop_adding(). The dip
    # invalidation there is hand-verified as 47.5 (see
    # test_compute_entry_plan_per_tranche_invalidation_dip_shared_breakout_own
    # above): invalidation = round(min(bear_lo=50) * (1 - 0.05), 2) = 47.5.
    # threshold = 47.5 * (1 + 0.03) = 48.925 ; price=48.5 <= 48.925 -> fires,
    # citing the dip invalidation (47.50), not any breakout tranche's.
    plan = _mixed_plan_for_rr_and_note_checks()
    technical = {"price": 48.5}
    signals = planning.compute_stop_adding({}, technical, None, plan, None)
    assert [s["code"] for s in signals] == ["NEAR_INVALIDATION"]
    assert "47.50" in signals[0]["message"]


def test_compute_stop_adding_near_invalidation_integration_breakout_only_plan_skipped():
    # End-to-end companion case: a real compute_entry_plan() output that is
    # breakout-only (price=10, every structural level above it -- same
    # fixture as test_compute_entry_plan_breakout_only_when_price_below_all_
    # structural_levels above) has no dip tranches at all. Price is set
    # right next to the sma50 breakout tranche's own failed-breakout
    # invalidation (round(15 * (1 - 0.05), 2) = 14.25) -- which would have
    # fired under the old "fall back to all tranches" behavior -- but per
    # fix #4 the signal must be skipped entirely since there is no
    # structural dip floor to check against.
    valuation = {
        "fair_value_range": {
            "bear": {"lo": 20},
            "base": {"lo": 25, "hi": 30},
            "bull": {"hi": 50},
        }
    }
    technical_for_plan = {
        "sma50": 15,
        "sma200": 18,
        "resistance_levels": [{"price": 22}],
        "high_52w": 40,
    }
    plan = planning.compute_entry_plan(valuation, technical_for_plan, 10)
    assert plan and all(t["kind"] == "breakout" for t in plan)  # sanity: dip-free

    technical = {"price": 14.3}
    signals = planning.compute_stop_adding({}, technical, None, plan, None)
    assert [s["code"] for s in signals] == []


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


# ---------------------------------------------------------------------------
# thesis_metric cycle position (select_thesis_metric -> _compute_cycle_position)
# ---------------------------------------------------------------------------


def test_select_thesis_metric_always_carries_cycle_key():
    # The "cycle" key is part of the documented shape (SPEC.md), present even
    # when it degrades to None.
    for result in (
        planning.select_thesis_metric("mature", [], {}),
        planning.select_thesis_metric("mature", [{"fy": 2023, "net_margin": 0.12}], {}),
    ):
        assert "cycle" in result


def test_select_thesis_metric_cycle_positions_current_in_trough_peak_range():
    # low=fy2021=0.10, high=fy2022=0.30, current=latest fy2023=0.20
    # position = (0.20 - 0.10) / (0.30 - 0.10) = 0.10 / 0.20 = 0.5
    ratios = [
        {"fy": 2021, "net_margin": 0.10},
        {"fy": 2022, "net_margin": 0.30},
        {"fy": 2023, "net_margin": 0.20},
    ]
    cycle = planning.select_thesis_metric("mature", ratios, {})["cycle"]
    assert cycle is not None
    assert cycle["low"] == 0.10 and cycle["low_fy"] == 2021
    assert cycle["high"] == 0.30 and cycle["high_fy"] == 2022
    assert cycle["current"] == 0.20 and cycle["current_fy"] == 2023
    assert cycle["position"] == 0.5
    assert cycle["n_years"] == 3
    assert cycle["is_cyclical"] is False
    # series is the full annual series, sorted ascending by fiscal year.
    assert cycle["series"] == [
        {"fy": 2021, "value": 0.10},
        {"fy": 2022, "value": 0.30},
        {"fy": 2023, "value": 0.20},
    ]


def test_select_thesis_metric_cycle_is_cyclical_flag_true_for_cyclical_sector():
    ratios = [
        {"fy": 2020, "gross_margin": 0.15},
        {"fy": 2021, "gross_margin": 0.45},
        {"fy": 2022, "gross_margin": 0.30},
    ]
    cycle = planning.select_thesis_metric("cyclical", ratios, {})["cycle"]
    assert cycle is not None
    assert cycle["is_cyclical"] is True
    # current = latest fy2022 = 0.30; low=0.15, high=0.45
    # position = (0.30 - 0.15) / (0.45 - 0.15) = 0.15 / 0.30 = 0.5
    assert cycle["position"] == 0.5


def test_select_thesis_metric_cycle_none_with_single_fiscal_year():
    ratios = [{"fy": 2023, "net_margin": 0.12}]
    assert planning.select_thesis_metric("mature", ratios, {})["cycle"] is None


def test_select_thesis_metric_cycle_none_when_series_is_flat():
    # Two years, identical values -> trough == peak -> no placeable position.
    ratios = [{"fy": 2022, "net_margin": 0.20}, {"fy": 2023, "net_margin": 0.20}]
    assert planning.select_thesis_metric("mature", ratios, {})["cycle"] is None


def test_select_thesis_metric_cycle_none_for_single_point_metrics_fallback():
    # growth_unprofitable with no ratios series falls back to a single
    # metrics figure -> no derivable trough/peak series -> cycle is None.
    result = planning.select_thesis_metric("growth_unprofitable", [], {"revenue_cagr_5y": 0.25})
    assert result["latest_value"] == "%25.0"
    assert result["cycle"] is None


# ---------------------------------------------------------------------------
# METODOLOJI.md Sec.7's quarterly thesis-invalidation rule:
# _classify_quarterly_series (pure classification), _establish_or_load_anchor
# (persisted day-1 anchor direction, sec_analyzer.store.thesis_anchors), and
# select_thesis_metric's end-to-end "quarterly_check" wiring.
#
# _classify_quarterly_series compares each quarter to the PRIOR quarter in a
# window (not a trend-window mean), against a per-metric deadband:
#   margin/ROE-shaped keys (net_margin, gross_margin, fcf_margin, roe): 1.0pp
#   yoy_revenue_growth: 2.0pp (momentum's own revenue-accel deadband)
# A delta inside the deadband is neutral (against_thesis=None): it neither
# breaks nor extends the trailing consecutive-against-thesis streak.
# invalidated = consecutive_against >= 2 (METODOLOJI Sec.7's "iki çeyrek
# üst üste tezin aksi yönde" rule).
# ---------------------------------------------------------------------------

from sec_analyzer.config import Config
from sec_analyzer.store import thesis_anchors as TA


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A throwaway SQLite path, also installed as Config.DB_PATH, so
    planning.py's anchor persistence (which always resolves the default
    Config.DB_PATH -- it never receives a db_path override) is isolated
    per test. Mirrors test_assumptions_cache.py / test_thesis_anchors.py."""
    p = str(tmp_path / "test.sqlite3")
    monkeypatch.setattr(Config, "DB_PATH", p)
    return p


def _qs(pairs):
    """Build an ascending [{"period_end", "value"}, ...] quarterly series
    from a list of (period_end, value) tuples."""
    return [{"period_end": pe, "value": v} for pe, v in pairs]


def _normalized_with_quarters(cik, netincome_quarters, revenue_quarters):
    """A normalized-facts-shaped dict carrying just enough quarterly
    NetIncome/Revenue data for quarterly_ratio_series (net_margin) to work,
    plus the `cik` select_thesis_metric's quarterly_check keys the
    persisted anchor on."""
    return {
        "cik": cik,
        "entity_name": "X",
        "currency": "USD",
        "annual": {},
        "quarterly": {
            "NetIncome": _qs(netincome_quarters),
            "Revenue": _qs(revenue_quarters),
        },
        "missing": [],
        "matched_tags": {},
    }


# --- _classify_quarterly_series -------------------------------------------

def test_classify_quarterly_series_two_consecutive_against_thesis_invalidates():
    # anchor_direction="improving" -> thesis_is_up=True; deadband=1.0pp (net_margin).
    # Q1=20.0 (no prior -> neutral)
    # Q2=22.0: diff=+2.0 (>=1.0pp) -> with-thesis (against=False)
    # Q3=19.0: diff=-3.0 -> against-thesis (against=True)
    # Q4=15.0: diff=-4.0 -> against-thesis (against=True)
    # Trailing streak walks back from Q4: True, True, then Q2=False stops it -> 2.
    series = _qs([
        ("2023-03-31", 20.0), ("2023-06-30", 22.0), ("2023-09-30", 19.0), ("2023-12-31", 15.0),
    ])
    result = planning._classify_quarterly_series("net_margin", series, "improving", 2022)
    assert result is not None
    assert result["anchor_direction"] == "improving"
    assert result["anchor_established_fy"] == 2022
    assert [q["against_thesis"] for q in result["quarters"]] == [None, False, True, True]
    assert result["consecutive_against"] == 2
    assert result["invalidated"] is True


def test_classify_quarterly_series_with_thesis_quarter_breaks_the_streak():
    # Same anchor/deadband as above, but a with-thesis quarter sits between
    # two against-thesis quarters -- only the TRAILING run counts.
    # Q1=20.0 (neutral, no prior)
    # Q2=17.0: diff=-3.0 -> against=True
    # Q3=21.0: diff=+4.0 -> against=False (breaks the streak)
    # Q4=19.0: diff=-2.0 -> against=True (streak restarts, only 1 so far)
    series = _qs([
        ("2023-03-31", 20.0), ("2023-06-30", 17.0), ("2023-09-30", 21.0), ("2023-12-31", 19.0),
    ])
    result = planning._classify_quarterly_series("net_margin", series, "improving", None)
    assert [q["against_thesis"] for q in result["quarters"]] == [None, True, False, True]
    assert result["consecutive_against"] == 1
    assert result["invalidated"] is False


def test_classify_quarterly_series_deadband_noise_is_neutral_extends_streak_without_counting():
    # A delta inside the 1.0pp deadband is neutral: it is skipped when
    # walking the trailing streak backward (doesn't break it) but also
    # doesn't add to the count (doesn't extend it) -- consecutive_against
    # ends up counting only the two real against-thesis quarters, not three.
    # Q1=20.0 (neutral, no prior)
    # Q2=17.0: diff=-3.0 -> against=True
    # Q3=17.5: diff=+0.5 (abs < 1.0pp deadband) -> neutral (against=None)
    # Q4=14.5: diff=-3.0 -> against=True
    series = _qs([
        ("2023-03-31", 20.0), ("2023-06-30", 17.0), ("2023-09-30", 17.5), ("2023-12-31", 14.5),
    ])
    result = planning._classify_quarterly_series("net_margin", series, "improving", None)
    assert [q["against_thesis"] for q in result["quarters"]] == [None, True, None, True]
    # Walking back from Q4: True(1) -> Q3 None (skipped, continues) -> Q2 True(2).
    assert result["consecutive_against"] == 2
    assert result["invalidated"] is True


def test_classify_quarterly_series_fewer_than_three_usable_quarters_returns_none():
    series = _qs([("2023-03-31", 20.0), ("2023-06-30", 22.0)])
    assert planning._classify_quarterly_series("net_margin", series, "improving", None) is None


def test_classify_quarterly_series_empty_series_returns_none():
    assert planning._classify_quarterly_series("net_margin", [], "improving", None) is None


def test_classify_quarterly_series_uses_per_metric_deadband():
    # Same +1.5pp-per-quarter series, anchor_direction="deteriorating" (thesis_is_up=False,
    # so a RISING value is against-thesis). Under yoy_revenue_growth's 2.0pp deadband,
    # 1.5pp is neutral every quarter -> no invalidation. Under net_margin's 1.0pp
    # deadband, the same 1.5pp deltas all clear the bar -> against every quarter.
    series = _qs([
        ("2023-03-31", 10.0), ("2023-06-30", 11.5), ("2023-09-30", 13.0), ("2023-12-31", 14.5),
    ])
    growth_result = planning._classify_quarterly_series("yoy_revenue_growth", series, "deteriorating", None)
    assert [q["against_thesis"] for q in growth_result["quarters"]] == [None, None, None, None]
    assert growth_result["consecutive_against"] == 0
    assert growth_result["invalidated"] is False

    margin_result = planning._classify_quarterly_series("net_margin", series, "deteriorating", None)
    assert [q["against_thesis"] for q in margin_result["quarters"]] == [None, True, True, True]
    assert margin_result["consecutive_against"] == 3
    assert margin_result["invalidated"] is True


# --- _establish_or_load_anchor ---------------------------------------------

def test_establish_or_load_anchor_establishes_on_first_call(db_path):
    anchor = planning._establish_or_load_anchor("111", "net_margin", "improving", 2023)
    assert anchor is not None
    assert anchor["cik"] == "111"
    assert anchor["metric_key"] == "net_margin"
    assert anchor["direction"] == "improving"
    assert anchor["established_fy"] == 2023
    assert TA.get_anchor("111") == anchor


def test_establish_or_load_anchor_sticks_once_established_even_if_trend_later_flips(db_path):
    planning._establish_or_load_anchor("112", "net_margin", "improving", 2022)
    # A later run's annual trend flips to deteriorating, but the metric_key
    # is unchanged -- the anchor must NOT be rewritten (that reversal is
    # exactly what the quarterly check exists to catch).
    loaded = planning._establish_or_load_anchor("112", "net_margin", "deteriorating", 2024)
    assert loaded["direction"] == "improving"
    assert loaded["established_fy"] == 2022


def test_establish_or_load_anchor_defers_when_trend_is_none_or_flat(db_path):
    assert planning._establish_or_load_anchor("113", "net_margin", None, 2023) is None
    assert planning._establish_or_load_anchor("113", "net_margin", "flat", 2023) is None
    # Nothing was persisted -- deferred, not silently established as None.
    assert TA.get_anchor("113") is None


def test_establish_or_load_anchor_reestablishes_on_metric_key_change():
    # Simulates a sector reclassification swapping the anchor metric
    # (e.g. net_margin -> gross_margin): the stored anchor's metric_key no
    # longer matches, so it is re-established with THIS run's trend.
    planning._establish_or_load_anchor("114", "net_margin", "improving", 2022)
    reestablished = planning._establish_or_load_anchor("114", "gross_margin", "deteriorating", 2023)
    assert reestablished["metric_key"] == "gross_margin"
    assert reestablished["direction"] == "deteriorating"
    assert reestablished["established_fy"] == 2023

    # A later run under the SAME (now gross_margin) metric_key, even with a
    # trend that flipped back to improving, must not overwrite it again.
    loaded_again = planning._establish_or_load_anchor("114", "gross_margin", "improving", 2024)
    assert loaded_again["metric_key"] == "gross_margin"
    assert loaded_again["direction"] == "deteriorating"
    assert loaded_again["established_fy"] == 2023


# --- select_thesis_metric integration (quarterly_check end-to-end) --------

def test_select_thesis_metric_quarterly_check_none_when_normalized_not_supplied(db_path):
    # Backward compatibility: existing call sites that don't pass `normalized`
    # simply get quarterly_check=None, same as a missing `cycle`.
    ratios = [{"fy": 2023, "net_margin": 0.234}, {"fy": 2022, "net_margin": 0.20}]
    result = planning.select_thesis_metric("mature", ratios, {})
    assert result["quarterly_check"] is None


def test_select_thesis_metric_quarterly_check_none_for_single_point_metrics_fallback(db_path):
    # growth_unprofitable with no ratios series falls back to metrics'
    # revenue_cagr_5y -- no series, so chosen_key stays None and
    # quarterly_check is never even attempted (even though normalized,
    # with a cik, IS supplied here).
    normalized = _normalized_with_quarters("222", [], [])
    result = planning.select_thesis_metric(
        "growth_unprofitable", [], {"revenue_cagr_5y": 0.25}, normalized=normalized
    )
    assert result["quarterly_check"] is None
    # And no anchor was persisted for this cik, since the check never ran.
    assert TA.get_anchor("222") is None


def test_select_thesis_metric_quarterly_check_none_for_brand_new_pair_with_no_trend_yet(db_path):
    # Only one fiscal year of ratios data -> trend is None (nothing to
    # compare against) -> _establish_or_load_anchor defers -> no anchor yet
    # -> quarterly_check is None, even though normalized/cik was supplied.
    ratios = [{"fy": 2023, "net_margin": 0.20}]
    normalized = _normalized_with_quarters("333", [], [])
    result = planning.select_thesis_metric("mature", ratios, {}, normalized=normalized)
    assert result["trend"] is None
    assert result["quarterly_check"] is None
    assert TA.get_anchor("333") is None


def test_select_thesis_metric_quarterly_check_invalidation_appends_turkish_warning(db_path):
    # Annual trend: fy2022=0.15 -> fy2023=0.20, diff=+0.05 >= 0.01 threshold
    # -> trend "improving" -> a brand-new anchor is established this same run
    # with direction="improving" (thesis_is_up=True), established_fy=2023.
    ratios = [{"fy": 2022, "net_margin": 0.15}, {"fy": 2023, "net_margin": 0.20}]
    # Quarterly net_margin (NetIncome/Revenue*100, Revenue=100 flat so the
    # ratio IS the raw NetIncome value): 20.0, 22.0, 19.0, 15.0 -- identical
    # to the hand-verified "two consecutive against-thesis" case above:
    # Q2 (+2.0) is with-thesis, Q3 (-3.0) and Q4 (-4.0) are against-thesis
    # back-to-back -> consecutive_against=2 -> invalidated=True.
    normalized = _normalized_with_quarters(
        "444",
        netincome_quarters=[
            ("2023-03-31", 20.0), ("2023-06-30", 22.0), ("2023-09-30", 19.0), ("2023-12-31", 15.0),
        ],
        revenue_quarters=[
            ("2023-03-31", 100.0), ("2023-06-30", 100.0), ("2023-09-30", 100.0), ("2023-12-31", 100.0),
        ],
    )
    result = planning.select_thesis_metric("mature", ratios, {}, normalized=normalized)

    assert result["trend"] == "improving"
    qc = result["quarterly_check"]
    assert qc is not None
    assert qc["anchor_direction"] == "improving"
    assert qc["anchor_established_fy"] == 2023
    assert qc["consecutive_against"] == 2
    assert qc["invalidated"] is True
    assert planning._THESIS_INVALIDATION_TRIGGERED_TR in result["rationale"]
    # The general Sec.7 rule sentence is always present too, in addition to
    # the invalidation-triggered one.
    assert planning._THESIS_INVALIDATION_RULE_TR in result["rationale"]

    # The anchor is now persisted for this cik/metric.
    anchor = TA.get_anchor("444")
    assert anchor is not None
    assert anchor["metric_key"] == "net_margin"
    assert anchor["direction"] == "improving"


def test_select_thesis_metric_quarterly_check_anchor_persists_across_calls(db_path):
    # First run establishes the anchor from that run's own annual trend.
    ratios = [{"fy": 2022, "net_margin": 0.15}, {"fy": 2023, "net_margin": 0.20}]
    normalized = _normalized_with_quarters(
        "555",
        netincome_quarters=[("2023-03-31", 20.0), ("2023-06-30", 21.0), ("2023-09-30", 22.0)],
        revenue_quarters=[("2023-03-31", 100.0), ("2023-06-30", 100.0), ("2023-09-30", 100.0)],
    )
    first = planning.select_thesis_metric("mature", ratios, {}, normalized=normalized)
    assert first["quarterly_check"]["anchor_direction"] == "improving"

    # A second run (e.g. next quarter) with the SAME cik/metric reuses the
    # already-established anchor rather than re-deriving it -- confirmed by
    # checking the stored anchor's established_fy/direction are unchanged.
    anchor_after_first = TA.get_anchor("555")
    second = planning.select_thesis_metric("mature", ratios, {}, normalized=normalized)
    assert second["quarterly_check"]["anchor_direction"] == "improving"
    assert TA.get_anchor("555") == anchor_after_first
