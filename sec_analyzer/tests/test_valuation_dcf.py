"""Hand-verified numeric tests for ``valuation.dcf``, ``valuation.reverse_dcf``,
and ``valuation.sanity`` (SPEC.md Sec.3-5).

Every numeric expectation here is derived independently by hand (shown in the
comment above each assertion) and checked against the implementation with
``pytest.approx``. This file intentionally does NOT import or reuse the
smoke-test fixtures in ``test_valuation_smoke.py``.
"""

import pytest

from sec_analyzer.valuation.dcf import dcf_per_share
from sec_analyzer.valuation.reverse_dcf import implied_growth, implied_growth_with_status
from sec_analyzer.valuation.sanity import clamp_assumptions, validate_assumptions

# ---------------------------------------------------------------------------
# 1. DCF hand-verified cases (SPEC Sec.4)
# ---------------------------------------------------------------------------


def test_dcf_per_share_hand_verified_case_a_no_debt_no_dilution():
    # fcf0=100, growth_5y=0.10, terminal_growth=0.03, r=0.10, shares=10,
    # dilution=0. F1 (FCFE-direct): dcf_per_share no longer takes a
    # net_debt parameter -- equity == ev directly, no net-debt subtraction.
    #
    # Years 1-5 grow at 10%:
    #   fcf1 = 100*1.10      = 110
    #   fcf2 = 110*1.10      = 121
    #   fcf3 = 121*1.10      = 133.1
    #   fcf4 = 133.1*1.10    = 146.41
    #   fcf5 = 146.41*1.10   = 161.051
    #
    # Years 6-10 fade linearly from 0.10 to 0.03 (g_y = 0.10 + (0.03-0.10)*
    # (y-5)/5):
    #   g6=0.086 -> fcf6 = 161.051*1.086    = 174.901386
    #   g7=0.072 -> fcf7 = 174.901386*1.072 = 187.494285792
    #   g8=0.058 -> fcf8 = 187.494285792*1.058 = 198.368954367936
    #   g9=0.044 -> fcf9 = 198.368954367936*1.044 = 207.097188360125
    #   g10=0.03 -> fcf10 = 207.097188360125*1.03 = 213.310104010929 (== terminal_growth, as required)
    #
    # Since growth_5y == r == 0.10 for years 1-5, each of those years'
    # present value collapses to fcf0 exactly: pv_y = fcf0*(1.10)^y/(1.10)^y
    # = 100. So pv1..pv5 = 100 each -> sum = 500.
    #
    # pv6..pv10 (discounting at 10%, using 1.10^y = 1.771561, 1.9487171,
    # 2.14358881, 2.357947691, 2.5937424601 for y=6..10):
    #   pv6  = 174.901386   / 1.771561    ~= 98.7273
    #   pv7  = 187.494285792/ 1.9487171   ~= 96.2347
    #   pv8  = 198.368954368/ 2.14358881  ~= 92.5408
    #   pv9  = 207.097188360/ 2.357947691 ~= 87.8294
    #   pv10 = 213.310104011/ 2.5937424601~= 82.2403
    #   sum(pv6..pv10) ~= 457.5725
    #
    # pv_sum = 500 + 457.5725 = 957.5725
    #
    # TV = fcf10*(1+g_t)/(r-g_t) = 213.310104011*1.03/0.07 = 219.709407131/0.07
    #    ~= 3138.7058
    # pv(TV) = 3138.7058 / 2.5937424601 ~= 1210.1069
    #
    # ev = pv_sum + pv(TV) ~= 957.5725 + 1210.1069 = 2167.6794
    # equity = ev (FCFE-direct, no net_debt subtraction) = 2167.6794
    # effective_shares = 10 * (1+0)**5 = 10
    # per_share = 2167.6794 / 10 = 216.7679
    result = dcf_per_share(
        fcf0=100.0, growth_5y=0.10, terminal_growth=0.03, discount_rate=0.10,
        shares=10.0, dilution_rate=0.0,
    )

    fcf_path = result["fcf_path"]
    assert len(fcf_path) == 10
    assert fcf_path[0] == pytest.approx(110.0, rel=1e-3)
    assert fcf_path[4] == pytest.approx(161.051, rel=1e-3)
    assert fcf_path[5] == pytest.approx(174.901386, rel=1e-3)
    assert fcf_path[9] == pytest.approx(213.310104, rel=1e-3)

    assert result["tv"] == pytest.approx(3138.7058, rel=1e-3)
    assert result["ev"] == pytest.approx(2167.6794, rel=1e-3)
    assert result["equity"] == pytest.approx(2167.6794, rel=1e-3)
    assert result["effective_shares"] == 10.0
    assert result["per_share"] == pytest.approx(216.7679, rel=1e-3)


def test_dcf_per_share_hand_verified_case_b_with_dilution_only():
    # F1 removed the net_debt parameter entirely (FCFE-direct: no net-debt
    # subtraction at all, ever) -- so this case (formerly "with net debt and
    # dilution") now only exercises dilution_rate. Same fcf0/growth_5y/
    # terminal_growth/r as case A above, so the fcf_path/ev/tv are IDENTICAL
    # (dilution_rate only affects effective_shares/per_share, not the
    # cash-flow projection or EV): ev == equity ~= 2167.6794 (derived above,
    # unchanged from case A since equity is no longer reduced by net_debt).
    #
    # shares=10, dilution_rate=0.02:
    #   equity = ev = 2167.6794 (NOT reduced by any net_debt -- F1)
    #   effective_shares = 10 * (1.02)**5
    #     1.02^2 = 1.0404
    #     1.02^4 = 1.0404^2 = 1.08243216
    #     1.02^5 = 1.08243216*1.02 = 1.1040808032
    #     effective_shares = 11.040808032
    #   per_share = 2167.6794 / 11.040808032 ~= 196.3316
    result = dcf_per_share(
        fcf0=100.0, growth_5y=0.10, terminal_growth=0.03, discount_rate=0.10,
        shares=10.0, dilution_rate=0.02,
    )

    assert result["ev"] == pytest.approx(2167.6794, rel=1e-3)
    # FCFE-direct: equity == ev exactly, no net-debt subtraction of any kind.
    assert result["equity"] == result["ev"]
    assert result["effective_shares"] == pytest.approx(11.040808032, rel=1e-6)
    assert result["per_share"] == pytest.approx(196.3316, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. DCF error paths (SPEC Sec.4)
# ---------------------------------------------------------------------------


def test_dcf_per_share_raises_when_discount_rate_at_or_below_terminal_growth():
    with pytest.raises(ValueError):
        dcf_per_share(100.0, 0.10, 0.05, discount_rate=0.05, shares=10.0)
    # Strictly below, too.
    with pytest.raises(ValueError):
        dcf_per_share(100.0, 0.10, 0.05, discount_rate=0.03, shares=10.0)


def test_dcf_per_share_raises_when_shares_is_none():
    with pytest.raises(ValueError):
        dcf_per_share(100.0, 0.10, 0.02, 0.10, shares=None)


def test_dcf_per_share_raises_when_shares_is_zero_or_negative():
    with pytest.raises(ValueError):
        dcf_per_share(100.0, 0.10, 0.02, 0.10, shares=0)
    with pytest.raises(ValueError):
        dcf_per_share(100.0, 0.10, 0.02, 0.10, shares=-5.0)


def test_dcf_per_share_raises_when_fcf0_is_none():
    with pytest.raises(ValueError):
        dcf_per_share(None, 0.10, 0.02, 0.10, shares=10.0)


# ---------------------------------------------------------------------------
# 3. Reverse DCF round-trip (SPEC Sec.5)
# ---------------------------------------------------------------------------


def test_implied_growth_round_trips_a_known_growth_rate():
    # Pick a known growth_star = 0.12 and use dcf_per_share itself (the
    # module under separate, independent test above) to derive the price
    # that growth rate produces; implied_growth must then bisect back to
    # (approximately) that same growth rate. F1: no net_debt parameter on
    # either dcf_per_share or implied_growth.
    growth_star = 0.12
    fcf0, terminal_growth, discount_rate = 100.0, 0.025, 0.10
    shares, dilution_rate = 20.0, 0.0

    price = dcf_per_share(fcf0, growth_star, terminal_growth, discount_rate, shares, dilution_rate)["per_share"]

    implied = implied_growth(price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)

    assert implied is not None
    assert implied == pytest.approx(growth_star, abs=1e-3)


def test_implied_growth_round_trips_a_negative_known_growth_rate():
    # Same idea with a negative growth_star, to exercise the low end of the
    # bisection bracket ([-0.20, 0.40]).
    growth_star = -0.05
    fcf0, terminal_growth, discount_rate = 100.0, 0.02, 0.09
    shares, dilution_rate = 15.0, 0.0

    price = dcf_per_share(fcf0, growth_star, terminal_growth, discount_rate, shares, dilution_rate)["per_share"]

    implied = implied_growth(price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)

    assert implied is not None
    assert implied == pytest.approx(growth_star, abs=1e-3)


def test_implied_growth_returns_none_when_price_unreachable_in_bracket():
    # per_share is monotonically increasing in growth_5y (higher growth ->
    # higher cash flows -> higher per-share value), so the maximum
    # per_share reachable within the bisection bracket [-0.20, 0.40] is the
    # value at growth_5y=0.40. A price far above that has no root in the
    # bracket -> None.
    fcf0, terminal_growth, discount_rate = 100.0, 0.025, 0.10
    shares, dilution_rate = 20.0, 0.0

    max_reachable_price = dcf_per_share(fcf0, 0.40, terminal_growth, discount_rate, shares, dilution_rate)["per_share"]
    unreachable_price = max_reachable_price * 10

    implied = implied_growth(unreachable_price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)
    assert implied is None


def test_implied_growth_returns_none_for_unusable_inputs():
    # price<=0, missing fcf0, non-positive shares, and r<=g_t must all
    # degrade to None rather than raising (reverse_dcf never raises).
    assert implied_growth(0.0, 100.0, 0.02, 0.10, 10.0) is None
    assert implied_growth(50.0, None, 0.02, 0.10, 10.0) is None
    assert implied_growth(50.0, 100.0, 0.02, 0.10, 0.0) is None
    assert implied_growth(50.0, 100.0, 0.05, 0.05, 10.0) is None


# ---------------------------------------------------------------------------
# 3a. Reverse DCF with bracket-boundary status (SPEC Sec.5, F5) --
# ``implied_growth_with_status``
# ---------------------------------------------------------------------------


def test_implied_growth_with_status_ok_case_matches_hand_verified_dcf():
    # Same round-trip idea as the "ok" tests above, but also asserts the
    # status string. growth_star=0.12, fcf0=100, terminal_growth=0.025,
    # discount_rate=0.10, shares=20 -- price generated by the independently
    # hand-verified dcf_per_share (see Sec.1 above); implied_growth_with_status
    # must bisect back to (approximately) growth_star with status "ok".
    growth_star = 0.12
    fcf0, terminal_growth, discount_rate = 100.0, 0.025, 0.10
    shares, dilution_rate = 20.0, 0.0

    price = dcf_per_share(fcf0, growth_star, terminal_growth, discount_rate, shares, dilution_rate)["per_share"]

    growth, status = implied_growth_with_status(price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)

    assert status == "ok"
    assert growth == pytest.approx(growth_star, abs=1e-3)


def test_implied_growth_with_status_above_bracket_when_price_unreachably_high():
    # Mirrors test_implied_growth_returns_none_when_price_unreachable_in_bracket:
    # a price far above what growth_5y=0.40 (the bracket's own ceiling) can
    # produce means the model's per-share value stays BELOW the market price
    # at both bracket ends -> "above_bracket" (price implies growth > +40%).
    fcf0, terminal_growth, discount_rate = 100.0, 0.025, 0.10
    shares, dilution_rate = 20.0, 0.0

    max_reachable_price = dcf_per_share(fcf0, 0.40, terminal_growth, discount_rate, shares, dilution_rate)["per_share"]
    unreachable_price = max_reachable_price * 10

    growth, status = implied_growth_with_status(
        unreachable_price, fcf0, terminal_growth, discount_rate, shares, dilution_rate
    )
    assert growth is None
    assert status == "above_bracket"


def test_implied_growth_with_status_below_bracket_when_price_unreachably_low():
    # Mirror case: a price far below what growth_5y=-0.20 (the bracket's own
    # floor) can produce means the model's per-share value stays ABOVE the
    # market price at both ends -> "below_bracket" (price implies growth <
    # -20%).
    fcf0, terminal_growth, discount_rate = 100.0, 0.025, 0.10
    shares, dilution_rate = 20.0, 0.0

    min_reachable_price = dcf_per_share(fcf0, -0.20, terminal_growth, discount_rate, shares, dilution_rate)["per_share"]
    # min_reachable_price is positive here (fcf0=100 > 0), so halving it
    # keeps the price positive while still landing below the floor value.
    unreachably_low_price = min_reachable_price * 0.5

    growth, status = implied_growth_with_status(
        unreachably_low_price, fcf0, terminal_growth, discount_rate, shares, dilution_rate
    )
    assert growth is None
    assert status == "below_bracket"


def test_implied_growth_with_status_no_data_for_unusable_inputs():
    # Mirrors test_implied_growth_returns_none_for_unusable_inputs, plus the
    # explicit "no_data" status for each unusable-input case.
    assert implied_growth_with_status(None, 100.0, 0.02, 0.10, 10.0) == (None, "no_data")
    assert implied_growth_with_status(50.0, None, 0.02, 0.10, 10.0) == (None, "no_data")
    assert implied_growth_with_status(50.0, 100.0, 0.02, 0.10, 0.0) == (None, "no_data")
    assert implied_growth_with_status(50.0, 100.0, 0.05, 0.05, 10.0) == (None, "no_data")


# ---------------------------------------------------------------------------
# 4. Sanity check (SPEC Sec.3) -- each rule individually
# ---------------------------------------------------------------------------


def _scenario(growth_5y=0.08, terminal_growth=0.02, discount_rate=0.10, story="s"):
    return {"growth_5y": growth_5y, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": story}


def _three_scenarios(base_overrides=None):
    base_overrides = base_overrides or {}
    return {
        "bear": _scenario(),
        "base": _scenario(**base_overrides),
        "bull": _scenario(),
    }


def test_validate_assumptions_fully_valid_set_returns_empty_list():
    assumptions = _three_scenarios()
    assert validate_assumptions(assumptions, is_unprofitable=False) == []


def test_validate_assumptions_rule_terminal_growth_too_high_fires_alone():
    # terminal_growth=0.05 > 0.04 max; discount_rate=0.10 stays above the
    # 0.07 min and above terminal_growth=0.05, so no other rule fires.
    assumptions = _three_scenarios({"terminal_growth": 0.05, "discount_rate": 0.10})
    violations = validate_assumptions(assumptions, is_unprofitable=False)

    assert len(violations) == 1
    assert "%4" in violations[0] and "Base" in violations[0]


def test_validate_assumptions_rule_discount_rate_below_minimum_fires_alone():
    # discount_rate=0.05 < 0.07 min; terminal_growth=0.02 keeps
    # discount_rate (0.05) > terminal_growth (0.02), and terminal_growth
    # stays <= 0.04, so no other rule fires.
    assumptions = _three_scenarios({"discount_rate": 0.05, "terminal_growth": 0.02})
    violations = validate_assumptions(assumptions, is_unprofitable=False)

    assert len(violations) == 1
    assert "alt sınır" in violations[0] and "Base" in violations[0]


def test_validate_assumptions_rule_discount_rate_below_minimum_unprofitable_threshold():
    # is_unprofitable raises the minimum from 0.07 to 0.10: discount_rate=
    # 0.08 is fine for a profitable filer but violates the unprofitable
    # 0.10 floor. terminal_growth=0.02 keeps 0.08 > terminal_growth (no
    # Gordon violation) and terminal_growth <= 0.04 (no terminal violation).
    assumptions = _three_scenarios({"discount_rate": 0.08, "terminal_growth": 0.02})
    assert validate_assumptions(assumptions, is_unprofitable=False) == []

    violations = validate_assumptions(assumptions, is_unprofitable=True)
    assert len(violations) == 1
    assert "zarar eden" in violations[0]


def test_validate_assumptions_rule_growth_5y_hard_max_fires_alone():
    # growth_5y=0.45 > 0.40 hard max; everything else stays within bounds.
    assumptions = _three_scenarios({"growth_5y": 0.45})
    violations = validate_assumptions(assumptions, is_unprofitable=False)

    assert len(violations) == 1
    assert "%40" in violations[0] and "Base" in violations[0]


def test_validate_assumptions_growth_5y_between_20_and_40_percent_is_allowed():
    # SPEC Sec.3: growth_5y > 0.20 is allowed by design; only > 0.40 is a
    # violation. 0.30 must NOT trigger anything.
    assumptions = _three_scenarios({"growth_5y": 0.30})
    assert validate_assumptions(assumptions, is_unprofitable=False) == []


def test_validate_assumptions_rule_missing_field_fires_alone_and_names_field():
    assumptions = _three_scenarios()
    del assumptions["bull"]["discount_rate"]

    violations = validate_assumptions(assumptions, is_unprofitable=False)

    assert len(violations) == 1
    assert "discount_rate" in violations[0] and "Bull" in violations[0]


def test_validate_assumptions_non_numeric_field_is_also_a_violation():
    assumptions = _three_scenarios()
    assumptions["bear"]["growth_5y"] = "yuksek"

    violations = validate_assumptions(assumptions, is_unprofitable=False)
    assert len(violations) == 1
    assert "growth_5y" in violations[0] and "Bear" in violations[0]


def test_validate_assumptions_rule_gordon_undefined_cannot_fire_in_true_isolation():
    # NOTE ON RULE INTERACTION (not a bug): the discount_rate<=terminal_growth
    # ("Gordon undefined") rule can never fire completely alone under the
    # SPEC's own numeric bounds. To trigger it you need discount_rate <=
    # terminal_growth; to avoid the "terminal_growth > 0.04" rule you need
    # terminal_growth <= 0.04; to avoid the "discount_rate < 0.07" rule you
    # need discount_rate >= 0.07. But discount_rate <= terminal_growth <=
    # 0.04 < 0.07 <= discount_rate is a contradiction. So whenever the
    # Gordon-undefined rule fires, at least one of the other two numeric
    # rules necessarily fires alongside it. This test documents that
    # interaction explicitly rather than asserting a (structurally
    # impossible) single-violation case.
    assumptions = _three_scenarios({"discount_rate": 0.03, "terminal_growth": 0.05})
    violations = validate_assumptions(assumptions, is_unprofitable=False)

    assert len(violations) == 3
    assert any("Gordon" in v for v in violations)
    assert any("%4" in v for v in violations)  # terminal_growth > 0.04
    assert any("alt sınır" in v for v in violations)  # discount_rate < 0.07


def test_validate_assumptions_never_raises_on_missing_scenario():
    assumptions = _three_scenarios()
    del assumptions["bear"]

    violations = validate_assumptions(assumptions, is_unprofitable=False)
    assert any("Bear" in v for v in violations)


# ---------------------------------------------------------------------------
# 5. Clamping (SPEC Sec.3, F5) -- ``sanity.clamp_assumptions``
# ---------------------------------------------------------------------------


def test_clamp_assumptions_caps_terminal_growth():
    # base terminal_growth=0.06 > 0.04 max -> capped to 0.04. bear/bull stay
    # at the _scenario() default (0.02), so bear(0.02)<=base(0.04)<=bull(0.02)
    # is the ONLY thing that could add a stray ordering note -- it doesn't
    # apply here since growth_5y (not terminal_growth) is what the ordering
    # check compares, and growth_5y stays at the default 0.08 for all three.
    assumptions = _three_scenarios({"terminal_growth": 0.06, "discount_rate": 0.10})
    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped["base"]["terminal_growth"] == pytest.approx(0.04)
    assert len(notes) == 1
    assert "%4" in notes[0] and "Base" in notes[0]
    # Untouched fields stay exactly as given.
    assert clamped["base"]["growth_5y"] == pytest.approx(0.08)
    assert clamped["base"]["discount_rate"] == pytest.approx(0.10)
    assert clamped["bear"] == assumptions["bear"]
    assert clamped["bull"] == assumptions["bull"]
    # The original input dict must not be mutated.
    assert assumptions["base"]["terminal_growth"] == 0.06


def test_clamp_assumptions_caps_growth_5y():
    # bull growth_5y=0.45 > 0.40 hard max -> capped to 0.40. bear=0.05,
    # base=0.10 chosen so bear<=base<=bull holds BOTH before (0.05<=0.10<=
    # 0.45) and after (0.05<=0.10<=0.40) clamping -- isolating the growth_5y
    # cap from the cross-scenario ordering check.
    assumptions = _three_scenarios()
    assumptions["bear"]["growth_5y"] = 0.05
    assumptions["base"]["growth_5y"] = 0.10
    assumptions["bull"]["growth_5y"] = 0.45

    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped["bull"]["growth_5y"] == pytest.approx(0.40)
    assert len(notes) == 1
    assert "%40" in notes[0] and "Bull" in notes[0]
    assert clamped["bear"]["growth_5y"] == pytest.approx(0.05)
    assert clamped["base"]["growth_5y"] == pytest.approx(0.10)


def test_clamp_assumptions_floors_discount_rate_profitable_threshold():
    # base discount_rate=0.05 < 0.07 (profitable floor) -> raised to 0.07.
    # terminal_growth=0.02 keeps 0.07 > 0.02 (no stray Gordon situation) and
    # <= 0.04 (no terminal cap).
    assumptions = _three_scenarios({"discount_rate": 0.05, "terminal_growth": 0.02})
    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped["base"]["discount_rate"] == pytest.approx(0.07)
    assert len(notes) == 1
    assert "%7" in notes[0] and "Base" in notes[0]
    assert "zarar eden" not in notes[0]


def test_clamp_assumptions_floors_discount_rate_unprofitable_threshold():
    # Same base discount_rate=0.08 is fine for a profitable filer (>=0.07)
    # but violates the unprofitable 0.10 floor -> raised to 0.10, and the
    # note must mention the unprofitable case.
    assumptions = _three_scenarios({"discount_rate": 0.08, "terminal_growth": 0.02})

    clamped_profitable, notes_profitable = clamp_assumptions(assumptions, is_unprofitable=False)
    assert clamped_profitable["base"]["discount_rate"] == pytest.approx(0.08)
    assert notes_profitable == []

    clamped_unprofitable, notes_unprofitable = clamp_assumptions(assumptions, is_unprofitable=True)
    assert clamped_unprofitable["base"]["discount_rate"] == pytest.approx(0.10)
    assert len(notes_unprofitable) == 1
    assert "zarar eden" in notes_unprofitable[0]


def test_clamp_assumptions_does_not_clamp_r_less_equal_g_t_case():
    # discount_rate=0.07 (already at/above the 0.07 floor -> the floor clamp
    # does NOT touch it) and terminal_growth=0.08 (above the 0.04 cap -> that
    # INDEPENDENT clamp fires and caps it to 0.04). Pre-clamp, r(0.07) <=
    # g_t(0.08) is a Gordon violation; clamp_assumptions has no special-case
    # logic for this relationship (per its docstring) -- the only reason the
    # relationship stops being violated after clamping is the ORDINARY
    # terminal_growth cap firing on its own, not a Gordon-specific fix. Proof:
    # discount_rate itself is completely untouched (no note mentions it, and
    # its clamped value is bit-for-bit the original 0.07).
    assumptions = _three_scenarios({"discount_rate": 0.07, "terminal_growth": 0.08})
    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped["base"]["terminal_growth"] == pytest.approx(0.04)
    assert clamped["base"]["discount_rate"] == 0.07  # untouched, bit-for-bit
    assert len(notes) == 1  # only the terminal_growth cap note -- nothing r/g_t-specific
    assert "%4" in notes[0]
    assert "Gordon" not in notes[0]


def test_clamp_assumptions_growth_ordering_violation_is_note_only_no_value_change():
    # bear(0.20) > base(0.10) > bull(0.05) violates bear<=base<=bull. None of
    # the three individual values exceed the per-scenario caps (all <=0.40)
    # or violate the discount-rate floor, so the ordering note should be the
    # ONLY note, and no growth_5y value should be rewritten (there's no
    # single "correct" reordering to apply).
    assumptions = _three_scenarios()
    assumptions["bear"]["growth_5y"] = 0.20
    assumptions["base"]["growth_5y"] = 0.10
    assumptions["bull"]["growth_5y"] = 0.05

    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped["bear"]["growth_5y"] == pytest.approx(0.20)
    assert clamped["base"]["growth_5y"] == pytest.approx(0.10)
    assert clamped["bull"]["growth_5y"] == pytest.approx(0.05)
    assert len(notes) == 1
    assert "sıralama" in notes[0]


def test_clamp_assumptions_fully_valid_set_passes_through_unchanged_with_no_notes():
    assumptions = _three_scenarios()
    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped == assumptions
    assert notes == []


def test_clamp_assumptions_never_raises_on_missing_or_garbage_input():
    clamped, notes = clamp_assumptions(None, is_unprofitable=False)
    assert clamped == {}
    assert notes == []

    assumptions = _three_scenarios()
    del assumptions["bear"]
    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)
    assert "bear" not in clamped
    assert clamped["base"] == assumptions["base"]


def test_clamp_assumptions_leaves_non_numeric_field_untouched():
    assumptions = _three_scenarios()
    assumptions["base"]["growth_5y"] = "yuksek"

    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped["base"]["growth_5y"] == "yuksek"
    assert notes == []
