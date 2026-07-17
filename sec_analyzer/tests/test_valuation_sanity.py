"""Regression tests for the ``clamp_assumptions`` -> ``validate_assumptions``
float-boundary bug in ``valuation.sanity`` (SPEC.md Sec.3, ERP-spread guard).

Background: ``_clamp_assumptions`` raises a too-thin ``discount_rate`` to
exactly ``terminal_growth + _MIN_ERP_SPREAD``. Re-deriving the spread from
that clamped value (``discount_rate - terminal_growth``) can land a hair
under ``_MIN_ERP_SPREAD`` due to floating-point rounding -- e.g. with
``terminal_growth=0.04``, ``0.04 + 0.045 - 0.04 == 0.04499999999999999``,
which is strictly less than the literal ``0.045`` in IEEE-754 arithmetic.
Before the fix, ``_validate_assumptions`` used a bare ``<`` comparison
against ``_MIN_ERP_SPREAD``, so it flagged a violation on a value
``clamp_assumptions`` had just declared valid. Since callers such as
``interpret.rule_based._default_assumptions`` run clamp then validate and
discard the entire (CAPM-based) assumption set for a flat fallback on any
violation, this silently disabled CAPM discount rates for any scenario
whose clamped rate landed exactly on the ERP-spread boundary.

This file does not re-test every individual sanity rule (see
``test_valuation_dcf.py`` Sec.4-5 for the full per-rule suite); it only
covers the clamp/validate boundary-agreement property the bug broke.
"""

import pytest

from sec_analyzer.valuation.sanity import clamp_assumptions, validate_assumptions


def _scenario(growth_5y=0.08, terminal_growth=0.02, discount_rate=0.10, story="s"):
    return {"growth_5y": growth_5y, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": story}


def _three_scenarios(bear_overrides=None, base_overrides=None, bull_overrides=None):
    bear_overrides = bear_overrides or {}
    base_overrides = base_overrides or {}
    bull_overrides = bull_overrides or {}
    return {
        "bear": _scenario(**bear_overrides),
        "base": _scenario(**base_overrides),
        "bull": _scenario(**bull_overrides),
    }


def test_clamp_then_validate_at_exact_erp_spread_boundary_is_consistent():
    # The exact case from the bug report: base_dr=0.09 already clears the
    # 0.085 boundary (0.04 + 0.045) so it is left untouched; bull_dr=0.08
    # sits inside the too-thin band (0.04 < 0.08 < 0.085) and gets raised to
    # 0.04 + 0.045 == 0.08499999999999999 in float64 -- reproducing the
    # exact rounding that used to fail re-validation.
    assumptions = _three_scenarios(
        base_overrides={"discount_rate": 0.09, "terminal_growth": 0.04},
        bull_overrides={"discount_rate": 0.08, "terminal_growth": 0.04},
        bear_overrides={"discount_rate": 0.09, "terminal_growth": 0.04},
    )

    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    # (a) bull was clamped to the boundary; base was left alone.
    assert clamped["bull"]["discount_rate"] == pytest.approx(0.085)
    assert clamped["base"]["discount_rate"] == pytest.approx(0.09)
    assert any("Bull" in n and "asgari risk primi" in n for n in notes)

    # (b) the clamped set must pass re-validation with zero violations --
    # this is the exact regression: before the fix, the float remainder of
    # clamped["bull"]["discount_rate"] - 0.04 (0.04499999999999999) compared
    # less than the literal 0.045 and spuriously fired the ERP-spread rule.
    violations = validate_assumptions(clamped, is_unprofitable=False)
    assert violations == []


def test_clamp_then_validate_round_trip_is_always_consistent():
    # General property: for any Gordon-defined input (discount_rate >
    # terminal_growth), clamp_assumptions's output must always pass
    # validate_assumptions -- a clamped set is never allowed to still be
    # reported as invalid. Sweeps a spread of terminal_growth values
    # (including the 0.04 cap) crossed with discount_rate values that land
    # below, exactly on, and above the ERP-spread boundary.
    terminal_growths = [0.0, 0.01, 0.02, 0.025, 0.03, 0.04]
    for terminal_growth in terminal_growths:
        boundary = terminal_growth + 0.045
        for discount_rate in (
            terminal_growth + 0.001,  # far below the spread -- needs clamping
            boundary - 1e-6,  # just below the boundary -- needs clamping
            boundary,  # exactly on the boundary -- must NOT be (re-)clamped
            boundary + 1e-6,  # just above the boundary -- already valid
            0.20,  # comfortably above -- already valid
        ):
            assumptions = _three_scenarios(
                base_overrides={"discount_rate": discount_rate, "terminal_growth": terminal_growth}
            )
            clamped, _ = clamp_assumptions(assumptions, is_unprofitable=False)
            violations = validate_assumptions(clamped, is_unprofitable=False)
            assert violations == [], (
                f"clamp->validate inconsistent for terminal_growth={terminal_growth!r}, "
                f"discount_rate={discount_rate!r}: {violations}"
            )


def test_clamp_leaves_value_already_at_boundary_untouched():
    # A discount_rate already exactly at terminal_growth + _MIN_ERP_SPREAD
    # (computed the same way the clamp itself would) must not be rewritten
    # again -- clamp is idempotent at the boundary.
    terminal_growth = 0.04
    at_boundary = terminal_growth + 0.045  # == 0.08499999999999999
    assumptions = _three_scenarios(base_overrides={"discount_rate": at_boundary, "terminal_growth": terminal_growth})

    clamped, notes = clamp_assumptions(assumptions, is_unprofitable=False)

    assert clamped["base"]["discount_rate"] == at_boundary
    assert notes == []
    assert validate_assumptions(clamped, is_unprofitable=False) == []


def test_undefined_gordon_case_on_raw_input_is_unaffected_by_the_epsilon_fix():
    # Sanity check that the epsilon tolerance is scoped to the Gordon-defined
    # (tg < dr) elif branch only: on a RAW (unclamped) input where
    # discount_rate <= terminal_growth, that hard violation must still fire,
    # exactly as before this fix (mirrors test_valuation_dcf.py's
    # ``test_validate_assumptions_erp_spread_guard_does_not_double_report_undefined_gordon``).
    # Note this says nothing about clamp's output: in this codebase
    # terminal_growth is always capped at 0.04 while the discount_rate floor
    # is 0.07 (0.10 if unprofitable), so a clamped discount_rate can never
    # remain <= a clamped terminal_growth -- see
    # test_valuation_dcf.py's ``test_clamp_assumptions_does_not_clamp_r_less_equal_g_t_case``
    # for that (already-covered) interaction.
    assumptions = _three_scenarios(base_overrides={"discount_rate": 0.04, "terminal_growth": 0.04})

    violations = validate_assumptions(assumptions, is_unprofitable=False)
    assert any("Gordon" in v for v in violations)
    assert not any("asgari risk primi" in v for v in violations)
