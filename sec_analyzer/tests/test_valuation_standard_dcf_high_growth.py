"""Tests for the standard two-stage DCF's high-growth flag (LEVER 4).

Background: WP5 raised the deterministic growth hard cap from 0.40 to 0.60
(``engine._HYPER_START_GROWTH_CAP``). The hyper-grower and mid-growth
revenue-first DCF paths each have their own arrival-point safety net (TAM
share / implied-revenue-multiple checks) for aggressive growth assumptions,
but the STANDARD two-stage DCF built by ``_build_dcf_scenarios`` (used for
``mature``-classified filers' headline) has none. On the LLM-provider path,
an LLM could propose growth in (0.40, 0.60] with zero aggressiveness signal.

``_build_dcf_scenarios`` now returns a third value, ``high_growth_flag``
(``True`` iff at least one scenario's ``growth_5y`` is strictly greater than
``_STANDARD_DCF_HIGH_GROWTH_FLAG`` = 0.40), and appends exactly one Turkish
note naming the triggering scenario(s) -- this is a reporting-only addition:
it must never change any computed ``per_share``/``lo``/``hi`` value or which
scenario ends up as the headline.
"""

from sec_analyzer.valuation.engine import _STANDARD_DCF_HIGH_GROWTH_FLAG, _build_dcf_scenarios

_FCF0 = 100.0
_SHARES = 50.0
_DILUTION = 0.0


def _assumptions(bear_g=0.05, base_g=0.10, bull_g=0.15):
    return {
        "bear": {"growth_5y": bear_g, "terminal_growth": 0.02, "discount_rate": 0.12},
        "base": {"growth_5y": base_g, "terminal_growth": 0.03, "discount_rate": 0.10},
        "bull": {"growth_5y": bull_g, "terminal_growth": 0.03, "discount_rate": 0.09},
    }


def test_no_flag_when_all_scenarios_at_or_below_threshold():
    """All growth_5y values <= 0.40 -> no flag, no note."""
    assumptions = _assumptions(bear_g=0.05, base_g=0.10, bull_g=0.15)
    scenarios, notes, high_growth_flag = _build_dcf_scenarios(assumptions, _FCF0, _SHARES, _DILUTION)

    assert scenarios is not None
    assert high_growth_flag is False
    assert not any("büyüme varsayımı %40" in note for note in notes)


def test_flag_set_and_note_added_when_base_growth_exceeds_threshold():
    """Base scenario's growth_5y > 0.40 -> flag True, one note naming Base."""
    assumptions = _assumptions(bear_g=0.05, base_g=0.55, bull_g=0.58)
    scenarios, notes, high_growth_flag = _build_dcf_scenarios(assumptions, _FCF0, _SHARES, _DILUTION)

    assert scenarios is not None
    assert high_growth_flag is True
    matching = [note for note in notes if "büyüme varsayımı %40" in note]
    assert len(matching) == 1
    assert "Base" in matching[0]
    assert "Bull" in matching[0]
    assert "Bear" not in matching[0]


def test_flag_note_mentions_all_triggering_scenarios():
    """All three scenarios above threshold -> single note naming all three."""
    assumptions = _assumptions(bear_g=0.45, base_g=0.50, bull_g=0.58)
    _, notes, high_growth_flag = _build_dcf_scenarios(assumptions, _FCF0, _SHARES, _DILUTION)

    assert high_growth_flag is True
    matching = [note for note in notes if "büyüme varsayımı %40" in note]
    assert len(matching) == 1
    assert "Bear" in matching[0]
    assert "Base" in matching[0]
    assert "Bull" in matching[0]


def test_flag_does_not_change_computed_values():
    """The flag/note addition must be purely additive -- per_share/lo/hi for
    a given growth assumption must be identical whether or not the flag
    threshold is crossed by some OTHER scenario in the same call."""
    low_assumptions = _assumptions(bear_g=0.05, base_g=0.10, bull_g=0.15)
    high_assumptions = _assumptions(bear_g=0.05, base_g=0.10, bull_g=0.55)

    low_scenarios, _, low_flag = _build_dcf_scenarios(low_assumptions, _FCF0, _SHARES, _DILUTION)
    high_scenarios, _, high_flag = _build_dcf_scenarios(high_assumptions, _FCF0, _SHARES, _DILUTION)

    assert low_flag is False
    assert high_flag is True
    # bear/base scenarios' assumptions are identical between the two calls,
    # so their computed values must match exactly.
    assert low_scenarios["bear"] == high_scenarios["bear"]
    assert low_scenarios["base"] == high_scenarios["base"]


def test_early_return_yields_false_flag_and_none_scenarios():
    """When fcf0/shares are unusable, the early-return branch must yield
    ``(None, notes, False)`` -- the 3-tuple shape holds even here."""
    scenarios, notes, high_growth_flag = _build_dcf_scenarios(_assumptions(), None, _SHARES, _DILUTION)
    assert scenarios is None
    assert high_growth_flag is False

    scenarios, notes, high_growth_flag = _build_dcf_scenarios(_assumptions(), _FCF0, 0, _DILUTION)
    assert scenarios is None
    assert high_growth_flag is False


def test_threshold_constant_value():
    """Sanity-pin the reference threshold to 0.40 (the pre-WP5 hard cap),
    used here only as a flag threshold, not a cap."""
    assert _STANDARD_DCF_HIGH_GROWTH_FLAG == 0.40
