"""Unit tests for growth-cap transparency + externally-funded growth.

SPEC.md Sec.22. Three things are covered:

* 22a -- both equity anchors report the growth path they ACTUALLY applied,
  so a caller can tell when ``g = b x ROE`` discarded the assumption.
* 22b -- ``rim_external_growth_per_share`` prices the growth the cap
  suppressed, funded by fair-value (value-neutral) equity issuance.
* 22c -- the engine discloses the cap in a note and in the
  ``fair_value_range`` growth label, instead of printing an assumption the
  model never used.

SoFi FY2025 supplies the hand-verified numbers: ``ni0 = 481.32M``,
``bve0 = 10,489.5M`` (so ``roe = 4.59%``), ``shares = 1,275.3M``.
"""

import pytest

from sec_analyzer.valuation.dcf import (
    fcfe_sustainable_growth_per_share,
    rim_external_growth_per_share,
    rim_per_share,
)
from sec_analyzer.valuation.engine import _rim_scenario_meta, run_valuation

_NI = 481.32e6
_BVE = 10489.5e6
_SHARES = 1275.3e6
_ROE = _NI / _BVE          # 0.045887...
_R = 0.10
_TG = 0.04


# ---------------------------------------------------------------------------
# 22a. The anchors report the growth they actually used
# ---------------------------------------------------------------------------


def test_rim_reports_the_cap_when_roe_binds():
    result = rim_per_share(_BVE, _NI, _ROE, 0.25, _TG, _R, _SHARES, terminal_roe=_R)

    assert result["growth_capped"] is True
    assert result["effective_growth_5y"] == pytest.approx(_ROE)
    # Years 1-5 are flat, so all five sit exactly at the cap.
    assert result["growth_path"][:5] == pytest.approx([_ROE] * 5)
    assert len(result["growth_path"]) == 10


def test_rim_reports_no_cap_when_growth_is_below_roe():
    result = rim_per_share(_BVE, _NI, 0.20, 0.05, _TG, _R, _SHARES, terminal_roe=_R)

    assert result["growth_capped"] is False
    assert result["effective_growth_5y"] == pytest.approx(0.05)
    assert result["growth_path"][:5] == pytest.approx([0.05] * 5)


def test_fcfe_anchor_reports_the_same_cap_fields():
    result = fcfe_sustainable_growth_per_share(
        _NI, _ROE, 0.25, _TG, _R, _SHARES, terminal_roe=_R
    )

    assert result["growth_capped"] is True
    assert result["effective_growth_5y"] == pytest.approx(_ROE)
    assert len(result["growth_path"]) == 10


def test_growth_above_the_cap_is_inert_in_the_value():
    """The defect that motivated Sec.22: once the assumption clears ROE the
    growth input stops changing the answer, so the report's stated growth
    describes nothing.

    Assumptions well above ROE are byte-identical, because the cap binds for
    years 1-9 and year 10 is pinned to ``terminal_growth`` regardless of
    ``growth_5y``. A ``growth_5y`` only just above ROE (5% here vs a 4.59%
    ROE) differs in the last decimals, since its years 6-10 fade DIPS BELOW
    the cap and is applied uncapped -- but it still agrees to the cent, which
    is the precision the report actually shows.
    """
    values = {
        g: rim_per_share(_BVE, _NI, _ROE, g, _TG, _R, _SHARES, terminal_roe=_R)["per_share"]
        for g in (0.05, 0.10, 0.25, 0.40)
    }

    assert len({round(values[g], 6) for g in (0.10, 0.25, 0.40)}) == 1
    assert len({round(v, 2) for v in values.values()}) == 1


def test_reporting_the_cap_does_not_change_any_computed_number():
    """Sec.22a is observational -- per_share/equity/paths must be untouched."""
    result = rim_per_share(_BVE, _NI, _ROE, 0.25, _TG, _R, _SHARES, terminal_roe=_R)

    # bve0 + PV(RI) with a zero terminal (terminal_roe == discount_rate).
    assert result["tv"] == pytest.approx(0.0)
    assert result["equity"] == pytest.approx(_BVE + sum(
        ri / (1 + _R) ** (i + 1) for i, ri in enumerate(result["ri_path"])
    ))
    assert result["per_share"] == pytest.approx(result["equity"] / _SHARES)


# ---------------------------------------------------------------------------
# 22b. Externally-funded growth
# ---------------------------------------------------------------------------


def test_growth_is_exactly_value_neutral_when_roe_equals_cost_of_equity():
    """The theoretical anchor: at ROE == r every year's residual income is
    zero, so value is book value regardless of how fast the firm grows."""
    values = [
        rim_external_growth_per_share(_BVE, _NI, _R, g, _TG, _R, _SHARES, terminal_roe=_R)
        for g in (0.05, 0.25, 0.40)
    ]

    for result in values:
        assert result["per_share"] == pytest.approx(_BVE / _SHARES)
        assert result["value_gap_per_share"] == pytest.approx(0.0)


def test_growth_destroys_value_when_roe_is_below_cost_of_equity():
    result = rim_external_growth_per_share(
        _BVE, _NI, _ROE, 0.25, _TG, _R, _SHARES, terminal_roe=_R
    )

    assert result["per_share"] < result["per_share_internal"]
    assert result["value_gap_per_share"] < 0
    assert result["external_funding_total"] > 0


def test_growth_adds_value_when_roe_exceeds_cost_of_equity():
    result = rim_external_growth_per_share(
        _BVE, _NI, 0.18, 0.25, _TG, _R, _SHARES, terminal_roe=_R
    )

    assert result["per_share"] > result["per_share_internal"]
    assert result["value_gap_per_share"] > 0


def test_faster_growth_monotonically_worsens_a_sub_cost_of_equity_filer():
    values = [
        rim_external_growth_per_share(
            _BVE, _NI, _ROE, g, _TG, _R, _SHARES, terminal_roe=_R
        )["per_share"]
        for g in (0.05, 0.15, 0.25, 0.35)
    ]

    assert values == sorted(values, reverse=True)


def test_share_count_is_not_inflated_by_the_issuance():
    """Issuance is modeled at fair value, so the injected capital and the
    claim it buys cancel -- today's share count is the divisor."""
    result = rim_external_growth_per_share(
        _BVE, _NI, _ROE, 0.25, _TG, _R, _SHARES, terminal_roe=_R
    )

    assert result["effective_shares"] == pytest.approx(_SHARES)
    assert result["per_share"] == pytest.approx(result["equity"] / _SHARES)


def test_external_funding_is_zero_when_retention_covers_the_growth():
    result = rim_external_growth_per_share(
        _BVE, _NI, 0.18, 0.05, _TG, _R, _SHARES, terminal_roe=_R
    )

    assert result["external_funding_total"] == pytest.approx(0.0)
    # Nothing to fund externally, so the two paths coincide.
    assert result["value_gap_per_share"] == pytest.approx(0.0)


def test_external_funding_path_matches_the_book_growth_shortfall():
    g = 0.25
    result = rim_external_growth_per_share(
        _BVE, _NI, _ROE, g, _TG, _R, _SHARES, terminal_roe=_R
    )

    # Year 1: retention supplies roe * bve0, the rest is issued.
    assert result["external_funding_path"][0] == pytest.approx(_BVE * (g - _ROE))
    assert result["bve_path"][0] == pytest.approx(_BVE * (1 + g))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bve0": 0.0},
        {"bve0": -1.0},
        {"shares": 0.0},
        {"roe": 0.0},
        {"roe": -0.05},
        {"discount_rate": 0.04},   # not strictly above terminal_growth
    ],
)
def test_invalid_inputs_raise_rather_than_being_fixed(kwargs):
    args = dict(
        bve0=_BVE, ni0=_NI, roe=_ROE, growth_5y=0.25,
        terminal_growth=_TG, discount_rate=_R, shares=_SHARES,
    )
    args.update(kwargs)

    with pytest.raises(ValueError):
        rim_external_growth_per_share(**args)


def test_none_net_income_raises():
    with pytest.raises(ValueError):
        rim_external_growth_per_share(_BVE, None, _ROE, 0.25, _TG, _R, _SHARES)


# ---------------------------------------------------------------------------
# 22c. Engine disclosure
# ---------------------------------------------------------------------------


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(**series):
    annual = {name: [_rec(fy, v) for fy, v in sorted(values.items(), reverse=True)]
              for name, values in series.items()}
    return {
        "cik": 1, "entity_name": "Growth Cap Test Co", "currency": "USD",
        "annual": annual, "quarterly": {}, "missing": [], "matched_tags": {},
    }


def _assumptions(growth=0.25):
    return {
        "bear": {"growth_5y": growth - 0.05, "terminal_growth": _TG,
                 "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": growth, "terminal_growth": _TG,
                 "discount_rate": _R, "story": "Baz."},
        "bull": {"growth_5y": growth + 0.05, "terminal_growth": _TG,
                 "discount_rate": 0.09, "story": "Boğa."},
    }


def _run(growth=0.25, roe=_ROE):
    normalized = _normalized(StockholdersEquity={2025: _BVE}, NetIncome={2025: _NI})
    return run_valuation(
        normalized, [{"fy": 2025, "roe": roe}],
        {"shares": _SHARES, "latest_fy": 2025, "fcf": None, "net_debt": 0.0},
        price=15.25, price_df=None, assumptions=_assumptions(growth),
        sector_type="financial",
    )


def test_engine_flags_the_cap_and_builds_the_diagnostic():
    rim = _run()["rim"]

    assert rim["growth_capped"] is True
    assert rim["assumed_growth_5y"] == pytest.approx(0.25)
    assert rim["effective_growth_5y"] == pytest.approx(_ROE)
    assert rim["external_growth"] is not None
    assert rim["external_growth"]["value_gap_per_share"] < 0


def test_engine_note_names_both_rates_and_the_consequence():
    notes = _run()["notes"]

    note = next(n for n in notes if "içsel finansman kısıtına takıldı" in n)
    assert "%25.0" in note          # the assumption that was discarded
    assert "%4.6" in note           # what the model actually used
    assert "değer yaratmaz, yok eder" in note


def test_fair_value_range_growth_label_reports_the_effective_rate():
    fvr = _run()["fair_value_range"]

    assert fvr["base"]["growth"].startswith("%4.6 büyüme")
    assert "varsayılan %25.0 içsel finansman kısıtıyla sınırlandı" in fvr["base"]["growth"]
    # Every scenario row must say so -- the band varies only by discount rate.
    assert "sınırlandı" in fvr["bear"]["growth"]
    assert "sınırlandı" in fvr["bull"]["growth"]


def test_no_cap_no_diagnostic_and_a_plain_growth_label():
    # ROE 20% comfortably funds the 25% assumption's capped equivalent... use a
    # low growth so the cap genuinely doesn't bind.
    result = _run(growth=0.05, roe=0.20)
    rim = result["rim"]

    assert rim["growth_capped"] is False
    assert rim["external_growth"] is None
    assert not any("içsel finansman kısıtına takıldı" in n for n in result["notes"])
    assert result["fair_value_range"]["base"]["growth"] == "%5.0 büyüme (kazanç + sürdürülebilir büyüme)"


def test_the_disclosure_does_not_move_the_fair_value():
    """Sec.22 changes labels and adds advisory keys -- never a number."""
    fvr = _run()["fair_value_range"]
    direct = rim_per_share(_BVE, _NI, _ROE, 0.25, _TG, _R, _SHARES, terminal_roe=_R)

    assert _run()["rim"]["per_share"] == pytest.approx(round(direct["per_share"], 2))
    assert fvr["base"]["lo"] is not None and fvr["base"]["hi"] is not None


def test_rim_scenario_meta_degrades_on_missing_inputs():
    assert _rim_scenario_meta(None, _assumptions()) == {}
    assert _rim_scenario_meta({}, _assumptions()) == {}
    assert _rim_scenario_meta({"scenarios": {}, "roe": 0.1}, _assumptions()) == {}
    # Non-positive ROE makes the reinvestment identity meaningless.
    assert _rim_scenario_meta(
        {"scenarios": {"base": {"per_share": 5.0}}, "roe": 0.0}, _assumptions()
    ) == {}
