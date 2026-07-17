"""Hand-verified numeric tests for the cyclical sustainable-growth FCFE
anchor (SPEC.md Sec.8e, post financial-review Fix A/Fix B): ``valuation.dcf``'s
``fcfe_sustainable_growth_per_share`` and ``valuation.engine``'s
``_build_cyclical_fcfe``/``_cyclical_fcfe_scenario_band`` and their
``run_valuation`` wiring (the ``_fcf_dcf_unreliable`` gate + the
``cf_base_ps >= epv_base_ps`` beats-floor guardrail), plus ``triangulate``'s
confidence cap when the cyclical FCFE anchor becomes the headline.

IMPLEMENTATION NOTE (source of truth, read before trusting the delta spec):
the delta spec's "Fix C" (ROE denominator = average StockholdersEquity across
fiscal years) was proposed but later REVERTED in the shipped code --
``_build_cyclical_fcfe`` divides by the SPOT latest-FY ``StockholdersEquity``
(via ``resolve_fundamental_fy``), exactly like ``_build_earnings_power``'s own
advisory ROE check. Every test below asserts the SPOT-equity behavior, not
the delta spec's averaged-equity proposal. Fix A (terminal ROE fades to the
scenario's own cost of equity, passed as ``terminal_roe=discount_rate``) and
Fix B (growth capped at ROE: ``g_eff = min(g, roe)`` for every projection
year, including the terminal year) are both live and are asserted here.

Every numeric expectation is derived from the documented formulas (year-by-
year growth path via ``_year_growth_rate``, reinvestment ``b = g_eff/roe``,
FCFE discounting, Gordon-growth terminal value) and cross-checked with an
independent from-scratch scratch script (reimplementing those formulas, NOT
calling ``dcf.py``) before finalizing, following
``test_valuation_mature_revenue.py``'s own methodology for a 10-year
two-stage projection. See ``test_valuation_dcf.py``'s module docstring for
the general approach; see ``test_valuation_earnings_power.py`` and
``test_valuation_mature_revenue.py`` for the EPV-floor/beats-floor guardrail
pattern this anchor mirrors.
"""

import pytest

from sec_analyzer.valuation.dcf import fcfe_sustainable_growth_per_share
from sec_analyzer.valuation.engine import _build_cyclical_fcfe, run_valuation
from sec_analyzer.valuation.triangulate import triangulate

# ---------------------------------------------------------------------------
# 1. fcfe_sustainable_growth_per_share (pure function, SPEC.md Sec.8e)
# ---------------------------------------------------------------------------


def test_fcfe_sustainable_growth_normal_case_growth_below_roe_hand_verified():
    # ni0=100, roe=0.15, growth_5y=0.10, terminal_growth=0.02, r=0.09,
    # shares=10, dilution=0, terminal_roe=None (falls back to roe=0.15).
    #
    # growth_5y(0.10) < roe(0.15) for the ENTIRE horizon (years 1-10 fade
    # from 0.10 down to 0.02, always < 0.15) -> Fix B's g_eff=min(g,roe)
    # cap never binds anywhere in this case; g_eff == the raw
    # _year_growth_rate value every year.
    #
    # Years 1-5 (g=0.10 flat): b = 0.10/0.15 = 0.666667 every year.
    #   ni1 = 100*1.10   = 110.0        fcfe1 = 110.0    *(1-0.666667) = 36.666667
    #   ni2 = 110*1.10   = 121.0        fcfe2 = 121.0    *(1-0.666667) = 40.333333
    #   ni3 = 121*1.10   = 133.1        fcfe3 = 133.1    *(1-0.666667) = 44.366667
    #   ni4 = 133.1*1.10 = 146.41       fcfe4 = 146.41   *(1-0.666667) = 48.803333
    #   ni5 = 146.41*1.10= 161.051      fcfe5 = 161.051  *(1-0.666667) = 53.683667
    # Years 6-10 fade g linearly from 0.10 to 0.02 (g_y = 0.10 - 0.08*(y-5)/5):
    #   y6  g=0.084  ni6 =161.051*1.084 =174.579284  b=0.084/0.15=0.56     fcfe6 = 76.814885
    #   y7  g=0.068  ni7 =174.579284*1.068=186.450675 b=0.068/0.15=0.453333 fcfe7=101.926369
    #   y8  g=0.052  ni8 =186.450675*1.052=196.146110 b=0.052/0.15=0.346667 fcfe8=128.148792
    #   y9  g=0.036  ni9 =196.146110*1.036=203.207370 b=0.036/0.15=0.240000 fcfe9=154.437602
    #   y10 g=0.020  ni10=203.207370*1.020=207.271518 b=0.020/0.15=0.133333 fcfe10=179.635315
    # (== terminal_growth at y10, as the two-stage fade formula requires.)
    #
    # Discounting at r=0.09 (1.09^y for y=1..10) and summing:
    #   pv1..pv10 = 33.639144, 33.947760, 34.259207, 34.573512, 34.890700,
    #               45.802206, 55.757214, 64.313558, 71.107362, 75.879899
    #   pv_sum = 484.170561 (cross-checked via scratch script)
    #
    # Terminal (terminal_roe=None -> falls back to roe=0.15):
    #   g_t_eff = min(0.02, 0.15) = 0.02 (uncapped, well below roe)
    #   ni_terminal = ni10*(1.02) = 207.271518*1.02 = 211.416948
    #   b_t = 0.02/0.15 = 0.133333
    #   fcfe_terminal = 211.416948*(1-0.133333) = 183.228022
    #   tv = fcfe_terminal/(r-g_t) = 183.228022/0.07 = 2617.543168
    #   pv_tv = tv/1.09^10 = 2617.543168/2.367364 = 1105.678522
    #
    # equity = pv_sum + pv_tv = 484.170561 + 1105.678522 = 1589.849083
    # effective_shares = 10*(1+0)^5 = 10
    # per_share = 1589.849083/10 = 158.984908
    result = fcfe_sustainable_growth_per_share(
        ni0=100.0, roe=0.15, growth_5y=0.10, terminal_growth=0.02,
        discount_rate=0.09, shares=10.0, dilution_rate=0.0, terminal_roe=None,
    )

    assert len(result["ni_path"]) == 10
    assert len(result["fcfe_path"]) == 10
    assert result["ni_path"][0] == pytest.approx(110.0, rel=1e-6)
    assert result["fcfe_path"][0] == pytest.approx(36.666667, rel=1e-5)
    assert result["ni_path"][9] == pytest.approx(207.271518, rel=1e-6)
    assert result["fcfe_path"][9] == pytest.approx(179.635315, rel=1e-5)

    assert result["tv"] == pytest.approx(2617.543168, rel=1e-6)
    assert result["equity"] == pytest.approx(1589.849082, rel=1e-6)
    assert result["ev"] == result["equity"]  # FCFE-direct, no net-debt bridge
    assert result["effective_shares"] == pytest.approx(10.0)
    assert result["per_share"] == pytest.approx(158.984908, rel=1e-6)


def test_fcfe_sustainable_growth_terminal_roe_fade_lowers_value_vs_none():
    # Same inputs as the normal case above, but compare terminal_roe=None
    # (falls back to roe=0.15, perpetual excess return) against
    # terminal_roe=0.09 (the scenario's own cost of equity -- Fix A's
    # Damodaran stable-phase convention: terminal ROE fades to the cost of
    # equity). Years 1-10 are UNCHANGED (terminal_roe only affects the
    # terminal-year reinvestment), so pv_sum is identical in both cases;
    # only the terminal value differs:
    #
    # terminal_roe=None -> troe_resolved=0.15 (as derived above):
    #   b_t=0.133333, fcfe_terminal=183.228022, tv=2617.543168,
    #   pv_tv=1105.678522 -> per_share = 158.984908 (from the test above).
    #
    # terminal_roe=0.09 -> troe_resolved=0.09:
    #   g_t_eff = min(0.02, 0.09) = 0.02 (still uncapped, below 0.09 too)
    #   b_t = 0.02/0.09 = 0.222222 (HIGHER than 0.133333 -- less of
    #     ni_terminal is distributed, since the lower terminal ROE means the
    #     firm must retain more to fund the same terminal growth rate)
    #   ni_terminal = 211.416948 (same as above, ni_path unaffected)
    #   fcfe_terminal = 211.416948*(1-0.222222) = 164.435404
    #   tv = 164.435404/0.07 = 2349.077202
    #   pv_tv = 2349.077202/2.367364 = 992.275596
    #   equity = 484.170561 + 992.275596 = 1476.446157
    #   per_share = 1476.446157/10 = 147.644616
    #
    # 147.644616 < 158.984908: fading the terminal ROE down to the cost of
    # equity strictly LOWERS the anchor (matches the delta spec's expected
    # direction for MU: base ~$89 -> ~$84).
    kwargs = dict(
        ni0=100.0, roe=0.15, growth_5y=0.10, terminal_growth=0.02,
        discount_rate=0.09, shares=10.0, dilution_rate=0.0,
    )
    result_none = fcfe_sustainable_growth_per_share(**kwargs, terminal_roe=None)
    result_fade = fcfe_sustainable_growth_per_share(**kwargs, terminal_roe=0.09)

    assert result_none["per_share"] == pytest.approx(158.984908, rel=1e-6)
    assert result_fade["per_share"] == pytest.approx(147.644616, rel=1e-6)
    assert result_fade["per_share"] < result_none["per_share"]

    # Years 1-10 (ni_path/fcfe_path) are untouched by terminal_roe -- only
    # the terminal value differs.
    assert result_fade["ni_path"] == result_none["ni_path"]
    assert result_fade["fcfe_path"] == result_none["fcfe_path"]
    assert result_fade["tv"] < result_none["tv"]


def test_fcfe_sustainable_growth_capped_at_roe_ceiling_hand_verified():
    # Fix B: growth is capped at ROE in EVERY projection year (including the
    # terminal year) -- "g_eff = min(g_year, roe)" both compounds earnings
    # AND drives the reinvestment rate b=g_eff/roe, so b never exceeds 1.0
    # (payout never goes negative) and a raw growth assumption above ROE
    # adds NO extra value beyond what growth==ROE would already produce.
    #
    # Construction: roe=0.10, terminal_growth=0.099 (just BELOW roe, so the
    # y10 point -- which always equals terminal_growth exactly, per
    # _year_growth_rate's formula -- is never capped in either case),
    # discount_rate=0.15 (same in both cases; only Gordon's r>g_t matters,
    # not growth_5y, so this can stay fixed).
    #
    # Case 1: growth_5y=0.30. Fade slope = (0.099-0.30)/5 = -0.0402/yr:
    #   y6=0.2598, y7=0.2196, y8=0.1794, y9=0.1392, y10=0.099
    # Case 2: growth_5y=0.50. Fade slope = (0.099-0.50)/5 = -0.0802/yr:
    #   y6=0.4198, y7=0.3396, y8=0.2594, y9=0.1792, y10=0.099
    #
    # In BOTH cases: y1-5 (0.30 and 0.50) and y6-9 (all values above) are
    # > roe(0.10) -> capped to g_eff=0.10 in every one of those 9 years,
    # identically in both cases. y10 (=0.099 in both, since it always
    # equals terminal_growth exactly) is < roe -> uncapped, identically
    # 0.099 in both cases too. So ni_path/fcfe_path (and therefore
    # per_share) must come out EXACTLY equal despite the raw growth_5y
    # differing by nearly 2x -- this is the "ceiling" behavior: value
    # cannot exceed what the ROE-funded growth path produces.
    #
    # (Cross-checked via scratch script: both cases give per_share =
    # 1.444379, equal to within float precision.)
    case1 = fcfe_sustainable_growth_per_share(
        ni0=100.0, roe=0.10, growth_5y=0.30, terminal_growth=0.099,
        discount_rate=0.15, shares=10.0, dilution_rate=0.0, terminal_roe=None,
    )
    case2 = fcfe_sustainable_growth_per_share(
        ni0=100.0, roe=0.10, growth_5y=0.50, terminal_growth=0.099,
        discount_rate=0.15, shares=10.0, dilution_rate=0.0, terminal_roe=None,
    )

    assert case1["per_share"] == pytest.approx(1.444379, rel=1e-5)
    assert case2["per_share"] == pytest.approx(case1["per_share"], rel=1e-9)
    assert case1["ni_path"] == pytest.approx(case2["ni_path"], rel=1e-9)

    # Every one of years 1-9's effective growth was capped to roe exactly
    # (ni grows at a flat 10% for 9 straight years despite the much higher
    # raw growth_5y assumptions): ni1 = 100*1.10 = 110.0; ni9 = ni5 grown at
    # 10% for 4 more years = 161.051*1.10^4 = 235.794769.
    assert case1["ni_path"][0] == pytest.approx(110.0, rel=1e-6)
    assert case1["ni_path"][8] == pytest.approx(235.794769, rel=1e-6)
    # Reinvestment b=g_eff/roe=0.10/0.10=1.0 for years 1-9 -> FCFE is
    # (near-)zero those years (payout never goes negative, per Fix B).
    assert case1["fcfe_path"][0] == pytest.approx(0.0, abs=1e-9)
    assert case1["fcfe_path"][7] == pytest.approx(0.0, abs=1e-9)
    # y10 uncapped (g_eff=0.099<roe): b=0.099/0.10=0.99, a small positive payout.
    assert case1["fcfe_path"][9] > 0


def test_fcfe_sustainable_growth_raises_on_invalid_inputs():
    valid = dict(
        ni0=100.0, roe=0.15, growth_5y=0.10, terminal_growth=0.02,
        discount_rate=0.09, shares=10.0,
    )

    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "ni0": None})

    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "shares": 0})
    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "shares": -5.0})
    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "shares": None})

    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "roe": 0.0})
    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "roe": -0.05})

    # discount_rate <= terminal_growth: Gordon undefined (strictly below AND
    # exactly equal both raise).
    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "discount_rate": 0.02})
    with pytest.raises(ValueError):
        fcfe_sustainable_growth_per_share(**{**valid, "discount_rate": 0.01})


# ---------------------------------------------------------------------------
# Fixture helpers for _build_cyclical_fcfe / run_valuation (mirrors
# test_valuation_mature_revenue.py's _annual/_normalized/_rec conventions).
# ---------------------------------------------------------------------------


def _annual(**concepts) -> dict:
    """Minimal ``normalized``-shaped dict for direct unit calls into
    ``_build_cyclical_fcfe``: ``_annual(StockholdersEquity={2023: 500.0})``
    -> ``{"annual": {"StockholdersEquity": [{"fy": 2023, "value": 500.0}]}}``."""
    return {
        "annual": {
            concept: [{"fy": fy, "value": value} for fy, value in by_fy.items()]
            for concept, by_fy in concepts.items()
        }
    }


_CYCLICAL_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity",
]


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(overrides: "dict[str, dict[int, float]]") -> dict:
    """Fuller ``normalized``-shaped fixture (mirrors
    ``test_valuation_earnings_power.py``'s ``_normalized_epv``) for full
    ``run_valuation`` integration tests: ``overrides`` is ``{concept: {fy:
    value}}``."""
    annual = {
        concept: [_rec(fy, value) for fy, value in (overrides.get(concept) or {}).items()] or None
        for concept in _CYCLICAL_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "Cyclical FCFE Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CYCLICAL_CONCEPTS},
        "missing": [c for c in _CYCLICAL_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CYCLICAL_CONCEPTS},
    }


def _cyclical_assumptions():
    """bear/base/bull assumptions shared by the ``_build_cyclical_fcfe``
    unit tests below (ni_norm=100, roe=0.15 -> equity=666.666...667)."""
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.01, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.02, "discount_rate": 0.09, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.08, "story": "Boğa."},
    }


# ---------------------------------------------------------------------------
# 2. _build_cyclical_fcfe (unit, SPEC.md Sec.8e)
# ---------------------------------------------------------------------------


def test_build_cyclical_fcfe_none_when_earnings_power_missing_or_incomplete():
    normalized = _annual(StockholdersEquity={2023: 500.0})
    metrics = {"latest_fy": 2023}
    assumptions = _cyclical_assumptions()

    # earnings_power is None entirely -- no note appended for this branch
    # (per the actual implementation: `if not earnings_power or ...: return
    # None, notes` where notes is still the empty list at that point).
    detail, notes = _build_cyclical_fcfe(assumptions, None, normalized, metrics, shares=10.0, dilution_rate=0.0)
    assert detail is None
    assert notes == []

    # earnings_power present but missing "cost_of_equity".
    detail2, notes2 = _build_cyclical_fcfe(
        assumptions, {"normalized_net_income": 100.0}, normalized, metrics, shares=10.0, dilution_rate=0.0
    )
    assert detail2 is None
    assert notes2 == []

    # earnings_power present but missing "normalized_net_income".
    detail3, notes3 = _build_cyclical_fcfe(
        assumptions, {"cost_of_equity": 0.09}, normalized, metrics, shares=10.0, dilution_rate=0.0
    )
    assert detail3 is None
    assert notes3 == []


def test_build_cyclical_fcfe_none_without_valid_shares():
    normalized = _annual(StockholdersEquity={2023: 500.0})
    metrics = {"latest_fy": 2023}
    assumptions = _cyclical_assumptions()
    earnings_power = {"normalized_net_income": 100.0, "cost_of_equity": 0.09}

    for bad_shares in (None, 0.0, -5.0):
        detail, notes = _build_cyclical_fcfe(
            assumptions, earnings_power, normalized, metrics, shares=bad_shares, dilution_rate=0.0
        )
        assert detail is None
        assert notes == []


def test_build_cyclical_fcfe_none_when_equity_missing_or_nonpositive():
    assumptions = _cyclical_assumptions()
    earnings_power = {"normalized_net_income": 100.0, "cost_of_equity": 0.09}
    metrics = {"latest_fy": 2023}

    # No StockholdersEquity at all.
    detail, notes = _build_cyclical_fcfe(
        assumptions, earnings_power, _annual(), metrics, shares=10.0, dilution_rate=0.0
    )
    assert detail is None
    assert any("özkaynak verisi eksik/negatif" in n for n in notes)

    # StockholdersEquity present but zero/negative.
    for bad_equity in (0.0, -50.0):
        normalized = _annual(StockholdersEquity={2023: bad_equity})
        detail2, notes2 = _build_cyclical_fcfe(
            assumptions, earnings_power, normalized, metrics, shares=10.0, dilution_rate=0.0
        )
        assert detail2 is None
        assert any("özkaynak verisi eksik/negatif" in n for n in notes2)


def test_build_cyclical_fcfe_none_when_roe_not_positive():
    # normalized_net_income <= 0 -> roe = ni_norm/equity <= 0 -> refused,
    # distinct note from the missing-equity case.
    assumptions = _cyclical_assumptions()
    earnings_power = {"normalized_net_income": -100.0, "cost_of_equity": 0.09}
    normalized = _annual(StockholdersEquity={2023: 500.0})
    metrics = {"latest_fy": 2023}

    detail, notes = _build_cyclical_fcfe(
        assumptions, earnings_power, normalized, metrics, shares=10.0, dilution_rate=0.0
    )
    assert detail is None
    assert any("ROE pozitif değil" in n for n in notes)


def test_build_cyclical_fcfe_builds_all_three_scenarios_with_bands_hand_verified():
    # ni_norm=100, equity=666.666...667 -> roe = 100/666.6666... = 0.15
    # exactly (matches Part 1's hand-verified roe=0.15 case): each
    # scenario's own discount_rate is ALSO passed through as terminal_roe
    # (Sec.8e addendum/Fix A), so this reuses this file's Part-1-verified
    # fcfe_sustainable_growth_per_share arithmetic directly:
    #   bear (g5=0.05, gt=0.01, r=0.12, terminal_roe=0.12): per_share=90.18
    #   base (g5=0.10, gt=0.02, r=0.09, terminal_roe=0.09): per_share=147.64
    #     (== the terminal-ROE-fade case hand-verified in Part 1 above)
    #   bull (g5=0.15, gt=0.03, r=0.08, terminal_roe=0.08): per_share=208.01
    # (Cross-checked via the same scratch script as Part 1.)
    #
    # Bands (_cyclical_fcfe_scenario_band: discount_rate +/- 0.01, growth_5y/
    # terminal_growth held fixed, terminal_roe=that nearby rate each time):
    #   bear: cells at r={0.11,0.12,0.13} -> lo=81.79, hi=100.22
    #   base: cells at r={0.08,0.09,0.10} -> lo=127.20, hi=173.78
    #   bull: cells at r={0.07,0.08,0.09} -> lo=172.44, hi=255.13
    #
    # reinvestment_base = min(base growth_5y=0.10, roe=0.15)/roe = 0.10/0.15
    #   = 0.6667 (rounded to 4dp, for display).
    equity = 100.0 / 0.15  # 666.666...667, ni_norm/equity == 0.15 exactly by construction
    normalized = _annual(StockholdersEquity={2023: equity})
    metrics = {"latest_fy": 2023}
    assumptions = _cyclical_assumptions()
    earnings_power = {"normalized_net_income": 100.0, "cost_of_equity": 0.09}

    detail, notes = _build_cyclical_fcfe(
        assumptions, earnings_power, normalized, metrics, shares=10.0, dilution_rate=0.0
    )

    assert detail is not None
    assert detail["normalized_net_income"] == pytest.approx(100.0)
    assert detail["roe"] == pytest.approx(0.15, abs=1e-4)
    assert detail["equity"] == pytest.approx(equity)
    assert detail["cost_of_equity"] == pytest.approx(0.09)
    assert detail["reinvestment_base"] == pytest.approx(0.6667, abs=1e-4)

    bear = detail["scenarios"]["bear"]
    assert bear["per_share"] == pytest.approx(90.18, abs=0.02)
    assert bear["lo"] == pytest.approx(81.79, abs=0.02)
    assert bear["hi"] == pytest.approx(100.22, abs=0.02)

    base = detail["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(147.64, abs=0.02)
    assert base["lo"] == pytest.approx(127.20, abs=0.02)
    assert base["hi"] == pytest.approx(173.78, abs=0.02)
    assert detail["per_share"] == base["per_share"]  # top-level mirrors base

    bull = detail["scenarios"]["bull"]
    assert bull["per_share"] == pytest.approx(208.01, abs=0.02)
    assert bull["lo"] == pytest.approx(172.44, abs=0.02)
    assert bull["hi"] == pytest.approx(255.13, abs=0.02)

    assert notes == []  # every scenario built cleanly, nothing to flag


def test_build_cyclical_fcfe_skips_a_scenario_with_discount_rate_at_or_below_terminal_growth():
    # bear/base valid (same as the full-build test above); bull's
    # discount_rate (0.02) is <= its own terminal_growth (0.03) -- Gordon
    # undefined -- so ONLY the bull scenario degrades to a None cell with a
    # note, while bear/base still build normally (mirrors
    # test_build_mature_revenue_dcf's/_build_dcf_scenarios' per-scenario
    # isolation).
    equity = 100.0 / 0.15
    normalized = _annual(StockholdersEquity={2023: equity})
    metrics = {"latest_fy": 2023}
    assumptions = _cyclical_assumptions()
    assumptions["bull"] = {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.02, "story": "Boğa."}
    earnings_power = {"normalized_net_income": 100.0, "cost_of_equity": 0.09}

    detail, notes = _build_cyclical_fcfe(
        assumptions, earnings_power, normalized, metrics, shares=10.0, dilution_rate=0.0
    )

    assert detail is not None  # bear/base still succeeded -> detail is built
    assert detail["scenarios"]["bull"] == {"per_share": None, "lo": None, "hi": None}
    assert any("Bull senaryosu için döngüsel FCFE varsayımları eksik veya geçersiz" in n for n in notes)

    # bear/base unaffected by bull's failure.
    assert detail["scenarios"]["bear"]["per_share"] == pytest.approx(90.18, abs=0.02)
    assert detail["scenarios"]["base"]["per_share"] == pytest.approx(147.64, abs=0.02)
    assert detail["per_share"] == detail["scenarios"]["base"]["per_share"]


# ---------------------------------------------------------------------------
# 3. run_valuation integration (SPEC.md Sec.8e): the _fcf_dcf_unreliable
#    gate + the cf_base_ps >= epv_base_ps beats-floor guardrail.
# ---------------------------------------------------------------------------

_CYCLICAL_RATIOS_SUPPRESSED = [{"fy": 2023, "fcf": 5.0}]
_CYCLICAL_METRICS_SUPPRESSED = {
    "shares": 10.0, "latest_fy": 2023, "fcf": 5.0, "net_debt": 0.0,
}


def _cyclical_run_assumptions():
    return {
        "bear": {"growth_5y": 0.04, "terminal_growth": 0.01, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.08, "terminal_growth": 0.02, "discount_rate": 0.09, "story": "Baz."},
        "bull": {"growth_5y": 0.12, "terminal_growth": 0.03, "discount_rate": 0.08, "story": "Boğa."},
    }


def test_run_valuation_cyclical_capex_suppressed_profitable_triggers_fcfe_headline():
    # MU-shaped fixture: single-FY (fy=2023) NI=100, Revenue=1000 (margin
    # 0.10; sanity guard trivially inactive, same one-year-fixture trick as
    # test_valuation_earnings_power.py) -> EPV base_value_per_share =
    # 100/0.09/10 = 111.111... -> base per_share=111.11 (dr_base=0.09 is
    # assumptions["base"]["discount_rate"]).
    #
    # StockholdersEquity=500 -> roe = 100/500 = 0.20 (> discount_rate=0.09
    # for every scenario -> growth genuinely adds value, per the docstring's
    # roe>r condition).
    #
    # Raw FCF-DCF deliberately capex-suppressed: fcf0=5.0 (100x below NI).
    # Cash-conversion guard: OCF=90 >= 0.8*NI(100)=80 -> cash_backed=True.
    # Investment-driven: CapEx=70, capex/ocf=70/90=0.778>=0.5 -> True.
    # Raw dcf base band hi (3x3 sensitivity grid, cross-checked via scratch
    # script) = 13.42 < 0.5*111.11=55.56 -> fcf_suppressed=True.
    # -> _fcf_dcf_unreliable's gate fires (all three conditions hold).
    #
    # Sustainable-growth FCFE (ni_norm=100, roe=0.20, terminal_roe=
    # discount_rate per scenario -- cross-checked via scratch script):
    #   bear (g5=0.04,gt=0.01,r=0.12): per_share=92.63, band=(84.52,102.31)
    #   base (g5=0.08,gt=0.02,r=0.09): per_share=149.67, band=(130.96,173.42)
    #   bull (g5=0.12,gt=0.03,r=0.08): per_share=206.53, band=(175.65,247.07)
    # base FCFE (149.67) >= EPV base (111.11) -> cf_beats_floor=True ->
    # cyclical_fcfe_headline=True, fair_value_range.base == the FCFE band,
    # NOT the (suppressed) raw dcf band and NOT the normalized_variant.
    normalized = _normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 70.0},
        "StockholdersEquity": {2023: 500.0},
    })
    assumptions = _cyclical_run_assumptions()

    result = run_valuation(
        normalized, _CYCLICAL_RATIOS_SUPPRESSED, _CYCLICAL_METRICS_SUPPRESSED,
        price=None, price_df=None, assumptions=assumptions, sector_type="cyclical",
    )

    # Fixture preconditions.
    assert result["earnings_power"] is not None
    assert result["earnings_power"]["scenarios"]["base"]["per_share"] == pytest.approx(111.11, abs=0.01)
    assert result["dcf"]["scenarios"]["base"]["hi"] < 55.56

    assert result["cyclical_fcfe_headline"] is True
    assert result["earnings_power_headline"] is False
    detail = result["cyclical_fcfe_detail"]
    assert detail is not None
    assert detail["roe"] == pytest.approx(0.20, abs=1e-4)
    assert detail["reinvestment_base"] == pytest.approx(0.4, abs=1e-4)  # min(0.08,0.20)/0.20

    base = detail["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(149.67, abs=0.02)
    assert base["per_share"] > result["earnings_power"]["scenarios"]["base"]["per_share"]

    # Headline fair_value_range must be the FCFE band, not EPV or the raw/
    # normalized FCF-DCF bands.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(130.96, abs=0.02)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(173.42, abs=0.02)

    assert any("sürdürülebilir-büyüme" in n and "FCFE çapasına dayandırıldı" in n for n in result["notes"])
    # Fix D's corrected disclosure note + Fix E's trough-excluded transparency note.
    assert any("Duyarlılık tablosu döngü-ortası normalize FCF-DCF tabanını" in n for n in result["notes"])
    assert any("DIŞLAR" in n and "yapısal re-rating" in n for n in result["notes"])


def test_run_valuation_cyclical_fcfe_below_epv_floor_keeps_epv_headline():
    # SAME suppressed-FCF/cash-backed/investment-driven preconditions as the
    # test above (identical NI/Revenue/OCF/CapEx/fcf0 -> EPV base=111.11,
    # gate still fires identically), but StockholdersEquity=2000 this time
    # -> roe = 100/2000 = 0.05, BELOW every scenario's discount_rate (0.08-
    # 0.12) -> growth genuinely DESTROYS value relative to the zero-growth
    # EPV floor (the textbook roe<r case the dcf.py docstring calls out).
    #
    # Sustainable-growth FCFE (cross-checked via scratch script):
    #   base (g5=0.08,gt=0.02,r=0.09,terminal_roe=0.09): per_share=81.35
    # 81.35 < EPV base(111.11) -> cf_beats_floor=False -> the FCFE anchor is
    # demoted to a secondary cross-check (still reported, never silently
    # dropped) and the headline falls back to the zero-growth EPV floor --
    # strictly better than the capex-suppressed raw FCF-DCF, per spec.
    normalized = _normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 70.0},
        "StockholdersEquity": {2023: 2000.0},
    })
    assumptions = _cyclical_run_assumptions()

    result = run_valuation(
        normalized, _CYCLICAL_RATIOS_SUPPRESSED, _CYCLICAL_METRICS_SUPPRESSED,
        price=None, price_df=None, assumptions=assumptions, sector_type="cyclical",
    )

    assert result["earnings_power"]["scenarios"]["base"]["per_share"] == pytest.approx(111.11, abs=0.01)

    assert result["cyclical_fcfe_headline"] is False
    assert result["earnings_power_headline"] is True

    # The FCFE detail must STILL be present (secondary cross-check), never
    # silently dropped just because it lost the guardrail comparison.
    detail = result["cyclical_fcfe_detail"]
    assert detail is not None
    assert detail["roe"] == pytest.approx(0.05, abs=1e-4)
    base = detail["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(81.35, abs=0.02)
    assert base["per_share"] < result["earnings_power"]["scenarios"]["base"]["per_share"]

    # Headline fair_value_range must be the EPV band, NOT the FCFE band.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["lo"]
    )
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["hi"]
    )
    assert result["fair_value_range"]["base"]["lo"] != pytest.approx(81.35, abs=0.02)

    assert any(
        "kazanç gücünü yansıtmıyor" in n and "kazanç-gücü (EPV) çapasına dayandırıldı" in n
        for n in result["notes"]
    )


def test_run_valuation_cyclical_fcf_not_suppressed_keeps_normalized_variant_headline():
    # A cyclical filer whose FCF is healthy (NOT capex-suppressed): 3 fiscal
    # years (2021-2023) of Revenue=1000/OCF=200/CapEx=100 (margin=(200-100)/
    # 1000=0.10 every year, no SBC) -> _normalized_fcf0's top-half average
    # margin = 0.10 -> normalized_fcf0 = 0.10*1000 = 100.0.
    #
    # Raw fcf0 (ttm, via ratios/metrics) = 80.0 -- deliberately DIFFERENT
    # from normalized_fcf0(100.0) so the assertions below can distinguish
    # "headline came from normalized_variant" from "headline came from the
    # raw dcf_scenarios" (they would otherwise coincide and the test would
    # be vacuous).
    #
    # EPV base (NI=100, dr_base=0.09, shares=10) = 111.11 (same as the other
    # two tests above). Raw dcf (fcf0=80.0) base per_share=163.00, band
    # (3x3 grid, cross-checked via scratch script) = (126.94, 214.64) --
    # hi(214.64) is NOT < 0.5*111.11=55.56 -> fcf_suppressed=False ->
    # _fcf_dcf_unreliable returns (False, None) regardless of cash-backed/
    # investment-driven -- the gate never even reaches the FCFE anchor.
    #
    # Falls to the pre-existing "elif normalized_variant is not None"
    # branch: normalized_variant (fcf0=100.0) base per_share=203.75, band
    # (cross-checked) = (158.67, 268.30) -- the EXISTING cycle-mid
    # normalized-FCF-DCF headline, byte-for-byte unchanged behavior; the
    # new cyclical_fcfe_headline machinery must never even fire here.
    normalized = _normalized({
        "NetIncome": {2023: 100.0},
        "Revenue": {2023: 1000.0, 2022: 1000.0, 2021: 1000.0},
        "OperatingCashFlow": {2023: 200.0, 2022: 200.0, 2021: 200.0},
        "CapEx": {2023: 100.0, 2022: 100.0, 2021: 100.0},
    })
    ratios = [{"fy": 2023, "fcf": 80.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 80.0, "net_debt": 0.0}
    assumptions = _cyclical_run_assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="cyclical",
    )

    assert result["earnings_power"]["scenarios"]["base"]["per_share"] == pytest.approx(111.11, abs=0.01)
    assert result["dcf"]["scenarios"]["base"]["per_share"] == pytest.approx(163.0, abs=0.02)
    assert result["dcf"]["scenarios"]["base"]["hi"] == pytest.approx(214.64, abs=0.02)

    assert result["cyclical_fcfe_headline"] is False
    assert result["earnings_power_headline"] is False
    assert result["cyclical_fcfe_detail"] is None  # gate never fired -> never built

    normalized_variant = result["dcf"]["normalized_variant"]
    assert normalized_variant is not None
    assert normalized_variant["base"]["per_share"] == pytest.approx(203.75, abs=0.02)

    # Headline fair_value_range must be the normalized_variant band, NOT
    # the raw dcf_scenarios band (they differ here by construction).
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(158.67, abs=0.02)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(268.30, abs=0.02)
    assert result["fair_value_range"]["base"]["lo"] != pytest.approx(
        result["dcf"]["scenarios"]["base"]["lo"], abs=0.02
    )

    assert any("döngü-ortası normalize edilmiş FCF'e dayandırıldı" in n for n in result["notes"])


# ---------------------------------------------------------------------------
# 4. triangulate -- cyclical_fcfe_headline confidence cap (mirrors
#    earnings_power_headline's/mature_revenue_headline's own cap; see
#    test_valuation_earnings_power.py's Part D for the identical-shape test).
# ---------------------------------------------------------------------------


def test_triangulate_cyclical_fcfe_headline_caps_high_confidence_to_medium():
    # Same three-way "ucuz" agreement fixture used by every other headline-
    # cap test in this codebase (dcf: price=90 < band.lo=100; reverse_dcf:
    # implied=0.05 < ref(0.10)-0.03=0.07; multiples: pe_pct=20 < 30) --
    # would normally be CONFIDENCE_HIGH. With cyclical_fcfe_headline=True,
    # confidence must be capped to ORTA (not YÜKSEK): DCF and multiples both
    # ultimately reflect the (correctly) suppressed FCF in this mode, so
    # three-way agreement is weaker evidence than usual.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="cyclical", cyclical_fcfe_headline=True,
    )
    assert result["signals"] == {"dcf": "ucuz", "reverse_dcf": "ucuz", "multiples": "ucuz"}
    assert result["confidence"] == "ORTA"
    assert result["direction"] == "ucuz"
    assert "kazanç-tabanlı FCFE çapasından" in result["rationale"]["confidence"]
    assert "ORTA'ya sınırlandı" in result["rationale"]["confidence"]


def test_triangulate_cyclical_fcfe_headline_false_preserves_high_confidence():
    # Regression: cyclical_fcfe_headline defaults to False, so the exact
    # same three-way agreement stays CONFIDENCE_HIGH, unchanged.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="cyclical",
    )
    assert result["confidence"] == "YÜKSEK"
    assert "ORTA'ya sınırlandı" not in result["rationale"]["confidence"]


def test_triangulate_cyclical_fcfe_headline_does_not_alter_already_medium_or_low_confidence():
    # The cap only ever applies when confidence would otherwise have been
    # HIGH; a 2-of-3 (ORTA) result must come out exactly as it would without
    # cyclical_fcfe_headline -- no further downgrade, no rationale suffix.
    two_of_three = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=80, ps_pct=None, pfcf_pct=None,
        sector_type="cyclical", cyclical_fcfe_headline=True,
    )
    assert two_of_three["confidence"] == "ORTA"
    assert "ORTA'ya sınırlandı" not in two_of_three["rationale"]["confidence"]

    scattered = triangulate(
        price=110, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.15,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="cyclical", cyclical_fcfe_headline=True,
    )
    assert scattered["confidence"] == "DÜŞÜK"
    assert "ORTA'ya sınırlandı" not in scattered["rationale"]["confidence"]
