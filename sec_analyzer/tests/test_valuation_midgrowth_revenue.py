"""Hand-verified numeric tests for the mid-growth, loss-making revenue-first
DCF (SPEC.md Sec.8d / Roadmap Madde 2) -- ``valuation.engine``'s
``_build_midgrowth_revenue_dcf`` and its ``run_valuation`` wiring, plus
``triangulate``'s confidence cap when the mid-growth revenue-first DCF
becomes the headline.

This is a revenue-first alternative to a multiples-only headline for
``growth_unprofitable`` filers growing the top line at a real but sub-hyper
rate (realized CAGR roughly 12-20%) that ``sector.detect_hyper_grower``
(needs > 20%) does not pick up. It reuses the mature path's
``_mature_start_growth``/``_mature_current_margin`` helpers (already
hand-verified in ``test_valuation_mature_revenue.py``) and the hyper path's
``_hyper_target_base``/``_hyper_scenario_band``, so this module does not
re-derive THEIR arithmetic; it hand-verifies the mid-growth-specific
wiring: the 20% target-margin cap, the 8-year fade horizon, the 12%
growth gate, the suppression guardrail, and the ``run_valuation``/
``triangulate`` integration.

See ``test_valuation_mature_revenue.py``'s module docstring/style: each
per-share/lo/hi expectation below was independently cross-checked with a
from-scratch scratch re-implementation of ``revenue_dcf.revenue_first_dcf``'s
documented growth-path/margin-path/PV/terminal-value formulas (NOT calling
the function itself) before being hardcoded as a ``pytest.approx``
expectation.
"""

import pytest

from sec_analyzer.valuation.engine import _build_midgrowth_revenue_dcf, run_valuation
from sec_analyzer.valuation.triangulate import triangulate

# ---------------------------------------------------------------------------
# Fixture helpers (mirrors test_valuation_mature_revenue.py's _annual/_rec/_normalized)
# ---------------------------------------------------------------------------


def _annual(**concepts) -> dict:
    return {
        "annual": {
            concept: [{"fy": fy, "value": value} for fy, value in by_fy.items()]
            for concept, by_fy in concepts.items()
        }
    }


_MIDGROWTH_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity", "Depreciation",
]


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(overrides: "dict[str, dict[int, float]]") -> dict:
    annual = {
        concept: [_rec(fy, value) for fy, value in (overrides.get(concept) or {}).items()] or None
        for concept in _MIDGROWTH_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "Midgrowth Revenue Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _MIDGROWTH_CONCEPTS},
        "missing": [c for c in _MIDGROWTH_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _MIDGROWTH_CONCEPTS},
    }


def _midgrowth_assumptions(discount_rate=0.15, terminal_growth=0.03):
    """Bear/base/bull with identical dr/tg across scenarios (a valid special
    case, mirrors ``test_valuation_mature_revenue.py``'s ``_mature_assumptions``
    -- ``_build_midgrowth_revenue_dcf`` only reads ``discount_rate``/
    ``terminal_growth`` per scenario; the growth story itself comes from
    realized data). ``growth_5y`` is a small increasing sequence (only
    consulted by the standard FCF-DCF leg/triangulation, never by this
    method) so the bear<=base<=bull ordering check never fires a note."""
    return {
        "bear": {"growth_5y": terminal_growth + 0.01, "terminal_growth": terminal_growth,
                  "discount_rate": discount_rate, "story": "Ayı."},
        "base": {"growth_5y": terminal_growth + 0.02, "terminal_growth": terminal_growth,
                  "discount_rate": discount_rate, "story": "Baz."},
        "bull": {"growth_5y": terminal_growth + 0.03, "terminal_growth": terminal_growth,
                  "discount_rate": discount_rate, "story": "Boğa."},
    }


# ---------------------------------------------------------------------------
# A. _build_midgrowth_revenue_dcf (unit) -- growth gate + full scenario build.
# ---------------------------------------------------------------------------


def test_growth_gate_rejects_below_12pct_min_growth():
    # realized CAGR = 0.08 < _MIDGROWTH_MIN_GROWTH (0.12) -> gate rejects
    # outright -> (None, note), regardless of terminal growth.
    normalized = _annual(Revenue={2023: 1000.0})
    metrics = {"revenue_cagr_5y": 0.08, "revenue_cagr_3y": None, "latest_fy": 2023, "shares": 100.0}
    assumptions = _midgrowth_assumptions(discount_rate=0.15, terminal_growth=0.03)

    detail, notes = _build_midgrowth_revenue_dcf(
        assumptions, normalized, metrics, ratios=[], price=None, shares=100.0
    )

    assert detail is None
    assert any("yetersiz" in n for n in notes)
    assert any("%8.0" in n for n in notes)


def test_growth_gate_rejects_at_or_below_terminal_growth():
    # realized CAGR = 0.12 -- NOT < the 12% floor (0.12<0.12 is False), so
    # the first clause doesn't fire. But terminal_growth is set to 0.15 here
    # (deliberately above the realized growth) -> the SECOND clause
    # (start_growth <= terminal_growth, "nothing left to fade") rejects
    # instead -> (None, note). Isolates the second clause from the first.
    normalized = _annual(Revenue={2023: 1000.0})
    metrics = {"revenue_cagr_5y": 0.12, "revenue_cagr_3y": None, "latest_fy": 2023, "shares": 100.0}
    assumptions = _midgrowth_assumptions(discount_rate=0.20, terminal_growth=0.15)

    detail, notes = _build_midgrowth_revenue_dcf(
        assumptions, normalized, metrics, ratios=[], price=None, shares=100.0
    )

    assert detail is None
    assert any("yetersiz" in n for n in notes)
    assert any("%12.0" in n for n in notes)


def test_none_without_valid_shares_or_revenue():
    metrics = {"revenue_cagr_5y": 0.15, "latest_fy": 2023, "shares": 100.0}
    assumptions = _midgrowth_assumptions()

    detail, notes = _build_midgrowth_revenue_dcf(
        assumptions, _annual(), metrics, ratios=[], price=None, shares=100.0
    )
    assert detail is None
    assert any("eksik" in n for n in notes)

    normalized = _annual(Revenue={2023: 1000.0})
    for bad_shares in (None, 0.0, -5.0):
        detail2, _ = _build_midgrowth_revenue_dcf(
            assumptions, normalized, metrics, ratios=[], price=None, shares=bad_shares
        )
        assert detail2 is None


def test_full_scenario_build_hand_verified():
    # --- Inputs ---
    # revenue0=1000 (FY2023, no FY2022 revenue -> latest_yoy=None ->
    # start_growth = realized CAGR = 0.15 exactly, unblended, same
    # _mature_start_growth mechanics as test_valuation_mature_revenue.py).
    #
    # Target margin: gross_margin=0.60 -> _hyper_target_base(0.60,
    # current_margin): ceiling=min(0.60*0.5, 0.30)=0.30; current_margin
    # (below) is -0.15, not >0, so base=ceiling=0.30; gm known -> min(0.30,
    # 0.60)=0.30. Then capped at _MIDGROWTH_TARGET_CAP (0.20):
    # target_margin_base = min(0.30, 0.20) = 0.20 (the mid-growth cap binds).
    #
    # Current (starting) margin: single FY2023, OCF=-100, CapEx=50, no SBC,
    # no Depreciation reported (so _maintenance_adjusted_margin's gate fails
    # on `dep is None` -> raw unchanged) -> current_margin =
    # (OCF-CapEx-SBC)/Revenue = (-100-50-0)/1000 = -0.15. capex_normalization
    # is None (capex/revenue=50/1000=0.05 <= 0.30 anyway).
    #
    # steady_state_year = _MIDGROWTH_STEADY_STATE_YEAR = 8.
    # annual_dilution: shares_yoy not supplied -> 0.0. price=None ->
    # financing_shares stays 0.0 (burn<0 note appended instead).
    # discount_rate=0.15, terminal_growth=0.03 for all 3 scenarios (a valid
    # degenerate case -- only target_margin differs per scenario, via the
    # 0.7/1.0/1.2 _MATURE_TARGET_MARGIN_SCALE).
    #
    # Revenue-first DCF (independently reimplemented from
    # revenue_dcf.py's documented formulas in a from-scratch scratch script,
    # NOT calling revenue_first_dcf; cross-checked before finalizing):
    #   bear (target=0.20*0.7=0.14): per_share=7.9083  -> 7.91; lo=6.16, hi=10.15
    #   base (target=0.20):          per_share=13.4053 -> 13.41; lo=10.77, hi=16.74
    #   bull (target=0.20*1.2=0.24): per_share=17.0700 -> 17.07; lo=13.85, hi=21.14
    normalized = _normalized({
        "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: -100.0},
        "CapEx": {2023: 50.0},
    })
    ratios = [{"fy": 2023, "gross_margin": 0.60}]
    metrics = {
        "revenue_cagr_5y": 0.15, "revenue_cagr_3y": None, "latest_fy": 2023,
        "shares": 100.0, "shares_yoy": None,
    }
    assumptions = _midgrowth_assumptions(discount_rate=0.15, terminal_growth=0.03)

    detail, notes = _build_midgrowth_revenue_dcf(
        assumptions, normalized, metrics, ratios, price=None, shares=100.0
    )

    assert detail is not None
    assert detail["start_growth"] == pytest.approx(0.15)
    assert detail["target_margin_base"] == pytest.approx(0.20)
    assert detail["current_margin"] == pytest.approx(-0.15)
    assert detail["steady_state_year"] == 8
    assert detail["annual_dilution"] == pytest.approx(0.0)
    assert detail["financing_shares"] == pytest.approx(0.0)
    assert detail["suppressed"] is False
    # The mid-growth path does not apply the Sec.3.6 CapEx relief.
    assert "capex_normalization" not in detail

    bear = detail["scenarios"]["bear"]
    assert bear["start_growth"] == pytest.approx(0.15)
    assert bear["target_fcf_margin"] == pytest.approx(0.14)
    assert bear["terminal_growth"] == pytest.approx(0.03)
    assert bear["discount_rate"] == pytest.approx(0.15)
    assert bear["per_share"] == pytest.approx(7.91, abs=0.01)
    assert bear["lo"] == pytest.approx(6.16, abs=0.02)
    assert bear["hi"] == pytest.approx(10.15, abs=0.02)

    base = detail["scenarios"]["base"]
    assert base["target_fcf_margin"] == pytest.approx(0.20)
    assert base["per_share"] == pytest.approx(13.41, abs=0.01)
    assert base["lo"] == pytest.approx(10.77, abs=0.02)
    assert base["hi"] == pytest.approx(16.74, abs=0.02)

    bull = detail["scenarios"]["bull"]
    assert bull["target_fcf_margin"] == pytest.approx(0.24)
    assert bull["per_share"] == pytest.approx(17.07, abs=0.01)
    assert bull["lo"] == pytest.approx(13.85, abs=0.02)
    assert bull["hi"] == pytest.approx(21.14, abs=0.02)


# ---------------------------------------------------------------------------
# B. run_valuation integration (SPEC.md Sec.8d).
# ---------------------------------------------------------------------------


def test_run_valuation_midgrowth_becomes_headline_hand_verified():
    # Same fixture as test_full_scenario_build_hand_verified above, run
    # through the full run_valuation wiring: sector_type="growth_unprofitable",
    # revenue_cagr_5y=0.15 is in the 12-20% band (does NOT trip
    # detect_hyper_grower, which needs > 20%), base per_share=13.41 > 0 (not
    # suppressed) -> midgrowth_revenue_headline=True, and fair_value_range
    # must reflect the mid-growth band (base lo=10.77, hi=16.74).
    normalized = _normalized({
        "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: -100.0},
        "CapEx": {2023: 50.0},
    })
    ratios = [{"fy": 2023, "gross_margin": 0.60, "fcf": -150.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": -150.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.15, "revenue_cagr_3y": None, "shares_yoy": None,
    }
    assumptions = _midgrowth_assumptions(discount_rate=0.15, terminal_growth=0.03)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    # Fixture precondition: hyper-grower mode never triggers (CAGR <= 20%
    # never triggers, regardless of clauses -- SPEC.md Sec.8).
    assert result["hyper_growth"] is False

    assert result["midgrowth_revenue_headline"] is True
    detail = result["midgrowth_revenue_detail"]
    assert detail is not None
    assert detail["steady_state_year"] == 8
    assert detail["target_margin_base"] <= 0.20
    assert detail["target_margin_base"] == pytest.approx(0.20)

    base = detail["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(13.41, abs=0.02)

    # Headline fair_value_range comes from the mid-growth band, not raw
    # FCF-DCF/multiples -- bands are present and finite.
    base_fvr = result["fair_value_range"]["base"]
    assert base_fvr["lo"] == pytest.approx(10.77, abs=0.02)
    assert base_fvr["hi"] == pytest.approx(16.74, abs=0.02)
    assert base_fvr["lo"] is not None and base_fvr["hi"] is not None
    for key in ("bear", "base", "bull"):
        assert result["fair_value_range"][key]["lo"] is not None
        assert result["fair_value_range"][key]["hi"] is not None

    # Reverse-DCF override: revenue-based reference (mirrors hyper-grower/
    # mature-revenue overrides), "gelir 5y"/"gelir 3y" label.
    assert result["reverse_dcf"]["realized_label"] in ("gelir 5y", "gelir 3y")
    assert result["reverse_dcf"]["realized_cagr_5y"] == pytest.approx(0.15)
    assert result["reverse_dcf"]["bracket_status"] == "ok"

    assert any("revenue-first DCF'e dayandırıldı" in n for n in result["notes"])
    # Sec.9 sensitivity exception note (mirrors EPV/mature).
    assert any("orta-büyüme revenue-first DCF'i" in n for n in result["notes"])


def test_run_valuation_growth_gate_rejection_falls_back_to_multiples():
    # realized CAGR = 0.08 < 12% floor -> the method's own growth gate
    # rejects -> midgrowth_revenue_headline=False, detail=None, and the
    # filer keeps its existing raw-FCF-DCF/multiples headline unchanged.
    normalized = _normalized({"Revenue": {2023: 1000.0}})
    ratios = [{"fy": 2023, "gross_margin": 0.60, "fcf": 50.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": 50.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.08, "revenue_cagr_3y": None, "shares_yoy": None,
    }
    assumptions = _midgrowth_assumptions(discount_rate=0.15, terminal_growth=0.03)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    assert result["midgrowth_revenue_headline"] is False
    assert result["midgrowth_revenue_detail"] is None
    assert not any("olgun revenue-first DCF için yetersiz" in n for n in result["notes"])  # not the mature note
    assert any("yetersiz" in n for n in result["notes"])

    # Headline falls back to the raw FCF-DCF band (the pre-existing
    # growth_unprofitable behavior) -- fair_value_range must equal
    # dcf.scenarios' own base band exactly, not a mid-growth one.
    dcf_base = result["dcf"]["scenarios"]["base"]
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(dcf_base["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(dcf_base["hi"])


def test_run_valuation_suppressed_base_falls_back_to_multiples():
    # Deeply negative current margin (current_margin=-1.0, from OCF=-900,
    # CapEx=100, revenue=1000 -> (-900-100-0)/1000=-1.0) with the SAME
    # growth/target-margin setup as the hand-verified build above (start_
    # growth=0.15, target_margin_base=0.20, ss=8, dr=0.15, tg=0.03,
    # shares=100) drives the base scenario deeply negative (independently
    # scratch-computed: per_share = -14.4634 -> -14.46, well below 0) ->
    # the suppression guardrail fires: suppressed=True,
    # midgrowth_revenue_headline stays False (detail is still returned, for
    # transparency, but never headlined), and the headline stays on
    # whatever the raw FCF-DCF produces (here: also None/empty, since a
    # single-year deeply negative fcf0 with no 3y average to fall back on
    # leaves fcf0 unusable) -- the critical assertion is that the headline
    # NEVER reflects the suppressed mid-growth band.
    normalized = _normalized({
        "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: -900.0},
        "CapEx": {2023: 100.0},
    })
    ratios = [{"fy": 2023, "gross_margin": 0.60, "fcf": -1000.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": -1000.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.15, "revenue_cagr_3y": None, "shares_yoy": None,
    }
    assumptions = _midgrowth_assumptions(discount_rate=0.15, terminal_growth=0.03)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    assert result["midgrowth_revenue_headline"] is False
    detail = result["midgrowth_revenue_detail"]
    assert detail is not None  # still reported, just not headlined
    assert detail["suppressed"] is True

    base_ps = detail["scenarios"]["base"]["per_share"]
    assert base_ps is not None
    assert base_ps <= 0
    assert base_ps == pytest.approx(-14.46, abs=0.02)

    assert any(
        "negatif özkaynak değeri" in n and "çarpan (multiples)" in n for n in result["notes"]
    )

    # The headline must NEVER equal the suppressed (negative) mid-growth band.
    fvr_base = result["fair_value_range"]["base"]
    if fvr_base["lo"] is not None:
        assert fvr_base["lo"] != pytest.approx(base_ps, abs=0.5)


def test_run_valuation_backward_compat_mature_sector_stays_false():
    # A "mature" sector_type filer never even attempts the mid-growth branch
    # (it's gated on sector_type == "growth_unprofitable" and not hyper) --
    # regression guard that this feature doesn't touch unrelated sectors.
    normalized = _normalized({"Revenue": {2023: 1000.0}})
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0, "revenue_cagr_5y": 0.15}
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["midgrowth_revenue_headline"] is False
    assert result["midgrowth_revenue_detail"] is None


def test_run_valuation_backward_compat_hyper_active_stays_false():
    # A growth_unprofitable filer that DOES trigger hyper-grower mode
    # (revenue_cagr_5y=0.50, fcf<=0 -- strong tier) never reaches the
    # mid-growth branch either (hyper-grower takes precedence in the
    # priority chain) -- regression guard for the "not hyper_growth_active"
    # gate. Reuses the healthy hyper-grower fixture shape from
    # test_valuation_hyper_suppression.py.
    normalized = _normalized({"Revenue": {2023: 1000.0}})
    ratios = [
        {"fy": 2023, "gross_margin": 0.60, "fcf": -50.0},
        {"fy": 2022, "fcf": 100.0}, {"fy": 2021, "fcf": 90.0},
    ]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": -50.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.50, "rnd_revenue": 0.0, "sbc_revenue": 0.0, "shares_yoy": None,
    }
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result = run_valuation(
        normalized, ratios, metrics, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    assert result["hyper_growth"] is True
    assert result["midgrowth_revenue_headline"] is False
    assert result["midgrowth_revenue_detail"] is None


# ---------------------------------------------------------------------------
# C. triangulate -- midgrowth_revenue_headline confidence cap (mirrors
#    mature_revenue_headline's cap; see test_valuation_mature_revenue.py's
#    own Part F for the identical-shape test).
# ---------------------------------------------------------------------------


def test_triangulate_midgrowth_headline_caps_high_confidence_to_medium():
    # Same three-way "ucuz" agreement fixture as the mature-revenue/EPV cap
    # tests -- would normally be CONFIDENCE_HIGH, direction "ucuz". With
    # midgrowth_revenue_headline=True, confidence must be capped to ORTA.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="growth_unprofitable", midgrowth_revenue_headline=True,
    )
    assert result["signals"] == {"dcf": "ucuz", "reverse_dcf": "ucuz", "multiples": "ucuz"}
    assert result["confidence"] == "ORTA"
    assert result["direction"] == "ucuz"
    assert "orta-büyüme revenue-first DCF'e dayanıyor" in result["rationale"]["confidence"]
    assert "ORTA ile sınırlandı" in result["rationale"]["confidence"]


def test_triangulate_midgrowth_headline_false_preserves_high_confidence():
    # Regression: defaults to False, so the exact same three-way agreement
    # stays CONFIDENCE_HIGH, unchanged from before this feature existed.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="growth_unprofitable",
    )
    assert result["confidence"] == "YÜKSEK"
    assert "ORTA ile sınırlandı" not in result["rationale"]["confidence"]


def test_triangulate_midgrowth_headline_does_not_alter_already_medium_confidence():
    # The cap only ever applies when confidence would otherwise have been
    # HIGH; a 2-of-3 (ORTA) result comes out exactly as it would without the
    # flag -- no further downgrade, no rationale suffix appended.
    two_of_three = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=80, ps_pct=None, pfcf_pct=None,
        sector_type="growth_unprofitable", midgrowth_revenue_headline=True,
    )
    assert two_of_three["confidence"] == "ORTA"
    assert "ORTA ile sınırlandı" not in two_of_three["rationale"]["confidence"]


def test_triangulate_earnings_power_and_midgrowth_both_true_ep_message_wins():
    # Documented mutual-exclusivity fallback (engine.py never sets both in
    # practice, but triangulate() must degrade predictably if it ever
    # happened): when both earnings_power_headline and
    # midgrowth_revenue_headline are True, the earnings_power_headline cap
    # message takes precedence (confidence still ORTA either way -- only
    # the rationale text differs), mirroring the mature-revenue/EPV
    # precedence test.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="growth_unprofitable", earnings_power_headline=True, midgrowth_revenue_headline=True,
    )
    assert result["confidence"] == "ORTA"
    assert "kazanç-gücüne dayanıyor" in result["rationale"]["confidence"]
    assert "orta-büyüme revenue-first DCF'e dayanıyor" not in result["rationale"]["confidence"]
