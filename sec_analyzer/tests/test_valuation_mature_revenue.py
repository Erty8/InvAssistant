"""Hand-verified numeric tests for the mature, FCF-suppressed-but-growing
revenue-first DCF (VALUATION.md Sec.4/4a addendum) -- ``valuation.engine``'s
``_mature_target_fcf_margin``/``_mature_current_margin``/``_mature_start_growth``/
``_build_mature_revenue_dcf`` and their ``run_valuation`` wiring, plus the
EPV-floor guardrail (a growth-inclusive revenue-first value that lands BELOW
the zero-growth EPV floor must NOT become the headline) and ``triangulate``'s
confidence cap when the mature revenue-first DCF becomes the headline.

See ``test_valuation_earnings_power.py``'s module docstring/style and
``test_valuation_dcf.py``'s general methodology (independent hand arithmetic
in a comment above each assertion, cross-checked with a from-scratch scratch
computation before finalizing, then verified with ``pytest.approx``). Parts
A-C unit-test the private helpers directly; Part D exercises the full
``run_valuation`` wiring (including the CRITICAL below-floor guardrail); Part
E covers the ``triangulate`` confidence cap.
"""

import pytest

from sec_analyzer.valuation.engine import (
    _build_mature_revenue_dcf,
    _mature_current_margin,
    _mature_start_growth,
    _mature_target_fcf_margin,
    run_valuation,
)
from sec_analyzer.valuation.triangulate import triangulate

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _annual(**concepts) -> dict:
    """Minimal ``normalized``-shaped dict for direct unit calls into the
    ``_mature_*`` helpers: ``_annual(Revenue={2023: 1000.0})`` ->
    ``{"annual": {"Revenue": [{"fy": 2023, "value": 1000.0}]}}``.
    ``to_annual_series`` only reads ``fy``/``value`` off each record, so this
    is sufficient (mirrors ``test_valuation_earnings_power.py``'s ``_annual``)."""
    return {
        "annual": {
            concept: [{"fy": fy, "value": value} for fy, value in by_fy.items()]
            for concept, by_fy in concepts.items()
        }
    }


_MATURE_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity", "OperatingIncome",
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
        for concept in _MATURE_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "Mature Revenue Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _MATURE_CONCEPTS},
        "missing": [c for c in _MATURE_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _MATURE_CONCEPTS},
    }


def _mature_assumptions(discount_rate=0.15, terminal_growth=0.03, growth_5y=None):
    """A bear/base/bull assumption dict. ``_build_mature_revenue_dcf`` only
    reads ``discount_rate``/``terminal_growth`` per scenario (the growth
    story itself comes from realized data, not this pipeline) -- kept
    identical across scenarios here to keep the hand arithmetic tractable
    (a valid special case: nothing in the spec requires the three scenarios'
    dr/tg to differ, only their target-margin scale, SPEC reviewer Finding 4).
    ``growth_5y`` (only consulted by the *standard* FCF-DCF leg and
    triangulation, never by the mature revenue-first method itself) defaults
    to a small positive number distinct from ``terminal_growth``."""
    g5 = growth_5y if growth_5y is not None else terminal_growth + 0.02
    return {
        "bear": {"growth_5y": g5, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": "Ayı."},
        "base": {"growth_5y": g5, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": "Baz."},
        "bull": {"growth_5y": g5, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": "Boğa."},
    }


# ---------------------------------------------------------------------------
# A. _mature_target_fcf_margin (unit) -- median-op x0.75x0.85 (NOPAT proxy),
#    hist-peak x1.5, cap 0.15 floor, current-margin floor (reviewer F5-6).
# ---------------------------------------------------------------------------


def test_mature_target_fcf_margin_nopat_anchor_binds():
    # op_margins (positive years only): FY2020 80/1000=0.08, FY2021
    # 120/1000=0.12 -> median([0.08,0.12]) = 0.10.
    # nopat = 0.10 * (1-0.25) * 0.85 = 0.10*0.75*0.85 = 0.06375.
    # hist-anchor: only OCF/CapEx data at FY2015 (outside the fy/fy-1/fy-2
    # window used by _mature_current_margin, see below): (300-100)/1000=0.20
    # -> hist_anchor = 0.20*1.5 = 0.30.
    # target = min(nopat=0.06375, hist=0.30, cap=0.15) = 0.06375 (nopat binds).
    # current_margin: fy=2023 (metrics), window = {2023,2022,2021}; none of
    # those years has OCF/CapEx data (only 2015 does) -> current_margin=0.0,
    # floor inactive (0.0 is not > 0).
    normalized = _annual(
        OperatingIncome={2020: 80.0, 2021: 120.0},
        Revenue={2020: 1000.0, 2021: 1000.0, 2015: 1000.0},
        OperatingCashFlow={2015: 300.0}, CapEx={2015: 100.0},
    )
    metrics = {"latest_fy": 2023}

    assert _mature_current_margin(normalized, metrics) == pytest.approx(0.0)
    target = _mature_target_fcf_margin(normalized, metrics, ratios=[])
    assert target == pytest.approx(0.06375)


def test_mature_target_fcf_margin_hist_anchor_binds():
    # op_margin: single FY2020, 250/1000=0.25 -> nopat=0.25*0.75*0.85=0.159375.
    # hist: FY2015 (150-100)/1000=0.05 -> hist_anchor=0.05*1.5=0.075.
    # target = min(nopat=0.159375, hist=0.075, cap=0.15) = 0.075 (hist binds).
    # current_margin=0.0 (same reasoning as above -- no OCF/CapEx in the
    # fy/fy-1/fy-2 window) -> floor inactive.
    normalized = _annual(
        OperatingIncome={2020: 250.0}, Revenue={2020: 1000.0, 2015: 1000.0},
        OperatingCashFlow={2015: 150.0}, CapEx={2015: 100.0},
    )
    metrics = {"latest_fy": 2023}

    target = _mature_target_fcf_margin(normalized, metrics, ratios=[])
    assert target == pytest.approx(0.075)


def test_mature_target_fcf_margin_no_absolute_cap_nopat_anchor_binds_uncapped():
    # op_margin: single FY2020, 400/1000=0.40 -> nopat=0.40*0.75*0.85=0.255.
    # hist: FY2015 (400-100)/1000=0.30 -> hist_anchor=0.30*1.5=0.45.
    # WP4: _MATURE_TARGET_CAP (0.15) is no longer part of this function's
    # min-candidates -- it is now only a reporting/flag threshold applied by
    # the caller (_build_mature_revenue_dcf), not baked into this return
    # value. So target = min(nopat=0.255, hist=0.45) = 0.255 (nopat binds,
    # uncapped) -- both anchors are far above the old 0.15 cap, and that's
    # now correctly reflected rather than silently truncated.
    normalized = _annual(
        OperatingIncome={2020: 400.0}, Revenue={2020: 1000.0, 2015: 1000.0},
        OperatingCashFlow={2015: 400.0}, CapEx={2015: 100.0},
    )
    metrics = {"latest_fy": 2023}

    target = _mature_target_fcf_margin(normalized, metrics, ratios=[])
    assert target == pytest.approx(0.255)


def test_mature_target_fcf_margin_current_margin_floor_raises_target():
    # A deliberately low op-anchor (nopat) so the current-margin floor is the
    # binding constraint (reviewer F6): single FY2023 op_margin=20/1000=0.02
    # -> nopat=0.02*0.75*0.85=0.01275.
    # hist: FY2023/2022/2021 all (200-100)/1000=0.10 -> hist_anchor=0.10*1.5=
    # 0.15 (== cap, doesn't matter which binds since both are 0.15).
    # Pre-floor target = min(nopat=0.01275, hist=0.15, cap=0.15) = 0.01275.
    # current_margin: fy/fy-1/fy-2 = 2023/2022/2021 all have Revenue=1000,
    # OCF=200, CapEx=100, no SBC -> margin=(200-100-0)/1000=0.10 every year
    # -> median([0.10,0.10,0.10])=0.10.
    # Floor: current_margin(0.10) > pre-floor target(0.01275) -> final target
    # = max(0.01275, 0.10) = 0.10 -- the floor dominates.
    normalized = _annual(
        OperatingIncome={2023: 20.0},
        Revenue={2023: 1000.0, 2022: 1000.0, 2021: 1000.0},
        OperatingCashFlow={2023: 200.0, 2022: 200.0, 2021: 200.0},
        CapEx={2023: 100.0, 2022: 100.0, 2021: 100.0},
    )
    metrics = {"latest_fy": 2023}

    assert _mature_current_margin(normalized, metrics) == pytest.approx(0.10)
    target = _mature_target_fcf_margin(normalized, metrics, ratios=[])
    assert target == pytest.approx(0.10)


def test_mature_target_fcf_margin_none_when_neither_anchor_available():
    # No OperatingIncome (no op-anchor) and no OperatingCashFlow/CapEx (no
    # hist-anchor) at all -> the method can't be built: None.
    normalized = _annual(Revenue={2020: 1000.0})
    metrics = {"latest_fy": 2023}

    assert _mature_target_fcf_margin(normalized, metrics, ratios=[]) is None


# ---------------------------------------------------------------------------
# B. _mature_current_margin (unit) -- 3-year median of the SBC-adjusted raw
#    FCF margin (reviewer F6: smooths a lone working-capital swing).
# ---------------------------------------------------------------------------


def test_mature_current_margin_median_of_three_full_years():
    # FY2023: (150-100-0)/1000 = 0.05
    # FY2022: (200-100-0)/1000 = 0.10
    # FY2021: (250-100-0)/1000 = 0.15
    # median([0.05, 0.10, 0.15]) = 0.10 (the middle value).
    normalized = _annual(
        Revenue={2023: 1000.0, 2022: 1000.0, 2021: 1000.0},
        OperatingCashFlow={2023: 150.0, 2022: 200.0, 2021: 250.0},
        CapEx={2023: 100.0, 2022: 100.0, 2021: 100.0},
    )
    metrics = {"latest_fy": 2023}

    assert _mature_current_margin(normalized, metrics) == pytest.approx(0.10)


def test_mature_current_margin_two_years_sbc_adjusted_and_averaged():
    # Only FY2023/2022 have usable data (FY2021 missing entirely -> skipped,
    # NOT treated as 0): FY2023 (180-100-30)/1000=0.05 (SBC=30 subtracted,
    # reviewer F6's SBC-as-expense convention); FY2022 (200-100-0)/1000=0.10
    # (no SBC entry -> treated as 0.0).
    # median of 2 elements = average = (0.05+0.10)/2 = 0.075.
    normalized = _annual(
        Revenue={2023: 1000.0, 2022: 1000.0},
        OperatingCashFlow={2023: 180.0, 2022: 200.0},
        CapEx={2023: 100.0, 2022: 100.0},
        SBC={2023: 30.0},
    )
    metrics = {"latest_fy": 2023}

    assert _mature_current_margin(normalized, metrics) == pytest.approx(0.075)


def test_mature_current_margin_zero_when_no_usable_data():
    normalized = _annual(Revenue={2023: 1000.0})  # no OCF/CapEx at all
    metrics = {"latest_fy": 2023}

    assert _mature_current_margin(normalized, metrics) == 0.0

    # fy itself unresolvable (no latest_fy/latest_fundamental_fy) -> 0.0 too.
    assert _mature_current_margin(normalized, {}) == 0.0


# ---------------------------------------------------------------------------
# C. _mature_start_growth (unit) -- CAGR (5y, falling back to 3y) blended
#    50/50 with the latest single-year revenue YoY (reviewer F4 pattern).
# ---------------------------------------------------------------------------


def test_mature_start_growth_blends_realized_cagr_with_latest_yoy():
    # realized (revenue_cagr_5y) = 0.30. latest_yoy = 1100/1000 - 1 = 0.10.
    # blended = 0.5*0.30 + 0.5*0.10 = 0.20.
    metrics = {"revenue_cagr_5y": 0.30, "revenue_cagr_3y": None, "latest_fy": 2023}
    normalized = _annual(Revenue={2023: 1100.0, 2022: 1000.0})

    assert _mature_start_growth(metrics, normalized) == pytest.approx(0.20)


def test_mature_start_growth_falls_back_to_3y_cagr_and_no_yoy_available():
    # revenue_cagr_5y missing -> falls back to revenue_cagr_3y (0.12). No
    # prior-year revenue available -> latest_yoy=None -> returns the
    # unblended realized figure as-is.
    metrics = {"revenue_cagr_5y": None, "revenue_cagr_3y": 0.12, "latest_fy": 2023}
    normalized = _annual()

    assert _mature_start_growth(metrics, normalized) == pytest.approx(0.12)


def test_mature_start_growth_none_when_realized_cagr_unavailable():
    # Both revenue_cagr_5y and revenue_cagr_3y are None -> None, regardless
    # of any YoY data (the method can't be built at all without SOME
    # realized-growth reference).
    metrics = {"revenue_cagr_5y": None, "revenue_cagr_3y": None, "latest_fy": 2023}
    normalized = _annual(Revenue={2023: 1100.0, 2022: 1000.0})

    assert _mature_start_growth(metrics, normalized) is None


# ---------------------------------------------------------------------------
# D. _build_mature_revenue_dcf (unit) -- growth gate + full scenario build
#    (SPEC reviewer Finding 2/4/8).
# ---------------------------------------------------------------------------


def test_build_mature_revenue_dcf_growth_gate_rejects_below_min_growth_threshold():
    # realized CAGR = 0.05 < _MATURE_REV_DCF_MIN_GROWTH (0.10) -> the first
    # clause of the growth gate rejects outright (regardless of terminal
    # growth) -> (None, note).
    normalized = _annual(Revenue={2023: 1000.0})
    metrics = {"revenue_cagr_5y": 0.05, "revenue_cagr_3y": None, "latest_fy": 2023, "shares": 100.0}
    assumptions = _mature_assumptions(discount_rate=0.15, terminal_growth=0.02)

    detail, notes = _build_mature_revenue_dcf(assumptions, normalized, metrics, ratios=[], price=None, shares=100.0)

    assert detail is None
    assert any("yetersiz" in n for n in notes)
    assert any("%5.0" in n for n in notes)


def test_build_mature_revenue_dcf_growth_gate_rejects_at_or_below_terminal_growth():
    # realized CAGR = 0.10 -- NOT < the min-growth threshold (0.10<0.10 is
    # False), so the first clause does not fire. But the scenario's own
    # terminal_growth is set to 0.12 here (deliberately above the realized
    # growth) -> the gate's SECOND clause (start_growth <= terminal_growth,
    # "nothing left to fade") rejects instead -> (None, note). Isolates the
    # second clause from the first (reviewer Finding 2's "nothing left to
    # fade" case), even though both clauses share the same note text in the
    # implementation.
    normalized = _annual(Revenue={2023: 1000.0})
    metrics = {"revenue_cagr_5y": 0.10, "revenue_cagr_3y": None, "latest_fy": 2023, "shares": 100.0}
    assumptions = _mature_assumptions(discount_rate=0.20, terminal_growth=0.12)

    detail, notes = _build_mature_revenue_dcf(assumptions, normalized, metrics, ratios=[], price=None, shares=100.0)

    assert detail is None
    assert any("yetersiz" in n for n in notes)
    assert any("%10.0" in n for n in notes)


def test_build_mature_revenue_dcf_full_scenario_build_hand_verified():
    # --- Inputs ---
    # revenue0=1000 (FY2023, no FY2022 revenue -> latest_yoy=None ->
    # start_growth = realized CAGR = 0.30 exactly, unblended).
    # target_margin_base: single-year OperatingIncome=300/Revenue=1000 ->
    # op_margin=0.30 -> nopat=0.30*0.75*0.85=0.19125. No OCF/CapEx data at
    # all -> hist_anchor=None -> target=min(nopat=0.19125)=0.19125 (WP4: no
    # longer clamped to the old 0.15 cap -- that's now only a flag threshold
    # the caller compares against, see the "above_reference" flag check
    # below). current_margin=0.0 (no OCF/CapEx data) -> floor inactive ->
    # target_margin_base=0.19125 (rounds to 0.1912 in the detail dict, see
    # below -- round(0.19124999999999998, 4) == 0.1912 due to the exact
    # floating-point value of 0.30*0.75*0.85).
    # ss=_MATURE_STEADY_STATE_YEAR=7. discount_rate=0.15, terminal_growth=
    # 0.06 for all three scenarios (a valid special case -- only the target
    # margin differs per scenario here, via the 0.7/1.0/1.2 scale).
    #
    # Growth path (fade from 0.30 to 0.06 over 7 years, denominator=ss-1=6):
    #   g_t = 0.30 - 0.04*min(t-1,6) for t=1..7, flat at 0.06 for t=8..10:
    #   g1=0.30, g2=0.26, g3=0.22, g4=0.18, g5=0.14, g6=0.10, g7=0.06 (== tg).
    # Margin path (base, target=0.19125, converge over 7 years):
    #   m_t = 0.19125*min(t,7)/7 = 0.02732143*min(t,7).
    # Revenue path is unaffected by the target margin (rev0=1000,
    # rev_t=rev_{t-1}*(1+g_t)): rev1=1300, rev2=1638, rev3=1998.36,
    # rev4=2358.0648, rev5=2688.193872, rev6=2957.013259, rev7=3134.434055,
    # rev8..10 continue fading at 0.06/yr -> 3322.500098, 3521.850104,
    # 3733.16111.
    # FCF_t=rev_t*m_t, discounted at r=0.15 (1.15^t), Gordon terminal value
    # off fcf10 at g=0.06: tv=fcf10*1.06/(0.15-0.06). Summed and divided by
    # effective_shares=100*(1+0)^7=100 (no dilution) gives the per-share
    # figures below.
    # (Re-derived by calling revenue_dcf.revenue_first_dcf directly with the
    # new uncapped target_base -- same growth-path/margin-path/PV/terminal-
    # value formulas the from-scratch scratch computation above this test
    # originally hand-verified for the capped case -- and cross-checked
    # against the real _build_mature_revenue_dcf output before finalizing.)
    #
    # target_base = 0.19125 (uncapped nopat anchor).
    # bear (target=0.19125*0.7=0.133875): per_share=25.377369 -> 25.38;
    #   3x3 sensitivity grid (start_growth +/-0.02, discount_rate +/-0.01)
    #   -> lo=21.06, hi=30.97.
    # base (target=0.19125):          per_share=36.253384 -> 36.25;
    #   -> lo=30.08, hi=44.25.
    # bull (target=0.19125*1.2=0.2295): per_share=43.504061 -> 43.50;
    #   -> lo=36.10, hi=53.10.
    normalized = _annual(Revenue={2023: 1000.0}, OperatingIncome={2023: 300.0})
    metrics = {"revenue_cagr_5y": 0.30, "revenue_cagr_3y": None, "latest_fy": 2023, "shares": 100.0}
    assumptions = _mature_assumptions(discount_rate=0.15, terminal_growth=0.06)

    detail, notes = _build_mature_revenue_dcf(assumptions, normalized, metrics, ratios=[], price=None, shares=100.0)

    assert detail is not None
    assert any("referans eşiğinin üzerinde" in n for n in notes)
    assert detail["start_growth"] == pytest.approx(0.30)
    assert detail["target_margin_base"] == pytest.approx(0.19125, abs=0.0001)
    assert detail["target_margin_flag"] == "above_reference"
    assert detail["current_margin"] == pytest.approx(0.0)
    assert detail["steady_state_year"] == 7

    bear = detail["scenarios"]["bear"]
    assert bear["start_growth"] == pytest.approx(0.30)
    assert bear["target_fcf_margin"] == pytest.approx(0.133875, abs=0.0001)
    assert bear["terminal_growth"] == pytest.approx(0.06)
    assert bear["discount_rate"] == pytest.approx(0.15)
    assert bear["per_share"] == pytest.approx(25.38, abs=0.01)
    assert bear["lo"] == pytest.approx(21.06, abs=0.02)
    assert bear["hi"] == pytest.approx(30.97, abs=0.02)

    base = detail["scenarios"]["base"]
    assert base["target_fcf_margin"] == pytest.approx(0.19125, abs=0.0001)
    assert base["per_share"] == pytest.approx(36.25, abs=0.01)
    assert base["lo"] == pytest.approx(30.08, abs=0.02)
    assert base["hi"] == pytest.approx(44.25, abs=0.02)

    bull = detail["scenarios"]["bull"]
    assert bull["target_fcf_margin"] == pytest.approx(0.2295, abs=0.0001)
    assert bull["per_share"] == pytest.approx(43.50, abs=0.01)
    assert bull["lo"] == pytest.approx(36.10, abs=0.02)
    assert bull["hi"] == pytest.approx(53.10, abs=0.02)


def test_build_mature_revenue_dcf_none_without_valid_shares_or_revenue():
    metrics = {"revenue_cagr_5y": 0.30, "latest_fy": 2023, "shares": 100.0}
    assumptions = _mature_assumptions()

    # Missing revenue at fy.
    detail, notes = _build_mature_revenue_dcf(assumptions, _annual(), metrics, ratios=[], price=None, shares=100.0)
    assert detail is None
    assert any("eksik" in n for n in notes)

    normalized = _annual(Revenue={2023: 1000.0})
    # shares missing / zero / negative.
    for bad_shares in (None, 0.0, -5.0):
        detail2, _ = _build_mature_revenue_dcf(assumptions, normalized, metrics, ratios=[], price=None, shares=bad_shares)
        assert detail2 is None


def test_build_mature_revenue_dcf_none_when_target_margin_unavailable():
    # Realized growth clears the gate (0.30 >= 0.10 and > terminal 0.02),
    # but neither op-anchor nor hist-anchor data exists -> target margin is
    # None -> the method degrades to (None, notes) even though the growth
    # story itself was fine.
    normalized = _annual(Revenue={2023: 1000.0})
    metrics = {"revenue_cagr_5y": 0.30, "revenue_cagr_3y": None, "latest_fy": 2023, "shares": 100.0}
    assumptions = _mature_assumptions(discount_rate=0.15, terminal_growth=0.02)

    detail, notes = _build_mature_revenue_dcf(assumptions, normalized, metrics, ratios=[], price=None, shares=100.0)

    assert detail is None
    assert any("hedef olgun FCF marj" in n for n in notes)


# ---------------------------------------------------------------------------
# E. run_valuation integration (VALUATION.md Sec.4/4a) + the CRITICAL
#    below-floor guardrail (added by this task, on top of the reviewer spec):
#    a growth-inclusive revenue-first value that lands BELOW the zero-growth
#    EPV floor must NOT become the headline.
# ---------------------------------------------------------------------------


def test_run_valuation_mature_revenue_first_beats_epv_floor_becomes_headline():
    # Mature, FCF-suppressed, genuinely-growing fixture (Amazon-shaped):
    # revenue_cagr_5y=0.15 (>= the 0.10 gate, and safely <= 0.20 so
    # sector.detect_hyper_grower's strong/gray tiers never fire -- CAGR at
    # or below 20% never triggers hyper-grower mode regardless of clauses).
    #
    # EPV floor (single-FY fixture, sanity guard trivially inactive, same
    # derivation as test_valuation_earnings_power.py): NI=100, dr_base=0.15,
    # shares=100 -> epv_base_per_share = 100/0.15/100 = 6.6667 (-> 6.67).
    #
    # Standard FCF-DCF leg deliberately starved (fcf0=1.0) -- its exact
    # 10-year value isn't hand-derived here (see test_valuation_dcf.py for
    # that methodology); what matters for the fixture precondition is that
    # it lands far under 0.5*epv_base=3.33 (suppressed).
    # Cash-conversion guard: OCF=150 >= 0.8*NI(100)=80 -> cash_backed=True.
    # Investment-driven: CapEx=75, capex/ocf=75/150=0.5 >= 0.5 -> True.
    # -> _fcf_dcf_unreliable's gate fires.
    #
    # Mature target margin: no OperatingIncome (nopat=None); hist-anchor =
    # (OCF-CapEx)/Revenue * 1.5 = (150-75)/1000*1.5 = 0.075*1.5 = 0.1125 ->
    # target=min(hist=0.1125, cap=0.15)=0.1125. current_margin (single FY,
    # no SBC) = (150-75-0)/1000=0.075 (< target, floor inactive).
    # start_growth: no FY2022 revenue -> latest_yoy=None -> start_growth =
    # realized CAGR = 0.15 exactly. Growth gate: 0.15>=0.10 and 0.15>0.03
    # (terminal) -> passes.
    #
    # Mature revenue-first DCF (base, target_margin=0.1125, current_margin=
    # 0.075, start_growth=0.15, terminal=0.03, dr=0.15, ss=7, revenue0=1000,
    # shares=100) -- independently scratch-computed (same methodology as the
    # unit test above): base per_share = 12.4203 -> 12.42, lo=10.75,
    # hi=14.47. 12.42 >= epv_base(6.67) -> the guardrail's "mr_beats_floor"
    # condition holds -> mature_revenue_headline=True, and fair_value_range
    # must reflect the mature revenue-first band, NOT the EPV floor.
    normalized = _normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 150.0}, "CapEx": {2023: 75.0},
    })
    ratios = [{"fy": 2023, "fcf": 1.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": 1.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.15, "revenue_cagr_3y": None,
    }
    assumptions = _mature_assumptions(discount_rate=0.15, terminal_growth=0.03)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    # Fixture preconditions.
    assert result["dcf"]["scenarios"]["base"]["hi"] < 3.33
    assert result["earnings_power"]["scenarios"]["base"]["per_share"] == pytest.approx(6.67, abs=0.01)

    assert result["mature_revenue_headline"] is True
    assert result["earnings_power_headline"] is False
    assert result["mature_revenue_detail"] is not None

    mr_base = result["mature_revenue_detail"]["scenarios"]["base"]
    assert mr_base["per_share"] == pytest.approx(12.42, abs=0.02)
    assert mr_base["per_share"] > result["earnings_power"]["scenarios"]["base"]["per_share"]

    # Headline fair_value_range must reflect the mature revenue-first band.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(10.75, abs=0.02)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(14.47, abs=0.02)
    assert "gerçekleşen büyüme" in result["fair_value_range"]["base"]["growth"]
    assert result["fair_value_range"]["base"]["discount_rate"] == "%15"

    # Reverse-DCF override: revenue-based reference (mirrors hyper-grower).
    assert result["reverse_dcf"]["realized_label"] == "gelir 5y"
    assert result["reverse_dcf"]["realized_cagr_5y"] == pytest.approx(0.15)
    assert result["reverse_dcf"]["bracket_status"] == "ok"

    assert any("revenue-first DCF'e dayandırıldı" in n for n in result["notes"])


def test_run_valuation_mature_revenue_first_below_epv_floor_guardrail_keeps_epv_headline():
    # CRITICAL regression guard (the guardrail added on top of the reviewer
    # spec): same growth gate/gate-firing preconditions as the test above
    # (revenue_cagr_5y=0.15, NI=100, dr_base=0.15 -> epv_base=6.67), but
    # CapEx=88 this time (OCF=90 unchanged) -> the mature target margin
    # collapses:
    #   hist-anchor = (90-88)/1000*1.5 = 0.002*1.5 = 0.003 -> target=
    #   min(hist=0.003, cap=0.15)=0.003 (nopat unavailable).
    #   current_margin = (90-88-0)/1000=0.002 (< target, floor inactive).
    #   investment_driven check (unaffected by target-margin shrinkage):
    #   capex/ocf=88/90=0.9778 >= 0.5 -> still True; cash_backed: ocf=90 >=
    #   0.8*100=80 -> still True -> the FCF-DCF-unreliable gate still fires,
    #   so _build_mature_revenue_dcf is still attempted (and its growth gate
    #   still passes -- growth story unaffected by margin inputs).
    #
    # Mature DCF (base, target_margin=0.003, current_margin=0.002,
    # start_growth=0.15, terminal=0.03, dr=0.15, ss=7, revenue0=1000,
    # shares=100) -- scratch-computed: base per_share = 0.3312 -> 0.33.
    # 0.33 < epv_base(6.67) -> the guardrail's "mr_beats_floor" condition
    # fails -> the revenue-first value is demoted to a secondary
    # cross-check (still reported under mature_revenue_detail); the
    # headline stays the EPV floor.
    normalized = _normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 88.0},
    })
    ratios = [{"fy": 2023, "fcf": 1.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": 1.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.15, "revenue_cagr_3y": None,
    }
    assumptions = _mature_assumptions(discount_rate=0.15, terminal_growth=0.03)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["earnings_power"]["scenarios"]["base"]["per_share"] == pytest.approx(6.67, abs=0.01)

    # The guardrail must refuse the revenue-first headline here.
    assert result["mature_revenue_headline"] is False
    assert result["earnings_power_headline"] is True

    # But the revenue-first detail must STILL be present (secondary
    # cross-check, per spec) -- never silently dropped just because it lost
    # the guardrail comparison.
    assert result["mature_revenue_detail"] is not None
    mr_base = result["mature_revenue_detail"]["scenarios"]["base"]
    assert mr_base["per_share"] == pytest.approx(0.33, abs=0.01)
    assert mr_base["per_share"] < result["earnings_power"]["scenarios"]["base"]["per_share"]

    # Headline fair_value_range must be the EPV band, NOT the revenue-first one.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["lo"]
    )
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["hi"]
    )
    assert result["fair_value_range"]["base"]["growth"] == "sıfır büyüme (kazanç gücü çapası)"

    # The below-floor advisory note must be present, citing both figures.
    assert any(
        "altında kaldığı için manşet EPV'de tutuldu" in n and "0.33" in n and "6.67" in n
        for n in result["notes"]
    )


def test_run_valuation_mature_low_realized_growth_gate_rejects_stays_epv_headline():
    # Same fixture as the guardrail-False test above (target-margin-
    # collapsed, CapEx=88) but with revenue_cagr_5y=0.05 this time -- below
    # _MATURE_REV_DCF_MIN_GROWTH (0.10) -> the growth gate itself rejects
    # (mature_revenue_detail is None, distinct from "built but lost the
    # guardrail comparison" above) -> falls back to the existing EPV-
    # headline behavior, identical to pre-this-feature.
    normalized = _normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 88.0},
    })
    ratios = [{"fy": 2023, "fcf": 1.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": 1.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.05, "revenue_cagr_3y": None,
    }
    assumptions = _mature_assumptions(discount_rate=0.15, terminal_growth=0.03)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["mature_revenue_headline"] is False
    assert result["earnings_power_headline"] is True
    assert result["mature_revenue_detail"] is None  # gate rejected, nothing to report

    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["lo"]
    )
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["hi"]
    )
    assert any("olgun revenue-first DCF için yetersiz" in n for n in result["notes"])


def test_run_valuation_mature_healthy_fcf_regression_stays_fcf_dcf_headline():
    # REGRESSION guard: a mature, profitable filer with genuinely healthy
    # FCF (fcf0=100, matching test_valuation_dcf.py's "case A" AND
    # test_valuation_earnings_power.py's own healthy-FCF regression test
    # exactly: growth_5y=0.10, terminal_growth=0.03, discount_rate=0.10,
    # shares=10, net_debt=0 -> base per_share ~=216.7679, hand-derived in
    # both those files) never even reaches the mature-revenue-first
    # attempt: _fcf_dcf_unreliable's gate doesn't fire at all (FCF isn't
    # suppressed relative to EPV, 216+ >> 0.5*epv_base), so
    # mature_revenue_detail stays None and the headline is unchanged from
    # pre-this-feature (raw FCF-DCF).
    normalized = _normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 20.0},
    })
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {
        "shares": 10.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.15, "revenue_cagr_3y": None,
    }
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["earnings_power"] is not None
    assert result["earnings_power_headline"] is False
    assert result["mature_revenue_headline"] is False
    assert result["mature_revenue_detail"] is None

    dcf_base = result["dcf"]["scenarios"]["base"]
    assert dcf_base["per_share"] == pytest.approx(216.7679, rel=1e-3)
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(dcf_base["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(dcf_base["hi"])
    assert result["fair_value_range"]["base"]["growth"] == "%10 büyüme"


# ---------------------------------------------------------------------------
# F. triangulate -- mature_revenue_headline confidence cap (mirrors
#    earnings_power_headline's cap; see test_valuation_earnings_power.py's
#    own Part D for the identical-shape EPV test).
# ---------------------------------------------------------------------------


def test_triangulate_mature_revenue_headline_caps_high_confidence_to_medium():
    # Same three-way "ucuz" agreement fixture as
    # test_valuation_engine.py/test_valuation_earnings_power.py's own cap
    # tests (dcf: price=90 < band.lo=100; reverse_dcf: implied=0.05 <
    # ref(0.10)-0.03=0.07; multiples: pe_pct=20 < 30) -- would normally be
    # CONFIDENCE_HIGH, direction "ucuz". With mature_revenue_headline=True,
    # confidence must be capped to ORTA: the DCF leg and the reverse-DCF leg
    # both derive from the same revenue-first model in this mode, so they
    # aren't independent evidence of one another.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature", mature_revenue_headline=True,
    )
    assert result["signals"] == {"dcf": "ucuz", "reverse_dcf": "ucuz", "multiples": "ucuz"}
    assert result["confidence"] == "ORTA"
    assert result["direction"] == "ucuz"
    assert "olgun revenue-first DCF'e dayanıyor" in result["rationale"]["confidence"]
    assert "ORTA ile sınırlandı" in result["rationale"]["confidence"]


def test_triangulate_mature_revenue_headline_false_preserves_high_confidence():
    # Regression: mature_revenue_headline defaults to False, so the exact
    # same three-way agreement stays CONFIDENCE_HIGH, unchanged from before
    # this feature existed.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["confidence"] == "YÜKSEK"
    assert "ORTA ile sınırlandı" not in result["rationale"]["confidence"]


def test_triangulate_mature_revenue_headline_does_not_alter_already_medium_or_low_confidence():
    # The cap only ever applies when confidence would otherwise have been
    # HIGH; a 2-of-3 (ORTA) result must come out exactly as it would
    # without mature_revenue_headline -- no further downgrade, and no
    # rationale suffix appended.
    two_of_three = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=80, ps_pct=None, pfcf_pct=None,
        sector_type="mature", mature_revenue_headline=True,
    )
    assert two_of_three["confidence"] == "ORTA"
    assert "ORTA ile sınırlandı" not in two_of_three["rationale"]["confidence"]


def test_triangulate_both_headline_flags_true_earnings_power_message_takes_precedence():
    # Documented mutual-exclusivity fallback (engine.py never sets both in
    # practice, but triangulate() itself must degrade predictably if it
    # ever happened): when both earnings_power_headline and
    # mature_revenue_headline are True, the earnings_power_headline cap
    # message takes precedence over the mature one (confidence still ORTA
    # either way -- only the rationale text differs).
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature", earnings_power_headline=True, mature_revenue_headline=True,
    )
    assert result["confidence"] == "ORTA"
    assert "kazanç-gücüne dayanıyor" in result["rationale"]["confidence"]
    assert "olgun revenue-first DCF'e dayanıyor" not in result["rationale"]["confidence"]
