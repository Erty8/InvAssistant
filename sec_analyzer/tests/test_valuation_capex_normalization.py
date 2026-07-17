"""Hand-verified tests for the maintenance/growth CapEx split
(``valuation.engine._maintenance_adjusted_margin``, SPEC.md Sec.3.6 /
Roadmap Madde 1).

Background: a capex-heavy hyper-grower (e.g. a data-center builder like
APLD) spends CapEx that is many multiples of its maintenance needs. Most of
that CapEx is *growth* CapEx that builds the very future revenue the
revenue-first projection already captures via its growth path, so
subtracting it from the *starting* FCF margin double-penalizes the same
expansion once as today's cash outflow, again as forgone terminal cash flow.
``_maintenance_adjusted_margin`` relieves the growth portion
(``capex - d&a``) from the starting margin when the filer is genuinely
capex-heavy (``capex/revenue > 0.30``) AND there is growth CapEx above
maintenance to relieve (``capex > d&a``); otherwise it returns the raw
margin unchanged (byte-for-byte pre-existing behavior).

Part A unit-tests the helper directly (hand-derived arithmetic, small
round numbers). Part B exercises the full ``run_valuation`` wiring on a
capex-heavy HYPER-grower fixture, demonstrating (with two otherwise-
identical fixtures that differ only in whether ``Depreciation`` is
reported) that the relief is what flips the base scenario from a
non-credible suppressed (<=0) value to a positive, published one.

Fixture shapes mirror ``test_valuation_hyper_suppression.py``'s
``_rec``/``_normalized``/``_assumptions`` helpers (Sec.12).
"""

import pytest

from sec_analyzer.valuation.engine import _maintenance_adjusted_margin, run_valuation

# ---------------------------------------------------------------------------
# Part A -- _maintenance_adjusted_margin (unit)
# ---------------------------------------------------------------------------


def _annual(**concepts) -> dict:
    """Minimal ``normalized``-shaped dict: ``_annual(Revenue={2023: 1000.0})``
    -> ``{"annual": {"Revenue": [{"fy": 2023, "value": 1000.0}]}}``.
    ``to_annual_series`` only reads ``fy``/``value`` off each record, so this
    is sufficient (mirrors ``test_valuation_mature_revenue.py``'s ``_annual``)."""
    return {
        "annual": {
            concept: [{"fy": fy, "value": value} for fy, value in by_fy.items()]
            for concept, by_fy in concepts.items()
        }
    }


def test_capex_heavy_applies_relief_hand_verified():
    # revenue=1000, capex=500 -> capex_intensity=500/1000=0.5 > 0.30 (gate 1
    # OK); capex(500) > d&a(100) (gate 2 OK) -> relief applies.
    # growth_capex = capex - d&a = 500 - 100 = 400.
    # ops_margin = raw_current_margin + growth_capex/revenue
    #            = -0.60 + 400/1000 = -0.60 + 0.40 = -0.20.
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 500.0}, Depreciation={2023: 100.0})
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.60)

    assert ops_margin == pytest.approx(-0.20)
    assert capex_normalization is not None
    assert capex_normalization["applied"] is True
    assert capex_normalization["capex_intensity"] == pytest.approx(0.5)
    assert capex_normalization["maintenance_capex"] == pytest.approx(100.0)
    assert capex_normalization["growth_capex"] == pytest.approx(400.0)
    assert capex_normalization["raw_current_margin"] == pytest.approx(-0.60)
    assert capex_normalization["ops_current_margin"] == pytest.approx(-0.20)


def test_capex_light_intensity_at_or_below_threshold_not_applied():
    # revenue=1000, capex=200 -> capex_intensity=0.2 <= 0.30 -> gate 1 fails
    # -> the raw margin is returned unchanged, capex_normalization is None,
    # regardless of capex > d&a (capex(200) > d&a(50) here, but that alone
    # isn't enough -- both gates must hold).
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 200.0}, Depreciation={2023: 50.0})
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.10)

    assert ops_margin == -0.10  # returned exactly as passed in, unchanged
    assert capex_normalization is None


def test_capex_heavy_but_at_or_below_maintenance_not_applied():
    # revenue=1000, capex=400 -> capex_intensity=0.4 > 0.30 (gate 1 OK), but
    # capex(400) <= d&a(500) -- no growth CapEx above maintenance to relieve
    # (gate 2 fails) -> raw margin unchanged, capex_normalization is None.
    # Isolates gate 2 from gate 1 (a filer under-investing relative to its
    # own depreciation gets no relief, even though it is "capex-heavy" by
    # the intensity ratio alone).
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 400.0}, Depreciation={2023: 500.0})
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.30)

    assert ops_margin == -0.30
    assert capex_normalization is None


def test_capex_equal_to_maintenance_not_applied_strict_inequality():
    # capex == d&a exactly (100 == 100) -- "capex > d&a" is a STRICT
    # inequality, so this must NOT apply relief even though intensity clears
    # the 30% gate.
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 500.0}, Depreciation={2023: 500.0})
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.45)

    assert ops_margin == -0.45
    assert capex_normalization is None


# ---------------------------------------------------------------------------
# Part A (edge cases) -- missing/invalid data degrades to "raw, unchanged".
# ---------------------------------------------------------------------------


def test_missing_depreciation_not_applied():
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 500.0})  # no Depreciation at all
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.60)

    assert ops_margin == -0.60
    assert capex_normalization is None


@pytest.mark.parametrize("dep_value", [0.0, -10.0])
def test_nonpositive_depreciation_not_applied(dep_value):
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 500.0}, Depreciation={2023: dep_value})
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.60)

    assert ops_margin == -0.60
    assert capex_normalization is None


@pytest.mark.parametrize("revenue_value", [0.0, -1000.0])
def test_nonpositive_revenue_not_applied(revenue_value):
    normalized = _annual(Revenue={2023: revenue_value}, CapEx={2023: 500.0}, Depreciation={2023: 100.0})
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.60)

    assert ops_margin == -0.60
    assert capex_normalization is None


def test_missing_capex_not_applied():
    normalized = _annual(Revenue={2023: 1000.0}, Depreciation={2023: 100.0})  # no CapEx at all
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.60)

    assert ops_margin == -0.60
    assert capex_normalization is None


def test_unresolvable_fiscal_year_not_applied():
    # metrics has neither latest_fundamental_fy nor latest_fy -- fy can't be
    # resolved at all, so the gate can't even be evaluated -> raw unchanged.
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 500.0}, Depreciation={2023: 100.0})
    metrics = {}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(normalized, metrics, -0.60)

    assert ops_margin == -0.60
    assert capex_normalization is None


# ---------------------------------------------------------------------------
# Part B -- run_valuation end-to-end on a capex-heavy HYPER-grower fixture.
#
# Two fixtures, identical except one omits `Depreciation` (fixture A: no
# relief possible, gate fails on `dep is None`) and the other reports it
# (fixture B: relief applies). This isolates the relief's effect using the
# real engine, not a synthetic before/after comparison.
#
# Shared setup: revenue0=1000, CapEx=4000, OCF=50 (so raw fcf = OCF-CapEx =
# -3950 -> metrics["fcf"]=-3950, sbc=0) -> raw_current_margin =
# -3950/1000 = -3.95. gross_margin=0.60, revenue_cagr_5y=1.0 (100%, clears
# detect_hyper_grower's strong tier > 25%; fcf<=0 also independently fires
# clause (a)). shares=100, price=None (keeps financing_shares=0 in both
# cases, so the ONLY difference in the revenue-first DCF inputs between A
# and B is current_margin itself).
#
# Fixture A (no Depreciation): capex_normalization gate fails (dep is None)
# -> current_margin stays raw (-3.95). Fixture B (Depreciation=100):
# capex_intensity=4000/1000=4.0>0.30, capex(4000)>dep(100) -> relief applies:
# growth_capex=4000-100=3900, ops_margin=-3.95+3900/1000=-3.95+3.90=-0.05.
#
# Base scenario uses the FIXED hyper rates (SPEC Sec.3): start_growth=
# min(1.0, 0.60)=0.60 (WP5: _HYPER_START_GROWTH_CAP raised 0.40 -> 0.60, so
# this is now capped instead of the old min(1.0,0.40)=0.40; no latest-YoY
# data supplied -> unblended), terminal_growth=0.025, discount_rate=0.12
# (base), steady_state_year=10. target_base (mature target margin) =
# _hyper_target_base(gm=0.60, current_margin): ceiling=min(0.60*0.5, 0.30)=
# 0.30; current_margin is <=0 in BOTH fixtures (not >0) -> base=ceiling=0.30;
# gm known -> min(0.30,0.60)=0.30 -- i.e. target_base=0.30 in BOTH cases (the
# relief only changes the STARTING margin, never the independently-derived
# mature target, exactly as documented -- "Only the starting margin is
# relieved").
#
# Revenue-first DCF (independently reimplemented from the module's own
# documented formulas -- growth-path/margin-path/PV/terminal-value -- in a
# from-scratch scratch script, NOT calling revenue_dcf.revenue_first_dcf;
# cross-checked before finalizing):
#
# WP3 discount-rate fade: `_assumptions()`'s base discount_rate is 0.10 ->
# mature_discount_rate = max(0.10, terminal(0.025) + sanity._MIN_ERP_SPREAD
# (0.045) = 0.07) = 0.10. The base scenario's revenue-first DCF now fades
# from its cohort rate (0.12) down to 0.10 by steady_state_year=10 instead of
# discounting flat at 0.12, and the terminal value discounts at 0.10.
# Re-derived (same from-scratch reimplementation, extended with the fade)
# and cross-checked against the real run_valuation() output:
#   base(revenue0=1000, start_growth=0.60, terminal=0.025, r=0.12 (fading to
#        mature=0.10), current_margin=-3.95, target=0.30, ss=10, shares=100,
#        dilution=0, financing=0) -> per_share = -262.38  (<=0 -> suppressed)
#   base(same inputs but current_margin=-0.05) -> per_share = +265.35
#     (band: lo=236.90, hi=297.14 over the +/-2pp start_growth x +/-1pp
#     discount_rate 3x3 grid, each row fading to the same shared 0.10)
# ---------------------------------------------------------------------------

_CAPEX_CONCEPTS = [
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


def _normalized(overrides):
    annual = {c: overrides.get(c) for c in _CAPEX_CONCEPTS}
    return {
        "cik": 1, "entity_name": "Capex Norm Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CAPEX_CONCEPTS},
        "missing": [c for c in _CAPEX_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CAPEX_CONCEPTS},
    }


def _assumptions():
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


_CAPEX_RATIOS = [{"fy": 2023, "gross_margin": 0.60, "fcf": -3950.0}]
_CAPEX_METRICS = {
    "shares": 100.0, "latest_fy": 2023, "fcf": -3950.0, "net_debt": 0.0,
    "revenue_cagr_5y": 1.0, "rnd_revenue": 0.0, "sbc_revenue": 0.0, "shares_yoy": None,
}


def test_run_valuation_without_depreciation_stays_suppressed(tmp_path):
    # No Depreciation reported at all -> gate fails (dep is None) ->
    # current_margin stays the deeply negative raw figure -> base per_share
    # <= 0 -> the existing (Sec.3's) suppression guard fires, same as
    # test_valuation_hyper_suppression.py's APLD-shaped fixture.
    #
    # This test is about the maintenance-CapEx-relief MECHANICS, not WP2's
    # risk-free-linked terminal growth, so it pins terminal_growth to the
    # old fixed 0.025 (matching the hand-verified -262.38 below, WP3
    # fade-adjusted) by pointing damodaran_dir at a nonexistent directory --
    # load_sector_data then returns None and the anchor falls back to
    # engine._HYPER_TERMINAL_GROWTH=0.025. Without this, run_valuation's
    # default damodaran_dir picks up the repo's real data/damodaran/erp.csv
    # (risk_free=4.20) and the anchor becomes 0.04, invalidating the
    # hand-verified per_share below.
    normalized = _normalized({"Revenue": [_rec(2023, 1000.0)], "CapEx": [_rec(2023, 4000.0)]})

    result = run_valuation(
        normalized, _CAPEX_RATIOS, _CAPEX_METRICS, price=None, price_df=None,
        assumptions=_assumptions(), sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    detail = result["hyper_growth_detail"]
    assert detail is not None
    assert detail["capex_normalization"] is None
    assert detail["suppressed"] is True

    base_ps = detail["scenarios"]["base"]["per_share"]
    assert base_ps is not None
    assert base_ps <= 0
    assert base_ps == pytest.approx(-262.38, abs=0.05)

    for key in ("bear", "base", "bull"):
        assert result["fair_value_range"][key]["lo"] is None
        assert result["fair_value_range"][key]["hi"] is None


def test_run_valuation_with_depreciation_reports_upside_but_headline_stays_suppressed(tmp_path):
    # SAME fixture, with Depreciation=100 added -- the ONLY difference from
    # the test above. capex_intensity=4000/1000=4.0>0.30, capex(4000)>
    # maintenance(max(100, 0.05*1000=50)=100) -> relief applies: growth_capex
    # =3900, ops_margin=-3.95+3.9=-0.05. Per the finance review, the relief
    # is NOT headlined: the headline scenarios keep using the actual (raw)
    # margin, so the base scenario stays suppressed exactly as in the
    # no-depreciation test; the relieved value is reported ONLY as an
    # explicitly-labeled aggressive UPSIDE inside capex_normalization.
    #
    # Same nonexistent-damodaran_dir trick as the sibling test above pins
    # terminal_growth to the hand-verified 0.025 (both the suppressed
    # -262.38 headline and the +265.35 upside below were derived under it,
    # WP3 fade-adjusted).
    normalized = _normalized({
        "Revenue": [_rec(2023, 1000.0)], "CapEx": [_rec(2023, 4000.0)],
        "Depreciation": [_rec(2023, 100.0)],
    })

    result = run_valuation(
        normalized, _CAPEX_RATIOS, _CAPEX_METRICS, price=None, price_df=None,
        assumptions=_assumptions(), sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    detail = result["hyper_growth_detail"]
    assert detail is not None

    capex_normalization = detail["capex_normalization"]
    assert capex_normalization is not None
    assert capex_normalization["applied"] is True
    assert capex_normalization["capex_intensity"] == pytest.approx(4.0)
    assert capex_normalization["maintenance_capex"] == pytest.approx(100.0)
    assert capex_normalization["growth_capex"] == pytest.approx(3900.0)
    assert capex_normalization["raw_current_margin"] == pytest.approx(-3.95)
    assert capex_normalization["ops_current_margin"] == pytest.approx(-0.05)

    # The relieved value is an aggressive UPSIDE, not the headline.
    assert capex_normalization["upside_per_share"] == pytest.approx(265.35, abs=0.05)
    assert capex_normalization["upside_lo"] == pytest.approx(236.90, abs=0.05)
    assert capex_normalization["upside_hi"] == pytest.approx(297.14, abs=0.05)

    # Headline stays suppressed: the base scenario uses the ACTUAL (raw)
    # margin, so it is the same suppressed value as the no-depreciation test.
    assert detail["suppressed"] is True
    base_ps = detail["scenarios"]["base"]["per_share"]
    assert base_ps is not None
    assert base_ps <= 0
    assert base_ps == pytest.approx(-262.38, abs=0.05)

    # fair_value_range stays empty (suppressed) -- the upside is never headlined.
    for key in ("bear", "base", "bull"):
        assert result["fair_value_range"][key]["lo"] is None
        assert result["fair_value_range"][key]["hi"] is None

    # Explanatory note names the growth-CapEx relief ("büyüme CapEx'i bakım
    # CapEx'inden ... ayrıldı") and flags it as an upside, not the headline.
    assert any("büyüme CapEx" in n for n in result["notes"])
