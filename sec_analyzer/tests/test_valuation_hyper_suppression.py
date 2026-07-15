"""Tests for the hyper-grower revenue-first DCF's non-credible-negative-value
suppression guard (``valuation.engine._build_hyper_growth``'s
``suppressed``/``suppressed_reason`` fields and their wiring into
``run_valuation``'s ``hyper_growth_detail``/``fair_value_range``/
``triangulation`` output).

Background (see the comment block above the guard in
``sec_analyzer/valuation/engine.py``, "Non-credible negative valuation
guard"): a capex-heavy hyper-grower (e.g. APLD/Applied Digital, where growth
CapEx is many multiples of revenue) can have a current FCF margin so deeply
negative that the revenue-first DCF's discounted early-year cash burn
exceeds its positive terminal value, producing a negative per-share equity
value for the base scenario. That is not a usable number, so the engine:

- keeps ``hyper_growth = True`` (detection still fired, scenarios stay in
  ``hyper_growth_detail`` for transparency), but
- sets ``hyper_growth_detail["suppressed"] = True`` with a non-empty Turkish
  ``suppressed_reason``,
- empties the headline ``fair_value_range`` (bear/base/bull all
  ``lo``/``hi`` = None) since neither the hyper band nor the P/B x ROE
  anchor apply here, and
- drops the DCF vote in ``triangulation`` back to ``"veri_yok"`` (no usable
  DCF band to compare price against).

This module does not re-derive revenue-first DCF arithmetic by hand (that is
already hand-verified in ``test_valuation_dcf.py``/the hyper-grower fixture
in ``test_valuation_engine.py``); instead it demonstrates -- with a rough
order-of-magnitude argument in the comment above Test A -- why the chosen
fixture's base per-share must come out <= 0, and asserts the resulting
suppression contract end-to-end.

Fixture shapes (``normalized``/``ratios``/``metrics``/``assumptions``,
``run_valuation`` call signature) mirror
``test_valuation_engine.py``'s ``_HYPER_*`` fixtures (Sec.12).
"""

import pytest

from sec_analyzer.valuation.engine import run_valuation

# ---------------------------------------------------------------------------
# Fixture helpers (mirrors test_valuation_engine.py's _rec/_normalized/_assumptions)
# ---------------------------------------------------------------------------

_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity",
]


def _rec(fy, value, end=None):
    return {
        "concept": None, "tag": None, "period_end": end or f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(overrides):
    annual = {c: overrides.get(c) for c in _CONCEPTS}
    return {
        "cik": 1, "entity_name": "Hyper Suppression Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CONCEPTS},
        "missing": [c for c in _CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CONCEPTS},
    }


def _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10):
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": base_growth, "terminal_growth": base_terminal, "discount_rate": base_discount, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


# ---------------------------------------------------------------------------
# Test A -- capex-heavy hyper-grower (APLD-shaped) is suppressed
# ---------------------------------------------------------------------------

# APLD-shaped fixture: small revenue growing very fast, but FCF deeply
# negative because growth CapEx dwarfs revenue.
#   latest_revenue = 144.0, fcf = -917.0, sbc = 0 (not supplied)
#   -> current_margin = (fcf - sbc) / revenue = -917.0 / 144.0 = -6.3681
#      (i.e. -636.8% FCF margin today).
# detect_hyper_grower: revenue_cagr_5y=1.5 (150%) > 0.25 strong-tier growth
# gate fires; clause (a) fcf=-917<=0 also fires independently -> triggered.
#
# Why the base scenario's per_share must come out <= 0 (order-of-magnitude,
# not an exact re-derivation -- the exact arithmetic lives in
# revenue_dcf.revenue_first_dcf, hand-verified elsewhere):
# start_growth for base = min(realized_cagr=1.5, cap 0.40) = 0.40, and the
# FCF margin path linearly interpolates from current_margin=-6.3681 up to a
# small positive target (<=0.30) over steady_state_year=10 years. Year 1
# revenue = 144*1.40 = 201.6, margin ~= -6.3681 + (target-(-6.3681))/10 ~=
# -5.72 -> FCF_1 ~= 201.6 * -5.72 ~= -1153. Every one of the first several
# years is comparably deeply negative (revenue keeps compounding at ~25-40%
# while margin is still deeply negative), so the UNDISCOUNTED cumulative
# burn alone is already many billions of dollars -- several orders of
# magnitude larger than any plausible discounted terminal value off a base
# of ~144 revenue growing to a multiple of that over 10 years at a <=30%
# margin (terminal FCF is at most on the order of a few hundred million,
# discounted back 10 years at 10% divides it by ~2.6 again). The NPV of the
# burn therefore dominates the NPV of the terminal value, so equity value
# (and hence per_share, dividing by ~280 shares plus dilution) is negative.
_APLD_CONCEPTS_OVERRIDES = {"Revenue": [_rec(2023, 144.0)]}
_APLD_RATIOS = [
    {"fy": 2023, "gross_margin": 0.20, "fcf": -917.0},
]
_APLD_METRICS = {
    "shares": 280.0, "latest_fy": 2023, "fcf": -917.0, "net_debt": 0.0,
    "revenue_cagr_5y": 1.5, "rnd_revenue": 0.0, "sbc_revenue": 0.0, "shares_yoy": None,
}


def test_hyper_grower_capex_heavy_negative_base_is_suppressed():
    normalized = _normalized(_APLD_CONCEPTS_OVERRIDES)
    assumptions = _assumptions()

    result = run_valuation(
        normalized, _APLD_RATIOS, _APLD_METRICS, price=28.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    # Detection still fires (mode stays "detected") even though it gets
    # suppressed downstream.
    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail is not None

    assert detail["suppressed"] is True
    assert isinstance(detail["suppressed_reason"], str)
    assert detail["suppressed_reason"]  # non-empty/truthy

    # The base scenario itself must actually be non-credible (per_share <= 0)
    # -- this is the condition the guard keys off of (SPEC contract).
    base_per_share = detail["scenarios"]["base"]["per_share"]
    assert base_per_share is not None
    assert base_per_share <= 0

    # Headline fair_value_range is emptied -- no usable DCF or P/B x ROE
    # anchor applies for this (non-financial) sector once the hyper band is
    # dropped, so every scenario's lo/hi must be None.
    for key in ("bear", "base", "bull"):
        assert result["fair_value_range"][key]["lo"] is None
        assert result["fair_value_range"][key]["hi"] is None

    # Triangulation's DCF vote is dropped back to "no data" rather than
    # comparing price against a negative (non-credible) band.
    assert result["triangulation"]["signals"]["dcf"] == "veri_yok"

    # Explanatory Turkish note surfaces at the top level too.
    assert any(
        "revenue-first DCF baz senaryosu negatif özkaynak" in n or "capex" in n.lower()
        for n in result["notes"]
    )


# ---------------------------------------------------------------------------
# Test B -- healthy hyper-grower (positive base per_share) is NOT suppressed
# ---------------------------------------------------------------------------

# Reuses the exact hyper-grower fixture already hand-verified end-to-end in
# test_valuation_engine.py's
# test_run_valuation_hyper_grower_uses_revenue_first_dcf_as_headline (Sec.12):
# revenue0=1000, fcf=-50 (fcf_margin=-5%, mild), gross_margin=0.60,
# revenue_cagr_5y=0.50, shares=100, price=50.0 -> base scenario
# start_growth=0.40, target_fcf_margin=0.30, discount_rate=0.12 (fixed hyper
# base rate), per_share~=99.97 (positive), and the base band (min/max over
# the +/-2pp start_growth x +/-1pp discount grid) is lo=79.91, hi=126.53.
# This is a regression guard that the new suppression logic does NOT fire on
# a normal (profitable-enough-at-target) hyper-grower.
_HEALTHY_CONCEPTS_OVERRIDES = {"Revenue": [_rec(2023, 1000.0)]}
_HEALTHY_RATIOS = [
    {"fy": 2023, "gross_margin": 0.60, "fcf": -50.0},
    {"fy": 2022, "fcf": 100.0},
    {"fy": 2021, "fcf": 90.0},
]
_HEALTHY_METRICS = {
    "shares": 100.0, "latest_fy": 2023, "fcf": -50.0, "net_debt": 0.0,
    "revenue_cagr_5y": 0.50, "rnd_revenue": 0.0, "sbc_revenue": 0.0, "shares_yoy": None,
}


def test_hyper_grower_healthy_positive_base_is_not_suppressed():
    normalized = _normalized(_HEALTHY_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, _HEALTHY_RATIOS, _HEALTHY_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail is not None

    assert detail["suppressed"] is False
    assert detail["suppressed_reason"] is None

    base_per_share = detail["scenarios"]["base"]["per_share"]
    assert base_per_share is not None
    assert base_per_share > 0
    assert base_per_share == pytest.approx(99.97, abs=0.05)

    # Headline fair_value_range.base is populated from the hyper band (not
    # emptied) -- same numbers as test_valuation_engine.py's hand-verified
    # grid derivation (lo=min, hi=max over the 3x3 start_growth x discount
    # grid): lo=79.91, hi=126.53.
    base_fvr = result["fair_value_range"]["base"]
    assert base_fvr["lo"] == pytest.approx(79.91)
    assert base_fvr["hi"] == pytest.approx(126.53)
    assert isinstance(base_fvr["lo"], (int, float))
    assert isinstance(base_fvr["hi"], (int, float))

    assert result["triangulation"]["signals"]["dcf"] != "veri_yok"
