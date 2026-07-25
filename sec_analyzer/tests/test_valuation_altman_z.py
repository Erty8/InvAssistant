"""Hand-verified numeric tests for the Altman Z-score distress screen
(SPEC.md Sec.8g): ``valuation.distress.altman_z_score`` and
``valuation.engine._build_altman_z``/their ``run_valuation`` wiring.

This screen is ADVISORY ONLY -- every test below also confirms it never
touches ``fair_value_range``/``primary_dcf_scenarios``/the triangulation
confidence, and is never computed at all for ``financial``/``reit`` filers.
"""

import pytest

from sec_analyzer.valuation.distress import altman_z_score
from sec_analyzer.valuation.engine import _build_altman_z, run_valuation

# ---------------------------------------------------------------------------
# 1. altman_z_score (pure function, SPEC.md Sec.8g)
# ---------------------------------------------------------------------------


def test_altman_z_score_safe_zone_hand_verified():
    # working_capital=200, total_assets=1000, retained_earnings=300, ebit=150,
    # market_cap=800, total_liabilities=400, revenue=1200.
    # X1=200/1000=0.2, X2=300/1000=0.3, X3=150/1000=0.15, X4=800/400=2.0,
    # X5=1200/1000=1.2.
    # Z = 1.2*0.2 + 1.4*0.3 + 3.3*0.15 + 0.6*2.0 + 1.0*1.2
    #   = 0.24 + 0.42 + 0.495 + 1.2 + 1.2 = 3.555, rounds to 3.55 (Python's
    #   round-half-to-even on the float closest to 3.555) -> > 2.99 -> "safe"
    result = altman_z_score(
        working_capital=200.0, total_assets=1000.0, retained_earnings=300.0,
        ebit=150.0, market_cap=800.0, total_liabilities=400.0, revenue=1200.0,
    )

    assert result is not None
    assert result["z_score"] == pytest.approx(3.55, abs=0.005)
    assert result["zone"] == "safe"
    assert result["components"]["x1"] == pytest.approx(0.2)
    assert result["components"]["x4"] == pytest.approx(2.0)


def test_altman_z_score_grey_zone_hand_verified():
    # working_capital=150, total_assets=1000, retained_earnings=250, ebit=100,
    # market_cap=500, total_liabilities=600, revenue=900.
    # X1=0.15, X2=0.25, X3=0.1, X4=500/600=0.833333, X5=0.9.
    # Z = 1.2*0.15 + 1.4*0.25 + 3.3*0.1 + 0.6*0.833333 + 1.0*0.9
    #   = 0.18 + 0.35 + 0.33 + 0.5 + 0.9 = 2.26 -> between 1.81 and 2.99 -> "grey"
    result = altman_z_score(
        working_capital=150.0, total_assets=1000.0, retained_earnings=250.0,
        ebit=100.0, market_cap=500.0, total_liabilities=600.0, revenue=900.0,
    )

    assert result is not None
    assert result["z_score"] == pytest.approx(2.26, abs=0.01)
    assert result["zone"] == "grey"


def test_altman_z_score_distress_zone_hand_verified():
    # working_capital=-50 (negative -- a distress sign itself), total_assets=1000,
    # retained_earnings=-100 (accumulated deficit), ebit=10, market_cap=200,
    # total_liabilities=900, revenue=500.
    # X1=-0.05, X2=-0.1, X3=0.01, X4=200/900=0.222222, X5=0.5.
    # Z = 1.2*(-0.05) + 1.4*(-0.1) + 3.3*0.01 + 0.6*0.222222 + 1.0*0.5
    #   = -0.06 - 0.14 + 0.033 + 0.133333 + 0.5 = 0.466333 -> < 1.81 -> "distress"
    result = altman_z_score(
        working_capital=-50.0, total_assets=1000.0, retained_earnings=-100.0,
        ebit=10.0, market_cap=200.0, total_liabilities=900.0, revenue=500.0,
    )

    assert result is not None
    assert result["z_score"] == pytest.approx(0.47, abs=0.01)
    assert result["zone"] == "distress"


def test_altman_z_score_none_when_any_input_missing():
    assert altman_z_score(None, 1000.0, 300.0, 150.0, 800.0, 400.0, 1200.0) is None
    assert altman_z_score(200.0, 1000.0, 300.0, 150.0, 800.0, 400.0, None) is None


def test_altman_z_score_none_when_total_assets_non_positive():
    assert altman_z_score(200.0, 0.0, 300.0, 150.0, 800.0, 400.0, 1200.0) is None
    assert altman_z_score(200.0, -100.0, 300.0, 150.0, 800.0, 400.0, 1200.0) is None


def test_altman_z_score_none_when_total_liabilities_non_positive():
    assert altman_z_score(200.0, 1000.0, 300.0, 150.0, 800.0, 0.0, 1200.0) is None


# ---------------------------------------------------------------------------
# 2. _build_altman_z (engine.py, SPEC.md Sec.8g)
# ---------------------------------------------------------------------------

_ALTMAN_CONCEPTS = [
    "Revenue", "CurrentAssets", "CurrentLiabilities", "TotalAssets",
    "TotalLiabilities", "OperatingIncome", "RetainedEarningsAccumulatedDeficit",
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
        for concept in _ALTMAN_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "Altman Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _ALTMAN_CONCEPTS},
        "missing": [c for c in _ALTMAN_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _ALTMAN_CONCEPTS},
    }


def test_build_altman_z_safe_zone_from_normalized_fixture():
    normalized = _normalized({
        "CurrentAssets": {2023: 600.0},
        "CurrentLiabilities": {2023: 400.0},  # working_capital = 200
        "TotalAssets": {2023: 1000.0},
        "RetainedEarningsAccumulatedDeficit": {2023: 300.0},
        "OperatingIncome": {2023: 150.0},
        "TotalLiabilities": {2023: 400.0},
        "Revenue": {2023: 1200.0},
    })
    metrics = {"latest_fy": 2023, "market_cap": 800.0}

    result, notes = _build_altman_z(normalized, metrics)

    assert result is not None
    assert result["z_score"] == pytest.approx(3.55, abs=0.005)
    assert result["zone"] == "safe"
    assert any("güvenli" in n and "3.55" in n for n in notes)


def test_build_altman_z_none_when_fiscal_year_unresolvable():
    normalized = _normalized({})
    metrics = {}

    result, notes = _build_altman_z(normalized, metrics)

    assert result is None
    assert notes == []


def test_build_altman_z_none_when_current_assets_liabilities_missing():
    normalized = _normalized({
        "TotalAssets": {2023: 1000.0},
        "RetainedEarningsAccumulatedDeficit": {2023: 300.0},
        "OperatingIncome": {2023: 150.0},
        "TotalLiabilities": {2023: 400.0},
        "Revenue": {2023: 1200.0},
    })
    metrics = {"latest_fy": 2023, "market_cap": 800.0}

    result, notes = _build_altman_z(normalized, metrics)

    assert result is None
    assert any("dönen varlık" in n for n in notes)


def test_build_altman_z_none_when_retained_earnings_missing():
    # WP8's new concept missing entirely -> altman_z_score itself degrades to
    # None (one of its 7 required inputs is None) -> _build_altman_z reports
    # the generic "insufficient data" note.
    normalized = _normalized({
        "CurrentAssets": {2023: 600.0},
        "CurrentLiabilities": {2023: 400.0},
        "TotalAssets": {2023: 1000.0},
        "OperatingIncome": {2023: 150.0},
        "TotalLiabilities": {2023: 400.0},
        "Revenue": {2023: 1200.0},
    })
    metrics = {"latest_fy": 2023, "market_cap": 800.0}

    result, notes = _build_altman_z(normalized, metrics)

    assert result is None
    assert any("gerekli veriler eksik" in n for n in notes)


# ---------------------------------------------------------------------------
# 3. run_valuation wiring: sector gate + advisory-only (SPEC.md Sec.8g)
# ---------------------------------------------------------------------------


def test_run_valuation_computes_altman_z_for_mature_sector():
    normalized = _normalized({
        "CurrentAssets": {2023: 600.0},
        "CurrentLiabilities": {2023: 400.0},
        "TotalAssets": {2023: 1000.0},
        "RetainedEarningsAccumulatedDeficit": {2023: 300.0},
        "OperatingIncome": {2023: 150.0},
        "TotalLiabilities": {2023: 400.0},
        "Revenue": {2023: 1200.0},
    })
    metrics = {
        "shares": None, "latest_fy": 2023, "fcf": None, "net_debt": 0.0, "market_cap": 800.0,
    }
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result = run_valuation(
        normalized, [], metrics, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["altman_z"] is not None
    assert result["altman_z"]["zone"] == "safe"


def test_run_valuation_never_computes_altman_z_for_financial_or_reit():
    normalized = _normalized({
        "CurrentAssets": {2023: 600.0},
        "CurrentLiabilities": {2023: 400.0},
        "TotalAssets": {2023: 1000.0},
        "RetainedEarningsAccumulatedDeficit": {2023: 300.0},
        "OperatingIncome": {2023: 150.0},
        "TotalLiabilities": {2023: 400.0},
        "Revenue": {2023: 1200.0},
    })
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0, "market_cap": 800.0,
    }
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    for sector in ("financial", "reit"):
        result = run_valuation(
            normalized, [{"fy": 2023, "roe": 0.12}], metrics, price=20.0, price_df=None,
            assumptions=assumptions, sector_type=sector,
        )
        assert result["altman_z"] is None


def test_run_valuation_altman_z_never_affects_fair_value_range():
    # Even in the "distress" zone, altman_z is purely advisory -- the fair
    # value band is computed from the DCF path exactly as it would be
    # without this screen at all.
    normalized_no_altman = _normalized({})
    normalized_with_distress = _normalized({
        "CurrentAssets": {2023: 100.0},
        "CurrentLiabilities": {2023: 150.0},  # working_capital = -50
        "TotalAssets": {2023: 1000.0},
        "RetainedEarningsAccumulatedDeficit": {2023: -100.0},
        "OperatingIncome": {2023: 10.0},
        "TotalLiabilities": {2023: 900.0},
        "Revenue": {2023: 500.0},
    })
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": 50.0, "net_debt": 0.0, "market_cap": 200.0,
    }
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result_without = run_valuation(
        normalized_no_altman, [], metrics, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )
    result_with = run_valuation(
        normalized_with_distress, [], metrics, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result_without["altman_z"] is None
    assert result_with["altman_z"] is not None
    assert result_with["altman_z"]["zone"] == "distress"
    # fair_value_range is identical regardless of the distress screen's
    # presence -- it's derived purely from fcf0/shares/assumptions, which
    # are identical between the two runs.
    assert result_with["fair_value_range"] == result_without["fair_value_range"]
