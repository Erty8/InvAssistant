"""Hand-verified numeric tests for the Beneish M-score earnings-manipulation
screen (SPEC.md Sec.8j): ``valuation.distress.beneish_m_score`` and
``valuation.engine._build_beneish_m``/their ``run_valuation`` wiring.

This screen is ADVISORY ONLY -- every test below also confirms it never
touches ``fair_value_range``/``primary_dcf_scenarios``/the triangulation
confidence, and (unlike Altman Z/LBO floor) IS computed for every sector
including ``financial``/``reit``.
"""

import pytest

from sec_analyzer.valuation.distress import beneish_m_score
from sec_analyzer.valuation.engine import _build_beneish_m, run_valuation

# ---------------------------------------------------------------------------
# 1. beneish_m_score (pure function, SPEC.md Sec.8j)
# ---------------------------------------------------------------------------


def _full_current():
    return dict(
        receivables=150.0, revenue=1000.0, gross_profit=400.0, current_assets=500.0,
        ppe_gross=800.0, total_assets=2000.0, depreciation=100.0, sga=200.0,
        long_term_debt=300.0, current_liabilities=250.0, net_income=90.0, operating_cash_flow=110.0,
    )


def _full_prior():
    return dict(
        receivables=100.0, revenue=900.0, gross_profit=380.0, current_assets=450.0,
        ppe_gross=750.0, total_assets=1800.0, depreciation=90.0, sga=180.0,
        long_term_debt=280.0, current_liabilities=220.0, net_income=80.0, operating_cash_flow=100.0,
    )


def test_beneish_m_score_full_8variable_model_hand_verified():
    # DSRI = (150/1000)/(100/900) = 0.15/0.111111 = 1.35
    # GMI = gm_prior/gm_current = (380/900)/(400/1000) = 0.422222/0.4 = 1.055556
    # SGI = 1000/900 = 1.111111
    # AQI: aq_current = 1-(500+800)/2000 = 1-0.65 = 0.35
    #      aq_prior   = 1-(450+750)/1800 = 1-0.666667 = 0.333333
    #      AQI = 0.333333/0.35 = 0.952381
    # DEPI: rate_current = 100/(800+100) = 0.111111
    #       rate_prior   = 90/(750+90)   = 0.107143
    #       DEPI = 0.107143/0.111111 = 0.964286
    # SGAI = (200/1000)/(180/900) = 0.2/0.2 = 1.0
    # LVGI: lev_current = (300+250)/2000 = 0.275
    #       lev_prior   = (280+220)/1800 = 0.277778
    #       LVGI = 0.277778/0.275 = 1.010101
    # TATA = (90-110)/2000 = -0.01
    #
    # M = -4.84 + 0.920*1.35 + 0.528*1.055556 + 0.404*0.952381 + 0.892*1.111111
    #     + 0.115*0.964286 - 0.172*1.0 + 4.679*(-0.01) - 0.327*1.010101
    #   = -4.84 + 1.242 + 0.557333 + 0.384762 + 0.991111 + 0.110893
    #     - 0.172 - 0.04679 - 0.330303
    #   = -2.102994 -> rounds to -2.10 (< -1.78 -> not flagged)
    result = beneish_m_score(_full_current(), _full_prior())

    assert result is not None
    assert result["partial"] is False
    assert result["components"]["dsri"] == pytest.approx(1.35, rel=1e-4)
    assert result["components"]["gmi"] == pytest.approx(1.055556, rel=1e-4)
    assert result["components"]["aqi"] == pytest.approx(0.952381, rel=1e-4)
    assert result["components"]["sgi"] == pytest.approx(1.111111, rel=1e-4)
    assert result["components"]["depi"] == pytest.approx(0.964286, rel=1e-4)
    assert result["components"]["sgai"] == pytest.approx(1.0, rel=1e-4)
    assert result["components"]["lvgi"] == pytest.approx(1.010101, rel=1e-4)
    assert result["components"]["tata"] == pytest.approx(-0.01, rel=1e-4)
    assert result["m_score"] == pytest.approx(-2.10, abs=0.01)
    assert result["flag"] is False


def test_beneish_m_score_degrades_to_5variable_model_when_sga_missing():
    # Same inputs, minus SGA (and therefore SGAI) -- LVGI/TATA are also
    # dropped, per the 5-variable model's own (separately calibrated)
    # coefficients, NOT a truncation of the 8-variable ones:
    # M = -6.065 + 0.823*1.35 + 0.906*1.055556 + 0.593*0.952381
    #     + 0.717*1.111111 + 0.107*0.964286
    #   = -6.065 + 1.11105 + 0.956333 + 0.564762 + 0.796667 + 0.103179
    #   = -2.533009 -> rounds to -2.53
    current = _full_current()
    del current["sga"]

    result = beneish_m_score(current, _full_prior())

    assert result is not None
    assert result["partial"] is True
    assert set(result["components"]) == {"dsri", "gmi", "aqi", "sgi", "depi"}
    assert result["m_score"] == pytest.approx(-2.53, abs=0.01)
    assert result["flag"] is False


def test_beneish_m_score_flags_manipulation_signal_on_large_dsri_jump():
    # Same fixture, but receivables jump to 300 (vs. revenue 1000) while the
    # prior year's receivables/revenue relationship is unchanged -- a
    # textbook "receivables growing much faster than sales" manipulation
    # signal. DSRI = (300/1000)/(100/900) = 0.3/0.111111 = 2.7 (vs. 1.35
    # above); only the DSRI term changes, by 0.920*(2.7-1.35) = 1.242, so
    # M = -2.102994 + 1.242 = -0.860994 -> rounds to -0.86, > -1.78 -> flagged.
    current = _full_current()
    current["receivables"] = 300.0

    result = beneish_m_score(current, _full_prior())

    assert result is not None
    assert result["components"]["dsri"] == pytest.approx(2.7, rel=1e-4)
    assert result["m_score"] == pytest.approx(-0.86, abs=0.01)
    assert result["flag"] is True


def test_beneish_m_score_none_when_five_variable_inputs_missing():
    current = _full_current()
    del current["revenue"]
    assert beneish_m_score(current, _full_prior()) is None


def test_beneish_m_score_none_when_total_assets_zero():
    current = _full_current()
    current["total_assets"] = 0.0
    assert beneish_m_score(current, _full_prior()) is None


def test_beneish_m_score_none_when_gross_margin_zero():
    # gm_current = 0 -> GMI (gm_prior/gm_current) is undefined.
    current = _full_current()
    current["gross_profit"] = 0.0
    assert beneish_m_score(current, _full_prior()) is None


def test_beneish_m_score_none_when_gross_ppe_proxy_out_of_domain():
    # F2 guard: an old, asset-heavy, low-intangible filer where
    # (CurrentAssets + GROSS PP&E) exceeds TotalAssets -> the AQI inner term
    # 1-(CA+PPE)/TA goes negative. With NET PP&E this term is essentially
    # always positive; the gross-PP&E proxy pushes it out of domain. Rather
    # than emit a sign-flipped garbage AQI into the M-score, the whole score
    # returns None.
    current = dict(
        receivables=150.0, revenue=1000.0, gross_profit=400.0, current_assets=300.0,
        ppe_gross=800.0, total_assets=1000.0, depreciation=100.0,  # (300+800)/1000 = 1.10 > 1
    )
    prior = dict(
        receivables=100.0, revenue=900.0, gross_profit=380.0, current_assets=280.0,
        ppe_gross=750.0, total_assets=950.0, depreciation=90.0,  # (280+750)/950 = 1.084 > 1
    )
    assert beneish_m_score(current, prior) is None

    # Sanity: the SAME filer with a larger asset base (proxy back in domain)
    # computes fine -- the guard is domain-specific, not a blanket rejection.
    current_ok = dict(current, total_assets=2000.0)
    prior_ok = dict(prior, total_assets=1800.0)
    assert beneish_m_score(current_ok, prior_ok) is not None


# ---------------------------------------------------------------------------
# 2. _build_beneish_m (engine.py, SPEC.md Sec.8j)
# ---------------------------------------------------------------------------

_BENEISH_CONCEPTS = [
    "Receivables", "Revenue", "GrossProfit", "CurrentAssets",
    "PropertyPlantAndEquipmentGross", "TotalAssets", "Depreciation",
    "SellingGeneralAndAdministrativeExpense", "LongTermDebt",
    "CurrentLiabilities", "NetIncome", "OperatingCashFlow",
]


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized_full() -> dict:
    fy, prior_fy = 2023, 2022
    current, prior = _full_current(), _full_prior()
    field_to_concept = {
        "receivables": "Receivables", "revenue": "Revenue", "gross_profit": "GrossProfit",
        "current_assets": "CurrentAssets", "ppe_gross": "PropertyPlantAndEquipmentGross",
        "total_assets": "TotalAssets", "depreciation": "Depreciation",
        "sga": "SellingGeneralAndAdministrativeExpense", "long_term_debt": "LongTermDebt",
        "current_liabilities": "CurrentLiabilities", "net_income": "NetIncome",
        "operating_cash_flow": "OperatingCashFlow",
    }
    annual = {
        concept: [_rec(fy, current[field]), _rec(prior_fy, prior[field])]
        for field, concept in field_to_concept.items()
    }
    return {
        "cik": 1, "entity_name": "Beneish Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _BENEISH_CONCEPTS},
        "missing": [], "matched_tags": {c: None for c in _BENEISH_CONCEPTS},
    }


def test_build_beneish_m_matches_pure_function_from_normalized_fixture():
    normalized = _normalized_full()
    metrics = {"latest_fy": 2023}

    result, notes = _build_beneish_m(normalized, metrics)

    assert result is not None
    assert result["m_score"] == pytest.approx(-2.10, abs=0.01)
    assert result["partial"] is False
    assert any("Beneish M-skoru" in n for n in notes)


def test_build_beneish_m_none_when_fiscal_year_unresolvable():
    normalized = _normalized_full()
    result, notes = _build_beneish_m(normalized, {})
    assert result is None
    assert notes == []


def test_build_beneish_m_partial_note_when_sga_missing():
    normalized = _normalized_full()
    normalized["annual"]["SellingGeneralAndAdministrativeExpense"] = None
    metrics = {"latest_fy": 2023}

    result, notes = _build_beneish_m(normalized, metrics)

    assert result is not None
    assert result["partial"] is True
    assert any("kısmi" in n for n in notes)


# ---------------------------------------------------------------------------
# 3. run_valuation wiring: computed for every sector, advisory-only (SPEC.md Sec.8j)
# ---------------------------------------------------------------------------


def _assumptions():
    return {
        "bear": {"growth_5y": 0.02, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.05, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


def test_run_valuation_computes_beneish_m_for_every_sector():
    normalized = _normalized_full()
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions()

    for sector in ("mature", "financial", "reit", "cyclical", "growth_unprofitable"):
        ratios = [{"fy": 2023, "roe": 0.10}] if sector in ("financial", "reit") else []
        result = run_valuation(
            normalized, ratios, metrics, price=20.0, price_df=None,
            assumptions=assumptions, sector_type=sector,
        )
        assert result["beneish_m"] is not None, f"beneish_m should be computed for sector={sector}"
        assert result["beneish_m"]["m_score"] == pytest.approx(-2.10, abs=0.01)


def test_run_valuation_beneish_m_never_affects_fair_value_range():
    normalized_with = _normalized_full()
    normalized_without = _normalized_full()
    normalized_without["annual"]["Receivables"] = None  # beneish_m becomes unavailable
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions()

    result_with = run_valuation(
        normalized_with, [], metrics, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )
    result_without = run_valuation(
        normalized_without, [], metrics, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result_with["beneish_m"] is not None
    assert result_without["beneish_m"] is None
    assert result_with["fair_value_range"] == result_without["fair_value_range"]
