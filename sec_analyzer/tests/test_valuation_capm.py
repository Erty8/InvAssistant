"""Unit tests for sec_analyzer.valuation.capm.

Every expected number is hand-computed in the test body so the arithmetic is
independently checkable, not merely re-derived from the implementation. No
network / no files: sector_data is a small in-memory dict of the shape
``load_sector_data`` returns.
"""

import pytest

from sec_analyzer.interpret.rule_based import default_assumptions
from sec_analyzer.valuation.capm import (
    _COST_OF_EQUITY_MAX,
    _DEFAULT_TAX_RATE,
    capm_rate,
    compute_cost_of_equity,
    relever_beta,
)


def _sector_data(beta=1.5, erp=4.23, risk_free=4.20):
    return {
        "multiples": [
            {"industry": "Semiconductor", "pe": 28.4, "ps": 6.1, "pfcf": None,
             "growth": None, "peg": None, "beta": beta},
        ],
        "erp": erp,
        "risk_free": risk_free,
    }


_SIC_SEMI = "SEMICONDUCTORS & RELATED DEVICES"


# ---- pure helpers ---------------------------------------------------------


def test_relever_beta_hamada():
    # 1.0 * (1 + (1-0.25)*0.5) = 1.0 * (1 + 0.375) = 1.375
    assert relever_beta(1.0, 0.5, 0.25) == pytest.approx(1.375)


def test_relever_beta_zero_leverage_is_identity():
    assert relever_beta(1.3, 0.0, 0.25) == pytest.approx(1.3)


def test_capm_rate_percent_inputs_to_decimal():
    # (4.0 + 1.2*5.0) / 100 = (4.0 + 6.0)/100 = 0.10
    assert capm_rate(4.0, 1.2, 5.0) == pytest.approx(0.10)


# ---- orchestrator ---------------------------------------------------------


def test_compute_cost_of_equity_full_path():
    # unlevered 1.5, D/E = 300/6000 = 0.05, tax 0.25 ->
    # levered = 1.5 * (1 + 0.75*0.05) = 1.5 * 1.0375 = 1.55625
    # Ke = (4.20 + 1.55625*4.23)/100 = (4.20 + 6.5829375)/100 = 0.107829375
    result = compute_cost_of_equity(
        _sector_data(), _SIC_SEMI, {"total_debt": 300.0, "market_cap": 6000.0}
    )
    assert result is not None
    assert result["de_ratio"] == pytest.approx(0.05)
    assert result["levered_beta"] == pytest.approx(1.55625)
    assert result["rate"] == pytest.approx(0.107829375)
    assert result["clamped"] is False
    assert result["industry"] == "Semiconductor"
    assert "CAPM" in result["detail"]


def test_compute_cost_of_equity_missing_leverage_uses_de_zero():
    # No market_cap -> D/E = 0 -> levered == unlevered == 1.5
    # Ke = (4.20 + 1.5*4.23)/100 = (4.20 + 6.345)/100 = 0.10545
    result = compute_cost_of_equity(_sector_data(), _SIC_SEMI, {"total_debt": 300.0})
    assert result is not None
    assert result["de_ratio"] == 0.0
    assert result["levered_beta"] == pytest.approx(1.5)
    assert result["rate"] == pytest.approx(0.10545)


def test_compute_cost_of_equity_floors_low_rate():
    # Utility-like: unlevered 0.4, no leverage, erp 4.23, rf 4.20 ->
    # raw = (4.20 + 0.4*4.23)/100 = (4.20 + 1.692)/100 = 0.05892 -> floored to 0.07
    result = compute_cost_of_equity(
        _sector_data(beta=0.4), _SIC_SEMI, {}, is_unprofitable=False
    )
    assert result is not None
    assert result["rate"] == pytest.approx(0.07)
    assert result["clamped"] is True


def test_compute_cost_of_equity_unprofitable_floor_is_higher():
    # Same low raw rate, but unprofitable floor is 10%.
    result = compute_cost_of_equity(
        _sector_data(beta=0.4), _SIC_SEMI, {}, is_unprofitable=True
    )
    assert result["rate"] == pytest.approx(0.10)
    assert result["clamped"] is True


def test_compute_cost_of_equity_caps_high_rate():
    # Absurd leverage drives the rate above the cap.
    result = compute_cost_of_equity(
        _sector_data(beta=2.0), _SIC_SEMI,
        {"total_debt": 10000.0, "market_cap": 1000.0},  # D/E = 10
    )
    assert result is not None
    assert result["rate"] == pytest.approx(_COST_OF_EQUITY_MAX)
    assert result["clamped"] is True


def test_default_tax_rate_used_when_unspecified():
    result = compute_cost_of_equity(
        _sector_data(), _SIC_SEMI, {"total_debt": 300.0, "market_cap": 6000.0}
    )
    assert result["tax_rate"] == _DEFAULT_TAX_RATE


# ---- None / fallback paths ------------------------------------------------


def test_none_when_no_sector_data():
    assert compute_cost_of_equity(None, _SIC_SEMI, {"market_cap": 100.0}) is None


def test_none_when_beta_missing_for_sector():
    assert compute_cost_of_equity(_sector_data(beta=None), _SIC_SEMI, {}) is None


def test_none_when_risk_free_missing():
    data = _sector_data()
    data["risk_free"] = None
    assert compute_cost_of_equity(data, _SIC_SEMI, {}) is None


def test_none_when_erp_missing():
    data = _sector_data()
    data["erp"] = None
    assert compute_cost_of_equity(data, _SIC_SEMI, {}) is None


def test_none_when_sic_does_not_match_any_industry():
    assert compute_cost_of_equity(_sector_data(), "ZZZ NONSENSE QREW", {}) is None


def test_never_raises_on_garbage_metrics():
    # Non-numeric leverage fields must not raise; they degrade to D/E = 0.
    result = compute_cost_of_equity(
        _sector_data(), _SIC_SEMI, {"total_debt": "oops", "market_cap": None}
    )
    assert result is not None
    assert result["de_ratio"] == 0.0


# ---- integration with rule_based.default_assumptions ----------------------


def test_default_assumptions_uses_capm_base_when_present():
    capm = {"rate": 0.13, "detail": "CAPM: rf %4.2 + βL 1.9 × ERP %4.23 = %13.0"}
    a = default_assumptions({"revenue_cagr_5y": 0.10}, "mature", capm=capm)
    # base = CAPM rate; bear = +2pp; bull = -1pp.
    assert a["base"]["discount_rate"] == pytest.approx(0.13)
    assert a["bear"]["discount_rate"] == pytest.approx(0.15)
    assert a["bull"]["discount_rate"] == pytest.approx(0.12)
    assert "CAPM" in a["base"]["story"]


def test_default_assumptions_flat_default_without_capm():
    a = default_assumptions({"revenue_cagr_5y": 0.10}, "mature")
    assert a["base"]["discount_rate"] == pytest.approx(0.10)
    assert "sınıflandırmasına göre" in a["base"]["story"]


def test_default_assumptions_low_capm_base_is_floored_not_discarded():
    # A low CAPM base (0.07) makes bull's -1pp delta dip to 0.06, which would
    # be invalid; the clamp floors it to 0.07 and CAPM survives (base stays
    # 0.07) rather than reverting to the flat minimal-safe 0.10 base.
    capm = {"rate": 0.07, "detail": "CAPM: rf %4.2 + βL 0.5 × ERP %4.23 = %7.0"}
    a = default_assumptions({"revenue_cagr_5y": 0.05}, "mature", capm=capm)
    assert a["base"]["discount_rate"] == pytest.approx(0.07)
    assert a["bull"]["discount_rate"] >= 0.07
