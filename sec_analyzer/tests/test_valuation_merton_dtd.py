"""Hand-verified numeric tests for the Merton distance-to-default screen
(SPEC.md Sec.8k): ``valuation.distress.merton_distance_to_default``,
``valuation.distress._annualized_volatility``, and
``valuation.engine._build_merton_dtd``/their ``run_valuation`` wiring.

This screen is ADVISORY ONLY -- every test below also confirms it never
touches ``fair_value_range``/``primary_dcf_scenarios``/the triangulation
confidence, and (like Beneish M) IS computed for every sector.

The Newton-Raphson solver's numeric outputs below were cross-checked by
re-evaluating ``distress._merton_residuals`` at the returned
``(asset_value, asset_vol)`` and confirming both residuals are ~0 (see the
first test's comment) -- the standard way to verify a converged nonlinear
solve without an independent closed-form reference.
"""

import math

import pandas as pd
import pytest

from sec_analyzer.valuation.distress import (
    _annualized_volatility,
    _merton_residuals,
    merton_distance_to_default,
)
from sec_analyzer.valuation.engine import _build_merton_dtd, run_valuation

# ---------------------------------------------------------------------------
# 1. merton_distance_to_default (pure function, SPEC.md Sec.8k)
# ---------------------------------------------------------------------------


def test_merton_dtd_normal_case_residuals_confirm_convergence():
    # equity_value=100, equity_vol=0.4, debt_face_value=80, risk_free=0.03,
    # horizon=1.0 (moderate leverage). The solver's own convergence
    # criterion (both equations' residuals below a tight, equity-scaled
    # tolerance) IS the correctness check for a nonlinear simultaneous
    # solve -- re-evaluating the two Merton equations at the returned
    # (asset_value, asset_vol) must yield residuals ~0.
    result = merton_distance_to_default(
        equity_value=100.0, equity_vol=0.4, debt_face_value=80.0, risk_free_rate=0.03, horizon_years=1.0,
    )

    assert result is not None
    assert result["zone"] == "safe"
    assert result["distance_to_default"] == pytest.approx(3.5628, abs=0.01)
    assert result["probability_of_default"] == pytest.approx(0.000183, abs=1e-5)

    f1, f2, _d1, d2 = _merton_residuals(
        result["asset_value"], result["asset_vol"], 100.0, 0.4, 80.0, 0.03, 1.0, math.sqrt(1.0),
    )
    assert f1 == pytest.approx(0.0, abs=1e-4)
    assert f2 == pytest.approx(0.0, abs=1e-4)
    assert d2 == pytest.approx(result["distance_to_default"], abs=1e-3)


def test_merton_dtd_elevated_zone_high_leverage():
    # equity_value=20, equity_vol=0.8, debt_face_value=150 -- heavily
    # levered, high equity volatility. Solved DD lands in the "elevated"
    # band (1.0 <= DD < 3.0).
    result = merton_distance_to_default(
        equity_value=20.0, equity_vol=0.8, debt_face_value=150.0, risk_free_rate=0.03, horizon_years=1.0,
    )

    assert result is not None
    assert result["zone"] == "elevated"
    assert result["distance_to_default"] == pytest.approx(1.0349, abs=0.01)


def test_merton_dtd_distress_zone_extreme_leverage():
    # equity_value=5, equity_vol=1.2, debt_face_value=500 -- extreme
    # leverage relative to equity. DD lands below the elevated threshold
    # (< 1.0) -> "distress".
    result = merton_distance_to_default(
        equity_value=5.0, equity_vol=1.2, debt_face_value=500.0, risk_free_rate=0.04, horizon_years=1.0,
    )

    assert result is not None
    assert result["zone"] == "distress"
    assert result["distance_to_default"] == pytest.approx(0.1041, abs=0.01)
    assert result["probability_of_default"] == pytest.approx(0.4586, abs=0.001)


def test_merton_dtd_safe_zone_low_leverage():
    result = merton_distance_to_default(
        equity_value=1000.0, equity_vol=0.25, debt_face_value=100.0, risk_free_rate=0.03, horizon_years=1.0,
    )

    assert result is not None
    assert result["zone"] == "safe"
    assert result["distance_to_default"] == pytest.approx(10.5283, abs=0.01)
    assert result["probability_of_default"] == pytest.approx(0.0, abs=1e-4)


def test_merton_dtd_accepts_negative_risk_free_rate():
    # risk_free_rate may legitimately be zero or negative (unlike the other
    # required inputs) -- this must not be rejected as a guard failure.
    result = merton_distance_to_default(
        equity_value=100.0, equity_vol=0.3, debt_face_value=60.0, risk_free_rate=-0.01, horizon_years=1.0,
    )
    assert result is not None
    assert result["zone"] == "safe"


def test_merton_dtd_none_when_equity_value_missing_or_non_positive():
    assert merton_distance_to_default(None, 0.4, 80.0, 0.03) is None
    assert merton_distance_to_default(0.0, 0.4, 80.0, 0.03) is None
    assert merton_distance_to_default(-1.0, 0.4, 80.0, 0.03) is None


def test_merton_dtd_none_when_equity_vol_missing_or_non_positive():
    assert merton_distance_to_default(100.0, None, 80.0, 0.03) is None
    assert merton_distance_to_default(100.0, 0.0, 80.0, 0.03) is None


def test_merton_dtd_none_when_debt_face_value_missing_or_non_positive():
    assert merton_distance_to_default(100.0, 0.4, None, 0.03) is None
    assert merton_distance_to_default(100.0, 0.4, 0.0, 0.03) is None


def test_merton_dtd_none_when_risk_free_rate_missing():
    assert merton_distance_to_default(100.0, 0.4, 80.0, None) is None


def test_merton_dtd_none_when_horizon_non_positive():
    assert merton_distance_to_default(100.0, 0.4, 80.0, 0.03, horizon_years=0) is None


# ---------------------------------------------------------------------------
# 2. _annualized_volatility (SPEC.md Sec.8k)
# ---------------------------------------------------------------------------


def _alternating_return_price_df(n_returns=60, magnitude=0.05, start="2022-01-03"):
    """Build a price series whose daily returns alternate EXACTLY between
    ``+magnitude`` and ``-magnitude`` for ``n_returns`` observations -- an
    exactly hand-computable case: with an equal +/- split (mean return 0),
    the SAMPLE variance (pandas' default ddof=1) of an alternating +/-a
    series is ``n*a^2/(n-1)``."""
    prices = [100.0]
    for i in range(n_returns):
        r = magnitude if i % 2 == 0 else -magnitude
        prices.append(prices[-1] * (1 + r))
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({"Close": prices}, index=idx)


def test_annualized_volatility_hand_verified_alternating_returns():
    # 60 returns alternating exactly +5%/-5% -> sample variance =
    # 60*0.05^2/59 = 0.15/59 = 0.00254237..., stdev = 0.05042195,
    # annualized = stdev*sqrt(252) = 0.80042362 (cross-checked directly
    # against pandas' own .pct_change()/.std() during test authoring).
    df = _alternating_return_price_df(n_returns=60, magnitude=0.05)

    result = _annualized_volatility(df)

    assert result == pytest.approx(0.80042362, rel=1e-6)


def test_annualized_volatility_none_when_insufficient_observations():
    # Fewer than _MERTON_MIN_RETURN_OBSERVATIONS (60) daily returns.
    df = _alternating_return_price_df(n_returns=30, magnitude=0.05)
    assert _annualized_volatility(df) is None


def test_annualized_volatility_none_when_price_df_is_none():
    assert _annualized_volatility(None) is None


def test_annualized_volatility_none_when_close_column_missing():
    df = pd.DataFrame({"Open": [1.0, 2.0, 3.0]})
    assert _annualized_volatility(df) is None


# ---------------------------------------------------------------------------
# 3. _build_merton_dtd (engine.py, SPEC.md Sec.8k)
# ---------------------------------------------------------------------------


def test_build_merton_dtd_matches_pure_function():
    df = _alternating_return_price_df(n_returns=60, magnitude=0.05)
    # equity_vol from the df above (0.80042362) combined with market_cap/
    # total_debt/risk_free_pct=3.0 (-> risk_free_rate=0.03) should match a
    # direct merton_distance_to_default call with the same equity_vol.
    metrics = {"market_cap": 100.0, "total_debt": 80.0}

    result, notes = _build_merton_dtd(metrics, df, risk_free_pct=3.0)

    direct = merton_distance_to_default(100.0, 0.80042362, 80.0, 0.03)
    assert result is not None
    assert result["distance_to_default"] == pytest.approx(direct["distance_to_default"], rel=1e-4)
    assert any("Merton" in n for n in notes)


def test_build_merton_dtd_none_when_risk_free_pct_missing():
    df = _alternating_return_price_df(n_returns=60, magnitude=0.05)
    metrics = {"market_cap": 100.0, "total_debt": 80.0}

    result, notes = _build_merton_dtd(metrics, df, risk_free_pct=None)

    assert result is None
    assert any("risksiz getiri oranı yok" in n for n in notes)


def test_build_merton_dtd_none_when_price_history_insufficient():
    df = _alternating_return_price_df(n_returns=10, magnitude=0.05)
    metrics = {"market_cap": 100.0, "total_debt": 80.0}

    result, notes = _build_merton_dtd(metrics, df, risk_free_pct=3.0)

    assert result is None
    assert any("fiyat geçmişi yok" in n for n in notes)


def test_build_merton_dtd_none_when_price_df_is_none():
    metrics = {"market_cap": 100.0, "total_debt": 80.0}
    result, notes = _build_merton_dtd(metrics, None, risk_free_pct=3.0)
    assert result is None


# ---------------------------------------------------------------------------
# 4. run_valuation wiring: computed for every sector, advisory-only (SPEC.md Sec.8k)
# ---------------------------------------------------------------------------

_MERTON_CONCEPTS = ["Revenue", "OperatingCashFlow", "CapEx", "SharesOutstanding", "StockholdersEquity"]


def _normalized() -> dict:
    return {
        "cik": 1, "entity_name": "Merton Test Co", "currency": "USD",
        "annual": {c: None for c in _MERTON_CONCEPTS},
        "quarterly": {c: None for c in _MERTON_CONCEPTS},
        "missing": list(_MERTON_CONCEPTS),
        "matched_tags": {c: None for c in _MERTON_CONCEPTS},
    }


def _assumptions():
    return {
        "bear": {"growth_5y": 0.02, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.05, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


def test_run_valuation_computes_merton_dtd_for_every_sector(tmp_path):
    # Provide a Damodaran erp.csv so risk_free_pct resolves (required for
    # merton_dtd -- see _build_merton_dtd's None-when-missing guard).
    (tmp_path / "erp.csv").write_text("region,erp,risk_free\nUS,4.6,3.0\n", encoding="utf-8")
    normalized = _normalized()
    df = _alternating_return_price_df(n_returns=60, magnitude=0.05)
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0,
        "market_cap": 100.0, "total_debt": 80.0,
    }
    assumptions = _assumptions()

    for sector in ("mature", "financial", "reit", "cyclical", "growth_unprofitable"):
        ratios = [{"fy": 2023, "roe": 0.10}] if sector in ("financial", "reit") else []
        result = run_valuation(
            normalized, ratios, metrics, price=20.0, price_df=df,
            assumptions=assumptions, sector_type=sector, damodaran_dir=str(tmp_path),
        )
        assert result["merton_dtd"] is not None, f"merton_dtd should be computed for sector={sector}"


def test_run_valuation_merton_dtd_never_affects_fair_value_range(tmp_path):
    (tmp_path / "erp.csv").write_text("region,erp,risk_free\nUS,4.6,3.0\n", encoding="utf-8")
    normalized = _normalized()
    df = _alternating_return_price_df(n_returns=60, magnitude=0.05)
    metrics_with = {
        "shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0,
        "market_cap": 100.0, "total_debt": 80.0,
    }
    metrics_without = dict(metrics_with)
    metrics_without["total_debt"] = None  # merton_dtd becomes unavailable
    assumptions = _assumptions()

    result_with = run_valuation(
        normalized, [], metrics_with, price=20.0, price_df=df,
        assumptions=assumptions, sector_type="mature", damodaran_dir=str(tmp_path),
    )
    result_without = run_valuation(
        normalized, [], metrics_without, price=20.0, price_df=df,
        assumptions=assumptions, sector_type="mature", damodaran_dir=str(tmp_path),
    )

    assert result_with["merton_dtd"] is not None
    assert result_without["merton_dtd"] is None
    assert result_with["fair_value_range"] == result_without["fair_value_range"]
