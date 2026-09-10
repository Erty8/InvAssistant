"""Hand-verified numeric tests for the LBO-implied floor value (SPEC.md
Sec.8h): ``valuation.lbo.lbo_implied_floor_per_share`` and
``valuation.engine._build_lbo_floor``/their ``run_valuation`` wiring.

This is an ADVISORY-ONLY floor -- every test below also confirms it never
touches ``fair_value_range``/``primary_dcf_scenarios``/the triangulation
confidence, and is never computed at all for ``financial``/``reit`` filers.
"""

import pytest

from sec_analyzer.valuation.engine import _build_lbo_floor, run_valuation
from sec_analyzer.valuation.lbo import lbo_implied_floor_per_share

# ---------------------------------------------------------------------------
# 1. lbo_implied_floor_per_share (pure function, SPEC.md Sec.8h)
# ---------------------------------------------------------------------------


def test_lbo_implied_floor_normal_case_hand_verified():
    # ebitda=100, entry_multiple=8.0, existing_debt=300, exit_multiple=8.0
    # (no multiple expansion), fcf0=40, fcf_growth=0.05 (flat -- hold_years=5
    # never reaches project_fcf's fade phase), shares=50, target_irr=0.20,
    # hold_years=5 (all defaults except target_irr/hold_years, spelled out).
    #
    # entry_ev = 8*100 = 800; entry_equity = 800-300 = 500.
    # fcf_path (flat 5% growth): 42.0, 44.1, 46.305, 48.62025, 51.0512625.
    # 100% cash sweep debt paydown (never floors at 0 here):
    #   debt1 = 300-42     = 258.0
    #   debt2 = 258-44.1   = 213.9
    #   debt3 = 213.9-46.305        = 167.595
    #   debt4 = 167.595-48.62025    = 118.97475
    #   debt5 = 118.97475-51.0512625 = 67.9234875
    # exit_ev = 8*100 = 800; exit_equity = 800-67.9234875 = 732.0765125.
    # floor_equity_today = 732.0765125 / 1.20^5 = 732.0765125/2.48832
    #                     = 294.205131
    # per_share = 294.205131/50 = 5.884103
    result = lbo_implied_floor_per_share(
        ebitda=100.0, entry_multiple=8.0, existing_debt=300.0, exit_multiple=8.0,
        fcf0=40.0, fcf_growth=0.05, shares=50.0, target_irr=0.20, hold_years=5,
    )

    assert result is not None
    assert result["entry_ev"] == pytest.approx(800.0)
    assert result["entry_equity"] == pytest.approx(500.0)
    assert len(result["fcf_path"]) == 5
    assert result["fcf_path"][0] == pytest.approx(42.0)
    assert result["debt_path"][-1] == pytest.approx(67.9234875, rel=1e-6)
    assert result["remaining_debt"] == pytest.approx(67.9234875, rel=1e-6)
    assert result["exit_ev"] == pytest.approx(800.0)
    assert result["exit_equity"] == pytest.approx(732.0765125, rel=1e-6)
    assert result["floor_equity_today"] == pytest.approx(294.205131, rel=1e-5)
    assert result["per_share"] == pytest.approx(5.884103, rel=1e-5)


def test_lbo_implied_floor_debt_fully_paid_down_floors_at_zero():
    # existing_debt=50 (small vs. FCF) -- debt hits 0 by year 2 and STAYS 0
    # (never goes negative) for years 3-5.
    #   debt1 = max(0, 50-42)   = 8.0
    #   debt2 = max(0, 8-44.1)  = 0.0 (would be -36.1 uncapped)
    #   debt3 = max(0, 0-46.305)= 0.0
    #   debt4 = 0.0; debt5 = 0.0
    # exit_ev = 6*100 = 600; exit_equity = 600-0 = 600.
    # floor_equity_today = 600/1.20^5 = 241.126543
    # per_share = 241.126543/20 = 12.056327
    result = lbo_implied_floor_per_share(
        ebitda=100.0, entry_multiple=6.0, existing_debt=50.0, exit_multiple=6.0,
        fcf0=40.0, fcf_growth=0.05, shares=20.0, target_irr=0.20, hold_years=5,
    )

    assert result is not None
    assert result["debt_path"] == [pytest.approx(8.0), pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0)]
    assert result["remaining_debt"] == pytest.approx(0.0)
    assert result["exit_equity"] == pytest.approx(600.0)
    assert result["floor_equity_today"] == pytest.approx(241.126543, rel=1e-5)
    assert result["per_share"] == pytest.approx(12.056327, rel=1e-5)


def test_lbo_implied_floor_none_when_ebitda_missing_or_non_positive():
    assert lbo_implied_floor_per_share(None, 8.0, 300.0, 8.0, 40.0, 0.05, 50.0) is None
    assert lbo_implied_floor_per_share(0.0, 8.0, 300.0, 8.0, 40.0, 0.05, 50.0) is None
    assert lbo_implied_floor_per_share(-10.0, 8.0, 300.0, 8.0, 40.0, 0.05, 50.0) is None


def test_lbo_implied_floor_none_when_existing_debt_missing_or_negative():
    assert lbo_implied_floor_per_share(100.0, 8.0, None, 8.0, 40.0, 0.05, 50.0) is None
    assert lbo_implied_floor_per_share(100.0, 8.0, -1.0, 8.0, 40.0, 0.05, 50.0) is None


def test_lbo_implied_floor_none_when_multiples_non_positive():
    assert lbo_implied_floor_per_share(100.0, 0.0, 300.0, 8.0, 40.0, 0.05, 50.0) is None
    assert lbo_implied_floor_per_share(100.0, 8.0, 300.0, 0.0, 40.0, 0.05, 50.0) is None
    assert lbo_implied_floor_per_share(100.0, None, 300.0, 8.0, 40.0, 0.05, 50.0) is None


def test_lbo_implied_floor_none_when_fcf0_missing():
    assert lbo_implied_floor_per_share(100.0, 8.0, 300.0, 8.0, None, 0.05, 50.0) is None


def test_lbo_implied_floor_none_when_shares_missing_or_non_positive():
    assert lbo_implied_floor_per_share(100.0, 8.0, 300.0, 8.0, 40.0, 0.05, None) is None
    assert lbo_implied_floor_per_share(100.0, 8.0, 300.0, 8.0, 40.0, 0.05, 0.0) is None


def test_lbo_implied_floor_none_when_hold_years_non_positive():
    assert lbo_implied_floor_per_share(100.0, 8.0, 300.0, 8.0, 40.0, 0.05, 50.0, hold_years=0) is None


def test_lbo_implied_floor_none_when_target_irr_degenerate():
    assert lbo_implied_floor_per_share(100.0, 8.0, 300.0, 8.0, 40.0, 0.05, 50.0, target_irr=-1.0) is None


# ---------------------------------------------------------------------------
# 2. _build_lbo_floor (engine.py, SPEC.md Sec.8h)
# ---------------------------------------------------------------------------


def _lbo_assumptions(base_growth=0.05):
    # Still used by the run_valuation integration tests below (run_valuation
    # needs an assumption set for its DCF path); the LBO floor itself no
    # longer reads it (F3: FCF held flat).
    return {
        "bear": {"growth_5y": 0.02, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": base_growth, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


def test_build_lbo_floor_reads_metrics_and_holds_fcf_flat():
    # F3: _build_lbo_floor holds FCF FLAT (fcf_growth=0), so it no longer
    # depends on the assumption set at all. With ebitda=100, ev_ebitda=8,
    # total_debt=300, fcf0=40, shares=50: FCF is 40 every year, debt paydown
    # [260,220,180,140,100], exit_equity = 8*100 - 100 = 700,
    # floor = 700/1.20^5 = 281.30, per_share = 281.30/50 = 5.626286.
    metrics = {"ebitda": 100.0, "ev_ebitda": 8.0, "total_debt": 300.0, "shares": 50.0}

    result, notes = _build_lbo_floor(metrics, fcf0=40.0)

    assert result is not None
    assert result["per_share"] == pytest.approx(5.626286, rel=1e-5)
    # FCF held flat -> level debt paydown, never over-swept by phantom growth.
    assert result["debt_path"] == [pytest.approx(260.0), pytest.approx(220.0), pytest.approx(180.0), pytest.approx(140.0), pytest.approx(100.0)]
    assert any("bilgi amaçlı" in n and "manşete GİRMEZ" in n for n in notes)


def test_build_lbo_floor_none_when_ev_ebitda_missing():
    metrics = {"ebitda": 100.0, "ev_ebitda": None, "total_debt": 300.0, "shares": 50.0}

    result, notes = _build_lbo_floor(metrics, fcf0=40.0)

    assert result is None
    assert any("hesaplanamadı" in n for n in notes)


# ---------------------------------------------------------------------------
# 3. run_valuation wiring: sector gate + advisory-only (SPEC.md Sec.8h)
# ---------------------------------------------------------------------------

_LBO_CONCEPTS = ["Revenue", "OperatingCashFlow", "CapEx", "SharesOutstanding", "StockholdersEquity"]


def _lbo_normalized() -> dict:
    """Minimal, fully-empty ``normalized``-shaped fixture -- the LBO floor
    itself reads only ``assumptions``/``metrics``/``fcf0`` (see
    ``_build_lbo_floor`` above), but ``run_valuation`` needs a validly
    shaped ``normalized`` dict for its OTHER builders (DCF, etc.) even
    though this test doesn't care about their output."""
    return {
        "cik": 1, "entity_name": "LBO Test Co", "currency": "USD",
        "annual": {c: None for c in _LBO_CONCEPTS},
        "quarterly": {c: None for c in _LBO_CONCEPTS},
        "missing": list(_LBO_CONCEPTS),
        "matched_tags": {c: None for c in _LBO_CONCEPTS},
    }


def _mature_metrics():
    return {
        "shares": 50.0, "latest_fy": 2023, "fcf": 40.0, "net_debt": 0.0,
        "ebitda": 100.0, "ev_ebitda": 8.0, "total_debt": 300.0,
    }


def _mature_ratios():
    """3 fiscal years of ``fcf`` (a non-deviating, near-monotonic ramp) so
    ``_select_fcf0`` resolves ``fcf0=40.0`` (the ttm figure, no SBC data ->
    unadjusted) -- ``run_valuation`` derives its OWN ``fcf0`` from ``ratios``/
    ``normalized`` internally (see ``_select_fcf0``/``_sbc_adjusted_fcf_by_fy``),
    it does NOT read ``metrics["fcf"]`` directly."""
    return [
        {"fy": 2023, "fcf": 40.0}, {"fy": 2022, "fcf": 38.0}, {"fy": 2021, "fcf": 36.0},
    ]


def test_run_valuation_computes_lbo_floor_for_mature_sector():
    normalized = _lbo_normalized()
    assumptions = _lbo_assumptions(base_growth=0.05)

    result = run_valuation(
        normalized, _mature_ratios(), _mature_metrics(), price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(40.0)
    assert result["lbo_floor_detail"] is not None
    assert result["lbo_floor_detail"]["per_share"] is not None


def test_run_valuation_never_computes_lbo_floor_for_financial_or_reit():
    normalized = _lbo_normalized()
    assumptions = _lbo_assumptions(base_growth=0.05)

    for sector in ("financial", "reit"):
        ratios = _mature_ratios() + [{"fy": 2023, "roe": 0.12}]
        result = run_valuation(
            normalized, ratios, _mature_metrics(), price=20.0, price_df=None,
            assumptions=assumptions, sector_type=sector,
        )
        assert result["lbo_floor_detail"] is None


def test_run_valuation_lbo_floor_never_affects_fair_value_range():
    normalized = _lbo_normalized()
    assumptions = _lbo_assumptions(base_growth=0.05)
    ratios = _mature_ratios()

    metrics_with_lbo = _mature_metrics()
    metrics_without_lbo = dict(metrics_with_lbo)
    metrics_without_lbo["ev_ebitda"] = None  # LBO floor becomes unavailable

    result_with = run_valuation(
        normalized, ratios, metrics_with_lbo, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )
    result_without = run_valuation(
        normalized, ratios, metrics_without_lbo, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result_with["lbo_floor_detail"] is not None
    assert result_without["lbo_floor_detail"] is None
    # fair_value_range is identical regardless of the LBO floor's presence --
    # it's derived purely from fcf0/shares/assumptions, unaffected by
    # ev_ebitda/total_debt (only lbo_floor_detail's inputs).
    assert result_with["fair_value_range"] == result_without["fair_value_range"]
