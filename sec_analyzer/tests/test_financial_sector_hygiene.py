"""Unit tests for deposit-funded detection + financial-sector metric hygiene.

SPEC.md Sec.20: (a) a deposit-funded balance sheet classifies as
``financial`` whatever the SIC says, (b) enterprise-value multiples are
suppressed for that sector, (c) the liquidity/leverage checklist items are
re-based. No network access, no randomness.
"""

import pytest

from sec_analyzer.interpret import rule_based
from sec_analyzer.valuation.engine import run_valuation
from sec_analyzer.valuation.sector import (
    SECTOR_CYCLICAL,
    SECTOR_FINANCIAL,
    SECTOR_MATURE,
    SECTOR_REIT,
    _is_deposit_funded,
    classify_sector,
)


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(**series):
    """A normalized dict carrying only the named concepts."""
    annual = {name: [_rec(fy, v) for fy, v in sorted(values.items(), reverse=True)]
              for name, values in series.items()}
    return {
        "cik": 1, "entity_name": "Hygiene Test Co", "currency": "USD",
        "annual": annual, "quarterly": {}, "missing": [], "matched_tags": {},
    }


# ---------------------------------------------------------------------------
# 20a. Deposit-funded detection
# ---------------------------------------------------------------------------


def test_deposit_funded_detects_a_bank_shaped_balance_sheet():
    # SoFi's real FY2025 figures: deposits 37.51B of 42.89B liabilities.
    normalized = _normalized(
        Deposits={2025: 37.51e9},
        TotalLiabilities={2025: 42.89e9},
    )

    assert _is_deposit_funded(normalized, {"latest_fy": 2025}) is True


def test_deposit_funded_is_false_for_an_ordinary_operating_company():
    normalized = _normalized(
        Deposits={2025: 1.0e9},
        TotalLiabilities={2025: 50.0e9},
    )

    assert _is_deposit_funded(normalized, {}) is False


def test_deposit_share_of_exactly_twenty_percent_does_not_trigger():
    normalized = _normalized(
        Deposits={2025: 200.0},
        TotalLiabilities={2025: 1_000.0},
    )

    assert _is_deposit_funded(normalized, {}) is False


def test_deposit_share_just_above_twenty_percent_triggers():
    normalized = _normalized(
        Deposits={2025: 201.0},
        TotalLiabilities={2025: 1_000.0},
    )

    assert _is_deposit_funded(normalized, {}) is True


def test_deposit_ratio_uses_a_single_fiscal_year_never_mixes_years():
    # Deposits only in 2025, liabilities only in 2024: no shared year, so the
    # ratio is not computable and must NOT be built from mismatched years
    # (37.51/1.0 would otherwise look like a bank).
    normalized = _normalized(
        Deposits={2025: 37.51e9},
        TotalLiabilities={2024: 1.0e9},
    )

    assert _is_deposit_funded(normalized, {}) is False


def test_deposit_ratio_prefers_the_newest_year_with_both_figures():
    # 2025 has deposits only; the decision falls back to 2024, where the
    # share is immaterial.
    normalized = _normalized(
        Deposits={2024: 100.0, 2025: 900.0},
        TotalLiabilities={2024: 1_000.0},
    )

    assert _is_deposit_funded(normalized, {}) is False


def test_deposit_funded_skips_a_non_positive_liabilities_year():
    normalized = _normalized(
        Deposits={2024: 500.0, 2025: 500.0},
        TotalLiabilities={2024: 1_000.0, 2025: 0.0},
    )

    assert _is_deposit_funded(normalized, {}) is True


@pytest.mark.parametrize("normalized", [None, {}, {"annual": None}, {"annual": "junk"}])
def test_deposit_funded_never_raises(normalized):
    assert _is_deposit_funded(normalized, {}) is False


def test_classify_sector_routes_a_deposit_funded_non_financial_sic_to_financial():
    # SIC 7372 (prepackaged software) would normally fall through to the
    # profitability check; deposit funding overrides it.
    normalized = _normalized(
        Deposits={2025: 37.51e9},
        TotalLiabilities={2025: 42.89e9},
        NetIncome={2025: 500.0e6},
    )

    assert classify_sector(7372, normalized, {"latest_fy": 2025}) == SECTOR_FINANCIAL


def test_deposit_override_never_displaces_a_reit_classification():
    normalized = _normalized(
        Deposits={2025: 37.51e9},
        TotalLiabilities={2025: 42.89e9},
    )

    assert classify_sector(6798, normalized, {"latest_fy": 2025}) == SECTOR_REIT


def test_deposit_override_beats_a_cyclical_sic():
    normalized = _normalized(
        Deposits={2025: 37.51e9},
        TotalLiabilities={2025: 42.89e9},
    )

    assert classify_sector(2911, normalized, {"latest_fy": 2025}) == SECTOR_FINANCIAL


def test_classification_without_deposits_is_unchanged():
    normalized = _normalized(NetIncome={2025: 500.0e6})

    assert classify_sector(7372, normalized, {"latest_fy": 2025}) == SECTOR_MATURE
    assert classify_sector(2911, normalized, {"latest_fy": 2025}) == SECTOR_CYCLICAL
    assert classify_sector(6199, normalized, {"latest_fy": 2025}) == SECTOR_FINANCIAL


# ---------------------------------------------------------------------------
# 20b. EV multiples are suppressed for a financial filer
# ---------------------------------------------------------------------------


def _ev_metrics():
    return {
        "shares": 100.0, "latest_fy": 2023, "fcf": 50.0,
        "ev": 5_000.0, "ev_ebit": 10.0, "ev_ebitda": 8.0,
        "net_debt": 2_000.0, "ebitda": 500.0,
    }


def _ev_assumptions():
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


def test_financial_filer_reports_no_ev_multiples():
    normalized = _normalized(
        StockholdersEquity={2023: 1_000.0},
        NetIncome={2023: 150.0},
        Deposits={2023: 800.0},
        TotalLiabilities={2023: 1_000.0},
    )

    result = run_valuation(
        normalized, [{"fy": 2023, "roe": 0.15}], _ev_metrics(),
        price=15.0, price_df=None, assumptions=_ev_assumptions(),
        sector_type="financial",
    )

    multiples = result["multiples"]
    assert multiples["ev_applicable"] is False
    assert multiples["current"]["ev_ebit"] is None
    assert multiples["current"]["ev_ebitda"] is None
    assert multiples["ev_ebit_percentile"] is None
    assert multiples["ev_ebitda_percentile"] is None
    # The leverage gate exists only to promote EV/EBITDA to primary.
    assert multiples["net_debt_to_ebitda"] is None
    assert multiples["leveraged"] is False


def test_financial_filer_gets_a_turkish_note_explaining_the_suppression():
    normalized = _normalized(
        StockholdersEquity={2023: 1_000.0},
        NetIncome={2023: 150.0},
    )

    result = run_valuation(
        normalized, [{"fy": 2023, "roe": 0.15}], _ev_metrics(),
        price=15.0, price_df=None, assumptions=_ev_assumptions(),
        sector_type="financial",
    )

    assert any("FD/FAVÖK" in note and "tanımsız" in note for note in result["notes"])


def test_financial_filer_emits_no_derived_ev_multiple_note():
    """The fy-mismatch back-fill must not manufacture an EV multiple (and its
    "derived from FYxxxx" note) for a sector where EV is undefined."""
    normalized = _normalized(
        StockholdersEquity={2023: 1_000.0},
        NetIncome={2023: 150.0},
        OperatingIncome={2023: 400.0},
        Depreciation={2023: 100.0},
    )
    metrics = dict(_ev_metrics(), ev_ebit=None, ev_ebitda=None)

    result = run_valuation(
        normalized, [{"fy": 2023, "roe": 0.15}], metrics,
        price=15.0, price_df=None, assumptions=_ev_assumptions(),
        sector_type="financial",
    )

    assert result["multiples"]["current"]["ev_ebitda"] is None
    assert not any("FD/FAVÖK" in note and "mali yılının" in note for note in result["notes"])


def test_non_financial_filer_keeps_its_ev_multiples():
    normalized = _normalized(
        StockholdersEquity={2023: 1_000.0},
        NetIncome={2023: 150.0},
        Revenue={2023: 2_000.0},
        OperatingCashFlow={2023: 300.0},
        CapEx={2023: 100.0},
    )

    result = run_valuation(
        normalized, [{"fy": 2023, "roe": 0.15}], _ev_metrics(),
        price=15.0, price_df=None, assumptions=_ev_assumptions(),
        sector_type="mature",
    )

    multiples = result["multiples"]
    assert multiples["ev_applicable"] is True
    assert multiples["current"]["ev_ebitda"] == pytest.approx(8.0)
    assert multiples["current"]["ev_ebit"] == pytest.approx(10.0)
    assert multiples["net_debt_to_ebitda"] == pytest.approx(4.0)
    assert not any("FD/FAVÖK" in note and "tanımsız" in note for note in result["notes"])


def test_reit_keeps_its_ev_multiples():
    """REITs are deliberately excluded from the suppression -- EV multiples are
    standard practice for them."""
    normalized = _normalized(
        StockholdersEquity={2023: 1_000.0},
        NetIncome={2023: 150.0},
        Depreciation={2023: 200.0},
    )

    result = run_valuation(
        normalized, [{"fy": 2023, "roe": 0.15}], _ev_metrics(),
        price=15.0, price_df=None, assumptions=_ev_assumptions(),
        sector_type="reit",
    )

    assert result["multiples"]["ev_applicable"] is True
    assert result["multiples"]["current"]["ev_ebitda"] == pytest.approx(8.0)


def test_suppression_does_not_mutate_the_caller_metrics_dict():
    metrics = _ev_metrics()
    normalized = _normalized(
        StockholdersEquity={2023: 1_000.0}, NetIncome={2023: 150.0},
    )

    run_valuation(
        normalized, [{"fy": 2023, "roe": 0.15}], metrics,
        price=15.0, price_df=None, assumptions=_ev_assumptions(),
        sector_type="financial",
    )

    assert metrics["ev_ebitda"] == pytest.approx(8.0)
    assert metrics["ev"] == pytest.approx(5_000.0)
    assert metrics["net_debt"] == pytest.approx(2_000.0)


# ---------------------------------------------------------------------------
# 20c. Liquidity / leverage checks are re-based
# ---------------------------------------------------------------------------


def _bank_like_inputs():
    """A profitable, deposit-funded filer: 3.97x liabilities/equity, and a
    current ratio that would fail the generic 1.0 bar."""
    normalized = _normalized(
        Revenue={2024: 2_675.0, 2025: 3_613.0},
        NetIncome={2024: 499.0, 2025: 481.0},
        OperatingCashFlow={2025: 600.0},
        StockholdersEquity={2025: 10_810.0},
        SharesOutstanding={2025: 1_200.0},
    )
    ratios = [{
        "fy": 2025, "net_margin": 0.133, "roe": 0.045,
        "current_ratio": 0.30, "debt_to_equity": 3.97,
        "yoy_revenue_growth": 0.35, "fcf": 600.0,
    }]
    return normalized, ratios


def _check(result, name):
    return next(c for c in result["score"]["checks"] if c["name"] == name)


def test_financial_filer_liquidity_check_is_not_applicable():
    normalized, ratios = _bank_like_inputs()

    result = rule_based.analyze(normalized, ratios, sector_type="financial")

    liquidity = _check(result, "Liquidity")
    assert liquidity["passed"] is None
    assert "not defined for a financial institution" in liquidity["detail"]
    assert "Liquidity" not in result["key_risks"]


def test_financial_filer_leverage_uses_the_deposit_inclusive_threshold():
    normalized, ratios = _bank_like_inputs()

    result = rule_based.analyze(normalized, ratios, sector_type="financial")

    leverage = _check(result, "Leverage")
    assert leverage["passed"] is True
    assert "Liabilities-to-equity incl. deposits" in leverage["detail"]
    assert "Leverage" not in result["key_risks"]


def test_a_genuinely_overlevered_bank_still_fails_the_leverage_check():
    normalized, ratios = _bank_like_inputs()
    ratios[0]["debt_to_equity"] = 12.5

    result = rule_based.analyze(normalized, ratios, sector_type="financial")

    assert _check(result, "Leverage")["passed"] is False


def test_financial_filer_with_negative_equity_still_auto_fails_leverage():
    normalized, ratios = _bank_like_inputs()
    normalized["annual"]["StockholdersEquity"] = [_rec(2025, -100.0)]

    result = rule_based.analyze(normalized, ratios, sector_type="financial")

    assert _check(result, "Leverage")["passed"] is False


def test_same_inputs_without_a_sector_keep_the_generic_thresholds():
    normalized, ratios = _bank_like_inputs()

    result = rule_based.analyze(normalized, ratios)

    assert _check(result, "Liquidity")["passed"] is False
    assert _check(result, "Leverage")["passed"] is False


def test_non_financial_sector_keeps_the_generic_thresholds():
    normalized, ratios = _bank_like_inputs()

    result = rule_based.analyze(normalized, ratios, sector_type="mature")

    assert _check(result, "Liquidity")["passed"] is False
    assert _check(result, "Leverage")["passed"] is False


# ---------------------------------------------------------------------------
# 20c. CLI ratio table
# ---------------------------------------------------------------------------


def test_cli_ratio_table_drops_the_current_ratio_column_for_a_financial(capsys):
    from sec_analyzer.cli import _print_ratios

    ratios = [{"fy": 2025, "net_margin": 0.13, "roe": 0.045,
               "current_ratio": 0.30, "yoy_revenue_growth": 0.35,
               "yoy_net_income_growth": -0.04}]

    _print_ratios(ratios, sector_type="financial")
    financial_out = capsys.readouterr().out
    assert "Current Ratio" not in financial_out
    assert "0.30" not in financial_out
    assert "Net Margin" in financial_out and "ROE" in financial_out

    _print_ratios(ratios)
    default_out = capsys.readouterr().out
    assert "Current Ratio" in default_out
    assert "0.30" in default_out
