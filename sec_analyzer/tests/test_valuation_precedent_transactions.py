"""Tests for precedent-transaction (M&A comps) reference data (SPEC.md
Sec.8i): ``valuation.precedent_transactions.load_precedent_transactions``/
``.find_industry_medians`` and their ``run_valuation`` wiring (the
EV/EBITDA sector-median gap fix for leveraged filers).

Loader-tolerance tests mirror ``test_damodaran.py``'s
``test_load_sector_data_*`` conventions (``tmp_path``-based real CSV I/O,
not mocked).
"""

import pytest

from sec_analyzer.valuation.engine import run_valuation
from sec_analyzer.valuation.precedent_transactions import (
    find_industry_medians,
    load_precedent_transactions,
)

# ---------------------------------------------------------------------------
# 1. load_precedent_transactions (loader tolerance, SPEC.md Sec.8i)
# ---------------------------------------------------------------------------


def test_load_precedent_transactions_parses_deals_csv(tmp_path):
    (tmp_path / "deals.csv").write_text(
        "industry,ev_ebitda,ev_revenue,control_premium,deal_count\n"
        "Semiconductor,12.4,4.8,0.32,18\n"
        "Retail (General),9.1,0.9,0.28,7\n",
        encoding="utf-8",
    )

    result = load_precedent_transactions(str(tmp_path))

    assert result == [
        {"industry": "Semiconductor", "ev_ebitda": 12.4, "ev_revenue": 4.8, "control_premium": 0.32, "deal_count": 18.0},
        {"industry": "Retail (General)", "ev_ebitda": 9.1, "ev_revenue": 0.9, "control_premium": 0.28, "deal_count": 7.0},
    ]


def test_load_precedent_transactions_degrades_missing_columns_to_none(tmp_path):
    # Only industry + ev_ebitda -- other optional columns are simply absent
    # from the header; each degrades to None rather than dropping the row.
    (tmp_path / "deals.csv").write_text(
        "industry,ev_ebitda\nSemiconductor,12.4\n", encoding="utf-8",
    )

    result = load_precedent_transactions(str(tmp_path))

    assert result == [
        {"industry": "Semiconductor", "ev_ebitda": 12.4, "ev_revenue": None, "control_premium": None, "deal_count": None},
    ]


def test_load_precedent_transactions_skips_rows_without_industry(tmp_path):
    (tmp_path / "deals.csv").write_text(
        "industry,ev_ebitda\n,12.4\nSemiconductor,9.0\n", encoding="utf-8",
    )

    result = load_precedent_transactions(str(tmp_path))

    assert result == [
        {"industry": "Semiconductor", "ev_ebitda": 9.0, "ev_revenue": None, "control_premium": None, "deal_count": None},
    ]


def test_load_precedent_transactions_returns_none_for_missing_directory(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    assert load_precedent_transactions(str(missing_dir)) is None


def test_load_precedent_transactions_returns_none_for_missing_file(tmp_path):
    # Directory exists but deals.csv doesn't.
    assert load_precedent_transactions(str(tmp_path)) is None


def test_load_precedent_transactions_returns_none_for_none_dir_path():
    assert load_precedent_transactions(None) is None


def test_load_precedent_transactions_returns_none_when_no_row_has_industry(tmp_path):
    (tmp_path / "deals.csv").write_text("industry,ev_ebitda\n,12.4\n,9.0\n", encoding="utf-8")
    assert load_precedent_transactions(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# 2. find_industry_medians (normalized exact match, SPEC.md Sec.8i)
# ---------------------------------------------------------------------------


def test_find_industry_medians_matches_normalized_name():
    deals = [
        {"industry": "Semiconductor", "ev_ebitda": 12.4, "ev_revenue": 4.8, "control_premium": 0.32, "deal_count": 18.0},
    ]
    result = find_industry_medians(deals, "Semiconductor")
    assert result is not None
    assert result["ev_ebitda"] == pytest.approx(12.4)


def test_find_industry_medians_case_and_punctuation_insensitive():
    deals = [{"industry": "Retail (General)", "ev_ebitda": 9.1, "ev_revenue": None, "control_premium": None, "deal_count": None}]
    # damodaran._normalize_text lowercases and collapses punctuation to spaces.
    result = find_industry_medians(deals, "retail general")
    assert result is not None
    assert result["ev_ebitda"] == pytest.approx(9.1)


def test_find_industry_medians_none_when_no_match():
    deals = [{"industry": "Semiconductor", "ev_ebitda": 12.4, "ev_revenue": None, "control_premium": None, "deal_count": None}]
    assert find_industry_medians(deals, "Retail (General)") is None


def test_find_industry_medians_none_when_deals_or_industry_empty():
    assert find_industry_medians(None, "Semiconductor") is None
    assert find_industry_medians([], "Semiconductor") is None
    assert find_industry_medians([{"industry": "Semiconductor", "ev_ebitda": 12.4}], None) is None


# ---------------------------------------------------------------------------
# 3. run_valuation wiring: fills the EV/EBITDA sector-median gap for a
#    leveraged filer, purely additively (SPEC.md Sec.8i).
# ---------------------------------------------------------------------------

_PT_CONCEPTS = ["Revenue", "OperatingCashFlow", "CapEx", "SharesOutstanding", "StockholdersEquity"]


def _normalized() -> dict:
    return {
        "cik": 1, "entity_name": "Precedent Test Co", "currency": "USD",
        "annual": {c: None for c in _PT_CONCEPTS},
        "quarterly": {c: None for c in _PT_CONCEPTS},
        "missing": list(_PT_CONCEPTS),
        "matched_tags": {c: None for c in _PT_CONCEPTS},
    }


def _assumptions():
    return {
        "bear": {"growth_5y": 0.02, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.05, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


def _leveraged_metrics():
    # net_debt/ebitda = 300/100 = 3.0 >= triangulate._LEVERAGE_EBITDA_RATIO
    # (1.0) -> leveraged=True -> FD/FAVÖK (EV/EBITDA) becomes the primary
    # own-history multiple/sector-comparison axis.
    return {
        "shares": 50.0, "latest_fy": 2023, "fcf": 40.0, "net_debt": 300.0,
        "ebitda": 100.0, "ev_ebitda": 12.0, "market_cap": 500.0,
    }


def test_run_valuation_fills_ev_ebitda_sector_median_from_precedent_transactions(tmp_path):
    (tmp_path / "deals.csv").write_text(
        "industry,ev_ebitda,ev_revenue,control_premium,deal_count\n"
        "Semiconductor,10.0,4.0,0.30,12\n",
        encoding="utf-8",
    )
    # No Damodaran multiples.csv/erp.csv -> sector_medians_result would
    # normally be None (no industry to key precedent-transaction lookup
    # off), so also drop a minimal multiples.csv/erp.csv into the SAME
    # damodaran_dir with a matching industry row and sic_description.
    (tmp_path / "multiples.csv").write_text(
        "industry,pe,ps,pfcf\nSemiconductor,20.0,3.0,18.0\n", encoding="utf-8",
    )
    (tmp_path / "erp.csv").write_text("region,erp,risk_free\nUS,4.6,4.2\n", encoding="utf-8")

    result = run_valuation(
        _normalized(), [], _leveraged_metrics(), price=100.0, price_df=None,
        assumptions=_assumptions(), sector_type="mature",
        damodaran_dir=str(tmp_path), sic_description="Semiconductor",
        precedent_transactions_dir=str(tmp_path),
    )

    assert result["multiples"]["leveraged"] is True
    sector_info = result["multiples"]["sector"]
    assert sector_info["industry"] == "Semiconductor"
    # The core WP12 fix: the EV/EBITDA sector median -- previously ALWAYS
    # None (Damodaran's own multiples.csv has no such column) -- is now
    # sourced from the precedent-transaction data.
    assert sector_info["ev_ebitda_median"] == pytest.approx(10.0)
    assert sector_info["precedent_transactions"]["control_premium"] == pytest.approx(0.30)
    # NOTE: the axis-b "comparison" block additionally requires an
    # own-history percentile (>= 5 years of price-backed history via
    # price_df, a separate, pre-existing mechanism this minimal fixture
    # doesn't provide -- price_df=None here) before it populates, so it
    # stays disabled in THIS fixture even though ev_ebitda_median is now
    # available -- that wiring is exercised by test_valuation_multiples.py,
    # not re-tested here.
    assert sector_info["comparison"]["label"] is None


def test_run_valuation_degrades_silently_when_no_precedent_transactions_dir(tmp_path):
    (tmp_path / "multiples.csv").write_text(
        "industry,pe,ps,pfcf\nSemiconductor,20.0,3.0,18.0\n", encoding="utf-8",
    )
    (tmp_path / "erp.csv").write_text("region,erp,risk_free\nUS,4.6,4.2\n", encoding="utf-8")
    missing_precedent_dir = tmp_path / "no_precedent_data_here"

    result = run_valuation(
        _normalized(), [], _leveraged_metrics(), price=100.0, price_df=None,
        assumptions=_assumptions(), sector_type="mature",
        damodaran_dir=str(tmp_path), sic_description="Semiconductor",
        precedent_transactions_dir=str(missing_precedent_dir),
    )

    sector_info = result["multiples"]["sector"]
    assert sector_info["ev_ebitda_median"] is None
    assert sector_info["precedent_transactions"] is None
    # Exactly today's pre-existing (pre-Sec.8i) behavior: axis-b comparison
    # stays disabled for the leveraged primary when no precedent data exists.
    assert sector_info["comparison"]["label"] is None
    assert sector_info["comparison"]["ratio"] is None


def test_run_valuation_precedent_transactions_never_a_new_triangulation_vote(tmp_path):
    # Confirm the 3-way triangulation signal set is unchanged regardless of
    # whether precedent-transaction data is present (SPEC.md Sec.8i: "not a
    # new triangulation vote").
    (tmp_path / "multiples.csv").write_text(
        "industry,pe,ps,pfcf\nSemiconductor,20.0,3.0,18.0\n", encoding="utf-8",
    )
    (tmp_path / "erp.csv").write_text("region,erp,risk_free\nUS,4.6,4.2\n", encoding="utf-8")
    (tmp_path / "deals.csv").write_text(
        "industry,ev_ebitda\nSemiconductor,10.0\n", encoding="utf-8",
    )

    result = run_valuation(
        _normalized(), [], _leveraged_metrics(), price=100.0, price_df=None,
        assumptions=_assumptions(), sector_type="mature",
        damodaran_dir=str(tmp_path), sic_description="Semiconductor",
        precedent_transactions_dir=str(tmp_path),
    )

    assert set(result["triangulation"]["signals"]) == {"dcf", "reverse_dcf", "multiples"}
