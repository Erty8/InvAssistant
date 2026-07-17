"""Tests for WP7 -- Beats-floor: kapı → çift-satır rapor.

Covers ``valuation.engine``'s new ``_growth_vs_floor`` classifier and its
wiring into both growth-inclusive-anchor-vs-EPV-floor call sites in
``_run_valuation``:

- the cyclical path (``cyclical_fcfe_detail["growth_vs_floor"]``, plus the
  new both-values note appended when the FCFE anchor lands BELOW the
  zero-growth EPV floor -- mirrors the note the mature path already had);
- the mature path (``mature_revenue_detail["growth_vs_floor"]`` only -- its
  both-values note already existed pre-WP7 and is unchanged).

Fixtures below are copied verbatim (same construction, same hand-derived
numbers) from the existing beats-floor/below-floor regression tests in
``test_valuation_cyclical_fcfe.py``
(``test_run_valuation_cyclical_capex_suppressed_profitable_triggers_fcfe_headline``
/ ``test_run_valuation_cyclical_fcfe_below_epv_floor_keeps_epv_headline``) and
``test_valuation_mature_revenue.py``
(``test_run_valuation_mature_revenue_first_beats_epv_floor_becomes_headline``
/ ``test_run_valuation_mature_revenue_first_below_epv_floor_guardrail_keeps_epv_headline``)
-- see those files for the full hand-verified numeric derivations; this file
only asserts the NEW ``growth_vs_floor``/note behavior on top of those
already-proven preconditions.
"""

import pytest

from sec_analyzer.valuation.engine import run_valuation

# ---------------------------------------------------------------------------
# Cyclical fixtures (mirrors test_valuation_cyclical_fcfe.py's _normalized/
# _rec/_annual conventions).
# ---------------------------------------------------------------------------

_CYCLICAL_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity",
]


def _cyclical_rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _cyclical_normalized(overrides: "dict[str, dict[int, float]]") -> dict:
    annual = {
        concept: [_cyclical_rec(fy, value) for fy, value in (overrides.get(concept) or {}).items()] or None
        for concept in _CYCLICAL_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "WP7 Cyclical Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CYCLICAL_CONCEPTS},
        "missing": [c for c in _CYCLICAL_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CYCLICAL_CONCEPTS},
    }


def _cyclical_run_assumptions():
    return {
        "bear": {"growth_5y": 0.04, "terminal_growth": 0.01, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.08, "terminal_growth": 0.02, "discount_rate": 0.09, "story": "Baz."},
        "bull": {"growth_5y": 0.12, "terminal_growth": 0.03, "discount_rate": 0.08, "story": "Boğa."},
    }


_CYCLICAL_RATIOS_SUPPRESSED = [{"fy": 2023, "fcf": 5.0}]
_CYCLICAL_METRICS_SUPPRESSED = {
    "shares": 10.0, "latest_fy": 2023, "fcf": 5.0, "net_debt": 0.0,
}


# ---------------------------------------------------------------------------
# 1. Cyclical: FCFE lands BELOW the EPV floor -- growth_vs_floor="destroys",
#    headline stays EPV, both-values note present (WP7's new note).
# ---------------------------------------------------------------------------


def test_cyclical_fcfe_below_epv_floor_sets_destroys_and_appends_both_values_note():
    # Identical fixture to test_valuation_cyclical_fcfe.py's
    # test_run_valuation_cyclical_fcfe_below_epv_floor_keeps_epv_headline:
    # NI=100, Revenue=1000, OCF=90, CapEx=70, StockholdersEquity=2000 ->
    # roe=0.05 (below every scenario's discount_rate) -> EPV base=111.11,
    # FCFE base=81.35 < 111.11 -> cf_beats_floor=False.
    normalized = _cyclical_normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 70.0},
        "StockholdersEquity": {2023: 2000.0},
    })
    assumptions = _cyclical_run_assumptions()

    result = run_valuation(
        normalized, _CYCLICAL_RATIOS_SUPPRESSED, _CYCLICAL_METRICS_SUPPRESSED,
        price=None, price_df=None, assumptions=assumptions, sector_type="cyclical",
    )

    # Preconditions (headline stays EPV, per the pre-existing regression test).
    assert result["cyclical_fcfe_headline"] is False
    assert result["earnings_power_headline"] is True

    # (a) headline stays EPV (fair_value_range mirrors the EPV band).
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["lo"]
    )
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(
        result["earnings_power"]["scenarios"]["base"]["hi"]
    )

    # (b) cyclical_fcfe_detail is still present/reported.
    detail = result["cyclical_fcfe_detail"]
    assert detail is not None
    base = detail["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(81.35, abs=0.02)

    # (c) growth_vs_floor == "destroys" (roe=0.05 < every scenario's discount_rate).
    assert detail["growth_vs_floor"] == "destroys"

    # (d) the new both-values note is present, citing both figures.
    assert any(
        "sürdürülebilir-büyüme FCFE de hesaplandı" in n
        and "81.35" in n and "111.11" in n and "YARATMADIĞINI" in n
        for n in result["notes"]
    )


def test_cyclical_fcfe_beats_epv_floor_sets_adds_no_destroys_note():
    # Identical fixture to test_valuation_cyclical_fcfe.py's
    # test_run_valuation_cyclical_capex_suppressed_profitable_triggers_fcfe_headline:
    # StockholdersEquity=500 -> roe=0.20 (above discount_rate) -> FCFE base
    # (149.67) >= EPV base (111.11) -> cf_beats_floor=True -> FCFE headline.
    normalized = _cyclical_normalized({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 70.0},
        "StockholdersEquity": {2023: 500.0},
    })
    assumptions = _cyclical_run_assumptions()

    result = run_valuation(
        normalized, _CYCLICAL_RATIOS_SUPPRESSED, _CYCLICAL_METRICS_SUPPRESSED,
        price=None, price_df=None, assumptions=assumptions, sector_type="cyclical",
    )

    assert result["cyclical_fcfe_headline"] is True
    assert result["earnings_power_headline"] is False

    detail = result["cyclical_fcfe_detail"]
    assert detail is not None
    assert detail["growth_vs_floor"] == "adds"

    # No below-floor "destroys" note should be present in the beats-floor case.
    assert not any("YARATMADIĞINI" in n for n in result["notes"])


# ---------------------------------------------------------------------------
# 2. Mature: beats-floor -> "adds", headline revenue-first;
#    below-floor -> "destroys", headline EPV (note text unchanged, pre-existing).
# ---------------------------------------------------------------------------


_MATURE_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity", "OperatingIncome",
]


def _mature_rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _mature_normalized(overrides: "dict[str, dict[int, float]]") -> dict:
    annual = {
        concept: [_mature_rec(fy, value) for fy, value in (overrides.get(concept) or {}).items()] or None
        for concept in _MATURE_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "WP7 Mature Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _MATURE_CONCEPTS},
        "missing": [c for c in _MATURE_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _MATURE_CONCEPTS},
    }


def _mature_assumptions(discount_rate=0.15, terminal_growth=0.03, growth_5y=None):
    g5 = growth_5y if growth_5y is not None else terminal_growth + 0.02
    return {
        "bear": {"growth_5y": g5, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": "Ayı."},
        "base": {"growth_5y": g5, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": "Baz."},
        "bull": {"growth_5y": g5, "terminal_growth": terminal_growth, "discount_rate": discount_rate, "story": "Boğa."},
    }


def test_mature_revenue_first_beats_epv_floor_sets_adds():
    # Identical fixture to test_valuation_mature_revenue.py's
    # test_run_valuation_mature_revenue_first_beats_epv_floor_becomes_headline:
    # NI=100, Revenue=1000, OCF=150, CapEx=75 -> EPV base=6.67, revenue-first
    # base=12.42 >= 6.67 -> mr_beats_floor=True -> revenue-first headline.
    normalized = _mature_normalized({
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

    assert result["mature_revenue_headline"] is True
    assert result["earnings_power_headline"] is False

    detail = result["mature_revenue_detail"]
    assert detail is not None
    assert detail["growth_vs_floor"] == "adds"


def test_mature_revenue_first_below_epv_floor_sets_destroys():
    # Identical fixture to test_valuation_mature_revenue.py's
    # test_run_valuation_mature_revenue_first_below_epv_floor_guardrail_keeps_epv_headline:
    # CapEx=88 (OCF=90) -> revenue-first base=0.33 < EPV base(6.67) ->
    # mr_beats_floor=False -> headline stays EPV.
    normalized = _mature_normalized({
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

    assert result["mature_revenue_headline"] is False
    assert result["earnings_power_headline"] is True

    detail = result["mature_revenue_detail"]
    assert detail is not None
    assert detail["growth_vs_floor"] == "destroys"

    # Pre-existing both-values note (unchanged text, only asserting it is
    # still there and still carries both figures).
    assert any(
        "altında kaldığı için manşet EPV'de tutuldu" in n and "0.33" in n and "6.67" in n
        for n in result["notes"]
    )
