"""Immaterial CapEx imputation (SPEC.md Sec.31, WP15).

AppLovin supplies the hand-verified numbers: CapEx was tagged through FY2024
(`$4.8M` against `$2.10B` OCF, 0.23%) and then dropped from the filings, so
FY2025's FCF read as undefined and fcf0 fell back to an older average.
"""

import pytest

from sec_analyzer.normalize.metrics import _compute_ttm_cash
from sec_analyzer.normalize.normalizer import _impute_immaterial_capex, to_annual_series, to_quarterly_series
from sec_analyzer.valuation.engine import _capex_imputation_notes

M = 1e6


def _rec(period_end, value, start=None, fy=None, fp="FY", form="10-K"):
    return {
        "concept": None, "tag": "T", "period_end": period_end,
        "fy": fy if fy is not None else int(period_end[:4]),
        "fp": fp, "form": form, "value": value,
        "filed": None, "start": start, "unit": "USD",
    }


def _app(capex_2024=4.8 * M):
    annual = {
        "Revenue": [_rec("2025-12-31", 5480 * M), _rec("2024-12-31", 3220 * M)],
        "OperatingCashFlow": [_rec("2025-12-31", 3970 * M), _rec("2024-12-31", 2100 * M),
                              _rec("2023-12-31", 1060 * M)],
        "CapEx": [_rec("2024-12-31", capex_2024), _rec("2023-12-31", 4.2 * M)],
    }
    # 10-Q OCF rows are year-to-date, as AppLovin reports them.
    quarterly = {
        "OperatingCashFlow": [
            _rec("2025-03-31", 830 * M, "2025-01-01", fp="Q1", form="10-Q"),
            _rec("2025-06-30", 1600 * M, "2025-01-01", fp="Q2", form="10-Q"),
            _rec("2025-09-30", 2650 * M, "2025-01-01", fp="Q3", form="10-Q"),
            _rec("2026-03-31", 1290 * M, "2026-01-01", fp="Q1", form="10-Q"),
            _rec("2026-06-30", 2160 * M, "2026-01-01", fp="Q2", form="10-Q"),
        ],
        "CapEx": None,
    }
    return annual, quarterly


def test_imputes_annual_and_quarterly_capex_after_the_last_tagged_year():
    annual, quarterly = _app()
    block = _impute_immaterial_capex(annual, quarterly)

    ratio = 4.8 / 2100
    assert block["reference_fy"] == 2024
    assert block["ratio"] == pytest.approx(ratio)
    assert block["imputed_annual_fys"] == [2025]
    assert block["imputed_quarter_ends"] == ["2025-03-31", "2025-06-30", "2025-09-30",
                                             "2026-03-31", "2026-06-30"]
    capex = to_annual_series({"annual": annual}, "CapEx")
    assert capex[2025] == pytest.approx(ratio * 3970 * M)
    assert capex[2024] == pytest.approx(4.8 * M)  # reported years untouched
    assert all(r.get("imputed") for r in quarterly["CapEx"])


def test_imputed_ytd_rows_difference_into_single_quarters_and_complete_ttm():
    annual, quarterly = _app()
    _impute_immaterial_capex(annual, quarterly)
    normalized = {"annual": annual, "quarterly": quarterly}

    ratio = 4.8 / 2100
    q = {r["period_end"]: r["value"] for r in to_quarterly_series(normalized, "CapEx")}
    # Q4 2025 derived from the imputed annual minus the imputed 9M row.
    assert q["2025-12-31"] == pytest.approx(ratio * (3970 - 2650) * M)
    assert q["2026-06-30"] == pytest.approx(ratio * (2160 - 1290) * M)

    ttm = _compute_ttm_cash(normalized, latest_fundamental_fy=2025)
    assert ttm["ttm_cash_complete"] is True
    assert ttm["ttm_capex"] == pytest.approx(ratio * ttm["ttm_ocf"])


def test_material_capex_gap_is_left_missing():
    # 10% of OCF is material: a gap must stay undefined, never guessed.
    annual, quarterly = _app(capex_2024=210 * M)
    block = _impute_immaterial_capex(annual, quarterly)

    assert block is None
    assert 2025 not in to_annual_series({"annual": annual}, "CapEx")
    assert quarterly["CapEx"] is None


def test_capex_above_two_percent_of_revenue_is_material():
    # 4% of OCF passes the OCF gate, but 2.6% of revenue does not.
    annual, quarterly = _app(capex_2024=84 * M)
    annual["Revenue"] = [_rec("2024-12-31", 3220 * M)]

    assert _impute_immaterial_capex(annual, quarterly) is None


def test_stale_reference_is_not_extrapolated():
    # Reference FY2021: FY2024 is ~3 years out (1096 days, imputed), FY2025
    # is ~4 years out (1461 days > 1200, left undefined).
    annual, quarterly = _app()
    annual["CapEx"] = [_rec("2021-12-31", 4.0 * M)]
    annual["OperatingCashFlow"] += [_rec("2022-12-31", 410 * M), _rec("2021-12-31", 360 * M)]
    quarterly["OperatingCashFlow"] = []

    block = _impute_immaterial_capex(annual, quarterly)

    assert block["imputed_annual_fys"] == [2022, 2023, 2024]
    assert 2025 not in to_annual_series({"annual": annual}, "CapEx")


def test_one_light_year_cannot_stand_in_for_heavy_capex_history():
    # FY2024 is immaterial, but FY2023 consumed 20% of OCF: not stable.
    annual, quarterly = _app()
    annual["CapEx"] = [_rec("2024-12-31", 4.8 * M), _rec("2023-12-31", 212 * M)]

    assert _impute_immaterial_capex(annual, quarterly) is None


def test_fully_tagged_filer_is_unchanged():
    annual, quarterly = _app()
    annual["CapEx"].insert(0, _rec("2025-12-31", 9.0 * M))
    quarterly["OperatingCashFlow"] = []

    assert _impute_immaterial_capex(annual, quarterly) is None


def test_engine_note_only_when_imputation_reaches_fcf0():
    normalized = {"capex_imputation": {"reference_fy": 2024, "ratio": 0.00228,
                                       "imputed_annual_fys": [2025],
                                       "imputed_quarter_ends": ["2026-06-30"]}}

    note = _capex_imputation_notes(normalized, {"latest_fy": 2025}, "latest_fy")
    assert len(note) == 1
    assert "after FY2024" in note[0] and "0.23%" in note[0] and "FY2025, 2026-06-30" in note[0]
    assert _capex_imputation_notes(normalized, {"latest_fy": 2026}, "latest_fy") == []
    assert _capex_imputation_notes({"capex_imputation": None}, {"latest_fy": 2025}, "ttm") == []
