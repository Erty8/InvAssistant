"""Hand-verified numeric tests for ``valuation.multiples`` and
``valuation.sensitivity`` (SPEC.md Sec.6, Sec.9).

Every numeric expectation is derived independently by hand (see the comment
above each assertion). Fixtures mimic ``normalize.normalizer.normalize_facts``'s
shape (annual records carry ``fy``/``period_end``/``value``), matching the
style of ``test_metrics.py``.
"""

import pandas as pd
import pytest

from sec_analyzer.valuation.multiples import multiples_history, percentile_position
from sec_analyzer.valuation.sensitivity import sensitivity_matrix

# ---------------------------------------------------------------------------
# 5. percentile_position (SPEC Sec.6)
# ---------------------------------------------------------------------------


def test_percentile_position_midrank_ties():
    # history = [10, 20, 20, 30, 40], current = 20.
    # less_count (strictly < 20) = {10} -> 1
    # equal_count (== 20) = {20, 20} -> 2
    # pct = (1 + 0.5*2) / 5 * 100 = (1 + 1) / 5 * 100 = 40.0
    result = percentile_position([10.0, 20.0, 20.0, 30.0, 40.0], 20.0)
    assert result == pytest.approx(40.0)


def test_percentile_position_no_ties_simple_case():
    # history = [10, 20, 30, 40, 50], current = 35.
    # less_count (<35) = {10,20,30} -> 3; equal_count = 0.
    # pct = 3/5*100 = 60.0
    result = percentile_position([10.0, 20.0, 30.0, 40.0, 50.0], 35.0)
    assert result == pytest.approx(60.0)


def test_percentile_position_drops_none_entries_before_counting():
    # Raw list has 6 entries but one is None -> only 5 valid values count:
    # [10, 20, 30, 40, 50], current = 25.
    # less_count (<25) = {10,20} -> 2; equal_count = 0.
    # pct = 2/5*100 = 40.0
    result = percentile_position([10.0, 20.0, None, 30.0, 40.0, 50.0], 25.0)
    assert result == pytest.approx(40.0)


def test_percentile_position_fewer_than_five_values_returns_none():
    # Only 4 non-None values -> below the _MIN_PERCENTILE_SAMPLE (5) floor.
    result = percentile_position([10.0, 20.0, 30.0, 40.0], 25.0)
    assert result is None


def test_percentile_position_none_current_returns_none():
    result = percentile_position([10.0, 20.0, 30.0, 40.0, 50.0], None)
    assert result is None


# ---------------------------------------------------------------------------
# 6. multiples_history (SPEC Sec.6)
# ---------------------------------------------------------------------------


def _mh_record(fy, value, end):
    return {
        "concept": None, "tag": None, "period_end": end,
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _mh_normalized():
    """Three fiscal years (2020-2022) of Revenue/EPS/Shares/OCF/CapEx.

    FY2020: EPS=0.8, Revenue=900,  Shares=95,  OCF=140, CapEx=50 -> fcf=90
    FY2021: EPS=1.0, Revenue=1000, Shares=100, OCF=150, CapEx=50 -> fcf=100
    FY2022: EPS=1.2, Revenue=1100, Shares=100, OCF=160, CapEx=60 -> fcf=100
    """
    concepts = {
        "Revenue": [
            _mh_record(2020, 900.0, "2020-12-31"),
            _mh_record(2021, 1000.0, "2021-12-31"),
            _mh_record(2022, 1100.0, "2022-12-31"),
        ],
        "EPS": [
            _mh_record(2020, 0.8, "2020-12-31"),
            _mh_record(2021, 1.0, "2021-12-31"),
            _mh_record(2022, 1.2, "2022-12-31"),
        ],
        "SharesOutstanding": [
            _mh_record(2020, 95.0, "2020-12-31"),
            _mh_record(2021, 100.0, "2021-12-31"),
            _mh_record(2022, 100.0, "2022-12-31"),
        ],
        "OperatingCashFlow": [
            _mh_record(2020, 140.0, "2020-12-31"),
            _mh_record(2021, 150.0, "2021-12-31"),
            _mh_record(2022, 160.0, "2022-12-31"),
        ],
        "CapEx": [
            _mh_record(2020, 50.0, "2020-12-31"),
            _mh_record(2021, 50.0, "2021-12-31"),
            _mh_record(2022, 60.0, "2022-12-31"),
        ],
    }
    return {"cik": 1, "entity_name": "Multiples Co", "currency": "USD", "annual": concepts, "quarterly": {}, "missing": [], "matched_tags": {}}


def _mh_price_df():
    """Daily closes starting 2021-01-04 -- deliberately does NOT reach back
    to cover FY2020's 2020-12-31 period end (earliest row is after it), so
    FY2020 must be skipped. FY2021 and FY2022 ARE covered."""
    dates = pd.to_datetime(["2021-01-04", "2021-12-31", "2022-06-15", "2022-12-29"])
    df = pd.DataFrame(
        {"Open": [1.0] * 4, "High": [1.0] * 4, "Low": [1.0] * 4, "Close": [8.0, 10.0, 13.0, 16.0], "Volume": [100] * 4},
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_multiples_history_skips_fy_not_covered_and_computes_correct_ratios():
    history = multiples_history(_mh_normalized(), _mh_price_df())

    # FY2020's period_end (2020-12-31) is before the earliest price row
    # (2021-01-04) -> no eligible price -> skipped entirely.
    fys = [h["fy"] for h in history]
    assert 2020 not in fys
    assert fys == [2021, 2022]

    # FY2021: last Close on/before 2021-12-31 is the exact-match row -> 10.0.
    #   pe   = 10.0 / 1.0        = 10.0
    #   ps   = 10.0*100 / 1000   = 1.0
    #   pfcf = 10.0*100 / 100    = 10.0   (fcf = 150-50 = 100)
    fy2021 = next(h for h in history if h["fy"] == 2021)
    assert fy2021["end"] == "2021-12-31"
    assert fy2021["price"] == pytest.approx(10.0)
    assert fy2021["pe"] == pytest.approx(10.0)
    assert fy2021["ps"] == pytest.approx(1.0)
    assert fy2021["pfcf"] == pytest.approx(10.0)

    # FY2022: last Close on/before 2022-12-31 -- no exact-date row, so the
    # last available (2022-12-29, Close=16.0) is used.
    #   pe   = 16.0 / 1.2         = 13.3333...
    #   ps   = 16.0*100 / 1100    = 1.454545...
    #   pfcf = 16.0*100 / 100     = 16.0   (fcf = 160-60 = 100)
    fy2022 = next(h for h in history if h["fy"] == 2022)
    assert fy2022["price"] == pytest.approx(16.0)
    assert fy2022["pe"] == pytest.approx(16.0 / 1.2, rel=1e-6)
    assert fy2022["ps"] == pytest.approx(16.0 * 100 / 1100, rel=1e-6)
    assert fy2022["pfcf"] == pytest.approx(16.0, rel=1e-6)


def test_multiples_history_returns_empty_list_when_price_df_is_none():
    history = multiples_history(_mh_normalized(), None)
    assert history == []


def test_multiples_history_never_raises_on_empty_normalized():
    assert multiples_history({}, _mh_price_df()) == []


# ---------------------------------------------------------------------------
# 7. sensitivity_matrix (SPEC Sec.9)
# ---------------------------------------------------------------------------


def test_sensitivity_matrix_shape_and_hand_verified_cells():
    # base growth=0.10, r=0.10, terminal_growth=0.03 (SAME numbers as the
    # DCF "case A" in test_valuation_dcf.py) -> center cell reuses that
    # hand-verified result: per_share ~= 216.7679 -> rounds to 216.77 (or
    # 216.77-ish; we assert with tolerance).
    #
    # growth_values = [0.08, 0.10, 0.12], dr_values = [0.09, 0.10, 0.11].
    #
    # Non-center cell [row0][col1] = growth=0.08, r=0.10 (terminal_growth
    # still 0.03):
    #   fcf1=108, fcf2=116.64, fcf3=125.9712, fcf4=136.048896, fcf5=146.93280768
    #   g6=0.07 -> fcf6=157.218104..., g7=0.06 -> fcf7=166.651190...,
    #   g8=0.05 -> fcf8=174.983750..., g9=0.04 -> fcf9=181.983100...,
    #   g10=0.03 -> fcf10=187.442593...
    #   Discounting at 10% (1.10^y as in case A) and summing PVs plus
    #   pv(TV) where TV = fcf10*1.03/0.07 = 2758.0839, pv(TV) ~= 1063.36:
    #   pv_sum(1..10) ~= 878.73, ev ~= 1942.09, equity ~= 1942.09 (no debt),
    #   per_share = 1942.09 / 10 ~= 194.21
    base_assumptions = {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10}
    result = sensitivity_matrix(base_assumptions, fcf0=100.0, shares=10.0, dilution_rate=0.0)

    assert result["growth_values"] == [0.08, 0.10, 0.12]
    assert result["dr_values"] == [0.09, 0.10, 0.11]
    assert len(result["matrix"]) == 3
    assert all(len(row) == 3 for row in result["matrix"])

    center_cell = result["matrix"][1][1]
    assert center_cell == pytest.approx(216.77, rel=1e-3)

    off_center_cell = result["matrix"][0][1]
    assert off_center_cell == pytest.approx(194.21, rel=1e-3)


def test_sensitivity_matrix_cell_is_none_when_discount_rate_at_or_below_terminal_growth():
    # base discount_rate=0.05, terminal_growth=0.04 -> dr_values =
    # [0.04, 0.05, 0.06]. Column 0 (r=0.04) has r <= terminal_growth(0.04)
    # -> the ENTIRE first column (all 3 growth rows) must be None, per
    # SPEC Sec.9 ("None if that cell has r <= g_t").
    base_assumptions = {"growth_5y": 0.08, "terminal_growth": 0.04, "discount_rate": 0.05}
    result = sensitivity_matrix(base_assumptions, fcf0=100.0, shares=10.0, dilution_rate=0.0)

    assert result["dr_values"][0] == 0.04
    for row in result["matrix"]:
        assert row[0] is None
    # Columns 1 and 2 (r=0.05, r=0.06) are both > terminal_growth(0.04) and
    # must be computable (not None).
    for row in result["matrix"]:
        assert row[1] is not None
        assert row[2] is not None


def test_sensitivity_matrix_high_uncertainty_flagged_true_when_spread_is_large():
    # base discount_rate=0.045 is very close to terminal_growth=0.03, so
    # (r - g_t) ranges only from 0.005 (col0, r=0.035) to 0.025 (col2,
    # r=0.055) across the 3x3 grid -- a 5x swing in the Gordon-growth
    # denominator alone. Since TV ~ fcf10*(1+g_t)/(r-g_t), the terminal
    # value (which dominates this low-discount-rate regime) swings by
    # roughly the same factor after accounting for the mild ~1.2x
    # offsetting effect of the differing 10-year discount factors --
    # comfortably exceeding the 60% (hi-lo)/base_cell threshold.
    base_assumptions = {"growth_5y": 0.05, "terminal_growth": 0.03, "discount_rate": 0.045}
    result = sensitivity_matrix(base_assumptions, fcf0=100.0, shares=10.0, dilution_rate=0.0)

    assert result["high_uncertainty"] is True
    # Re-derive the flag independently from the matrix's own reported
    # lo/hi/base_cell using the documented formula, as an internal-
    # consistency cross-check on top of the plausibility argument above.
    base_cell = result["matrix"][1][1]
    expected_flag = (result["hi"] - result["lo"]) / abs(base_cell) > 0.60
    assert expected_flag is True


def test_sensitivity_matrix_high_uncertainty_flagged_false_when_spread_is_small():
    # base discount_rate=0.12 is far from terminal_growth=0.02: (r - g_t)
    # only ranges from 0.09 (r=0.11) to 0.11 (r=0.13) across the grid -- a
    # mild ~22% swing in the Gordon-growth denominator, nowhere near
    # enough (even combined with the modest growth_5y +/-2pp swing) to
    # push the per-share spread past the 60% threshold.
    base_assumptions = {"growth_5y": 0.10, "terminal_growth": 0.02, "discount_rate": 0.12}
    result = sensitivity_matrix(base_assumptions, fcf0=100.0, shares=10.0, dilution_rate=0.0)

    assert result["high_uncertainty"] is False
    base_cell = result["matrix"][1][1]
    expected_flag = (result["hi"] - result["lo"]) / abs(base_cell) > 0.60
    assert expected_flag is False


def test_sensitivity_matrix_returns_none_when_shares_unusable():
    assert sensitivity_matrix({"growth_5y": 0.1, "terminal_growth": 0.02, "discount_rate": 0.1}, fcf0=100.0, shares=0.0) is None
    assert sensitivity_matrix({"growth_5y": 0.1, "terminal_growth": 0.02, "discount_rate": 0.1}, fcf0=None, shares=10.0) is None
