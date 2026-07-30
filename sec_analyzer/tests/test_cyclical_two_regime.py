"""Unit tests for the cyclical work packages: WP8/9/10.

SPEC.md Sec.24 (quarterly fiscal-year anchoring + TTM base), Sec.25
(through-cycle statistics + historical P/B band) and Sec.26 (two-regime
valuation + implied break probability).

Micron supplies the hand-verified numbers throughout: a late-August fiscal
year (which is what breaks calendar-year quarter grouping), and a FY2016-2025
net-margin series running from -37.5% to +46.5%.
"""

import pytest

from sec_analyzer.normalize.normalizer import (
    _fiscal_year_end_month,
    _quarter_fiscal_year,
    to_quarterly_series,
)
from sec_analyzer.normalize.metrics import _compute_ttm
from sec_analyzer.valuation import cyclical

# MU's real annual net margins, FY2016-FY2025.
_MU_MARGINS = {
    2016: -0.022, 2017: 0.250, 2018: 0.465, 2019: 0.270, 2020: 0.125,
    2021: 0.212, 2022: 0.282, 2023: -0.375, 2024: 0.031, 2025: 0.228,
}


def _rec(period_end, value, fy=None, start=None):
    return {
        "concept": None, "tag": None, "period_end": period_end,
        "fy": fy if fy is not None else int(period_end[:4]),
        "fp": "Q1", "form": "10-Q", "value": value,
        "filed": None, "start": start, "unit": "USD",
    }


# ---------------------------------------------------------------------------
# 24a. Quarterly fiscal-year anchoring
# ---------------------------------------------------------------------------


def test_fiscal_year_end_month_reads_the_annual_records():
    normalized = {"annual": {"Revenue": [
        {"period_end": "2025-08-28", "value": 1.0},
        {"period_end": "2024-08-29", "value": 1.0},
    ]}}

    assert _fiscal_year_end_month(normalized) == 8


def test_fiscal_year_end_month_is_none_without_annual_records():
    assert _fiscal_year_end_month({"annual": {}}) is None
    assert _fiscal_year_end_month({}) is None


@pytest.mark.parametrize(
    "period_end, fy_end_month, expected",
    [
        # August fiscal year (Micron): Q1 ends in November, i.e. the NEXT
        # calendar year from its own fiscal year's label.
        ("2025-11-27", 8, 2026),
        ("2026-02-26", 8, 2026),
        ("2026-05-28", 8, 2026),
        ("2026-08-27", 8, 2026),
        # September fiscal year (Apple): the December quarter is FY+1.
        ("2025-12-27", 9, 2026),
        ("2026-03-28", 9, 2026),
        # December fiscal year: everything stays on its calendar year.
        ("2026-03-31", 12, 2026),
        ("2026-12-31", 12, 2026),
    ],
)
def test_quarter_fiscal_year_anchors_on_the_year_end(period_end, fy_end_month, expected):
    assert _quarter_fiscal_year(period_end, fy_end_month) == expected


def test_quarter_fiscal_year_falls_back_without_a_year_end_month():
    assert _quarter_fiscal_year("2026-03-31", None) == 2026


def test_q4_derivation_is_correct_for_an_august_fiscal_year():
    """The MU defect: the calendar-2025 bucket held FY2025's Q2/Q3 plus
    FY2026's Q1, so Q4 came out 37.38 - 30.99 = 6.39 instead of 11.32."""
    normalized = {
        "annual": {"Revenue": [_rec("2025-08-28", 37.38e9)]},
        "quarterly": {"Revenue": [
            _rec("2024-11-28", 8.71e9, start="2024-08-30"),
            _rec("2025-02-27", 8.05e9, start="2024-11-29"),
            _rec("2025-05-29", 9.30e9, start="2025-02-28"),
            _rec("2025-11-27", 13.64e9, start="2025-08-29"),
        ]},
    }

    quarters = {q["period_end"]: q for q in to_quarterly_series(normalized, "Revenue")}

    assert "2025-08-28" in quarters, "Q4 must be derived"
    q4 = quarters["2025-08-28"]
    assert q4["derived"] is True
    assert q4["value"] == pytest.approx(11.32e9, rel=1e-3)
    # The FY2026 Q1 must NOT have been absorbed into FY2025's derivation.
    assert quarters["2025-11-27"]["value"] == pytest.approx(13.64e9)


def test_december_fiscal_year_grouping_is_unchanged():
    normalized = {
        "annual": {"Revenue": [_rec("2025-12-31", 400.0)]},
        "quarterly": {"Revenue": [
            _rec("2025-03-31", 90.0, start="2025-01-01"),
            _rec("2025-06-30", 100.0, start="2025-04-01"),
            _rec("2025-09-30", 105.0, start="2025-07-01"),
        ]},
    }

    quarters = {q["period_end"]: q for q in to_quarterly_series(normalized, "Revenue")}

    assert quarters["2025-12-31"]["value"] == pytest.approx(105.0)
    assert quarters["2025-12-31"]["derived"] is True


# ---------------------------------------------------------------------------
# 24b. TTM
# ---------------------------------------------------------------------------


def _ttm_normalized():
    """MU's real last four quarters (Q4 FY2025 through Q3 FY2026)."""
    def q(period_end, revenue, net_income, start):
        return period_end, revenue, net_income, start
    rows = [
        q("2025-08-28", 11.32e9, 3.20e9, "2025-05-30"),
        q("2025-11-27", 13.64e9, 5.24e9, "2025-08-29"),
        q("2026-02-26", 23.86e9, 13.79e9, "2025-11-28"),
        q("2026-05-28", 41.46e9, 28.24e9, "2026-02-27"),
    ]
    return {
        "annual": {
            "Revenue": [_rec("2025-08-28", 37.38e9)],
            "NetIncome": [_rec("2025-08-28", 8.54e9)],
        },
        "quarterly": {
            "Revenue": [_rec(pe, rev, start=st) for pe, rev, _, st in rows],
            "NetIncome": [_rec(pe, ni, start=st) for pe, _, ni, st in rows],
        },
    }


def test_ttm_sums_the_last_four_quarters():
    ttm = _compute_ttm(_ttm_normalized(), shares=1.1253e9, price=739.0,
                       latest_fy_net_income=8.54e9)

    assert ttm["ttm_complete"] is True
    assert ttm["ttm_quarters"] == 4
    assert ttm["ttm_period_end"] == "2026-05-28"
    assert ttm["ttm_revenue"] == pytest.approx(90.28e9, rel=1e-4)
    assert ttm["ttm_net_income"] == pytest.approx(50.47e9, rel=1e-4)
    assert ttm["ttm_net_margin"] == pytest.approx(0.5591, rel=1e-3)
    assert ttm["ttm_eps"] == pytest.approx(44.85, rel=1e-3)
    assert ttm["pe_ttm"] == pytest.approx(16.48, rel=1e-3)
    # The whole point: 5.9x the fiscal-year base the anchors use.
    assert ttm["ttm_vs_fy_net_income"] == pytest.approx(5.91, rel=1e-2)


def test_incomplete_ttm_window_reports_partials_but_no_pe():
    normalized = _ttm_normalized()
    for concept in ("Revenue", "NetIncome"):
        normalized["quarterly"][concept] = normalized["quarterly"][concept][-2:]

    ttm = _compute_ttm(normalized, shares=1.1253e9, price=739.0, latest_fy_net_income=8.54e9)

    assert ttm["ttm_complete"] is False
    assert ttm["ttm_quarters"] == 2
    assert ttm["ttm_net_income"] is not None
    assert ttm["pe_ttm"] is None


def test_ttm_is_empty_without_quarterly_data():
    ttm = _compute_ttm({"annual": {}, "quarterly": {}}, shares=1.0, price=10.0,
                       latest_fy_net_income=1.0)

    assert ttm["ttm_quarters"] == 0
    assert ttm["ttm_complete"] is False
    assert ttm["ttm_net_income"] is None


# ---------------------------------------------------------------------------
# 25. Through-cycle statistics
# ---------------------------------------------------------------------------


def _ratios(margins=None):
    return [{"fy": fy, "net_margin": m} for fy, m in sorted((margins or _MU_MARGINS).items())]


def _normalized_revenue_series():
    return {"annual": {
        "Revenue": [_rec(f"{fy}-08-28", v) for fy, v in
                    ((2023, 15.54e9), (2024, 25.11e9), (2025, 37.38e9))],
        "StockholdersEquity": [_rec("2025-08-28", 54.16e9)],
    }}


def _history():
    # Ten years of observed P/B and P/E.
    pbs = [1.26, 1.50, 1.79, 2.10, 3.56, 1.40, 2.00, 1.60, 1.90, 2.20]
    pes = [8.0, 9.5, 10.0, 10.5, 12.0, 7.0, 11.0, 13.0, 9.0, 14.0]
    return [{"fy": 2016 + i, "pb": pbs[i], "pe": pes[i]} for i in range(10)]


def _metrics(**over):
    base = {
        "shares": 1.1253e9, "price": 739.0,
        "ttm_complete": True, "ttm_net_margin": 0.5591,
        "ttm_revenue": 90.28e9, "pe_ttm": 16.48,
    }
    base.update(over)
    return base


def test_through_cycle_stats_summarizes_the_margin_distribution():
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(), _metrics()
    )

    assert stats["years"] == 10
    assert stats["window_short"] is False
    assert stats["margin_mean"] == pytest.approx(0.1466, abs=1e-4)
    assert stats["margin_trough"] == pytest.approx(-0.375)
    assert stats["margin_trough_fy"] == 2023
    assert stats["margin_peak"] == pytest.approx(0.465)
    assert stats["margin_peak_fy"] == 2018
    # TTM margin (55.9%) is above every historical year.
    assert stats["margin_current_basis"] == "ttm"
    assert stats["margin_percentile"] == pytest.approx(100.0)


def test_through_cycle_stats_falls_back_to_the_fiscal_year_margin():
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(),
        _metrics(ttm_complete=False),
    )

    assert stats["margin_current_basis"] == "fy"
    assert stats["margin_current"] == pytest.approx(_MU_MARGINS[2025])


def test_through_cycle_stats_needs_a_minimum_history():
    short = {fy: m for fy, m in list(_MU_MARGINS.items())[:4]}

    assert cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(short), _history(), _metrics()
    ) is None


def test_short_window_is_flagged():
    six = {fy: m for fy, m in list(_MU_MARGINS.items())[:6]}
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(six), _history(), _metrics()
    )

    assert stats["years"] == 6
    assert stats["window_short"] is True


def test_pb_band_comes_from_the_observed_history():
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(), _metrics()
    )

    assert stats["pb_trough"] == pytest.approx(1.26)
    assert stats["pb_peak"] == pytest.approx(3.56)
    assert stats["pb_median"] == pytest.approx(1.845)
    # Current P/B is derived from price and today's book value per share.
    assert stats["pb_current"] == pytest.approx(739.0 / (54.16e9 / 1.1253e9), rel=1e-3)


def test_normalized_revenue_substitutes_ttm_for_the_newest_year():
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(), _metrics()
    )

    assert stats["revenue_basis_years"] == [2023, 2024, "TTM"]
    assert stats["normalized_revenue"] == pytest.approx((15.54e9 + 25.11e9 + 90.28e9) / 3)


def test_normalized_revenue_keeps_annual_points_without_a_ttm_window():
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(),
        _metrics(ttm_complete=False),
    )

    assert stats["revenue_basis_years"] == [2023, 2024, 2025]
    assert stats["normalized_revenue"] == pytest.approx((15.54e9 + 25.11e9 + 37.38e9) / 3)


def test_through_cycle_stats_never_raises():
    assert cyclical.through_cycle_stats(None, None, None, None) is None
    assert cyclical.through_cycle_stats({"annual": "junk"}, _ratios(), None, _metrics()) is None


# ---------------------------------------------------------------------------
# 26. Two regimes, blend, implied probability
# ---------------------------------------------------------------------------


def _cycle(price=739.0, **metric_over):
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(), _metrics(**metric_over)
    )
    return cyclical.two_regime_valuation(stats, _metrics(**metric_over), price)


def test_two_regime_builds_both_regimes():
    cycle = _cycle()

    assert cycle["regime_a"]["center"] is not None
    assert cycle["regime_b"]["center"] is not None
    assert cycle["regime_b"]["revenue_basis"] == "TTM"


def test_regime_a_center_takes_the_lower_of_the_two_reads():
    cycle = _cycle()
    stats, a = cycle["stats"], cycle["regime_a"]

    earnings_read = a["normalized_eps"] * stats["pe_median"]
    book_read = stats["pb_median"] * stats["book_value_per_share"]
    assert a["center"] == pytest.approx(min(earnings_read, book_read))


def test_regime_a_band_never_crosses_itself():
    cycle = _cycle()
    a = cycle["regime_a"]

    assert a["low"] <= a["center"] <= a["high"]
    # MU's trough P/B floor lands just above the earnings-based centre, so the
    # disagreement must be recorded rather than silently ordered away.
    assert a["band_crossed"] is True


def test_regime_b_sensitivity_is_a_full_margin_by_multiple_grid():
    b = _cycle()["regime_b"]

    assert set(b["sensitivity"]) == {"0.30", "0.38", "0.45"}
    for row in b["sensitivity"].values():
        assert set(row) == {"12.0", "13.5", "15.0"}
    # A corner check: 38% margin x 13.5x on the TTM revenue base.
    expected = 0.38 * 90.28e9 / 1.1253e9 * 13.5
    assert b["sensitivity"]["0.38"]["13.5"] == pytest.approx(expected, rel=1e-6)
    assert b["center"] == pytest.approx(expected, rel=1e-6)


def test_regime_b_needs_a_complete_ttm_window():
    cycle = _cycle(ttm_complete=False)

    assert cycle["regime_b"] is None
    assert cycle["p_implied"] is None


def test_blend_interpolates_between_the_two_centers():
    cycle = _cycle()
    fv_a = cycle["regime_a"]["center"]
    fv_b = cycle["regime_b"]["center"]

    assert cycle["blend"]["0.25"] == pytest.approx(0.25 * fv_b + 0.75 * fv_a)
    assert cycle["blend"]["0.50"] == pytest.approx(0.5 * (fv_a + fv_b))


def test_implied_probability_solves_the_blend():
    cycle = _cycle(price=None) or {}
    # Price exactly at the 50/50 blend must imply p = 0.5.
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(), _metrics()
    )
    probe = cyclical.two_regime_valuation(stats, _metrics(), price=1.0)
    midpoint = 0.5 * (probe["regime_a"]["center"] + probe["regime_b"]["center"])

    at_mid = cyclical.two_regime_valuation(stats, _metrics(), price=midpoint)
    assert at_mid["p_implied"] == pytest.approx(0.5)
    assert at_mid["p_implied_status"] == "ok"


def test_price_above_both_regimes_is_reported_not_clamped():
    cycle = _cycle(price=739.0)

    assert cycle["p_implied"] > 1.0
    assert cycle["p_implied_status"] == "above_range"
    assert "hiçbir olasılık karışımı açıklamıyor" in cycle["verdict_sentence"]


def test_price_below_both_regimes_is_reported_not_clamped():
    cycle = _cycle(price=10.0)

    assert cycle["p_implied"] < 0.0
    assert cycle["p_implied_status"] == "below_range"


def test_verdict_sentence_uses_the_required_shape_when_in_range():
    stats = cyclical.through_cycle_stats(
        _normalized_revenue_series(), _ratios(), _history(), _metrics()
    )
    probe = cyclical.two_regime_valuation(stats, _metrics(), price=1.0)
    midpoint = 0.5 * (probe["regime_a"]["center"] + probe["regime_b"]["center"])

    sentence = cyclical.two_regime_valuation(stats, _metrics(), price=midpoint)["verdict_sentence"]

    assert "olasılık vermeyi" in sentence
    assert "zorunlu kılıyor" in sentence
    assert "üstünde ise fiyat ucuz" in sentence


# ---- flags ----


def test_regime_change_premium_fires_above_the_historical_peak():
    flags = _cycle()["flags"]

    # P/B ~15.4 against a historical peak of 3.56.
    assert flags["regime_change_premium"] is True


def test_peak_annualization_fires_far_above_the_through_cycle_mean():
    assert _cycle()["flags"]["peak_annualization"] is True


def test_peak_cycle_pe_trap_needs_both_a_peak_margin_and_a_low_ttm_pe():
    # TTM P/E 16.5 is above the 15x bar, so the trap does not fire...
    assert _cycle()["flags"]["peak_cycle_pe_trap"] is False
    # ...but the same peak margin at a 9x TTM P/E is exactly the trap.
    assert _cycle(pe_ttm=9.0)["flags"]["peak_cycle_pe_trap"] is True


def test_two_regime_never_raises():
    assert cyclical.two_regime_valuation(None, {}, 10.0) is None
    assert cyclical.two_regime_valuation({"junk": True}, {"shares": 0}, 10.0) is None
