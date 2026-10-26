"""TTM cash-flow base for ``fcf0`` (SPEC.md Sec.30, WP14).

NVDA supplies the hand-verified numbers: a late-January fiscal year, so on
2026-10-26 the latest 10-K (FY2026, to 2026-01-25) is two quarters behind the
newest 10-Q (to 2026-07-26).

    FY2026:  OCF 102.72 - CapEx 6.04 - SBC 6.39 = 90.29  ($B)
    TTM:     OCF 134.36 - CapEx 7.36 - SBC 7.24 = 119.76 ($B)
"""

import pytest

from sec_analyzer.normalize.metrics import _compute_ttm_cash
from sec_analyzer.normalize.normalizer import normalize_facts
from sec_analyzer.valuation.engine import _select_fcf0, _ttm_staleness_notes, run_valuation

B = 1e9

# NVDA's single quarters Q3 FY2026 .. Q2 FY2027: (period_end, start, OCF, CapEx, SBC).
_NVDA_QUARTERS = [
    ("2025-10-26", "2025-07-28", 23.75, 1.64, 1.65),
    ("2026-01-25", "2025-10-27", 36.19, 1.28, 1.63),
    ("2026-04-26", "2026-01-26", 50.34, 1.76, 1.93),
    ("2026-07-26", "2026-04-27", 24.08, 2.68, 2.03),
]


def _rec(period_end, value, start=None, fp="Q1", form="10-Q", fy=None):
    return {
        "concept": None, "tag": None, "period_end": period_end,
        "fy": fy if fy is not None else int(period_end[:4]),
        "fp": fp, "form": form, "value": value,
        "filed": None, "start": start, "unit": "USD",
    }


def _nvda_normalized(quarters=_NVDA_QUARTERS, with_sbc=True, fy_end="2026-01-25"):
    annual = {"OperatingCashFlow": [_rec(fy_end, 102.72 * B, "2025-01-27", "FY", "10-K", 2026)]}
    quarterly = {
        "OperatingCashFlow": [_rec(pe, ocf * B, st) for pe, st, ocf, _, _ in quarters],
        "CapEx": [_rec(pe, cx * B, st) for pe, st, _, cx, _ in quarters],
    }
    if with_sbc:
        quarterly["SBC"] = [_rec(pe, sbc * B, st) for pe, st, _, _, sbc in quarters]
    return {"annual": annual, "quarterly": quarterly}


# ---------------------------------------------------------------------------
# 30a. _compute_ttm_cash
# ---------------------------------------------------------------------------


def test_ttm_cash_sums_four_contiguous_quarters():
    ttm = _compute_ttm_cash(_nvda_normalized(), latest_fundamental_fy=2026)

    assert ttm["ttm_cash_complete"] is True
    assert ttm["ttm_cash_period_end"] == "2026-07-26"
    assert ttm["ttm_ocf"] == pytest.approx(134.36 * B)
    assert ttm["ttm_capex"] == pytest.approx(7.36 * B)
    assert ttm["ttm_sbc"] == pytest.approx(7.24 * B)
    assert ttm["ttm_fcf_sbc_adj"] == pytest.approx(119.76 * B)
    assert ttm["ttm_cash_newer_than_fy"] is True


def test_ttm_cash_window_with_a_missing_quarter_is_incomplete():
    # Drop Q4 FY2026: the newest four now span 2025-07-27 .. 2026-07-26
    # (364 days > 300), so the sum is not a twelve-month figure.
    quarters = [("2025-07-27", "2025-04-28", 15.37, 1.90, 1.62)] + [
        q for q in _NVDA_QUARTERS if q[0] != "2026-01-25"
    ]
    ttm = _compute_ttm_cash(_nvda_normalized(quarters), latest_fundamental_fy=2026)

    assert ttm["ttm_cash_complete"] is False
    assert ttm["ttm_fcf_sbc_adj"] is None


def test_ttm_cash_mismatched_period_ends_are_incomplete():
    normalized = _nvda_normalized()
    normalized["quarterly"]["CapEx"] = normalized["quarterly"]["CapEx"][:-1]  # CapEx stops a quarter early

    ttm = _compute_ttm_cash(normalized, latest_fundamental_fy=2026)

    assert ttm["ttm_cash_complete"] is False
    assert ttm["ttm_fcf_sbc_adj"] is None


def test_ttm_cash_missing_sbc_counts_as_zero():
    ttm = _compute_ttm_cash(_nvda_normalized(with_sbc=False), latest_fundamental_fy=2026)

    assert ttm["ttm_cash_complete"] is True
    assert ttm["ttm_sbc"] == 0.0
    assert ttm["ttm_fcf_sbc_adj"] == pytest.approx((134.36 - 7.36) * B)


def test_ttm_cash_window_ending_on_the_fiscal_year_is_not_newer():
    # MU-style: the fiscal year just closed, so TTM == FY and adds nothing.
    ttm = _compute_ttm_cash(_nvda_normalized(fy_end="2026-07-26"), latest_fundamental_fy=2026)

    assert ttm["ttm_cash_complete"] is True
    assert ttm["ttm_cash_newer_than_fy"] is False


def test_ttm_cash_is_empty_without_quarterly_data():
    ttm = _compute_ttm_cash({"annual": {}, "quarterly": {}}, latest_fundamental_fy=2026)

    assert ttm["ttm_cash_complete"] is False
    assert ttm["ttm_fcf_sbc_adj"] is None
    assert ttm["ttm_cash_newer_than_fy"] is False


# ---------------------------------------------------------------------------
# 30b/30c. _select_fcf0 candidate and label
# ---------------------------------------------------------------------------

# NVDA's SBC-adjusted annual FCF ($B), FY2024-FY2026: a monotonic ramp.
_NVDA_FCF_BY_FY = {2024: 23.6 * B, 2025: 56.1 * B, 2026: 90.29 * B}


def _ttm_metrics(ttm_fcf, newer=True, complete=True):
    return {
        "latest_fy": 2026, "latest_fundamental_fy": 2026,
        "ttm_cash_complete": complete, "ttm_cash_newer_than_fy": newer,
        "ttm_fcf_sbc_adj": ttm_fcf,
    }


def test_select_fcf0_uses_newer_complete_ttm():
    # TTM 119.76 vs prior avg (56.1+23.6)/2 = 39.85 -> deviates, but
    # (23.6, 56.1, 119.76) is a monotonic ramp -> kept, labelled "ttm".
    fcf0, source, note = _select_fcf0(_ttm_metrics(119.76 * B), _NVDA_FCF_BY_FY)

    assert fcf0 == pytest.approx(119.76 * B)
    assert source == "ttm"
    assert "trailing-twelve-month" in note


def test_select_fcf0_keeps_latest_fy_when_ttm_is_not_newer():
    fcf0, source, _ = _select_fcf0(_ttm_metrics(119.76 * B, newer=False), _NVDA_FCF_BY_FY)

    assert fcf0 == pytest.approx(90.29 * B)
    assert source == "latest_fy"


def test_select_fcf0_keeps_latest_fy_when_ttm_is_incomplete():
    fcf0, source, _ = _select_fcf0(_ttm_metrics(None, complete=False), _NVDA_FCF_BY_FY)

    assert fcf0 == pytest.approx(90.29 * B)
    assert source == "latest_fy"


def test_select_fcf0_noisy_ttm_falls_back_to_average_including_it():
    # SMCI-style: TTM 5.0 against priors 40.0/30.0 -> deviates (|5-35|/35 = 86%)
    # and (30, 40, 5) is not monotonic -> 3y average of (5.0, 40.0, 30.0) = 25.0.
    fcf_by_fy = {2024: 30.0 * B, 2025: 40.0 * B, 2026: 45.0 * B}
    fcf0, source, note = _select_fcf0(_ttm_metrics(5.0 * B), fcf_by_fy)

    assert source == "3y_avg"
    assert fcf0 == pytest.approx(25.0 * B)
    assert "trailing-twelve-month" in note


def test_select_fcf0_negative_ttm_falls_back_to_average():
    fcf_by_fy = {2024: 30.0 * B, 2025: 40.0 * B, 2026: 45.0 * B}
    fcf0, source, _ = _select_fcf0(_ttm_metrics(-7.0 * B), fcf_by_fy)

    assert source == "3y_avg"
    assert fcf0 == pytest.approx((-7.0 + 40.0 + 30.0) / 3 * B)


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def _engine_inputs(ttm_fcf):
    normalized = {"cik": 1, "entity_name": "Engine Test Co", "currency": "USD",
                  "annual": {}, "quarterly": {}, "missing": [], "matched_tags": {}}
    ratios = [{"fy": fy, "fcf": v} for fy, v in _NVDA_FCF_BY_FY.items()]
    metrics = {"shares": 24.3 * B, "fcf": 90.29 * B, "net_debt": 0.0, **_ttm_metrics(ttm_fcf)}
    assumptions = {
        s: {"growth_5y": g, "terminal_growth": 0.03, "discount_rate": r, "story": s}
        for s, g, r in (("bear", 0.05, 0.12), ("base", 0.10, 0.10), ("bull", 0.15, 0.09))
    }
    return normalized, ratios, metrics, assumptions


def test_run_valuation_dcf_base_scales_with_ttm_fcf0():
    # FCFE-direct: per-share value is linear in fcf0, so the TTM run's base
    # is exactly 119.76/90.29 times the fiscal-year run's.
    def base(ttm_fcf, newer):
        normalized, ratios, metrics, assumptions = _engine_inputs(ttm_fcf)
        metrics["ttm_cash_newer_than_fy"] = newer
        result = run_valuation(normalized, ratios, metrics, price=None, price_df=None,
                               assumptions=assumptions, sector_type="mature")
        return result

    fy_run, ttm_run = base(119.76 * B, newer=False), base(119.76 * B, newer=True)

    assert fy_run["fcf0_source"] == "latest_fy"
    assert ttm_run["fcf0_source"] == "ttm"
    ratio = (ttm_run["dcf"]["scenarios"]["base"]["per_share"]
             / fy_run["dcf"]["scenarios"]["base"]["per_share"])
    assert ratio == pytest.approx(119.76 / 90.29, rel=1e-3)


def test_staleness_note_names_the_ttm_dcf_base():
    metrics = {"ttm_complete": True, "ttm_vs_fy_net_income": 1.6,
               "ttm_net_income": 192.9 * B, "ttm_period_end": "2026-07-26"}

    ttm_note = _ttm_staleness_notes(metrics, "ttm")[0]
    fy_note = _ttm_staleness_notes(metrics, "latest_fy")[0]

    assert "The FCF-DCF uses the TTM cash-flow base; every other anchor is computed" in ttm_note
    assert "Every anchor is computed from the annual series" in fy_note


# ---------------------------------------------------------------------------
# As-of mode: a quarter filed after as_of never enters the window
# ---------------------------------------------------------------------------


def _facts():
    def row(start, end, val, form, fp, filed, fy):
        return {"start": start, "end": end, "val": val, "form": form, "fp": fp,
                "filed": filed, "fy": fy, "accn": f"0000000000-{filed}"}

    def tag(rows):
        return {"units": {"USD": rows}}

    ocf, capex = [], []
    # FY2026 10-K plus YTD 10-Q rows, mirroring how NVDA reports cash flow.
    for target, values in ((ocf, (102.72, 50.34, 74.42, 66.53, 42.78, 27.41)),
                           (capex, (6.04, 1.76, 4.43, 4.76, 3.12, 1.23))):
        fy, q1, h1, py9m, pyh1, pyq1 = (v * B for v in values)
        target += [
            row("2025-01-27", "2025-04-27", pyq1, "10-Q", "Q1", "2025-05-28", 2026),
            row("2025-01-27", "2025-07-27", pyh1, "10-Q", "Q2", "2025-08-27", 2026),
            row("2025-01-27", "2025-10-26", py9m, "10-Q", "Q3", "2025-11-19", 2026),
            row("2025-01-27", "2026-01-25", fy, "10-K", "FY", "2026-02-25", 2026),
            row("2026-01-26", "2026-04-26", q1, "10-Q", "Q1", "2026-05-27", 2027),
            row("2026-01-26", "2026-07-26", h1, "10-Q", "Q2", "2026-08-26", 2027),
        ]
    return {"cik": 1045810, "entityName": "Test", "facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": tag(ocf),
        "PaymentsToAcquirePropertyPlantAndEquipment": tag(capex),
    }}}


def test_as_of_excludes_quarters_filed_later():
    full = _compute_ttm_cash(normalize_facts(_facts()), latest_fundamental_fy=2026)
    as_of = _compute_ttm_cash(normalize_facts(_facts(), as_of="2026-06-30"), latest_fundamental_fy=2026)

    assert full["ttm_cash_period_end"] == "2026-07-26"
    assert as_of["ttm_cash_period_end"] == "2026-04-26"
