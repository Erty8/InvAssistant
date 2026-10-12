"""Tests for SPEC.md Sec.28 (WP12): reverse-DCF cash-flow basis + discount-rate sensitivity.

Independent oracle: ``_ref_ps`` / ``_ref_solve`` below re-implement the Sec.4 DCF
(years 1-5 grow at g, years 6-10 fade linearly to g_t, Gordon terminal value on
year 10, all discounted at r, divided by shares) and a high-precision bisection
WITHOUT calling ``dcf.py`` / ``reverse_dcf.py``.

Golden (non-cyclical invariance): ``fixtures/rate_basis_golden.json`` was generated
from the PRE-WP12 code (``git archive HEAD sec_analyzer`` into a scratch dir, then
``run_valuation`` on the ``_GOLDEN_CASES`` inputs below, serialized with
``json.dumps(sort_keys=True, default=str)``). To regenerate (only if a pre-existing
key legitimately changes), repeat that procedure against the commit before WP12.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from sec_analyzer.valuation import reverse_dcf
from sec_analyzer.valuation.engine import run_valuation
from sec_analyzer.valuation.triangulate import SIGNAL_CHEAP, SIGNAL_EXPENSIVE

# ---------------------------------------------------------------------------
# Independent DCF oracle (SPEC Sec.4)
# ---------------------------------------------------------------------------

_LO, _HI = -0.20, 0.60


def _ref_ps(fcf0, g, gt, r, shares):
    prev, pv = fcf0, 0.0
    for y in range(1, 11):
        gy = g if y <= 5 else g + (gt - g) * (y - 5) / 5.0
        prev *= 1.0 + gy
        pv += prev / (1.0 + r) ** y
    pv += prev * (1.0 + gt) / (r - gt) / (1.0 + r) ** 10
    return pv / shares


def _ref_solve(price, fcf0, gt, r, shares):
    """Return (growth_or_None, status) using 200-step bisection (full precision)."""
    d_lo = _ref_ps(fcf0, _LO, gt, r, shares) - price
    d_hi = _ref_ps(fcf0, _HI, gt, r, shares) - price
    if (d_lo > 0) == (d_hi > 0):
        return None, ("above_bracket" if d_hi < 0 else "below_bracket")
    lo, hi = _LO, _HI
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if (_ref_ps(fcf0, mid, gt, r, shares) - price > 0) == (d_lo > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, "ok"


_TOL = 2e-4  # solver: half-width < 1e-4 then 4dp rounding (5e-5)

_FCF0, _SHARES, _GT, _R = 100.0, 10.0, 0.03, 0.10


def _sens(price, fcf0=_FCF0, gt=_GT, r=_R, shares=_SHARES, **kw):
    kw.setdefault("rate_floor", 0.07)
    kw.setdefault("min_erp_spread", 0.045)
    return reverse_dcf.implied_growth_rate_sensitivity(price, fcf0, gt, r, shares, 0.0, **kw)


def test_oracle_matches_closed_form_when_g_equals_terminal_growth():
    # Hand derivation. With g = g_t = 0.03 every year grows at 3%, so
    # FCF_y = 100*1.03^y and PV_y = 100*q^y with q = 1.03/1.10 = 0.9363636.
    #   sum_{y=1..10} = 100*q*(1-q^10)/(1-q);  q^10 = exp(10 ln q) = 0.51836
    #                 = 100*0.9363636*(0.48164)/(0.0636364) = 708.7
    #   TV part       = 100*q^10*1.03/(0.10-0.03) = 762.7
    #   total 1471.4 -> per share (10 shares) = 147.14
    q = 1.03 / 1.10
    total = 100 * q * (1 - q ** 10) / (1 - q) + 100 * q ** 10 * 1.03 / 0.07
    assert total == pytest.approx(1471.4, rel=1e-3)
    assert _ref_ps(100.0, 0.03, 0.03, 0.10, 10.0) == pytest.approx(total / 10.0, rel=1e-12)
    # and the production solver inverts it back to 3%
    growth, status = reverse_dcf.implied_growth_with_status(total / 10.0, 100.0, 0.03, 0.10, 10.0)
    assert status == "ok" and growth == pytest.approx(0.03, abs=_TOL)


# ---------------------------------------------------------------------------
# 1. Shifted-rate values
# ---------------------------------------------------------------------------

def test_rate_shifts_constant_and_order():
    assert reverse_dcf.RATE_SHIFTS_BP == (-50, 50, 100)


def test_rate_sensitivity_values_match_independent_solve():
    # Price generated from the oracle at g=10%, r=10% (~216.77, the well known
    # "case A"): base implied growth ~= 0.10. Every shift is then re-solved
    # independently at r_s = 0.095 / 0.105 / 0.11 with g_t fixed at 0.03.
    price = _ref_ps(_FCF0, 0.10, _GT, _R, _SHARES)
    block = _sens(price)
    assert block is not None
    assert block["basis"] == "raw"
    assert block["base_discount_rate"] == 0.1
    assert block["terminal_growth"] == 0.03
    assert block["base_status"] == "ok"
    assert block["base_implied_growth"] == pytest.approx(0.10, abs=_TOL)

    shifts = block["shifts"]
    assert [s["shift_bp"] for s in shifts] == [-50, 50, 100]
    # exact 6dp rates: no float drift (0.095 + 0.005 == 0.1 style)
    assert [s["discount_rate"] for s in shifts] == [0.095, 0.105, 0.11]

    base_growth = block["base_implied_growth"]
    prev = None
    for s in shifts:
        expected, status = _ref_solve(price, _FCF0, _GT, s["discount_rate"], _SHARES)
        assert status == "ok" and s["status"] == "ok"
        assert s["skip_reason"] is None
        assert s["implied_growth"] == pytest.approx(expected, abs=_TOL)
        # delta_pp reproducible from the displayed 4dp values
        assert s["delta_pp"] == pytest.approx(round((s["implied_growth"] - base_growth) * 100, 2), abs=1e-9)
        if prev is not None:
            assert s["implied_growth"] >= prev  # monotone non-decreasing in shift
        prev = s["implied_growth"]
    # a higher discount rate needs more growth; lower rate needs less
    assert shifts[0]["implied_growth"] < base_growth < shifts[1]["implied_growth"] < shifts[2]["implied_growth"]
    assert shifts[0]["delta_pp"] < 0 < shifts[1]["delta_pp"] < shifts[2]["delta_pp"]


def test_rate_sensitivity_reuses_supplied_base_result_and_is_deterministic():
    price = _ref_ps(_FCF0, 0.10, _GT, _R, _SHARES)
    a = _sens(price)
    b = _sens(price)
    assert a == b
    c = _sens(price, base_result=(0.1234, "ok"))
    assert c["base_implied_growth"] == 0.1234
    # delta is measured against the supplied base
    s = c["shifts"][1]
    assert s["delta_pp"] == pytest.approx(round((s["implied_growth"] - 0.1234) * 100, 2))


def test_rate_sensitivity_custom_shifts_and_basis_echo():
    price = _ref_ps(_FCF0, 0.10, _GT, _R, _SHARES)
    block = _sens(price, shifts_bp=(25, 200), basis="normalized")
    assert block["basis"] == "normalized"
    assert [s["shift_bp"] for s in block["shifts"]] == [25, 200]
    assert [s["discount_rate"] for s in block["shifts"]] == [0.1025, 0.12]


# ---------------------------------------------------------------------------
# 2. Downward-shift guards: skipped, never clamped; order r<=g_t, floor, spread
# ---------------------------------------------------------------------------

def _first(block):
    return block["shifts"][0]


def _assert_skipped(entry, r_s, reason):
    assert entry["status"] == "skipped"
    assert entry["skip_reason"] == reason
    assert entry["implied_growth"] is None
    assert entry["delta_pp"] is None
    assert entry["discount_rate"] == r_s  # true shifted rate, not the guard value


def test_skip_r_at_or_below_terminal_growth_fires_first():
    # r=0.095, g_t=0.09 -> r-50bp = 0.09 == g_t (<=). Floor OK (0.09>=0.07) but
    # spread 0 < 0.045 also breaches: rule 1 must be the one reported.
    block = _sens(150.0, gt=0.09, r=0.095)
    _assert_skipped(_first(block), 0.09, "rate_at_or_below_terminal_growth")
    # Explicit floor/spread of 0 -> r<=g_t is the only rule that can fire.
    block = _sens(150.0, gt=0.09, r=0.095, rate_floor=0.0, min_erp_spread=0.0)
    _assert_skipped(_first(block), 0.09, "rate_at_or_below_terminal_growth")


def test_skip_r_below_terminal_growth_beats_floor_and_spread():
    # r=0.0705, g_t=0.068: r-50bp = 0.0655 < g_t AND < 0.07 floor AND spread<0.
    block = _sens(150.0, gt=0.068, r=0.0705)
    _assert_skipped(_first(block), 0.0655, "rate_at_or_below_terminal_growth")


def test_skip_below_rate_floor():
    # r=0.0725, g_t=0.02: r-50bp = 0.0675 < 0.07; spread 0.0475 >= 0.045 OK.
    block = _sens(150.0, gt=0.02, r=0.0725)
    _assert_skipped(_first(block), 0.0675, "below_rate_floor")


def test_floor_beats_erp_spread():
    # r=0.0725, g_t=0.03: r_s=0.0675 breaches BOTH floor (0.07) and spread
    # (0.0375 < 0.045). Floor is rule 2 -> reported.
    block = _sens(150.0, gt=0.03, r=0.0725)
    _assert_skipped(_first(block), 0.0675, "below_rate_floor")


def test_skip_below_erp_spread():
    # r=0.0895, g_t=0.04: r_s=0.0845 (>= floor); spread 0.0445 < 0.045.
    block = _sens(150.0, gt=0.04, r=0.0895)
    _assert_skipped(_first(block), 0.0845, "below_erp_spread")


def test_erp_spread_boundary_is_not_skipped():
    # r=0.09, g_t=0.04: r_s=0.085, spread exactly 0.045 (float 0.04499999..).
    # The Sec.3 epsilon (1e-9) must let it run.
    price = _ref_ps(_FCF0, 0.10, 0.04, 0.09, _SHARES)
    block = _sens(price, gt=0.04, r=0.09)
    assert block["shifts"][0]["status"] == "ok"
    assert block["shifts"][0]["discount_rate"] == 0.085


def test_rate_floor_boundary_is_not_skipped():
    # r=0.075, g_t=0.02: r_s = 0.07 == floor -> runs.
    price = _ref_ps(_FCF0, 0.10, 0.02, 0.075, _SHARES)
    block = _sens(price, gt=0.02, r=0.075)
    assert block["shifts"][0]["status"] == "ok"


def test_unprofitable_floor_10pct_passed_explicitly():
    # growth_unprofitable floor 0.10: base 0.10 -> -50bp = 0.095 skipped;
    # base 0.1025 -> -50bp = 0.0525?? no: 0.0975 skipped; base 0.105 -> 0.10 runs.
    b1 = _sens(150.0, gt=0.03, r=0.10, rate_floor=0.10)
    _assert_skipped(_first(b1), 0.095, "below_rate_floor")
    price = _ref_ps(_FCF0, 0.10, 0.03, 0.105, _SHARES)
    b2 = _sens(price, gt=0.03, r=0.105, rate_floor=0.10)
    assert _first(b2)["status"] == "ok" and _first(b2)["discount_rate"] == 0.1


def test_upward_shifts_never_skipped_even_when_base_sits_on_guards():
    # base r = g_t + 0.045 exactly (typical low-beta filer): only -50bp is skipped.
    block = _sens(150.0, gt=0.04, r=0.085)
    assert block["shifts"][0]["status"] == "skipped"
    assert block["shifts"][0]["skip_reason"] == "below_erp_spread"
    assert block["shifts"][1]["status"] != "skipped"
    assert block["shifts"][2]["status"] != "skipped"
    assert all(s["skip_reason"] is None for s in block["shifts"][1:])


def test_skipped_shift_does_not_change_other_entries():
    block = _sens(150.0, gt=0.04, r=0.085)
    assert len(block["shifts"]) == 3
    assert [s["shift_bp"] for s in block["shifts"]] == [-50, 50, 100]


# ---------------------------------------------------------------------------
# 3. Bracket-edge statuses per shift
# ---------------------------------------------------------------------------

def test_plus_100bp_goes_above_bracket_while_base_is_ok():
    # P_hi(r) = per-share at g=+60% is strictly decreasing in r. Choose the price
    # midway between P_hi(0.105) and P_hi(0.11): reachable at r=0.10 and 0.105
    # (P_hi > price) but NOT reachable at 0.11 (P_hi < price).
    price = (_ref_ps(_FCF0, _HI, _GT, 0.105, _SHARES) + _ref_ps(_FCF0, _HI, _GT, 0.11, _SHARES)) / 2.0
    assert _ref_ps(_FCF0, _HI, _GT, 0.10, _SHARES) > price
    block = _sens(price)
    assert block["base_status"] == "ok"
    s = block["shifts"]
    assert s[0]["status"] == "ok" and s[1]["status"] == "ok"
    assert s[2]["status"] == "above_bracket"
    assert s[2]["implied_growth"] is None
    assert s[2]["delta_pp"] is None
    assert s[2]["skip_reason"] is None
    assert s[1]["delta_pp"] is not None


def test_minus_50bp_goes_below_bracket_while_base_is_ok():
    # P_lo(r) = per-share at g=-20% is strictly decreasing in r, so a lower rate
    # RAISES the floor value. Price between P_lo(0.10) and P_lo(0.095):
    # base r=0.10 brackets it, r=0.095 has P_lo > price -> below_bracket.
    price = (_ref_ps(_FCF0, _LO, _GT, 0.10, _SHARES) + _ref_ps(_FCF0, _LO, _GT, 0.095, _SHARES)) / 2.0
    block = _sens(price)
    assert block["base_status"] == "ok"
    assert block["shifts"][0]["status"] == "below_bracket"
    assert block["shifts"][0]["implied_growth"] is None
    assert block["shifts"][0]["delta_pp"] is None
    assert block["shifts"][1]["status"] == "ok"
    assert block["shifts"][1]["delta_pp"] is not None


def test_base_not_ok_makes_every_delta_none():
    # Price so high that even the base is above_bracket.
    price = _ref_ps(_FCF0, _HI, _GT, _R, _SHARES) * 10
    block = _sens(price)
    assert block["base_status"] == "above_bracket"
    assert block["base_implied_growth"] is None
    assert all(s["delta_pp"] is None for s in block["shifts"])
    assert all(s["status"] == "above_bracket" for s in block["shifts"])


# ---------------------------------------------------------------------------
# Unusable inputs -> None (never raises)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(price=None), dict(price=0.0), dict(price=-5.0), dict(price=float("nan")),
    dict(fcf0=None), dict(shares=None), dict(shares=0.0), dict(shares=-1.0),
    dict(r=None), dict(gt=None), dict(r="x"), dict(gt=float("nan")),
])
def test_unusable_inputs_return_none(kwargs):
    args = dict(price=150.0, fcf0=_FCF0, gt=_GT, r=_R, shares=_SHARES)
    args.update(kwargs)
    assert _sens(args.pop("price"), **args) is None


def test_base_rate_at_or_below_terminal_growth_does_not_crash():
    # Post-clamp this cannot happen in the engine, but the pure function must not raise.
    block = _sens(150.0, gt=0.10, r=0.10)
    if block is not None:
        assert block["base_status"] == "no_data"
        assert all(s["delta_pp"] is None for s in block["shifts"])


# ---------------------------------------------------------------------------
# Engine fixtures (mirrors test_valuation_engine.py helpers)
# ---------------------------------------------------------------------------

_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity", "Depreciation",
]


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31", "fy": fy, "fp": "FY",
        "form": "10-K", "value": value, "filed": None, "start": None, "unit": "USD",
    }


def _normalized(overrides):
    annual = {c: overrides.get(c) for c in _CONCEPTS}
    return {
        "cik": 1, "entity_name": "Rate Basis Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CONCEPTS},
        "missing": [c for c in _CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CONCEPTS},
    }


def _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10):
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": base_growth, "terminal_growth": base_terminal, "discount_rate": base_discount, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


def _no_damodaran_dir():
    """A directory that does not exist: no CSV, no live FRED -> deterministic macro."""
    path = os.path.join(tempfile.gettempdir(), "sec_analyzer_no_damodaran_rate_basis")
    assert not os.path.exists(path)
    return path


# Cyclical fixture with a normalized variant (same as
# test_run_valuation_cyclical_sector_produces_normalized_variant):
#   constant FCF margin 0.10 in FY2021-23, latest revenue 1200
#   -> normalized_fcf0 = 0.10 * 1200 = 120 ; raw ttm fcf0 = 200 (metrics["fcf"]).
_CYC_NORM_FCF0, _CYC_RAW_FCF0 = 120.0, 200.0


def _cyclical_case(revenue_cagr_5y=0.05, revenue_cagr_3y=None):
    normalized = _normalized({
        "Revenue": [_rec(2021, 1000.0), _rec(2022, 1100.0), _rec(2023, 1200.0)],
        "OperatingCashFlow": [_rec(2021, 150.0), _rec(2022, 176.0), _rec(2023, 300.0)],
        "CapEx": [_rec(2021, 50.0), _rec(2022, 66.0), _rec(2023, 180.0)],
    })
    ratios = [{"fy": 2023, "fcf": 200.0}, {"fy": 2022, "fcf": 110.0}, {"fy": 2021, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 200.0, "net_debt": 0.0}
    if revenue_cagr_5y is not None:
        metrics["revenue_cagr_5y"] = revenue_cagr_5y
    if revenue_cagr_3y is not None:
        metrics["revenue_cagr_3y"] = revenue_cagr_3y
    return normalized, ratios, metrics


def _run_cyclical(price, **kw):
    normalized, ratios, metrics = _cyclical_case(**kw)
    return run_valuation(
        normalized, ratios, metrics, price=price, price_df=None,
        assumptions=_assumptions(), sector_type="cyclical", damodaran_dir=_no_damodaran_dir(),
    )


# ---------------------------------------------------------------------------
# 4. Engine: cyclical with a normalized variant
# ---------------------------------------------------------------------------

# Price 260 ~= per-share of the normalized FCF (120) at g=10%, r=10%, g_t=3%
# (216.7659 * 1.2 = 260.12) -> normalized implied growth ~= 0.10. The same price
# on the raw fcf0=200 needs far less growth (raw implied < 0.02), so the two
# solves straddle the +-3pp reference band around a 5% revenue CAGR:
#   normalized 0.10 > 0.05 + 0.03 -> expensive ; raw < 0.05 - 0.03 -> cheap.
_CYC_PRICE = 260.0


def test_engine_cyclical_normalized_basis_drives_headline():
    result = _run_cyclical(_CYC_PRICE)
    assert result["fcf0"] == pytest.approx(_CYC_RAW_FCF0)
    assert result["dcf"]["normalized_variant"] is not None
    rd = result["reverse_dcf"]

    exp_norm, st_norm = _ref_solve(_CYC_PRICE, _CYC_NORM_FCF0, 0.03, 0.10, 10.0)
    exp_raw, st_raw = _ref_solve(_CYC_PRICE, _CYC_RAW_FCF0, 0.03, 0.10, 10.0)
    assert exp_norm == pytest.approx(0.10, abs=2e-3)  # sanity of the fixture
    assert exp_raw < 0.02 and exp_norm - exp_raw > 0.06  # signals must differ

    assert rd["fcf0_basis"] == "normalized"
    assert rd["implied_growth"] == pytest.approx(exp_norm, abs=_TOL)
    assert rd["bracket_status"] == st_norm == "ok"
    assert rd["implied_growth_raw"] == pytest.approx(exp_raw, abs=_TOL)
    assert rd["bracket_status_raw"] == st_raw == "ok"
    assert rd["implied_growth"] != rd["implied_growth_raw"]

    # Reference growth = realized REVENUE CAGR, not FCF CAGR.
    assert rd["realized_label"].startswith("revenue")
    assert rd["realized_label"] == "revenue 5y"
    assert rd["realized_cagr_5y"] == pytest.approx(0.05)


def test_engine_cyclical_reference_falls_back_to_revenue_3y_then_none():
    rd = _run_cyclical(_CYC_PRICE, revenue_cagr_5y=None, revenue_cagr_3y=0.06)["reverse_dcf"]
    assert rd["fcf0_basis"] == "normalized"
    assert rd["realized_label"] == "revenue 3y"
    assert rd["realized_cagr_5y"] == pytest.approx(0.06)
    rd = _run_cyclical(_CYC_PRICE, revenue_cagr_5y=None, revenue_cagr_3y=None)["reverse_dcf"]
    assert rd["fcf0_basis"] == "normalized"
    assert rd["realized_cagr_5y"] is None
    assert rd["realized_label"] is None


def test_engine_cyclical_triangulation_signal_uses_normalized_value():
    result = _run_cyclical(_CYC_PRICE)
    # normalized implied ~0.10 vs revenue CAGR 0.05 -> +5pp > 3pp margin -> expensive.
    # (The raw solve ~<0.02 would have said cheap.)
    assert result["triangulation"]["signals"]["reverse_dcf"] == SIGNAL_EXPENSIVE
    assert result["triangulation"]["signals"]["reverse_dcf"] != SIGNAL_CHEAP


def test_engine_cyclical_rate_sensitivity_on_normalized_fcf():
    result = _run_cyclical(_CYC_PRICE)
    rd = result["reverse_dcf"]
    rs = rd["rate_sensitivity"]
    assert rs is not None
    assert rs["basis"] == "normalized"
    assert rs["base_discount_rate"] == 0.1
    assert rs["terminal_growth"] == 0.03
    assert rs["base_status"] == "ok"
    assert rs["base_implied_growth"] == rd["implied_growth"]  # reused, not re-solved
    assert [s["discount_rate"] for s in rs["shifts"]] == [0.095, 0.105, 0.11]
    for s in rs["shifts"]:
        exp, st = _ref_solve(_CYC_PRICE, _CYC_NORM_FCF0, 0.03, s["discount_rate"], 10.0)
        assert s["status"] == st == "ok"
        assert s["implied_growth"] == pytest.approx(exp, abs=_TOL)
    grown = [s["implied_growth"] for s in rs["shifts"]]
    assert grown == sorted(grown)


def test_engine_cyclical_note_explains_basis():
    notes = _run_cyclical(_CYC_PRICE)["notes"]
    matching = [n for n in notes if n.startswith("Cyclical: the reverse DCF solves on mid-cycle normalized FCF")]
    assert len(matching) == 1
    note = matching[0]
    assert "instead of the latest single-year FCF" in note
    assert "normalized vs" in note and "on the raw figure" in note
    assert "None" not in note and "nan" not in note.lower()


def test_engine_cyclical_without_normalized_variant_stays_raw():
    # Every FY has a non-positive margin -> normalized_variant is None (existing
    # fixture in test_valuation_engine.py) -> basis stays "raw".
    normalized = _normalized({
        "Revenue": [_rec(2022, 1000.0), _rec(2023, 1000.0), _rec(2024, 1000.0)],
        "OperatingCashFlow": [_rec(2022, 50.0), _rec(2023, 80.0), _rec(2024, 100.0)],
        "CapEx": [_rec(2022, 100.0), _rec(2023, 150.0), _rec(2024, 200.0)],
    })
    ratios = [{"fy": 2024, "fcf": 50.0}]
    metrics = {"shares": 10.0, "latest_fy": 2024, "fcf": 50.0, "net_debt": 0.0, "revenue_cagr_5y": 0.05}
    result = run_valuation(
        normalized, ratios, metrics, price=80.0, price_df=None,
        assumptions=_assumptions(), sector_type="cyclical", damodaran_dir=_no_damodaran_dir(),
    )
    assert result["dcf"]["normalized_variant"] is None
    rd = result["reverse_dcf"]
    assert rd["fcf0_basis"] == "raw"
    assert rd["implied_growth_raw"] is None
    assert rd["bracket_status_raw"] is None
    assert not (rd["realized_label"] or "").startswith("revenue")
    assert rd["rate_sensitivity"]["basis"] == "raw"
    exp, st = _ref_solve(80.0, 50.0, 0.03, 0.10, 10.0)
    assert st == "ok"
    assert rd["implied_growth"] == pytest.approx(exp, abs=_TOL)
    assert not any(n.startswith("Cyclical: the reverse DCF solves") for n in result["notes"])


def test_engine_empty_inputs_have_new_keys_and_do_not_crash():
    result = run_valuation({}, [], {}, price=None, price_df=None, assumptions={}, sector_type="mature",
                           damodaran_dir=_no_damodaran_dir())
    rd = result["reverse_dcf"]
    assert rd["fcf0_basis"] == "raw"
    assert rd["implied_growth_raw"] is None
    assert rd["bracket_status_raw"] is None
    assert rd["rate_sensitivity"] is None


def test_engine_early_return_shape_has_new_keys():
    from sec_analyzer.valuation import engine
    empty = engine._empty_valuation("mature", None)
    rd = empty["reverse_dcf"]
    assert rd["fcf0_basis"] == "raw"
    assert rd["implied_growth_raw"] is None
    assert rd["bracket_status_raw"] is None
    assert rd["rate_sensitivity"] is None


# ---------------------------------------------------------------------------
# 5. Non-cyclical byte-for-byte invariance against a golden from pre-WP12 code
# ---------------------------------------------------------------------------

_NEW_KEYS = ("fcf0_basis", "implied_growth_raw", "bracket_status_raw", "rate_sensitivity")
_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "rate_basis_golden.json"

_HYPER_RATIOS = [
    {"fy": 2023, "gross_margin": 0.60, "fcf": -50.0},
    {"fy": 2022, "fcf": 100.0},
    {"fy": 2021, "fcf": 90.0},
]
_HYPER_METRICS = {
    "shares": 100.0, "latest_fy": 2023, "fcf": -50.0, "net_debt": 0.0,
    "revenue_cagr_5y": 0.50, "rnd_revenue": 0.0, "sbc_revenue": 0.0, "shares_yoy": None,
}


def _golden_inputs(name):
    """Return (normalized, ratios, metrics, price, sector_type) for a golden case."""
    if name == "mature_profitable":
        normalized = _normalized({
            "Revenue": [_rec(2021, 1000.0), _rec(2022, 1040.0), _rec(2023, 1080.0)],
            "NetIncome": [_rec(2021, 100.0), _rec(2022, 105.0), _rec(2023, 110.0)],
            "OperatingCashFlow": [_rec(2021, 160.0), _rec(2022, 165.0), _rec(2023, 170.0)],
            "CapEx": [_rec(2021, 50.0), _rec(2022, 52.0), _rec(2023, 54.0)],
        })
        ratios = [{"fy": 2023, "fcf": 116.0}, {"fy": 2022, "fcf": 113.0}, {"fy": 2021, "fcf": 110.0}]
        metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 116.0, "net_debt": 0.0, "revenue_cagr_5y": 0.04}
        return normalized, ratios, metrics, 150.0, "mature"
    if name == "growth_unprofitable_nonhyper":
        normalized = _normalized({
            "Revenue": [_rec(2021, 800.0), _rec(2022, 900.0), _rec(2023, 1000.0)],
            "NetIncome": [_rec(2021, -30.0), _rec(2022, -20.0), _rec(2023, -10.0)],
            "OperatingCashFlow": [_rec(2021, 30.0), _rec(2022, 45.0), _rec(2023, 60.0)],
            "CapEx": [_rec(2021, 20.0), _rec(2022, 25.0), _rec(2023, 30.0)],
        })
        ratios = [{"fy": 2023, "fcf": 30.0}, {"fy": 2022, "fcf": 20.0}, {"fy": 2021, "fcf": 10.0}]
        metrics = {"shares": 20.0, "latest_fy": 2023, "fcf": 30.0, "net_debt": 0.0, "revenue_cagr_5y": 0.11}
        return normalized, ratios, metrics, 40.0, "growth_unprofitable"
    if name == "cyclical_no_variant":
        normalized = _normalized({
            "Revenue": [_rec(2022, 1000.0), _rec(2023, 1000.0), _rec(2024, 1000.0)],
            "OperatingCashFlow": [_rec(2022, 50.0), _rec(2023, 80.0), _rec(2024, 100.0)],
            "CapEx": [_rec(2022, 100.0), _rec(2023, 150.0), _rec(2024, 200.0)],
        })
        ratios = [{"fy": 2024, "fcf": 50.0}]
        metrics = {"shares": 10.0, "latest_fy": 2024, "fcf": 50.0, "net_debt": 0.0}
        return normalized, ratios, metrics, 20.0, "cyclical"
    if name == "hyper_revenue_mode":
        return _normalized({"Revenue": [_rec(2023, 1000.0)]}), _HYPER_RATIOS, _HYPER_METRICS, 50.0, "growth_unprofitable"
    raise KeyError(name)


_GOLDEN_CASES = {
    # name: (expected fcf0_basis, expect rate_sensitivity is None)
    "mature_profitable": ("raw", False),
    "growth_unprofitable_nonhyper": ("raw", False),
    "cyclical_no_variant": ("raw", False),
    "hyper_revenue_mode": ("revenue", True),
}


def _run_golden_case(name):
    normalized, ratios, metrics, price, sector = _golden_inputs(name)
    return run_valuation(
        normalized, ratios, metrics, price=price, price_df=None,
        assumptions=_assumptions(), sector_type=sector, damodaran_dir=_no_damodaran_dir(),
    )


def _dump(obj):
    return json.dumps(obj, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_GOLDEN_CASES))
def test_non_cyclical_output_is_byte_identical_to_pre_wp12_golden(name):
    golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))[name]
    result = _run_golden_case(name)
    basis, sens_none = _GOLDEN_CASES[name]

    rd = result["reverse_dcf"]
    assert rd["fcf0_basis"] == basis
    assert rd["implied_growth_raw"] is None
    assert rd["bracket_status_raw"] is None
    if sens_none:
        assert rd["rate_sensitivity"] is None
    else:
        assert rd["rate_sensitivity"] is not None
        assert rd["rate_sensitivity"]["basis"] == "raw"

    stripped = json.loads(_dump(result))
    for key in _NEW_KEYS:
        assert key in stripped["reverse_dcf"]
        del stripped["reverse_dcf"][key]
    assert _dump(stripped) == _dump(golden)


def test_golden_cases_exercise_intended_headline_modes():
    # Guards the fixtures themselves against silently drifting into another mode.
    mature = _run_golden_case("mature_profitable")
    assert not mature["hyper_growth"] and not mature["mature_revenue_headline"]
    assert not mature["midgrowth_revenue_headline"]
    assert mature["reverse_dcf"]["implied_growth"] is not None
    hyper = _run_golden_case("hyper_revenue_mode")
    assert hyper["hyper_growth"] is True


def test_raw_basis_rate_sensitivity_matches_independent_solve():
    # Mature filer: fcf0 = metrics["fcf"] = 116 (ttm), r=0.10, g_t=0.03, 10 shares.
    result = _run_golden_case("mature_profitable")
    rs = result["reverse_dcf"]["rate_sensitivity"]
    fcf0 = result["fcf0"]
    for s in rs["shifts"]:
        exp, st = _ref_solve(150.0, fcf0, 0.03, s["discount_rate"], 10.0)
        assert s["status"] == st
        if st == "ok":
            assert s["implied_growth"] == pytest.approx(exp, abs=_TOL)


# ---------------------------------------------------------------------------
# 6. CLI rendering never crashes (SPEC Sec.28d)
# ---------------------------------------------------------------------------

def _block(shifts, terminal_growth=0.04):
    return {"basis": "raw", "base_discount_rate": 0.095, "terminal_growth": terminal_growth,
            "base_implied_growth": 0.2310, "base_status": "ok", "shifts": shifts}


def _entry(shift_bp, growth, status="ok", reason=None):
    return {"shift_bp": shift_bp, "discount_rate": 0.1, "implied_growth": growth,
            "delta_pp": None, "status": status, "skip_reason": reason}


def test_cli_rate_sens_line_matches_spec_example():
    from sec_analyzer.cli import _rate_sensitivity_line
    val = {"reverse_dcf": {"rate_sensitivity": _block(
        [_entry(-50, 0.211), _entry(50, 0.248), _entry(100, 0.265)])}}
    assert _rate_sensitivity_line(val) == (
        "Rate sens.: r −50bp 21.1% · +50bp 24.8% · +100bp 26.5% (g_t fixed 4.0%)"
    )


def test_cli_rate_sens_line_skipped_and_bracket_forms():
    from sec_analyzer.cli import _rate_sensitivity_line
    for reason, label in (("below_rate_floor", "floor"), ("below_erp_spread", "ERP spread"),
                          ("rate_at_or_below_terminal_growth", "r ≤ g_t")):
        val = {"reverse_dcf": {"rate_sensitivity": _block(
            [_entry(-50, None, "skipped", reason), _entry(50, 0.2), _entry(100, 0.3)])}}
        assert _rate_sensitivity_line(val).startswith(f"Rate sens.: r −50bp — ({label}) · +50bp 20.0%")
    val = {"reverse_dcf": {"rate_sensitivity": _block(
        [_entry(-50, None, "below_bracket"), _entry(50, 0.2), _entry(100, None, "above_bracket")])}}
    line = _rate_sensitivity_line(val)
    assert "−50bp <−20%" in line and "+100bp >60%" in line


@pytest.mark.parametrize("val", [
    {}, {"reverse_dcf": None}, {"reverse_dcf": {}}, {"reverse_dcf": {"rate_sensitivity": None}},
    {"reverse_dcf": {"rate_sensitivity": {}}}, {"reverse_dcf": {"rate_sensitivity": {"shifts": []}}},
    {"reverse_dcf": {"rate_sensitivity": "garbage"}},
])
def test_cli_rate_sens_line_omitted_when_missing(val):
    from sec_analyzer.cli import _rate_sensitivity_line
    assert _rate_sensitivity_line(val) is None


def test_cli_rate_sens_line_tolerates_malformed_entries_and_missing_numbers():
    from sec_analyzer.cli import _rate_sensitivity_line
    val = {"reverse_dcf": {"rate_sensitivity": {"terminal_growth": None, "shifts": [
        {}, {"shift_bp": 50}, {"shift_bp": "x", "status": "ok", "implied_growth": None},
    ]}}}
    line = _rate_sensitivity_line(val)
    assert line is not None
    assert "None" not in line and "nan" not in line.lower()
    assert "—" in line


def test_cli_reverse_dcf_line_raw_suffix_renders_bracket_status():
    # MU-style case: the raw solve is out of bracket, so implied_growth_raw is
    # None -- the suffix must say ">60%"/"<−20%", not "—".
    from sec_analyzer.cli import _reverse_dcf_line
    base = {"implied_growth": 0.10, "realized_cagr_5y": 0.05, "realized_label": "revenue 5y",
            "fcf0_basis": "normalized", "implied_growth_raw": None}
    line = _reverse_dcf_line({}, {"reverse_dcf": {**base, "bracket_status_raw": "above_bracket"}})
    assert line.endswith("[normalized FCF; raw: >60%]")
    line = _reverse_dcf_line({}, {"reverse_dcf": {**base, "bracket_status_raw": "below_bracket"}})
    assert line.endswith("[normalized FCF; raw: <−20%]")
    # Headline also out of bracket (no commentary): bracket text, suffix kept.
    line = _reverse_dcf_line({}, {"reverse_dcf": {**base, "implied_growth": None, "bracket_status": "above_bracket",
                                                  "bracket_status_raw": "above_bracket"}})
    assert "implies a 10y >60% CAGR" in line and line.endswith("[normalized FCF; raw: >60%]")


def test_cli_reverse_dcf_line_normalized_suffix_and_missing_raw():
    from sec_analyzer.cli import _reverse_dcf_line
    base = {"implied_growth": 0.10, "realized_cagr_5y": 0.05, "realized_label": "revenue 5y",
            "fcf0_basis": "normalized"}
    line = _reverse_dcf_line({}, {"reverse_dcf": {**base, "implied_growth_raw": 0.01}})
    assert line.endswith("[normalized FCF; raw: 1.0%]")
    line = _reverse_dcf_line({}, {"reverse_dcf": {**base, "implied_growth_raw": None}})
    assert line.endswith("[normalized FCF; raw: —]")
    assert "None" not in line and "nan" not in line.lower()
    # raw basis: no suffix
    line = _reverse_dcf_line({}, {"reverse_dcf": {**base, "fcf0_basis": "raw"}})
    assert "[normalized FCF" not in line
    # pre-WP12 shaped dict (keys absent): no suffix, no crash
    line = _reverse_dcf_line({}, {"reverse_dcf": {"implied_growth": 0.1, "realized_cagr_5y": None,
                                                    "realized_label": None}})
    assert "None" not in line and "nan" not in line.lower()


# ---------------------------------------------------------------------------
# SPEC Sec.10/Sec.28a: normalized reverse-DCF confidence ceiling
# ---------------------------------------------------------------------------

def _all_fair_triangulation(**flags):
    from sec_analyzer.valuation.triangulate import triangulate
    # price inside the band -> DCF fair; implied == reference -> rDCF fair;
    # P/E at the 50th percentile -> multiples fair: three-way agreement.
    return triangulate(100.0, {"lo": 90.0, "hi": 110.0}, 0.10, 0.10, 0.10, 50.0, None, None, "cyclical", **flags)


def test_triangulate_normalized_reverse_dcf_caps_high_to_medium():
    from sec_analyzer.valuation.triangulate import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
    assert _all_fair_triangulation()["confidence"] == CONFIDENCE_HIGH
    capped = _all_fair_triangulation(normalized_reverse_dcf=True)
    assert capped["confidence"] == CONFIDENCE_MEDIUM
    assert "normalized FCF" in capped["rationale"]["confidence"]


def test_triangulate_headline_cap_message_takes_precedence_over_normalized_flag():
    from sec_analyzer.valuation.triangulate import CONFIDENCE_MEDIUM
    both = _all_fair_triangulation(normalized_reverse_dcf=True, cyclical_fcfe_headline=True)
    assert both["confidence"] == CONFIDENCE_MEDIUM
    assert "FCFE anchor" in both["rationale"]["confidence"]
    assert "normalized FCF" not in both["rationale"]["confidence"]


def test_engine_passes_normalized_flag_to_triangulate(monkeypatch):
    from sec_analyzer.valuation import engine, triangulate as tri_mod
    seen = {}
    real = tri_mod.triangulate

    def spy(*args, **kwargs):
        seen["flag"] = kwargs.get("normalized_reverse_dcf")
        return real(*args, **kwargs)

    monkeypatch.setattr(engine.triangulate, "triangulate", spy)
    result = _run_cyclical(260.0)
    assert result["reverse_dcf"]["fcf0_basis"] == "normalized"
    assert seen["flag"] is True
