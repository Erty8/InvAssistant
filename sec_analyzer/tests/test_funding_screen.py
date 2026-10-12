"""Hand-verified, fully offline tests for the self-funded-growth screen
(``sec_analyzer.screener.funding``, FUNDING_SPEC Sec.5-8) and its CLI wiring.

Fixtures are tiny synthetic companyfacts documents. Network / pipeline
boundaries (``resolve_cik``, ``get_company_facts``, ``SecHttpClient``,
``run_headless_pipeline``, the FRED fetch) are monkeypatched; the code under
test is not mocked.

The standard company ("GOOD", Dec fiscal year, latest filing = 9M to
2026-09-30) and its hand-derived numbers:

    Revenue   FY25 100, 9M26 80, 9M25 70, FY24 90, 9M24 60
              TTM 100+80-70 = 110; prior TTM 90+70-60 = 100; growth 10%
    CFO       FY25 50, 9M26 40, 9M25 30            TTM 50+40-30 = 60
    CapEx     FY25 10, 9M26 8,  9M25 6             TTM 10+8-6   = 12
    FCF       60 - 12 = 48            margin 48/110 = 0.436364
    FY margin FY25 (50-10)/100 = 0.40; FY24 (40-8)/90 = 0.355556
    SBC       FY25 10, 9M26 9, 9M25 7              TTM 12
    FCF-SBC   48 - 12 = 36            margin 36/110 = 0.327273
    EBIT      30, 25, 20                           TTM 35
    D&A       8, 6, 5                              TTM 9   -> EBITDA 44
    Debt 30 (noncurrent), cash 40 + ST inv 20     -> net debt 30-60 = -30
    ND/EBITDA -30/44 = -0.681818 (net cash -> leverage passes)
    Interest  3, 2, 2                              TTM 3   -> cover 35/3 = 11.667
    Rule of 40  100*0.10 + 100*0.436364 = 53.6364
    Dilution  Q3-26 101 / Q3-25 100 - 1 = 1%
    Cover shares 60 + 40 (two classes, one accn) = 100
"""

import json
import re
import sys

import pytest

from sec_analyzer import cli
from sec_analyzer.screener import funding
from sec_analyzer.valuation import reverse_dcf

E = "2026-09-30"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _row(start, end, val, form="10-K", filed="2026-02-15", accn="A"):
    r = {"end": end, "val": val, "form": form, "filed": filed, "accn": accn}
    if start is not None:
        r["start"] = start
    return r


def _inst(end, val, form="10-Q", filed="2026-10-25"):
    return _row(None, end, val, form, filed)


def _dec_flow(fy25, y26=None, p25=None, fy24=None, p24=None):
    rows = [_row("2025-01-01", "2025-12-31", fy25, "10-K", "2026-02-15")]
    if fy24 is not None:
        rows.append(_row("2024-01-01", "2024-12-31", fy24, "10-K", "2025-02-15"))
    if y26 is not None:
        rows.append(_row("2026-01-01", "2026-09-30", y26, "10-Q", "2026-10-25"))
    if p25 is not None:
        rows.append(_row("2025-01-01", "2025-09-30", p25, "10-Q", "2025-10-25"))
    if p24 is not None:
        rows.append(_row("2024-01-01", "2024-09-30", p24, "10-Q", "2024-10-25"))
    return rows


def _shares_q(cur, prior):
    return ("shares", [
        _row("2026-07-01", E, cur, "10-Q"),
        _row("2025-07-01", "2025-09-30", prior, "10-Q"),
    ])


def _cover(classes, accn="Q3"):
    return ("shares", [
        _row(None, "2026-10-20", v, "10-Q", "2026-10-25", accn) for v in classes
    ])


def _doc(us_gaap=None, dei=None):
    def ns(d):
        out = {}
        for tag, v in (d or {}).items():
            unit, rows = v if isinstance(v, tuple) else ("USD", v)
            out[tag] = {"units": {unit: rows}}
        return out
    return {"facts": {"us-gaap": ns(us_gaap), "dei": ns(dei)}}


def _good_us_gaap():
    return {
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            _dec_flow(100, 80, 70, fy24=90, p24=60),
        "NetCashProvidedByUsedInOperatingActivities":
            _dec_flow(50, 40, 30, fy24=40, p24=25),
        "PaymentsToAcquirePropertyPlantAndEquipment":
            _dec_flow(10, 8, 6, fy24=8),
        "ShareBasedCompensation": _dec_flow(10, 9, 7),
        "OperatingIncomeLoss": _dec_flow(30, 25, 20),
        "DepreciationDepletionAndAmortization": _dec_flow(8, 6, 5),
        "InterestExpense": _dec_flow(3, 2, 2),
        "LongTermDebtNoncurrent": [_inst(E, 30)],
        "CashAndCashEquivalentsAtCarryingValue": [_inst(E, 40)],
        "ShortTermInvestments": [_inst(E, 20)],
        "WeightedAverageNumberOfDilutedSharesOutstanding": _shares_q(101, 100),
    }


def good_facts(over=None, drop=(), dei=None):
    us = _good_us_gaap()
    us.update(over or {})
    for t in drop:
        us.pop(t, None)
    return _doc(us, dei if dei is not None else
                {"EntityCommonStockSharesOutstanding": _cover([60, 40])})


def fy_only_facts(rev, rev_prev, cfo, capex=None, extra=None):
    """FY-2025-only company (TTM end == FY end, no YTD data)."""
    us = {
        "Revenues": [
            _row("2025-01-01", "2025-12-31", rev),
            _row("2024-01-01", "2024-12-31", rev_prev),
        ] if rev_prev is not None else [_row("2025-01-01", "2025-12-31", rev)],
        "NetCashProvidedByUsedInOperatingActivities":
            [_row("2025-01-01", "2025-12-31", cfo)],
    }
    if capex is not None:
        us["PaymentsToAcquirePropertyPlantAndEquipment"] = \
            [_row("2025-01-01", "2025-12-31", capex)]
    us.update(extra or {})
    return _doc(us)


def screen(facts, ticker="T"):
    return funding.screen_facts(ticker, "Test Co", "0000000001", facts)


# ---------------------------------------------------------------------------
# Standard company: every metric by hand
# ---------------------------------------------------------------------------

def test_good_company_metrics_hand_verified():
    r = screen(good_facts())
    assert r["status"] == "ok"
    assert r["ttm_end"] == E and r["fy_end"] == "2025-12-31" and r["is_ttm"] is True
    assert r["revenue_tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert r["revenue"] == pytest.approx(110)
    assert r["rev_growth"] == pytest.approx(0.10)
    assert r["cfo"] == pytest.approx(60)
    assert r["capex"] == pytest.approx(12)
    assert r["fcf"] == pytest.approx(48)
    assert r["fcf_margin"] == pytest.approx(48 / 110)
    assert r["fcf_margin_trend"] == pytest.approx([32 / 90, 0.40, 48 / 110])
    assert r["sbc"] == pytest.approx(12)
    assert r["sbc_rev"] == pytest.approx(12 / 110)
    assert r["fcf_sbc_margin"] == pytest.approx(36 / 110)
    assert r["ebit"] == pytest.approx(35) and r["ebitda"] == pytest.approx(44)
    assert r["debt"] == pytest.approx(30)
    assert r["liquidity"] == pytest.approx(60)
    assert r["net_debt"] == pytest.approx(-30)
    assert r["net_debt_ebitda"] == pytest.approx(-30 / 44)
    assert r["interest_cov"] == pytest.approx(35 / 3)
    assert r["rule_of_40"] == pytest.approx(10 + 4800 / 110)
    assert r["dilution"] == pytest.approx(0.01)
    assert r["cover_shares"] == pytest.approx(100)
    assert r["cover_shares_as_of"] == "2026-10-20"
    assert r["notes"] == []
    # Interest coverage is n/a for a net-cash company even though computed.
    assert r["tests"] == {
        "fcf_margin": "pass", "leverage": "pass", "interest_cov": "n/a",
        "rule_of_40": "pass", "dilution": "pass",
    }
    assert r["fails"] == [] and r["group"] == "A"


# 4 (screen level): the trend omits the TTM point when TTM end == FY end.
def test_trend_has_two_points_when_ttm_end_is_fy_end():
    # FY25: rev 200, CFO 50, capex 10 -> (50-10)/200 = 0.20
    # FY24: rev 160, CFO n/a in prior year -> fy_margin(prev) uses annual CFO
    r = screen(fy_only_facts(200, 160, 50, 10))
    assert r["is_ttm"] is False
    assert len(r["fcf_margin_trend"]) == 2
    assert r["fcf_margin_trend"][1] == pytest.approx(0.20)
    assert r["fcf_margin_trend"][0] is None    # no FY24 CFO


# ---------------------------------------------------------------------------
# 5. Revenue tag alignment
# ---------------------------------------------------------------------------

# Revenue tags that only have FY data end on 2025-12-31 != CFO end
# 2026-09-30; only the IncludingAssessedTax tag has a TTM ending on E, so it
# is chosen: TTM = 100 + 80 - 70 = 110.
def test_revenue_tag_aligned_to_cfo_end_picks_assessed_tax_tag():
    over = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": _dec_flow(1000),
        "Revenues": _dec_flow(1000),
        "RevenueFromContractWithCustomerIncludingAssessedTax":
            _dec_flow(100, 80, 70, fy24=90, p24=60),
    }
    r = screen(good_facts(over))
    assert r["status"] == "ok"
    assert r["revenue_tag"] == "RevenueFromContractWithCustomerIncludingAssessedTax"
    assert r["revenue"] == pytest.approx(110)
    assert r["fcf_margin"] == pytest.approx(48 / 110)


def test_revenue_with_no_tag_on_cfo_end_is_error_row():
    r = screen(good_facts({
        "RevenueFromContractWithCustomerExcludingAssessedTax": _dec_flow(100, fy24=90),
    }), ticker="NOREV")
    assert set(r) == {"ticker", "status", "error"}
    assert r["ticker"] == "NOREV" and r["status"] == "error"
    assert r["error"]


def test_missing_revenue_entirely_is_error_row():
    r = screen(good_facts(drop=["RevenueFromContractWithCustomerExcludingAssessedTax"]))
    assert set(r) == {"ticker", "status", "error"} and r["status"] == "error"


# ---------------------------------------------------------------------------
# 4 (screen level). Missing pieces: notes and n/a semantics
# ---------------------------------------------------------------------------

# Missing CapEx counts as 0: FCF = CFO = 60, margin 60/110; note added.
def test_missing_capex_treated_as_zero_with_note():
    r = screen(good_facts(drop=["PaymentsToAcquirePropertyPlantAndEquipment"]))
    assert r["capex"] == 0.0
    assert r["fcf"] == pytest.approx(60)
    assert r["fcf_margin"] == pytest.approx(60 / 110)
    assert any("CapEx" in n for n in r["notes"])


# Missing SBC -> fcf_sbc_margin None (not a silently un-adjusted number).
def test_missing_sbc_gives_none_margin_and_note():
    r = screen(good_facts(drop=["ShareBasedCompensation"]))
    assert r["sbc"] is None and r["sbc_rev"] is None and r["fcf_sbc_margin"] is None
    assert any("SBC" in n for n in r["notes"])


# Missing D&A counts as 0: EBITDA = EBIT = 35; note.
def test_missing_da_treated_as_zero_with_note():
    r = screen(good_facts(drop=["DepreciationDepletionAndAmortization"]))
    assert r["ebitda"] == pytest.approx(35)
    assert r["net_debt_ebitda"] == pytest.approx(-30 / 35)
    assert any("D&A" in n for n in r["notes"])


# Missing operating income: EBITDA, ND/EBITDA and coverage are None.
def test_missing_operating_income_nulls_ebitda_and_coverage():
    r = screen(good_facts(drop=["OperatingIncomeLoss"]))
    assert r["ebit"] is None and r["ebitda"] is None
    assert r["net_debt_ebitda"] is None and r["interest_cov"] is None
    # net cash: leverage still passes, coverage n/a
    assert r["tests"]["leverage"] == "pass"
    assert r["tests"]["interest_cov"] == "n/a"


# Missing dilution data -> n/a (not a fail) and a note.
def test_missing_dilution_is_na_with_note():
    r = screen(good_facts(drop=["WeightedAverageNumberOfDilutedSharesOutstanding"]))
    assert r["dilution"] is None
    assert r["tests"]["dilution"] == "n/a"
    assert "dilution" not in r["fails"]
    assert any("dilution" in n for n in r["notes"])


# Missing revenue history -> rev_growth None -> rule_of_40 None -> core FAIL.
# Revenue FY25 only: growth is None; fcf margin = (40-10)/200 = 15% passes.
def test_missing_growth_fails_rule_of_40_core_test():
    r = screen(fy_only_facts(200, None, 40, 10))
    assert r["rev_growth"] is None and r["rule_of_40"] is None
    assert r["tests"]["rule_of_40"] == "fail"
    assert r["tests"]["fcf_margin"] == "pass"


# Net borrower with no interest data: coverage n/a + note.
# Debt 200, cash 40 + STI 20 -> net debt 140; EBITDA 44 -> 3.18x -> leverage
# fails. No interest -> coverage n/a (not fail).
def test_net_borrower_without_interest_is_na_coverage():
    r = screen(good_facts({"LongTermDebtNoncurrent": [_inst(E, 200)]},
                          drop=["InterestExpense"]))
    assert r["net_debt"] == pytest.approx(140)
    assert r["net_debt_ebitda"] == pytest.approx(140 / 44)
    assert r["tests"]["leverage"] == "fail"
    assert r["interest_cov"] is None and r["tests"]["interest_cov"] == "n/a"
    assert any("coverage" in n for n in r["notes"])
    assert r["fails"] == ["leverage"] and r["group"] == "B"


# Net borrower with interest 5 (TTM: 5+4-4): coverage 35/5 = 7.0 < 8 -> fail.
def test_net_borrower_low_coverage_fails():
    r = screen(good_facts({
        "LongTermDebtNoncurrent": [_inst(E, 200)],
        "InterestExpense": _dec_flow(5, 4, 4),
    }))
    assert r["interest_cov"] == pytest.approx(7.0)
    assert r["tests"]["interest_cov"] == "fail"
    assert r["fails"] == ["leverage", "interest_cov"] and r["group"] == "C"


# Interest reported only for the fiscal year (4): FY value used with a note;
# coverage 35/4 = 8.75 -> pass (net borrower, debt 200).
def test_interest_fy_fallback_used_with_note():
    r = screen(good_facts({
        "LongTermDebtNoncurrent": [_inst(E, 200)],
        "InterestExpense": _dec_flow(4),
    }))
    assert r["interest_cov"] == pytest.approx(35 / 4)
    assert r["tests"]["interest_cov"] == "pass"
    assert any("interest" in n for n in r["notes"])


# EBITDA <= 0 for a net borrower: leverage FAILS (EBIT -20, D&A 9 -> -11).
def test_negative_ebitda_net_borrower_fails_leverage():
    r = screen(good_facts({
        "LongTermDebtNoncurrent": [_inst(E, 200)],
        "OperatingIncomeLoss": _dec_flow(-30, -25, -20),  # TTM -35
    }))
    assert r["ebitda"] == pytest.approx(-35 + 9)
    assert r["net_debt_ebitda"] is None
    assert r["tests"]["leverage"] == "fail"


# Dilution quarterly primary; annual fallback when the quarters are absent.
def test_dilution_quarterly_primary_and_annual_fallback_in_screen():
    r = screen(good_facts({
        "WeightedAverageNumberOfDilutedSharesOutstanding": ("shares", [
            _row("2026-07-01", E, 105, "10-Q"),
            _row("2025-07-01", "2025-09-30", 100, "10-Q"),
            _row("2025-01-01", "2025-12-31", 150, "10-K"),
            _row("2024-01-01", "2024-12-31", 100, "10-K"),
        ]),
    }))
    assert r["dilution"] == pytest.approx(0.05)       # quarterly, not 50%
    assert r["tests"]["dilution"] == "fail" and r["group"] == "B"

    r = screen(good_facts({
        "WeightedAverageNumberOfDilutedSharesOutstanding": ("shares", [
            _row("2025-01-01", "2025-12-31", 102, "10-K"),
            _row("2024-01-01", "2024-12-31", 100, "10-K"),
        ]),
    }))
    assert r["dilution"] == pytest.approx(0.02)       # annual fallback
    assert r["tests"]["dilution"] == "pass"


# 6 (screen level): multi-class cover shares summed per accn.
def test_cover_shares_multiclass_in_row():
    r = screen(good_facts(dei={"EntityCommonStockSharesOutstanding": ("shares", [
        _row(None, "2026-10-20", 60, "10-Q", "2026-10-25", "Q3"),
        _row(None, "2026-10-20", 30, "10-Q", "2026-10-25", "Q3"),
        _row(None, "2026-10-20", 10, "10-Q", "2026-10-25", "Q3"),
        _row(None, "2026-07-20", 90, "10-Q", "2026-07-25", "Q2"),
    ])}))
    assert r["cover_shares"] == pytest.approx(100)


# ---------------------------------------------------------------------------
# 8. Exact threshold boundaries and group/sort rules
# ---------------------------------------------------------------------------

def _tests(**kw):
    base = {"fcf_margin": 0.5, "net_debt": -1.0, "net_debt_ebitda": -0.1,
            "interest_cov": 20.0, "rule_of_40": 60.0, "dilution": 0.0}
    base.update(kw)
    return funding.evaluate_tests(base)[0]


def test_fcf_margin_boundary_15_percent():
    assert _tests(fcf_margin=0.15)["fcf_margin"] == "pass"
    assert _tests(fcf_margin=0.1499)["fcf_margin"] == "fail"
    assert _tests(fcf_margin=None)["fcf_margin"] == "fail"


def test_leverage_boundary_one_x():
    nb = dict(net_debt=10.0)
    assert _tests(net_debt_ebitda=0.99, **nb)["leverage"] == "pass"
    assert _tests(net_debt_ebitda=1.0, **nb)["leverage"] == "fail"   # strict <
    assert _tests(net_debt_ebitda=None, **nb)["leverage"] == "fail"  # EBITDA<=0
    assert _tests(net_debt=0.0, net_debt_ebitda=None)["leverage"] == "pass"
    assert _tests(net_debt=-5.0, net_debt_ebitda=None)["leverage"] == "pass"
    assert _tests(net_debt=None)["leverage"] == "fail"


def test_interest_coverage_boundary_8x_and_na_rules():
    nb = dict(net_debt=10.0)
    assert _tests(interest_cov=8.0, **nb)["interest_cov"] == "pass"
    assert _tests(interest_cov=7.99, **nb)["interest_cov"] == "fail"
    assert _tests(interest_cov=None, **nb)["interest_cov"] == "n/a"
    # Net cash (net debt <= 0): n/a even with a terrible coverage figure.
    assert _tests(interest_cov=1.0, net_debt=-1.0)["interest_cov"] == "n/a"
    assert _tests(interest_cov=1.0, net_debt=0.0)["interest_cov"] == "n/a"


def test_rule_of_40_boundary():
    assert _tests(rule_of_40=40.0)["rule_of_40"] == "pass"
    assert _tests(rule_of_40=39.99)["rule_of_40"] == "fail"
    assert _tests(rule_of_40=None)["rule_of_40"] == "fail"


def test_dilution_boundary_3_percent():
    assert _tests(dilution=0.03)["dilution"] == "pass"
    assert _tests(dilution=0.0301)["dilution"] == "fail"
    assert _tests(dilution=None)["dilution"] == "n/a"
    assert _tests(dilution=-0.05)["dilution"] == "pass"


def test_evaluate_tests_empty_row_fails_core_and_na_optional():
    tests, fails = funding.evaluate_tests({})
    assert tests == {"fcf_margin": "fail", "leverage": "fail",
                     "interest_cov": "n/a", "rule_of_40": "fail",
                     "dilution": "n/a"}
    assert fails == ["fcf_margin", "leverage", "rule_of_40"]
    assert funding.evaluate_tests(None)[1] == fails


# Whole-pipeline exact boundary: revenue 200 vs 160 -> growth 25%; CFO 40,
# capex 10 -> FCF 30 -> margin exactly 15%; Rule of 40 = 25 + 15 = 40.
# No debt (net debt 0), no interest, no dilution data.
def test_screen_exact_boundaries_pass_end_to_end():
    r = screen(fy_only_facts(200, 160, 40, 10))
    assert r["fcf_margin"] == pytest.approx(0.15, abs=1e-12)
    assert r["rule_of_40"] == pytest.approx(40.0, abs=1e-9)
    assert r["tests"]["fcf_margin"] == "pass"
    assert r["tests"]["rule_of_40"] == "pass"
    assert r["tests"]["leverage"] == "pass"
    assert r["tests"]["interest_cov"] == "n/a"
    assert r["tests"]["dilution"] == "n/a"
    assert r["group"] == "A"


def test_assign_group_counts():
    assert funding.assign_group(0) == "A"
    assert funding.assign_group(1) == "B"
    assert funding.assign_group(2) == "C"
    assert funding.assign_group(5) == "C"
    assert funding.assign_group(None) is None


# Group C via the screen: flat revenue (growth 0), FCF 10/100 = 10%:
# fcf_margin fails (<15) and rule_of_40 = 0 + 10 = 10 fails -> 2 fails -> C.
def test_group_c_from_screen():
    r = screen(fy_only_facts(100, 100, 10))
    assert r["fails"] == ["fcf_margin", "rule_of_40"]
    assert r["group"] == "C"


def test_sort_order_group_then_rule_of_40_then_ticker_errors_last():
    def ok(t, g, r40):
        return {"ticker": t, "status": "ok", "group": g, "rule_of_40": r40}
    rows = [
        {"ticker": "ZERR", "status": "error", "error": "x"},
        ok("C1", "C", 90.0),
        ok("B2", "B", 10.0),
        ok("A_NONE", "A", None),
        ok("A_LOW", "A", 45.0),
        {"ticker": "AERR", "status": "error", "error": "y"},
        ok("A_HIGH_B", "A", 70.0),
        ok("A_HIGH_A", "A", 70.0),
        ok("B1", "B", 30.0),
    ]
    out = [r["ticker"] for r in funding.sort_rows(rows)]
    assert out == ["A_HIGH_A", "A_HIGH_B", "A_LOW", "A_NONE", "B1", "B2",
                   "C1", "AERR", "ZERR"]
    # Input is not mutated.
    assert [r["ticker"] for r in rows][0] == "ZERR"


# ---------------------------------------------------------------------------
# 7. Foreign filers, run_funding_screen, CLI (all offline)
# ---------------------------------------------------------------------------

FOREIGN = _doc({
    "Revenues": [_row("2025-01-01", "2025-12-31", 100, "20-F")],
    "NetCashProvidedByUsedInOperatingActivities":
        [_row("2025-01-01", "2025-12-31", 50, "20-F")],
})


def _weak_facts():
    return fy_only_facts(100, 100, 10)


def _mid_facts():
    # dilution 105/100 -> 5% -> one fail (dilution) -> group B
    return good_facts({"WeightedAverageNumberOfDilutedSharesOutstanding": _shares_q(105, 100)})


COMPANIES = {
    "GOOD": good_facts(),
    "MID": _mid_facts(),
    "WEAK": _weak_facts(),
    "FOREIGN": FOREIGN,
}


@pytest.fixture
def offline(monkeypatch):
    """Patch every network boundary used by ``run_funding_screen``."""
    from sec_analyzer.fetch import companyfacts, tickers
    from sec_analyzer import http_client

    class FakeClient:
        pass

    monkeypatch.setattr(http_client, "SecHttpClient", FakeClient)

    # get_company_facts receives only the cik; encode the ticker in it.
    def resolve(ticker, client, no_cache=False):
        if ticker not in COMPANIES:
            raise ValueError(f"Unknown ticker {ticker}")
        return f"T:{ticker}", f"{ticker} Inc"

    def get_facts(cik, client, no_cache=False):
        return COMPANIES[cik.split(":", 1)[1]]

    monkeypatch.setattr(tickers, "resolve_cik", resolve)
    monkeypatch.setattr(companyfacts, "get_company_facts", get_facts)
    return None


def test_foreign_filer_without_10k_is_error_row():
    r = funding.screen_facts("ASML", "ASML", "1", FOREIGN)
    assert r == {"ticker": "ASML", "status": "error", "error": funding.NO_10K_ERROR}
    assert "foreign filer" in r["error"]


def test_empty_or_garbage_facts_are_error_rows_not_exceptions():
    for bad in ({}, None, "x", {"facts": {}}, _doc({})):
        r = funding.screen_facts("X", None, None, bad)
        assert set(r) == {"ticker", "status", "error"} and r["status"] == "error"


def test_run_funding_screen_returns_rows_without_raising(offline):
    rows = funding.run_funding_screen(["foreign", "weak", "NOPE", "good", "mid"])
    assert [r["ticker"] for r in rows] == ["GOOD", "MID", "WEAK", "FOREIGN", "NOPE"]
    assert [r.get("group") for r in rows] == ["A", "B", "C", None, None]
    foreign, nope = rows[3], rows[4]
    assert set(foreign) == {"ticker", "status", "error"}
    assert foreign["error"] == funding.NO_10K_ERROR
    assert set(nope) == {"ticker", "status", "error"}
    assert "Unknown ticker" in nope["error"]


def test_cli_funding_screen_exits_normally_with_foreign_and_unknown(
    offline, monkeypatch, capsys, tmp_path
):
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "sec_analyzer", "funding-screen", "--tickers", "good,foreign,nope",
        "--json", str(out_json), "--quiet",
    ])
    cli.main()                              # must not raise / sys.exit
    out = capsys.readouterr().out
    assert "Group A (all tests passed): 1" in out
    assert "Errors: 2" in out
    assert "FOREIGN" in out and "foreign filer" in out
    assert "NOPE" in out
    assert funding.DISCLAIMER in out
    assert not re.search(r"\bNone\b|\bnan\b", out)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["meta"]["universe"] == "custom"
    assert payload["meta"]["generated_from"] == "SEC companyfacts"
    assert payload["meta"]["disclaimer"] == funding.DISCLAIMER
    assert payload["meta"]["thresholds"]["fcf_margin_min"] == 0.15
    assert payload["meta"]["thresholds"]["dilution_max"] == 0.03
    assert [r["ticker"] for r in payload["rows"]] == ["GOOD", "FOREIGN", "NOPE"]
    assert payload["rows"][0]["group"] == "A"


# ---------------------------------------------------------------------------
# 9. Rendering
# ---------------------------------------------------------------------------

def _no_none_nan(text):
    assert not re.search(r"\bNone\b|\bnan\b|\binf\b", text, re.IGNORECASE), text


def test_render_missing_values_as_dash():
    sparse = {
        "ticker": "SPARSE", "status": "ok", "group": "C", "ttm_end": None,
        "rev_growth": None, "fcf_margin": float("nan"),
        "fcf_sbc_margin": None, "fcf_margin_trend": [None, float("nan"), 0.25],
        "net_debt_ebitda": None, "interest_cov": float("inf"),
        "rule_of_40": None, "dilution": None, "fails": [],
    }
    err = {"ticker": "BAD", "status": "error", "error": "boom"}
    text = funding.render_table([sparse, err], with_valuation=False)
    _no_none_nan(text)
    line = next(l for l in text.splitlines() if l.startswith("SPARSE"))
    assert line.count("—") >= 8
    assert "—/—/25" in line                      # trend: missing points as dash
    assert "BAD: boom" in text
    assert "Group C (2+ tests failed): 1" in text
    assert "Group A (all tests passed): 0" in text


def test_render_valuation_columns_missing_and_error():
    rows = funding.sort_rows([
        {**screen(good_facts(), "OKV"),
         "valuation": {"p_fcf": None, "implied_growth": None, "price_reading": None}},
        {**screen(good_facts(), "ERRV"), "valuation": {"error": "no price"}},
        {**screen(good_facts(), "NOV")},                 # no valuation key at all
    ])
    text = funding.render_table(rows, with_valuation=True)
    _no_none_nan(text)
    assert "P/FCF" in text and "Price reading" in text
    assert "error: no price" in text


def test_render_real_screen_row_has_values_and_no_none():
    text = funding.render_table([screen(good_facts(), "GOOD")])
    _no_none_nan(text)
    row = next(l for l in text.splitlines() if l.startswith("GOOD"))
    assert "2026-09-30" in row and "10.0%" in row and "43.6%" in row
    assert "32.7%" in row and "36/40/44" in row
    assert "-0.68x" in row and "53.6" in row and "1.0%" in row


def test_build_payload_nulls_non_finite():
    row = {"ticker": "X", "status": "ok", "fcf_margin": float("nan"),
           "interest_cov": float("inf"), "trend": [float("-inf"), 1.0]}
    payload = funding.build_payload([row], "custom")
    text = json.dumps(payload, allow_nan=False)        # raises on NaN/Inf
    data = json.loads(text)["rows"][0]
    assert data["fcf_margin"] is None and data["interest_cov"] is None
    assert data["trend"] == [None, 1.0]


# ---------------------------------------------------------------------------
# 10. Valuation overlay
# ---------------------------------------------------------------------------

BASE = {"discount_rate": 0.10, "terminal_growth": 0.03}


def _row_for_overlay(**kw):
    row = {"ticker": "T", "fcf": 48.0, "sbc": 12.0, "rev_growth": 0.10,
           "cover_shares": 100.0}
    row.update(kw)
    return row


@pytest.fixture
def spy(monkeypatch):
    """Replace ``implied_growth_with_status`` with a recording stub."""
    state = {"calls": [], "result": (0.10, "ok")}

    def fake(price, fcf0, terminal_growth, discount_rate, shares, dilution_rate=0.0):
        state["calls"].append((price, fcf0, terminal_growth, discount_rate,
                               shares, dilution_rate))
        return state["result"]

    monkeypatch.setattr(reverse_dcf, "implied_growth_with_status", fake)
    return state


# fcf0 = fcf - sbc = 48 - 12 = 36; market cap = 10 * 100 = 1000;
# P/FCF = 1000 / 48 = 20.8333.
def test_overlay_passes_fcf_minus_sbc_to_reverse_dcf(spy):
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, "2026-10-01", BASE, 999.0)
    assert spy["calls"] == [(10.0, 36.0, 0.03, 0.10, 100.0, 0.0)]
    assert v["fcf0"] == pytest.approx(36)
    assert v["shares"] == 100.0 and v["shares_source"] == "cover"
    assert v["market_cap"] == pytest.approx(1000)
    assert v["p_fcf"] == pytest.approx(1000 / 48)
    assert v["price"] == 10.0 and v["price_as_of"] == "2026-10-01"
    assert v["discount_rate"] == 0.10 and v["terminal_growth"] == 0.03
    assert v["bracket_status"] == "ok"


def test_overlay_uses_engine_shares_when_cover_missing(spy):
    v = funding.valuation_overlay(_row_for_overlay(cover_shares=None), 10.0, None, BASE, 80.0)
    assert spy["calls"][0][4] == 80.0
    assert v["shares"] == 80.0 and v["shares_source"] == "engine"
    assert v["p_fcf"] == pytest.approx(800 / 48)


def test_overlay_no_shares_at_all_is_no_data(spy):
    v = funding.valuation_overlay(_row_for_overlay(cover_shares=None), 10.0, None, BASE, None)
    assert spy["calls"] == []
    assert v["bracket_status"] == "no_data" and v["price_reading"] == "n/a"
    assert v["reason"] == "shares missing"
    assert v["shares"] is None and v["p_fcf"] is None


def test_overlay_p_fcf_none_for_non_positive_fcf(spy):
    # P/FCF needs fcf > 0, so it is None for fcf = -5.
    v = funding.valuation_overlay(_row_for_overlay(fcf=-5.0), 10.0, None, BASE, None)
    assert v["p_fcf"] is None
    # Amendment 1 (Sec.6.4): fcf0 = -5 - 12 = -17 <= 0 never reaches the solver.
    assert spy["calls"] == []
    assert v["bracket_status"] == "no_data" and v["reason"] == "FCF\u2212SBC \u2264 0"


def test_overlay_missing_sbc_is_no_data_real_function():
    # Real reverse_dcf: fcf0=None -> no_data (never a silently un-adjusted FCF).
    v = funding.valuation_overlay(_row_for_overlay(sbc=None), 10.0, None, BASE, None)
    assert v["fcf0"] is None
    assert v["implied_growth"] is None
    assert v["bracket_status"] == "no_data"
    assert v["price_reading"] == "n/a" and v["reason"] == "SBC missing"
    assert v["gap_pp"] is None


def test_overlay_missing_sbc_never_calls_solver(spy):
    funding.valuation_overlay(_row_for_overlay(sbc=None), 10.0, None, BASE, None)
    assert spy["calls"] == []


# Reading boundaries, rev_growth = 10%: gap = implied - 10pp.
#   implied 20.00% -> gap +10.00pp -> NOT above +10 -> in line
#   implied 20.01% -> gap +10.01pp -> acceleration
#   implied  0.00% -> gap -10.00pp -> NOT below -10 -> in line
#   implied -0.01% -> gap -10.01pp -> deceleration
@pytest.mark.parametrize("implied,expected", [
    (0.2000, funding.PRICE_READING_FLAT),
    (0.2001, funding.PRICE_READING_UP),
    (0.0000, funding.PRICE_READING_FLAT),
    (-0.0001, funding.PRICE_READING_DOWN),
    (0.10, funding.PRICE_READING_FLAT),
    (0.45, funding.PRICE_READING_UP),
    (-0.15, funding.PRICE_READING_DOWN),
])
def test_overlay_price_reading_boundaries(spy, implied, expected):
    spy["result"] = (implied, "ok")
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, BASE, None)
    assert v["price_reading"] == expected
    assert v["gap_pp"] == pytest.approx(100 * (implied - 0.10), abs=1e-6)


def test_overlay_reading_none_when_rev_growth_missing(spy):
    spy["result"] = (0.12, "ok")
    v = funding.valuation_overlay(_row_for_overlay(rev_growth=None), 10.0, None, BASE, None)
    assert v["price_reading"] is None and v["gap_pp"] is None
    assert v["implied_growth"] == 0.12


@pytest.mark.parametrize("status,label", [
    ("above_bracket", funding.PRICE_READING_ABOVE_BRACKET),
    ("below_bracket", funding.PRICE_READING_BELOW_BRACKET),
    ("no_data", "n/a"),
])
def test_overlay_reports_non_ok_bracket_status(spy, status, label):
    spy["result"] = (None, status)
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, BASE, None)
    assert v["bracket_status"] == status
    assert v["price_reading"] == label
    assert v["gap_pp"] is None


# Real function: a huge price (1e9/share on 100 shares vs FCF0 36) cannot be
# reached by 60% growth -> above_bracket; a tiny price (0.01) is below even
# -20% growth value -> below_bracket.
def test_overlay_real_bracket_statuses():
    hi = funding.valuation_overlay(_row_for_overlay(), 1e9, None, BASE, None)
    assert hi["bracket_status"] == "above_bracket"
    assert hi["price_reading"] == "implied growth above +60% (off scale)"
    assert hi["implied_growth"] is None
    lo = funding.valuation_overlay(_row_for_overlay(), 0.01, None, BASE, None)
    assert lo["bracket_status"] == "below_bracket"
    assert lo["price_reading"] == "implied growth below \u221220% (off scale)"


def test_overlay_bad_inputs_never_raise():
    assert funding.valuation_overlay(_row_for_overlay(), None, None, BASE, None) == \
        {"error": "missing price"}
    assert funding.valuation_overlay(_row_for_overlay(), -1.0, None, BASE, None) == \
        {"error": "missing price"}
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, None, None)
    assert v["bracket_status"] == "no_data"           # no assumptions
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None,
                                  {"discount_rate": 0.03, "terminal_growth": 0.05}, None)
    assert v["bracket_status"] == "no_data"           # r <= g_t


def test_overlay_exception_becomes_error_dict(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaput")
    monkeypatch.setattr(reverse_dcf, "implied_growth_with_status", boom)
    assert funding.valuation_overlay(_row_for_overlay(), 10.0, None, BASE, None) == \
        {"error": "RuntimeError: kaput"}


def _pipeline_factory(fail_for=(), price=10.0, spy_calls=None):
    def fake_pipeline(ticker, years, no_cache, as_of, fred_rate):
        if spy_calls is not None:
            spy_calls.append(ticker)
        if ticker in fail_for:
            raise RuntimeError("boom")
        print("progress noise that must not reach the table")
        result = {"valuation": {"assumptions": {"base": dict(BASE)}}}
        return result, price, "2026-10-01", {"shares": 90.0}
    return fake_pipeline


def test_run_with_valuation_attaches_blocks_and_isolates_failure(
    offline, monkeypatch, spy, capsys
):
    from sec_analyzer import calibrate
    spy["result"] = (0.12, "ok")
    seen = []
    monkeypatch.setattr(calibrate, "run_headless_pipeline",
                        _pipeline_factory(fail_for=("WEAK",), spy_calls=seen))
    monkeypatch.setattr(cli, "_fetch_risk_free_asof", lambda *a, **k: None)

    plain = {r["ticker"]: r for r in funding.run_funding_screen(["GOOD", "WEAK", "FOREIGN"])}
    rows = funding.run_funding_screen(["GOOD", "WEAK", "FOREIGN"], valuation=True)
    by = {r["ticker"]: r for r in rows}

    assert seen == ["GOOD", "WEAK"]                    # error rows skipped
    assert "valuation" not in by["FOREIGN"]
    assert by["WEAK"]["valuation"] == {"error": "boom"}
    # The screen row itself is intact despite the overlay failure.
    w = dict(by["WEAK"]); w.pop("valuation")
    assert w == plain["WEAK"]
    assert by["WEAK"]["group"] == "C"

    g = by["GOOD"]["valuation"]
    assert g["fcf0"] == pytest.approx(36)
    assert g["shares"] == pytest.approx(100) and g["shares_source"] == "cover"
    assert g["price"] == 10.0 and g["implied_growth"] == 0.12
    # gap = 12 - 10 = 2pp -> in line
    assert g["price_reading"] == funding.PRICE_READING_FLAT
    assert spy["calls"][0] == (10.0, pytest.approx(36), 0.03, 0.10,
                               pytest.approx(100), 0.0)
    # The pipeline's own stdout noise was swallowed.
    assert "progress noise" not in capsys.readouterr().out
    text = funding.render_table(rows, with_valuation=True)
    assert "error: boom" in text
    _no_none_nan(text)


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------

def test_two_runs_give_identical_rows_and_table(offline):
    tickers = ["WEAK", "GOOD", "FOREIGN", "MID", "NOPE"]
    a = funding.run_funding_screen(tickers)
    b = funding.run_funding_screen(list(tickers))
    assert a == b
    assert funding.render_table(a) == funding.render_table(b)
    assert json.dumps(funding.build_payload(a, "custom"), sort_keys=True) == \
        json.dumps(funding.build_payload(b, "custom"), sort_keys=True)
    # Input order does not change the output order either.
    c = funding.run_funding_screen(list(reversed(tickers)))
    assert [r["ticker"] for r in c] == [r["ticker"] for r in a]


# ---------------------------------------------------------------------------
# Amendment 1 (FUNDING_SPEC Sec.5.1, 5.2, 6.4-6.6, 10)
# ---------------------------------------------------------------------------

def test_reading_labels_exact():
    assert funding.PRICE_READING_UP == "price implies FCF growth well above TTM sales growth"
    assert funding.PRICE_READING_DOWN == "price implies FCF growth well below TTM sales growth"
    assert funding.PRICE_READING_FLAT == "price-implied FCF growth near TTM sales growth"


@pytest.mark.parametrize("kw,reason", [
    (dict(price_reliable=False), "unreliable price"),
    (dict(sector_type="financial"), "FCF-DCF not applicable (financial/REIT)"),
    (dict(sector_type="reit"), "FCF-DCF not applicable (financial/REIT)"),
    (dict(dcf_enabled=False), "FCF-DCF not applicable (financial/REIT)"),
])
def test_overlay_engine_guards(spy, kw, reason):
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, BASE, None, **kw)
    assert spy["calls"] == []
    assert v["bracket_status"] == "no_data" and v["implied_growth"] is None
    assert v["reason"] == reason and v["price_reading"] == "n/a"


def test_overlay_unreliable_price_suppresses_p_fcf(spy):
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, BASE, None,
                                  price_reliable=False)
    assert v["p_fcf"] is None
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, BASE, None,
                                  price_reliable=True, sector_type="mature",
                                  dcf_enabled=True)
    assert v["p_fcf"] == pytest.approx(1000 / 48) and len(spy["calls"]) == 1


def test_overlay_non_positive_fcf_minus_sbc_is_no_data_with_real_solver():
    # fcf 10 - sbc 12 = -2: must NOT be reported as above_bracket.
    v = funding.valuation_overlay(_row_for_overlay(fcf=10.0), 10.0, None, BASE, None)
    assert v["bracket_status"] == "no_data" and v["reason"] == "FCF\u2212SBC \u2264 0"
    assert v["p_fcf"] == pytest.approx(1000 / 10)


def test_overlay_capex_missing_note_blocks(spy):
    row = _row_for_overlay(notes=["capex_missing: no CapEx tag in the filing; treated as 0"])
    v = funding.valuation_overlay(row, 10.0, None, BASE, None)
    assert spy["calls"] == [] and v["reason"] == "CapEx not tagged"


@pytest.mark.parametrize("base", [
    {"discount_rate": 0.069, "terminal_growth": 0.03},
    {"discount_rate": 0.10, "terminal_growth": 0.041},
])
def test_overlay_assumption_bounds(spy, base):
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, base, None)
    assert spy["calls"] == [] and v["reason"] == "assumptions out of bounds"
    ok = funding.valuation_overlay(
        _row_for_overlay(), 10.0, None, {"discount_rate": 0.07, "terminal_growth": 0.04}, None)
    assert ok["reason"] is None and len(spy["calls"]) == 1


def test_overlay_ytd_suffix_and_audit_fields(spy):
    spy["result"] = (0.45, "ok")
    eng = {"implied_growth": 0.30, "fcf0": 10.0, "fcf0_source": "ttm"}
    v = funding.valuation_overlay(_row_for_overlay(rev_growth_basis="ytd"), 10.0,
                                  "2026-10-01", BASE, None, engine_reverse_dcf=eng)
    assert v["price_reading"] == funding.PRICE_READING_UP + " (YTD growth basis)"
    assert v["engine_implied_growth"] == 0.30
    assert v["engine_fcf0"] == 10.0 and v["engine_fcf0_source"] == "ttm"
    assert v["price_as_of"] == "2026-10-01" and v["shares_source"] == "cover"
    assert v["base_divergence"]                      # 36 vs 10 -> > 50%
    v = funding.valuation_overlay(_row_for_overlay(), 10.0, None, BASE, None,
                                  engine_reverse_dcf={"fcf0": 30.0})
    assert v["base_divergence"] is None              # 36/30-1 = 20%


def test_capex_tag_present_but_misaligned_gives_none_fcf():
    # CapEx only has FY data (ends 2025-12-31 != E): never a silent OCF-only FCF.
    r = screen(good_facts({"PaymentsToAcquirePropertyPlantAndEquipment": _dec_flow(10)}))
    assert r["fcf"] is None and r["fcf_margin"] is None and r["rule_of_40"] is None
    assert r["tests"]["fcf_margin"] == "fail" and r["tests"]["rule_of_40"] == "fail"
    assert any("CapEx" in n for n in r["notes"])
    assert not any("capex_missing" in n for n in r["notes"])


def test_no_capex_tag_adds_capex_missing_note():
    r = screen(good_facts(drop=["PaymentsToAcquirePropertyPlantAndEquipment"]))
    assert any("capex_missing" in n for n in r["notes"])


def test_no_debt_tags_with_interest_makes_net_debt_unknown():
    r = screen(good_facts(drop=["LongTermDebtNoncurrent"]))   # interest TTM 3 > 0
    assert r["net_debt"] is None and r["net_debt_ebitda"] is None
    assert r["tests"]["leverage"] == "fail"
    assert any("net_debt unknown" in n for n in r["notes"])
    # No interest either -> debt genuinely none: net cash-like, leverage passes.
    r = screen(good_facts(drop=["LongTermDebtNoncurrent", "InterestExpense"]))
    assert r["net_debt"] == pytest.approx(-60) and r["tests"]["leverage"] == "pass"


def test_net_revenue_guard_switches_when_more_than_5_percent_apart():
    over = {"RevenuesNetOfInterestExpense": _dec_flow(60, 50, 45, fy24=50, p24=35)}
    r = screen(good_facts(over))            # net TTM = 60+50-45 = 65 vs 110
    assert r["revenue_tag"] == "RevenuesNetOfInterestExpense"
    assert r["revenue"] == pytest.approx(65)
    assert any("net-revenue guard" in n for n in r["notes"])
    # Within 5% (108 vs 110): unchanged.
    over = {"RevenuesNetOfInterestExpense": _dec_flow(98, 80, 70, fy24=90, p24=60)}
    r = screen(good_facts(over))            # net TTM 108 -> 1.8% apart
    assert r["revenue_tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_rev_growth_basis_ttm_and_ytd():
    assert screen(good_facts())["rev_growth_basis"] == "ttm"
    # No FY24 / 9M24 history: prior TTM unavailable -> YTD growth 80/70-1.
    r = screen(good_facts({
        "RevenueFromContractWithCustomerExcludingAssessedTax": _dec_flow(100, 80, 70)}))
    assert r["rev_growth_basis"] == "ytd"
    assert r["rev_growth"] == pytest.approx(80 / 70 - 1)
    assert any("YTD" in n for n in r["notes"])


def test_interest_fy_fallback_bounded_to_400_days():
    # FY ends 2025-12-31; E = 2026-09-30 (273 days): used, note names the FY end.
    r = screen(good_facts({"LongTermDebtNoncurrent": [_inst(E, 200)],
                           "InterestExpense": _dec_flow(4)}))
    assert r["interest_cov"] == pytest.approx(35 / 4)
    assert any("2025-12-31" in n for n in r["notes"])
    # Interest FY ends 2024-12-31 (639 days before E): too old -> None.
    r = screen(good_facts({"LongTermDebtNoncurrent": [_inst(E, 200)],
                           "InterestExpense": [_row("2024-01-01", "2024-12-31", 4)]}))
    assert r["interest_cov"] is None and r["tests"]["interest_cov"] == "n/a"


def test_threshold_rounding_103_over_100_passes_dilution():
    assert 103 / 100 - 1 > 0.03                      # raw float noise
    assert funding.evaluate_tests({"dilution": 103 / 100 - 1})[0]["dilution"] == "pass"


def test_render_valuation_headers_and_reason():
    text = funding.render_table([{
        **screen(good_facts(), "OKV"),
        "valuation": {"p_fcf": None, "implied_growth": None,
                      "price_reading": "n/a", "reason": "SBC missing"}}], True)
    assert "P/FCF (pre-SBC)" in text and "Impl. FCF-SBC g (5y)" in text
    assert "n/a (SBC missing)" in text


def test_disclaimer_has_implied_growth_and_limitations():
    assert "5-year growth" in funding.DISCLAIMER
    assert funding.DISCLAIMER.rstrip().endswith("higher implied growth.")
    assert "cover-page shares are basic" in funding.DISCLAIMER