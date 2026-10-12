"""Hand-verified tests for ``sec_analyzer.normalize.ttm`` (FUNDING_SPEC Sec.4).

All fixtures are tiny synthetic SEC companyfacts documents; nothing touches
the network. Expected values are derived by hand in the comment above each
test.
"""

import pytest

from sec_analyzer.normalize import ttm


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _row(start, end, val, form="10-K", filed="2026-02-15", accn="A"):
    r = {"end": end, "val": val, "form": form, "filed": filed, "accn": accn}
    if start is not None:
        r["start"] = start
    return r


def _doc(us_gaap=None, dei=None):
    """companyfacts dict. ``us_gaap``/``dei``: ``{tag: rows}`` or
    ``{tag: (unit, rows)}``."""
    def ns(d):
        out = {}
        for tag, v in (d or {}).items():
            unit, rows = v if isinstance(v, tuple) else ("USD", v)
            out[tag] = {"units": {unit: rows}}
        return out
    return {"facts": {"us-gaap": ns(us_gaap), "dei": ns(dei)}}


def _dec_flow(fy25, y26=None, p25=None, fy24=None, p24=None):
    """Calendar-year flow rows: FY2025, 9M 2026, 9M 2025, FY2024, 9M 2024."""
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


def _inst(end, val, form="10-Q", filed="2026-10-25"):
    return _row(None, end, val, form, filed)


# ---------------------------------------------------------------------------
# 1. TTM formula
# ---------------------------------------------------------------------------

# Hand derivation (FY Dec, latest 10-Q = 9M to 2026-09-30):
#   TTM        = A_F + Y - P           = 100 + 80 - 70          = 110
#   prior TTM  = A_{F-1} + P - P_{-1}  = 90 + 70 - 60           = 100
#   growth     = 110 / 100 - 1                                  = 10%
def test_ttm_formula_and_prior_ttm():
    facts = _doc({"Revenues": _dec_flow(100, 80, 70, fy24=90, p24=60)})
    r = ttm.ttm_flow(facts, ["Revenues"])
    assert r["tag"] == "Revenues"
    assert r["fy_end"] == "2025-12-31"
    assert r["ttm_end"] == "2026-09-30"
    assert r["ttm"] == pytest.approx(110)
    assert r["ttm_prior"] == pytest.approx(100)
    assert r["ttm"] / r["ttm_prior"] - 1 == pytest.approx(0.10)
    assert r["ytd"] == 80 and r["ytd_prior"] == 70
    assert r["ytd_growth"] is None
    assert r["annual"] == {"2024-12-31": 90, "2025-12-31": 100}


# Hand derivation: without the 9M-2024 fact the prior TTM cannot be built, so
# ytd_growth = Y / P - 1 = 80 / 70 - 1 = 1/7 = 0.142857...
def test_ytd_growth_when_prior_ttm_unavailable():
    facts = _doc({"Revenues": _dec_flow(100, 80, 70, fy24=90)})
    r = ttm.ttm_flow(facts, ["Revenues"])
    assert r["ttm"] == pytest.approx(110)
    assert r["ttm_prior"] is None
    assert r["ytd_growth"] == pytest.approx(1 / 7)


# Hand derivation: YTD present but no prior-year YTD -> TTM falls back to the
# annual value (100), ending on the FY end; prior = FY-1 = 90.
def test_ytd_without_prior_ytd_falls_back_to_annual():
    facts = _doc({"Revenues": _dec_flow(100, 80, None, fy24=90)})
    r = ttm.ttm_flow(facts, ["Revenues"])
    assert r["ttm"] == 100
    assert r["ttm_end"] == "2025-12-31"
    assert r["ytd"] is None and r["ytd_prior"] is None
    assert r["ttm_prior"] == 90


# Hand derivation: FY-only data -> TTM = 100 at FY end, prior = 90.
def test_fy_only_ttm_is_annual():
    facts = _doc({"Revenues": _dec_flow(100, fy24=90)})
    r = ttm.ttm_flow(facts, ["Revenues"])
    assert (r["ttm"], r["ttm_end"], r["ttm_prior"]) == (100, "2025-12-31", 90)


def test_most_recently_filed_fact_wins():
    # Same (start, end) key filed twice: the restated 105 (filed later) wins.
    rows = [
        _row("2025-01-01", "2025-12-31", 100, "10-K", "2026-02-15"),
        _row("2025-01-01", "2025-12-31", 105, "10-K/A", "2026-04-01"),
    ]
    r = ttm.ttm_flow(_doc({"Revenues": rows}), ["Revenues"])
    assert r["ttm"] == 105


def test_ttm_flow_none_cases():
    assert ttm.ttm_flow({}, ["Revenues"]) is None
    assert ttm.ttm_flow(_doc({}), ["Revenues"]) is None
    assert ttm.ttm_flow(None, ["Revenues"]) is None
    assert ttm.ttm_flow(_doc({"Revenues": _dec_flow(100)}), []) is None
    # 10-Q only: no annual data, so no TTM.
    q_only = [_row("2026-01-01", "2026-09-30", 80, "10-Q")]
    assert ttm.ttm_flow(_doc({"Revenues": q_only}), ["Revenues"]) is None
    # 20-F rows are ignored entirely.
    f20 = [_row("2025-01-01", "2025-12-31", 100, "20-F")]
    assert ttm.ttm_flow(_doc({"Revenues": f20}), ["Revenues"]) is None


def test_malformed_rows_are_skipped_not_raised():
    rows = [
        "garbage",
        {"end": "bad-date", "start": "2025-01-01", "val": 5, "form": "10-K"},
        {"end": "2025-12-31", "start": "2025-01-01", "val": None, "form": "10-K"},
        {"end": "2025-12-31", "start": "2025-01-01", "val": True, "form": "10-K"},
        _row("2025-01-01", "2025-12-31", 100),
    ]
    r = ttm.ttm_flow(_doc({"Revenues": rows}), ["Revenues"])
    assert r["ttm"] == 100


# Tag choice: with want_end the first tag whose TTM ends there wins even if an
# earlier tag is stale; without it the latest ttm_end wins.
def test_tag_choice_want_end_and_latest():
    stale = _dec_flow(1000)                        # TTM ends 2025-12-31
    fresh = _dec_flow(100, 80, 70, fy24=90)        # TTM ends 2026-09-30
    facts = _doc({"A": stale, "B": fresh})
    assert ttm.ttm_flow(facts, ["A", "B"])["tag"] == "B"            # latest end
    assert ttm.ttm_flow(facts, ["A", "B"], want_end="2026-09-30")["tag"] == "B"
    assert ttm.ttm_flow(facts, ["A", "B"], want_end="2025-12-31")["tag"] == "A"
    # Nothing ends on want_end: falls back to the latest-ending tag.
    r = ttm.ttm_flow(facts, ["A", "B"], want_end="2020-01-01")
    assert r["tag"] == "B"
    # Tie on ttm_end goes to list order.
    tie = _doc({"A": _dec_flow(1), "B": _dec_flow(2)})
    assert ttm.ttm_flow(tie, ["B", "A"])["tag"] == "B"


# ---------------------------------------------------------------------------
# 2. January fiscal year / 52-53 week shift
# ---------------------------------------------------------------------------

# Hand derivation (FY ends 2026-01-31, prior FY ended 2025-02-01, i.e. 364
# days earlier; 9M to 2026-10-31; prior 9M ended 2025-10-30 (2 days off
# E-364 = 2025-11-01); 9M two years back ended 2024-11-05 (3 days off
# 2024-11-02)):
#   TTM       = 200 + 150 - 120 = 230
#   prior TTM = 180 + 120 - 100 = 200   -> growth 15%
# A 91-day distractor ending exactly on 2025-11-01 (value 999) must be
# rejected by the duration tolerance.
def test_january_fiscal_year_matches_shifted_prior_periods():
    rows = [
        _row("2025-02-02", "2026-01-31", 200, "10-K", "2026-03-20"),
        _row("2024-02-04", "2025-02-01", 180, "10-K", "2025-03-20"),
        _row("2026-02-01", "2026-10-31", 150, "10-Q", "2026-12-05"),
        _row("2025-02-02", "2025-10-30", 120, "10-Q", "2025-12-05"),
        _row("2024-02-06", "2024-11-05", 100, "10-Q", "2024-12-05"),
        _row("2025-08-03", "2025-11-01", 999, "10-Q", "2025-12-05"),
    ]
    r = ttm.ttm_flow(_doc({"Revenues": rows}), ["Revenues"])
    assert r["fy_end"] == "2026-01-31"
    assert r["ttm_end"] == "2026-10-31"
    assert r["ytd_prior"] == 120
    assert r["ttm"] == pytest.approx(230)
    assert r["ttm_prior"] == pytest.approx(200)
    assert r["ttm"] / r["ttm_prior"] - 1 == pytest.approx(0.15)


# Prior-year YTD outside the +-10 day end tolerance is NOT matched:
# E = 2026-09-30, E-364 = 2025-10-01; a prior YTD ending 2025-10-15 (14 days
# off) is rejected -> TTM falls back to the annual value.
def test_prior_ytd_outside_tolerance_not_matched():
    rows = _dec_flow(100, 80, None, fy24=90)
    rows.append(_row("2025-01-01", "2025-10-15", 70, "10-Q"))
    r = ttm.ttm_flow(_doc({"Revenues": rows}), ["Revenues"])
    assert r["ttm"] == 100 and r["ttm_end"] == "2025-12-31"


# Prior FY must end within +-10 days of FY end - 365: a prior FY ending
# 2025-01-31 vs target 2024-12-31 (31 days off) is not used as A_{F-1}.
def test_prior_fy_outside_tolerance_ignored():
    rows = [
        _row("2025-01-01", "2025-12-31", 100, "10-K"),
        _row("2024-02-01", "2025-01-31", 90, "10-K"),
    ]
    r = ttm.ttm_flow(_doc({"Revenues": rows}), ["Revenues"])
    assert r["ttm_prior"] is None


# ---------------------------------------------------------------------------
# 3. AMZN trap
# ---------------------------------------------------------------------------

# A 10-Q tags a 365-day trailing-12-month duration (2025-03-31..2026-03-31,
# value 999). Only the 10-K (100, FY 2025-12-31) may count as the fiscal year.
def test_amzn_trap_10q_365_day_duration_is_not_a_fiscal_year():
    rows = _dec_flow(100, fy24=90)
    rows.append(_row("2025-03-31", "2026-03-31", 999, "10-Q", "2026-04-28"))
    r = ttm.ttm_flow(_doc({"Revenues": rows}), ["Revenues"])
    assert r["fy_end"] == "2025-12-31"
    assert "2026-03-31" not in r["annual"]
    assert r["ttm"] == 100
    assert r["ttm_end"] == "2025-12-31"


def test_amzn_trap_only_10q_trailing_year_means_no_annual_data():
    rows = [_row("2025-03-31", "2026-03-31", 999, "10-Q")]
    assert ttm.ttm_flow(_doc({"Revenues": rows}), ["Revenues"]) is None


# ---------------------------------------------------------------------------
# 4. Balance sheet / ORCL debt
# ---------------------------------------------------------------------------

E = "2026-09-30"

# Hand derivation (ORCL style): LongTermNotesAndLoans 50 + NotesPayableCurrent
# 10 = 60 debt; liquidity = cash 20 + short-term investments 5 = 25;
# net debt = 60 - 25 = 35.
def test_balance_orcl_notes_and_current_notes_summed():
    facts = _doc({
        "LongTermNotesAndLoans": [_inst(E, 50)],
        "NotesPayableCurrent": [_inst(E, 10)],
        "CashAndCashEquivalentsAtCarryingValue": [_inst(E, 20)],
        "ShortTermInvestments": [_inst(E, 5)],
    })
    b = ttm.balance(facts, E)
    assert b["debt"] == pytest.approx(60)
    assert b["liquidity"] == pytest.approx(25)
    assert b["net_debt"] == pytest.approx(35)


# Hand derivation: no long-term (noncurrent) tag, so the combined LongTermDebt
# (70, which already contains the 10 current portion) is used and the current
# tag is NOT added again: debt = 70 + commercial paper 5 = 75. Liquidity = 30
# (cash falls back to the restricted-cash tag) -> net debt 45.
def test_balance_long_term_debt_fallback_does_not_double_count_current():
    facts = _doc({
        "LongTermDebt": [_inst(E, 70)],
        "LongTermDebtCurrent": [_inst(E, 10)],
        "CommercialPaper": [_inst(E, 5)],
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": [_inst(E, 30)],
    })
    b = ttm.balance(facts, E)
    assert b["debt"] == pytest.approx(75)
    assert b["liquidity"] == pytest.approx(30)
    assert b["net_debt"] == pytest.approx(45)


# Hand derivation: noncurrent 80 + current 20 = 100; ShortTermBorrowings 7
# -> 107; no cash -> net debt 107.
def test_balance_noncurrent_plus_current_plus_short_term():
    facts = _doc({
        "LongTermDebtNoncurrent": [_inst(E, 80)],
        "LongTermDebtCurrent": [_inst(E, 20)],
        "ShortTermBorrowings": [_inst(E, 7)],
    })
    assert ttm.balance(facts, E)["net_debt"] == pytest.approx(107)


def test_balance_empty_and_wrong_date():
    assert ttm.balance(_doc({}), E) == {
        "debt": 0.0, "liquidity": 0.0, "net_debt": 0.0, "debt_tag_found": False}
    facts = _doc({"LongTermDebtNoncurrent": [_inst("2026-06-30", 80)]})
    b = ttm.balance(facts, E)
    assert b["debt"] == 0.0 and b["debt_tag_found"] is False   # only exact-end facts count


def test_balance_debt_tag_found_true_for_any_debt_list():
    assert ttm.balance(_doc({"LongTermDebtNoncurrent": [_inst(E, 5)]}), E)["debt_tag_found"]
    assert ttm.balance(_doc({"CommercialPaper": [_inst(E, 5)]}), E)["debt_tag_found"]
    assert ttm.balance(_doc({"LongTermDebt": [_inst(E, 5)]}), E)["debt_tag_found"]
    assert ttm.balance(_doc({"DebtCurrent": [_inst(E, 5)]}), E)["debt_tag_found"]
    assert not ttm.balance(_doc({"CashAndCashEquivalentsAtCarryingValue": [_inst(E, 5)]}), E)["debt_tag_found"]


def test_instant_ignores_duration_facts_and_takes_latest_filed():
    rows = [
        _row("2026-07-01", E, 999, "10-Q"),          # duration: ignored
        _inst(E, 10, filed="2026-10-25"),
        _inst(E, 12, form="10-Q/A", filed="2026-11-10"),
    ]
    facts = _doc({"Cash": rows, "Other": [_inst(E, 3)]})
    assert ttm.instant(facts, ["Cash"], E) == 12
    # List order: the first tag with a fact wins.
    assert ttm.instant(facts, ["Missing", "Other", "Cash"], E) == 3
    assert ttm.instant(facts, ["Missing"], E) is None


# ---------------------------------------------------------------------------
# Diluted share change
# ---------------------------------------------------------------------------

def _shares(rows):
    return {"WeightedAverageNumberOfDilutedSharesOutstanding": ("shares", rows)}


# Hand derivation: quarter to 2026-09-30 = 102, quarter to 2025-09-30 = 100:
#   102 / 100 - 1 = +2%. The annual figures (130 vs 100 = +30%) must NOT be
#   used when the quarterly pair exists.
def test_dilution_quarterly_primary():
    rows = [
        _row("2026-07-01", E, 102, "10-Q"),
        _row("2025-07-01", "2025-09-30", 100, "10-Q"),
        _row("2025-01-01", "2025-12-31", 130, "10-K"),
        _row("2024-01-01", "2024-12-31", 100, "10-K"),
    ]
    assert ttm.diluted_share_change(_doc(_shares(rows)), E) == pytest.approx(0.02)


# Hand derivation: no quarterly pair -> annual fallback: 104 / 100 - 1 = 4%.
def test_dilution_annual_fallback():
    rows = [
        _row("2025-01-01", "2025-12-31", 104, "10-K"),
        _row("2024-01-01", "2024-12-31", 100, "10-K"),
    ]
    assert ttm.diluted_share_change(_doc(_shares(rows)), E) == pytest.approx(0.04)


# A 9-month (272 day) row is not a quarter, so it cannot form the pair; one
# annual row alone is not enough for the fallback either.
def test_dilution_insufficient_data_is_none():
    rows = [
        _row("2026-01-01", E, 102, "10-Q"),
        _row("2025-01-01", "2025-09-30", 100, "10-Q"),
        _row("2025-01-01", "2025-12-31", 104, "10-K"),
    ]
    assert ttm.diluted_share_change(_doc(_shares(rows)), E) is None
    assert ttm.diluted_share_change(_doc({}), E) is None
    assert ttm.diluted_share_change(_doc(_shares(rows)), "not-a-date") is None


# ---------------------------------------------------------------------------
# 6. Cover shares
# ---------------------------------------------------------------------------

def _cover(rows):
    return {"EntityCommonStockSharesOutstanding": ("shares", rows)}


# Hand derivation: latest filing (accn Q3, filed 2026-10-25) has classes
# 50 + 30 + 20 = 100. The older filing (accn Q2: 40 + 30 = 70) must not be
# added (would be 170).
def test_cover_shares_sums_classes_of_latest_filing_only():
    rows = [
        _row(None, "2026-10-20", 50, "10-Q", "2026-10-25", ) | {"accn": "Q3"},
        _row(None, "2026-10-20", 30, "10-Q", "2026-10-25") | {"accn": "Q3"},
        _row(None, "2026-10-20", 20, "10-Q", "2026-10-25") | {"accn": "Q3"},
        _row(None, "2026-07-20", 40, "10-Q", "2026-07-25") | {"accn": "Q2"},
        _row(None, "2026-07-20", 30, "10-Q", "2026-07-25") | {"accn": "Q2"},
    ]
    shares, as_of = ttm.cover_shares(_doc(dei=_cover(rows)))
    assert shares == pytest.approx(100)
    assert as_of == "2026-10-20"


def test_cover_shares_unavailable():
    assert ttm.cover_shares(_doc()) == (None, None)
    assert ttm.cover_shares({}) == (None, None)
    assert ttm.cover_shares(None) == (None, None)
