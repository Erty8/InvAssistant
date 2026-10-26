"""Data-reading fixes (SPEC.md Sec.32): net-income and debt tag fallbacks,
20-F/40-F annual reports, and the latest 10-Q share count.

Each case mirrors a real filer whose fundamentals the engine misread:
Broadcom (ProfitLoss from FY2025), Freeport (common-holder income only),
Broadcom/Oracle/Hims debt tags, Sea Ltd (20-F), Oklo/CoreWeave share counts.
"""

from sec_analyzer.normalize.metrics import compute_metrics
from sec_analyzer.normalize.normalizer import normalize_facts, to_annual_series
from sec_analyzer.normalize.ratios import compute_ratios


def _row(end, val, *, fy, fp="FY", form="10-K", start=None, filed=None):
    row = {"end": end, "val": val, "fy": fy, "fp": fp, "form": form,
           "filed": filed or f"{int(end[:4]) + 1}-02-15", "accn": f"acc-{end}-{form}"}
    if start:
        row["start"] = start
    return row


def _facts(usgaap, dei=None):
    facts = {"us-gaap": {tag: {"units": {"USD": rows}} for tag, rows in usgaap.items()}}
    if dei:
        facts["dei"] = {tag: {"units": {"shares": rows}} for tag, rows in dei.items()}
    return {"entityName": "Test Co", "cik": 1, "facts": facts}


def _annual(tag_rows):
    return {tag: [_row(f"{fy}-12-31", v, fy=fy, start=f"{fy}-01-01") for fy, v in rows.items()]
            for tag, rows in tag_rows.items()}


# ---------------------------------------------------------------------------
# Net income
# ---------------------------------------------------------------------------


def test_profitloss_fills_years_net_income_loss_left_empty():
    # Broadcom: NetIncomeLoss through FY2024, ProfitLoss from FY2025.
    n = normalize_facts(_facts(_annual({
        "NetIncomeLoss": {2023: 14.08e9, 2024: 5.89e9},
        "ProfitLoss": {2024: 9.99e9, 2025: 23.13e9},
    })))

    series = to_annual_series(n, "NetIncome")
    assert series[2025] == 23.13e9
    assert series[2024] == 5.89e9  # NetIncomeLoss still wins where both exist


def test_common_holder_income_preferred_over_consolidated_profitloss():
    # Freeport: ProfitLoss includes large noncontrolling interests.
    n = normalize_facts(_facts(_annual({
        "NetIncomeLossAvailableToCommonStockholdersBasic": {2025: 2.20e9},
        "ProfitLoss": {2025: 4.15e9},
    })))

    assert to_annual_series(n, "NetIncome")[2025] == 2.20e9


# ---------------------------------------------------------------------------
# Debt
# ---------------------------------------------------------------------------


def _debt_metrics(balance):
    usgaap = {tag: [_row(f"{fy}-12-31", v, fy=fy) for fy, v in rows.items()] for tag, rows in balance.items()}
    usgaap["Revenues"] = [_row("2025-12-31", 10e9, fy=2025, start="2025-01-01")]
    n = normalize_facts(_facts(usgaap))
    return compute_metrics(n, compute_ratios(n), 10.0)


def test_debt_read_from_capital_lease_combined_tag():
    # Broadcom's 10-Ks since FY2022 skip LongTermDebtNoncurrent.
    m = _debt_metrics({"LongTermDebtAndCapitalLeaseObligations": {2025: 61.98e9},
                       "LongTermDebtCurrent": {2025: 3.15e9}})
    assert m["total_debt"] == 61.98e9 + 3.15e9


def test_debt_read_from_notes_tags():
    # Oracle: notes payable instead of LongTermDebt*.
    m = _debt_metrics({"LongTermNotesPayable": {2025: 122.34e9}, "DebtCurrent": {2025: 7.20e9}})
    assert m["total_debt"] == 122.34e9 + 7.20e9


def test_convertible_debt_counted_when_it_is_the_only_debt():
    # Hims & Hers: convertible notes only.
    m = _debt_metrics({"ConvertibleDebtNoncurrent": {2025: 0.97e9}})
    assert m["total_debt"] == 0.97e9


def test_standard_debt_tags_still_win():
    m = _debt_metrics({"LongTermDebtNoncurrent": {2025: 78.33e9},
                       "LongTermDebtAndCapitalLeaseObligations": {2025: 80.0e9},
                       "LongTermDebtCurrent": {2025: 12.35e9},
                       "DebtCurrent": {2025: 15.0e9}})
    assert m["total_debt"] == 78.33e9 + 12.35e9


# ---------------------------------------------------------------------------
# 20-F / 40-F annual reports
# ---------------------------------------------------------------------------


def test_20f_and_40f_annual_facts_are_read():
    usgaap = {"NetIncomeLoss": [
        _row("2025-12-31", 1.578e9, fy=2025, form="20-F", start="2025-01-01"),
        _row("2024-12-31", 0.444e9, fy=2024, form="40-F", start="2024-01-01"),
    ]}
    series = to_annual_series(normalize_facts(_facts(usgaap)), "NetIncome")
    assert series == {2025: 1.578e9, 2024: 0.444e9}


def test_amended_and_quarterly_forms_still_not_annual():
    usgaap = {"NetIncomeLoss": [
        _row("2025-12-31", 1.0e9, fy=2025, form="20-F/A", start="2025-01-01"),
        _row("2025-12-31", 2.0e9, fy=2025, form="6-K", start="2025-01-01"),
    ]}
    assert to_annual_series(normalize_facts(_facts(usgaap)), "NetIncome") == {}


# ---------------------------------------------------------------------------
# Latest share count
# ---------------------------------------------------------------------------


def _share_metrics(dei_rows=None, usgaap_share_rows=None):
    usgaap = {"Revenues": [_row("2025-12-31", 1e9, fy=2025, start="2025-01-01")]}
    if usgaap_share_rows:
        facts = _facts(usgaap)
        facts["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
            "units": {"shares": usgaap_share_rows}}
    else:
        facts = _facts(usgaap, dei={"EntityCommonStockSharesOutstanding": dei_rows})
    n = normalize_facts(facts)
    return compute_metrics(n, compute_ratios(n), 26.35)


def test_newer_10q_cover_count_replaces_annual():
    # Oklo: 173.6M on the 10-K cover, 186.0M on the Aug 10-Q cover.
    m = _share_metrics([
        _row("2026-03-10", 173.6e6, fy=2025, filed="2026-03-12"),
        _row("2026-08-04", 186.0e6, fy=2026, fp="Q2", form="10-Q", filed="2026-08-06"),
    ])
    assert m["shares"] == 186.0e6
    assert m["shares_as_of"] == "2026-08-04"
    assert m["market_cap"] == 26.35 * 186.0e6


def test_out_of_band_cover_count_is_ignored():
    # A single class of a multi-class filer (half the annual count).
    m = _share_metrics([
        _row("2026-02-01", 12_230e6, fy=2025, filed="2026-02-04"),
        _row("2026-04-20", 5_800e6, fy=2026, fp="Q1", form="10-Q", filed="2026-04-25"),
    ])
    assert m["shares"] == 12_230e6
    assert m["shares_as_of"] == "2026-02-01"


def test_weighted_average_filer_uses_newer_quarter_of_same_tag():
    # CoreWeave: no dei cover count; FY2025 weighted average includes pre-IPO months.
    m = _share_metrics(usgaap_share_rows=[
        _row("2025-12-31", 436.0e6, fy=2025, start="2025-01-01"),
        _row("2026-06-30", 551.0e6, fy=2026, fp="Q2", form="10-Q", start="2026-04-01", filed="2026-08-10"),
    ])
    assert m["shares"] == 551.0e6
