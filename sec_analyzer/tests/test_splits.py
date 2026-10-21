"""Unit tests for share-split detection and basis normalization (SPEC.md Sec.29).

Fixtures are hand-built companyfacts-shaped dicts modelled on Netflix's
10-for-1 split of November 2025: filings before the split report pre-split
share counts and EPS, the first post-split 10-K restates its comparative
years, and the cover-page count is never restated.
"""

import pytest

from sec_analyzer.normalize import splits
from sec_analyzer.normalize.normalizer import normalize_facts, to_annual_series, to_quarterly_series

_DILUTED = "WeightedAverageNumberOfDilutedSharesOutstanding"


def _annual(fy, val, accn, filed):
    return {"start": f"{fy}-01-01", "end": f"{fy}-12-31", "val": val, "accn": accn,
            "fy": fy, "fp": "FY", "form": "10-K", "filed": filed}


def _quarter(start, end, val, accn, filed, fp="Q3"):
    return {"start": start, "end": end, "val": val, "accn": accn,
            "fy": int(end[:4]), "fp": fp, "form": "10-Q", "filed": filed}


def _cover(end, val, accn, filed, form="10-K", fp="FY"):
    return {"end": end, "val": val, "accn": accn, "fy": int(end[:4]), "fp": fp, "form": form, "filed": filed}


def _facts(diluted, cover, eps):
    return {
        "entityName": "Split Co",
        "cik": 1,
        "facts": {
            "us-gaap": {
                _DILUTED: {"units": {"shares": diluted}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
            },
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover}}},
        },
    }


def _ten_for_one_facts():
    """FY2023/FY2024 10-Ks and a Q3'25 10-Q pre-split, FY2025 10-K post-split."""
    diluted = [
        _annual(2022, 451_290_000, "A23", "2024-01-26"),
        _annual(2023, 449_498_000, "A23", "2024-01-26"),
        _annual(2022, 451_290_000, "A24", "2025-01-27"),
        _annual(2023, 449_498_000, "A24", "2025-01-27"),
        _annual(2024, 439_261_000, "A24", "2025-01-27"),
        _quarter("2025-07-01", "2025-09-30", 434_000_000, "Q325", "2025-10-22"),
        _annual(2023, 4_494_980_000, "A25", "2026-01-23"),
        _annual(2024, 4_392_610_000, "A25", "2026-01-23"),
        _annual(2025, 4_343_863_000, "A25", "2026-01-23"),
    ]
    cover = [
        _cover("2023-12-31", 433_000_000, "A23", "2024-01-26"),
        _cover("2024-12-31", 427_757_100, "A24", "2025-01-27"),
        _cover("2025-09-30", 423_732_334, "Q325", "2025-10-22", form="10-Q", fp="Q3"),
        _cover("2025-12-31", 4_222_162_150, "A25", "2026-01-23"),
    ]
    eps = [
        _annual(2022, 9.95, "A24", "2025-01-27"),
        _annual(2023, 12.03, "A24", "2025-01-27"),
        _annual(2024, 19.83, "A24", "2025-01-27"),
        _annual(2023, 1.20, "A25", "2026-01-23"),
        _annual(2024, 1.98, "A25", "2026-01-23"),
        _annual(2025, 2.53, "A25", "2026-01-23"),
    ]
    return _facts(diluted, cover, eps)


def test_forward_split_moves_unrestated_share_counts_and_eps_onto_new_basis():
    result = normalize_facts(_ten_for_one_facts())

    shares = to_annual_series(result, "SharesOutstanding")
    assert shares[2023] == pytest.approx(4_330_000_000)
    assert shares[2024] == pytest.approx(4_277_571_000)
    assert shares[2025] == pytest.approx(4_222_162_150)  # already post-split, untouched

    eps = to_annual_series(result, "EPS")
    assert eps[2022] == pytest.approx(0.995)  # never re-filed after the split
    assert eps[2023] == pytest.approx(1.20)   # the post-split restatement wins as before
    assert eps[2025] == pytest.approx(2.53)


def test_detected_event_is_reported_with_its_basis_window():
    result = normalize_facts(_ten_for_one_facts())

    assert result["share_splits"] == [{
        "factor": 10.0,
        "old_basis_until": "2025-01-27",
        "new_basis_from": "2026-01-23",
        "corroborating_periods": 2,
    }]


def test_adjusted_record_keeps_reported_value_and_factor():
    result = normalize_facts(_ten_for_one_facts())

    fy2024 = next(r for r in result["annual"]["SharesOutstanding"] if r["fy"] == 2024)
    assert fy2024["reported_value"] == 427_757_100
    assert fy2024["split_factor"] == 10.0
    fy2025 = next(r for r in result["annual"]["SharesOutstanding"] if r["fy"] == 2025)
    assert "split_factor" not in fy2025 and "reported_value" not in fy2025
    assert all("_accn" not in r for r in result["annual"]["SharesOutstanding"])


def test_filing_inside_the_window_is_placed_by_its_share_level():
    # The Q3'25 10-Q (filed 2025-10-22) falls between the last old-basis and
    # the first new-basis filing; its 423.7M cover count is near the old
    # basis, so its figures are scaled up. (The quarterly bucket keeps the
    # quarter-length weighted-average row, 434.0M, for this period.)
    result = normalize_facts(_ten_for_one_facts())

    quarters = {r["period_end"]: r for r in to_quarterly_series(result, "SharesOutstanding")}
    assert quarters["2025-09-30"]["value"] == pytest.approx(4_340_000_000)


def test_share_jump_without_restatement_is_not_a_split():
    # A stock-for-stock merger doubles the share count, but no past period is
    # ever re-reported at twice its value.
    diluted = [
        _annual(2023, 100_000_000, "A23", "2024-02-01"),
        _annual(2023, 100_000_000, "A24", "2025-02-01"),
        _annual(2024, 200_000_000, "A24", "2025-02-01"),
    ]
    cover = [
        _cover("2023-12-31", 100_000_000, "A23", "2024-02-01"),
        _cover("2024-12-31", 200_000_000, "A24", "2025-02-01"),
    ]
    result = normalize_facts(_facts(diluted, cover, []))

    assert result["share_splits"] == []
    assert to_annual_series(result, "SharesOutstanding") == {2023: 100_000_000, 2024: 200_000_000}


def test_single_restated_period_is_not_enough():
    diluted = [
        _annual(2023, 100_000_000, "A23", "2024-02-01"),
        _annual(2023, 200_000_000, "A24", "2025-02-01"),
    ]
    result = normalize_facts(_facts(diluted, [], []))

    assert result["share_splits"] == []


def test_restatement_by_a_non_split_ratio_is_ignored():
    diluted = [
        _annual(2022, 100_000_000, "A23", "2024-02-01"),
        _annual(2023, 100_000_000, "A23", "2024-02-01"),
        _annual(2022, 107_000_000, "A24", "2025-02-01"),
        _annual(2023, 107_000_000, "A24", "2025-02-01"),
    ]
    result = normalize_facts(_facts(diluted, [], []))

    assert result["share_splits"] == []


def test_reverse_split_divides_old_share_counts():
    diluted = [
        _annual(2022, 1_000_000_000, "A23", "2024-02-01"),
        _annual(2023, 1_000_000_000, "A23", "2024-02-01"),
        _annual(2022, 100_000_000, "A24", "2025-02-01"),
        _annual(2023, 100_000_000, "A24", "2025-02-01"),
        _annual(2024, 100_000_000, "A24", "2025-02-01"),
    ]
    cover = [
        _cover("2023-12-31", 1_000_000_000, "A23", "2024-02-01"),
        _cover("2024-12-31", 100_000_000, "A24", "2025-02-01"),
    ]
    eps = [_annual(2022, -0.05, "A23", "2024-02-01")]
    result = normalize_facts(_facts(diluted, cover, eps))

    assert result["share_splits"][0]["factor"] == pytest.approx(0.1)
    assert to_annual_series(result, "SharesOutstanding")[2023] == pytest.approx(100_000_000)
    assert to_annual_series(result, "EPS")[2022] == pytest.approx(-0.5)


def test_two_splits_seen_through_one_restatement_are_not_a_third_split():
    # 2:1 splits in late 2022 and mid 2023. FY2020/FY2021 were last filed
    # before the first split and re-filed only after the second, so they
    # jump 4x in one step -- that must not become a separate 4:1 event.
    diluted = [
        # FY2020 / FY2021: filed 2022-02 (old), re-filed 2023-09 (x4).
        _annual(2020, 100_000_000, "K22", "2022-02-01"),
        _annual(2021, 100_000_000, "K22", "2022-02-01"),
        _annual(2020, 400_000_000, "K23", "2023-09-01"),
        _annual(2021, 400_000_000, "K23", "2023-09-01"),
        # Quarters that pin each split separately.
        _quarter("2022-01-01", "2022-03-31", 100_000_000, "Q122", "2022-05-01", fp="Q1"),
        _quarter("2022-04-01", "2022-06-30", 100_000_000, "Q222", "2022-08-01", fp="Q2"),
        _quarter("2022-01-01", "2022-03-31", 200_000_000, "Q123", "2023-01-15", fp="Q1"),
        _quarter("2022-04-01", "2022-06-30", 200_000_000, "Q123", "2023-01-15", fp="Q1"),
        _quarter("2022-10-01", "2022-12-31", 200_000_000, "Q123", "2023-01-15", fp="Q1"),
        _quarter("2023-01-01", "2023-03-31", 200_000_000, "Q223", "2023-05-01", fp="Q2"),
        _quarter("2022-10-01", "2022-12-31", 400_000_000, "K23", "2023-09-01", fp="FY"),
        _quarter("2023-01-01", "2023-03-31", 400_000_000, "K23", "2023-09-01", fp="FY"),
    ]
    cover = [_cover("2021-12-31", 100_000_000, "K22", "2022-02-01")]
    result = normalize_facts(_facts(diluted, cover, []))

    assert [e["factor"] for e in result["share_splits"]] == [2.0, 2.0]
    assert to_annual_series(result, "SharesOutstanding")[2021] == pytest.approx(400_000_000)


def test_as_of_before_the_restating_filing_sees_no_split():
    result = normalize_facts(_ten_for_one_facts(), as_of="2025-12-31")

    assert result["share_splits"] == []
    assert to_annual_series(result, "SharesOutstanding")[2024] == 427_757_100


def test_same_inputs_give_same_outputs():
    assert normalize_facts(_ten_for_one_facts()) == normalize_facts(_ten_for_one_facts())


def test_filer_without_split_is_unchanged():
    facts = _ten_for_one_facts()
    facts["facts"]["us-gaap"][_DILUTED]["units"]["shares"] = [
        r for r in facts["facts"]["us-gaap"][_DILUTED]["units"]["shares"] if r["accn"] != "A25"
    ]
    result = normalize_facts(facts)

    assert result["share_splits"] == []
    assert all("split_factor" not in r for r in result["annual"]["SharesOutstanding"])


def test_malformed_facts_never_raise():
    garbage = {"us-gaap": {_DILUTED: {"units": {"shares": [{"val": "x"}, {"filed": None}, {}]}}}}
    assert splits.detect_share_splits(garbage) is None
    assert splits.detect_share_splits({}) is None
    assert splits.detect_share_splits(None) is None
