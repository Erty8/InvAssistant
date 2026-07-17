"""Unit tests for sec_analyzer.normalize.normalizer.

All fixtures are small, hand-built dicts that mimic the shape of a real SEC
companyfacts document (see the module docstring of ``normalizer.py`` for the
shape). No network access or fixtures files are used -- everything is
constructed inline so these tests run instantly and without sec_analyzer's
config/http_client layers.
"""

from sec_analyzer.normalize.concepts import CONCEPTS
from sec_analyzer.normalize.normalizer import (
    format_table,
    latest_annual_value,
    normalize_facts,
    to_annual_series,
)


def _make_facts(usgaap=None, ifrs_full=None, entity_name="Test Co", cik=320193):
    """Build a minimal companyfacts-shaped dict for a test fixture."""
    facts = {}
    if usgaap is not None:
        facts["us-gaap"] = usgaap
    if ifrs_full is not None:
        facts["ifrs-full"] = ifrs_full
    return {"entityName": entity_name, "cik": cik, "facts": facts}


def _usd_tag(rows):
    """Wrap a list of fact rows in the ``{"units": {"USD": [...]}}`` shape."""
    return {"units": {"USD": rows}}


def test_fallback_tag_priority_uses_first_available_tag():
    """Revenue should fall back to `Revenues` when the preferred tag is absent."""
    usgaap = {
        "Revenues": _usd_tag(
            [
                {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 1_000_000,
                    "accn": "0000000000-23-000001",
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    # matched_tags values are now lists of contributing tags, ordered by
    # fallback priority. Only `Revenues` was present here.
    assert result["matched_tags"]["Revenue"] == ["Revenues"]
    assert result["annual"]["Revenue"] is not None
    assert result["annual"]["Revenue"][0]["value"] == 1_000_000
    assert result["annual"]["Revenue"][0]["tag"] == "Revenues"
    # The preferred tag was never present, so it must not appear anywhere.
    assert "RevenueFromContractWithCustomerExcludingAssessedTax" not in usgaap


def test_restatement_dedup_keeps_latest_filed_value():
    """Two rows for the same period, different `filed` dates -> latest wins."""
    usgaap = {
        "NetIncomeLoss": _usd_tag(
            [
                {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 500,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                },
                {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 550,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-06-15",  # a later restatement (e.g. in a 10-K/A)
                },
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    annual = result["annual"]["NetIncome"]
    assert annual is not None
    assert len(annual) == 1
    assert annual[0]["value"] == 550
    assert annual[0]["filed"] == "2023-06-15"


def test_depreciation_concept_extracted_as_annual_flow_series():
    """Depreciation (Package 2's new REIT-FFO concept) should extract from a
    cash-flow-statement D&A tag exactly like any other annual flow concept
    (Revenue, NetIncome): a full ~12-month span with a usable value yields
    one annual row, and the concept is classified as a "flow" (both
    ``start``/``end`` meaningful), not a "stock" snapshot.
    """
    usgaap = {
        "DepreciationAndAmortization": _usd_tag(
            [
                {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 250_000,
                    "accn": "0000000000-23-000001",
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    assert result["matched_tags"]["Depreciation"] == ["DepreciationAndAmortization"]
    annual = result["annual"]["Depreciation"]
    assert annual is not None
    assert annual[0]["value"] == 250_000
    assert annual[0]["fy"] == 2022


def test_depreciation_concept_fallback_tag_priority():
    """`DepreciationDepletionAndAmortization` (the broadest combined tag) is
    tried first; when absent, the normalizer falls back to
    `DepreciationAndAmortization`, exactly mirroring the Revenue fallback
    test above."""
    usgaap = {
        "DepreciationAndAmortization": _usd_tag(
            [
                {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 100_000,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    assert result["matched_tags"]["Depreciation"] == ["DepreciationAndAmortization"]
    assert "DepreciationDepletionAndAmortization" not in usgaap


def test_missing_concept_is_reported_without_raising():
    """A concept whose tags are entirely absent should land in `missing`."""
    usgaap = {
        "Assets": _usd_tag(
            [
                {
                    "end": "2022-12-31",
                    "val": 12345,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ]
        ),
        # Note: no "Liabilities" key at all.
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    assert "TotalLiabilities" in result["missing"]
    assert result["annual"]["TotalLiabilities"] is None
    assert result["quarterly"]["TotalLiabilities"] is None
    assert result["matched_tags"]["TotalLiabilities"] is None
    # The concept that *was* present should still be extracted normally.
    assert result["annual"]["TotalAssets"] is not None


def test_ifrs_full_only_filer_has_no_crash_and_all_concepts_missing():
    """A 20-F filer with only `ifrs-full` facts should degrade gracefully."""
    ifrs_full = {
        "Assets": _usd_tag(
            [{"end": "2022-12-31", "val": 999, "fy": 2022, "fp": "FY", "form": "20-F", "filed": "2023-04-01"}]
        ),
    }
    result = normalize_facts(_make_facts(ifrs_full=ifrs_full))

    assert set(result["missing"]) == set(CONCEPTS.keys())
    for concept in CONCEPTS:
        assert result["annual"][concept] is None
        assert result["quarterly"][concept] is None
        assert result["matched_tags"][concept] is None


def test_no_facts_at_all_does_not_raise():
    """A completely empty `facts` dict must not crash normalize_facts."""
    result = normalize_facts(_make_facts())

    assert set(result["missing"]) == set(CONCEPTS.keys())
    assert result["entity_name"] == "Test Co"


def test_annual_and_quarterly_split_for_point_in_time_concept():
    """A 10-K FY row and a 10-Q row for the same tag land in different buckets."""
    usgaap = {
        "Assets": _usd_tag(
            [
                {"end": "2022-12-31", "val": 100, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
                {"end": "2022-09-30", "val": 90, "fy": 2022, "fp": "Q3", "form": "10-Q", "filed": "2022-11-01"},
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    annual = result["annual"]["TotalAssets"]
    quarterly = result["quarterly"]["TotalAssets"]

    assert annual is not None and len(annual) == 1
    assert annual[0]["value"] == 100
    assert annual[0]["form"] == "10-K"

    assert quarterly is not None and len(quarterly) == 1
    assert quarterly[0]["value"] == 90
    assert quarterly[0]["form"] == "10-Q"


def test_flow_concept_annual_bucket_rejects_stray_quarter_in_10k():
    """A 10-K/FY row for a flow concept spanning ~1 quarter should be excluded."""
    usgaap = {
        "NetIncomeLoss": _usd_tag(
            [
                {
                    # Mislabeled/stray quarter-length period inside a 10-K.
                    "start": "2022-10-01",
                    "end": "2022-12-31",
                    "val": 111,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                },
                {
                    # The genuine full-year figure.
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 444,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                },
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    annual = result["annual"]["NetIncome"]
    assert annual is not None
    assert len(annual) == 1
    assert annual[0]["value"] == 444


def test_quarterly_flow_concept_prefers_quarter_length_span_over_ytd():
    """When a 10-Q reports both a quarter and a YTD figure for the same end
    date, the quarter-length (~80-100 day) span should be preferred."""
    usgaap = {
        "NetIncomeLoss": _usd_tag(
            [
                {
                    # Year-to-date (Q1+Q2+Q3), ~270 days.
                    "start": "2022-01-01",
                    "end": "2022-09-30",
                    "val": 900,
                    "fy": 2022,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2022-11-01",
                },
                {
                    # Quarter-only figure, ~91 days.
                    "start": "2022-07-01",
                    "end": "2022-09-30",
                    "val": 300,
                    "fy": 2022,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2022-11-01",
                },
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    quarterly = result["quarterly"]["NetIncome"]
    assert quarterly is not None
    assert len(quarterly) == 1
    assert quarterly[0]["value"] == 300


def test_to_annual_series_and_latest_annual_value():
    usgaap = {
        "Assets": _usd_tag(
            [
                {"end": "2021-12-31", "val": 100, "fy": 2021, "fp": "FY", "form": "10-K", "filed": "2022-02-01"},
                {"end": "2022-12-31", "val": 200, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
            ]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    series = to_annual_series(result, "TotalAssets")
    assert series == {2021: 100, 2022: 200}
    assert latest_annual_value(result, "TotalAssets") == 200

    # Helpers must handle missing concepts gracefully rather than raising.
    assert to_annual_series(result, "TotalLiabilities") == {}
    assert latest_annual_value(result, "TotalLiabilities") is None


def test_years_window_limits_distinct_fiscal_years():
    rows = []
    for i, year in enumerate(range(2015, 2023)):
        rows.append(
            {
                "end": f"{year}-12-31",
                "val": i,
                "fy": year,
                "fp": "FY",
                "form": "10-K",
                "filed": f"{year + 1}-02-01",
            }
        )
    usgaap = {"Assets": _usd_tag(rows)}
    result = normalize_facts(_make_facts(usgaap=usgaap), years=3)

    annual = result["annual"]["TotalAssets"]
    assert annual is not None
    assert len(annual) == 3
    assert [r["fy"] for r in annual] == [2022, 2021, 2020]


def test_comparative_columns_labeled_by_period_not_filing_fy():
    """A single 10-K carries prior-year comparative columns all stamped with
    the *filing's* fy/fp. The derived fiscal year must come from each row's
    period_end, not that shared fy, so the comparatives don't collapse or
    mislabel.

    This mirrors Apple's FY2025 10-K: one filing, one filed date, three USD
    rows for Revenue/NetIncome, every row stamped fy=2025/fp=FY/form=10-K but
    with distinct period-end dates.
    """
    filed = "2025-10-31"
    rev_rows = [
        {"start": "2024-09-29", "end": "2025-09-27", "val": 416_161_000_000,
         "fy": 2025, "fp": "FY", "form": "10-K", "filed": filed},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 391_035_000_000,
         "fy": 2025, "fp": "FY", "form": "10-K", "filed": filed},
        {"start": "2022-09-25", "end": "2023-09-30", "val": 383_285_000_000,
         "fy": 2025, "fp": "FY", "form": "10-K", "filed": filed},
    ]
    ni_rows = [
        {"start": "2024-09-29", "end": "2025-09-27", "val": 100_000_000_000,
         "fy": 2025, "fp": "FY", "form": "10-K", "filed": filed},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 93_736_000_000,
         "fy": 2025, "fp": "FY", "form": "10-K", "filed": filed},
        {"start": "2022-09-25", "end": "2023-09-30", "val": 96_995_000_000,
         "fy": 2025, "fp": "FY", "form": "10-K", "filed": filed},
    ]
    usgaap = {
        "Revenues": _usd_tag(rev_rows),
        "NetIncomeLoss": _usd_tag(ni_rows),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap, entity_name="Apple Inc."), years=5)

    annual_rev = result["annual"]["Revenue"]
    assert annual_rev is not None
    # Three distinct periods survive -- they must NOT collapse into one fy.
    assert len(annual_rev) == 3
    # Derived fiscal years come from period_end, sorted descending.
    assert [r["fy"] for r in annual_rev] == [2025, 2024, 2023]
    # The original SEC fy is preserved verbatim on every record.
    assert all(r["reported_fy"] == 2025 for r in annual_rev)

    # to_annual_series must key by the corrected fy, mapping each period to
    # its own value (proving no shift/duplication).
    rev_series = to_annual_series(result, "Revenue")
    assert rev_series == {
        2025: 416_161_000_000,
        2024: 391_035_000_000,
        2023: 383_285_000_000,
    }
    ni_series = to_annual_series(result, "NetIncome")
    assert ni_series == {
        2025: 100_000_000_000,
        2024: 93_736_000_000,
        2023: 96_995_000_000,
    }


def test_concept_split_across_tags_over_time():
    """A filer can report one concept under different tags across years.

    This mirrors NVIDIA's Revenue: older years under the priority-0 tag
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` and recent years
    under the priority-1 tag ``Revenues``, with neither tag covering all
    periods. The normalizer must MERGE rows across both tags so no year is
    blank, and report both contributing tags (priority-ordered) in
    ``matched_tags``.
    """
    old_tag_rows = [
        {"start": "2021-02-01", "end": "2022-01-30", "val": 26_914,
         "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-02-18"},
        {"start": "2020-02-03", "end": "2021-01-31", "val": 16_675,
         "fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-02-26"},
    ]
    new_tag_rows = [
        {"start": "2022-01-31", "end": "2023-01-29", "val": 26_974,
         "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-02-24"},
        {"start": "2023-01-30", "end": "2024-01-28", "val": 60_922,
         "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-02-21"},
        {"start": "2024-01-29", "end": "2025-01-26", "val": 130_497,
         "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-02-26"},
    ]
    usgaap = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": _usd_tag(old_tag_rows),
        "Revenues": _usd_tag(new_tag_rows),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap, entity_name="NVIDIA Corp"), years=6)

    # All five years present, none blank, sourced from BOTH tags.
    rev_series = to_annual_series(result, "Revenue")
    assert rev_series == {
        2021: 16_675,
        2022: 26_914,
        2023: 26_974,
        2024: 60_922,
        2025: 130_497,
    }

    annual_rev = result["annual"]["Revenue"]
    assert [r["fy"] for r in annual_rev] == [2025, 2024, 2023, 2022, 2021]
    # Each record keeps the specific tag its value came from.
    by_fy = {r["fy"]: r for r in annual_rev}
    assert by_fy[2021]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert by_fy[2022]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert by_fy[2025]["tag"] == "Revenues"

    # matched_tags is the priority-ordered list of contributing tags.
    assert result["matched_tags"]["Revenue"] == [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ]
    assert "Revenue" not in result["missing"]


def test_higher_priority_tag_wins_for_same_period():
    """When two fallback tags report the SAME period_end, the higher-priority
    tag's value wins; within one tag, the latest `filed` still wins."""
    high_priority_rows = [
        # priority-0 tag (RevenueFromContract...): two filings, same period.
        {"start": "2021-01-01", "end": "2021-12-31", "val": 1000,
         "fy": 2021, "fp": "FY", "form": "10-K", "filed": "2022-02-01"},
        {"start": "2021-01-01", "end": "2021-12-31", "val": 1010,  # later restatement
         "fy": 2021, "fp": "FY", "form": "10-K", "filed": "2022-08-01"},
    ]
    low_priority_rows = [
        # priority-1 tag (Revenues): same period, filed even later, but must
        # NOT win because its tag is lower priority.
        {"start": "2021-01-01", "end": "2021-12-31", "val": 9999,
         "fy": 2021, "fp": "FY", "form": "10-K", "filed": "2023-01-01"},
    ]
    usgaap = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": _usd_tag(high_priority_rows),
        "Revenues": _usd_tag(low_priority_rows),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap), years=5)

    annual_rev = result["annual"]["Revenue"]
    assert annual_rev is not None and len(annual_rev) == 1
    # Higher-priority tag wins over lower-priority even though the latter was
    # filed later; within the winning tag, the latest filing (1010) wins.
    assert annual_rev[0]["value"] == 1010
    assert annual_rev[0]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    # Only the winning tag contributed to the surviving record.
    assert result["matched_tags"]["Revenue"] == [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ]


def test_format_table_smoke_test_does_not_raise():
    usgaap = {
        "Assets": _usd_tag(
            [{"end": "2022-12-31", "val": 123456789, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"}]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap, entity_name="Acme Corp"))

    table = format_table(result)
    assert isinstance(table, str)
    assert "Acme Corp" in table
    assert "FY2022" in table


def test_format_table_with_no_annual_data_does_not_raise():
    result = normalize_facts(_make_facts(entity_name="Empty Co"))
    table = format_table(result)
    assert isinstance(table, str)
    assert "Empty Co" in table


def _unit_tag(unit_key, rows):
    """Wrap a list of fact rows in the ``{"units": {<unit_key>: [...]}}``
    shape -- like ``_usd_tag`` but for non-USD unit keys (e.g. per-share
    concepts reported under ``"USD/shares"`` or ``"shares"``)."""
    return {"units": {unit_key: rows}}


def test_eps_concept_extracted_from_usd_per_shares_unit():
    """EPS is reported under the `USD/shares` unit key, not `USD` -- the
    normalizer must consult `concepts.CONCEPT_UNITS` to find it."""
    usgaap = {
        "EarningsPerShareDiluted": _unit_tag(
            "USD/shares",
            [
                {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 6.13,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ],
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    annual = result["annual"]["EPS"]
    assert annual is not None
    assert annual[0]["value"] == 6.13
    assert annual[0]["unit"] == "USD/shares"
    assert "EPS" not in result["missing"]


def test_shares_outstanding_extracted_from_shares_unit():
    """SharesOutstanding is reported under the `shares` unit key."""
    usgaap = {
        "CommonStockSharesOutstanding": _unit_tag(
            "shares",
            [
                {
                    "end": "2022-12-31",
                    "val": 15_000_000_000,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ],
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    annual = result["annual"]["SharesOutstanding"]
    assert annual is not None
    assert annual[0]["value"] == 15_000_000_000
    assert annual[0]["unit"] == "shares"


def test_usd_concept_still_extracted_and_tagged_with_usd_unit():
    """A plain monetary concept (default unit_keys == ["USD"]) should still
    extract normally and carry `unit == "USD"` on its records."""
    usgaap = {
        "Assets": _usd_tag(
            [{"end": "2022-12-31", "val": 500, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"}]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    annual = result["annual"]["TotalAssets"]
    assert annual is not None
    assert annual[0]["value"] == 500
    assert annual[0]["unit"] == "USD"


def test_shares_outstanding_prefers_dei_cover_page_tag():
    """`EntityCommonStockSharesOutstanding` is a dei-taxonomy tag (not
    us-gaap) -- see `concepts.TAG_TAXONOMY`. It's the highest-priority
    fallback for SharesOutstanding, and must be looked up in
    `facts["dei"]`, not `facts["us-gaap"]`."""
    usgaap = {
        "Revenues": _usd_tag(
            [
                {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                    "val": 1000,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ]
        ),
    }
    dei = {
        "EntityCommonStockSharesOutstanding": {
            "units": {
                "shares": [
                    {
                        "end": "2022-12-31",
                        "val": 12_000_000,
                        "fy": 2022,
                        "fp": "FY",
                        "form": "10-K",
                        "filed": "2023-02-01",
                    }
                ]
            }
        },
    }
    facts_json = {
        "entityName": "Dei Test Co",
        "cik": 999,
        "facts": {"us-gaap": usgaap, "dei": dei},
    }
    result = normalize_facts(facts_json)

    annual = result["annual"]["SharesOutstanding"]
    assert annual is not None
    assert len(annual) == 1
    assert annual[0]["value"] == 12_000_000
    assert annual[0]["unit"] == "shares"
    assert annual[0]["tag"] == "EntityCommonStockSharesOutstanding"
    assert result["matched_tags"]["SharesOutstanding"] == ["EntityCommonStockSharesOutstanding"]

    # A concept whose tags are all us-gaap (Revenue) is unaffected by the
    # presence of a `dei` taxonomy sub-dict elsewhere in `facts`.
    assert result["annual"]["Revenue"] is not None
    assert result["annual"]["Revenue"][0]["value"] == 1000


def test_dei_tag_absent_falls_back_to_usgaap_shares_tag():
    """When the dei cover-page tag isn't present at all, SharesOutstanding
    must still fall back to the us-gaap tags, exactly as before this
    taxonomy support was added."""
    usgaap = {
        "CommonStockSharesOutstanding": _unit_tag(
            "shares",
            [
                {
                    "end": "2022-12-31",
                    "val": 15_000_000_000,
                    "fy": 2022,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2023-02-01",
                }
            ],
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    annual = result["annual"]["SharesOutstanding"]
    assert annual is not None
    assert annual[0]["value"] == 15_000_000_000
    assert annual[0]["tag"] == "CommonStockSharesOutstanding"


def test_concept_with_tag_present_but_wrong_unit_is_missing():
    """A tag that exists but only carries units outside the concept's
    acceptable list must not contribute any rows -- the concept is reported
    as missing rather than picking up an incompatible unit."""
    usgaap = {
        # EPS only accepts "USD/shares" (see CONCEPT_UNITS); here the tag is
        # present but mistagged with a plain "USD" unit.
        "EarningsPerShareDiluted": _usd_tag(
            [{"end": "2022-12-31", "val": 6.13, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"}]
        ),
    }
    result = normalize_facts(_make_facts(usgaap=usgaap))

    assert result["annual"]["EPS"] is None
    assert result["quarterly"]["EPS"] is None
    assert "EPS" in result["missing"]
