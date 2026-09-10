"""Unit tests for the financial-filer net-revenue basis (SPEC.md Sec.19).

No network access -- these call ``normalizer._apply_net_revenue_basis``
directly on synthetic ``annual``/``quarterly`` buckets shaped exactly like
``normalize_facts`` produces, plus one end-to-end pass through
``normalize_facts`` on a synthetic companyfacts document to prove the wiring.

The headline case is real: SoFi (CIK 1818874) reports FY2025 contract
revenue of $0.619B under the tag ``CONCEPTS["Revenue"]`` prefers, against
$3.613B of total net revenue under ``RevenuesNetOfInterestExpense``.
"""

import pytest

from sec_analyzer.normalize.normalizer import (
    _apply_net_revenue_basis,
    normalize_facts,
    to_annual_series,
)

# SoFi's real reported figures, in USD.
_SOFI_REPORTED = {2022: 0.377e9, 2023: 0.421e9, 2024: 0.503e9, 2025: 0.619e9}
_SOFI_NET = {2022: 1.574e9, 2023: 2.123e9, 2024: 2.675e9, 2025: 3.613e9}


def _record(fy, value, tag="SomeTag"):
    """A minimal annual record, matching normalize_facts' record shape."""
    return {
        "concept": None,
        "tag": tag,
        "period_end": f"{fy}-12-31",
        "fy": fy,
        "fp": "FY",
        "form": "10-K",
        "value": value,
        "filed": f"{fy + 1}-02-20",
        "start": f"{fy}-01-01",
    }


def _records(series, tag="SomeTag"):
    return [_record(fy, value, tag) for fy, value in sorted(series.items(), reverse=True)]


def _buckets(reported=None, net=None, interest=None, noninterest=None, quarterly=None):
    """Build the four mutable structures ``_apply_net_revenue_basis`` takes."""
    annual = {}
    if reported is not None:
        annual["Revenue"] = _records(reported, tag="RevenueFromContractWithCustomerExcludingAssessedTax")
    if net is not None:
        annual["NetRevenue"] = _records(net, tag="RevenuesNetOfInterestExpense")
    if interest is not None:
        annual["InterestIncome"] = _records(interest)
    if noninterest is not None:
        annual["NoninterestIncome"] = _records(noninterest)

    quarterly = dict(quarterly or {})
    matched_tags = {
        "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax"],
        "NetRevenue": ["RevenuesNetOfInterestExpense"],
    }
    missing = []
    return annual, quarterly, matched_tags, missing


# ---------------------------------------------------------------------------
# The headline case: a lender whose as-reported tag is a fee-only slice.
# ---------------------------------------------------------------------------


def test_sofi_shape_swaps_to_net_revenue():
    annual, quarterly, matched_tags, missing = _buckets(
        reported=_SOFI_REPORTED, net=_SOFI_NET
    )

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["basis"] == "net_revenue"
    assert result["swapped"] is True
    # Hand-computed: abs(0.619 - 3.613) / 3.613 = 0.82867... -> 0.8287 is
    # FY2025's, but FY2022's abs(0.377 - 1.574) / 1.574 = 0.76049 is smaller,
    # so the max across the four overlapping years is FY2024's
    # abs(0.503 - 2.675) / 2.675 = 0.81196... Recompute explicitly:
    expected = max(
        abs(_SOFI_REPORTED[fy] - _SOFI_NET[fy]) / abs(_SOFI_NET[fy])
        for fy in _SOFI_NET
    )
    assert result["max_divergence"] == pytest.approx(round(expected, 4))
    assert result["divergent_fys"] == [2022, 2023, 2024, 2025]
    assert result["dropped_fys"] == []

    swapped = {r["fy"]: r["value"] for r in annual["Revenue"]}
    assert swapped == _SOFI_NET
    assert result["rejected_annual"] == _SOFI_REPORTED
    assert result["rejected_tags"] == ["RevenueFromContractWithCustomerExcludingAssessedTax"]
    assert matched_tags["Revenue"] == ["RevenuesNetOfInterestExpense"]


def test_swap_note_is_turkish_and_names_the_worst_year():
    annual, quarterly, matched_tags, missing = _buckets(
        reported=_SOFI_REPORTED, net=_SOFI_NET
    )

    note = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)["note"]

    assert note
    worst_fy = max(
        _SOFI_NET,
        key=lambda fy: abs(_SOFI_REPORTED[fy] - _SOFI_NET[fy]) / abs(_SOFI_NET[fy]),
    )
    assert worst_fy == 2025
    assert f"FY{worst_fy}" in note
    assert "net gelir" in note
    # Turkish decimal comma, not an English decimal point: 82.9% -> "%82,9".
    assert "%82,9" in note
    assert "%82.9" not in note


# ---------------------------------------------------------------------------
# Threshold behavior.
# ---------------------------------------------------------------------------


def test_agreement_within_tolerance_does_not_swap():
    # 3% apart: the two tags describe the same thing, keep as-reported.
    net = {2024: 1_000.0, 2025: 2_000.0}
    reported = {2024: 1_030.0, 2025: 2_060.0}
    annual, quarterly, matched_tags, missing = _buckets(reported=reported, net=net)

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["basis"] == "as_reported"
    assert result["swapped"] is False
    assert result["max_divergence"] == pytest.approx(0.03)
    assert result["divergent_fys"] == []
    assert result["rejected_annual"] is None
    assert {r["fy"]: r["value"] for r in annual["Revenue"]} == reported


def test_exactly_five_percent_divergence_does_not_swap():
    # The threshold is strictly-above, so exactly 5.0% stays as-reported.
    net = {2025: 1_000.0}
    reported = {2025: 1_050.0}
    annual, quarterly, matched_tags, missing = _buckets(reported=reported, net=net)

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["max_divergence"] == pytest.approx(0.05)
    assert result["swapped"] is False
    assert {r["fy"]: r["value"] for r in annual["Revenue"]} == reported


def test_just_above_five_percent_divergence_swaps():
    net = {2025: 1_000.0}
    reported = {2025: 1_051.0}
    annual, quarterly, matched_tags, missing = _buckets(reported=reported, net=net)

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["swapped"] is True
    assert {r["fy"]: r["value"] for r in annual["Revenue"]} == net


def test_understatement_and_overstatement_both_trigger():
    """Divergence is absolute: the SoFi under-statement and an aggregator-style
    gross over-statement must both be caught."""
    net = {2025: 1_000.0}

    under_annual, q1, t1, m1 = _buckets(reported={2025: 200.0}, net=net)
    over_annual, q2, t2, m2 = _buckets(reported={2025: 1_800.0}, net=net)

    assert _apply_net_revenue_basis(under_annual, q1, t1, m1)["swapped"] is True
    assert _apply_net_revenue_basis(over_annual, q2, t2, m2)["swapped"] is True


# ---------------------------------------------------------------------------
# The ordinary non-financial filer, and the missing-input edges.
# ---------------------------------------------------------------------------


def test_no_net_revenue_tag_leaves_everything_untouched():
    reported = {2024: 500.0, 2025: 600.0}
    annual, quarterly, matched_tags, missing = _buckets(reported=reported)
    before = [dict(r) for r in annual["Revenue"]]

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["basis"] == "as_reported"
    assert result["swapped"] is False
    assert result["max_divergence"] is None
    assert result["divergent_fys"] == []
    assert result["dropped_fys"] == []
    assert result["rejected_annual"] is None
    assert result["gross_annual"] is None
    assert result["note"] is None
    assert annual["Revenue"] == before
    assert matched_tags["Revenue"] == ["RevenueFromContractWithCustomerExcludingAssessedTax"]


def test_net_revenue_present_but_revenue_absent_swaps():
    net = {2024: 1_000.0, 2025: 1_200.0}
    annual, quarterly, matched_tags, missing = _buckets(net=net)
    missing.append("Revenue")

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["swapped"] is True
    assert result["max_divergence"] is None  # empty overlap
    assert {r["fy"]: r["value"] for r in annual["Revenue"]} == net
    assert "Revenue" not in missing


def test_zero_net_revenue_year_is_excluded_from_the_overlap():
    # A zero net figure cannot be a divergence denominator; it must not raise
    # and must not count as an agreeing year.
    net = {2024: 0.0, 2025: 1_000.0}
    reported = {2024: 500.0, 2025: 1_000.0}
    annual, quarterly, matched_tags, missing = _buckets(reported=reported, net=net)

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["max_divergence"] == pytest.approx(0.0)
    assert result["swapped"] is False


# ---------------------------------------------------------------------------
# No mixed basis.
# ---------------------------------------------------------------------------


def test_years_without_a_net_counterpart_are_dropped_not_spliced():
    net = {2024: 2_000.0, 2025: 3_000.0}
    reported = {2021: 100.0, 2022: 200.0, 2024: 300.0, 2025: 400.0}
    annual, quarterly, matched_tags, missing = _buckets(reported=reported, net=net)

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["swapped"] is True
    assert result["dropped_fys"] == [2021, 2022]
    survivors = {r["fy"]: r["value"] for r in annual["Revenue"]}
    assert survivors == net
    # No as-reported value survived anywhere in the series.
    assert not set(survivors.values()) & set(reported.values())


def test_quarterly_revenue_is_cleared_when_net_has_no_quarterly_rows():
    annual, quarterly, matched_tags, missing = _buckets(
        reported=_SOFI_REPORTED,
        net=_SOFI_NET,
        quarterly={"Revenue": _records({2025: 150.0}), "NetRevenue": None},
    )

    _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert quarterly["Revenue"] is None


def test_quarterly_revenue_is_replaced_when_net_has_quarterly_rows():
    net_q = _records({2025: 900.0}, tag="RevenuesNetOfInterestExpense")
    annual, quarterly, matched_tags, missing = _buckets(
        reported=_SOFI_REPORTED,
        net=_SOFI_NET,
        quarterly={"Revenue": _records({2025: 150.0}), "NetRevenue": net_q},
    )

    _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert [r["value"] for r in quarterly["Revenue"]] == [900.0]


def test_swapped_series_is_a_copy_not_an_alias():
    annual, quarterly, matched_tags, missing = _buckets(
        reported=_SOFI_REPORTED, net=_SOFI_NET
    )

    _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)
    annual["Revenue"][0]["value"] = -1.0

    assert annual["NetRevenue"][0]["value"] == _SOFI_NET[2025]
    assert annual["Revenue"] is not annual["NetRevenue"]


# ---------------------------------------------------------------------------
# gross_annual is informational only.
# ---------------------------------------------------------------------------


def test_gross_annual_sums_interest_and_noninterest_income():
    annual, quarterly, matched_tags, missing = _buckets(
        reported=_SOFI_REPORTED,
        net=_SOFI_NET,
        interest={2024: 2.808e9, 2025: 3.375e9},
        noninterest={2024: 0.958e9, 2025: 1.394e9},
    )

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["gross_annual"] == {
        2024: pytest.approx(3.766e9),
        2025: pytest.approx(4.769e9),
    }
    # It is never used as the revenue basis -- the swap took net revenue.
    assert {r["fy"]: r["value"] for r in annual["Revenue"]} == _SOFI_NET


def test_gross_annual_skips_years_missing_either_leg():
    annual, quarterly, matched_tags, missing = _buckets(
        reported=_SOFI_REPORTED,
        net=_SOFI_NET,
        interest={2024: 100.0, 2025: 200.0},
        noninterest={2025: 50.0},
    )

    result = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

    assert result["gross_annual"] == {2025: pytest.approx(250.0)}


# ---------------------------------------------------------------------------
# Never raises.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annual",
    [
        # Not a list of records at all.
        {"NetRevenue": "not-a-list"},
        # A non-numeric value that blows up the divergence arithmetic.
        {
            "NetRevenue": [{"fy": 2025, "value": "abc"}],
            "Revenue": [{"fy": 2025, "value": 1.0}],
        },
        # A null record inside an otherwise well-formed list.
        {"NetRevenue": [None]},
    ],
)
def test_malformed_input_degrades_to_as_reported(annual):
    original = {key: value for key, value in annual.items()}

    result = _apply_net_revenue_basis(annual, {}, {}, [])

    assert result["basis"] == "as_reported"
    assert result["swapped"] is False
    # The buckets are left exactly as they were -- no half-applied swap.
    assert annual == original


# ---------------------------------------------------------------------------
# End-to-end through normalize_facts.
# ---------------------------------------------------------------------------


def _usd_fact(fy, value):
    return {
        "start": f"{fy}-01-01",
        "end": f"{fy}-12-31",
        "val": value,
        "fy": fy,
        "fp": "FY",
        "form": "10-K",
        "filed": f"{fy + 1}-02-20",
    }


def test_normalize_facts_exposes_revenue_basis_and_applies_the_swap():
    facts = {
        "cik": 1818874,
        "entityName": "Synthetic Lender, Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_usd_fact(fy, v) for fy, v in _SOFI_REPORTED.items()]}
                },
                "RevenuesNetOfInterestExpense": {
                    "units": {"USD": [_usd_fact(fy, v) for fy, v in _SOFI_NET.items()]}
                },
            }
        },
    }

    normalized = normalize_facts(facts, years=5)

    assert normalized["revenue_basis"]["basis"] == "net_revenue"
    assert to_annual_series(normalized, "Revenue") == _SOFI_NET
    assert to_annual_series(normalized, "NetRevenue") == _SOFI_NET


def test_normalize_facts_revenue_basis_key_always_present():
    facts = {
        "cik": 320193,
        "entityName": "Synthetic Manufacturer, Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_usd_fact(2025, 1_000.0)]}
                },
            }
        },
    }

    normalized = normalize_facts(facts, years=5)

    assert normalized["revenue_basis"]["basis"] == "as_reported"
    assert normalized["revenue_basis"]["swapped"] is False
    assert to_annual_series(normalized, "Revenue") == {2025: 1_000.0}
