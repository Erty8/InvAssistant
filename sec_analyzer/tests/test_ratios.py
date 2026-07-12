"""Unit tests for sec_analyzer.normalize.ratios.

``compute_ratios`` only depends on the normalized-facts *shape* produced by
``normalizer.normalize_facts`` (a dict of concept -> list of records with
``fy``/``value``/``period_end`` keys), so these fixtures build that shape
directly rather than round-tripping through a full companyfacts document.
"""

from sec_analyzer.normalize.ratios import compute_ratios


def _record(fy, period_end, value):
    """Build a minimal annual record dict, matching normalizer's record shape."""
    return {
        "concept": None,
        "tag": None,
        "period_end": period_end,
        "fy": fy,
        "fp": "FY",
        "form": "10-K",
        "value": value,
        "filed": None,
        "start": None,
    }


def _normalized(annual_overrides):
    """Build a minimal normalized-facts dict with an `annual` bucket.

    ``annual_overrides`` supplies the concepts under test; any concept not
    given defaults to ``None`` (missing), matching what normalize_facts
    would produce for a concept with no matching tag.
    """
    concepts = [
        "Revenue",
        "NetIncome",
        "TotalAssets",
        "TotalLiabilities",
        "StockholdersEquity",
        "OperatingCashFlow",
        "Cash",
        "CurrentAssets",
        "CurrentLiabilities",
        "GrossProfit",
        "OperatingIncome",
        "CapEx",
    ]
    annual = {c: annual_overrides.get(c) for c in concepts}
    return {
        "cik": 1,
        "entity_name": "Ratio Test Co",
        "currency": "USD",
        "annual": annual,
        "quarterly": {c: None for c in concepts},
        "missing": [c for c in concepts if annual[c] is None],
        "matched_tags": {c: None for c in concepts},
    }


def test_compute_ratios_two_fiscal_years():
    normalized = _normalized(
        {
            "Revenue": [
                _record(2023, "2023-12-31", 1000),
                _record(2022, "2022-12-31", 800),
            ],
            "NetIncome": [
                _record(2023, "2023-12-31", 100),
                _record(2022, "2022-12-31", 50),
            ],
            "StockholdersEquity": [
                _record(2023, "2023-12-31", 500),
                _record(2022, "2022-12-31", 400),
            ],
        }
    )

    results = compute_ratios(normalized)

    assert [r["fy"] for r in results] == [2023, 2022]

    fy2023 = results[0]
    assert fy2023["period_end"] == "2023-12-31"
    assert fy2023["net_margin"] == 0.1
    assert fy2023["roe"] == 0.2
    assert fy2023["yoy_revenue_growth"] == 0.25
    assert fy2023["yoy_net_income_growth"] == 1.0
    # No CurrentAssets/CurrentLiabilities data was supplied.
    assert fy2023["current_ratio"] is None

    fy2022 = results[1]
    assert fy2022["net_margin"] == 0.0625
    assert fy2022["roe"] == 0.125
    # No prior (2021) data exists, so growth rates must be None, not an error.
    assert fy2022["yoy_revenue_growth"] is None
    assert fy2022["yoy_net_income_growth"] is None


def test_current_ratio_computed_when_available():
    normalized = _normalized(
        {
            "CurrentAssets": [_record(2023, "2023-12-31", 300)],
            "CurrentLiabilities": [_record(2023, "2023-12-31", 150)],
        }
    )

    results = compute_ratios(normalized)
    assert len(results) == 1
    assert results[0]["current_ratio"] == 2.0
    # Everything else is missing input data and must be None, not an error.
    assert results[0]["net_margin"] is None
    assert results[0]["roe"] is None


def test_missing_inputs_yield_none_without_raising():
    """A fully empty annual bucket must not raise and returns no rows."""
    normalized = _normalized({})
    assert compute_ratios(normalized) == []


def test_growth_guards_against_non_positive_prior_base():
    """A zero or negative prior-year value must not produce a growth ratio."""
    normalized = _normalized(
        {
            "Revenue": [
                _record(2023, "2023-12-31", 500),
                _record(2022, "2022-12-31", 0),
            ],
            "NetIncome": [
                _record(2023, "2023-12-31", 20),
                _record(2022, "2022-12-31", -30),
            ],
        }
    )

    results = compute_ratios(normalized)
    fy2023 = next(r for r in results if r["fy"] == 2023)

    assert fy2023["yoy_revenue_growth"] is None  # prior revenue was 0
    assert fy2023["yoy_net_income_growth"] is None  # prior net income was negative


def test_extended_margins_and_returns_two_year_fixture():
    """gross_margin, operating_margin, roa, and debt_to_equity across two
    fiscal years, computed from GrossProfit/OperatingIncome/TotalAssets/
    TotalLiabilities/StockholdersEquity/Revenue/NetIncome inputs."""
    normalized = _normalized(
        {
            "Revenue": [
                _record(2023, "2023-12-31", 1000),
                _record(2022, "2022-12-31", 800),
            ],
            "GrossProfit": [
                _record(2023, "2023-12-31", 600),
                _record(2022, "2022-12-31", 400),
            ],
            "OperatingIncome": [
                _record(2023, "2023-12-31", 300),
                _record(2022, "2022-12-31", 200),
            ],
            "NetIncome": [
                _record(2023, "2023-12-31", 100),
                _record(2022, "2022-12-31", 50),
            ],
            "TotalAssets": [
                _record(2023, "2023-12-31", 2000),
                _record(2022, "2022-12-31", 1600),
            ],
            "TotalLiabilities": [
                _record(2023, "2023-12-31", 1200),
                _record(2022, "2022-12-31", 1000),
            ],
            "StockholdersEquity": [
                _record(2023, "2023-12-31", 800),
                _record(2022, "2022-12-31", 600),
            ],
        }
    )

    results = compute_ratios(normalized)
    fy2023 = next(r for r in results if r["fy"] == 2023)
    fy2022 = next(r for r in results if r["fy"] == 2022)

    assert fy2023["gross_margin"] == 0.6
    assert fy2023["operating_margin"] == 0.3
    assert fy2023["roa"] == 0.05
    assert fy2023["debt_to_equity"] == 1.5

    assert fy2022["gross_margin"] == 0.5
    assert fy2022["operating_margin"] == 0.25
    assert fy2022["roa"] == round(50 / 1600, 4)
    assert fy2022["debt_to_equity"] == round(1000 / 600, 4)


def test_fcf_and_fcf_margin_computed_from_ocf_minus_capex():
    """fcf is a raw USD figure (OperatingCashFlow - CapEx), not a ratio;
    fcf_margin divides that figure by Revenue."""
    normalized = _normalized(
        {
            "Revenue": [_record(2023, "2023-12-31", 1000)],
            "OperatingCashFlow": [_record(2023, "2023-12-31", 300)],
            "CapEx": [_record(2023, "2023-12-31", 120)],
        }
    )

    results = compute_ratios(normalized)
    assert len(results) == 1
    assert results[0]["fcf"] == 180
    assert results[0]["fcf_margin"] == 0.18


def test_fcf_is_none_when_capex_missing():
    """fcf must not fall back to OperatingCashFlow alone when CapEx data
    isn't available -- that would silently overstate free cash flow."""
    normalized = _normalized(
        {
            "Revenue": [_record(2023, "2023-12-31", 1000)],
            "OperatingCashFlow": [_record(2023, "2023-12-31", 300)],
            # No CapEx supplied.
        }
    )

    results = compute_ratios(normalized)
    assert len(results) == 1
    assert results[0]["fcf"] is None
    assert results[0]["fcf_margin"] is None
