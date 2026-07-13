"""Unit tests for sec_analyzer.normalize.metrics.

``compute_metrics`` only depends on the normalized-facts *shape* produced by
``normalizer.normalize_facts`` (a dict of concept -> list of records with
``fy``/``value`` keys) plus a ``ratios`` list shaped like
``ratios.compute_ratios``'s output, so these fixtures build that shape
directly -- matching the style of ``test_ratios.py``.
"""

from sec_analyzer.normalize.metrics import compute_metrics

_CONCEPTS = [
    "SharesOutstanding", "EPS", "LongTermDebt", "LongTermDebtCurrent",
    "Cash", "Revenue", "SBC", "RnD", "Buyback", "DividendsPaid",
    "OperatingCashFlow", "CapEx",
]


def _record(fy, value):
    """Build a minimal annual record dict -- only ``fy``/``value`` matter
    for ``to_annual_series``, which ``compute_metrics`` relies on exclusively."""
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(annual_overrides):
    annual = {c: annual_overrides.get(c) for c in _CONCEPTS}
    return {
        "cik": 1, "entity_name": "Metrics Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CONCEPTS},
        "missing": [c for c in _CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CONCEPTS},
    }


def _full_normalized():
    """A filer with 4 fiscal years of data (2018, 2020, 2022, 2023), enough
    to exercise both the 3y and 5y revenue CAGR windows."""
    return _normalized(
        {
            "SharesOutstanding": [
                _record(2023, 100),
                _record(2022, 90),
            ],
            "EPS": [_record(2023, 2.0)],
            "LongTermDebt": [_record(2023, 300)],
            "LongTermDebtCurrent": [_record(2023, 50)],
            "Cash": [_record(2023, 80)],
            "Revenue": [
                _record(2023, 1000),
                _record(2020, 500),
                _record(2018, 250),
            ],
            "SBC": [_record(2023, 50)],
            "RnD": [_record(2023, 40)],
            "Buyback": [_record(2023, 20)],
            "DividendsPaid": [_record(2023, 10)],
        }
    )


_FULL_RATIOS = [{"fy": 2023, "fcf": 180, "net_margin": 0.1}]


def test_full_metrics_with_price():
    normalized = _full_normalized()
    metrics = compute_metrics(normalized, _FULL_RATIOS, price=20.0)

    assert metrics["latest_fy"] == 2023
    assert metrics["price"] == 20.0
    assert metrics["shares"] == 100
    assert metrics["eps"] == 2.0
    assert metrics["market_cap"] == 2000.0
    assert metrics["total_debt"] == 350.0
    assert metrics["net_debt"] == 270.0
    assert metrics["pe"] == 10.0
    assert metrics["ps"] == 2.0
    assert metrics["pfcf"] == round(2000.0 / 180, 4)
    assert metrics["revenue_cagr_3y"] == round((1000 / 500) ** (1 / 3) - 1, 4)
    assert metrics["revenue_cagr_5y"] == round((1000 / 250) ** (1 / 5) - 1, 4)
    assert metrics["sbc_revenue"] == 0.05
    assert metrics["rnd_revenue"] == 0.04
    assert metrics["shares_yoy"] == round(100 / 90 - 1, 4)
    assert metrics["buyback_latest"] == 20
    assert metrics["dividends_latest"] == 10
    assert metrics["fcf"] == 180
    assert metrics["fcf_per_share"] == round(180 / 100, 2)


def test_eps_non_positive_makes_pe_none_but_other_metrics_survive():
    normalized = _full_normalized()
    normalized["annual"]["EPS"] = [_record(2023, -1.0)]

    metrics = compute_metrics(normalized, _FULL_RATIOS, price=20.0)

    assert metrics["pe"] is None
    # market_cap/ps/pfcf don't depend on EPS, so they're unaffected.
    assert metrics["market_cap"] == 2000.0
    assert metrics["ps"] == 2.0
    assert metrics["pfcf"] == round(2000.0 / 180, 4)


def test_missing_price_nulls_price_dependent_metrics_but_keeps_others():
    normalized = _full_normalized()
    metrics = compute_metrics(normalized, _FULL_RATIOS, price=None)

    assert metrics["price"] is None
    assert metrics["market_cap"] is None
    assert metrics["pe"] is None
    assert metrics["ps"] is None
    assert metrics["pfcf"] is None

    # Non-price-dependent metrics are still computed.
    assert metrics["revenue_cagr_3y"] == round((1000 / 500) ** (1 / 3) - 1, 4)
    assert metrics["revenue_cagr_5y"] == round((1000 / 250) ** (1 / 5) - 1, 4)
    assert metrics["sbc_revenue"] == 0.05
    assert metrics["rnd_revenue"] == 0.04
    assert metrics["shares_yoy"] == round(100 / 90 - 1, 4)
    assert metrics["fcf"] == 180


def test_total_debt_treats_missing_current_portion_as_zero():
    normalized = _normalized(
        {
            "LongTermDebt": [_record(2023, 300)],
            # No LongTermDebtCurrent at all.
            "Cash": [_record(2023, 80)],
        }
    )
    metrics = compute_metrics(normalized, [], price=None)

    assert metrics["total_debt"] == 300
    assert metrics["net_debt"] == 220


def test_total_debt_is_none_when_both_debt_concepts_missing():
    normalized = _normalized({"Cash": [_record(2023, 80)]})
    metrics = compute_metrics(normalized, [], price=None)

    assert metrics["total_debt"] is None
    assert metrics["net_debt"] is None


def test_no_data_at_all_returns_all_none_without_raising():
    normalized = _normalized({})
    metrics = compute_metrics(normalized, [], price=None)

    assert metrics["latest_fy"] is None
    assert metrics["market_cap"] is None
    assert metrics["pe"] is None
    assert metrics["shares_yoy"] is None
    assert metrics["fcf"] is None


def test_shares_yoy_none_when_prior_year_missing():
    normalized = _normalized({"SharesOutstanding": [_record(2023, 100)]})
    metrics = compute_metrics(normalized, [], price=None)

    assert metrics["shares_yoy"] is None
    assert metrics["shares"] == 100
