"""Unit tests for sec_analyzer.normalize.metrics.

``compute_metrics`` only depends on the normalized-facts *shape* produced by
``normalizer.normalize_facts`` (a dict of concept -> list of records with
``fy``/``value`` keys) plus a ``ratios`` list shaped like
``ratios.compute_ratios``'s output, so these fixtures build that shape
directly -- matching the style of ``test_ratios.py``.
"""

import pytest

from sec_analyzer.normalize.metrics import compute_metrics, resolve_fundamental_fy
from sec_analyzer.normalize.normalizer import to_annual_series

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


# ---------------------------------------------------------------------------
# "Ghost fiscal year" anchor split: latest_fy (ALL series, incl.
# SharesOutstanding) vs latest_fundamental_fy (every series EXCEPT
# SharesOutstanding). A dei cover-page share count can carry a fiscal year
# newer than the filer's actual financial statements (e.g. AMZN) -- fixture
# below reproduces that mismatch on purpose.
# ---------------------------------------------------------------------------


def _ghost_year_normalized():
    """SharesOutstanding (cover-page, point-in-time) runs one fiscal year
    ahead (FY2022-2025) of every fundamental series (FY2022-2024) -- the
    "ghost year" is FY2025, which has share-count data but NO revenue/EPS/
    OCF/CapEx data at all."""
    return _normalized(
        {
            "SharesOutstanding": [
                _record(2025, 110),
                _record(2024, 100),
                _record(2023, 90),
                _record(2022, 80),
            ],
            "EPS": [
                _record(2024, 5.0),
                _record(2023, 4.0),
                _record(2022, 3.0),
            ],
            # 1000 -(10%/yr)-> 1100 -> 1210 -> 1331: revenue_cagr_3y off the
            # FY2024 anchor is then exactly (1331/1000)**(1/3) - 1 == 0.10
            # (since 1.1**3 == 1.331), a clean hand-checkable number.
            "Revenue": [
                _record(2024, 1331),
                _record(2023, 1210),
                _record(2022, 1100),
                _record(2021, 1000),
            ],
            "OperatingCashFlow": [
                _record(2024, 250),
                _record(2023, 235),
                _record(2022, 220),
            ],
            "CapEx": [
                _record(2024, 50),
                _record(2023, 45),
                _record(2022, 40),
            ],
        }
    )


# ratios' fy=2024 fcf (200) agrees with OCF(250) - CapEx(50) = 200 at the
# same FY, so the fcf figure is the same whether it comes from `ratios` or
# the OCF-CapEx fallback -- self-consistent fixture.
_GHOST_YEAR_RATIOS = [{"fy": 2024, "fcf": 200.0}]


def test_ghost_year_latest_fy_is_the_shares_only_cover_page_year():
    normalized = _ghost_year_normalized()
    metrics = compute_metrics(normalized, _GHOST_YEAR_RATIOS, price=13.31)

    # latest_fy is the max across ALL series, including SharesOutstanding
    # -> the ghost year FY2025 (no fundamental data exists for it at all).
    assert metrics["latest_fy"] == 2025
    # latest_fundamental_fy excludes SharesOutstanding -> the real anchor,
    # FY2024, where the financial statements actually report.
    assert metrics["latest_fundamental_fy"] == 2024


def test_ghost_year_shares_and_market_cap_use_the_freshest_share_count():
    normalized = _ghost_year_normalized()
    metrics = compute_metrics(normalized, _GHOST_YEAR_RATIOS, price=13.31)

    # shares must still reflect FY2025 (the freshest count), NOT FY2024.
    assert metrics["shares"] == 110
    assert metrics["market_cap"] == pytest.approx(13.31 * 110)


def test_ghost_year_fundamental_reads_anchor_on_latest_fundamental_fy_not_latest_fy():
    normalized = _ghost_year_normalized()
    metrics = compute_metrics(normalized, _GHOST_YEAR_RATIOS, price=13.31)

    # Sanity check on the fixture itself: FY2025 (the ghost year) truly has
    # no fundamental data in the underlying series. If `latest_fy` (2025)
    # had wrongly been used as the fundamental anchor (the pre-fix bug),
    # every read below would resolve against a fiscal year with nothing in
    # it and come back None.
    assert 2025 not in to_annual_series(normalized, "Revenue")
    assert 2025 not in to_annual_series(normalized, "EPS")
    assert 2025 not in to_annual_series(normalized, "OperatingCashFlow")
    assert 2025 not in to_annual_series(normalized, "CapEx")

    # revenue_cagr_3y: (1331/1000)**(1/3) - 1 == 0.10 exactly, computed off
    # the FY2024 anchor (latest_fundamental_fy) -- NOT None, as it would be
    # if anchored on the ghost FY2025 (which has no Revenue entry at all,
    # let alone one 3 years back at FY2022... which also isn't FY2021 so
    # the "old" lookup would have failed doubly).
    assert metrics["revenue_cagr_3y"] == pytest.approx(0.10, abs=1e-6)

    # fcf comes from ratios' FY2024 entry (latest_fundamental_fy): 200,
    # positive.
    assert metrics["fcf"] == 200.0

    # price-dependent multiples are all computed off the FY2024 anchor +
    # the fresh FY2025 share count -- none of them collapse to None.
    assert metrics["pe"] is not None
    assert metrics["ps"] is not None
    assert metrics["pfcf"] is not None
    # pe = price / eps = 13.31 / 5.0 = 2.662 exactly.
    assert metrics["pe"] == pytest.approx(2.662, abs=1e-6)
    # ps = market_cap / revenue = (13.31*110) / 1331 = 1464.1 / 1331 = 1.1
    assert metrics["ps"] == pytest.approx(1.1, abs=1e-6)
    # pfcf = market_cap / fcf = 1464.1 / 200 = 7.3205
    assert metrics["pfcf"] == pytest.approx(1464.1 / 200, abs=1e-6)


# ---------------------------------------------------------------------------
# resolve_fundamental_fy unit tests (B)
# ---------------------------------------------------------------------------


def test_resolve_fundamental_fy_returns_explicit_key_when_present_and_not_none():
    assert resolve_fundamental_fy({"latest_fundamental_fy": 2022, "latest_fy": 2024}) == 2022


def test_resolve_fundamental_fy_falls_back_to_latest_fy_when_key_absent():
    # Mirrors hand-built metrics dicts in other test modules (e.g.
    # test_valuation_engine.py) that never produce "latest_fundamental_fy"
    # at all -- resolve_fundamental_fy must fall back to "latest_fy" so
    # those existing dicts keep behaving exactly as before.
    assert resolve_fundamental_fy({"latest_fy": 2023}) == 2023


def test_resolve_fundamental_fy_falls_back_to_latest_fy_when_value_is_none():
    assert resolve_fundamental_fy({"latest_fundamental_fy": None, "latest_fy": 2023}) == 2023


def test_resolve_fundamental_fy_handles_missing_or_none_metrics_dict():
    assert resolve_fundamental_fy({}) is None
    assert resolve_fundamental_fy(None) is None


# ---------------------------------------------------------------------------
# Regression guarantee (C): when every series (including SharesOutstanding)
# tops out at the same fiscal year, latest_fundamental_fy == latest_fy and
# every metric is unaffected by the anchor split.
# ---------------------------------------------------------------------------


def test_latest_fundamental_fy_matches_latest_fy_and_metrics_are_unchanged_without_mismatch():
    normalized = _full_normalized()
    metrics = compute_metrics(normalized, _FULL_RATIOS, price=20.0)

    assert metrics["latest_fundamental_fy"] == metrics["latest_fy"] == 2023

    # Same values as test_full_metrics_with_price -- the anchor split must
    # not change a single figure when there's no fiscal-year mismatch.
    assert metrics["shares"] == 100
    assert metrics["eps"] == 2.0
    assert metrics["market_cap"] == 2000.0
    assert metrics["pe"] == 10.0
    assert metrics["ps"] == 2.0
    assert metrics["pfcf"] == round(2000.0 / 180, 4)
    assert metrics["revenue_cagr_3y"] == round((1000 / 500) ** (1 / 3) - 1, 4)
    assert metrics["revenue_cagr_5y"] == round((1000 / 250) ** (1 / 5) - 1, 4)
    assert metrics["fcf"] == 180
