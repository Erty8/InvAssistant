"""Tests for the advisory cycle block on SIC 3674 filers (SPEC.md Sec.26).

A semiconductor filer routed OUT of ``"cyclical"`` (realized revenue CAGR
above 15%, SPEC.md Sec.8) must still get the advisory ``valuation["cycle"]``
block when ``run_valuation(..., sic=3674)`` is used -- without any change to
the headline valuation.
"""

import pytest

from sec_analyzer.valuation.engine import run_valuation
from sec_analyzer.valuation.sector import is_semiconductor_sic

_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity",
]

_MARGINS = {
    2016: -0.02, 2017: 0.25, 2018: 0.46, 2019: 0.27, 2020: 0.12,
    2021: 0.21, 2022: 0.28, 2023: -0.37, 2024: 0.03, 2025: 0.23,
}

_METRICS = {
    "shares": 10.0, "latest_fy": 2025, "net_debt": 0.0,
    "ttm_complete": True, "ttm_revenue": 1000.0, "ttm_net_margin": 0.23,
}
_CANNOT_COMPUTE = "Cycle position could not be computed"


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized():
    annual = {c: None for c in _CONCEPTS}
    annual["Revenue"] = [_rec(2025, 1000.0)]
    annual["NetIncome"] = [_rec(2025, 100.0)]
    return {
        "cik": 1, "entity_name": "Semi Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CONCEPTS},
        "missing": [c for c in _CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CONCEPTS},
    }


def _assumptions():
    return {
        "bear": {"growth_5y": 0.04, "terminal_growth": 0.01, "discount_rate": 0.12, "story": "Bear."},
        "base": {"growth_5y": 0.08, "terminal_growth": 0.02, "discount_rate": 0.09, "story": "Base."},
        "bull": {"growth_5y": 0.12, "terminal_growth": 0.03, "discount_rate": 0.08, "story": "Bull."},
    }


def _ratios(margins):
    return [{"fy": fy, "net_margin": m} for fy, m in sorted(margins.items())]


def _run(sector_type, margins, **kwargs):
    return run_valuation(
        _normalized(), _ratios(margins), _METRICS, price=100.0, price_df=None,
        assumptions=_assumptions(), sector_type=sector_type, **kwargs,
    )


def test_is_semiconductor_sic():
    assert is_semiconductor_sic(3674) is True
    assert is_semiconductor_sic("3674") is True
    assert is_semiconductor_sic(3672) is False
    assert is_semiconductor_sic(None) is False
    assert is_semiconductor_sic("abc") is False


def test_semi_in_mature_bucket_gets_advisory_cycle_without_changing_headline():
    with_sic = _run("mature", _MARGINS, sic=3674)
    without_sic = _run("mature", _MARGINS, sic=None)

    assert with_sic["cycle"] is not None
    assert without_sic["cycle"] is None
    assert any("valued outside the cyclical bucket" in n for n in with_sic["notes"])
    assert not any("valued outside the cyclical bucket" in n for n in without_sic["notes"])
    # The extra note precedes the verdict sentence.
    notes = with_sic["notes"]
    idx_extra = next(i for i, n in enumerate(notes) if "valued outside the cyclical bucket" in n)
    assert notes[idx_extra + 1] == with_sic["cycle"]["verdict_sentence"]

    for key in ("fair_value_range", "primary_dcf_scenarios", "triangulation", "verdict", "sector_type"):
        assert with_sic.get(key) == without_sic.get(key), key


def test_semi_in_mature_bucket_with_thin_history_has_no_cycle_and_no_noise_note():
    thin = {fy: m for fy, m in _MARGINS.items() if fy >= 2023}
    result = _run("mature", thin, sic=3674)

    assert result["cycle"] is None
    assert not any(_CANNOT_COMPUTE in n for n in result["notes"])


@pytest.mark.parametrize("sic", [None, 3674, 2911])
def test_cyclical_with_thin_history_still_emits_cannot_compute_note(sic):
    thin = {fy: m for fy, m in _MARGINS.items() if fy >= 2023}
    result = _run("cyclical", thin, sic=sic)

    assert result["cycle"] is None
    assert any(_CANNOT_COMPUTE in n for n in result["notes"])
