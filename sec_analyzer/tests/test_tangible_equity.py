"""Unit tests for the tangible-equity diagnostics: ROTCE + P/TBV.

SPEC.md Sec.23. These are REPORTED measures, not a new valuation anchor -- the
tests below pin both the arithmetic and the deliberate boundary: P/TBV never
becomes a primary multiple and never reaches triangulation, and the RIM anchor
keeps compounding on book equity (residual income is accounting-invariant, so
rebasing it on tangible equity would move the value only through the model's
10-year truncation).

SoFi FY2025 supplies the hand-verified numbers: equity ``10,489.5M``,
goodwill ``1,390M``, other intangibles ``220M`` -> tangible equity
``8,879.5M``; net income ``481.32M``.
"""

import pytest

from sec_analyzer.cli import _tangible_multiples_line
from sec_analyzer.normalize.metrics import compute_metrics
from sec_analyzer.normalize.ratios import compute_ratios
from sec_analyzer.valuation.multiples import multiples_history
from sec_analyzer.valuation.engine import run_valuation

_EQUITY = 10489.5e6
_GOODWILL = 1390.0e6
_INTANGIBLES = 220.0e6
_TANGIBLE = _EQUITY - _GOODWILL - _INTANGIBLES        # 8,879.5M
_NI = 481.32e6
_SHARES = 1275.3e6


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(**series):
    annual = {name: [_rec(fy, v) for fy, v in sorted(values.items(), reverse=True)]
              for name, values in series.items()}
    return {
        "cik": 1, "entity_name": "Tangible Test Co", "currency": "USD",
        "annual": annual, "quarterly": {}, "missing": [], "matched_tags": {},
    }


def _sofi_normalized(**extra):
    series = dict(
        StockholdersEquity={2025: _EQUITY},
        NetIncome={2025: _NI},
        Goodwill={2025: _GOODWILL},
        IntangibleAssets={2025: _INTANGIBLES},
    )
    series.update(extra)
    return _normalized(**series)


# ---------------------------------------------------------------------------
# 23b. Tangible equity + ROTCE
# ---------------------------------------------------------------------------


def test_rotce_deducts_goodwill_and_intangibles():
    row = compute_ratios(_sofi_normalized())[0]

    assert row["tangible_equity"] == pytest.approx(_TANGIBLE)
    # compute_ratios rounds every ratio to 4dp (`_safe_div`), same as roe.
    assert row["rotce"] == pytest.approx(round(_NI / _TANGIBLE, 4))
    # ROTCE must read HIGHER than ROE whenever intangibles are present.
    assert row["rotce"] > row["roe"]


def test_missing_goodwill_and_intangibles_are_treated_as_zero():
    row = compute_ratios(
        _normalized(StockholdersEquity={2025: _EQUITY}, NetIncome={2025: _NI})
    )[0]

    assert row["tangible_equity"] == pytest.approx(_EQUITY)
    # A genuinely goodwill-free filer: ROTCE IS ROE, not a degenerate value.
    assert row["rotce"] == pytest.approx(row["roe"])


def test_only_one_of_the_two_deductions_present():
    row = compute_ratios(
        _normalized(
            StockholdersEquity={2025: _EQUITY}, NetIncome={2025: _NI},
            Goodwill={2025: _GOODWILL},
        )
    )[0]

    assert row["tangible_equity"] == pytest.approx(_EQUITY - _GOODWILL)


def test_goodwill_exceeding_equity_makes_rotce_unavailable():
    """A non-positive tangible base makes the ratio meaningless, not negative."""
    row = compute_ratios(
        _normalized(
            StockholdersEquity={2025: 1_000.0}, NetIncome={2025: 100.0},
            Goodwill={2025: 1_500.0},
        )
    )[0]

    assert row["tangible_equity"] == pytest.approx(-500.0)
    assert row["rotce"] is None


def test_tangible_equity_is_none_without_book_equity():
    row = compute_ratios(_normalized(NetIncome={2025: _NI}, Goodwill={2025: _GOODWILL}))[0]

    assert row["tangible_equity"] is None
    assert row["rotce"] is None


# ---------------------------------------------------------------------------
# 23c. P/TBV
# ---------------------------------------------------------------------------


def test_metrics_reports_tangible_book_and_ptbv():
    normalized = _sofi_normalized(SharesOutstanding={2025: _SHARES})
    ratios = compute_ratios(normalized)

    metrics = compute_metrics(normalized, ratios, price=15.25)

    assert metrics["tangible_equity"] == pytest.approx(_TANGIBLE)
    assert metrics["tbv_per_share"] == pytest.approx(_TANGIBLE / _SHARES)
    assert metrics["ptbv"] == pytest.approx(15.25 / (_TANGIBLE / _SHARES))


def test_ptbv_is_none_without_a_price():
    normalized = _sofi_normalized(SharesOutstanding={2025: _SHARES})
    metrics = compute_metrics(normalized, compute_ratios(normalized), price=None)

    assert metrics["tbv_per_share"] is not None
    assert metrics["ptbv"] is None


def test_ptbv_is_none_when_tangible_equity_is_non_positive():
    normalized = _normalized(
        StockholdersEquity={2025: 1_000.0}, NetIncome={2025: 100.0},
        Goodwill={2025: 1_500.0}, SharesOutstanding={2025: 100.0},
    )
    metrics = compute_metrics(normalized, compute_ratios(normalized), price=10.0)

    assert metrics["tangible_equity"] is None
    assert metrics["ptbv"] is None


def test_multiples_history_carries_ptbv():
    import pandas as pd

    normalized = _sofi_normalized(SharesOutstanding={2025: _SHARES})
    price_df = pd.DataFrame(
        {"Close": [15.25]}, index=pd.to_datetime(["2025-12-31"])
    )

    history = multiples_history(normalized, price_df)

    assert history
    row = next(h for h in history if h["fy"] == 2025)
    assert row["ptbv"] == pytest.approx(15.25 * _SHARES / _TANGIBLE)


def test_multiples_history_ptbv_none_without_tangible_equity():
    import pandas as pd

    normalized = _normalized(
        NetIncome={2025: _NI}, SharesOutstanding={2025: _SHARES},
        Revenue={2025: 3_613e6},
    )
    price_df = pd.DataFrame({"Close": [15.25]}, index=pd.to_datetime(["2025-12-31"]))

    history = multiples_history(normalized, price_df)

    assert all(h["ptbv"] is None for h in history)


# ---------------------------------------------------------------------------
# 23c/23d. Engine wiring
# ---------------------------------------------------------------------------


def _assumptions():
    return {
        "bear": {"growth_5y": 0.20, "terminal_growth": 0.04, "discount_rate": 0.12, "story": "a"},
        "base": {"growth_5y": 0.25, "terminal_growth": 0.04, "discount_rate": 0.10, "story": "b"},
        "bull": {"growth_5y": 0.30, "terminal_growth": 0.04, "discount_rate": 0.09, "story": "c"},
    }


def _run(sector_type="financial", price=15.25):
    normalized = _sofi_normalized()
    ratios = compute_ratios(normalized)
    metrics = compute_metrics(normalized, ratios, price=price)
    metrics["shares"] = _SHARES
    metrics["latest_fy"] = 2025
    # compute_metrics derives tangible figures off its own share count; realign
    # them to the explicit share count this fixture pins.
    metrics["tbv_per_share"] = _TANGIBLE / _SHARES
    metrics["ptbv"] = None if price is None else price / (_TANGIBLE / _SHARES)
    return run_valuation(
        normalized, ratios, metrics, price=price, price_df=None,
        assumptions=_assumptions(), sector_type=sector_type,
    )


def test_engine_exposes_ptbv_and_its_percentile_slot():
    multiples = _run()["multiples"]

    assert multiples["current"]["ptbv"] == pytest.approx(15.25 / (_TANGIBLE / _SHARES))
    # No price history in this fixture, so the percentile is not computable --
    # but the key must exist so consumers can read a stable shape.
    assert "ptbv_percentile" in multiples
    assert multiples["ptbv_percentile"] is None


def test_rim_block_carries_the_tangible_figures():
    rim = _run()["rim"]

    assert rim["tangible_equity"] == pytest.approx(_TANGIBLE)
    assert rim["tbv_per_share"] == pytest.approx(_TANGIBLE / _SHARES)
    assert rim["rotce"] == pytest.approx(round(_NI / _TANGIBLE, 4))
    # ...but the anchor itself still compounds on BOOK equity.
    assert rim["bve0"] == pytest.approx(_EQUITY)
    assert rim["roe"] == pytest.approx(round(_NI / _EQUITY, 4))


def test_ptbv_does_not_become_the_primary_multiple_or_reach_triangulation():
    """The deliberate boundary in Sec.23c: reporting the number must not change
    any filer's multiples signal."""
    result = _run()
    tri = result["triangulation"]

    assert "F/MDD" not in (tri.get("rationale") or {}).get("multiples", "")
    comparison = ((result["multiples"].get("sector") or {}).get("comparison") or {})
    assert comparison.get("label") != "F/MDD"


def test_ptbv_is_computed_for_non_financial_sectors_too():
    """Sec.23 computes the raw figures for every filer; only the DISPLAY is
    gated on sector."""
    multiples = _run(sector_type="mature")["multiples"]

    assert multiples["current"]["ptbv"] is not None


# ---------------------------------------------------------------------------
# 23e. CLI line
# ---------------------------------------------------------------------------


def test_cli_tangible_line_renders_for_a_financial_filer():
    valuation = {
        "sector_type": "financial",
        "multiples": {"current": {"ptbv": 2.19}, "ptbv_percentile": 34.0},
        "rim": {"rotce": 0.0542},
    }

    line = _tangible_multiples_line(valuation)

    assert "MDD çarpanı:" in line
    # The label must not run into the value -- it is exactly one char shorter
    # than the card's label column.
    assert "MDD çarpanı: " in line
    assert "F/MDD 2.19×" in line
    assert "34. pctile" in line
    assert "ROTCE %5.4" in line


def test_cli_tangible_line_omits_rotce_when_unavailable():
    valuation = {
        "sector_type": "financial",
        "multiples": {"current": {"ptbv": 2.19}, "ptbv_percentile": None},
        "rim": None,
    }

    line = _tangible_multiples_line(valuation)

    assert "F/MDD 2.19×" in line
    assert "pctile" not in line
    assert "ROTCE" not in line


@pytest.mark.parametrize("sector_type", ["mature", "reit", "cyclical", "growth_unprofitable"])
def test_cli_tangible_line_is_financial_only(sector_type):
    valuation = {
        "sector_type": sector_type,
        "multiples": {"current": {"ptbv": 2.19}, "ptbv_percentile": 34.0},
        "rim": {"rotce": 0.0542},
    }

    assert _tangible_multiples_line(valuation) is None


def test_cli_tangible_line_omitted_without_a_ptbv_figure():
    valuation = {
        "sector_type": "financial",
        "multiples": {"current": {"ptbv": None}, "ptbv_percentile": None},
        "rim": {"rotce": 0.0542},
    }

    assert _tangible_multiples_line(valuation) is None
