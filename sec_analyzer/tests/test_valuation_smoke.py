"""Smoke tests for the valuation package.

These only check shapes and obvious round-trip behavior (dcf with simple
numbers, an end-to-end engine run on a minimal fake fixture, sanity catching
an undefined Gordon-growth input). Thorough, hand-verified numeric tests are
a separate follow-up; this file exists to catch import errors, contract
drift, and crashes.
"""

from sec_analyzer.valuation.dcf import dcf_per_share, project_fcf
from sec_analyzer.valuation.engine import run_valuation
from sec_analyzer.valuation.sanity import validate_assumptions
from sec_analyzer.valuation.sector import classify_sector


# ---------------------------------------------------------------------------
# dcf.py
# ---------------------------------------------------------------------------


def test_project_fcf_ten_years_fading_growth():
    path = project_fcf(fcf0=100.0, growth_5y=0.10, terminal_growth=0.02)

    assert len(path) == 10
    # Year 1: 100 * 1.10
    assert path[0] == 100.0 * 1.10
    # Year 5 growth is still 0.10.
    assert path[4] == path[3] * 1.10
    # Year 10 growth must equal terminal_growth exactly.
    assert path[9] == path[8] * (1 + 0.02)


def test_dcf_per_share_returns_expected_shape():
    # F1 (FCFE-direct): dcf_per_share no longer takes a net_debt parameter --
    # equity == ev directly (no net-debt subtraction).
    result = dcf_per_share(
        fcf0=100.0, growth_5y=0.10, terminal_growth=0.02, discount_rate=0.10,
        shares=100.0, dilution_rate=0.0,
    )

    assert set(result) == {"per_share", "ev", "equity", "fcf_path", "tv", "effective_shares"}
    assert len(result["fcf_path"]) == 10
    assert result["effective_shares"] == 100.0
    assert result["equity"] == result["ev"]
    assert result["per_share"] == result["equity"] / 100.0
    assert result["per_share"] > 0


def test_dcf_per_share_raises_on_bad_rate_relationship():
    try:
        dcf_per_share(100.0, 0.10, 0.05, discount_rate=0.03, shares=10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when discount_rate <= terminal_growth")


def test_dcf_per_share_raises_on_missing_fcf0():
    try:
        dcf_per_share(None, 0.10, 0.02, 0.10, shares=10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when fcf0 is None")


def test_dcf_per_share_raises_on_bad_shares():
    try:
        dcf_per_share(100.0, 0.10, 0.02, 0.10, shares=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when shares is falsy/<=0")


def test_dcf_per_share_dilution_uses_year5_compounding():
    with_dilution = dcf_per_share(100.0, 0.10, 0.02, 0.10, shares=100.0, dilution_rate=0.05)
    assert with_dilution["effective_shares"] == 100.0 * (1.05 ** 5)


# ---------------------------------------------------------------------------
# sanity.py
# ---------------------------------------------------------------------------


def _valid_assumptions():
    return {
        "bear": {"growth_5y": 0.03, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı senaryosu."},
        "base": {"growth_5y": 0.08, "terminal_growth": 0.025, "discount_rate": 0.10, "story": "Baz senaryo."},
        "bull": {"growth_5y": 0.13, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa senaryosu."},
    }


def test_validate_assumptions_accepts_a_clean_set():
    violations = validate_assumptions(_valid_assumptions(), is_unprofitable=False)
    assert violations == []


def test_validate_assumptions_catches_discount_rate_at_or_below_terminal_growth():
    assumptions = _valid_assumptions()
    assumptions["base"]["discount_rate"] = 0.02
    assumptions["base"]["terminal_growth"] = 0.025

    violations = validate_assumptions(assumptions, is_unprofitable=False)
    assert any("Gordon" in v for v in violations)


def test_validate_assumptions_catches_missing_field():
    assumptions = _valid_assumptions()
    del assumptions["bull"]["growth_5y"]

    violations = validate_assumptions(assumptions, is_unprofitable=False)
    assert any("growth_5y" in v for v in violations)


def test_validate_assumptions_never_raises_on_garbage_input():
    violations = validate_assumptions(None, is_unprofitable=False)
    assert isinstance(violations, list)
    assert violations  # missing scenarios entirely -> violations, not a crash


# ---------------------------------------------------------------------------
# sector.py
# ---------------------------------------------------------------------------


def test_classify_sector_reit_and_financial_and_missing_sic():
    assert classify_sector(6798, {}, {}) == "reit"
    assert classify_sector(6021, {}, {}) == "financial"
    assert classify_sector(None, {}, {}) == "mature"


# ---------------------------------------------------------------------------
# engine.py end-to-end on a minimal fake fixture
# ---------------------------------------------------------------------------


def _fake_normalized():
    def rec(fy, value):
        return {
            "concept": None, "tag": None, "period_end": f"{fy}-12-31",
            "fy": fy, "fp": "FY", "form": "10-K", "value": value,
            "filed": None, "start": None, "unit": "USD",
        }

    concepts = {
        "Revenue": [rec(2023, 1000.0), rec(2022, 900.0), rec(2021, 800.0)],
        "NetIncome": [rec(2023, 100.0), rec(2022, 80.0), rec(2021, 60.0)],
        "OperatingCashFlow": [rec(2023, 150.0), rec(2022, 130.0), rec(2021, 110.0)],
        "CapEx": [rec(2023, 40.0), rec(2022, 35.0), rec(2021, 30.0)],
        "Cash": [rec(2023, 200.0)],
        "LongTermDebt": [rec(2023, 300.0)],
        "LongTermDebtCurrent": [rec(2023, 20.0)],
        "SharesOutstanding": [rec(2023, 100.0), rec(2022, 98.0)],
        "EPS": [rec(2023, 1.0)],
        "SBC": [rec(2023, 10.0)],
        "StockholdersEquity": [rec(2023, 500.0)],
    }
    all_concepts = [
        "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
        "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
        "SBC", "StockholdersEquity",
    ]
    annual = {c: concepts.get(c) for c in all_concepts}
    return {
        "cik": 1, "entity_name": "Fixture Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in all_concepts},
        "missing": [c for c in all_concepts if annual[c] is None],
        "matched_tags": {c: None for c in all_concepts},
    }


def _fake_ratios():
    return [
        {"fy": 2023, "net_margin": 0.10, "roe": 0.20, "fcf": 110.0, "fcf_margin": 0.11},
        {"fy": 2022, "net_margin": 0.089, "roe": 0.18, "fcf": 95.0, "fcf_margin": 0.106},
        {"fy": 2021, "net_margin": 0.075, "roe": 0.15, "fcf": 80.0, "fcf_margin": 0.10},
    ]


def _fake_metrics():
    return {
        "price": 20.0, "shares": 100.0, "eps": 1.0, "market_cap": 2000.0,
        "total_debt": 320.0, "net_debt": 120.0, "pe": 20.0, "ps": 2.0,
        "pfcf": round(2000.0 / 110.0, 4), "revenue_cagr_3y": 0.118,
        "revenue_cagr_5y": None, "sbc_revenue": 0.01, "shares_yoy": round(100 / 98 - 1, 4),
        "buyback_latest": None, "dividends_latest": None, "rnd_revenue": None,
        "fcf": 110.0, "fcf_per_share": 1.10, "latest_fy": 2023,
    }


def _fake_assumptions():
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı senaryosu."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.025, "discount_rate": 0.10, "story": "Baz senaryo."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa senaryosu."},
    }


def test_run_valuation_end_to_end_on_minimal_fixture():
    normalized = _fake_normalized()
    ratios = _fake_ratios()
    metrics = _fake_metrics()
    assumptions = _fake_assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    # Top-level shape.
    for key in (
        "sector_type", "fcf0", "fcf0_source", "dcf", "pb_roe", "fair_value_range",
        "reverse_dcf", "multiples", "sensitivity", "triangulation", "assumptions", "notes",
    ):
        assert key in result

    assert result["sector_type"] == "mature"
    assert result["dcf"]["enabled"] is True
    assert result["dcf"]["scenarios"] is not None
    for key in ("bear", "base", "bull"):
        scenario = result["dcf"]["scenarios"][key]
        assert scenario["per_share"] is not None
        assert scenario["lo"] <= scenario["per_share"] <= scenario["hi"]

    for key in ("bear", "base", "bull"):
        fv = result["fair_value_range"][key]
        assert fv["lo"] is not None and fv["hi"] is not None
        assert fv["growth"].endswith("büyüme")
        assert fv["discount_rate"].startswith("%")

    assert result["triangulation"]["confidence"] in ("YÜKSEK", "ORTA", "DÜŞÜK")
    assert isinstance(result["notes"], list)

    # Sector-relative multiples comparison block (VALUATION.md Sec.7 axis-b)
    # is always shaped, even with no price history / no sector data: the five
    # keys exist so the report renderer can read them unconditionally. Here
    # (price_df=None -> no percentile history -> no primary) every value is
    # None, i.e. axis-b is disabled and the pure own-history signal stands.
    comparison = result["multiples"]["sector"]["comparison"]
    assert set(comparison) == {"label", "current", "median", "ratio", "bucket"}
    assert all(comparison[k] is None for k in comparison)


def test_run_valuation_disables_dcf_for_financial_sector():
    normalized = _fake_normalized()
    ratios = _fake_ratios()
    metrics = _fake_metrics()
    assumptions = _fake_assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=20.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    assert result["dcf"]["enabled"] is False
    assert result["dcf"]["disabled_reason"]
    assert result["pb_roe"] is not None
    assert result["pb_roe"]["scenarios"]["base"]["per_share"] is not None


def test_run_valuation_never_raises_on_empty_inputs():
    result = run_valuation(
        {}, [], {}, price=None, price_df=None, assumptions={}, sector_type="mature",
    )
    assert result["dcf"]["scenarios"] is None
    assert result["fair_value_range"]["base"]["lo"] is None
    assert isinstance(result["notes"], list) and result["notes"]
