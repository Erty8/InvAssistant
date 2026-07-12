"""Unit tests for sec_analyzer.interpret.rule_based (the "script" provider).

No network access, no LLM, no randomness -- these tests build synthetic
``normalized``/``ratios`` fixtures directly (matching the shapes produced by
``normalize_facts``/``compute_ratios``) and call ``analyze()`` on them, the
same way ``test_ratios.py`` builds fixtures for ``compute_ratios`` without
round-tripping through a full companyfacts document.
"""

from sec_analyzer.interpret import rule_based

# Concepts rule_based.analyze() pulls directly from the normalized dict via
# to_annual_series(). Everything else the checklist needs (margins, ROE/ROA,
# current ratio, debt/equity, FCF) is supplied through the `ratios` fixture
# instead, since that's what the real compute_ratios() output would carry.
_RAW_CONCEPTS = (
    "Revenue",
    "NetIncome",
    "OperatingCashFlow",
    "StockholdersEquity",
    "SharesOutstanding",
    "EPS",
    "DividendsPaid",
)


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


def _normalized(entity_name, annual_overrides):
    """Build a minimal normalized-facts dict with an `annual` bucket.

    ``annual_overrides`` supplies the concepts under test (each a list of
    ``_record(...)`` dicts); any concept not given defaults to ``None``
    (missing), matching what normalize_facts would produce for a concept
    with no matching tag.
    """
    annual = {c: annual_overrides.get(c) for c in _RAW_CONCEPTS}
    return {
        "cik": 1,
        "entity_name": entity_name,
        "currency": "USD",
        "annual": annual,
        "quarterly": {c: None for c in _RAW_CONCEPTS},
        "missing": [c for c in _RAW_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _RAW_CONCEPTS},
    }


def _ratio_row(fy, period_end, **overrides):
    """Build one per-fiscal-year ratio dict matching compute_ratios' shape.

    All fields default to ``None`` (as compute_ratios would report for
    missing inputs); pass keyword overrides for the fields under test.
    """
    row = {
        "fy": fy,
        "period_end": period_end,
        "net_margin": None,
        "roe": None,
        "current_ratio": None,
        "yoy_revenue_growth": None,
        "yoy_net_income_growth": None,
        "gross_margin": None,
        "operating_margin": None,
        "roa": None,
        "debt_to_equity": None,
        "fcf": None,
        "fcf_margin": None,
    }
    row.update(overrides)
    return row


def _find_check(score, name):
    """Look up one checklist entry by name from an analyze() result's score."""
    for check in score["checks"]:
        if check["name"] == name:
            return check
    raise AssertionError(f"No check named {name!r} in {[c['name'] for c in score['checks']]}")


# ---------------------------------------------------------------------------
# Fixture 1: a healthy, growing, profitable company across 3 fiscal years.
# ---------------------------------------------------------------------------


def _healthy_company():
    normalized = _normalized(
        "Healthy Co",
        {
            "Revenue": [
                _record(2023, "2023-12-31", 1000),
                _record(2022, "2022-12-31", 900),
                _record(2021, "2021-12-31", 800),
            ],
            "NetIncome": [
                _record(2023, "2023-12-31", 120),
                _record(2022, "2022-12-31", 95),
                _record(2021, "2021-12-31", 80),
            ],
            "OperatingCashFlow": [_record(2023, "2023-12-31", 150)],
            "StockholdersEquity": [_record(2023, "2023-12-31", 500)],
            "SharesOutstanding": [_record(2023, "2023-12-31", 20)],
            "EPS": [_record(2023, "2023-12-31", 6.0)],
            "DividendsPaid": [_record(2023, "2023-12-31", 30)],
        },
    )
    ratios = [
        _ratio_row(
            2023,
            "2023-12-31",
            net_margin=0.12,
            roe=0.24,
            current_ratio=1.5,
            yoy_revenue_growth=(1000 - 900) / 900,
            yoy_net_income_growth=(120 - 95) / 95,
            gross_margin=0.4,
            operating_margin=0.2,
            roa=0.1,
            debt_to_equity=0.8,
            fcf=50,
            fcf_margin=0.05,
        ),
        _ratio_row(
            2022,
            "2022-12-31",
            net_margin=95 / 900,
            yoy_revenue_growth=(900 - 800) / 800,
            yoy_net_income_growth=(95 - 80) / 80,
        ),
        _ratio_row(2021, "2021-12-31", net_margin=80 / 800),
    ]
    return normalized, ratios


def test_healthy_company_scores_strong():
    normalized, ratios = _healthy_company()
    result = rule_based.analyze(normalized, ratios)

    assert result["fundamental_verdict"].startswith("strong")
    assert result["_provider"] == "script"
    assert result["_model"] == "rule-based-v1"

    score = result["score"]
    assert score["max_points"] > 0
    assert 0 <= score["points"] <= score["max_points"]
    # A company that clears every check on this fixture should not have any
    # confirmed failures.
    assert all(c["passed"] is not False for c in score["checks"])


def test_healthy_company_fair_value_range_is_positive_and_ordered():
    normalized, ratios = _healthy_company()
    result = rule_based.analyze(normalized, ratios)

    fv = result["fair_value_range"]
    assert fv["unit"] == "USD per share"
    assert fv["low"] is not None and fv["high"] is not None
    assert fv["low"] > 0
    assert fv["low"] < fv["high"]
    assert "Graham" in fv["basis"]


def test_healthy_company_key_ratios_populated():
    normalized, ratios = _healthy_company()
    result = rule_based.analyze(normalized, ratios)

    key_ratios = result["key_ratios"]
    assert key_ratios  # non-empty
    assert key_ratios["net_margin"] == 0.12
    assert key_ratios["current_ratio"] == 1.5
    assert key_ratios["debt_to_equity"] == 0.8


def test_analyze_is_deterministic():
    """Calling analyze() twice on identical inputs must yield identical output."""
    normalized, ratios = _healthy_company()
    first = rule_based.analyze(normalized, ratios)
    second = rule_based.analyze(normalized, ratios)
    assert first == second


# ---------------------------------------------------------------------------
# Fixture 2: everything missing (e.g. an IFRS/20-F filer with no us-gaap data).
# ---------------------------------------------------------------------------


def test_all_data_missing_reports_insufficient_without_raising():
    normalized = _normalized("Mystery Filer", {})
    ratios = []

    result = rule_based.analyze(normalized, ratios)

    assert "insufficient" in result["fundamental_verdict"].lower()
    assert result["fair_value_range"]["low"] is None
    assert result["fair_value_range"]["high"] is None
    assert result["score"]["max_points"] == 0
    assert result["score"]["points"] == 0
    assert result["_provider"] == "script"
    assert result["summary"]  # a non-empty explanatory summary is still produced


# ---------------------------------------------------------------------------
# Fixture 3: profitable, but a weak balance sheet and negative free cash flow.
# ---------------------------------------------------------------------------


def test_weak_balance_sheet_fails_specific_checks():
    normalized = _normalized(
        "Levered Co",
        {
            "Revenue": [_record(2023, "2023-12-31", 600)],
            "NetIncome": [_record(2023, "2023-12-31", 50)],
            "StockholdersEquity": [_record(2023, "2023-12-31", 100)],
        },
    )
    ratios = [
        _ratio_row(
            2023,
            "2023-12-31",
            net_margin=50 / 600,
            roe=0.5,
            current_ratio=0.6,
            debt_to_equity=3.5,
            fcf=-20,
            fcf_margin=-20 / 600,
        )
    ]

    result = rule_based.analyze(normalized, ratios)
    score = result["score"]

    assert _find_check(score, "Liquidity")["passed"] is False
    assert _find_check(score, "Leverage")["passed"] is False
    assert _find_check(score, "FCF positive")["passed"] is False
    # The company is still profitable on this fixture -- that check should
    # not be swept up as a failure by the weak balance sheet.
    assert _find_check(score, "Profitable")["passed"] is True
