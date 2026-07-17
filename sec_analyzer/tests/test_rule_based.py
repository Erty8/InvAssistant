"""Unit tests for sec_analyzer.interpret.rule_based (the "script" provider).

No network access, no LLM, no randomness -- these tests build synthetic
``normalized``/``ratios``/``metrics``/``technical``/``red_flags``/
``catalyst`` fixtures directly (matching the shapes produced by
``normalize_facts``/``compute_ratios``/``compute_metrics``/
``technical_verdict``/``detect_red_flags``/``estimate_next_earnings``) and
call ``analyze()`` on them, the same way ``test_ratios.py`` builds fixtures
for ``compute_ratios`` without round-tripping through a full companyfacts
document.
"""

import pytest

from sec_analyzer.interpret import rule_based
from sec_analyzer.valuation.sanity import validate_assumptions

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


def test_healthy_company_checklist_score_is_high():
    normalized, ratios = _healthy_company()
    result = rule_based.analyze(normalized, ratios)

    assert result["_provider"] == "script"
    assert result["_model"] == "rule-based-v2"
    assert result["fundamental_verdict"] in ("UCUZ", "MAKUL", "PAHALI")

    score = result["score"]
    assert score["max_points"] > 0
    assert 0 <= score["points"] <= score["max_points"]
    # A company that clears every check on this fixture should not have any
    # confirmed failures, and should clear the "strong" tier (>=75%).
    assert all(c["passed"] is not False for c in score["checks"])
    assert score["points"] / score["max_points"] >= 0.75


def test_healthy_company_scenario_ordering_and_transparency():
    """Every scenario must expose lo<hi plus non-empty growth/discount_rate/
    note strings (transparency), and bear <= base <= bull by construction
    (increasing growth, decreasing discount rate from bear to bull)."""
    normalized, ratios = _healthy_company()
    metrics = {"fcf_per_share": 5.0, "revenue_cagr_3y": 0.1}

    result = rule_based.analyze(normalized, ratios, metrics=metrics)
    fv = result["fair_value_range"]

    assert set(fv.keys()) == {"bear", "base", "bull"}
    for name in ("bear", "base", "bull"):
        scenario = fv[name]
        assert scenario["lo"] is not None and scenario["hi"] is not None
        assert scenario["lo"] > 0
        assert scenario["lo"] < scenario["hi"]
        assert scenario["growth"]
        assert scenario["discount_rate"]
        assert scenario["note"]

    assert fv["bear"]["lo"] <= fv["base"]["lo"] <= fv["bull"]["lo"]
    assert fv["bear"]["hi"] <= fv["base"]["hi"] <= fv["bull"]["hi"]
    assert fv["bear"]["discount_rate"] == "%12"
    assert fv["base"]["discount_rate"] == "%10"
    assert fv["bull"]["discount_rate"] == "%9"


def test_healthy_company_fundamental_verdict_matches_price_vs_base_band():
    """UCUZ below the base band's low, MAKUL inside it, PAHALI above its high."""
    normalized, ratios = _healthy_company()
    metrics = {"fcf_per_share": 5.0, "revenue_cagr_3y": 0.1}

    band_only = rule_based.analyze(normalized, ratios, metrics=metrics)
    lo, hi = band_only["fair_value_range"]["base"]["lo"], band_only["fair_value_range"]["base"]["hi"]
    assert lo is not None and hi is not None

    cheap = rule_based.analyze(normalized, ratios, metrics=dict(metrics, price=lo - 1))
    fair = rule_based.analyze(normalized, ratios, metrics=dict(metrics, price=(lo + hi) / 2))
    expensive = rule_based.analyze(normalized, ratios, metrics=dict(metrics, price=hi + 1))

    assert cheap["fundamental_verdict"] == "UCUZ"
    assert fair["fundamental_verdict"] == "MAKUL"
    assert expensive["fundamental_verdict"] == "PAHALI"


def test_analyze_is_deterministic():
    """Calling analyze() twice on identical inputs must yield identical output."""
    normalized, ratios = _healthy_company()
    metrics = {"fcf_per_share": 5.0, "revenue_cagr_3y": 0.1, "price": 80.0}
    first = rule_based.analyze(normalized, ratios, metrics=metrics)
    second = rule_based.analyze(normalized, ratios, metrics=metrics)
    assert first == second


# ---------------------------------------------------------------------------
# Fixture 2: everything missing (e.g. an IFRS/20-F filer with no us-gaap data).
# ---------------------------------------------------------------------------


def test_all_data_missing_reports_null_band_without_raising():
    normalized = _normalized("Mystery Filer", {})
    ratios = []

    result = rule_based.analyze(normalized, ratios)

    fv = result["fair_value_range"]
    for name in ("bear", "base", "bull"):
        assert fv[name]["lo"] is None
        assert fv[name]["hi"] is None
        # Transparency: growth/discount_rate stay populated even when no
        # per-share anchor was available to compute lo/hi.
        assert fv[name]["growth"]
        assert fv[name]["discount_rate"]

    assert result["fundamental_verdict"] == "MAKUL"
    assert "eksik" in result["horizon_note"] or "Not:" in result["horizon_note"]
    assert result["score"]["max_points"] == 0
    assert result["score"]["points"] == 0
    assert result["_provider"] == "script"
    assert result["_model"] == "rule-based-v2"
    assert result["summary"]  # a non-empty explanatory summary is still produced


# ---------------------------------------------------------------------------
# Fixture 3: profitable, but a weak balance sheet and negative free cash flow.
# ---------------------------------------------------------------------------


def _weak_balance_sheet_fixture():
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
    return normalized, ratios


def test_weak_balance_sheet_fails_specific_checks():
    normalized, ratios = _weak_balance_sheet_fixture()

    result = rule_based.analyze(normalized, ratios)
    score = result["score"]

    assert _find_check(score, "Liquidity")["passed"] is False
    assert _find_check(score, "Leverage")["passed"] is False
    assert _find_check(score, "FCF positive")["passed"] is False
    # The company is still profitable on this fixture -- that check should
    # not be swept up as a failure by the weak balance sheet.
    assert _find_check(score, "Profitable")["passed"] is True


def test_weak_balance_sheet_key_risks_include_failed_checks_and_red_flags():
    normalized, ratios = _weak_balance_sheet_fixture()
    red_flags = [
        {"code": "DILUTION", "message": "Hisse sayısı hızla artıyor", "detail": "..."},
    ]

    result = rule_based.analyze(normalized, ratios, red_flags=red_flags)

    assert "Liquidity" in result["key_risks"]
    assert "Leverage" in result["key_risks"]
    assert "FCF positive" in result["key_risks"]
    assert "Hisse sayısı hızla artıyor" in result["key_risks"]
    assert len(result["key_risks"]) <= 5
    assert result["red_flags_comment"] == "Hisse sayısı hızla artıyor"


def test_no_red_flags_yields_yok_comment():
    normalized, ratios = _healthy_company()
    result = rule_based.analyze(normalized, ratios)
    assert result["red_flags_comment"] == "yok"


# ---------------------------------------------------------------------------
# catalyst / technical_verdict / profile_fit / horizon_note
# ---------------------------------------------------------------------------


def test_catalyst_defaults_to_bilinmiyor_and_uses_label_when_given():
    normalized, ratios = _healthy_company()

    without_catalyst = rule_based.analyze(normalized, ratios)
    assert without_catalyst["catalyst"] == "bilinmiyor"

    catalyst = {"estimate_date": "2026-08-27", "label": "Q2 earnings ~27 Ağu", "based_on": "x"}
    with_catalyst = rule_based.analyze(normalized, ratios, catalyst=catalyst)
    assert with_catalyst["catalyst"] == "Q2 earnings ~27 Ağu"


def test_technical_verdict_text_reflects_technical_arg():
    normalized, ratios = _healthy_company()

    without_technical = rule_based.analyze(normalized, ratios)
    assert without_technical["technical_verdict"] == "VERİ YOK (fiyat verisi alınamadı)"

    technical = {"verdict": "NÖTR", "verdict_detail": "yetersiz veri"}
    with_technical = rule_based.analyze(normalized, ratios, technical=technical)
    assert with_technical["technical_verdict"] == "NÖTR (yetersiz veri)"


def test_profile_fit_is_kismen_and_mentions_missing_file_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(rule_based.Config, "PROFIL_PATH", str(tmp_path / "does_not_exist.md"))
    normalized, ratios = _healthy_company()

    result = rule_based.analyze(normalized, ratios)

    assert result["profile_fit"]["verdict"] == "KISMEN"
    assert "PROFIL.md" in result["profile_fit"]["reason"]


def test_profile_fit_is_kismen_and_mentions_llm_when_profile_exists(monkeypatch, tmp_path):
    profile_path = tmp_path / "PROFIL.md"
    profile_path.write_text("Risk toleransı: düşük.", encoding="utf-8")
    monkeypatch.setattr(rule_based.Config, "PROFIL_PATH", str(profile_path))
    normalized, ratios = _healthy_company()

    result = rule_based.analyze(normalized, ratios)

    assert result["profile_fit"]["verdict"] == "KISMEN"
    assert "LLM" in result["profile_fit"]["reason"]


def test_horizon_note_5y_includes_cyclical_trap_flag_message_when_present():
    normalized, ratios = _healthy_company()
    red_flags = [
        {"code": "CYCLICAL_TRAP", "message": "Düşük P/E yanıltıcı olabilir", "detail": "..."},
    ]

    result = rule_based.analyze(normalized, ratios, red_flags=red_flags, horizon="5y")

    assert "5 yıllık" in result["horizon_note"]
    assert "Düşük P/E yanıltıcı olabilir" in result["horizon_note"]


def test_horizon_note_5y_without_cyclical_trap_says_not_triggered():
    normalized, ratios = _healthy_company()

    result = rule_based.analyze(normalized, ratios, horizon="5y")

    assert "tetiklenmedi" in result["horizon_note"]


def test_horizon_note_3m_mentions_katalizor():
    normalized, ratios = _healthy_company()

    result = rule_based.analyze(normalized, ratios, horizon="3m")

    assert "katalizör" in result["horizon_note"]


# ---------------------------------------------------------------------------
# default_assumptions() -- the two-phase flow's phase-1 deterministic
# fallback (SPEC.md Sec.12). Independent of analyze()/_fair_value_scenarios
# above -- see the module docstring.
# ---------------------------------------------------------------------------

def test_default_assumptions_always_passes_validate_assumptions():
    """The formula is designed to never violate a sanity rule -- check this
    across a spread of inputs, including missing/extreme metrics."""
    cases = [
        ({}, None),
        ({}, "mature"),
        ({}, "growth_unprofitable"),
        ({"revenue_cagr_5y": 0.35}, "cyclical"),
        ({"revenue_cagr_5y": -0.3}, "financial"),
        ({"revenue_cagr_3y": 0.6}, "reit"),
        (None, "growth_unprofitable"),
    ]
    for metrics, sector_type in cases:
        assumptions = rule_based.default_assumptions(metrics, sector_type)
        violations = validate_assumptions(assumptions, is_unprofitable=(sector_type == "growth_unprofitable"))
        assert violations == [], f"metrics={metrics!r} sector_type={sector_type!r} -> {violations}"


def test_default_assumptions_shape_and_scenario_ordering():
    assumptions = rule_based.default_assumptions({"revenue_cagr_5y": 0.10}, "mature")

    assert set(assumptions.keys()) == {"bear", "base", "bull"}
    for scenario in assumptions.values():
        assert set(scenario.keys()) == {"growth_5y", "terminal_growth", "discount_rate", "story"}
        assert scenario["terminal_growth"] == 0.025
        assert scenario["story"]

    assert assumptions["bear"]["growth_5y"] < assumptions["base"]["growth_5y"] < assumptions["bull"]["growth_5y"]
    assert assumptions["base"]["growth_5y"] == 0.10


def test_default_assumptions_prefers_5y_cagr_over_3y():
    assumptions = rule_based.default_assumptions({"revenue_cagr_5y": 0.15, "revenue_cagr_3y": 0.30}, "mature")
    assert assumptions["base"]["growth_5y"] == 0.15


def test_default_assumptions_falls_back_to_3y_then_flat_4_percent():
    with_3y = rule_based.default_assumptions({"revenue_cagr_3y": 0.20}, "mature")
    assert with_3y["base"]["growth_5y"] == 0.20

    with_neither = rule_based.default_assumptions({}, "mature")
    assert with_neither["base"]["growth_5y"] == 0.04


def test_default_assumptions_clamps_extreme_growth():
    too_high = rule_based.default_assumptions({"revenue_cagr_5y": 0.90}, "mature")
    assert too_high["base"]["growth_5y"] == 0.25

    too_low = rule_based.default_assumptions({"revenue_cagr_5y": -0.80}, "mature")
    assert too_low["base"]["growth_5y"] == -0.05


def test_default_assumptions_raises_discount_rate_floor_when_unprofitable():
    profitable = rule_based.default_assumptions({}, "mature")
    unprofitable = rule_based.default_assumptions({}, "growth_unprofitable")

    assert profitable["base"]["discount_rate"] == pytest.approx(0.10)
    assert unprofitable["base"]["discount_rate"] == pytest.approx(0.12)
    assert unprofitable["bear"]["discount_rate"] == pytest.approx(0.14)
    assert unprofitable["bull"]["discount_rate"] == pytest.approx(0.11)


def test_default_assumptions_is_deterministic():
    metrics = {"revenue_cagr_5y": 0.12}
    first = rule_based.default_assumptions(metrics, "cyclical")
    second = rule_based.default_assumptions(metrics, "cyclical")
    assert first == second


# ---------------------------------------------------------------------------
# risk_free_pct global fallback for terminal_growth (see
# rule_based._terminal_growth_anchor's 3-step resolution order): covers
# filers whose SIC doesn't match any Damodaran industry, so `capm` is None
# even though the GLOBAL risk-free rate (independent of SIC matching) is
# still available -- this must not also flatten terminal_growth to the old
# 2.5% constant.
# ---------------------------------------------------------------------------

def test_default_assumptions_uses_global_risk_free_pct_when_capm_is_none():
    metrics = {"revenue_cagr_5y": 0.06}
    assumptions = rule_based.default_assumptions(metrics, "mature", capm=None, risk_free_pct=4.20)
    for scenario in assumptions.values():
        assert scenario["terminal_growth"] == pytest.approx(0.04)


def test_default_assumptions_keeps_flat_default_when_no_risk_free_source_at_all():
    metrics = {"revenue_cagr_5y": 0.06}
    assumptions = rule_based.default_assumptions(metrics, "mature", capm=None, risk_free_pct=None)
    for scenario in assumptions.values():
        assert scenario["terminal_growth"] == pytest.approx(0.025)


def test_default_assumptions_capm_risk_free_takes_precedence_over_global_fallback():
    metrics = {"revenue_cagr_5y": 0.06}
    assumptions = rule_based.default_assumptions(
        metrics, "mature", capm={"risk_free": 3.0}, risk_free_pct=4.20
    )
    for scenario in assumptions.values():
        assert scenario["terminal_growth"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# commentary() -- the two-phase flow's phase-2 deterministic ("script"
# provider) commentary over an already-computed `valuation` dict.
# ---------------------------------------------------------------------------


def _valuation(dcf_signal="ucuz", confidence="YÜKSEK", sector_type="mature", high_uncertainty=False):
    """A `valuation` dict matching `sec_analyzer/valuation/SPEC.md` Sec.11,
    parameterized by the fields `commentary()` reads from."""
    return {
        "sector_type": sector_type,
        "fcf0": 120.0,
        "fcf0_source": "ttm",
        "dcf": {"enabled": True, "disabled_reason": None, "scenarios": None, "normalized_variant": None},
        "pb_roe": None,
        "fair_value_range": {
            "bear": {"lo": 72.0, "hi": 88.0, "growth": "%8 büyüme", "discount_rate": "%12", "note": "bear"},
            "base": {"lo": 90.0, "hi": 110.0, "growth": "%12 büyüme", "discount_rate": "%10", "note": "base"},
            "bull": {"lo": 108.0, "hi": 132.0, "growth": "%16 büyüme", "discount_rate": "%9", "note": "bull"},
        },
        "reverse_dcf": {"implied_growth": 0.19, "realized_cagr_5y": 0.14, "realized_label": "5y"},
        "multiples": {
            "history": [], "current": {"pe": None, "ps": None, "pfcf": None},
            "pe_percentile": None, "ps_percentile": None, "pfcf_percentile": None,
            "history_years": 0,
            "sector": {"available": False, "industry": None, "pe_median": None, "ps_median": None, "pfcf_median": None},
        },
        "sensitivity": {"high_uncertainty": high_uncertainty} if high_uncertainty is not None else None,
        "triangulation": {
            "signals": {"dcf": dcf_signal, "reverse_dcf": dcf_signal, "multiples": dcf_signal},
            "confidence": confidence,
            "direction": dcf_signal,
        },
        "assumptions": {},
        "notes": ["fcf0 = TTM FCF kullanıldı."],
    }


def test_commentary_returns_exactly_the_phase2_schema_keys():
    result = rule_based.commentary(_valuation())

    assert set(result.keys()) == {
        "fundamental_verdict", "profile_fit", "reverse_dcf_comment", "cyclical_risk",
        "horizon_note", "key_risks", "red_flags_comment", "catalyst", "summary",
    }
    # Deliberately not present -- interpret_results() always injects these.
    assert "fair_value_range" not in result
    assert "technical_verdict" not in result
    assert "confidence" not in result
    assert "valuation" not in result
    # Deliberately not present -- these four are computed downstream by
    # sec_analyzer.interpret.planning and injected uniformly for every
    # provider in analyzer._postprocess_phase2_result(), not by
    # commentary() itself (see test_interpret.py for the injection tests).
    assert "scenario_returns" not in result
    assert "entry_plan" not in result
    assert "stop_adding" not in result
    assert "thesis_metric" not in result


def test_commentary_fundamental_verdict_matches_dcf_triangulation_signal():
    assert rule_based.commentary(_valuation(dcf_signal="ucuz"))["fundamental_verdict"] == "UCUZ"
    assert rule_based.commentary(_valuation(dcf_signal="makul"))["fundamental_verdict"] == "MAKUL"
    assert rule_based.commentary(_valuation(dcf_signal="pahali"))["fundamental_verdict"] == "PAHALI"
    assert rule_based.commentary(_valuation(dcf_signal="veri_yok"))["fundamental_verdict"] == "MAKUL"


def test_commentary_reverse_dcf_comment_reflects_implied_vs_realized_growth():
    expensive = rule_based.commentary(_valuation())["reverse_dcf_comment"]
    assert "%19" in expensive and "%14" in expensive
    assert "pahalılık" in expensive

    cheap_valuation = _valuation()
    cheap_valuation["reverse_dcf"] = {"implied_growth": 0.05, "realized_cagr_5y": 0.14, "realized_label": "5y"}
    cheap = rule_based.commentary(cheap_valuation)["reverse_dcf_comment"]
    assert "ucuzluk" in cheap

    fair_valuation = _valuation()
    fair_valuation["reverse_dcf"] = {"implied_growth": 0.14, "realized_cagr_5y": 0.13, "realized_label": "5y"}
    fair = rule_based.commentary(fair_valuation)["reverse_dcf_comment"]
    assert "uyumlu" in fair

    no_data_valuation = _valuation()
    no_data_valuation["reverse_dcf"] = {"implied_growth": None, "realized_cagr_5y": None, "realized_label": None}
    no_data = rule_based.commentary(no_data_valuation)["reverse_dcf_comment"]
    assert "hesaplanamadı" in no_data


def test_commentary_cyclical_risk_reflects_sector_type_and_cyclical_trap_flag():
    cyclical = rule_based.commentary(_valuation(sector_type="cyclical"))["cyclical_risk"]
    assert "döngüsel" in cyclical.lower()

    red_flags = [{"code": "CYCLICAL_TRAP", "message": "Düşük P/E yanıltıcı olabilir", "detail": "..."}]
    with_flag = rule_based.commentary(_valuation(sector_type="cyclical"), red_flags=red_flags)["cyclical_risk"]
    assert "Düşük P/E yanıltıcı olabilir" in with_flag


def test_commentary_horizon_note_flags_high_uncertainty():
    calm = rule_based.commentary(_valuation(high_uncertainty=False), horizon="1y")["horizon_note"]
    assert "yüksek belirsizlik" not in calm

    uncertain = rule_based.commentary(_valuation(high_uncertainty=True), horizon="1y")["horizon_note"]
    assert "yüksek belirsizlik" in uncertain


def test_commentary_key_risks_include_red_flags_and_valuation_notes():
    red_flags = [{"code": "SBC_HIGH", "message": "Hisse bazlı ödemeler yüksek", "detail": "..."}]
    result = rule_based.commentary(_valuation(), red_flags=red_flags)

    assert "Hisse bazlı ödemeler yüksek" in result["key_risks"]
    assert "fcf0 = TTM FCF kullanıldı." in result["key_risks"]
    assert len(result["key_risks"]) <= 5


def test_commentary_catalyst_and_red_flags_comment_defaults():
    result = rule_based.commentary(_valuation())
    assert result["catalyst"] == "bilinmiyor"
    assert result["red_flags_comment"] == "yok"

    catalyst = {"estimate_date": "2026-08-27", "label": "Q2 earnings ~27 Ağu", "based_on": "x"}
    red_flags = [{"code": "X", "message": "bir bayrak", "detail": "..."}]
    with_data = rule_based.commentary(_valuation(), red_flags=red_flags, catalyst=catalyst)
    assert with_data["catalyst"] == "Q2 earnings ~27 Ağu"
    assert with_data["red_flags_comment"] == "bir bayrak"


def test_commentary_summary_is_grammatical_with_and_without_fair_value_band():
    # With a band present, the summary reads "...baz senaryoda $X-Y
    # aralığını işaret ediyor; ...".
    with_band = rule_based.commentary(_valuation())["summary"]
    assert "baz senaryoda $90.00-110.00 aralığını işaret ediyor;" in with_band

    # Without a band, it must NOT glue "hesaplayamadı" and "işaret ediyor"
    # together into the old ungrammatical sentence -- it should read
    # "...baz senaryoda adil değer aralığı hesaplayamadı; ...".
    no_band_valuation = _valuation()
    no_band_valuation["fair_value_range"] = {
        key: {"lo": None, "hi": None, "growth": None, "discount_rate": None, "note": None}
        for key in ("bear", "base", "bull")
    }
    without_band = rule_based.commentary(no_band_valuation)["summary"]
    assert "baz senaryoda adil değer aralığı hesaplayamadı;" in without_band
    assert "hesaplayamadı işaret ediyor" not in without_band


def test_commentary_handles_none_valuation_without_raising():
    result = rule_based.commentary(None)
    assert result["fundamental_verdict"] == "MAKUL"
    assert result["catalyst"] == "bilinmiyor"


def test_commentary_is_deterministic():
    valuation = _valuation()
    first = rule_based.commentary(valuation, horizon="5y")
    second = rule_based.commentary(valuation, horizon="5y")
    assert first == second
