"""Unit tests for sec_analyzer.interpret.analyzer's two-phase valuation flow.

These tests never touch the network: only the LLM transport boundary
(`_call_ollama` / `_call_anthropic`, or the shared `_dispatch_llm_call`) is
monkeypatched out. Everything else -- fence-stripping, JSON parsing, sanity
validation (`sec_analyzer.valuation.sanity.validate_assumptions`), the
revision-retry loop, the deterministic-fallback wiring, and the
provider-agnostic phase-2 post-processing (`technical_verdict`/`confidence`/
`fair_value_range` injection, the `fundamental_verdict` cross-check, and
`_provider`/`_model`/`_horizon`/`_weights` stamping) -- runs for real.
"""

import json

import pytest

from sec_analyzer.interpret import analyzer, rule_based
from sec_analyzer.interpret.ollama_client import OllamaError


def _normalized(annual=None):
    """Minimal normalized-facts dict; content is irrelevant unless a test
    needs specific annual data (e.g. to drive run_valuation for real)."""
    return {
        "entity_name": "X",
        "currency": "USD",
        "annual": annual or {},
        "quarterly": {},
        "missing": [],
        "matched_tags": {},
    }


def _record(fy, period_end, value):
    return {"fy": fy, "period_end": period_end, "value": value}


def _financials_for_valuation():
    """A small but complete normalized/ratios/metrics fixture realistic
    enough for `sec_analyzer.valuation.engine.run_valuation` to produce a
    real (non-degraded) result -- used by the fully-offline end-to-end test."""
    normalized = _normalized(
        {
            "Revenue": [
                _record(2023, "2023-12-31", 1000.0),
                _record(2022, "2022-12-31", 900.0),
                _record(2021, "2021-12-31", 800.0),
            ],
            "NetIncome": [_record(2023, "2023-12-31", 120.0)],
            "OperatingCashFlow": [_record(2023, "2023-12-31", 150.0)],
            "CapEx": [_record(2023, "2023-12-31", 30.0)],
            "SharesOutstanding": [_record(2023, "2023-12-31", 100.0)],
        }
    )
    ratios = [{"fy": 2023, "period_end": "2023-12-31", "fcf": 120.0, "roe": 0.2}]
    metrics = {
        "price": 50.0, "shares": 100.0, "eps": 1.2, "net_debt": 0.0,
        "pe": None, "ps": None, "pfcf": None,
        "revenue_cagr_3y": 0.1, "revenue_cagr_5y": None,
        "sbc_revenue": None, "shares_yoy": None, "fcf": 120.0, "latest_fy": 2023,
    }
    return normalized, ratios, metrics


def _valuation_fixture(dcf_signal="ucuz", confidence="YÜKSEK", sector_type="mature", divergence=None):
    """A `valuation` dict matching `sec_analyzer/valuation/SPEC.md` Sec.11
    (the shape `run_valuation` returns), parameterized by the triangulated
    DCF signal so tests can drive the fundamental_verdict cross-check."""
    return {
        "sector_type": sector_type,
        "fcf0": 120.0,
        "fcf0_source": "ttm",
        "dcf": {
            "enabled": True,
            "disabled_reason": None,
            "scenarios": {
                "bear": {"per_share": 80.0, "lo": 72.0, "hi": 88.0},
                "base": {"per_share": 100.0, "lo": 90.0, "hi": 110.0},
                "bull": {"per_share": 120.0, "lo": 108.0, "hi": 132.0},
            },
            "normalized_variant": None,
        },
        "pb_roe": None,
        "fair_value_range": {
            "bear": {"lo": 72.0, "hi": 88.0, "growth": "%8 büyüme", "discount_rate": "%12", "note": "bear"},
            "base": {"lo": 90.0, "hi": 110.0, "growth": "%12 büyüme", "discount_rate": "%10", "note": "base"},
            "bull": {"lo": 108.0, "hi": 132.0, "growth": "%16 büyüme", "discount_rate": "%9", "note": "bull"},
        },
        "reverse_dcf": {"implied_growth": 0.19, "realized_cagr_5y": 0.14, "realized_label": "5y"},
        "multiples": {
            "history": [],
            "current": {"pe": None, "ps": None, "pfcf": None},
            "pe_percentile": None, "ps_percentile": None, "pfcf_percentile": None,
            "history_years": 0,
            "sector": {"available": False, "industry": None, "pe_median": None, "ps_median": None, "pfcf_median": None},
        },
        "sensitivity": None,
        "triangulation": {
            "signals": {"dcf": dcf_signal, "reverse_dcf": dcf_signal, "multiples": dcf_signal},
            "confidence": confidence,
            "direction": dcf_signal,
            "divergence": divergence,
        },
        "assumptions": {
            "bear": {"growth_5y": 0.08, "terminal_growth": 0.025, "discount_rate": 0.12, "story": "s"},
            "base": {"growth_5y": 0.12, "terminal_growth": 0.025, "discount_rate": 0.10, "story": "s"},
            "bull": {"growth_5y": 0.16, "terminal_growth": 0.025, "discount_rate": 0.09, "story": "s"},
        },
        "notes": [],
    }


def _valid_assumptions_json(bear=0.08, base=0.12, bull=0.16, discount_rate=0.10, sector_type="mature"):
    return json.dumps(
        {
            "assumptions": {
                "bear": {"growth_5y": bear, "terminal_growth": 0.025, "discount_rate": discount_rate + 0.02, "story": "Bear."},
                "base": {"growth_5y": base, "terminal_growth": 0.025, "discount_rate": discount_rate, "story": "Base."},
                "bull": {"growth_5y": bull, "terminal_growth": 0.025, "discount_rate": discount_rate - 0.01, "story": "Bull."},
            },
            "sector_type": sector_type,
        }
    )


def _commentary_json(**overrides):
    payload = {
        "fundamental_verdict": "MAKUL",
        "profile_fit": {"verdict": "KISMEN", "reason": "test"},
        "reverse_dcf_comment": "test",
        "cyclical_risk": "test",
        "horizon_note": "test",
        "key_risks": [],
        "red_flags_comment": "yok",
        "catalyst": "bilinmiyor",
        "summary": "test",
    }
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# _strip_json_fence / _parse_model_json (unchanged by the two-phase refactor)
# ---------------------------------------------------------------------------


def test_strip_json_fence_removes_json_tagged_fence():
    text = '```json\n{"a": 1}\n```'
    assert analyzer._strip_json_fence(text) == '{"a": 1}'


def test_strip_json_fence_removes_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert analyzer._strip_json_fence(text) == '{"a": 1}'


def test_strip_json_fence_passes_through_unfenced_text():
    text = '  {"a": 1}  '
    assert analyzer._strip_json_fence(text) == '{"a": 1}'


def test_parse_model_json_valid():
    assert analyzer._parse_model_json('{"summary": "ok"}') == {"summary": "ok"}


def test_parse_model_json_fenced():
    text = '```json\n{"summary": "ok"}\n```'
    assert analyzer._parse_model_json(text) == {"summary": "ok"}


def test_parse_model_json_garbage_returns_error_dict():
    result = analyzer._parse_model_json("not json at all")
    assert result["error"] == "parse_failed"
    assert result["raw"] == "not json at all"
    assert "summary" in result


# ---------------------------------------------------------------------------
# propose_assumptions() -- phase 1
# ---------------------------------------------------------------------------


def test_propose_assumptions_script_provider_returns_deterministic_default():
    result = analyzer.propose_assumptions(_normalized(), [], metrics={"revenue_cagr_3y": 0.1}, provider="script")

    assert result["_provider"] == "script"
    assert result["sector_type"] == "mature"
    assert result["assumptions"] == rule_based.default_assumptions({"revenue_cagr_3y": 0.1}, "mature")
    assert analyzer.validate_assumptions(result["assumptions"]) == []


def test_propose_assumptions_uses_valid_llm_proposal_as_is(monkeypatch):
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: _valid_assumptions_json(sector_type="cyclical")
    )

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama")

    assert result["sector_type"] == "cyclical"
    assert result["assumptions"]["base"]["growth_5y"] == 0.12
    assert result["assumptions"]["base"]["discount_rate"] == 0.10
    assert result["_provider"] == "ollama"


def test_propose_assumptions_retries_once_on_sanity_violation_then_succeeds(monkeypatch):
    """First LLM response has an invalid discount rate (< 7%); the second
    (revision) response is valid -- the valid, revised proposal must win,
    and the transport must have been called exactly twice."""
    responses = [
        _valid_assumptions_json(discount_rate=0.05),  # violates discount_rate >= 0.07
        _valid_assumptions_json(discount_rate=0.11),  # valid
    ]
    calls = []

    def _fake_call_ollama(system, user, model, host):
        calls.append(user)
        return responses[len(calls) - 1]

    monkeypatch.setattr(analyzer, "_call_ollama", _fake_call_ollama)

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama")

    assert len(calls) == 2
    # The revision call must carry the violation list, in Turkish, per SPEC.md Sec.12.
    assert "ihlal edildi" in calls[1]
    assert result["assumptions"]["base"]["discount_rate"] == 0.11
    assert analyzer.validate_assumptions(result["assumptions"]) == []


def test_propose_assumptions_falls_back_after_two_invalid_responses(monkeypatch):
    """Both the original and the revised proposal are invalid -> the
    deterministic default_assumptions fallback is used instead of ever
    handing back an assumption set that fails validate_assumptions."""
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: _valid_assumptions_json(discount_rate=0.05)
    )

    result = analyzer.propose_assumptions(_normalized(), [], metrics={"revenue_cagr_3y": 0.1}, provider="ollama")

    assert result["_provider"] == "script"
    assert result["assumptions"] == rule_based.default_assumptions({"revenue_cagr_3y": 0.1}, result["sector_type"])


def test_propose_assumptions_falls_back_when_llm_unavailable(monkeypatch):
    def _raise(system, user, model, host):
        raise OllamaError("Ollama is not running.")

    monkeypatch.setattr(analyzer, "_call_ollama", _raise)

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama", sector_hint="reit")

    assert result["_provider"] == "script"
    assert result["sector_type"] == "reit"
    assert result["assumptions"] == rule_based.default_assumptions({}, "reit")


def test_propose_assumptions_falls_back_on_unparseable_json(monkeypatch):
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: "not json at all")

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama")

    assert result["_provider"] == "script"
    assert analyzer.validate_assumptions(result["assumptions"]) == []


def test_propose_assumptions_uses_sector_hint_when_llm_omits_sector_type(monkeypatch):
    payload = json.loads(_valid_assumptions_json())
    del payload["sector_type"]
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: json.dumps(payload))

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama", sector_hint="financial")

    assert result["sector_type"] == "financial"


# ---------------------------------------------------------------------------
# propose_assumptions() -- optional hyper_growth_extras (HYPER_SPEC.md Sec.5)
# ---------------------------------------------------------------------------


def _hyper_growth_extras_payload():
    return {
        "tam_usd": 5_000_000_000.0,
        "tam_rationale": "Global bulut depolama pazarı.",
        "per_scenario": {
            "bear": {"target_fcf_margin": 0.15, "steady_state_year": 8, "probability": 0.25},
            "base": {"target_fcf_margin": 0.25, "steady_state_year": 10, "probability": 0.5},
            "bull": {"target_fcf_margin": 0.35, "steady_state_year": 10, "probability": 0.25},
        },
    }


def test_propose_assumptions_extracts_hyper_growth_extras_when_present(monkeypatch):
    payload = json.loads(_valid_assumptions_json(sector_type="hyper_growth"))
    payload["hyper_growth_extras"] = _hyper_growth_extras_payload()
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: json.dumps(payload))

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama")

    assert result["sector_type"] == "hyper_growth"
    assert result["hyper_growth_extras"] == _hyper_growth_extras_payload()


def test_propose_assumptions_hyper_growth_extras_is_none_when_absent(monkeypatch):
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: _valid_assumptions_json())

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama")

    assert result["hyper_growth_extras"] is None


def test_propose_assumptions_hyper_growth_extras_is_none_when_malformed(monkeypatch):
    payload = json.loads(_valid_assumptions_json())
    payload["hyper_growth_extras"] = "not a dict"  # malformed -- must never raise
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: json.dumps(payload))

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama")

    assert result["hyper_growth_extras"] is None


def test_propose_assumptions_script_provider_hyper_growth_extras_is_none():
    result = analyzer.propose_assumptions(_normalized(), [], metrics={"revenue_cagr_3y": 0.1}, provider="script")

    assert result["hyper_growth_extras"] is None


def test_propose_assumptions_revision_keeps_original_hyper_growth_extras(monkeypatch):
    """The revision request only asks the provider to resend "assumptions"
    and "sector_type" (_build_phase1_revision_payload) -- hyper_growth_extras
    from the ORIGINAL (pre-revision) response must survive even though the
    revised response naturally omits it."""
    original = json.loads(_valid_assumptions_json(discount_rate=0.05))  # violates discount_rate >= 0.07
    original["hyper_growth_extras"] = _hyper_growth_extras_payload()
    revised = _valid_assumptions_json(discount_rate=0.11)  # valid, no hyper_growth_extras key at all

    responses = [json.dumps(original), revised]
    calls = []

    def _fake_call_ollama(system, user, model, host):
        calls.append(user)
        return responses[len(calls) - 1]

    monkeypatch.setattr(analyzer, "_call_ollama", _fake_call_ollama)

    result = analyzer.propose_assumptions(_normalized(), [], metrics={}, provider="ollama")

    assert len(calls) == 2
    assert result["assumptions"]["base"]["discount_rate"] == 0.11
    assert result["hyper_growth_extras"] == _hyper_growth_extras_payload()


# ---------------------------------------------------------------------------
# interpret_results() -- phase 2 (code-enforced post-processing)
# ---------------------------------------------------------------------------


def test_interpret_results_injects_confidence_and_fair_value_range_from_valuation(monkeypatch):
    """Even if the model tried to supply its own confidence/fair_value_range,
    the valuation dict's own figures must always win."""
    monkeypatch.setattr(
        analyzer,
        "_call_ollama",
        lambda system, user, model, host: json.dumps(
            {
                **json.loads(_commentary_json()),
                "confidence": "should be ignored",
                "fair_value_range": {"bear": {"lo": 1}, "base": {"lo": 2}, "bull": {"lo": 3}},
            }
        ),
    )
    valuation = _valuation_fixture(dcf_signal="makul", confidence="ORTA")

    result = analyzer.interpret_results(
        _normalized(), [], {}, None, None, None, valuation, provider="ollama",
    )

    assert result["confidence"] == "ORTA"
    assert result["fair_value_range"] == valuation["fair_value_range"]
    assert result["valuation"] is valuation


def test_interpret_results_overrides_contradicting_fundamental_verdict(monkeypatch, caplog):
    """The LLM says UCUZ, but the deterministic DCF signal says pahali --
    an outright contradiction on the ucuz<->pahali axis must be overridden."""
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json(fundamental_verdict="UCUZ")
    )
    valuation = _valuation_fixture(dcf_signal="pahali")

    with caplog.at_level("WARNING"):
        result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="ollama")

    assert result["fundamental_verdict"] == "PAHALI"
    assert any("overriding" in message for message in caplog.messages)


def test_interpret_results_does_not_override_makul_against_a_direction_signal(monkeypatch):
    """MAKUL vs. a directional (ucuz/pahali) code signal is not a
    contradiction on the ucuz<->pahali axis -- MAKUL must survive."""
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json(fundamental_verdict="MAKUL")
    )
    valuation = _valuation_fixture(dcf_signal="pahali")

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="ollama")

    assert result["fundamental_verdict"] == "MAKUL"


def test_interpret_results_keeps_agreeing_fundamental_verdict(monkeypatch):
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json(fundamental_verdict="UCUZ")
    )
    valuation = _valuation_fixture(dcf_signal="ucuz")

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="ollama")

    assert result["fundamental_verdict"] == "UCUZ"


def test_interpret_results_yuksek_beklenti_signal_always_wins_over_llm_verdict(monkeypatch):
    """Unlike the ucuz<->pahali cross-check (which only overrides an
    outright-opposite verdict, letting MAKUL survive unopposed either way),
    the hyper-grower 'yuksek_beklenti' code signal always wins outright --
    no provider is ever asked to produce this 4th value itself, so there is
    nothing legitimate for it to contribute here. Covers every verdict a
    provider's phase-2 contract can actually produce (UCUZ/MAKUL/PAHALI)."""
    for llm_guess in ("UCUZ", "MAKUL", "PAHALI"):
        monkeypatch.setattr(
            analyzer, "_call_ollama",
            lambda system, user, model, host, guess=llm_guess: _commentary_json(fundamental_verdict=guess),
        )
        valuation = _valuation_fixture(dcf_signal="yuksek_beklenti")

        result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="ollama")

        assert result["fundamental_verdict"] == "YÜKSEK BEKLENTİ FİYATLANMIŞ", (
            f"expected the code signal to win over the provider's {llm_guess!r} guess"
        )


def test_interpret_results_yuksek_beklenti_signal_overrides_script_providers_makul_default():
    """The "script" provider's own rule_based.commentary computes its
    fundamental_verdict from a SEPARATE, older _DCF_SIGNAL_TO_VERDICT copy
    (rule_based.py) that doesn't recognize "yuksek_beklenti" and falls back
    to "MAKUL" for it (rule_based._fundamental_verdict_from_valuation). That
    "MAKUL" must not survive interpret_results()'s cross-check -- the
    "yuksek_beklenti" code signal overrides it too, same as it would any
    other provider's guess."""
    valuation = _valuation_fixture(dcf_signal="yuksek_beklenti")

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="script")

    assert result["fundamental_verdict"] == "YÜKSEK BEKLENTİ FİYATLANMIŞ"


def test_interpret_results_divergence_override_wins_over_any_verdict(monkeypatch):
    """The model-market divergence governor (action="verdict") restates the
    headline as MODEL-PİYASA AYRIŞMASI regardless of what any provider (or the
    ucuz<->pahali reconcile) produced -- it's a deterministic, numbers-driven
    relabel applied last. Confidence comes straight from the (already-floored)
    triangulation.confidence, and the thesis-metric rationale is reframed."""
    up_divergence = {"direction": "ucuz", "action": "verdict", "factor": 3.25, "band_edge": 434.38}
    for llm_guess in ("UCUZ", "MAKUL", "PAHALI"):
        monkeypatch.setattr(
            analyzer, "_call_ollama",
            lambda system, user, model, host, guess=llm_guess: _commentary_json(fundamental_verdict=guess),
        )
        valuation = _valuation_fixture(
            dcf_signal="ucuz", confidence="DÜŞÜK", divergence=up_divergence,
        )
        result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="ollama")
        assert result["fundamental_verdict"] == "MODEL-PİYASA AYRIŞMASI"
        assert result["confidence"] == "DÜŞÜK"
        assert "ayrışma" in result["thesis_metric"]["rationale"].lower()


def test_interpret_results_downside_divergence_log_only_does_not_relabel(monkeypatch):
    """A down-side divergence is log-only (action="log_only"): the verdict is
    NOT relabeled -- the normal reconcile stands."""
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json(fundamental_verdict="PAHALI")
    )
    down_divergence = {"direction": "pahali", "action": "log_only", "factor": 0.4, "band_edge": 40.0}
    valuation = _valuation_fixture(dcf_signal="pahali", divergence=down_divergence)
    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="ollama")
    assert result["fundamental_verdict"] == "PAHALI"


def test_interpret_results_overwrites_technical_verdict_from_technical_arg(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "_call_ollama",
        lambda system, user, model, host: _commentary_json(technical_verdict="should be ignored"),
    )
    technical = {"verdict": "AŞIRI ALIM", "verdict_detail": "RSI 74, SMA50 +%12"}
    valuation = _valuation_fixture()

    result = analyzer.interpret_results(_normalized(), [], {}, technical, None, None, valuation, provider="ollama")

    assert result["technical_verdict"] == "AŞIRI ALIM (RSI 74, SMA50 +%12)"


def test_interpret_results_technical_verdict_falls_back_when_no_technical_given(monkeypatch):
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json())
    valuation = _valuation_fixture()

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="ollama")

    assert result["technical_verdict"] == "VERİ YOK (fiyat verisi alınamadı)"


def test_interpret_results_stamps_horizon_weights_and_valuation(monkeypatch):
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json())
    valuation = _valuation_fixture()

    result = analyzer.interpret_results(
        _normalized(), [], {}, None, None, None, valuation, provider="ollama", horizon="5y"
    )

    assert result["_provider"] == "ollama"
    assert result["_horizon"] == "5y"
    assert result["_weights"] == {"fundamental": 0.8, "technical": 0.2}
    assert result["valuation"] == valuation


# ---------------------------------------------------------------------------
# interpret_results() -- the four sec_analyzer.interpret.planning fields
# (scenario_returns/entry_plan/stop_adding/thesis_metric) are ALWAYS
# code-computed and injected by _postprocess_phase2_result(), for every
# provider -- see planning.py's own hand-verified numeric tests in
# test_planning.py for the arithmetic itself; these tests only check that
# analyzer.py actually wires the real inputs (price/valuation/technical/
# ratios/red_flags/metrics) through to planning.py and stamps the results
# onto the phase-2 output, for both the LLM branch and the "script" branch.
# ---------------------------------------------------------------------------


def test_interpret_results_injects_all_four_planning_fields_llm_branch(monkeypatch):
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json())
    # bear.lo=72/hi=88, base.lo=90/hi=110, bull.lo=108/hi=132 (see _valuation_fixture)
    # price=100 -> base: ret_lo=(90/100-1)*100=-10.0, ret_hi=(110/100-1)*100=10.0
    valuation = _valuation_fixture()
    metrics = {"price": 100.0}

    result = analyzer.interpret_results(
        _normalized(), [], metrics, None, None, None, valuation, provider="ollama",
    )

    assert set(result["scenario_returns"].keys()) == {"bear", "base", "bull"}
    assert result["scenario_returns"]["base"] == {"ret_lo_pct": -10.0, "ret_hi_pct": 10.0}
    assert isinstance(result["entry_plan"], list)
    assert len(result["entry_plan"]) > 0
    assert isinstance(result["stop_adding"], list)
    assert set(result["thesis_metric"].keys()) == {"name", "latest_value", "trend", "rationale"}


def test_interpret_results_injects_all_four_planning_fields_script_branch():
    valuation = _valuation_fixture()
    metrics = {"price": 100.0}

    result = analyzer.interpret_results(_normalized(), [], metrics, None, None, None, valuation, provider="script")

    assert result["scenario_returns"]["base"] == {"ret_lo_pct": -10.0, "ret_hi_pct": 10.0}
    assert isinstance(result["entry_plan"], list) and len(result["entry_plan"]) > 0
    assert isinstance(result["stop_adding"], list)
    assert set(result["thesis_metric"].keys()) == {"name", "latest_value", "trend", "rationale"}


def test_interpret_results_uses_ratios_and_red_flags_for_thesis_metric_and_stop_adding(monkeypatch):
    """thesis_metric's sector-driven ratio lookup and stop_adding's
    ACTIVE_RED_FLAG signal both depend on the `ratios`/`red_flags` arguments
    interpret_results() is given -- confirm they actually reach planning.py
    (and aren't silently dropped) via the script branch."""
    valuation = _valuation_fixture(sector_type="cyclical")
    ratios = [{"fy": 2023, "gross_margin": 0.45}]
    red_flags = [{"code": "X", "message": "bir bayrak", "detail": "..."}]

    result = analyzer.interpret_results(
        _normalized(), ratios, {"price": 100.0}, None, red_flags, None, valuation, provider="script",
    )

    assert result["thesis_metric"]["name"] == "Brüt Kâr Marjı"
    assert result["thesis_metric"]["latest_value"] == "%45.0"
    assert any(s["code"] == "ACTIVE_RED_FLAG" for s in result["stop_adding"])


def test_interpret_results_planning_fields_degrade_gracefully_without_price_or_valuation():
    """No metrics/technical price and no usable fair_value_range -- the four
    planning fields must still be present with their documented degraded
    shapes (all-None scenario_returns, empty entry_plan/stop_adding lists,
    thesis_metric with latest_value None), never missing and never raising."""
    valuation = {
        "sector_type": None,
        "fair_value_range": None,
        "sensitivity": None,
        "triangulation": {"signals": {}, "confidence": None, "direction": None},
        "assumptions": {},
        "notes": [],
    }

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="script")

    assert result["scenario_returns"] == {
        "bear": {"ret_lo_pct": None, "ret_hi_pct": None},
        "base": {"ret_lo_pct": None, "ret_hi_pct": None},
        "bull": {"ret_lo_pct": None, "ret_hi_pct": None},
    }
    assert result["entry_plan"] == []
    assert result["stop_adding"] == []
    assert result["thesis_metric"]["latest_value"] is None


def test_interpret_results_fills_catalyst_when_model_omits_it(monkeypatch):
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json(catalyst=""))
    catalyst = {"estimate_date": "2026-08-27", "label": "Q2 earnings ~27 Ağu", "based_on": "x"}
    valuation = _valuation_fixture()

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, catalyst, valuation, provider="ollama")

    assert result["catalyst"] == "Q2 earnings ~27 Ağu"


def test_interpret_results_ollama_error_is_caught_and_returned_as_error_dict(monkeypatch):
    def _raise(system, user, model, host):
        raise OllamaError("Ollama is not running.")

    monkeypatch.setattr(analyzer, "_call_ollama", _raise)

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, _valuation_fixture(), provider="ollama")

    assert result["error"] == "Ollama is not running."
    assert result["_provider"] == "ollama"
    assert "summary" in result


def test_interpret_results_unknown_provider_returns_error_without_raising():
    result = analyzer.interpret_results(
        _normalized(), [], {}, None, None, None, _valuation_fixture(), provider="not-a-real-provider"
    )

    assert "error" in result
    assert result["_provider"] == "not-a-real-provider"


def test_interpret_results_gemma_alias_stamps_canonical_ollama_provider(monkeypatch):
    monkeypatch.setattr(analyzer, "_call_ollama", lambda system, user, model, host: _commentary_json())

    result = analyzer.interpret_results(
        _normalized(), [], {}, None, None, None, _valuation_fixture(), provider="gemma"
    )

    assert result["_provider"] == "ollama"


# ---------------------------------------------------------------------------
# interpret_results() -- "script" provider, fully offline
# ---------------------------------------------------------------------------


def test_interpret_results_script_provider_is_fully_offline_and_postprocessed():
    valuation = _valuation_fixture(dcf_signal="pahali", confidence="ORTA")

    result = analyzer.interpret_results(_normalized(), [], {}, None, None, None, valuation, provider="script")

    assert "error" not in result
    assert result["_provider"] == "script"
    assert result["_model"] == "rule-based-v2"
    assert result["confidence"] == "ORTA"
    assert result["fair_value_range"] == valuation["fair_value_range"]
    assert result["fundamental_verdict"] == "PAHALI"
    assert result["valuation"] == valuation
    assert result["technical_verdict"] == "VERİ YOK (fiyat verisi alınamadı)"
    assert result["reverse_dcf_comment"]


# ---------------------------------------------------------------------------
# interpret() -- the two-phase orchestration wrapper
# ---------------------------------------------------------------------------


def test_interpret_with_precomputed_valuation_skips_phase_one(monkeypatch):
    """Passing `valuation=...` must go straight to phase 2 -- phase 1 must
    not even be attempted."""

    def _fail(*args, **kwargs):
        raise AssertionError("propose_assumptions() must not be called when valuation is already given")

    monkeypatch.setattr(analyzer, "propose_assumptions", _fail)
    valuation = _valuation_fixture()

    result = analyzer.interpret(_normalized(), [], provider="script", valuation=valuation)

    assert result["valuation"] == valuation
    assert result["_provider"] == "script"


def test_interpret_script_provider_end_to_end_offline():
    """No `valuation` given, no LLM mocked at all -- the "script" provider
    must run phase 1 (deterministic defaults) -> the real valuation engine
    -> phase 2 (deterministic commentary) with no network access whatsoever."""
    normalized, ratios, metrics = _financials_for_valuation()

    result = analyzer.interpret(normalized, ratios, provider="script", metrics=metrics)

    assert "error" not in result
    assert result["_provider"] == "script"
    assert result["valuation"]["sector_type"] == "mature"
    assert result["fair_value_range"] == result["valuation"]["fair_value_range"]
    assert result["confidence"] == result["valuation"]["triangulation"]["confidence"]
    assert result["fundamental_verdict"] in ("UCUZ", "MAKUL", "PAHALI")


def test_interpret_sic_from_submissions_overrides_llm_sector_guess(monkeypatch):
    """A known SIC code (6798 = REIT) must drive the deterministic engine's
    sector_type even if phase 1's own guess disagrees."""
    captured = {}

    def _fake_propose_assumptions(normalized, ratios, metrics, sector_hint=None, **kwargs):
        return {"assumptions": rule_based.default_assumptions(metrics, "mature"), "sector_type": "mature", "_provider": "script"}

    def _fake_run_valuation(normalized, ratios, metrics, price, price_df, assumptions, sector_type, **kwargs):
        captured["sector_type"] = sector_type
        return _valuation_fixture(sector_type=sector_type)

    monkeypatch.setattr(analyzer, "propose_assumptions", _fake_propose_assumptions)
    monkeypatch.setattr(analyzer, "run_valuation", _fake_run_valuation)

    analyzer.interpret(
        _normalized(), [], provider="script", metrics={"price": 50.0},
        submissions={"sic": 6798, "sicDescription": "Real Estate Investment Trusts"},
    )

    assert captured["sector_type"] == "reit"


def test_interpret_falls_back_to_phase1_sector_guess_when_sic_missing(monkeypatch):
    captured = {}

    def _fake_propose_assumptions(normalized, ratios, metrics, sector_hint=None, **kwargs):
        assert sector_hint is None
        return {"assumptions": rule_based.default_assumptions(metrics, "cyclical"), "sector_type": "cyclical", "_provider": "script"}

    def _fake_run_valuation(normalized, ratios, metrics, price, price_df, assumptions, sector_type, **kwargs):
        captured["sector_type"] = sector_type
        return _valuation_fixture(sector_type=sector_type)

    monkeypatch.setattr(analyzer, "propose_assumptions", _fake_propose_assumptions)
    monkeypatch.setattr(analyzer, "run_valuation", _fake_run_valuation)

    analyzer.interpret(_normalized(), [], provider="script", metrics={"price": 50.0}, submissions=None)

    assert captured["sector_type"] == "cyclical"


def test_interpret_threads_hyper_growth_extras_from_phase1_into_run_valuation(monkeypatch):
    """interpret() must pass phase 1's `hyper_growth_extras` (HYPER_SPEC.md
    Sec.5) through to `run_valuation`, keyword-only, so an LLM-refined
    TAM/margin can actually reach the engine's `_build_hyper_growth`."""
    captured = {}
    extras = _hyper_growth_extras_payload()

    def _fake_propose_assumptions(normalized, ratios, metrics, sector_hint=None, **kwargs):
        return {
            "assumptions": rule_based.default_assumptions(metrics, "mature"),
            "sector_type": "mature",
            "hyper_growth_extras": extras,
            "_provider": "script",
        }

    def _fake_run_valuation(normalized, ratios, metrics, price, price_df, assumptions, sector_type, **kwargs):
        captured["hyper_growth_extras"] = kwargs.get("hyper_growth_extras")
        return _valuation_fixture(sector_type=sector_type)

    monkeypatch.setattr(analyzer, "propose_assumptions", _fake_propose_assumptions)
    monkeypatch.setattr(analyzer, "run_valuation", _fake_run_valuation)

    analyzer.interpret(_normalized(), [], provider="script", metrics={"price": 50.0})

    assert captured["hyper_growth_extras"] == extras


def test_interpret_hyper_growth_extras_defaults_to_none_when_phase1_has_none(monkeypatch):
    captured = {}

    def _fake_run_valuation(normalized, ratios, metrics, price, price_df, assumptions, sector_type, **kwargs):
        captured["hyper_growth_extras"] = kwargs.get("hyper_growth_extras")
        return _valuation_fixture(sector_type=sector_type)

    monkeypatch.setattr(analyzer, "run_valuation", _fake_run_valuation)

    analyzer.interpret(_normalized(), [], provider="script", metrics={"price": 50.0})

    assert captured["hyper_growth_extras"] is None


def test_interpret_unknown_provider_returns_error_without_raising():
    result = analyzer.interpret(_normalized(), [], provider="not-a-real-provider")

    assert "error" in result
    assert result["_provider"] == "not-a-real-provider"


def test_interpret_dispatches_to_anthropic(monkeypatch):
    monkeypatch.setattr(
        analyzer, "_call_anthropic", lambda system, user, model, api_key: _commentary_json(summary="hosted")
    )
    # Avoid depending on a real ANTHROPIC_API_KEY being configured.
    monkeypatch.setattr(analyzer.Config, "require_anthropic_key", classmethod(lambda cls: "fake-key"))

    result = analyzer.interpret(_normalized(), [], provider="anthropic", valuation=_valuation_fixture())

    assert result["summary"] == "hosted"
    assert result["_provider"] == "anthropic"
    assert result["_model"] == analyzer.Config.ANTHROPIC_MODEL


def test_interpret_backward_compatible_defaults_to_configured_provider(monkeypatch):
    """Calling interpret(normalized, ratios) with no provider must still
    work, using Config.ANALYZER_PROVIDER."""
    monkeypatch.setattr(analyzer.Config, "ANALYZER_PROVIDER", "script")

    result = analyzer.interpret(_normalized(), [], valuation=_valuation_fixture())

    assert result["_provider"] == "script"
    # New keyword-only params (horizon, metrics, technical, red_flags,
    # catalyst, valuation, submissions) are all optional; omitting the ones
    # not exercised above still yields a fully post-processed result with
    # the "1y" default horizon.
    assert result["_horizon"] == "1y"
    assert result["_weights"] == {"fundamental": 0.5, "technical": 0.5}
    assert result["technical_verdict"] == "VERİ YOK (fiyat verisi alınamadı)"
    assert result["catalyst"] == "bilinmiyor"


def test_interpret_stamps_horizon_and_weights(monkeypatch):
    result = analyzer.interpret(_normalized(), [], provider="script", horizon="5y", valuation=_valuation_fixture())

    assert result["_horizon"] == "5y"
    assert result["_weights"] == {"fundamental": 0.8, "technical": 0.2}


def test_interpret_never_raises_on_unexpected_internal_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(analyzer, "propose_assumptions", _boom)

    result = analyzer.interpret(_normalized(), [], provider="script")

    assert result["error"] == "boom"
    assert result["_provider"] == "script"
