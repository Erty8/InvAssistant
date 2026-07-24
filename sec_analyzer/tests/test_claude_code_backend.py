"""Tests for the Claude Code (`claude -p`) subprocess backend.

Priority order mirrors the patch spec: the ANTHROPIC_API_KEY billing guard is
the single most important guarantee (it prevents silently billing the API
account instead of the subscription), then graceful failure when `claude`
isn't installed, then the JSON-in-JSON envelope parsing, then the fallback
chain that degrades a failed AI backend to the deterministic rule-based path.
All tests are offline: the subprocess and PATH lookup are stubbed.
"""

import json
import subprocess

import pytest

from sec_analyzer.interpret import analyzer
from sec_analyzer.interpret.backends import claude_code as cc


# --------------------------------------------------------------------------
# Billing guard (MOST IMPORTANT): refuse to run if ANTHROPIC_API_KEY is set.
# --------------------------------------------------------------------------

def test_api_key_guard_raises_before_any_subprocess(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-run")

    # If the guard is bypassed and we reach subprocess/which, fail loudly.
    def _boom(*a, **k):
        raise AssertionError("subprocess/which must not be reached when API key is set")

    monkeypatch.setattr(cc.subprocess, "run", _boom)
    monkeypatch.setattr(cc.shutil, "which", _boom)

    with pytest.raises(cc.ClaudeCodeError) as excinfo:
        cc.call_claude_code("sys", "user")
    # Message names the misbilling risk and the remedy.
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)
    assert "abonelik" in str(excinfo.value)


def test_api_key_guard_helper_directly(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    with pytest.raises(cc.ClaudeCodeError):
        cc._assert_no_api_key()


def test_no_guard_trip_when_key_absent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Should not raise on the guard step itself.
    cc._assert_no_api_key()


# --------------------------------------------------------------------------
# Graceful failure when `claude` is not installed.
# --------------------------------------------------------------------------

def test_missing_claude_binary_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cc.shutil, "which", lambda _bin: None)
    with pytest.raises(cc.ClaudeCodeError) as excinfo:
        cc.call_claude_code("sys", "user")
    assert "PATH" in str(excinfo.value)


# --------------------------------------------------------------------------
# Envelope parsing (JSON-in-JSON): peel the CC result envelope.
# --------------------------------------------------------------------------

def test_extract_model_text_from_result_field():
    inner = '{"fundamental_verdict": "MAKUL"}'
    envelope = json.dumps({"type": "result", "subtype": "success", "result": inner})
    assert cc._extract_model_text(envelope) == inner


def test_extract_model_text_prefers_result_then_text():
    envelope = json.dumps({"text": "from-text"})
    assert cc._extract_model_text(envelope) == "from-text"


def test_extract_model_text_raises_on_error_envelope():
    envelope = json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True})
    with pytest.raises(cc.ClaudeCodeError):
        cc._extract_model_text(envelope)


def test_extract_model_text_raises_on_non_json():
    with pytest.raises(cc.ClaudeCodeError):
        cc._extract_model_text("not json at all")


def test_extract_model_text_raises_on_empty():
    with pytest.raises(cc.ClaudeCodeError):
        cc._extract_model_text("   ")


def test_extract_model_text_raises_when_no_text_field():
    with pytest.raises(cc.ClaudeCodeError):
        cc._extract_model_text(json.dumps({"type": "result", "usage": {}}))


# --------------------------------------------------------------------------
# Subprocess invocation success/failure paths (run() stubbed).
# --------------------------------------------------------------------------

def _fake_completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_call_claude_code_success_returns_inner_text(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cc.shutil, "which", lambda _bin: "/usr/bin/claude")
    inner = '{"fundamental_verdict": "UCUZ"}'
    envelope = json.dumps({"type": "result", "result": inner})

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return _fake_completed(stdout=envelope)

    monkeypatch.setattr(cc.subprocess, "run", _fake_run)

    out = cc.call_claude_code("SYSTEM", "USER", model="claude-x")
    assert out == inner
    # Prompt is passed via stdin (not argv), and system is embedded ahead of user.
    assert captured["input"] == "SYSTEM\n\nUSER"
    assert "--output-format" in captured["cmd"] and "json" in captured["cmd"]
    assert "--model" in captured["cmd"]


def test_call_claude_code_nonzero_exit_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cc.shutil, "which", lambda _bin: "/usr/bin/claude")
    monkeypatch.setattr(cc.subprocess, "run", lambda cmd, **k: _fake_completed(stderr="bad", returncode=2))
    with pytest.raises(cc.ClaudeCodeError):
        cc.call_claude_code("s", "u")


def test_call_claude_code_timeout_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cc.shutil, "which", lambda _bin: "/usr/bin/claude")

    def _timeout(cmd, **k):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=k.get("timeout", 1))

    monkeypatch.setattr(cc.subprocess, "run", _timeout)
    with pytest.raises(cc.ClaudeCodeError) as excinfo:
        cc.call_claude_code("s", "u")
    assert "timeout" in str(excinfo.value).lower()


# --------------------------------------------------------------------------
# Dispatcher + fallback chain: claude_code failure -> rule-based + note.
# --------------------------------------------------------------------------

def test_dispatch_routes_claude_code(monkeypatch):
    monkeypatch.setattr(analyzer, "call_claude_code", lambda system, user, model=None: '{"ok": true}')
    raw, model = analyzer._dispatch_llm_call("claude_code", "sys", "usr", "m", None, None)
    assert raw == '{"ok": true}'
    assert model == "m"


def _capture_cc_model(monkeypatch):
    seen = {}

    def _fake_cc(system, user, model=None):
        seen["model"] = model
        return '{"ok": true}'

    monkeypatch.setattr(analyzer, "call_claude_code", _fake_cc)
    return seen


def test_claude_code_phase1_uses_strong_model_by_default(monkeypatch):
    seen = _capture_cc_model(monkeypatch)
    _, resolved = analyzer._dispatch_llm_call(
        "claude_code", "s", "u", None, None, None, phase="assumptions"
    )
    assert seen["model"] == analyzer.Config.CLAUDE_CODE_MODEL_ASSUMPTIONS == "opus"
    assert resolved == "opus"


def test_claude_code_phase2_uses_cheaper_model_by_default(monkeypatch):
    seen = _capture_cc_model(monkeypatch)
    _, resolved = analyzer._dispatch_llm_call(
        "claude_code", "s", "u", None, None, None, phase="commentary"
    )
    assert seen["model"] == analyzer.Config.CLAUDE_CODE_MODEL_COMMENTARY == "sonnet"
    assert resolved == "sonnet"


def test_claude_code_default_phase_is_commentary(monkeypatch):
    seen = _capture_cc_model(monkeypatch)
    analyzer._dispatch_llm_call("claude_code", "s", "u", None, None, None)
    assert seen["model"] == "sonnet"


def test_explicit_model_overrides_per_phase_default(monkeypatch):
    seen = _capture_cc_model(monkeypatch)
    _, resolved = analyzer._dispatch_llm_call(
        "claude_code", "s", "u", "claude-opus-4-8", None, None, phase="commentary"
    )
    assert seen["model"] == "claude-opus-4-8"  # explicit wins over the sonnet default
    assert resolved == "claude-opus-4-8"


def test_fallback_helper_degrades_claude_code_to_script(monkeypatch):
    calls = []

    def _fake_ir(*a, **k):
        calls.append(k.get("provider"))
        if k.get("provider") == "claude_code":
            return {"error": "claude bulunamadı", "summary": "x", "_provider": "claude_code"}
        return {"fundamental_verdict": "MAKUL", "summary": "ok", "_provider": "script"}

    monkeypatch.setattr(analyzer, "interpret_results", _fake_ir)
    result, note = analyzer._interpret_results_with_fallback(
        "claude_code", {}, [], {}, None, None, None, {},
        "claude_code", None, None, None, "1y", None,
    )
    assert calls == ["claude_code", "script"]
    assert "error" not in result
    assert note and "claude_code" in note and "claude bulunamadı" in note


def test_fallback_helper_passes_through_non_claude_code(monkeypatch):
    monkeypatch.setattr(analyzer, "interpret_results", lambda *a, **k: {"error": "boom", "_provider": "ollama"})
    result, note = analyzer._interpret_results_with_fallback(
        "ollama", {}, [], {}, None, None, None, {},
        "ollama", None, None, None, "1y", None,
    )
    assert note is None
    assert result["error"] == "boom"  # ollama behavior unchanged


def test_llm_report_live_compares_llm_vs_baseline(monkeypatch):
    metrics = {"revenue_cagr_5y": 0.12, "revenue_cagr_3y": 0.10, "latest_fundamental_fy": 2024}
    phase1 = {
        "_provider": "claude_code",
        "assumptions": {
            "bear": {"growth_5y": 0.08, "terminal_growth": 0.03, "discount_rate": 0.12, "story": "temkinli"},
            "base": {"growth_5y": 0.12, "terminal_growth": 0.05, "discount_rate": 0.11, "story": "iyimser"},
            "bull": {"growth_5y": 0.16, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "agresif"},
        },
    }
    r = analyzer._build_llm_report(phase1, "claude_code", "mature", metrics, None, None, None)
    assert r["used_llm"] is True
    assert r["provider"] == "claude_code"
    assert r["model"] == "opus"
    base = r["scenarios"]["base"]["fields"]
    # terminal 0.05 is out of range -> clamped to 0.04 (flagged), with a note.
    assert base["terminal_growth"]["llm"] == 0.05
    assert base["terminal_growth"]["used"] == pytest.approx(0.04)
    assert base["terminal_growth"]["clamped"] is True
    # baseline present and delta-vs-baseline computed.
    assert base["discount_rate"]["baseline"] is not None
    assert base["discount_rate"]["delta_vs_baseline_pp"] is not None
    assert r["scenarios"]["base"]["story"] == "iyimser"
    assert r["clamp_notes"]


def test_llm_report_fallback_when_requested_llm_but_downgraded(monkeypatch):
    r = analyzer._build_llm_report(
        {"_provider": "script", "assumptions": {}}, "claude_code", "mature", {}, None, None, "claude bulunamadı"
    )
    assert r["used_llm"] is False
    assert "claude bulunamadı" in r["fallback"]


def test_llm_report_none_for_pure_script_run():
    assert analyzer._build_llm_report(
        {"_provider": "script", "assumptions": {}}, "script", "mature", {}, None, None, None
    ) is None


def test_interpret_end_to_end_adds_backend_note_on_claude_code_failure(monkeypatch):
    """interpret() with a cached phase-1 override + claude_code phase-2 that
    fails: numbers come from the override, commentary degrades to rule-based,
    and the result carries the Turkish backend_note."""
    def _fake_ir(*a, **k):
        if k.get("provider") == "claude_code":
            return {"error": "claude -p PATH'te bulunamadı", "_provider": "claude_code"}
        return {"fundamental_verdict": "MAKUL", "fair_value_range": {}, "summary": "ok", "_provider": "script"}

    monkeypatch.setattr(analyzer, "interpret_results", _fake_ir)

    normalized = {"entity_name": "X", "currency": "USD", "annual": {}, "quarterly": {}, "missing": []}
    metrics = {"price": 50.0, "shares": 100.0, "latest_fundamental_fy": 2023, "latest_fy": 2023}
    override = {
        "assumptions": {
            "bear": {"growth_5y": 0.06, "terminal_growth": 0.03, "discount_rate": 0.11, "story": "x"},
            "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.11, "story": "x"},
            "bull": {"growth_5y": 0.14, "terminal_growth": 0.03, "discount_rate": 0.11, "story": "x"},
        },
        "sector_type": "mature", "hyper_growth_extras": None,
        "_provider": "cached:ollama", "_assumption_set_id": 3,
    }
    result = analyzer.interpret(
        normalized, [], provider="claude_code", metrics=metrics, phase1_override=override
    )
    assert "error" not in result
    assert result.get("backend_note") and "claude_code" in result["backend_note"]
    # Provenance from the cached set is still stamped.
    assert result["_assumption_set_id"] == 3
