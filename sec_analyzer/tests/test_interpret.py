"""Unit tests for sec_analyzer.interpret.analyzer.

These tests never touch the network: the raw "send this, get text back"
calls (`_call_ollama` / `_call_anthropic`) are monkeypatched out, so the
suite exercises fence-stripping, JSON parsing, and provider dispatch only.
"""

import pytest

from sec_analyzer.interpret import analyzer
from sec_analyzer.interpret.ollama_client import OllamaError


def _normalized():
    """Minimal normalized-facts dict; content is irrelevant to these tests."""
    return {
        "entity_name": "X",
        "currency": "USD",
        "annual": {},
        "quarterly": {},
        "missing": [],
        "matched_tags": {},
    }


# ---------------------------------------------------------------------------
# _strip_json_fence
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


# ---------------------------------------------------------------------------
# _parse_model_json
# ---------------------------------------------------------------------------


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
# interpret() provider dispatch
# ---------------------------------------------------------------------------


def test_interpret_dispatches_to_ollama(monkeypatch):
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: '```json\n{"summary": "local"}\n```'
    )

    result = analyzer.interpret(_normalized(), [], provider="ollama")

    assert result["summary"] == "local"
    assert result["_provider"] == "ollama"
    assert result["_model"] == analyzer.Config.OLLAMA_MODEL


def test_interpret_dispatches_to_anthropic(monkeypatch):
    monkeypatch.setattr(
        analyzer, "_call_anthropic", lambda system, user, model, api_key: '{"summary": "hosted"}'
    )
    # Avoid depending on a real ANTHROPIC_API_KEY being configured.
    monkeypatch.setattr(analyzer.Config, "require_anthropic_key", classmethod(lambda cls: "fake-key"))

    result = analyzer.interpret(_normalized(), [], provider="anthropic")

    assert result["summary"] == "hosted"
    assert result["_provider"] == "anthropic"
    assert result["_model"] == analyzer.Config.ANTHROPIC_MODEL


def test_interpret_gemma_alias_dispatches_to_ollama(monkeypatch):
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: '{"summary": "local"}'
    )

    result = analyzer.interpret(_normalized(), [], provider="gemma")

    assert result["_provider"] == "ollama"


def test_interpret_unknown_provider_returns_error_without_raising():
    result = analyzer.interpret(_normalized(), [], provider="not-a-real-provider")

    assert "error" in result
    assert result["_provider"] == "not-a-real-provider"


def test_interpret_ollama_error_is_caught_and_returned_as_error_dict(monkeypatch):
    def _raise(system, user, model, host):
        raise OllamaError("Ollama is not running.")

    monkeypatch.setattr(analyzer, "_call_ollama", _raise)

    result = analyzer.interpret(_normalized(), [], provider="ollama")

    assert result["error"] == "Ollama is not running."
    assert result["_provider"] == "ollama"
    assert "summary" in result


def test_interpret_backward_compatible_defaults_to_configured_provider(monkeypatch):
    """Calling interpret(normalized, ratios) with no provider must still work."""
    monkeypatch.setattr(analyzer.Config, "ANALYZER_PROVIDER", "ollama")
    monkeypatch.setattr(
        analyzer, "_call_ollama", lambda system, user, model, host: '{"summary": "default path"}'
    )

    result = analyzer.interpret(_normalized(), [])

    assert result["summary"] == "default path"
    assert result["_provider"] == "ollama"
