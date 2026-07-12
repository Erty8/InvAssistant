"""Minimal client for a local Ollama server's chat API.

This module is intentionally small and focused: it knows how to turn a
(system prompt, user message) pair into a single raw text response from a
local Ollama model, and how to translate the various ways that call can
fail (server not running, model not pulled, timeout, bad HTTP status) into
one friendly exception type. Prompt construction and response-JSON parsing
are the responsibility of :mod:`sec_analyzer.interpret.analyzer`.
"""

import logging

import requests

logger = logging.getLogger(__name__)

#: Ollama's chat endpoint is always this path relative to the server host.
_CHAT_PATH = "/api/chat"


class OllamaError(Exception):
    """Raised when a local Ollama chat request cannot be completed.

    The message is written to be shown directly to an end user (it includes
    remediation hints such as ``ollama serve`` / ``ollama pull <model>``).
    """


def chat_json(system: str, user: str, model: str, host: str, timeout: int = 300) -> str:
    """Send a chat request to a local Ollama server and return the raw reply text.

    Uses Ollama's ``format: "json"`` option so the model is constrained to
    emit valid JSON; the caller is still responsible for parsing that text
    (see ``analyzer._parse_model_json``).

    Args:
        system: System prompt (methodology + output-format instructions).
        user: User message (the compact JSON payload of financial data).
        model: Name of a model already pulled in Ollama, e.g. ``"gemma4:latest"``.
        host: Base URL of the Ollama server, e.g. ``"http://localhost:11434"``.
        timeout: Request timeout in seconds. Local generation can be slow on
            CPU-only machines, so this defaults to a generous 300s.

    Returns:
        The model's raw response text (``response["message"]["content"]``).

    Raises:
        OllamaError: If the server is unreachable, the request times out,
            the model isn't available locally, or the server returns a
            malformed response.
    """
    url = f"{host.rstrip('/')}{_CHAT_PATH}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }

    try:
        resp = requests.post(url, json=body, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {host}. Is Ollama running? "
            "Start it with 'ollama serve' (or launch the Ollama app), then "
            f"make sure the model is pulled with 'ollama pull {model}'."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            f"Ollama at {host} did not respond within {timeout}s. The model "
            "may still be loading, or the machine may be too slow to run it "
            "-- try again, increase the timeout, or use a smaller model."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Request to Ollama at {host} failed: {exc}") from exc

    if resp.status_code == 404:
        raise OllamaError(
            f"Ollama model '{model}' was not found on {host}. Pull it first "
            f"with 'ollama pull {model}'."
        )
    if resp.status_code != 200:
        snippet = resp.text[:200] if resp.text else ""
        raise OllamaError(
            f"Ollama at {host} returned HTTP {resp.status_code}: {snippet}"
        )

    try:
        data = resp.json()
        content = data["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OllamaError(
            f"Ollama at {host} returned an unexpected response shape: {exc}"
        ) from exc

    return content
