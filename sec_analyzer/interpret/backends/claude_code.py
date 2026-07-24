"""Claude Code (`claude -p`) subprocess backend for the interpret layer.

This backend drives the locally-installed Claude Code CLI as a subprocess
instead of calling the hosted HTTP API, so analysis is billed to the user's
Claude *subscription* rather than to a metered API account. It exposes the
same contract as the API/Ollama transports in
:mod:`sec_analyzer.interpret.analyzer`: given a ``(system, user)`` prompt pair
it returns the model's RAW reply text (the schema JSON the analyzer will then
fence-strip and parse).

Billing safety is the whole point of this file, so it is enforced first and
unconditionally: if ``ANTHROPIC_API_KEY`` is present in the environment,
``claude -p`` silently routes to API billing, so the backend REFUSES to run
(raises) rather than quietly charging the wrong account.

The CLI's ``--output-format json`` wraps the model's answer in a Claude Code
result envelope, so parsing here is two-layer (JSON-in-JSON): this module peels
off the CC envelope and returns the inner model text; the analyzer's existing
``_parse_model_json`` then parses that text as our own schema.

Every failure mode (API-key guard, `claude` not installed, non-zero exit,
timeout, unparseable envelope) raises :class:`ClaudeCodeError`. The caller
catches it and degrades to the deterministic rule-based path -- the valuation
engine never stops.
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Optional

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Env-var whose mere presence redirects `claude -p` from subscription to API
#: billing. The guard below treats any non-empty value as "set".
_API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Exact operator-facing message for the billing guard (Turkish, per spec).
_API_KEY_GUARD_MESSAGE = (
    "ANTHROPIC_API_KEY ortamda set — claude -p bu durumda abonelik yerine API "
    "hesabına fatura keser. Backend iptal edildi. Key'i unset edin veya "
    "llm_backend=api seçin."
)


class ClaudeCodeError(Exception):
    """Raised when the Claude Code subprocess backend cannot complete.

    The message is written to be shown to an end user / logged; the caller is
    expected to catch it and fall back to the deterministic rule-based path.
    """


def _assert_no_api_key() -> None:
    """Billing guard (MANDATORY, first): refuse to run if an API key is set.

    ``claude -p`` uses whatever credentials it finds; an ``ANTHROPIC_API_KEY``
    in the environment silently makes it bill the API account instead of the
    subscription. Since preventing that silent misbilling is the entire reason
    this backend exists, this check is unconditional and must never be
    removed or bypassed.
    """
    if os.environ.get(_API_KEY_ENV):
        logger.error(_API_KEY_GUARD_MESSAGE)
        raise ClaudeCodeError(_API_KEY_GUARD_MESSAGE)


def _extract_model_text(envelope_text: str) -> str:
    """Peel the Claude Code ``--output-format json`` envelope to the model text.

    Claude Code returns a JSON result object whose final assistant answer lives
    in a ``result`` field (older/other shapes use ``text``/``content``). This
    returns that inner string verbatim -- it does NOT parse it as our analysis
    schema (the analyzer's ``_parse_model_json`` does that next). Raises
    :class:`ClaudeCodeError` if the envelope isn't the expected JSON shape or
    carries no usable text, and surfaces a Claude Code ``is_error`` envelope as
    an error rather than treating its payload as a model answer.

    Args:
        envelope_text: Raw stdout from ``claude -p --output-format json``.

    Returns:
        The inner model reply text (still possibly a fenced JSON string).
    """
    if not envelope_text or not envelope_text.strip():
        raise ClaudeCodeError("Claude Code boş çıktı döndürdü (stdout yok).")
    try:
        envelope = json.loads(envelope_text)
    except (ValueError, TypeError) as exc:
        raise ClaudeCodeError(
            f"Claude Code çıktısı JSON zarfı olarak ayrıştırılamadı: {exc}"
        ) from exc

    if not isinstance(envelope, dict):
        raise ClaudeCodeError(
            "Claude Code JSON zarfı beklenen nesne biçiminde değil."
        )

    # A CC envelope can itself signal failure (e.g. subtype 'error_*').
    if envelope.get("is_error") or envelope.get("subtype") in ("error_max_turns", "error_during_execution"):
        raise ClaudeCodeError(
            f"Claude Code hata zarfı döndürdü: "
            f"{envelope.get('subtype') or envelope.get('result') or 'bilinmeyen hata'}"
        )

    for key in ("result", "text", "content"):
        value = envelope.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise ClaudeCodeError(
        "Claude Code zarfında model metni bulunamadı (result/text/content yok)."
    )


def call_claude_code(
    system: str,
    user: str,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """Run one interpretation via the local `claude -p` subprocess.

    Same return contract as ``analyzer._call_ollama``: the
    model's RAW reply text (the analysis-schema JSON, possibly fenced), which
    the caller feeds through ``analyzer._parse_model_json``.

    Order of operations (each step raises :class:`ClaudeCodeError` on failure):

    1. **Billing guard** (:func:`_assert_no_api_key`) -- refuse if an API key
       is set. Runs before anything else.
    2. Verify ``claude`` is on PATH (``shutil.which``).
    3. Invoke ``claude -p --output-format json`` (plus ``--model`` when given),
       passing the combined system+user prompt on **stdin** (avoids the OS
       argument-length limit on our long prompts).
    4. Unwrap the Claude Code JSON envelope (:func:`_extract_model_text`).

    Args:
        system: System prompt (methodology + output-format instructions).
        user: User message (the compact JSON payload for the phase).
        model: Optional model id/name passed to ``claude --model``. ``None``
            uses whatever the CLI is configured to use.
        timeout: Per-call timeout in seconds; defaults to
            ``Config.CLAUDE_CODE_TIMEOUT``.

    Returns:
        The model's raw reply text.

    Raises:
        ClaudeCodeError: On the API-key guard, a missing ``claude`` binary, a
            non-zero exit, a timeout, or an unparseable/empty envelope.
    """
    _assert_no_api_key()

    binary = Config.CLAUDE_CODE_BIN
    resolved_bin = shutil.which(binary)
    if resolved_bin is None:
        raise ClaudeCodeError(
            f"Claude Code CLI ('{binary}') PATH'te bulunamadı. Kurulum: "
            "'npm install -g @anthropic-ai/claude-code', sonra 'claude login'. "
            "(Masaüstü uygulaması yeterli değildir; ayrı CLI gerekir. Kuruluysa "
            "yolu CLAUDE_CODE_BIN ile verebilirsiniz.)"
        )

    # `claude -p` takes a single prompt; there is no separate system channel in
    # headless mode, so the system instruction is embedded ahead of the user
    # payload (per spec). The whole thing goes over stdin.
    prompt = f"{system}\n\n{user}"
    cmd = [resolved_bin, "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]

    resolved_timeout = timeout if timeout is not None else Config.CLAUDE_CODE_TIMEOUT
    logger.info("Requesting Claude Code analysis via %s (model=%s)", resolved_bin, model or "default")

    try:
        completed = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=resolved_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCodeError(
            f"Claude Code {resolved_timeout}s içinde yanıt vermedi (timeout)."
        ) from exc
    except OSError as exc:  # e.g. the binary vanished/became non-executable between which() and run()
        raise ClaudeCodeError(f"Claude Code çalıştırılamadı: {exc}") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        logger.error("Claude Code exited %s: %s", completed.returncode, stderr[:500])
        raise ClaudeCodeError(
            f"Claude Code hata koduyla çıktı ({completed.returncode}): "
            f"{stderr[:200] or 'stderr yok'}"
        )

    if completed.stderr and completed.stderr.strip():
        # Non-fatal: `claude` may emit diagnostics on stderr even on success.
        logger.debug("Claude Code stderr: %s", completed.stderr.strip()[:500])

    return _extract_model_text(completed.stdout)
