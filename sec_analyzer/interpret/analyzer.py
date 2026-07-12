"""LLM-powered fundamental interpretation of normalized SEC financials.

This module is the only place in ``sec_analyzer`` that talks to a language
model for analysis. It takes the tidy structures produced by the
``normalize`` layer (:func:`sec_analyzer.normalize.normalizer.normalize_facts`
and :func:`sec_analyzer.normalize.ratios.compute_ratios`) and asks an LLM to
produce a structured, conservative fundamental read: a fair-value range, a
verdict on fundamental quality, cyclicality commentary, and a short summary.

Three backends are supported, selected via ``Config.ANALYZER_PROVIDER`` (or
the ``provider`` argument to :func:`interpret`):

* ``"ollama"`` (**default**) -- a local Gemma model served by Ollama. Free,
  private, and requires no API key, but does require Ollama to be running
  locally with the model already pulled (see
  :mod:`sec_analyzer.interpret.ollama_client`).
* ``"anthropic"`` -- the hosted Claude API. Requires ``ANTHROPIC_API_KEY``.
* ``"script"`` -- a deterministic, script-based (no-AI) fundamental screen
  computed entirely with plain arithmetic; no network access, no API key,
  and no LLM involved at all (see
  :mod:`sec_analyzer.interpret.rule_based`).

Design goals:

* **Never crash the CLI.** Every failure mode -- a missing API key, Ollama
  not running, a network error, an API error, or a response that isn't valid
  JSON -- is caught and turned into a small error dict with a human-readable
  ``summary`` instead of propagating an exception.
* **Import-safe without optional dependencies.** The ``anthropic`` package is
  an optional extra; if it isn't installed, importing this module still
  succeeds, and :func:`interpret` returns a friendly error dict instructing
  the user to install it (only when the Anthropic backend is actually
  selected).
* **Configurable methodology.** The system prompt is built from an optional
  user-supplied ``METODOLOJI.md`` file (see :func:`load_methodology`) so
  different valuation philosophies can be swapped in without touching code.
* **Shared prompt/parsing, swappable transport.** Building the system/user
  prompts and parsing the model's JSON reply is identical regardless of
  backend; only the raw "send this, get text back" call
  (:func:`_call_anthropic` / :func:`_call_ollama`) differs per provider.
"""

import json
import logging
import os
import re
from typing import List, Optional

from sec_analyzer.config import Config, ConfigError
from sec_analyzer.interpret.ollama_client import OllamaError, chat_json
from sec_analyzer.normalize.normalizer import to_annual_series

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only when the optional
    # dependency is genuinely absent.
    anthropic = None

logger = logging.getLogger(__name__)

#: Maximum tokens requested from Claude for a single interpretation call.
_MAX_TOKENS = 2000

#: Default valuation/analysis framework, used whenever no ``METODOLOJI.md``
#: is supplied (see Config.METODOLOJI_PATH and load_methodology()).
DEFAULT_METHODOLOGY = """\
You are a careful, conservative equity fundamentals analyst. You are given a
company's normalized SEC financial figures (raw USD, not scaled) covering
several recent fiscal years, plus a set of derived ratios. Use ONLY the
figures provided -- do not invent numbers, do not assume information you
were not given, and do not rely on outside knowledge of the company's stock
price, market capitalization, or recent news.

Assess the business along these dimensions:

1. Revenue and earnings trend: is revenue growing, flat, or declining across
   the available years? Is net income tracking revenue, or diverging from it
   (margin expansion/compression)?
2. Profitability and margin quality: examine net margin over time. Favor
   businesses with stable or improving margins over those with volatile or
   eroding margins.
3. Return on equity (ROE) quality: a high ROE built on strong net income is
   healthier than one inflated by a thin or negative equity base -- note this
   distinction when equity is unusually small.
4. Balance-sheet strength: use the current ratio as a liquidity signal, and
   the relationship between total liabilities and total assets/equity (when
   available) as a leverage signal. Flag balance sheets that look stretched.
5. Cash-flow durability: compare operating cash flow to net income across
   years. Earnings that are not backed by cash flow over time are a
   yellow flag.
6. Growth trajectory and cyclicality: use the year-over-year revenue and net
   income growth figures to judge whether growth is steady, accelerating,
   decelerating, or cyclical/volatile. Note if the available history is too
   short or too choppy to draw a confident trend.

For a fair-value range: reason like a conservative analyst using simple,
transparent methods grounded strictly in the provided figures -- e.g. a
range of plausible earnings or cash-flow multiples applied to trailing
figures, or a simplified discounted-cash-flow sketch using the observed
growth rate and a conservative discount rate. Always state your basis and
assumptions plainly. If the data provided is too sparse, too short a
history, or too inconsistent to support a fair-value estimate, say so
explicitly and return null values rather than fabricating a number.

Be explicit about uncertainty throughout: call out missing concepts, thin
history, inconsistent trends, or anything else that limits confidence in
the analysis. This analysis is educational commentary on public financial
filings, not personalized investment advice."""

#: Explicit output-format contract appended to whatever methodology text is
#: used as the system prompt. Kept separate from the methodology itself so
#: the JSON contract is always enforced regardless of which methodology
#: (default or user-supplied) is active.
_OUTPUT_CONTRACT = """\
Respond with ONLY a single JSON object -- no prose before or after it, and
no markdown code fences. The JSON object must match exactly this schema:

{
  "fair_value_range": {
    "low": <number or null>,
    "high": <number or null>,
    "unit": <string, e.g. "USD per share" or "USD market cap">,
    "basis": <string explaining the method and key assumptions used>
  },
  "fundamental_verdict": <string, e.g. "strong", "adequate", or "weak",
    followed by a short rationale>,
  "cyclical_risk": <string describing growth stability / cyclicality>,
  "key_ratios": {<ratio name>: <value>, ...},
  "summary": <string, a short plain-language summary of the analysis>
}

This is an educational, non-personalized analysis of public SEC filings, not
investment advice. If the provided data is insufficient (e.g. many missing
concepts, very few fiscal years, or inconsistent figures) say so plainly in
"summary" and set "fair_value_range.low"/"high" to null rather than
guessing."""

#: Matches a fenced code block wrapping the whole response, with or without
#: a "json" language tag, e.g. ```json\n{...}\n``` or ```\n{...}\n```.
_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE
)


def load_methodology() -> str:
    """Return the analysis methodology to use as the system prompt.

    If ``Config.METODOLOJI_PATH`` points at an existing, non-empty file, its
    contents are used verbatim so users can supply their own valuation
    philosophy without touching code. Otherwise -- or if reading the file
    fails for any reason -- :data:`DEFAULT_METHODOLOGY` is used.

    Returns:
        The methodology text (never empty).
    """
    path = Config.METODOLOJI_PATH
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                logger.info("Using custom methodology from %s", path)
                return content
            logger.info(
                "Methodology file %s is empty; using the default methodology.",
                path,
            )
        else:
            logger.info(
                "No methodology file found at %s; using the default methodology.",
                path,
            )
    except OSError as exc:
        logger.warning(
            "Failed to read methodology file %s (%s); using the default "
            "methodology.",
            path, exc,
        )
    return DEFAULT_METHODOLOGY


def _build_system_prompt() -> str:
    """Combine the active methodology with the fixed JSON output contract."""
    return f"{load_methodology()}\n\n{_OUTPUT_CONTRACT}"


def _build_user_payload(normalized: dict, ratios: List[dict]) -> str:
    """Build the compact JSON payload sent as the user message.

    Includes the entity name, currency, a per-concept ``{fy: value}`` annual
    series (non-``None`` entries only), the full ratios list, and the list
    of concepts that could not be matched to any SEC tag. Serialized
    compactly (no extra whitespace) to keep token usage low.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.

    Returns:
        A compact JSON string.
    """
    annual_by_concept = {}
    for concept in normalized.get("annual") or {}:
        series = to_annual_series(normalized, concept)
        if series:
            annual_by_concept[concept] = {
                str(fy): value for fy, value in sorted(series.items(), reverse=True)
            }

    payload = {
        "entity_name": normalized.get("entity_name"),
        "currency": normalized.get("currency", "USD"),
        "scale_note": "All monetary figures are raw USD amounts (not scaled to thousands/millions).",
        "annual": annual_by_concept,
        "ratios": ratios,
        "missing": normalized.get("missing") or [],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _strip_json_fence(text: str) -> str:
    """Strip a wrapping ```json ... ``` / ``` ... ``` markdown code fence.

    Robust to input that has no fence at all -- in that case the (stripped)
    input is returned unchanged.

    Args:
        text: Raw model output.

    Returns:
        The text with any wrapping code fence removed, whitespace-trimmed.
    """
    if not text:
        return text
    stripped = text.strip()
    match = _FENCE_PATTERN.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _parse_model_json(text: str) -> dict:
    """Strip any code fence and parse a model's reply as JSON.

    Shared by every backend: whatever raw text a provider hands back goes
    through the same fence-stripping and parsing logic, so the JSON output
    contract is enforced identically regardless of which LLM produced it.

    Args:
        text: Raw model output (expected to be JSON, optionally fenced).

    Returns:
        The parsed dict on success. On failure, a dict of the form
        ``{"error": "parse_failed", "raw": text, "summary": ...}`` -- this
        function never raises.
    """
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Model response was not valid JSON: %s", exc)
        return {
            "error": "parse_failed",
            "raw": text,
            "summary": "Model did not return valid JSON.",
        }


def _call_anthropic(system: str, user: str, model: str, api_key: str) -> str:
    """Send one request to the Anthropic API and return the raw reply text.

    Args:
        system: System prompt (methodology + output-format instructions).
        user: User message (the compact JSON payload of financial data).
        model: Anthropic model ID, e.g. ``"claude-opus-4-8"``.
        api_key: Anthropic API key.

    Returns:
        The concatenated text of the response's text content blocks.

    Raises:
        anthropic.APIError: On any API-level failure.
        RuntimeError: If the ``anthropic`` package is not installed.
    """
    if anthropic is None:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run "
            "'pip install anthropic' to use the anthropic analyzer provider."
        )
    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Requesting Anthropic analysis using model %s", model)
    response = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def _call_ollama(system: str, user: str, model: str, host: str) -> str:
    """Send one request to a local Ollama server and return the raw reply text.

    Args:
        system: System prompt (methodology + output-format instructions).
        user: User message (the compact JSON payload of financial data).
        model: Name of a model already pulled in Ollama, e.g. ``"gemma4:latest"``.
        host: Base URL of the Ollama server, e.g. ``"http://localhost:11434"``.

    Returns:
        The model's raw response text.

    Raises:
        OllamaError: If the server is unreachable, times out, doesn't have
            the requested model, or returns a malformed response.
    """
    logger.info("Requesting Ollama analysis using model %s at %s", model, host)
    return chat_json(system=system, user=user, model=model, host=host)


def interpret(
    normalized: dict,
    ratios: List[dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
) -> dict:
    """Ask an LLM to produce a structured fundamental analysis.

    Builds a system prompt from :func:`load_methodology` plus a fixed JSON
    output contract, sends the normalized figures and ratios as a compact
    JSON user message to the selected backend, and parses the response back
    into a dict matching that contract.

    This function is designed to never raise: any failure (unknown provider,
    missing ``anthropic`` package, missing API key, Ollama not running or
    missing the requested model, network/API error, or a response that
    isn't valid JSON) is caught and returned as
    ``{"error": ..., "summary": ...}`` so callers (in particular the CLI)
    can display a warning and continue rather than crashing.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
        provider: Which backend to use: ``"ollama"``, ``"anthropic"``, or
            ``"script"`` (deterministic, no-AI). Defaults to
            ``Config.ANALYZER_PROVIDER`` (itself ``"ollama"`` unless
            overridden via the ``ANALYZER_PROVIDER`` environment variable).
            ``"gemma"`` is accepted as an alias for ``"ollama"``; ``"rule"``,
            ``"rules"``, ``"none"``, ``"noai"``, and ``"no-ai"`` are all
            accepted as aliases for ``"script"``.
        model: Model ID/name to use. Defaults to ``Config.OLLAMA_MODEL`` for
            the ollama provider or ``Config.ANTHROPIC_MODEL`` for anthropic.
        api_key: Anthropic API key to use (anthropic provider only).
            Defaults to the key returned by ``Config.require_anthropic_key()``.
        host: Base URL of the Ollama server (ollama provider only). Defaults
            to ``Config.OLLAMA_HOST``.

    Returns:
        On success, a dict matching the documented schema
        (``fair_value_range``, ``fundamental_verdict``, ``cyclical_risk``,
        ``key_ratios``, ``summary``), plus ``_provider``/``_model`` keys
        identifying what produced it. On failure, a dict of the form
        ``{"error": <str>, "summary": <str>}``, optionally with a ``"raw"``
        key holding the unparsed model output.
    """
    resolved_provider = (provider or Config.ANALYZER_PROVIDER or "ollama").lower()

    if resolved_provider in ("script", "rule", "rules", "none", "noai", "no-ai"):
        try:
            from sec_analyzer.interpret.rule_based import analyze as _rule_analyze

            return _rule_analyze(normalized, ratios)
        except Exception as exc:  # noqa: BLE001 - never let interpret() raise
            logger.exception("Unexpected error during script-based interpret()")
            return {
                "error": str(exc),
                "summary": "An unexpected error occurred while running the script-based analysis.",
                "_provider": "script",
            }

    system_prompt = _build_system_prompt()
    user_payload = _build_user_payload(normalized, ratios)

    if resolved_provider in ("ollama", "gemma"):
        resolved_model = model or Config.OLLAMA_MODEL
        resolved_host = host or Config.OLLAMA_HOST
        try:
            raw_text = _call_ollama(system_prompt, user_payload, resolved_model, resolved_host)
        except OllamaError as exc:
            logger.error("Ollama analysis failed: %s", exc)
            return {
                "error": str(exc),
                "summary": "Local Ollama analysis is unavailable; see error for details.",
                "_provider": "ollama",
            }
        except Exception as exc:  # noqa: BLE001 - never let interpret() raise
            logger.exception("Unexpected error during Ollama interpret()")
            return {
                "error": str(exc),
                "summary": "An unexpected error occurred while running the local analysis.",
                "_provider": "ollama",
            }
        result = _parse_model_json(raw_text)
        result.setdefault("_provider", "ollama")
        result.setdefault("_model", resolved_model)
        return result

    if resolved_provider == "anthropic":
        if anthropic is None:
            message = (
                "The 'anthropic' package is not installed. Run "
                "'pip install anthropic' to enable the anthropic analyzer provider."
            )
            logger.error(message)
            return {"error": "anthropic_not_installed", "summary": message, "_provider": "anthropic"}

        try:
            resolved_key = api_key or Config.require_anthropic_key()
            resolved_model = model or Config.ANTHROPIC_MODEL
            raw_text = _call_anthropic(system_prompt, user_payload, resolved_model, resolved_key)
        except ConfigError as exc:
            logger.error("Cannot run Anthropic analysis: %s", exc)
            return {
                "error": str(exc),
                "summary": "Anthropic API key is not configured; skipping analysis.",
                "_provider": "anthropic",
            }
        except anthropic.APIError as exc:
            logger.exception("Anthropic API error during interpret()")
            return {
                "error": str(exc),
                "summary": "The Anthropic API returned an error; analysis unavailable.",
                "_provider": "anthropic",
            }
        except Exception as exc:  # noqa: BLE001 - never let interpret() raise
            logger.exception("Unexpected error during interpret()")
            return {
                "error": str(exc),
                "summary": "An unexpected error occurred while running the analysis.",
                "_provider": "anthropic",
            }
        result = _parse_model_json(raw_text)
        result.setdefault("_provider", "anthropic")
        result.setdefault("_model", resolved_model)
        return result

    message = f"Unknown analyzer provider {resolved_provider!r}."
    logger.error(message)
    return {
        "error": message,
        "summary": "Set ANALYZER_PROVIDER to 'ollama', 'anthropic', or 'script'.",
        "_provider": resolved_provider,
    }
