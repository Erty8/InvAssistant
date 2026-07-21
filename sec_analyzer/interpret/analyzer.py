"""Two-phase LLM-powered valuation interpretation of normalized SEC financials.

This module is the only place in ``sec_analyzer`` that talks to a language
model for analysis. Per ``sec_analyzer/valuation/SPEC.md`` Sec.12, the flow
is split into two phases so that fair-value NUMBERS are always produced by
deterministic Python (:mod:`sec_analyzer.valuation`), never by an LLM:

* **Phase 1 -- assumption proposal** (:func:`propose_assumptions`): the LLM
  (or, for the ``"script"`` provider, :func:`sec_analyzer.interpret.
  rule_based.default_assumptions`) proposes a bear/base/bull growth/
  terminal-growth/discount-rate set plus a sector-type guess. Every
  proposal is run through :func:`sec_analyzer.valuation.sanity.
  validate_assumptions`; a violating LLM proposal gets exactly one
  revision request before falling back to the deterministic default.
* **Phase 2 -- commentary** (:func:`interpret_results`): the deterministic
  :func:`sec_analyzer.valuation.engine.run_valuation` turns phase 1's
  assumptions into an actual fair-value band (DCF/P-B x ROE scenarios,
  reverse DCF, multiples percentiles, sensitivity, triangulation). The LLM
  (or, for ``"script"``, :func:`sec_analyzer.interpret.rule_based.
  commentary`) is handed that complete ``valuation`` dict and asked to
  comment on it -- never to recompute or contradict any number in it.
  Several fields are still code-enforced on top of whatever the provider
  returns (see :func:`_postprocess_phase2_result`): ``technical_verdict``,
  ``confidence``, ``fair_value_range``, a cross-check of
  ``fundamental_verdict`` against the deterministic DCF signal, and the
  attached ``valuation`` dict itself.
* :func:`interpret` is a thin backward-compatible wrapper that runs both
  phases (phase 1 -> :func:`sec_analyzer.valuation.run_valuation` -> phase
  2) when no precomputed ``valuation`` is supplied, or just phase 2 when
  one is -- so existing callers (``sec_analyzer.cli``, ``sec_analyzer.web.
  app``) keep working unchanged.

Unified phase-2 output schema (identical across all three providers, and
identical to what pre-two-phase callers already expect)::

    {
      "fair_value_range": {
        "bear": {"lo": <num|null>, "hi": <num|null>, "growth": <str>,
                  "discount_rate": <str>, "note": <str>},
        "base": {...}, "bull": {...}
      },
      "fundamental_verdict": "UCUZ" | "MAKUL" | "PAHALI",
      "technical_verdict": <str -- always overwritten by this module from
        the ``technical`` argument; no provider, including the LLMs, ever
        decides this itself>,
      "confidence": "YÜKSEK" | "ORTA" | "DÜŞÜK" (from
        valuation.triangulation.confidence),
      "profile_fit": {"verdict": "UYUMLU" | "KISMEN" | "UYUMSUZ", "reason": <str>},
      "reverse_dcf_comment": <str>, "cyclical_risk": <str>,
      "horizon_note": <str>, "key_risks": [<str>, ...],
      "red_flags_comment": <str>, "catalyst": <str>, "summary": <str>,
      "valuation": <the full dict from valuation.engine.run_valuation>,
      "scenario_returns": {
        "bear": {"ret_lo_pct": <num|null>, "ret_hi_pct": <num|null>},
        "base": {...}, "bull": {...}
      },
      "entry_plan": [
        {"n": <int>, "trigger": <str>, "price_zone": {"lo": <num>, "hi": <num>},
         "size_pct": <num>, "invalidation": <num>, "target": <num|null>,
         "rr": <num|null>, "note": <str|null>}, ...
      ],
      "stop_adding": [{"code": <str>, "message": <str>}, ...],
      "thesis_metric": {"name": <str>, "latest_value": <str|null>,
        "trend": <str|null>, "rationale": <str>, "cycle": <dict|null>},
      "_provider": <str>, "_model": <str>,
      "_horizon": <str>, "_weights": {"fundamental": <float>, "technical": <float>}
    }

``scenario_returns``, ``entry_plan``, ``stop_adding``, and ``thesis_metric``
are ALWAYS computed by deterministic code
(:mod:`sec_analyzer.interpret.planning`) and injected into every provider's
result by :func:`_postprocess_phase2_result`, mirroring how
``fair_value_range``/``confidence`` are injected -- no provider, including
the LLMs, computes these fields itself (see ``METODOLOJI.md`` Sec.1 items
4-7).

All fair-value figures are USD per share.

Three backends are supported, selected via ``Config.ANALYZER_PROVIDER`` (or
the ``provider`` argument to :func:`interpret`/:func:`propose_assumptions`/
:func:`interpret_results`):

* ``"ollama"`` (**default**) -- a local Gemma model served by Ollama. Free,
  private, and requires no API key, but does require Ollama to be running
  locally with the model already pulled (see
  :mod:`sec_analyzer.interpret.ollama_client`).
* ``"anthropic"`` -- the hosted Claude API. Requires ``ANTHROPIC_API_KEY``.
* ``"script"`` -- deterministic, script-based (no-AI) phase 1/phase 2
  computed entirely with plain arithmetic and templates; no network access,
  no API key, and no LLM involved at all (see
  :mod:`sec_analyzer.interpret.rule_based`).

Design goals:

* **Never crash the CLI.** Every failure mode -- a missing API key, Ollama
  not running, a network error, an API error, or a response that isn't valid
  JSON -- is caught and turned into a small error dict with a human-readable
  ``summary`` instead of propagating an exception.
* **Import-safe without optional dependencies.** The ``anthropic`` package is
  an optional extra; if it isn't installed, importing this module still
  succeeds, and every public function returns a friendly error dict
  instructing the user to install it (only when the Anthropic backend is
  actually selected).
* **Configurable methodology.** The system prompt is built from
  ``METODOLOJI.md`` (:func:`load_methodology`) and ``VALUATION.md``
  (:func:`load_valuation_rules`) so the analysis philosophy and the
  deterministic valuation engine's rules can be swapped in without touching
  code.
* **Shared prompt/parsing, swappable transport.** Building the system/user
  prompts and parsing the model's JSON reply is identical regardless of
  backend; only the raw "send this, get text back" call
  (:func:`_call_anthropic` / :func:`_call_ollama`, dispatched via
  :func:`_dispatch_llm_call`) differs per provider.
"""

import json
import logging
import os
import re
from typing import List, Optional, Tuple

from sec_analyzer.config import Config, ConfigError
from sec_analyzer.interpret import planning, rule_based
from sec_analyzer.interpret.ollama_client import OllamaError, chat_json
from sec_analyzer.normalize.normalizer import to_annual_series
from sec_analyzer.valuation import damodaran, run_valuation, validate_assumptions
from sec_analyzer.valuation.capm import compute_cost_of_equity
from sec_analyzer.valuation.sector import classify_sector

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only when the optional
    # dependency is genuinely absent.
    anthropic = None

logger = logging.getLogger(__name__)

#: Maximum tokens requested from Claude for a single interpretation call.
_MAX_TOKENS = 2000

#: Provider aliases that select the deterministic, no-AI "script" backend
#: for both phases.
_SCRIPT_PROVIDER_ALIASES = ("script", "rule", "rules", "none", "noai", "no-ai")

#: Non-canonical provider aliases normalized before the resolved provider is
#: stamped into a result's ``_provider`` field (``_dispatch_llm_call`` still
#: accepts the alias directly -- this only affects the stamp).
_PROVIDER_ALIASES = {"gemma": "ollama"}


def _canonical_provider(resolved_provider: str) -> str:
    """Normalize a provider alias (e.g. ``"gemma"``) to its canonical name
    (``"ollama"``) for stamping into ``_provider``/logging. Transport
    dispatch (:func:`_dispatch_llm_call`) still accepts the alias itself."""
    return _PROVIDER_ALIASES.get(resolved_provider, resolved_provider)

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

Be explicit about uncertainty throughout: call out missing concepts, thin
history, inconsistent trends, or anything else that limits confidence in
the analysis. This analysis is educational commentary on public financial
filings, not personalized investment advice."""

#: Default valuation rules, used whenever no ``VALUATION.md`` is supplied
#: (see Config.VALUATION_PATH and load_valuation_rules()). Summarizes the
#: binding rules enforced in code by ``sec_analyzer.valuation`` (sanity
#: bounds, sector-to-method map, reverse-DCF/triangulation interpretation)
#: so an LLM backend still has the full picture even without the richer
#: (Turkish) VALUATION.md file this package ships with.
DEFAULT_VALUATION_RULES = """\
This analysis uses a two-phase valuation flow. In phase 1 you ONLY propose
growth/terminal-growth/discount-rate assumptions (never a fair-value
number); in phase 2 you are given a complete set of already-computed
fair-value figures and ONLY comment on them. Deterministic Python code
(not you) turns phase-1 assumptions into phase-2 numbers, so the same
assumptions always produce the same fair-value band.

Phase-1 hard limits (a violating proposal is rejected and you get one
chance to revise; if still invalid, a deterministic default is substituted
instead of your proposal), per bear/base/bull scenario:

- terminal_growth must not exceed 4%.
- discount_rate is a levered COST OF EQUITY (özkaynak maliyeti), NOT a WACC
  -- the DCF is FCFE-direct (the projected free cash flow is already a
  levered/equity cash flow, so no net-debt bridge is applied), so it must be
  discounted at a cost of equity. For a typical profitable large-cap this is
  usually ~8-12%, higher for a riskier, smaller, or unprofitable company.
- discount_rate must be at least 7% (at least 10% if the company is
  currently unprofitable) -- a low discount rate hides risk.
- discount_rate must always be strictly greater than terminal_growth (the
  Gordon-growth terminal value is undefined otherwise; this is never
  silently "fixed"), and by a comfortable margin -- a discount rate only a
  point or two above terminal_growth implies an implausibly thin equity
  risk premium and is rejected too.
- growth_5y above 20% is allowed (the model structurally fades growth
  toward terminal_growth after year 5), but above 40% is rejected as
  implausible.
- Every numeric field must be a real number, and "story" must be a
  concrete, one-sentence rationale (not a vague label like "optimistic
  scenario").

Sector-to-method map (final classification is made deterministically from
the filer's SIC code by application code -- your "sector_type" guess is
only used as a fallback when SIC is unavailable):

- financial: a P/B x ROE anchor is used instead of a cash-flow DCF (FCF-based
  DCF is unreliable for banks/insurers).
- reit: an FFO-based Gordon growth model (FFO = net income + depreciation/
  amortization) is used instead of a cash-flow DCF, with P/FFO-based
  multiples (FCF-based DCF is unreliable for REITs).
- growth_unprofitable: a P/S multiple and reverse DCF are weighted most
  heavily in the triangulation (P/E and P/FCF percentiles are usually
  unavailable).
- cyclical: an additional normalized-earnings DCF variant (median
  historical FCF margin x latest revenue) is computed alongside the
  standard DCF.
- mature: standard DCF plus historical/sector-relative multiples.

Reverse-DCF interpretation: compare the price-implied growth rate against
the realized revenue CAGR. Implied growth more than 3 percentage points
above the realized rate is an expensiveness signal; more than 3 points
below is a cheapness signal; within that band is roughly consistent with
the realized trend.

Triangulation and confidence: three independent methods (DCF/P-B x ROE,
reverse DCF, multiples percentile) each vote cheap/fair/expensive/no-data.
All three agreeing is high confidence; two agreeing is medium; a scattered
or mostly-missing read is low confidence -- this confidence level is always
computed by code, never asserted by you. Your "fundamental_verdict" must
never contradict the DCF (or P/B x ROE) signal on the cheap-vs-expensive
axis; application code overrides it (and logs the override) if it does."""

#: Explicit output-format contract for phase 1 (assumption proposal). Kept
#: separate from the methodology/valuation-rules text so the JSON contract
#: is always enforced regardless of which text (default or user-supplied)
#: precedes it.
_PHASE1_OUTPUT_CONTRACT = """\
Respond with ONLY a single JSON object -- no prose before or after it, and
no markdown code fences. The JSON object must match exactly this schema:

{
  "assumptions": {
    "bear": {"growth_5y": <number, decimal fraction e.g. 0.08 = 8%>,
              "terminal_growth": <number, decimal fraction>,
              "discount_rate": <number, decimal fraction>,
              "story": <string, one concrete Turkish sentence>},
    "base": {"growth_5y": <number>, "terminal_growth": <number>,
              "discount_rate": <number>, "story": <string>},
    "bull": {"growth_5y": <number>, "terminal_growth": <number>,
              "discount_rate": <number>, "story": <string>}
  },
  "sector_type": <string, exactly one of "cyclical", "financial",
    "growth_unprofitable", "hyper_growth", "mature", "reit" -- your best
    guess from the figures given; the final classification is made
    deterministically from the filer's SIC code by application code and may
    override this; "hyper_growth" is also detected independently and
    deterministically from the financials by application code -- your guess
    here is only ever a hint>,
  "hyper_growth_extras": <OPTIONAL object -- include ONLY when your
    "sector_type" guess above is "hyper_growth", and only to refine the
    deterministic revenue-first DCF (never required):
    {"tam_usd": <number, total addressable market in raw USD>,
     "tam_rationale": <string, one concrete Turkish sentence>,
     "per_scenario": {
       "bear": {"target_fcf_margin": <number, decimal fraction -- the
                   mature-state FCF margin this scenario converges to>,
                 "steady_state_year": <integer, year by which growth and
                   margin are both fully converged>,
                 "probability": <number, decimal fraction>},
       "base": {...same three keys...},
       "bull": {...same three keys...}
     }}
    Omit this key entirely (or omit any of its sub-fields) when you have no
    genuine TAM/margin refinement to add -- the deterministic engine fills
    in every omitted piece on its own.>
}

All rates are decimal fractions (0.08 = 8%), never percent numbers (8).
Respect every hard limit listed above for terminal_growth, discount_rate,
and growth_5y. bear's growth_5y should be <= base's <= bull's.

Do NOT compute or return any fair-value number, per-share figure, or dollar
amount here -- that is done by deterministic Python code downstream. This
phase ONLY proposes assumption ranges (plus, optionally, the hyper-grower
refinements above)."""

#: Explicit output-format contract for phase 2 (commentary on an
#: already-computed valuation).
_PHASE2_OUTPUT_CONTRACT = """\
You are given, in the user payload's "valuation" object, a COMPLETE set of
already-computed fair-value figures (DCF or P/B x ROE scenarios, reverse
DCF, multiples percentiles, sensitivity, triangulation signals) produced by
deterministic Python code. Every number in that object is final -- you must
NOT recompute, adjust, "correct", or contradict any of it. Your only job is
to comment on it.

Respond with ONLY a single JSON object -- no prose before or after it, and
no markdown code fences. The JSON object must match exactly this schema:

{
  "fundamental_verdict": <string, exactly one of "UCUZ", "MAKUL", "PAHALI"
    -- judge this from valuation.triangulation and valuation.fair_value_range
    (base band) vs. the current price; application code will override this
    field if it contradicts valuation.triangulation.signals.dcf>,
  "profile_fit": {"verdict": <string, exactly one of "UYUMLU", "KISMEN",
    "UYUMSUZ">, "reason": <string, one sentence, judged from the investor
    profile text in this system prompt and the current horizon>},
  "reverse_dcf_comment": <string -- interpret valuation.reverse_dcf
    (implied_growth vs. realized_cagr_5y/realized_label) per the reverse-DCF
    rule described above, stating the numbers explicitly>,
  "cyclical_risk": <string describing growth stability / cyclicality,
    referencing valuation.sector_type and any cyclical-trap red flag
    supplied in the user payload>,
  "horizon_note": <string, one sentence on what the current horizon
    emphasizes for this filer, and, if valuation.sensitivity.high_uncertainty
    is true, an explicit note about that>,
  "key_risks": [<string>, ...],
  "red_flags_comment": <string -- "yok" if no red flags were supplied,
    otherwise a short synthesis of them>,
  "catalyst": <string -- the upcoming catalyst label if one was supplied,
    otherwise "bilinmiyor">,
  "summary": <string, a short plain-language paragraph summarizing the
    analysis, referencing the base fair-value band and the confidence level
    in valuation.triangulation.confidence>
}

Do NOT include "fair_value_range", "technical_verdict", "confidence",
"valuation", "scenario_returns", "entry_plan", "stop_adding", or
"thesis_metric" keys in your response -- those are always supplied/overwritten
by application code regardless of what you write.

This is an educational, non-personalized analysis of public SEC filings, not
investment advice."""

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


def load_valuation_rules() -> str:
    """Return the valuation-engine rules to inject into the system prompt.

    Mirrors :func:`load_methodology`: if ``Config.VALUATION_PATH`` points at
    an existing, non-empty file, its contents are used verbatim (the
    package ships a richer, Turkish ``VALUATION.md`` at the default path --
    see that file for the full rule set this function's default only
    summarizes). Otherwise -- or if reading the file fails for any reason
    -- :data:`DEFAULT_VALUATION_RULES` is used. Injected into both the
    phase-1 and phase-2 system prompts, right after the methodology and
    before (phase 2 only) the investor profile.

    Returns:
        The valuation-rules text (never empty).
    """
    path = Config.VALUATION_PATH
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                logger.info("Using custom valuation rules from %s", path)
                return content
            logger.info(
                "Valuation rules file %s is empty; using the default valuation rules.",
                path,
            )
        else:
            logger.info(
                "No valuation rules file found at %s; using the default valuation rules.",
                path,
            )
    except OSError as exc:
        logger.warning(
            "Failed to read valuation rules file %s (%s); using the default "
            "valuation rules.",
            path, exc,
        )
    return DEFAULT_VALUATION_RULES


#: Neutral-default note used in place of an investor profile when
#: ``Config.PROFIL_PATH`` doesn't exist (or can't be read) -- tells the
#: model to assume a neutral investor and nudges the user to create one.
_NEUTRAL_PROFILE_NOTE = (
    "Profil dosyası yok; nötr bir yatırımcı varsay ve kullanıcıya PROFIL.md "
    "oluşturmasını öner."
)

#: Per-horizon guidance text, formatted with the (fundamental_pct,
#: technical_pct) weights resolved from ``Config.HORIZON_WEIGHTS``. Used by
#: :func:`_build_horizon_instruction`.
_HORIZON_GUIDANCE = {
    "3m": (
        "Vade: 3m. Sinyal ağırlıkları: fundamental %{fw:.0f} / teknik %{tw:.0f}. "
        "Bu ufukta teknik ve momentum sinyalleri (RSI, SMA50, volatilite) öncelikli "
        "olmalı; yaklaşan katalizör (kazanç tarihi vb.) kritik önemdedir."
    ),
    "1y": (
        "Vade: 1y. Sinyal ağırlıkları: fundamental %{fw:.0f} / teknik %{tw:.0f}. "
        "Bu ufukta fundamental ve teknik sinyaller dengeli şekilde değerlendirilmelidir."
    ),
    "5y": (
        "Vade: 5y. Sinyal ağırlıkları: fundamental %{fw:.0f} / teknik %{tw:.0f}. "
        "Bu ufukta fundamental sinyaller öncelikli olmalı; RSI gibi kısa vadeli "
        "göstergeler önemsizdir; döngüsel tepe (cyclical trap) kontrolü zorunludur."
    ),
}


def _load_profile() -> str:
    """Return the investor-profile section of the system prompt.

    If ``Config.PROFIL_PATH`` points at an existing, non-empty file, its
    contents are returned prefixed with a "## Yatırımcı Profili" heading so
    the model can weigh ``profile_fit`` against it. Otherwise -- or if
    reading the file fails for any reason -- :data:`_NEUTRAL_PROFILE_NOTE`
    is returned instead.

    Returns:
        The profile section text (never empty).
    """
    path = Config.PROFIL_PATH
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                logger.info("Using investor profile from %s", path)
                return f"## Yatırımcı Profili\n{content}"
            logger.info("Profile file %s is empty; using the neutral-default note.", path)
        else:
            logger.info("No profile file found at %s; using the neutral-default note.", path)
    except OSError as exc:
        logger.warning(
            "Failed to read profile file %s (%s); using the neutral-default note.",
            path, exc,
        )
    return _NEUTRAL_PROFILE_NOTE


def _build_horizon_instruction(horizon: str) -> str:
    """Render the horizon-specific instruction from ``Config.HORIZON_WEIGHTS``.

    Args:
        horizon: One of ``"3m"``, ``"1y"``, ``"5y"``. Unrecognized values
            fall back to the ``"1y"`` weights/guidance.

    Returns:
        A one-paragraph instruction naming the resolved fundamental/technical
        signal weights and horizon-specific guidance.
    """
    fundamental_weight, technical_weight = Config.HORIZON_WEIGHTS.get(
        horizon, Config.HORIZON_WEIGHTS["1y"]
    )
    template = _HORIZON_GUIDANCE.get(horizon, _HORIZON_GUIDANCE["1y"])
    return template.format(fw=fundamental_weight * 100, tw=technical_weight * 100)


def _build_phase1_system_prompt() -> str:
    """Assemble the phase-1 (assumption proposal) system prompt: methodology
    -> valuation rules -> the phase-1 JSON output contract. Deliberately
    excludes the investor profile and horizon instruction (SPEC.md Sec.12) --
    phase 1 proposes assumption ranges only, which don't depend on either."""
    sections = [load_methodology(), load_valuation_rules(), _PHASE1_OUTPUT_CONTRACT]
    return "\n\n".join(sections)


def _build_phase2_system_prompt(horizon: str = "1y") -> str:
    """Assemble the phase-2 (commentary) system prompt, in a fixed order:
    methodology -> valuation rules -> investor profile -> horizon
    instruction -> the phase-2 JSON output contract."""
    sections = [
        load_methodology(),
        load_valuation_rules(),
        _load_profile(),
        _build_horizon_instruction(horizon),
        _PHASE2_OUTPUT_CONTRACT,
    ]
    return "\n\n".join(sections)


def _annual_by_concept(normalized: dict) -> dict:
    """Build the per-concept ``{fy: value}`` annual-series payload fragment
    shared by both the phase-1 and phase-2 user payloads (non-``None``
    entries only, most-recent fiscal year first)."""
    annual_by_concept = {}
    for concept in normalized.get("annual") or {}:
        series = to_annual_series(normalized, concept)
        if series:
            annual_by_concept[concept] = {
                str(fy): value for fy, value in sorted(series.items(), reverse=True)
            }
    return annual_by_concept


def _build_phase1_user_payload(
    normalized: dict,
    ratios: List[dict],
    metrics: Optional[dict],
    sector_hint: Optional[str],
    horizon: str,
) -> str:
    """Build the compact JSON payload sent as the phase-1 user message.

    Includes the entity name, currency, the per-concept annual series, the
    full ratios list, the list of unmatched concepts, the valuation metrics
    (revenue CAGRs in particular), an optional deterministic sector guess
    (see :func:`sec_analyzer.valuation.sector.classify_sector`), and the
    horizon (context only -- phase 1 does not weight scenarios by horizon).
    Serialized compactly to keep token usage low.
    """
    payload = {
        "entity_name": normalized.get("entity_name"),
        "currency": normalized.get("currency", "USD"),
        "scale_note": "All monetary figures are raw USD amounts (not scaled to thousands/millions).",
        "annual": _annual_by_concept(normalized),
        "ratios": ratios,
        "missing": normalized.get("missing") or [],
        "metrics": metrics or {},
        "sector_hint": sector_hint,
        "horizon": horizon,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _build_phase1_revision_payload(original_user_payload: str, violations: List[str]) -> str:
    """Append a Turkish sanity-violation list to the original phase-1 user
    payload, requesting the single allowed revision (SPEC.md Sec.12)."""
    violation_lines = "\n".join(f"- {v}" for v in violations)
    return (
        f"{original_user_payload}\n\n"
        "Şu sınırlar ihlal edildi, varsayımları revize et:\n"
        f"{violation_lines}\n\n"
        'Aynı JSON şemasıyla (yalnızca "assumptions" ve "sector_type") '
        "düzeltilmiş önerini gönder."
    )


def _build_phase2_user_payload(
    normalized: dict,
    ratios: List[dict],
    metrics: Optional[dict],
    technical: Optional[dict],
    red_flags: Optional[List[dict]],
    catalyst: Optional[dict],
    valuation: dict,
) -> str:
    """Build the compact JSON payload sent as the phase-2 user message.

    Same financial/technical/red-flags/catalyst payload phase 2 has always
    sent, plus the full ``valuation`` dict (the deterministic figures phase
    2 comments on and must never recompute).

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None``.
        technical: The merged indicators + verdict dict from
            :mod:`sec_analyzer.technical`, or ``None``.
        red_flags: The list of ``{"code", "message", "detail"}`` dicts from
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``.
        catalyst: The ``{"estimate_date", "label", "based_on"}`` dict from
            :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
            ``None``.
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation`.

    Returns:
        A compact JSON string.
    """
    payload = {
        "entity_name": normalized.get("entity_name"),
        "currency": normalized.get("currency", "USD"),
        "scale_note": "All monetary figures are raw USD amounts (not scaled to thousands/millions).",
        "annual": _annual_by_concept(normalized),
        "ratios": ratios,
        "missing": normalized.get("missing") or [],
    }

    price = None
    as_of = None
    if metrics:
        payload["metrics"] = metrics
        price = metrics.get("price")
    if technical:
        payload["technical"] = {
            "verdict": technical.get("verdict"),
            "verdict_detail": technical.get("verdict_detail"),
            "horizon_summary": technical.get("horizon_summary"),
            "price": technical.get("price"),
            "as_of": technical.get("as_of"),
            "rsi14": technical.get("rsi14"),
            "sma50": technical.get("sma50"),
            "sma200": technical.get("sma200"),
            "dist_sma50_pct": technical.get("dist_sma50_pct"),
            "range_position_pct": technical.get("range_position_pct"),
            "volatility_20d": technical.get("volatility_20d"),
            "sma50_above_sma200": technical.get("sma50_above_sma200"),
        }
        if price is None:
            price = technical.get("price")
        if as_of is None:
            as_of = technical.get("as_of")
    payload["current_price"] = price
    payload["price_as_of"] = as_of

    if red_flags:
        payload["red_flags"] = red_flags
    if catalyst:
        payload["catalyst"] = catalyst.get("label")

    payload["valuation"] = valuation

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

    Shared by every backend and both phases: whatever raw text a provider
    hands back goes through the same fence-stripping and parsing logic, so
    the JSON output contract is enforced identically regardless of which
    LLM produced it.

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
        logger.warning(
            "Model response was not valid JSON: %s; response starts with: %r",
            exc,
            cleaned[:200] if cleaned else "<empty>",
        )
        return {
            "error": "parse_failed",
            "raw": text,
            "summary": "Model did not return valid JSON.",
        }


def _call_anthropic(system: str, user: str, model: str, api_key: str) -> str:
    """Send one request to the Anthropic API and return the raw reply text.

    Args:
        system: System prompt (methodology + valuation rules + output-
            format instructions, phase-dependent).
        user: User message (the compact JSON payload for the current phase).
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
        system: System prompt (methodology + valuation rules + output-
            format instructions, phase-dependent).
        user: User message (the compact JSON payload for the current phase).
        model: Name of a model already pulled in Ollama, e.g. ``"gemma4:latest"``.
        host: Base URL of the Ollama server, e.g. ``"http://localhost:11434"``.

    Returns:
        The model's raw response text.

    Raises:
        OllamaError: If the server is unreachable, times out, doesn't have
            the requested model, or returns a malformed response.
    """
    logger.info("Requesting Ollama analysis using model %s at %s", model, host)
    return chat_json(
        system=system, user=user, model=model, host=host,
        timeout=Config.OLLAMA_TIMEOUT, num_ctx=Config.OLLAMA_NUM_CTX,
    )


def _dispatch_llm_call(
    resolved_provider: str,
    system: str,
    user: str,
    model: Optional[str],
    api_key: Optional[str],
    host: Optional[str],
) -> Tuple[str, str]:
    """Call the resolved LLM provider and return ``(raw_text, resolved_model)``.

    Shared transport dispatch used by both :func:`propose_assumptions` and
    :func:`interpret_results` -- everything about building prompts and
    handling the result differs by phase, but "which function do I call for
    this provider" does not.

    Args:
        resolved_provider: ``"ollama"``, ``"gemma"`` (an ollama alias), or
            ``"anthropic"``. The ``"script"`` provider never reaches this
            function -- callers branch to the rule-based path before
            building a system/user prompt at all.
        system: System prompt for the current phase.
        user: User payload for the current phase.
        model: Model override, or ``None`` to use the provider's configured
            default.
        api_key: Anthropic API key override (anthropic provider only).
        host: Ollama host override (ollama provider only).

    Returns:
        ``(raw_text, resolved_model)``.

    Raises:
        OllamaError: Ollama transport failure (server unreachable, timeout,
            model not pulled, malformed response).
        ConfigError: Anthropic API key not configured.
        RuntimeError: The ``anthropic`` package is not installed.
        ValueError: ``resolved_provider`` is none of the above.
        Exception: Any other provider-level failure (e.g.
            ``anthropic.APIError``) propagates as-is for the caller to
            catch.
    """
    if resolved_provider in ("ollama", "gemma"):
        resolved_model = model or Config.OLLAMA_MODEL
        resolved_host = host or Config.OLLAMA_HOST
        raw_text = _call_ollama(system, user, resolved_model, resolved_host)
        return raw_text, resolved_model

    if resolved_provider == "anthropic":
        if anthropic is None:
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run "
                "'pip install anthropic' to enable the anthropic analyzer provider."
            )
        resolved_key = api_key or Config.require_anthropic_key()
        resolved_model = model or Config.ANTHROPIC_MODEL
        raw_text = _call_anthropic(system, user, resolved_model, resolved_key)
        return raw_text, resolved_model

    raise ValueError(f"Unknown analyzer provider {resolved_provider!r}.")


def _extract_phase1_fields(parsed: dict) -> "Tuple[Optional[dict], Optional[str], Optional[dict]]":
    """Pull ``(assumptions, sector_type, hyper_growth_extras)`` out of a
    parsed phase-1 response.

    Only checks the shape (an ``assumptions`` dict with all three scenario
    keys present as dicts) -- the actual numeric/story validation is
    ``sanity.validate_assumptions``'s job, run separately by the caller.
    ``hyper_growth_extras`` (HYPER_SPEC.md Sec.5) is optional and only
    shape-checked (must be a dict, else dropped to ``None``) -- the engine's
    ``_build_hyper_growth`` defensively validates its actual contents
    (``tam_usd``, per-scenario overrides) on its own.

    Returns:
        ``(None, None, None)`` if ``parsed`` isn't a usable phase-1 response
        at all (a parse-failure dict, or missing/malformed ``assumptions``).
    """
    if not isinstance(parsed, dict) or "error" in parsed:
        return None, None, None
    assumptions = parsed.get("assumptions")
    if not isinstance(assumptions, dict) or not all(
        isinstance(assumptions.get(key), dict) for key in ("bear", "base", "bull")
    ):
        return None, None, None
    sector_type = parsed.get("sector_type")
    if not isinstance(sector_type, str):
        sector_type = None
    hyper_growth_extras = parsed.get("hyper_growth_extras")
    if not isinstance(hyper_growth_extras, dict):
        hyper_growth_extras = None
    return assumptions, sector_type, hyper_growth_extras


def _fallback_assumptions_result(
    metrics: dict,
    sector_hint: Optional[str],
    capm: Optional[dict] = None,
    risk_free_pct: Optional[float] = None,
) -> dict:
    """The deterministic phase-1 fallback result (SPEC.md Sec.12): used for
    the ``"script"`` provider and for every LLM failure mode.

    ``hyper_growth_extras`` is always ``None`` here -- the deterministic
    fallback never has an LLM-refined TAM/margin to offer (HYPER_SPEC.md
    Sec.5); the engine's own deterministic hyper-grower inputs (Sec.3) are
    unaffected and still apply when ``sector.detect_hyper_grower`` fires.

    ``capm`` (the :func:`sec_analyzer.valuation.capm.compute_cost_of_equity`
    result, or ``None``) is forwarded to ``default_assumptions`` so the base
    discount rate is the firm's CAPM cost of equity when it was computable.
    ``risk_free_pct`` (the global risk-free rate, independent of any
    SIC/industry matching) is likewise forwarded as the ``terminal_growth``
    fallback source used only when ``capm`` is absent or lacks its own
    numeric ``risk_free`` (see
    :func:`sec_analyzer.interpret.rule_based._terminal_growth_anchor`).
    """
    sector_type = sector_hint or "mature"
    return {
        "assumptions": rule_based.default_assumptions(
            metrics, sector_type, capm=capm, risk_free_pct=risk_free_pct
        ),
        "sector_type": sector_type,
        "hyper_growth_extras": None,
        "_provider": "script",
    }


def propose_assumptions(
    normalized: dict,
    ratios: List[dict],
    metrics: Optional[dict] = None,
    sector_hint: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    horizon: str = "1y",
    capm: Optional[dict] = None,
    risk_free_pct: Optional[float] = None,
) -> dict:
    """Phase 1 of the two-phase valuation flow: propose bear/base/bull
    growth/terminal-growth/discount-rate assumptions (SPEC.md Sec.12).

    Builds the phase-1 system prompt (:func:`_build_phase1_system_prompt`)
    and user payload (:func:`_build_phase1_user_payload`), asks the selected
    backend for a proposal, and runs it through
    :func:`sec_analyzer.valuation.sanity.validate_assumptions`. If that
    finds violations, the LLM is re-called exactly once with the violation
    list appended (in Turkish) asking for a revision; if the revised
    proposal is still invalid (or the LLM call fails at any point, or the
    provider is ``"script"``), the deterministic
    :func:`sec_analyzer.interpret.rule_based.default_assumptions` fallback
    is used instead. Never raises.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None``.
        sector_hint: A preliminary sector-type guess (typically from
            :func:`sec_analyzer.valuation.sector.classify_sector` when the
            filer's SIC code is already known), passed to the LLM as
            context and used as the fallback ``sector_type`` if the LLM is
            unavailable or its own guess is unusable. ``None`` if unknown.
        provider: Which backend to use: ``"ollama"``, ``"anthropic"``, or
            ``"script"`` (deterministic, no-AI). Defaults to
            ``Config.ANALYZER_PROVIDER``. ``"gemma"`` aliases ``"ollama"``;
            ``"rule"``, ``"rules"``, ``"none"``, ``"noai"``, ``"no-ai"``
            alias ``"script"``.
        model: Model ID/name to use. Defaults to ``Config.OLLAMA_MODEL`` for
            the ollama provider or ``Config.ANTHROPIC_MODEL`` for anthropic.
        api_key: Anthropic API key to use (anthropic provider only).
        host: Base URL of the Ollama server (ollama provider only).
        horizon: Investment horizon, included in the user payload as
            context only -- phase 1 does not weight assumptions by horizon.
        capm: The optional
            :func:`sec_analyzer.valuation.capm.compute_cost_of_equity`
            result, forwarded to the deterministic fallback as its CAPM base
            discount rate. ``None`` if unavailable.
        risk_free_pct: The global risk-free rate (a PERCENTAGE number, e.g.
            ``4.20`` for 4.2%), independent of any SIC/industry matching.
            Forwarded to the deterministic fallback as the ``terminal_growth``
            fallback source used only when ``capm`` is absent or lacks its
            own numeric ``risk_free`` (see
            :func:`sec_analyzer.interpret.rule_based._terminal_growth_anchor`).
            ``None`` if unavailable.

    Returns:
        ``{"assumptions": {"bear": {...}, "base": {...}, "bull": {...}},
        "sector_type": <str>, "hyper_growth_extras": <dict or None>,
        "_provider": <str>}`` -- always a validated (or
        deterministically-substituted) assumption set.
        ``hyper_growth_extras`` (HYPER_SPEC.md Sec.5) is the optional
        LLM-supplied TAM/margin refinement dict when the provider included
        a usable one, else ``None`` (the engine's own deterministic
        hyper-grower inputs still apply either way). Never an error dict:
        any failure degrades to the deterministic fallback instead.
    """
    try:
        return _propose_assumptions(
            normalized or {}, ratios or [], metrics or {}, sector_hint,
            provider, model, api_key, host, horizon, capm, risk_free_pct,
        )
    except Exception:  # noqa: BLE001 - phase 1 must never raise
        logger.exception("propose_assumptions() failed unexpectedly; falling back to deterministic defaults.")
        return _fallback_assumptions_result(metrics or {}, sector_hint, capm, risk_free_pct)


def _propose_assumptions(
    normalized: dict,
    ratios: List[dict],
    metrics: dict,
    sector_hint: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    host: Optional[str],
    horizon: str,
    capm: Optional[dict] = None,
    risk_free_pct: Optional[float] = None,
) -> dict:
    resolved_provider = (provider or Config.ANALYZER_PROVIDER or "ollama").lower()

    if resolved_provider in _SCRIPT_PROVIDER_ALIASES:
        return _fallback_assumptions_result(metrics, sector_hint, capm, risk_free_pct)

    system_prompt = _build_phase1_system_prompt()
    user_payload = _build_phase1_user_payload(normalized, ratios, metrics, sector_hint, horizon)

    try:
        raw_text, resolved_model = _dispatch_llm_call(resolved_provider, system_prompt, user_payload, model, api_key, host)
    except Exception as exc:  # noqa: BLE001 - any transport/config failure -> deterministic fallback
        logger.warning(
            "Phase-1 assumption proposal via %s failed (%s); using deterministic defaults.",
            resolved_provider, exc,
        )
        return _fallback_assumptions_result(metrics, sector_hint, capm, risk_free_pct)

    parsed = _parse_model_json(raw_text)
    assumptions, llm_sector_type, hyper_growth_extras = _extract_phase1_fields(parsed)

    if assumptions is None:
        logger.warning(
            "Phase-1 response from %s was not usable JSON; using deterministic defaults.",
            resolved_provider,
        )
        return _fallback_assumptions_result(metrics, sector_hint, capm, risk_free_pct)

    sector_type = llm_sector_type or sector_hint or "mature"
    violations = validate_assumptions(assumptions, is_unprofitable=(sector_type == "growth_unprofitable"))

    if violations:
        logger.info(
            "Phase-1 assumptions from %s violated sanity checks (%s); requesting one revision.",
            resolved_provider, violations,
        )
        revision_user_payload = _build_phase1_revision_payload(user_payload, violations)
        try:
            raw_text, _ = _dispatch_llm_call(
                resolved_provider, system_prompt, revision_user_payload, resolved_model, api_key, host
            )
        except Exception as exc:  # noqa: BLE001 - revision call failed -> deterministic fallback
            logger.warning(
                "Phase-1 revision call via %s failed (%s); using deterministic defaults.",
                resolved_provider, exc,
            )
            return _fallback_assumptions_result(metrics, sector_hint, capm, risk_free_pct)

        revised_parsed = _parse_model_json(raw_text)
        # The revision request explicitly asks the provider to resend only
        # "assumptions" and "sector_type" (_build_phase1_revision_payload),
        # so hyper_growth_extras is not expected in this response -- the
        # original response's extras (if any) carry over unchanged.
        revised_assumptions, revised_sector_type, _ = _extract_phase1_fields(revised_parsed)
        if revised_assumptions is None:
            logger.warning(
                "Phase-1 revision response from %s was not usable JSON; using deterministic defaults.",
                resolved_provider,
            )
            return _fallback_assumptions_result(metrics, sector_hint, capm, risk_free_pct)

        sector_type = revised_sector_type or sector_type
        violations = validate_assumptions(
            revised_assumptions, is_unprofitable=(sector_type == "growth_unprofitable")
        )
        if violations:
            logger.warning(
                "Phase-1 assumptions from %s still invalid after revision (%s); using deterministic defaults.",
                resolved_provider, violations,
            )
            return _fallback_assumptions_result(metrics, sector_hint, capm, risk_free_pct)
        assumptions = revised_assumptions

    return {
        "assumptions": assumptions,
        "sector_type": sector_type,
        "hyper_growth_extras": hyper_growth_extras,
        "_provider": _canonical_provider(resolved_provider),
    }


#: The hyper-grower-only verdict label (HYPER_SPEC.md Sec.4) -- the phase-2
#: contract's "fundamental_verdict" enum only ever asks a provider for
#: UCUZ/MAKUL/PAHALI, so this string can only ever originate from
#: :data:`_DCF_SIGNAL_TO_VERDICT` / :func:`_reconcile_fundamental_verdict`,
#: never from a provider's own output.
_HIGH_EXPECTATION_VERDICT = "YÜKSEK BEKLENTİ FİYATLANMIŞ"

#: Model–market divergence verdict (the DOWN-price mirror of
#: :data:`_HIGH_EXPECTATION_VERDICT`). Emitted deterministically by
#: :func:`_postprocess_phase2_result` when the triangulation governor
#: (``valuation.triangulation.divergence``, see ``valuation/triangulate.py``)
#: flags an up-side divergence (``action == "verdict"``): the base fair-value
#: band sits more than ~2x above price, so the three method votes read off ONE
#: assumption set rather than independently confirming "cheap". The honest
#: headline is then a model↔market disagreement, not "UCUZ". Applied AFTER the
#: reconcile step, overriding whatever verdict any provider (LLM or script)
#: produced; the low confidence is already set by the governor itself.
_DIVERGENCE_VERDICT = "MODEL-PİYASA AYRIŞMASI"

#: Map from a triangulation direction signal to the schema's verdict
#: string; "veri_yok" deliberately has no entry.
_DCF_SIGNAL_TO_VERDICT = {
    "ucuz": "UCUZ",
    "makul": "MAKUL",
    "pahali": "PAHALI",
    "yuksek_beklenti": _HIGH_EXPECTATION_VERDICT,
}

#: The only two verdicts that can contradict each other on the
#: cheap-vs-expensive axis -- "MAKUL" on either side is never a
#: contradiction (SPEC.md Sec.12).
_OPPOSITE_VERDICT = {"UCUZ": "PAHALI", "PAHALI": "UCUZ"}


def _reconcile_fundamental_verdict(llm_verdict: Optional[str], dcf_signal: Optional[str], provider: str) -> str:
    """Cross-check a provider's ``fundamental_verdict`` against the
    deterministic DCF (or P/B x ROE) triangulation signal.

    Only an outright contradiction on the ucuz<->pahali axis is overridden:
    a "MAKUL" from either side, or a missing/"veri_yok" code signal, is not
    a contradiction (there's nothing to override against). The provider
    can never win a direct disagreement with the code-computed signal.

    In hyper-grower mode (HYPER_SPEC.md Sec.4), ``dcf_signal`` may be
    ``"yuksek_beklenti"`` -- a valid, non-contradictory code-computed state
    (not an error case) that maps to its own
    :data:`_HIGH_EXPECTATION_VERDICT` label rather than being squeezed into
    UCUZ/MAKUL/PAHALI. Unlike the ucuz<->pahali cross-check above, this
    mapping always wins outright: no provider (an LLM, or the "script"
    provider's own ``rule_based._fundamental_verdict_from_valuation``
    fallback, which defaults unrecognized signals to "MAKUL") is ever asked
    for this 4th value, so there is nothing for it to legitimately
    contribute here -- it is purely a code-side classification of the
    already-computed bands.

    Args:
        llm_verdict: The provider's own ``fundamental_verdict`` (or
            ``None``/anything not in ``{"UCUZ", "MAKUL", "PAHALI"}``, e.g.
            when JSON parsing failed).
        dcf_signal: ``valuation["triangulation"]["signals"]["dcf"]`` --
            ``"ucuz"``, ``"makul"``, ``"pahali"``, ``"yuksek_beklenti"``, or
            ``"veri_yok"``.
        provider: Resolved provider name, used only for the log message
            when an override actually happens.

    Returns:
        The final ``fundamental_verdict`` string.
    """
    code_verdict = _DCF_SIGNAL_TO_VERDICT.get(dcf_signal)
    llm_verdict = llm_verdict if llm_verdict in ("UCUZ", "MAKUL", "PAHALI") else None

    if code_verdict is None:
        return llm_verdict or "MAKUL"
    if code_verdict == _HIGH_EXPECTATION_VERDICT:
        return code_verdict
    if llm_verdict is None:
        return code_verdict
    if _OPPOSITE_VERDICT.get(llm_verdict) == code_verdict:
        logger.warning(
            "%s provider's fundamental_verdict (%s) contradicted the DCF triangulation "
            "signal (%s) on the ucuz<->pahali axis; overriding with the code signal.",
            provider, llm_verdict, code_verdict,
        )
        return code_verdict
    return llm_verdict


def _postprocess_phase2_result(
    result: dict,
    provider: str,
    model: str,
    horizon: str,
    technical: Optional[dict],
    catalyst: Optional[dict],
    valuation: dict,
    ratios: Optional[List[dict]] = None,
    red_flags: Optional[List[dict]] = None,
    metrics: Optional[dict] = None,
) -> dict:
    """Apply the fixed, provider-agnostic phase-2 post-processing rules
    (SPEC.md Sec.12 step 2) that no provider -- LLM or ``"script"`` -- can
    override:

    * ``technical_verdict`` is ALWAYS overwritten from ``technical`` (the
      technical read is code-derived, exactly as in the pre-two-phase flow).
    * ``confidence`` is ALWAYS ``valuation["triangulation"]["confidence"]``.
    * ``fair_value_range`` is ALWAYS ``valuation["fair_value_range"]``.
    * ``fundamental_verdict`` is cross-checked against
      ``valuation["triangulation"]["signals"]["dcf"]`` via
      :func:`_reconcile_fundamental_verdict` -- an outright contradiction on
      the ucuz<->pahali axis is overridden (and logged).
    * ``scenario_returns``, ``entry_plan``, ``stop_adding``, and
      ``thesis_metric`` are ALWAYS computed by
      :mod:`sec_analyzer.interpret.planning` (METODOLOJI.md Sec.1 items
      4-7) and injected the same way as ``fair_value_range``/``confidence``
      -- no provider computes these itself.
    * The full ``valuation`` dict is attached under ``result["valuation"]``.
    * ``catalyst`` is filled in from ``catalyst["label"]`` when the provider
      left it empty/missing.
    * ``_provider``/``_model``/``_horizon``/``_weights`` are stamped.

    Args:
        result: The parsed result dict (may also be an ``{"error": ...}``
            dict from a failed JSON parse; this function still stamps it,
            matching the pre-two-phase behavior).
        provider: Resolved provider name, e.g. ``"ollama"``.
        model: Resolved model identifier/name.
        horizon: Investment horizon used for this call.
        technical: The merged indicators + verdict dict, or ``None``.
        catalyst: The catalyst dict, or ``None``.
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`, or
            ``None`` -- used by :func:`sec_analyzer.interpret.planning.
            select_thesis_metric`.
        red_flags: The list of ``{"code", "message", "detail"}`` dicts from
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None`` -- used by :func:`sec_analyzer.interpret.planning.
            compute_stop_adding`.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None`` -- its ``price`` (falling back to ``technical``'s) is
            used by :func:`sec_analyzer.interpret.planning.
            compute_scenario_returns`/:func:`~sec_analyzer.interpret.
            planning.compute_entry_plan`.

    Returns:
        ``result``, mutated in place and returned for convenience.
    """
    if technical and technical.get("verdict") is not None:
        detail = technical.get("verdict_detail") or ""
        result["technical_verdict"] = f"{technical['verdict']} ({detail})" if detail else technical["verdict"]
    else:
        result["technical_verdict"] = "VERİ YOK (fiyat verisi alınamadı)"

    triangulation = valuation.get("triangulation") or {}
    result["confidence"] = triangulation.get("confidence")
    result["fair_value_range"] = valuation.get("fair_value_range")

    dcf_signal = (triangulation.get("signals") or {}).get("dcf")
    result["fundamental_verdict"] = _reconcile_fundamental_verdict(
        result.get("fundamental_verdict"), dcf_signal, provider
    )

    # Model–market divergence override (governor, action="verdict"; see
    # _DIVERGENCE_VERDICT). Deterministic, numbers-driven, applied LAST so it
    # overrides any provider's verdict; confidence is already floored to DÜŞÜK
    # by the governor via triangulation.confidence above.
    divergence = triangulation.get("divergence") or {}
    divergence_active = divergence.get("action") == "verdict"
    if divergence_active:
        result["fundamental_verdict"] = _DIVERGENCE_VERDICT

    if not result.get("catalyst"):
        result["catalyst"] = catalyst.get("label") if catalyst else "bilinmiyor"
    # Surface the raw estimated-earnings ISO date (deterministic, from
    # estimate_next_earnings) so the report can compute a swing "N days to
    # earnings" proximity flag; independent of the free-text catalyst label.
    result["catalyst_estimate_date"] = catalyst.get("estimate_date") if catalyst else None

    price = (metrics or {}).get("price")
    if price is None:
        price = (technical or {}).get("price")

    # scenario_returns is read from valuation["fair_value_range"] by
    # reference, not result["fair_value_range"] -- both point at the same
    # dict, but this makes explicit that planning never mutates valuation.
    result["scenario_returns"] = planning.compute_scenario_returns(valuation.get("fair_value_range"), price)
    result["entry_plan"] = planning.compute_entry_plan(valuation, technical, price)
    result["stop_adding"] = planning.compute_stop_adding(
        valuation, technical, red_flags, result["entry_plan"], catalyst
    )
    result["thesis_metric"] = planning.select_thesis_metric(valuation.get("sector_type"), ratios, metrics)
    if divergence_active:
        # In a divergence the kill-switch is whether the market-priced
        # assumption (a growth collapse the model rejects) actually
        # materializes, so reframe the thesis metric's rationale around that
        # referee while keeping its computed name/value/trend intact.
        thesis_metric = dict(result["thesis_metric"] or {})
        thesis_metric["rationale"] = (
            "Model-piyasa ayrışması: hakem, piyasanın fiyatladığı büyüme yavaşlamasının gerçekleşip "
            "gerçekleşmemesidir. Bu metriğin önümüzdeki çeyreklerdeki seyri tezi doğrular ya da çürütür."
        )
        result["thesis_metric"] = thesis_metric

    result["valuation"] = valuation
    result["_provider"] = provider
    result["_model"] = model
    fundamental_weight, technical_weight = Config.HORIZON_WEIGHTS.get(
        horizon, Config.HORIZON_WEIGHTS["1y"]
    )
    result["_horizon"] = horizon
    result["_weights"] = {"fundamental": fundamental_weight, "technical": technical_weight}
    return result


def interpret_results(
    normalized: dict,
    ratios: List[dict],
    metrics: Optional[dict],
    technical: Optional[dict],
    red_flags: Optional[List[dict]],
    catalyst: Optional[dict],
    valuation: dict,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    horizon: str = "1y",
) -> dict:
    """Phase 2 of the two-phase valuation flow: comment on an
    already-computed ``valuation`` (SPEC.md Sec.12).

    Builds the phase-2 system prompt (:func:`_build_phase2_system_prompt`)
    and user payload (:func:`_build_phase2_user_payload`, which embeds the
    full ``valuation`` dict), asks the selected backend for commentary, and
    applies the fixed post-processing rules in
    :func:`_postprocess_phase2_result` (``technical_verdict``,
    ``confidence``, ``fair_value_range``, the ``fundamental_verdict``
    cross-check, the attached ``valuation`` dict, and provider/horizon
    stamping -- none of which any provider can override).

    This function is designed to never raise: any failure (unknown
    provider, missing ``anthropic`` package, missing API key, Ollama not
    running or missing the requested model, network/API error, or a
    response that isn't valid JSON) is caught and returned as
    ``{"error": ..., "summary": ...}`` so callers (in particular the CLI)
    can display a warning and continue rather than crashing.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None``.
        technical: The merged indicators + verdict dict from
            :mod:`sec_analyzer.technical`, or ``None``. Its
            ``verdict``/``verdict_detail`` always win for
            ``technical_verdict`` regardless of what the provider produced.
        red_flags: The list of ``{"code", "message", "detail"}`` dicts from
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``.
        catalyst: The ``{"estimate_date", "label", "based_on"}`` dict from
            :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
            ``None``.
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation`. Attached
            verbatim under the returned result's ``"valuation"`` key.
        provider: Which backend to use: ``"ollama"``, ``"anthropic"``, or
            ``"script"`` (deterministic, no-AI -- uses
            :func:`sec_analyzer.interpret.rule_based.commentary`, fully
            offline). Defaults to ``Config.ANALYZER_PROVIDER``.
        model: Model ID/name to use.
        api_key: Anthropic API key to use (anthropic provider only).
        host: Base URL of the Ollama server (ollama provider only).
        horizon: Investment horizon: ``"3m"``, ``"1y"``, or ``"5y"``.

    Returns:
        On success, a dict matching this module's unified phase-2 output
        schema (see the module docstring). On failure, a dict of the form
        ``{"error": <str>, "summary": <str>}``, optionally with a ``"raw"``
        key holding the unparsed model output.
    """
    resolved_provider = (provider or Config.ANALYZER_PROVIDER or "ollama").lower()
    valuation = valuation or {}

    if resolved_provider in _SCRIPT_PROVIDER_ALIASES:
        try:
            result = rule_based.commentary(
                valuation, metrics=metrics, technical=technical, red_flags=red_flags,
                catalyst=catalyst, horizon=horizon,
            )
        except Exception as exc:  # noqa: BLE001 - never let interpret_results() raise
            logger.exception("Unexpected error during script-based interpret_results()")
            return {
                "error": str(exc),
                "summary": "An unexpected error occurred while running the script-based analysis.",
                "_provider": "script",
            }
        return _postprocess_phase2_result(
            result, "script", "rule-based-v2", horizon, technical, catalyst, valuation,
            ratios=ratios, red_flags=red_flags, metrics=metrics,
        )

    system_prompt = _build_phase2_system_prompt(horizon)
    user_payload = _build_phase2_user_payload(normalized, ratios, metrics, technical, red_flags, catalyst, valuation)

    try:
        raw_text, resolved_model = _dispatch_llm_call(resolved_provider, system_prompt, user_payload, model, api_key, host)
    except OllamaError as exc:
        logger.error("Ollama phase-2 interpretation failed: %s", exc)
        return {
            "error": str(exc),
            "summary": "Local Ollama analysis is unavailable; see error for details.",
            "_provider": "ollama",
        }
    except ConfigError as exc:
        logger.error("Cannot run Anthropic phase-2 interpretation: %s", exc)
        return {
            "error": str(exc),
            "summary": "Anthropic API key is not configured; skipping analysis.",
            "_provider": "anthropic",
        }
    except ValueError as exc:
        logger.error(str(exc))
        return {
            "error": str(exc),
            "summary": "Set ANALYZER_PROVIDER to 'ollama', 'anthropic', or 'script'.",
            "_provider": resolved_provider,
        }
    except Exception as exc:  # noqa: BLE001 - covers RuntimeError (anthropic not installed),
        # anthropic.APIError, and anything else a provider transport can raise.
        logger.exception("Unexpected error during interpret_results() (%s)", resolved_provider)
        return {
            "error": str(exc),
            "summary": "An unexpected error occurred while running the analysis.",
            "_provider": resolved_provider,
        }

    result = _parse_model_json(raw_text)
    return _postprocess_phase2_result(
        result, _canonical_provider(resolved_provider), resolved_model, horizon, technical, catalyst, valuation,
        ratios=ratios, red_flags=red_flags, metrics=metrics,
    )


def interpret(
    normalized: dict,
    ratios: List[dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    horizon: str = "1y",
    metrics: Optional[dict] = None,
    technical: Optional[dict] = None,
    red_flags: Optional[List[dict]] = None,
    catalyst: Optional[dict] = None,
    valuation: Optional[dict] = None,
    submissions: Optional[dict] = None,
    price_df=None,
    as_of=None,
    fred_rate: Optional[dict] = None,
) -> dict:
    """Backward-compatible entry point orchestrating the full two-phase flow.

    Same signature existing callers (``sec_analyzer.cli``,
    ``sec_analyzer.web.app``) already use, plus two new optional keyword
    arguments:

    * If ``valuation`` is given (a precomputed dict from
      :func:`sec_analyzer.valuation.engine.run_valuation`), this function
      only runs phase 2 (:func:`interpret_results`) against it.
    * If ``valuation`` is ``None`` (the common case for existing callers),
      this function runs the full flow internally: resolve a sector
      (deterministically from ``submissions["sic"]`` via
      :func:`sec_analyzer.valuation.sector.classify_sector` when available,
      else falling back to phase 1's own sector-type guess) -> phase 1
      (:func:`propose_assumptions`) -> the deterministic engine
      (:func:`sec_analyzer.valuation.run_valuation`) -> phase 2
      (:func:`interpret_results`).

    Callers that don't yet have ``submissions``/a price-history DataFrame
    on hand (e.g. today's ``web.app``) still work -- SIC-based sector
    classification degrades to phase 1's guess, and price-history-dependent
    multiples degrade to ``None`` (see ``run_valuation``'s own degradation
    rules), never a crash.

    This function is designed to never raise: any unexpected failure (this
    wrapper's own orchestration, not phase 1/2's already-defensive internals)
    is caught and returned as ``{"error": ..., "summary": ...}``.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
        provider: Which backend to use: ``"ollama"``, ``"anthropic"``, or
            ``"script"`` (deterministic, no-AI). Defaults to
            ``Config.ANALYZER_PROVIDER``. Used for both phases.
        model: Model ID/name to use for both phases.
        api_key: Anthropic API key to use (anthropic provider only).
        host: Base URL of the Ollama server (ollama provider only).
        horizon: Investment horizon: ``"3m"``, ``"1y"``, or ``"5y"``.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None``.
        technical: The merged indicators + verdict dict from
            :mod:`sec_analyzer.technical`, or ``None``.
        red_flags: The list of ``{"code", "message", "detail"}`` dicts from
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``.
        catalyst: The ``{"estimate_date", "label", "based_on"}`` dict from
            :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
            ``None``.
        valuation: A precomputed valuation dict to skip straight to phase
            2, or ``None`` to run the full flow. Defaults to ``None``.
        submissions: The raw dict from
            :func:`sec_analyzer.fetch.companyfacts.get_submissions`
            (``sic``/``sicDescription`` used for sector classification and
            Damodaran sector-median matching), or ``None``. Ignored when
            ``valuation`` is already given.
        price_df: The DataFrame returned by
            :func:`sec_analyzer.fetch.prices.get_price_history`, or
            ``None`` -- multiples history degrades gracefully without it.
            Ignored when ``valuation`` is already given.

    Returns:
        On success, a dict matching this module's unified phase-2 output
        schema (see the module docstring). On failure, a dict of the form
        ``{"error": <str>, "summary": <str>, "_provider": <str>}``.
    """
    resolved_provider_for_errors = (provider or Config.ANALYZER_PROVIDER or "ollama").lower()
    try:
        normalized = normalized or {}
        ratios = ratios or []
        metrics = metrics or {}

        if valuation is not None:
            return interpret_results(
                normalized, ratios, metrics, technical, red_flags, catalyst, valuation,
                provider=provider, model=model, api_key=api_key, host=host, horizon=horizon,
            )

        sic = (submissions or {}).get("sic")
        sic_description = (submissions or {}).get("sicDescription")
        sector_hint = classify_sector(sic, normalized, metrics) if sic is not None else None

        # Firm-specific CAPM cost of equity (rf + betaL x ERP) from the local
        # Damodaran reference data, used as the deterministic base discount
        # rate in place of the flat sector-agnostic default. None (missing
        # beta/ERP/risk-free, or unmatched SIC) leaves the flat default in
        # place; the LLM path uses this only as its fallback base. Loaded
        # once into `sector_data` and reused for both the CAPM lookup and
        # the global risk-free fallback below, rather than reading the CSVs
        # twice.
        sector_data = damodaran.load_sector_data(Config.DAMODARAN_DIR, as_of=as_of, fred_rate=fred_rate)
        capm = compute_cost_of_equity(
            sector_data,
            sic_description,
            metrics,
            is_unprofitable=(sector_hint == "growth_unprofitable"),
        )
        # Global risk-free rate (erp.csv), independent of SIC/industry
        # matching -- used by the phase-1 fallback's terminal-growth anchor
        # when `capm` itself is None (e.g. the SIC didn't match any
        # Damodaran industry) so terminal growth isn't ALSO flattened to the
        # old constant on top of the flat discount rate.
        risk_free_pct = sector_data.get("risk_free") if sector_data else None

        phase1 = propose_assumptions(
            normalized, ratios, metrics, sector_hint=sector_hint,
            provider=provider, model=model, api_key=api_key, host=host, horizon=horizon,
            capm=capm, risk_free_pct=risk_free_pct,
        )
        assumptions = phase1["assumptions"]
        # If SIC is known, its deterministic classification (sector_hint)
        # always wins over the LLM's own guess (SPEC.md Sec.8); otherwise
        # fall back to whatever phase 1 proposed.
        sector_type = sector_hint or phase1.get("sector_type") or "mature"

        price = metrics.get("price")
        if price is None and technical:
            price = technical.get("price")

        valuation_result = run_valuation(
            normalized, ratios, metrics, price, price_df, assumptions, sector_type,
            sic_description=sic_description, hyper_growth_extras=phase1.get("hyper_growth_extras"),
            as_of=as_of, fred_rate=fred_rate,
        )

        return interpret_results(
            normalized, ratios, metrics, technical, red_flags, catalyst, valuation_result,
            provider=provider, model=model, api_key=api_key, host=host, horizon=horizon,
        )
    except Exception as exc:  # noqa: BLE001 - interpret() must never raise
        logger.exception("Unexpected error during interpret()")
        return {
            "error": str(exc),
            "summary": "An unexpected error occurred while running the analysis.",
            "_provider": resolved_provider_for_errors,
        }
