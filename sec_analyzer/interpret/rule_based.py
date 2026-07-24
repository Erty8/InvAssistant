"""Deterministic, script-based (no-AI) fundamental analysis.

This module is the "script" analyzer provider: an alternative to the LLM
backends in :mod:`sec_analyzer.interpret.analyzer` that produces the same
bear/base/bull output schema using only plain arithmetic over the normalized
SEC figures, derived ratios, and (optionally) the valuation metrics,
technical indicators, red flags, and earnings-catalyst estimate produced by
the rest of the pipeline -- no network access, no language model, and no
randomness. Given the same inputs, :func:`analyze` always returns the same
output.

Why this exists alongside the LLM providers:

* **No dependency, no cost, no privacy concern.** It requires neither
  Ollama nor the Claude Code CLI, needs no API key, and never sends financial
  data anywhere.
* **Fully auditable.** Every judgment it makes is expressed as an explicit,
  named check (see the ``score`` key) or an explicit formula with visible
  assumptions (see the scenario ``growth``/``discount_rate``/``note``
  fields), so a reader can see exactly why a company scored the way it did
  -- unlike an LLM's free-form reasoning.
* **A useful baseline.** It's a simple, transparent sanity check that can be
  compared against (or substituted for) the LLM providers' output.

The methodology is a fixed, ten-point checklist covering profitability,
margin trend, growth, liquidity, leverage, cash-flow quality, and
shareholder returns (see :func:`_build_checks`), plus a transparent
bear/base/bull fair-value band derived from a two-stage discounted
cash-flow sketch (see :func:`_fair_value_scenarios`), and a cyclicality read
based on the volatility of year-over-year revenue growth (see
:func:`_cyclical_risk`). It is a heuristic screen, not investment advice --
see the "not investment advice" note baked into every summary.

This module also implements the "script" provider's half of the two-phase
valuation flow (``sec_analyzer/valuation/SPEC.md`` Sec.12; see
:mod:`sec_analyzer.interpret.analyzer` for the full flow):

* :func:`default_assumptions` -- the deterministic phase-1 fallback used
  whenever an LLM proposal can't be trusted (the ``"script"`` provider, an
  unavailable/unparseable LLM, or a proposal that still fails
  ``valuation.sanity.validate_assumptions`` after one revision round).
* :func:`commentary` -- the deterministic phase-2 analog to
  :func:`analyze`: template-based Turkish commentary over an already-
  computed ``valuation`` dict (from ``valuation.engine.run_valuation``)
  rather than computing its own fair-value band.

These two are independent of :func:`analyze`/:func:`_fair_value_scenarios`
(the older single-phase screen, kept as-is for backward compatibility) --
they don't share scenario-construction code because they solve different
problems: :func:`analyze` derives its own bear/base/bull band from a
two-stage DCF sketch, while :func:`default_assumptions` only proposes
growth/discount-rate *assumptions* for the real ``valuation`` engine to turn
into numbers, and :func:`commentary` only narrates numbers that already
exist.

Design goals, matching the rest of ``sec_analyzer.interpret``:

* **Never raise.** Every code path -- including a filer with no usable
  us-gaap data at all (e.g. an IFRS/20-F filer) -- returns the documented
  schema with ``None``/"insufficient data" placeholders rather than an
  exception.
* **Same unified output schema as the LLM providers** (see
  :mod:`sec_analyzer.interpret.analyzer`'s module docstring for the full
  schema), so callers (CLI, web UI) can treat ``analyze()`` and
  ``interpret()`` interchangeably. The extra ``score`` key is additive;
  consumers that don't know about it can ignore it.
"""

import logging
import os
import statistics
from typing import Dict, List, Optional, Tuple

from sec_analyzer.config import Config
from sec_analyzer.normalize.normalizer import to_annual_series
from sec_analyzer.valuation import sanity
from sec_analyzer.valuation.sanity import clamp_assumptions, validate_assumptions

logger = logging.getLogger(__name__)

#: Values stamped into every result's ``_provider``/``_model`` fields,
#: mirroring how the LLM providers identify themselves in ``analyzer.py``.
_PROVIDER = "script"
_MODEL = "rule-based-v2"

#: Nominal look-back window (in fiscal years) for the growth-CAGR checks and
#: the growth anchor used in the fair-value scenarios. If exactly this many
#: years of history aren't available, the oldest year that *is* available
#: (within the window) is used instead -- see ``_windowed_cagr``.
_GROWTH_WINDOW_YEARS = 3

#: Minimum number of non-null YoY revenue growth observations required
#: before ``_cyclical_risk`` will attempt a stdev-based read at all.
_MIN_GROWTH_OBSERVATIONS_FOR_CYCLICAL = 3

#: Thresholds (stdev of YoY revenue growth, as a fraction) separating the
#: "low"/"moderate"/"high" cyclicality buckets.
_CYCLICAL_STDEV_LOW = 0.05
_CYCLICAL_STDEV_MODERATE = 0.15

#: Verdict tier boundaries, as a fraction of evaluable checks passed. Used
#: only for the internal ``score``-derived narrative tier (strong/adequate/
#: weak), not for the schema's ``fundamental_verdict`` (UCUZ/MAKUL/PAHALI),
#: which is driven by price vs. the base-scenario band instead.
_VERDICT_STRONG_PCT = 0.75
_VERDICT_ADEQUATE_PCT = 0.50

#: Growth anchor used in the fair-value scenarios is clamped to this range
#: (0% to 25%) regardless of the raw computed CAGR, to keep even the bull
#: scenario conservative for a company on a brief growth tear.
_GROWTH_ANCHOR_MIN = 0.0
_GROWTH_ANCHOR_MAX = 0.25

#: Per-scenario growth multiplier and discount rate. Growth for a scenario
#: is ``growth_anchor * multiplier`` (bull additionally clamped to
#: ``bull_growth_clamp_max``); the discount rate is fixed per scenario.
_SCENARIOS: Dict[str, dict] = {
    "bear": {"multiplier": 0.5, "discount_rate": 0.12},
    "base": {"multiplier": 1.0, "discount_rate": 0.10},
    "bull": {"multiplier": 1.5, "discount_rate": 0.09, "growth_clamp_max": 0.30},
}

#: Terminal (perpetuity) growth rate used past the 5-year explicit forecast
#: window in every scenario. Always strictly below every scenario's discount
#: rate above, so the terminal-value denominator is always positive.
_TERMINAL_GROWTH = 0.025

#: Number of years of explicit (non-terminal) cash flows projected.
_PROJECTION_YEARS = 5

#: Annual concepts pulled directly from the normalized facts. Everything
#: else the checklist needs (margins, ROE/ROA, current ratio, debt/equity,
#: FCF) is already computed per fiscal year in the ``ratios`` list, so this
#: module only needs to pull raw series for the handful of concepts the
#: ratios list doesn't cover.
_RAW_CONCEPTS = (
    "Revenue",
    "NetIncome",
    "OperatingCashFlow",
    "StockholdersEquity",
    "SharesOutstanding",
    "EPS",
    "DividendsPaid",
)


def _fy_label(fy: Optional[int]) -> str:
    """Render a fiscal year for use in a human-readable detail string."""
    return f"FY{fy}" if fy is not None else "the most recent fiscal year"


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def _check(name: str, passed: Optional[bool], detail: str) -> dict:
    """Build one checklist entry in the documented ``score.checks`` shape."""
    return {"name": name, "passed": passed, "detail": detail}


def _score(checks: List[dict]) -> Tuple[int, int]:
    """Tally ``(points, max_points)`` from a checklist.

    Checks with ``passed is None`` ("n/a", missing data) are excluded from
    both the numerator and the denominator -- they neither help nor hurt
    the score, they just aren't counted.
    """
    evaluable = [c for c in checks if c["passed"] is not None]
    points = sum(1 for c in evaluable if c["passed"])
    return points, len(evaluable)


def _verdict_label(points: int, max_points: int) -> str:
    """Turn a ``(points, max_points)`` tally into an internal strong/
    adequate/weak narrative tier, used only for the ``summary`` text. This is
    distinct from the schema's ``fundamental_verdict``, which is UCUZ/MAKUL/
    PAHALI based on price vs. the base fair-value band.
    """
    if max_points == 0:
        return "insufficient data (0/0 checks evaluable)"
    pct = points / max_points
    if pct >= _VERDICT_STRONG_PCT:
        tier = "strong"
    elif pct >= _VERDICT_ADEQUATE_PCT:
        tier = "adequate"
    else:
        tier = "weak"
    return f"{tier} ({points}/{max_points} checks passed)"


def _windowed_cagr(
    series: Dict[int, float], latest_fy: Optional[int], window: int = _GROWTH_WINDOW_YEARS
) -> Optional[Tuple[float, int, int]]:
    """Compute a CAGR from ``latest_fy`` back up to ``window`` fiscal years.

    Prefers the fiscal year exactly ``window`` years before ``latest_fy`` if
    it's present in ``series``; otherwise falls back to the oldest year
    available within that window (i.e. whatever history actually exists, up
    to ``window`` years). Requires both endpoints to be strictly positive --
    a CAGR computed across a loss-making year is not a meaningful growth
    rate -- and at least two distinct fiscal years of data.

    Returns:
        ``(cagr, oldest_fy, n_years)`` on success, or ``None`` if the CAGR
        isn't computable (missing latest value, no earlier year within the
        window, or a non-positive endpoint).
    """
    if latest_fy is None or latest_fy not in series:
        return None
    latest_val = series[latest_fy]
    target_fy = latest_fy - window

    if target_fy in series:
        oldest_fy = target_fy
    else:
        candidates = [fy for fy in series if target_fy <= fy < latest_fy]
        if not candidates:
            return None
        oldest_fy = min(candidates)

    oldest_val = series[oldest_fy]
    n_years = latest_fy - oldest_fy
    if n_years <= 0 or latest_val is None or oldest_val is None:
        return None
    if latest_val <= 0 or oldest_val <= 0:
        return None

    cagr = (latest_val / oldest_val) ** (1.0 / n_years) - 1.0
    return cagr, oldest_fy, n_years


def _resolve_latest_fy(ratios: List[dict], series_dicts: List[Dict[int, float]]) -> Optional[int]:
    """Determine the most recent fiscal year to evaluate.

    ``ratios`` is already sorted by fiscal year descending (see
    ``compute_ratios``), so its first row's ``fy`` is preferred. If
    ``ratios`` is empty (e.g. a filer with too little annual data to
    compute any ratio), fall back to the maximum fiscal year seen across
    any of the raw concept series.
    """
    if ratios:
        return ratios[0].get("fy")
    candidates: set = set()
    for series in series_dicts:
        candidates |= set(series)
    return max(candidates) if candidates else None


def _build_checks(
    latest_fy: Optional[int], series: Dict[str, Dict[int, float]], ratio_by_fy: Dict[int, dict]
) -> List[dict]:
    """Run the fixed ten-point deterministic checklist for ``latest_fy``.

    Every check is defensive: if a required figure is missing, the check's
    ``passed`` is ``None`` ("n/a due to missing data") rather than being
    counted as a failure. See the module docstring for the rationale.
    """
    fy_lbl = _fy_label(latest_fy)
    latest_ratios = ratio_by_fy.get(latest_fy, {}) if latest_fy is not None else {}
    checks: List[dict] = []

    # 1. Profitable: latest NetIncome > 0.
    ni = series["NetIncome"].get(latest_fy) if latest_fy is not None else None
    if ni is None:
        checks.append(_check("Profitable", None, f"NetIncome is unavailable for {fy_lbl}."))
    else:
        checks.append(
            _check(
                "Profitable",
                ni > 0,
                f"NetIncome ({fy_lbl}) = {ni:,.0f}.",
            )
        )

    # 2. Margin quality: latest net_margin >= 8%.
    net_margin = latest_ratios.get("net_margin")
    if net_margin is None:
        checks.append(_check("Margin quality", None, f"net_margin is not computable for {fy_lbl}."))
    else:
        checks.append(
            _check(
                "Margin quality",
                net_margin >= 0.08,
                f"Net margin ({fy_lbl}) = {net_margin:.1%} (threshold 8.0%).",
            )
        )

    # 3. Margin trend: net_margin not deteriorating >20% relative vs 3y
    #    earlier (or the oldest available year before latest_fy).
    margin_by_fy = {fy: r.get("net_margin") for fy, r in ratio_by_fy.items() if r.get("net_margin") is not None}
    if latest_fy not in margin_by_fy:
        checks.append(_check("Margin trend", None, f"net_margin is not computable for {fy_lbl}."))
    else:
        earlier_candidates = [fy for fy in margin_by_fy if fy < latest_fy]
        if not earlier_candidates:
            checks.append(
                _check("Margin trend", None, "Only one fiscal year of net-margin data is available.")
            )
        else:
            target_fy = latest_fy - 3
            earlier_fy = target_fy if target_fy in margin_by_fy else min(earlier_candidates)
            earlier_margin = margin_by_fy[earlier_fy]
            latest_margin = margin_by_fy[latest_fy]
            if earlier_margin == 0:
                checks.append(
                    _check(
                        "Margin trend",
                        None,
                        f"Net margin at FY{earlier_fy} was exactly 0%; relative change is undefined.",
                    )
                )
            else:
                rel_change = (latest_margin - earlier_margin) / abs(earlier_margin)
                checks.append(
                    _check(
                        "Margin trend",
                        rel_change > -0.20,
                        f"Net margin {fy_lbl} {latest_margin:.1%} vs FY{earlier_fy} "
                        f"{earlier_margin:.1%} (relative change {rel_change:+.1%}; "
                        f"threshold -20%).",
                    )
                )

    # 4. Revenue growth: multi-year revenue CAGR > 0.
    rev_cagr = _windowed_cagr(series["Revenue"], latest_fy)
    if rev_cagr is None:
        checks.append(
            _check(
                "Revenue growth",
                None,
                "Insufficient revenue history (need at least 2 fiscal years with "
                "positive values) to compute a growth CAGR.",
            )
        )
    else:
        cagr, oldest_fy, n_years = rev_cagr
        checks.append(
            _check(
                "Revenue growth",
                cagr > 0,
                f"Revenue CAGR FY{oldest_fy}->{fy_lbl} ({n_years}y) = {cagr:.1%}.",
            )
        )

    # 5. Earnings growth: multi-year net income CAGR > 0 (only if both
    #    endpoints are positive; _windowed_cagr already enforces that).
    ni_cagr = _windowed_cagr(series["NetIncome"], latest_fy)
    if ni_cagr is None:
        checks.append(
            _check(
                "Earnings growth",
                None,
                "Insufficient net-income history, or a loss in one of the two "
                "endpoints, so no earnings CAGR was computed.",
            )
        )
    else:
        cagr, oldest_fy, n_years = ni_cagr
        checks.append(
            _check(
                "Earnings growth",
                cagr > 0,
                f"Net income CAGR FY{oldest_fy}->{fy_lbl} ({n_years}y) = {cagr:.1%}.",
            )
        )

    # 6. Liquidity: latest current_ratio >= 1.0.
    current_ratio = latest_ratios.get("current_ratio")
    if current_ratio is None:
        checks.append(_check("Liquidity", None, f"current_ratio is not computable for {fy_lbl}."))
    else:
        checks.append(
            _check(
                "Liquidity",
                current_ratio >= 1.0,
                f"Current ratio ({fy_lbl}) = {current_ratio:.2f} (threshold 1.00).",
            )
        )

    # 7. Leverage: latest debt_to_equity <= 2.0, or an automatic fail if
    #    equity itself is non-positive.
    equity = series["StockholdersEquity"].get(latest_fy) if latest_fy is not None else None
    debt_to_equity = latest_ratios.get("debt_to_equity")
    if equity is not None and equity <= 0:
        checks.append(
            _check(
                "Leverage",
                False,
                f"StockholdersEquity ({fy_lbl}) = {equity:,.0f} (negative equity).",
            )
        )
    elif debt_to_equity is None:
        checks.append(
            _check(
                "Leverage",
                None,
                f"debt_to_equity is not computable for {fy_lbl}.",
            )
        )
    else:
        checks.append(
            _check(
                "Leverage",
                debt_to_equity <= 2.0,
                f"Debt-to-equity ({fy_lbl}) = {debt_to_equity:.2f} (threshold 2.00).",
            )
        )

    # 8. Earnings quality: OperatingCashFlow >= 0.8 x NetIncome.
    ocf = series["OperatingCashFlow"].get(latest_fy) if latest_fy is not None else None
    if ocf is None or ni is None:
        checks.append(
            _check(
                "Earnings quality",
                None,
                f"OperatingCashFlow and/or NetIncome unavailable for {fy_lbl}.",
            )
        )
    else:
        threshold = 0.8 * ni
        checks.append(
            _check(
                "Earnings quality",
                ocf >= threshold,
                f"OperatingCashFlow ({fy_lbl}) = {ocf:,.0f} vs 0.8x NetIncome = {threshold:,.0f}.",
            )
        )

    # 9. FCF positive: latest fcf > 0.
    fcf = latest_ratios.get("fcf")
    if fcf is None:
        checks.append(
            _check(
                "FCF positive",
                None,
                f"Free cash flow is not computable for {fy_lbl} (missing OperatingCashFlow or CapEx).",
            )
        )
    else:
        checks.append(_check("FCF positive", fcf > 0, f"Free cash flow ({fy_lbl}) = {fcf:,.0f}."))

    # 10. Shareholder returns: DividendsPaid > 0 in the latest FY. Counts
    #     only when the concept has *any* data -- not paying a dividend is
    #     not itself a red flag (many strong companies retain all earnings).
    dividends_series = series["DividendsPaid"]
    if not dividends_series:
        checks.append(
            _check(
                "Shareholder returns",
                None,
                "No dividend data reported; many strong companies retain all earnings.",
            )
        )
    else:
        dividend = dividends_series.get(latest_fy)
        if dividend is None:
            checks.append(
                _check(
                    "Shareholder returns",
                    None,
                    f"DividendsPaid has no value for {fy_lbl} specifically.",
                )
            )
        else:
            checks.append(
                _check(
                    "Shareholder returns",
                    dividend > 0,
                    f"Dividends paid ({fy_lbl}) = {dividend:,.0f}.",
                )
            )

    return checks


def _cyclical_risk(ratios: List[dict]) -> str:
    """Assess cyclicality from the volatility of YoY revenue growth.

    Uses the population-style sample standard deviation
    (``statistics.stdev``) of the available ``yoy_revenue_growth`` values.
    Requires at least :data:`_MIN_GROWTH_OBSERVATIONS_FOR_CYCLICAL`
    observations; with fewer, there simply isn't enough history to say
    anything about volatility.
    """
    growth_rows = [r for r in ratios if r.get("yoy_revenue_growth") is not None]
    if len(growth_rows) < _MIN_GROWTH_OBSERVATIONS_FOR_CYCLICAL:
        return (
            "Insufficient history to assess cyclicality (need at least "
            f"{_MIN_GROWTH_OBSERVATIONS_FOR_CYCLICAL} fiscal years of "
            "year-over-year revenue growth)."
        )

    growth_values = [r["yoy_revenue_growth"] for r in growth_rows]
    stdev = statistics.stdev(growth_values)
    if stdev < _CYCLICAL_STDEV_LOW:
        level = "low"
    elif stdev < _CYCLICAL_STDEV_MODERATE:
        level = "moderate"
    else:
        level = "high"

    negative_years = sorted((r["fy"] for r in growth_rows if r["yoy_revenue_growth"] < 0), reverse=True)
    text = (
        f"{level} cyclicality (stdev of YoY revenue growth = {stdev:.1%} "
        f"across {len(growth_values)} fiscal years)."
    )
    if negative_years:
        text += f" Revenue declined year-over-year in FY {', '.join(str(fy) for fy in negative_years)}."
    return text


def _growth_anchor(
    latest_fy: Optional[int], series: Dict[str, Dict[int, float]], metrics: Optional[dict]
) -> Tuple[float, str]:
    """Determine the growth-rate anchor ``g`` used by every fair-value scenario.

    Preference order: a locally-computed 3-year net-income CAGR (via
    :func:`_windowed_cagr`) if computable; otherwise a 3-year revenue CAGR,
    preferring ``metrics["revenue_cagr_3y"]`` when ``metrics`` was supplied
    (it uses a stricter "exact fiscal year" window than the local fallback);
    otherwise 0.0. The raw value is clamped to
    ``[_GROWTH_ANCHOR_MIN, _GROWTH_ANCHOR_MAX]`` so even the bull scenario
    stays conservative.

    Returns:
        ``(g, source_text)`` -- the clamped growth anchor and a
        human-readable description of where it came from, for the
        transparency notes embedded in each scenario.
    """
    fy_lbl = _fy_label(latest_fy)
    ni_cagr = _windowed_cagr(series["NetIncome"], latest_fy)
    if ni_cagr is not None:
        raw_g, oldest_fy, n_years = ni_cagr
        source = f"{n_years} yıllık net kâr CAGR (FY{oldest_fy}->{fy_lbl})"
    else:
        rev_cagr_3y = (metrics or {}).get("revenue_cagr_3y")
        if rev_cagr_3y is not None:
            raw_g = rev_cagr_3y
            source = "3 yıllık gelir CAGR (metrics)"
        else:
            rev_cagr = _windowed_cagr(series["Revenue"], latest_fy)
            if rev_cagr is not None:
                raw_g, oldest_fy, n_years = rev_cagr
                source = f"{n_years} yıllık gelir CAGR (FY{oldest_fy}->{fy_lbl})"
            else:
                raw_g = 0.0
                source = "hesaplanabilir büyüme geçmişi yok (varsayılan %0)"

    g = _clamp(raw_g, _GROWTH_ANCHOR_MIN, _GROWTH_ANCHOR_MAX)
    return g, source


def _fps_anchor(
    latest_fy: Optional[int], series: Dict[str, Dict[int, float]], metrics: Optional[dict]
) -> Tuple[Optional[float], Optional[str]]:
    """Determine the per-share cash-flow figure the DCF scenarios are built on.

    Prefers ``metrics["fcf_per_share"]`` if it's positive; otherwise falls
    back to the latest fiscal year's EPS if that's positive; otherwise
    ``None`` (no scenario can be computed).

    Returns:
        ``(fps, anchor_label)``, where ``anchor_label`` is ``"FCF/hisse"``
        or ``"EPS"``. Both are ``None`` if neither anchor is usable.
    """
    fcf_per_share = (metrics or {}).get("fcf_per_share")
    if fcf_per_share is not None and fcf_per_share > 0:
        return fcf_per_share, "FCF/hisse"

    eps = series["EPS"].get(latest_fy) if latest_fy is not None else None
    if eps is not None and eps > 0:
        return eps, "EPS"

    return None, None


def _two_stage_pv(fps: float, growth: float, discount_rate: float) -> float:
    """Two-stage discounted-cash-flow present value per share.

    Explicit stage: ``fps`` grown at ``growth`` for
    :data:`_PROJECTION_YEARS` years, discounted at ``discount_rate``.
    Terminal stage: a Gordon-growth perpetuity (grown at
    :data:`_TERMINAL_GROWTH`, which is always below every scenario's
    ``discount_rate`` by construction) on the final projected year's cash
    flow, discounted back from the end of the projection window.
    """
    pv = 0.0
    for t in range(1, _PROJECTION_YEARS + 1):
        pv += fps * (1 + growth) ** t / (1 + discount_rate) ** t

    final_year_cf = fps * (1 + growth) ** _PROJECTION_YEARS
    terminal_value = final_year_cf * (1 + _TERMINAL_GROWTH) / (discount_rate - _TERMINAL_GROWTH)
    pv += terminal_value / (1 + discount_rate) ** _PROJECTION_YEARS
    return pv


def _scenario(
    fps: Optional[float],
    anchor_label: Optional[str],
    g: float,
    g_source: str,
    multiplier: float,
    discount_rate: float,
    growth_clamp_max: Optional[float] = None,
) -> dict:
    """Build one bear/base/bull scenario dict.

    ``growth`` and ``discount_rate`` are always populated as human-readable
    strings -- even when ``lo``/``hi`` are null -- so every assumption stays
    visible ("cam kutu" transparency) regardless of whether a fair-value
    number could actually be computed.
    """
    growth = g * multiplier
    if growth_clamp_max is not None:
        growth = min(growth, growth_clamp_max)

    growth_str = f"%{growth * 100:.0f} büyüme"
    discount_rate_str = f"%{discount_rate * 100:.0f}"

    if fps is None:
        return {
            "lo": None,
            "hi": None,
            "growth": growth_str,
            "discount_rate": discount_rate_str,
            "note": (
                "FCF/hisse pozitif değil ve EPS de pozitif değil; per-share nakit "
                "akışı çapası bulunamadığından bu senaryo hesaplanamadı."
            ),
        }

    pv = _two_stage_pv(fps, growth, discount_rate)
    lo = round(0.9 * pv, 2)
    hi = round(1.1 * pv, 2)
    note = (
        f"{anchor_label} çapası ({fps:.2f}); büyüme kaynağı: {g_source} "
        f"(uygulanan büyüme {growth_str}); {_PROJECTION_YEARS} yıllık iki aşamalı "
        f"model + %{_TERMINAL_GROWTH * 100:.1f} terminal büyüme; iskonto oranı "
        f"{discount_rate_str}."
    )
    return {"lo": lo, "hi": hi, "growth": growth_str, "discount_rate": discount_rate_str, "note": note}


def _fair_value_scenarios(
    latest_fy: Optional[int], series: Dict[str, Dict[int, float]], metrics: Optional[dict]
) -> Dict[str, dict]:
    """Build the ``{"bear", "base", "bull"}`` fair-value scenario dict.

    See the module docstring and :func:`_scenario`/:func:`_two_stage_pv` for
    the full methodology. All figures are USD per share.
    """
    g, g_source = _growth_anchor(latest_fy, series, metrics)
    fps, anchor_label = _fps_anchor(latest_fy, series, metrics)

    return {
        name: _scenario(
            fps, anchor_label, g, g_source,
            params["multiplier"], params["discount_rate"],
            growth_clamp_max=params.get("growth_clamp_max"),
        )
        for name, params in _SCENARIOS.items()
    }


def _current_price(metrics: Optional[dict], technical: Optional[dict]) -> Optional[float]:
    """Resolve the current market price from ``metrics`` (preferred) or
    ``technical`` (fallback), whichever has a usable value."""
    if metrics and metrics.get("price") is not None:
        return metrics["price"]
    if technical and technical.get("price") is not None:
        return technical["price"]
    return None


def _fundamental_verdict(price: Optional[float], base_scenario: dict) -> Tuple[str, bool]:
    """Classify the current price against the base-scenario band.

    Returns:
        ``(verdict, price_or_band_missing)`` where ``verdict`` is one of
        ``"UCUZ"``, ``"MAKUL"``, ``"PAHALI"``, and ``price_or_band_missing``
        is ``True`` when the classification defaulted to "MAKUL" purely
        because the price or the base band was unavailable (used to phrase
        ``horizon_note`` accordingly).
    """
    lo, hi = base_scenario.get("lo"), base_scenario.get("hi")
    if price is None or lo is None or hi is None:
        return "MAKUL", True
    if price < lo:
        return "UCUZ", False
    if price <= hi:
        return "MAKUL", False
    return "PAHALI", False


def _profile_fit() -> dict:
    """Judge profile fit -- always ``"KISMEN"`` for the deterministic
    provider, since it cannot actually interpret free-text profile
    preferences the way an LLM can; the reason differs depending on whether
    ``PROFIL.md`` exists at all.
    """
    if not (Config.PROFIL_PATH and os.path.exists(Config.PROFIL_PATH)):
        return {
            "verdict": "KISMEN",
            "reason": "PROFIL.md bulunamadı; nötr profil varsayıldı (dosyayı oluşturmanız önerilir).",
        }
    return {
        "verdict": "KISMEN",
        "reason": "Deterministik modül profil metnini yorumlayamaz; ayrıntılı uyum için LLM provider kullanın.",
    }


def _horizon_note(horizon: str, red_flags: Optional[List[dict]], price_or_band_missing: bool) -> str:
    """One-sentence note on what the given horizon emphasizes, plus (for a
    5-year horizon) the result of the cyclical-trap check, plus a note when
    price/band data was missing for the fundamental_verdict classification.
    """
    weights = Config.HORIZON_WEIGHTS.get(horizon, Config.HORIZON_WEIGHTS["1y"])
    fundamental_pct, technical_pct = weights[0] * 100, weights[1] * 100

    if horizon == "3m":
        note = (
            f"3 aylık ufukta teknik ve momentum sinyalleri öncelikli (fundamental "
            f"%{fundamental_pct:.0f} / teknik %{technical_pct:.0f}); yaklaşan katalizör kritik önemdedir."
        )
    elif horizon == "5y":
        note = (
            f"5 yıllık ufukta fundamental sinyaller öncelikli (fundamental "
            f"%{fundamental_pct:.0f} / teknik %{technical_pct:.0f}); RSI gibi kısa vadeli "
            "göstergeler önemsizdir."
        )
        cyclical_flag = next(
            (f for f in (red_flags or []) if f.get("code") == "CYCLICAL_TRAP"), None
        )
        if cyclical_flag:
            note += f" Döngüsel tepe riski kontrolü tetiklendi: {cyclical_flag.get('message', '')}."
        else:
            note += " Döngüsel tepe riski kontrolü tetiklenmedi."
    else:
        note = (
            f"{horizon} ufkunda fundamental (%{fundamental_pct:.0f}) ve teknik "
            f"(%{technical_pct:.0f}) sinyaller dengeli şekilde değerlendirilir."
        )

    if price_or_band_missing:
        note += " Not: güncel fiyat ve/veya baz değer bandı eksik olduğu için fiyat-bant karşılaştırması yapılamadı."
    return note


def _key_risks(checks: List[dict], red_flags: Optional[List[dict]]) -> List[str]:
    """Failed checklist item names plus red-flag messages, capped at 5."""
    risks = [c["name"] for c in checks if c["passed"] is False]
    risks += [f.get("message") for f in (red_flags or []) if f.get("message")]
    return risks[:5]


def _red_flags_comment(red_flags: Optional[List[dict]]) -> str:
    """``"yok"`` if no red flags fired, otherwise their messages joined."""
    if not red_flags:
        return "yok"
    return "; ".join(f.get("message", "") for f in red_flags if f.get("message"))


def _catalyst_text(catalyst: Optional[dict]) -> str:
    """The catalyst's human-readable label, or ``"bilinmiyor"`` if none."""
    if catalyst and catalyst.get("label"):
        return catalyst["label"]
    return "bilinmiyor"


def _technical_verdict_text(technical: Optional[dict]) -> str:
    """Render ``technical``'s verdict/detail into the schema's
    ``technical_verdict`` string, matching the format
    :mod:`sec_analyzer.interpret.analyzer` uses when it overwrites this
    field for the LLM providers -- so a direct call to :func:`analyze`
    (bypassing ``interpret()``) still gets a sensible value here.
    """
    if technical and technical.get("verdict") is not None:
        detail = technical.get("verdict_detail") or ""
        return f"{technical['verdict']} ({detail})" if detail else technical["verdict"]
    return "VERİ YOK (fiyat verisi alınamadı)"


def _build_summary(
    entity_name: Optional[str],
    latest_fy: Optional[int],
    checks: List[dict],
    points: int,
    max_points: int,
    tier_label: str,
    cyclical: str,
    fair_value_range: Dict[str, dict],
    fundamental_verdict: str,
    horizon: str,
) -> str:
    """Assemble the plain-English ``summary`` paragraph from the findings."""
    name = entity_name or "This filer"

    if max_points == 0:
        return (
            f"{name} could not be screened: none of the deterministic checklist "
            "items had enough SEC data to evaluate. This is typical for filers "
            "reporting only under the 'ifrs-full' taxonomy (e.g. foreign private "
            "issuers on Form 20-F) or with too little annual history on file. "
            "Deterministic rule-based screen of SEC filings; educational use "
            "only, not investment advice."
        )

    fy_lbl = _fy_label(latest_fy)
    passed_names = [c["name"] for c in checks if c["passed"] is True]
    failed_names = [c["name"] for c in checks if c["passed"] is False]
    na_count = sum(1 for c in checks if c["passed"] is None)

    sentences = [
        f"{name} scores {points}/{max_points} on the deterministic checklist for {fy_lbl} ({tier_label})."
    ]
    if passed_names:
        sentences.append(f"Strengths flagged by the screen: {', '.join(passed_names)}.")
    if failed_names:
        sentences.append(f"Weak spots flagged by the screen: {', '.join(failed_names)}.")
    sentences.append(cyclical)

    base = fair_value_range.get("base", {})
    if base.get("lo") is not None:
        sentences.append(
            f"A base-scenario two-stage DCF puts fair value roughly between "
            f"${base['lo']:.2f} and ${base['hi']:.2f} per share, versus a current price "
            f"read as {fundamental_verdict} for a {horizon} horizon."
        )
    else:
        sentences.append(
            "A per-share fair-value estimate could not be computed (no positive FCF/share "
            "or EPS anchor was available); see the base scenario's note for why."
        )
    if na_count:
        sentences.append(f"{na_count} checklist item(s) could not be evaluated due to missing data.")
    sentences.append(
        "Deterministic rule-based screen of SEC filings; educational use only, not investment advice."
    )
    return " ".join(sentences)


def _error_result() -> dict:
    """The fixed schema-shaped result returned when :func:`_analyze` raises
    unexpectedly, matching the field set :func:`analyze` normally returns.
    """
    def _null_scenario(discount_rate: float) -> dict:
        return {
            "lo": None,
            "hi": None,
            "growth": "%0 büyüme",
            "discount_rate": f"%{discount_rate * 100:.0f}",
            "note": "An internal error prevented a fair-value estimate.",
        }

    return {
        "fair_value_range": {
            name: _null_scenario(params["discount_rate"]) for name, params in _SCENARIOS.items()
        },
        "fundamental_verdict": "MAKUL",
        "technical_verdict": "VERİ YOK (fiyat verisi alınamadı)",
        "profile_fit": _profile_fit(),
        "cyclical_risk": "insufficient history to assess cyclicality (an internal error occurred).",
        "horizon_note": "Bir iç hata oluştu; ufuk notu üretilemedi.",
        "key_risks": [],
        "red_flags_comment": "yok",
        "catalyst": "bilinmiyor",
        "summary": (
            "An internal error prevented the deterministic screen from completing. "
            "Deterministic rule-based screen of SEC filings; educational use only, "
            "not investment advice."
        ),
        "_provider": _PROVIDER,
        "_model": _MODEL,
        "score": {"points": 0, "max_points": 0, "checks": []},
    }


def analyze(
    normalized: dict,
    ratios: List[dict],
    metrics: Optional[dict] = None,
    technical: Optional[dict] = None,
    red_flags: Optional[List[dict]] = None,
    catalyst: Optional[dict] = None,
    horizon: str = "1y",
) -> dict:
    """Run the deterministic, script-based fundamental screen.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None``. Supplies the current price, ``fcf_per_share``, and
            ``revenue_cagr_3y`` used by the fair-value scenarios.
        technical: The merged indicators + verdict dict from
            :mod:`sec_analyzer.technical`, or ``None``. Supplies the
            ``technical_verdict`` text and a price fallback.
        red_flags: The list of ``{"code", "message", "detail"}`` dicts from
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``/``[]``.
        catalyst: The ``{"estimate_date", "label", "based_on"}`` dict from
            :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
            ``None``.
        horizon: Investment horizon: ``"3m"``, ``"1y"``, or ``"5y"``.
            Controls ``horizon_note`` wording and (indirectly, since the
            weights are informational here) which signals are emphasized.

    Returns:
        A dict matching the unified bear/base/bull schema documented in
        :mod:`sec_analyzer.interpret.analyzer` -- ``fair_value_range``,
        ``fundamental_verdict``, ``technical_verdict``, ``profile_fit``,
        ``cyclical_risk``, ``horizon_note``, ``key_risks``,
        ``red_flags_comment``, ``catalyst``, ``summary``, ``_provider``,
        ``_model`` -- plus an additional ``score`` key (``{"points",
        "max_points", "checks"}``) that makes the checklist auditable.
        Never raises: any unexpected internal error is caught and turned
        into an "insufficient data" result with the same schema.
    """
    try:
        return _analyze(normalized or {}, ratios or [], metrics, technical, red_flags, catalyst, horizon)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("rule_based.analyze() failed unexpectedly; returning an insufficient-data result.")
        return _error_result()


def _analyze(
    normalized: dict,
    ratios: List[dict],
    metrics: Optional[dict],
    technical: Optional[dict],
    red_flags: Optional[List[dict]],
    catalyst: Optional[dict],
    horizon: str,
) -> dict:
    """Do the actual work for :func:`analyze` (split out so the latter can
    wrap it in a single top-level try/except)."""
    entity_name = normalized.get("entity_name")

    series = {concept: to_annual_series(normalized, concept) for concept in _RAW_CONCEPTS}
    ratio_by_fy = {r["fy"]: r for r in ratios if r.get("fy") is not None}

    latest_fy = _resolve_latest_fy(ratios, list(series.values()))

    checks = _build_checks(latest_fy, series, ratio_by_fy)
    points, max_points = _score(checks)
    tier_label = _verdict_label(points, max_points)
    cyclical = _cyclical_risk(ratios)

    fair_value_range = _fair_value_scenarios(latest_fy, series, metrics)
    price = _current_price(metrics, technical)
    fundamental_verdict, price_or_band_missing = _fundamental_verdict(price, fair_value_range["base"])

    summary = _build_summary(
        entity_name, latest_fy, checks, points, max_points, tier_label,
        cyclical, fair_value_range, fundamental_verdict, horizon,
    )

    return {
        "fair_value_range": fair_value_range,
        "fundamental_verdict": fundamental_verdict,
        "technical_verdict": _technical_verdict_text(technical),
        "profile_fit": _profile_fit(),
        "cyclical_risk": cyclical,
        "horizon_note": _horizon_note(horizon, red_flags, price_or_band_missing),
        "key_risks": _key_risks(checks, red_flags),
        "red_flags_comment": _red_flags_comment(red_flags),
        "catalyst": _catalyst_text(catalyst),
        "summary": summary,
        "_provider": _PROVIDER,
        "_model": _MODEL,
        "score": {"points": points, "max_points": max_points, "checks": checks},
    }


# ---------------------------------------------------------------------------
# Two-phase valuation flow (SPEC.md Sec.12): the "script" provider's phase-1
# assumption fallback and phase-2 commentary. Independent of analyze()/
# _fair_value_scenarios() above -- see the module docstring.
# ---------------------------------------------------------------------------

#: Phase-1 fallback: flat growth used when no revenue CAGR is available at
#: all (neither 5y nor 3y).
_DEFAULT_GROWTH_FALLBACK = 0.04

#: Phase-1 fallback: the base-scenario growth anchor is always clamped to
#: this range, regardless of source, mirroring the sanity-check spirit
#: (conservative even when a raw CAGR is extreme).
_DEFAULT_GROWTH_CLAMP_MIN = -0.05
_DEFAULT_GROWTH_CLAMP_MAX = 0.25

#: Phase-1 fallback: bear/bull growth is the base growth anchor +/- this
#: many percentage points.
_DEFAULT_SCENARIO_GROWTH_DELTA = 0.05

#: Phase-1 fallback: terminal growth used ONLY when no risk-free rate is
#: available (see :func:`_terminal_growth_anchor`), identical across all
#: three scenarios.
_DEFAULT_TERMINAL_GROWTH = 0.025

#: Phase-1 fallback: base discount rate, raised for a currently-unprofitable
#: filer (higher risk premium), plus per-scenario deltas.
_DEFAULT_DISCOUNT_RATE_BASE = 0.10
_DEFAULT_DISCOUNT_RATE_BASE_UNPROFITABLE = 0.12
_DEFAULT_DISCOUNT_RATE_BEAR_DELTA = 0.02
_DEFAULT_DISCOUNT_RATE_BULL_DELTA = -0.01

#: Turkish sector labels used in the phase-1 fallback's ``story`` sentences.
_SECTOR_LABELS_TR = {
    "cyclical": "döngüsel sektör",
    "financial": "finansal sektör",
    "growth_unprofitable": "henüz kâr etmeyen büyüme şirketi",
    "reit": "GYO",
    "mature": "olgun sektör",
}

_SCENARIO_LABELS_TR = {"bear": "Kötümser (bear)", "base": "Temel (base)", "bull": "İyimser (bull)"}


def _default_growth_anchor(metrics: Optional[dict]) -> Tuple[float, str]:
    """Resolve :func:`default_assumptions`'s base-scenario growth anchor.

    Prefers ``metrics["revenue_cagr_5y"]``, then ``metrics["revenue_cagr_3y"]``,
    then a flat 4% fallback (SPEC Sec.12); clamped to
    ``[_DEFAULT_GROWTH_CLAMP_MIN, _DEFAULT_GROWTH_CLAMP_MAX]`` regardless of
    source. Returns ``(g, source_text)`` for the transparency notes embedded
    in each scenario's ``story``.
    """
    metrics = metrics or {}
    raw = metrics.get("revenue_cagr_5y")
    if raw is not None:
        source = f"5 yıllık gelir CAGR (%{raw * 100:.1f})"
    else:
        raw = metrics.get("revenue_cagr_3y")
        if raw is not None:
            source = f"3 yıllık gelir CAGR (%{raw * 100:.1f})"
        else:
            raw = _DEFAULT_GROWTH_FALLBACK
            source = f"gelir CAGR verisi yok, varsayılan %{_DEFAULT_GROWTH_FALLBACK * 100:.0f}"
    return _clamp(raw, _DEFAULT_GROWTH_CLAMP_MIN, _DEFAULT_GROWTH_CLAMP_MAX), source


def _terminal_growth_anchor(
    capm: Optional[dict], risk_free_pct: Optional[float] = None
) -> Tuple[float, bool]:
    """Resolve the single shared terminal-growth rule: ``min(risk_free, 4%)``.

    Damodaran's practical rule of thumb is that a stable perpetuity growth
    rate should never exceed the risk-free rate (roughly nominal long-run
    GDP growth) -- see :data:`sec_analyzer.valuation.sanity._TERMINAL_GROWTH_MAX`
    for the 4% ceiling that still applies on top of it. This replaces the old
    flat 2.5% constant used for every scenario regardless of macro
    conditions, and deliberately does NOT differentiate by cohort: a
    hyper-grower that actually reaches its steady state is, by definition, a
    mature company at that point, so its terminal growth must not be set
    LOWER than a mature firm's just because it started out risky. That risk
    is already priced into the discount rate and the scenario probabilities
    -- cutting the terminal growth a third time on top of those would be a
    layered penalty for the same risk. (See
    :func:`sec_analyzer.valuation.engine._build_hyper_growth` for the other
    half of this rule, applied to the hyper-grower revenue-first DCF, which
    doesn't go through this function since it has no ``assumptions`` dict.)

    Resolution order (first numeric hit wins):

    1. ``capm["risk_free"]``, when ``capm`` is present and carries a numeric
       (non-bool) value.
    2. ``risk_free_pct`` -- the GLOBAL risk-free rate (e.g. straight off
       Damodaran's ``erp.csv``, independent of any SIC/industry matching),
       used when ``capm`` is absent or lacks a numeric ``risk_free``. This
       covers filers whose SIC doesn't match any Damodaran industry (so
       :func:`sec_analyzer.valuation.capm.compute_cost_of_equity` bails to
       ``None``) without also flattening their terminal growth to the old
       constant on top of the flat discount rate.
    3. :data:`_DEFAULT_TERMINAL_GROWTH`, when neither of the above yields a
       usable number.

    Args:
        capm: The optional :func:`sec_analyzer.valuation.capm.compute_cost_of_equity`
            result (carries ``risk_free`` as a PERCENTAGE number, e.g.
            ``4.20`` for 4.2%), or ``None``.
        risk_free_pct: The global risk-free rate as a PERCENTAGE number
            (e.g. ``4.20`` for 4.2%), used only as the step-2 fallback above.
            ``None`` if unavailable.

    Returns:
        ``(terminal_growth, from_risk_free)``: the resolved decimal
        fraction, and whether it was derived from a risk-free rate (``capm``'s
        own or the ``risk_free_pct`` fallback -- both count as ``True``) or
        fell back to :data:`_DEFAULT_TERMINAL_GROWTH` (``False``, when
        neither source carries a numeric value). Never raises.
    """
    risk_free_source = capm.get("risk_free") if capm else None
    if not (isinstance(risk_free_source, (int, float)) and not isinstance(risk_free_source, bool)):
        risk_free_source = risk_free_pct
    if isinstance(risk_free_source, (int, float)) and not isinstance(risk_free_source, bool):
        return min(risk_free_source / 100.0, sanity._TERMINAL_GROWTH_MAX), True
    return _DEFAULT_TERMINAL_GROWTH, False


def _default_story(
    scenario: str,
    growth: float,
    discount_rate: float,
    growth_source: str,
    sector_type: Optional[str],
    capm: Optional[dict] = None,
    terminal_growth: float = _DEFAULT_TERMINAL_GROWTH,
    terminal_from_risk_free: bool = False,
) -> str:
    """Build the transparent ("cam kutu"), Turkish ``story`` sentence for one
    :func:`default_assumptions` scenario, naming the inputs used.

    When ``capm`` (the :func:`sec_analyzer.valuation.capm.compute_cost_of_equity`
    result) is present, the discount-rate clause reports the CAPM derivation
    (rf + βL × ERP) rather than the old flat sector-classification default: the
    ``base`` scenario shows the full formula, ``bear``/``bull`` reference the
    CAPM base ± their scenario margin.

    ``terminal_growth``/``terminal_from_risk_free`` (see
    :func:`_terminal_growth_anchor`) add a clause naming whether the terminal
    growth used is tied to the risk-free rate (capped at 4%) or fell back to
    the flat :data:`_DEFAULT_TERMINAL_GROWTH` because no risk-free data was
    available.
    """
    scenario_label = _SCENARIO_LABELS_TR[scenario]
    capm_rate = capm.get("rate") if capm else None
    if isinstance(capm_rate, (int, float)) and not isinstance(capm_rate, bool):
        if scenario == "base" and capm.get("detail"):
            rate_clause = f"iskonto oranı %{discount_rate * 100:.1f} ({capm['detail']})"
        else:
            rate_clause = (
                f"iskonto oranı %{discount_rate * 100:.1f} "
                f"(CAPM tabanı %{capm_rate * 100:.1f} ± senaryo marjı)"
            )
    else:
        sector_label = _SECTOR_LABELS_TR.get(sector_type, "belirlenmemiş sektör")
        rate_clause = (
            f"iskonto oranı %{discount_rate * 100:.1f} "
            f"({sector_label} sınıflandırmasına göre)"
        )
    if terminal_from_risk_free:
        terminal_clause = (
            f"terminal büyüme %{terminal_growth * 100:.1f} (risksiz getiri oranına bağlı, üst sınır %4)"
        )
    else:
        terminal_clause = (
            f"terminal büyüme %{terminal_growth * 100:.1f} (risksiz getiri verisi yok, sabit varsayılan)"
        )
    return (
        f"{scenario_label} senaryo: deterministik varsayılan -- büyüme kaynağı {growth_source}, "
        f"bu senaryoda uygulanan büyüme %{growth * 100:.1f}, {rate_clause}, {terminal_clause}; "
        "LLM kullanılamadığı veya önerisi doğrulama sınırlarını aşıp geçersiz kaldığı için otomatik "
        "varsayılan devreye girdi."
    )


def _minimal_safe_assumptions() -> dict:
    """A hardcoded assumption set that trivially passes ``validate_assumptions``.

    Used only if :func:`default_assumptions` itself hits an unexpected
    internal error (or, defensively, produces an invalid set despite the
    formula above being designed to never do so) -- the absolute last
    resort before the valuation engine would otherwise receive nothing.
    """
    note = "İç bir hata nedeniyle en muhafazakar sabit varsayımlar kullanıldı."
    return {
        "bear": {"growth_5y": -0.01, "terminal_growth": 0.025, "discount_rate": 0.12, "story": note},
        "base": {"growth_5y": 0.04, "terminal_growth": 0.025, "discount_rate": 0.10, "story": note},
        "bull": {"growth_5y": 0.09, "terminal_growth": 0.025, "discount_rate": 0.09, "story": note},
    }


def default_assumptions(
    metrics: Optional[dict],
    sector_type: Optional[str] = None,
    capm: Optional[dict] = None,
    risk_free_pct: Optional[float] = None,
) -> dict:
    """Deterministic phase-1 fallback assumptions (SPEC.md Sec.12).

    Used by :func:`sec_analyzer.interpret.analyzer.propose_assumptions`
    whenever the LLM proposal can't be trusted: the ``"script"`` provider,
    an unavailable/unparseable LLM, or an LLM proposal that still violates
    ``valuation.sanity.validate_assumptions`` after one revision round.
    Deterministic given the same ``metrics``/``sector_type``/``capm`` --
    always passes ``validate_assumptions`` by construction (base growth is
    clamped, every discount rate stays comfortably above both its
    sector-specific floor and the terminal growth rate via
    ``clamp_assumptions``'s ERP-spread guard, and ``growth_5y`` never
    approaches the 40% hard ceiling).

    Args:
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics` (uses
            ``revenue_cagr_5y``, falling back to ``revenue_cagr_3y``, then a
            flat 4%), or ``None``.
        sector_type: One of ``valuation.sector.classify_sector``'s buckets.
            ``"growth_unprofitable"`` raises the base discount rate from 10%
            to 12% (and every scenario's floor with it); anything else
            (including ``None``) uses the standard 10% floor.
        capm: The optional
            :func:`sec_analyzer.valuation.capm.compute_cost_of_equity` result.
            When present (and carrying a numeric ``rate``), its firm-specific
            CAPM cost of equity REPLACES the flat sector-agnostic base
            discount rate; the bear/bull scenarios keep their usual deltas
            around that CAPM base. ``None`` (no Damodaran beta/ERP/risk-free)
            preserves the historical flat 10%/12% default. Its ``risk_free``
            field (see :func:`_terminal_growth_anchor`) also now drives
            ``terminal_growth`` for all three scenarios: ``min(risk_free,
            4%)``, falling back to the flat 2.5% constant when absent.
        risk_free_pct: The global risk-free rate (a PERCENTAGE number, e.g.
            ``4.20`` for 4.2%), used by :func:`_terminal_growth_anchor` as
            the ``terminal_growth`` fallback ONLY when ``capm`` is absent or
            lacks a numeric ``risk_free`` of its own -- e.g. a filer whose
            SIC doesn't match any Damodaran industry, so ``capm`` is
            ``None`` even though the global rate (independent of SIC
            matching) is still available. ``None`` preserves the flat 2.5%
            constant, matching historical behavior.

    Returns:
        ``{"bear": {...}, "base": {...}, "bull": {...}}`` (SPEC.md Sec.2
        shape) -- ``growth_5y``/``terminal_growth``/``discount_rate`` as
        decimal fractions, plus a Turkish ``story`` naming the inputs used.
        Never raises.
    """
    try:
        return _default_assumptions(metrics or {}, sector_type, capm, risk_free_pct)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("default_assumptions() failed unexpectedly; returning a minimal-safe fallback.")
        return _minimal_safe_assumptions()


def _default_assumptions(
    metrics: dict,
    sector_type: Optional[str],
    capm: Optional[dict] = None,
    risk_free_pct: Optional[float] = None,
) -> dict:
    base_growth, growth_source = _default_growth_anchor(metrics)
    bear_growth = base_growth - _DEFAULT_SCENARIO_GROWTH_DELTA
    bull_growth = base_growth + _DEFAULT_SCENARIO_GROWTH_DELTA

    # WP2: one shared terminal-growth rule for every scenario -- see
    # _terminal_growth_anchor's docstring for the rationale (no cohort
    # differentiation; risk is priced in discount_rate/probabilities, not
    # the terminal growth rate).
    terminal_growth, terminal_from_risk_free = _terminal_growth_anchor(capm, risk_free_pct)

    is_unprofitable = sector_type == "growth_unprofitable"
    # CAPM firm-specific cost of equity (rf + βL x ERP) takes precedence over
    # the flat sector-agnostic default when the Damodaran reference data made
    # it computable; otherwise fall back to the historical constant. Either
    # way, sanity.clamp_assumptions (inside run_valuation) floors and ERP-
    # spread-guards every per-scenario rate downstream.
    capm_rate = capm.get("rate") if capm else None
    if isinstance(capm_rate, (int, float)) and not isinstance(capm_rate, bool):
        base_dr = capm_rate
    else:
        base_dr = _DEFAULT_DISCOUNT_RATE_BASE_UNPROFITABLE if is_unprofitable else _DEFAULT_DISCOUNT_RATE_BASE
    bear_dr = base_dr + _DEFAULT_DISCOUNT_RATE_BEAR_DELTA
    bull_dr = base_dr + _DEFAULT_DISCOUNT_RATE_BULL_DELTA

    scenario_inputs = {"bear": (bear_growth, bear_dr), "base": (base_growth, base_dr), "bull": (bull_growth, bull_dr)}

    assumptions = {
        name: {
            "growth_5y": growth,
            "terminal_growth": terminal_growth,
            "discount_rate": dr,
            "story": _default_story(
                name, growth, dr, growth_source, sector_type, capm,
                terminal_growth=terminal_growth, terminal_from_risk_free=terminal_from_risk_free,
            ),
        }
        for name, (growth, dr) in scenario_inputs.items()
    }

    # A firm-specific CAPM base can sit low enough (low-beta sectors) that the
    # bull scenario's -1pp delta would dip below the discount-rate floor. Apply
    # the same clamp the engine applies downstream so this deterministic set
    # still passes validate_assumptions and CAPM survives, rather than bailing
    # to the flat minimal-safe fallback. Idempotent (no-op with no CAPM, where
    # the constant deltas are already in range).
    assumptions, _ = clamp_assumptions(assumptions, is_unprofitable=is_unprofitable)

    violations = validate_assumptions(assumptions, is_unprofitable=is_unprofitable)
    if violations:
        # Should be unreachable by construction -- fall back to the
        # hardcoded minimal-safe set rather than ever handing back
        # something the engine would just reject anyway.
        logger.error(
            "default_assumptions() produced invalid assumptions (%s); using the minimal-safe fallback.",
            violations,
        )
        return _minimal_safe_assumptions()
    return assumptions


#: Map from a triangulation direction signal to the schema's verdict string.
#: "veri_yok" deliberately has no entry -- there is nothing to map it to.
_DCF_SIGNAL_TO_VERDICT = {"ucuz": "UCUZ", "makul": "MAKUL", "pahali": "PAHALI"}


def _fundamental_verdict_from_valuation(valuation: dict) -> str:
    """Read the DCF (or P/B x ROE) triangulation signal straight off
    ``valuation`` rather than re-deriving it -- this keeps the "script"
    provider's ``fundamental_verdict`` trivially consistent with the
    code-enforced cross-check ``interpret_results`` applies to every
    provider, so it's never overridden for its own output."""
    signal = ((valuation.get("triangulation") or {}).get("signals") or {}).get("dcf")
    return _DCF_SIGNAL_TO_VERDICT.get(signal, "MAKUL")


#: Reverse-DCF comment margin (SPEC.md Sec.6), matching
#: ``valuation.triangulate._REVERSE_DCF_MARGIN``.
_REVERSE_DCF_COMMENT_MARGIN = 0.03


def _reverse_dcf_comment(valuation: dict) -> str:
    """Template-based Turkish reverse-DCF comment, built from
    ``valuation["reverse_dcf"]`` (SPEC.md Sec.6)."""
    reverse = valuation.get("reverse_dcf") or {}
    implied = reverse.get("implied_growth")
    realized = reverse.get("realized_cagr_5y")
    label = reverse.get("realized_label") or "geçmiş"

    if implied is None:
        return "Ters DCF hesaplanamadı; fiyatın ima ettiği büyüme oranı belirlenemedi."

    implied_pct = f"%{implied * 100:.1f}"
    if realized is None:
        return (
            f"Fiyat, 10 yıllık ufukta {implied_pct} büyüme ima ediyor; karşılaştırma için "
            "gerçekleşen gelir büyümesi verisi yok."
        )

    realized_pct = f"%{realized * 100:.1f}"
    diff = implied - realized
    if diff > _REVERSE_DCF_COMMENT_MARGIN:
        judgment = "piyasa, şirketin geçmiş performansından daha hızlı bir büyümeyi fiyatlıyor -- pahalılık sinyali."
    elif diff < -_REVERSE_DCF_COMMENT_MARGIN:
        judgment = "piyasa, şirketin geçmiş performansından daha kötümser bir senaryo fiyatlıyor -- ucuzluk sinyali."
    else:
        judgment = "fiyat, gerçekleşen büyüme trendiyle makul ölçüde uyumlu."

    return f"Fiyat, 10 yıllık ufukta {implied_pct} büyüme ima ediyor (gerçekleşen {label}: {realized_pct}); {judgment}"


def _cyclical_risk_from_valuation(valuation: dict, red_flags: Optional[List[dict]]) -> str:
    """Template-based cyclicality read from ``valuation["sector_type"]`` plus
    any ``CYCLICAL_TRAP`` red flag -- the phase-2 analog of :func:`_cyclical_risk`,
    which instead reads the YoY-revenue-growth stdev directly (unavailable
    to :func:`commentary`, which only receives ``valuation``, not ``ratios``)."""
    sector_type = valuation.get("sector_type")

    if sector_type == "cyclical":
        text = (
            "Şirket döngüsel bir sektörde sınıflandırıldı; standart DCF'e ek olarak normalize "
            "edilmiş kazanç senaryosu (tüm yılların FCF marjı medyanı) da hesaplandı."
        )
    elif sector_type == "growth_unprofitable":
        text = "Şirket henüz kâr etmiyor; büyüme senaryoları ve ters DCF, P/E çarpanlarından daha belirleyici."
    elif sector_type == "financial":
        text = "Finansal sınıflandırma nedeniyle döngüsellik P/B x ROE çapası üzerinden değerlendirildi."
    elif sector_type == "reit":
        text = (
            "GYO sınıflandırması nedeniyle döngüsellik FFO tabanlı değerleme "
            "(net kâr + amortisman) üzerinden değerlendirildi."
        )
    else:
        text = "Olgun sektör sınıflandırması altında döngüsellik riski sınırlı kabul edildi."

    cyclical_flag = next((f for f in (red_flags or []) if f.get("code") == "CYCLICAL_TRAP"), None)
    if cyclical_flag:
        text += f" Döngüsel tepe riski (cyclical trap) bayrağı tetiklendi: {cyclical_flag.get('message', '')}."
    return text


def _horizon_note_from_valuation(horizon: str, valuation: dict, price_or_band_missing: bool) -> str:
    """Phase-2 analog of :func:`_horizon_note`, additionally surfacing
    ``valuation["sensitivity"]["high_uncertainty"]`` (SPEC.md Sec.5)."""
    weights = Config.HORIZON_WEIGHTS.get(horizon, Config.HORIZON_WEIGHTS["1y"])
    fundamental_pct, technical_pct = weights[0] * 100, weights[1] * 100

    if horizon == "3m":
        note = (
            f"3 aylık ufukta teknik ve momentum sinyalleri öncelikli (fundamental "
            f"%{fundamental_pct:.0f} / teknik %{technical_pct:.0f})."
        )
    elif horizon == "5y":
        note = (
            f"5 yıllık ufukta fundamental sinyaller öncelikli (fundamental "
            f"%{fundamental_pct:.0f} / teknik %{technical_pct:.0f})."
        )
    else:
        note = (
            f"{horizon} ufkunda fundamental (%{fundamental_pct:.0f}) ve teknik "
            f"(%{technical_pct:.0f}) sinyaller dengeli şekilde değerlendirilir."
        )

    if (valuation.get("sensitivity") or {}).get("high_uncertainty"):
        note += " Duyarlılık matrisi yüksek belirsizlik gösteriyor (bant genişliği baz hücrenin %60'ından fazla)."
    if price_or_band_missing:
        note += " Not: güncel fiyat ve/veya adil değer bandı eksik olduğu için karşılaştırma yapılamadı."
    return note


def _key_risks_from_valuation(valuation: dict, red_flags: Optional[List[dict]]) -> List[str]:
    """Red-flag messages plus ``valuation["notes"]`` (Turkish engine
    warnings), capped at 5 -- the phase-2 analog of :func:`_key_risks`,
    which instead lists failed checklist item names (unavailable here since
    :func:`commentary` doesn't run the checklist)."""
    risks = [f.get("message") for f in (red_flags or []) if f.get("message")]
    risks += [note for note in (valuation.get("notes") or []) if note]
    return risks[:5]


def _commentary_error_result() -> dict:
    """The fixed schema-shaped result returned when :func:`commentary`'s
    implementation raises unexpectedly."""
    return {
        "fundamental_verdict": "MAKUL",
        "profile_fit": _profile_fit(),
        "reverse_dcf_comment": "Ters DCF yorumu üretilemedi (iç hata).",
        "cyclical_risk": "Döngüsellik değerlendirilemedi (iç hata).",
        "horizon_note": "Bir iç hata oluştu; ufuk notu üretilemedi.",
        "key_risks": [],
        "red_flags_comment": "yok",
        "catalyst": "bilinmiyor",
        "summary": (
            "Bir iç hata nedeniyle deterministik yorum tamamlanamadı. Eğitim amaçlı bir "
            "değerlendirmedir, yatırım tavsiyesi değildir."
        ),
    }


def commentary(
    valuation: Optional[dict],
    metrics: Optional[dict] = None,
    technical: Optional[dict] = None,
    red_flags: Optional[List[dict]] = None,
    catalyst: Optional[dict] = None,
    horizon: str = "1y",
) -> dict:
    """Deterministic, template-based phase-2 commentary (SPEC.md Sec.12.2).

    The "script" provider's phase-2 analog to :func:`default_assumptions`:
    given an already-computed ``valuation`` dict (from
    :func:`sec_analyzer.valuation.engine.run_valuation`), produces the same
    commentary fields an LLM would for
    :func:`sec_analyzer.interpret.analyzer.interpret_results`'s output
    contract, using only template sentences over ``valuation``'s own
    figures -- no network access, no LLM, no randomness. The caller
    (``interpret_results``) still applies the same code-enforced
    post-processing (``technical_verdict``, ``confidence``,
    ``fair_value_range`` injection, ``valuation`` attachment, provider
    stamping) on top of whatever this function returns, exactly as it does
    for the LLM providers.

    Args:
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation`, or ``None``
            (treated as ``{}``).
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None``. Only used (via :func:`_current_price`) to detect
            whether a price/base-band comparison was possible for
            ``horizon_note``.
        technical: The merged indicators + verdict dict from
            :mod:`sec_analyzer.technical`, or ``None`` (same role as
            ``metrics`` -- a price fallback).
        red_flags: The list of ``{"code", "message", "detail"}`` dicts from
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``/``[]``.
        catalyst: The ``{"estimate_date", "label", "based_on"}`` dict from
            :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
            ``None``.
        horizon: Investment horizon: ``"3m"``, ``"1y"``, or ``"5y"``.

    Returns:
        ``{"fundamental_verdict", "profile_fit", "reverse_dcf_comment",
        "cyclical_risk", "horizon_note", "key_risks", "red_flags_comment",
        "catalyst", "summary"}`` -- matches the phase-2 LLM output contract
        exactly (deliberately no ``fair_value_range``/``technical_verdict``/
        ``valuation``/``confidence`` keys; ``interpret_results`` always adds
        those). Never raises.
    """
    try:
        return _commentary(valuation or {}, metrics, technical, red_flags, catalyst, horizon)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("commentary() failed unexpectedly; returning an insufficient-data result.")
        return _commentary_error_result()


def _commentary(
    valuation: dict,
    metrics: Optional[dict],
    technical: Optional[dict],
    red_flags: Optional[List[dict]],
    catalyst: Optional[dict],
    horizon: str,
) -> dict:
    fair_value_range = valuation.get("fair_value_range") or {}
    base = fair_value_range.get("base") or {}
    price = _current_price(metrics, technical)
    price_or_band_missing = price is None or base.get("lo") is None or base.get("hi") is None

    fundamental_verdict = _fundamental_verdict_from_valuation(valuation)
    confidence = (valuation.get("triangulation") or {}).get("confidence") or "DÜŞÜK"
    sector_type = valuation.get("sector_type")

    if base.get("lo") is not None and base.get("hi") is not None:
        band_clause = f"baz senaryoda ${base['lo']:.2f}-{base['hi']:.2f} aralığını işaret ediyor"
    else:
        band_clause = "baz senaryoda adil değer aralığı hesaplayamadı"

    summary = (
        f"Değerleme motoru {band_clause}; üçgenleme güveni {confidence}. "
        f"Fiyat okuması: {fundamental_verdict}. Sektör sınıflandırması: {sector_type or 'bilinmiyor'}. "
        "Deterministik, kural tabanlı yorum; eğitim amaçlıdır, yatırım tavsiyesi değildir."
    )

    return {
        "fundamental_verdict": fundamental_verdict,
        "profile_fit": _profile_fit(),
        "reverse_dcf_comment": _reverse_dcf_comment(valuation),
        "cyclical_risk": _cyclical_risk_from_valuation(valuation, red_flags),
        "horizon_note": _horizon_note_from_valuation(horizon, valuation, price_or_band_missing),
        "key_risks": _key_risks_from_valuation(valuation, red_flags),
        "red_flags_comment": _red_flags_comment(red_flags),
        "catalyst": _catalyst_text(catalyst),
        "summary": summary,
    }
