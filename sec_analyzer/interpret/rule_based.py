"""Deterministic, script-based (no-AI) fundamental analysis.

This module is the "script" analyzer provider: an alternative to the LLM
backends in :mod:`sec_analyzer.interpret.analyzer` that produces the exact
same output schema using only plain arithmetic over the normalized SEC
figures and derived ratios -- no network access, no language model, and no
randomness. Given the same inputs, :func:`analyze` always returns the same
output.

Why this exists alongside the LLM providers:

* **No dependency, no cost, no privacy concern.** It requires neither
  Ollama nor an Anthropic API key, and never sends financial data anywhere.
* **Fully auditable.** Every judgment it makes is expressed as an explicit,
  named check (see the ``score`` key) with the numbers that drove it, so a
  reader can see exactly why a company scored the way it did -- unlike an
  LLM's free-form reasoning.
* **A useful baseline.** It's a simple, transparent sanity check that can be
  compared against (or substituted for) the LLM providers' output.

The methodology is a fixed, ten-point checklist covering profitability,
margin trend, growth, liquidity, leverage, cash-flow quality, and
shareholder returns (see :func:`_build_checks`), plus a conservative
per-share fair-value band derived from the classic Benjamin Graham growth
formula ``V = EPS x (8.5 + 2g)`` (see :func:`_fair_value`), and a
cyclicality read based on the volatility of year-over-year revenue growth
(see :func:`_cyclical_risk`). It is a heuristic screen, not investment
advice -- see the "not investment advice" note baked into every summary.

Design goals, matching the rest of ``sec_analyzer.interpret``:

* **Never raise.** Every code path -- including a filer with no usable
  us-gaap data at all (e.g. an IFRS/20-F filer) -- returns the documented
  schema with ``None``/"insufficient data" placeholders rather than an
  exception.
* **Same output schema as the LLM providers**, so callers (CLI, web UI) can
  treat ``analyze()`` and ``interpret()`` interchangeably. The extra
  ``score`` key is additive; consumers that don't know about it can ignore
  it.
"""

import logging
import math
import statistics
from typing import Dict, List, Optional, Tuple

from sec_analyzer.normalize.normalizer import to_annual_series

logger = logging.getLogger(__name__)

#: Values stamped into every result's ``_provider``/``_model`` fields,
#: mirroring how the LLM providers identify themselves in ``analyzer.py``.
_PROVIDER = "script"
_MODEL = "rule-based-v1"

#: Nominal look-back window (in fiscal years) for the growth-CAGR checks and
#: the growth estimate used in the fair-value formula. If exactly this many
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

#: Verdict tier boundaries, as a fraction of evaluable checks passed.
_VERDICT_STRONG_PCT = 0.75
_VERDICT_ADEQUATE_PCT = 0.50

#: Growth estimate used in the fair-value formula is clamped to this range
#: (0% to 15%) regardless of the raw computed CAGR, to keep the estimate
#: conservative even for a company on a brief growth tear.
_FAIR_VALUE_GROWTH_MIN = 0.0
_FAIR_VALUE_GROWTH_MAX = 0.15

#: Base (no-growth) multiple in the Graham growth formula V = EPS x (8.5 + 2g).
_GRAHAM_BASE_MULTIPLE = 8.5

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
    """Turn a ``(points, max_points)`` tally into the ``fundamental_verdict`` string."""
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


def _fair_value(latest_fy: Optional[int], series: Dict[str, Dict[int, float]]) -> dict:
    """Compute a conservative per-share fair-value band from the latest EPS.

    Uses the classic Benjamin Graham growth formula, ``V = EPS x (8.5 + 2g)``
    (``g`` expressed in whole percentage points), applied as a range rather
    than a single point estimate: the low end uses half the growth credit
    (``8.5 + g``) and the high end the full formula (``8.5 + 2g``). ``g`` is
    a multi-year earnings (preferred) or revenue CAGR, clamped to
    ``[0%, 15%]`` to keep the estimate conservative.

    When both StockholdersEquity and SharesOutstanding are available for
    the latest fiscal year, a Graham number (``sqrt(22.5 x EPS x BVPS)``) is
    computed as a cross-check and used to pull the low end of the range
    down further if it comes in below the growth-formula low end.

    Returns:
        A dict of the form
        ``{"low": float|None, "high": float|None, "unit": "USD per share", "basis": str}``.
    """
    fy_lbl = _fy_label(latest_fy)
    eps_series = series["EPS"]
    eps = eps_series.get(latest_fy) if latest_fy is not None else None

    if not eps_series or eps is None:
        return {
            "low": None,
            "high": None,
            "unit": "USD per share",
            "basis": (
                "EPS is not available for the latest fiscal year, so no "
                "per-share fair-value estimate can be computed."
            ),
        }

    if eps <= 0:
        return {
            "low": None,
            "high": None,
            "unit": "USD per share",
            "basis": (
                f"Latest EPS ({fy_lbl}) is {eps:.2f}, which is zero or negative. "
                "The Graham growth formula (V = EPS x (8.5 + 2g)) assumes "
                "positive earnings and does not produce a meaningful estimate here."
            ),
        }

    ni_cagr = _windowed_cagr(series["NetIncome"], latest_fy)
    rev_cagr = _windowed_cagr(series["Revenue"], latest_fy)
    if ni_cagr is not None:
        raw_g, g_src = ni_cagr[0], f"the {ni_cagr[2]}-year net income CAGR (FY{ni_cagr[1]}->{fy_lbl})"
    elif rev_cagr is not None:
        raw_g, g_src = (
            rev_cagr[0],
            f"the {rev_cagr[2]}-year revenue CAGR (FY{rev_cagr[1]}->{fy_lbl}, net income "
            "growth was not computable)",
        )
    else:
        raw_g, g_src = 0.0, "no computable growth history (assumed 0%)"

    g = _clamp(raw_g, _FAIR_VALUE_GROWTH_MIN, _FAIR_VALUE_GROWTH_MAX)
    g_pct = g * 100
    low = round(eps * (_GRAHAM_BASE_MULTIPLE + g_pct), 2)
    high = round(eps * (_GRAHAM_BASE_MULTIPLE + 2 * g_pct), 2)

    basis = (
        "Benjamin Graham growth formula, V = EPS x (8.5 + 2g), applied as a "
        "conservative range: low = EPS x (8.5 + g), high = EPS x (8.5 + 2g). "
        f"EPS = {eps:.2f} ({fy_lbl}); g = {g:.1%}, derived from {g_src} and "
        f"clamped to [0%, 15%]"
    )
    if abs(raw_g - g) > 1e-9:
        basis += f" (raw computed growth was {raw_g:.1%})"
    basis += "."

    equity = series["StockholdersEquity"].get(latest_fy) if latest_fy is not None else None
    shares = series["SharesOutstanding"].get(latest_fy) if latest_fy is not None else None
    if equity is not None and shares is not None and shares > 0:
        bvps = equity / shares
        if bvps > 0:
            graham_number = round(math.sqrt(22.5 * eps * bvps), 2)
            basis += (
                f" Cross-check: Graham number = sqrt(22.5 x EPS x BVPS) = "
                f"sqrt(22.5 x {eps:.2f} x {bvps:.2f}) = {graham_number:.2f} "
                f"(BVPS = StockholdersEquity/SharesOutstanding = {bvps:.2f})."
            )
            if graham_number < low:
                basis += f" Low end of the range clamped down to the Graham number ({graham_number:.2f})."
                low = graham_number

    basis += (
        " This is a heuristic screen computed only from the figures provided, "
        "not a price target or investment recommendation."
    )

    return {"low": low, "high": high, "unit": "USD per share", "basis": basis}


def _key_ratios(ratio_by_fy: Dict[int, dict], latest_fy: Optional[int]) -> dict:
    """Return the latest-fiscal-year subset of ratios for the ``key_ratios`` field.

    Only non-``None`` values are included, so a caller (e.g. the web UI)
    never has to render a ``null`` ratio.
    """
    row = ratio_by_fy.get(latest_fy, {}) if latest_fy is not None else {}
    fields = (
        "net_margin",
        "gross_margin",
        "operating_margin",
        "roe",
        "roa",
        "current_ratio",
        "debt_to_equity",
        "fcf_margin",
    )
    return {field: row[field] for field in fields if row.get(field) is not None}


def _build_summary(
    entity_name: Optional[str],
    latest_fy: Optional[int],
    checks: List[dict],
    points: int,
    max_points: int,
    verdict: str,
    cyclical: str,
    fair_value: dict,
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

    sentences = [f"{name} scores {points}/{max_points} on the deterministic checklist for {fy_lbl} ({verdict})."]
    if passed_names:
        sentences.append(f"Strengths flagged by the screen: {', '.join(passed_names)}.")
    if failed_names:
        sentences.append(f"Weak spots flagged by the screen: {', '.join(failed_names)}.")
    sentences.append(cyclical)
    if fair_value.get("low") is not None:
        sentences.append(
            f"A conservative Graham-style estimate puts fair value roughly between "
            f"${fair_value['low']:.2f} and ${fair_value['high']:.2f} per share."
        )
    else:
        sentences.append("A per-share fair-value estimate could not be computed; see the basis field for why.")
    if na_count:
        sentences.append(f"{na_count} checklist item(s) could not be evaluated due to missing data.")
    sentences.append(
        "Deterministic rule-based screen of SEC filings; educational use only, not investment advice."
    )
    return " ".join(sentences)


def analyze(normalized: dict, ratios: List[dict]) -> dict:
    """Run the deterministic, script-based fundamental screen.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.

    Returns:
        A dict matching the same schema as
        :func:`sec_analyzer.interpret.analyzer.interpret` --
        ``fair_value_range``, ``fundamental_verdict``, ``cyclical_risk``,
        ``key_ratios``, ``summary``, ``_provider``, ``_model`` -- plus an
        additional ``score`` key (``{"points", "max_points", "checks"}``)
        that makes the verdict auditable. Never raises: any unexpected
        internal error is caught and turned into an "insufficient data"
        result with the same schema.
    """
    try:
        return _analyze(normalized or {}, ratios or [])
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("rule_based.analyze() failed unexpectedly; returning an insufficient-data result.")
        return {
            "fair_value_range": {
                "low": None,
                "high": None,
                "unit": "USD per share",
                "basis": "An internal error prevented a fair-value estimate.",
            },
            "fundamental_verdict": "insufficient data (0/0 checks evaluable)",
            "cyclical_risk": "insufficient history to assess cyclicality (an internal error occurred).",
            "key_ratios": {},
            "summary": (
                "An internal error prevented the deterministic screen from completing. "
                "Deterministic rule-based screen of SEC filings; educational use only, "
                "not investment advice."
            ),
            "_provider": _PROVIDER,
            "_model": _MODEL,
            "score": {"points": 0, "max_points": 0, "checks": []},
        }


def _analyze(normalized: dict, ratios: List[dict]) -> dict:
    """Do the actual work for :func:`analyze` (split out so the latter can
    wrap it in a single top-level try/except)."""
    entity_name = normalized.get("entity_name")

    series = {concept: to_annual_series(normalized, concept) for concept in _RAW_CONCEPTS}
    ratio_by_fy = {r["fy"]: r for r in ratios if r.get("fy") is not None}

    latest_fy = _resolve_latest_fy(ratios, list(series.values()))

    checks = _build_checks(latest_fy, series, ratio_by_fy)
    points, max_points = _score(checks)
    verdict = _verdict_label(points, max_points)
    cyclical = _cyclical_risk(ratios)
    fair_value = _fair_value(latest_fy, series)
    key_ratios = _key_ratios(ratio_by_fy, latest_fy)
    summary = _build_summary(entity_name, latest_fy, checks, points, max_points, verdict, cyclical, fair_value)

    return {
        "fair_value_range": fair_value,
        "fundamental_verdict": verdict,
        "cyclical_risk": cyclical,
        "key_ratios": key_ratios,
        "summary": summary,
        "_provider": _PROVIDER,
        "_model": _MODEL,
        "score": {"points": points, "max_points": max_points, "checks": checks},
    }
