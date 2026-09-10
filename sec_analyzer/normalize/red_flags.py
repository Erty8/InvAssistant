"""Detect deterministic, rule-based "red flags" from normalized SEC facts.

This module is a thin sibling of ``sec_analyzer.normalize.ratios`` and
``sec_analyzer.normalize.metrics``: it consumes their outputs (the
normalized annual facts, the per-fiscal-year ratio list, and the valuation
metrics dict) and runs a fixed set of quality/valuation checks that are
worth calling out explicitly to the end user, rather than left implicit in
a ratio table. Each rule either fires (appending one flag)
or doesn't -- there is no scoring here, unlike
``sec_analyzer.interpret.rule_based``'s checklist.

Every rule is defensive: a rule with insufficient data simply doesn't fire
(it is never a false positive due to missing data), and ``detect_red_flags``
itself never raises -- an unexpected internal error is caught and logged,
yielding an empty flag list rather than propagating.
"""

import logging
from typing import Dict, List, Optional

from sec_analyzer.normalize.metrics import resolve_fundamental_fy
from sec_analyzer.normalize.normalizer import to_annual_series

logger = logging.getLogger(__name__)

#: Minimum number of consecutive most-recent fiscal years for which
#: Receivables growth must outpace Revenue growth before RECEIVABLES_OUTPACE
#: fires.
_RECEIVABLES_STREAK_MIN = 2

#: shares_yoy threshold above which DILUTION fires (5%).
_DILUTION_THRESHOLD = 0.05

#: sbc_revenue threshold above which SBC_HIGH fires (10%).
_SBC_REVENUE_THRESHOLD = 0.10

#: CYCLICAL_TRAP requires the latest net_margin to be within this fraction
#: of its historical max (i.e. margin >= _CYCLICAL_MARGIN_NEAR_PEAK * max).
_CYCLICAL_MARGIN_NEAR_PEAK = 0.9

#: CYCLICAL_TRAP requires a P/E strictly below this to fire.
_CYCLICAL_PE_MAX = 15.0

#: CYCLICAL_TRAP requires at least this many fiscal years of net_margin
#: history to be evaluable at all.
_CYCLICAL_MIN_MARGIN_YEARS = 4


def _flag(code: str, message: str, detail: str) -> dict:
    """Build one red-flag entry in the documented shape."""
    return {"code": code, "message": message, "detail": detail}


def _yoy_growth_series(series: Dict[int, float]) -> Dict[int, float]:
    """Return ``{fy: (val - val_prev) / val_prev}`` for every ``fy`` in
    ``series`` whose prior fiscal year is also present with a strictly
    positive value. A non-positive prior-year base makes a percentage
    growth rate meaningless, so those years are simply omitted."""
    growth: Dict[int, float] = {}
    for fy, val in series.items():
        prev = series.get(fy - 1)
        if val is None or prev is None or prev <= 0:
            continue
        growth[fy] = (val - prev) / prev
    return growth


def _check_receivables_outpace(normalized: dict) -> Optional[dict]:
    """RECEIVABLES_OUTPACE: Receivables YoY growth > Revenue YoY growth for
    2+ consecutive most-recent fiscal years. Growing receivables faster than
    revenue can mean the company is recognizing revenue before collecting
    cash for it (aggressive channel stuffing, looser payment terms, or
    weakening collections)."""
    receivables_series = to_annual_series(normalized, "Receivables")
    revenue_series = to_annual_series(normalized, "Revenue")
    if not receivables_series or not revenue_series:
        return None

    recv_growth = _yoy_growth_series(receivables_series)
    rev_growth = _yoy_growth_series(revenue_series)
    common_fys = set(recv_growth) & set(rev_growth)
    if not common_fys:
        return None

    fy = max(common_fys)
    streak_years: List[int] = []
    while fy in recv_growth and fy in rev_growth and recv_growth[fy] > rev_growth[fy]:
        streak_years.append(fy)
        fy -= 1

    if len(streak_years) < _RECEIVABLES_STREAK_MIN:
        return None

    breakdown = "; ".join(
        f"FY{y}: receivables {recv_growth[y] * 100:+.1f}% vs revenue {rev_growth[y] * 100:+.1f}%"
        for y in streak_years
    )
    return _flag(
        "RECEIVABLES_OUTPACE",
        "Receivables are growing faster than revenue",
        f"Over the last {len(streak_years)} year(s), receivables growth has outpaced revenue "
        f"growth ({breakdown}). This can signal weakening collections or revenue being "
        "recognized early.",
    )


def _check_ocf_negative(normalized: dict, metrics: dict) -> Optional[dict]:
    """OCF_NEGATIVE: latest FY reports a net profit on paper, but operating
    cash flow for the same year is negative -- a classic earnings-quality
    warning sign (profit isn't converting into cash)."""
    latest_fy = resolve_fundamental_fy(metrics)
    if latest_fy is None:
        return None

    ni = to_annual_series(normalized, "NetIncome").get(latest_fy)
    ocf = to_annual_series(normalized, "OperatingCashFlow").get(latest_fy)
    if ni is None or ocf is None:
        return None
    if not (ni > 0 and ocf < 0):
        return None

    return _flag(
        "OCF_NEGATIVE",
        "Profitable on paper but operating cash flow is negative",
        f"FY{latest_fy}: net income {ni:,.0f} (positive) while cash flow from operating "
        f"activities is {ocf:,.0f} (negative). Earnings quality may be low.",
    )


def _check_dilution(metrics: dict) -> Optional[dict]:
    """DILUTION: shares outstanding grew more than 5% year-over-year --
    existing shareholders are being meaningfully diluted."""
    shares_yoy = metrics.get("shares_yoy")
    if shares_yoy is None or shares_yoy <= _DILUTION_THRESHOLD:
        return None

    return _flag(
        "DILUTION",
        "Share count is rising fast (dilution risk)",
        f"Shares outstanding grew {shares_yoy * 100:.1f}% year-over-year "
        f"(threshold {_DILUTION_THRESHOLD * 100:.0f}%). Existing shareholders are being "
        "meaningfully diluted.",
    )


def _check_sbc_high(metrics: dict) -> Optional[dict]:
    """SBC_HIGH: stock-based compensation is more than 10% of revenue --
    a large, often under-appreciated non-cash cost that dilutes shareholders
    over time even though it doesn't show up in operating cash flow."""
    sbc_revenue = metrics.get("sbc_revenue")
    if sbc_revenue is None or sbc_revenue <= _SBC_REVENUE_THRESHOLD:
        return None

    return _flag(
        "SBC_HIGH",
        "Stock-based compensation is high relative to revenue",
        f"Stock-based compensation (SBC) is {sbc_revenue * 100:.1f}% of revenue "
        f"(threshold {_SBC_REVENUE_THRESHOLD * 100:.0f}%). This means reported profitability "
        "may overstate the real cash economics of the business.",
    )


def _check_cyclical_trap(ratios: List[dict], metrics: dict, horizon: str) -> Optional[dict]:
    """CYCLICAL_TRAP: latest net_margin sits near its historical peak AND
    the stock trades at a low P/E. For a cyclical business, a low P/E at
    peak margins is often a value trap -- margins (and the "E" in P/E) tend
    to mean-revert downward from a cyclical top, not a genuine bargain.

    This check always runs regardless of ``horizon``, but is flagged in the
    message as a mandatory consideration specifically for a 5-year horizon,
    where riding out a full margin cycle is a real risk.
    """
    margin_by_fy = {
        r["fy"]: r["net_margin"]
        for r in ratios
        if r.get("fy") is not None and r.get("net_margin") is not None
    }
    if len(margin_by_fy) < _CYCLICAL_MIN_MARGIN_YEARS:
        return None

    latest_fy = resolve_fundamental_fy(metrics)
    latest_margin = margin_by_fy.get(latest_fy)
    pe = metrics.get("pe")
    if latest_margin is None or pe is None:
        return None

    max_margin = max(margin_by_fy.values())
    if max_margin <= 0:
        return None
    if latest_margin < _CYCLICAL_MARGIN_NEAR_PEAK * max_margin:
        return None
    if pe >= _CYCLICAL_PE_MAX:
        return None

    horizon_note = (
        "This check is mandatory on a 5-year horizon: "
        if horizon == "5y"
        else "This matters for a long-term assessment: "
    )
    return _flag(
        "CYCLICAL_TRAP",
        "A low P/E may be misleading (cyclical peak risk)",
        f"{horizon_note}FY{latest_fy} net margin is {latest_margin * 100:.1f}%, very close to "
        f"its historical peak ({max_margin * 100:.1f}%), and P/E is {pe:.1f} "
        f"(below the {_CYCLICAL_PE_MAX:.0f} threshold). If margins normalize down from a "
        "cyclical peak, today's low P/E could be misleading.",
    )


def detect_red_flags(normalized: dict, ratios: list, metrics: dict, horizon: str = "1y") -> List[dict]:
    """Run the fixed red-flag checklist and return every flag that fires.

    Args:
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.
        ratios: The list returned by
            ``sec_analyzer.normalize.ratios.compute_ratios``.
        metrics: The dict returned by
            ``sec_analyzer.normalize.metrics.compute_metrics``.
        horizon: Investment horizon hint (``"1y"``, ``"5y"``, ...); only
            affects the wording of ``CYCLICAL_TRAP``'s message, not whether
            any rule fires.

    Returns:
        A list of ``{"code": str, "message": str, "detail": str}`` dicts,
        one per rule that fired, in the fixed rule order documented in the
        module docstring. Empty list if nothing fires. Never raises.
    """
    try:
        return _detect_red_flags(normalized or {}, ratios or [], metrics or {}, horizon)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("detect_red_flags() failed unexpectedly; returning no flags.")
        return []


def _detect_red_flags(normalized: dict, ratios: list, metrics: dict, horizon: str) -> List[dict]:
    checks = (
        _check_receivables_outpace(normalized),
        _check_ocf_negative(normalized, metrics),
        _check_dilution(metrics),
        _check_sbc_high(metrics),
        _check_cyclical_trap(ratios, metrics, horizon),
    )
    return [flag for flag in checks if flag is not None]
