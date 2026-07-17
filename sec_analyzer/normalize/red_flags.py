"""Detect deterministic, rule-based "red flags" from normalized SEC facts.

This module is a thin sibling of ``sec_analyzer.normalize.ratios`` and
``sec_analyzer.normalize.metrics``: it consumes their outputs (the
normalized annual facts, the per-fiscal-year ratio list, and the valuation
metrics dict) and runs a fixed set of quality/valuation checks that are
worth calling out explicitly to a Turkish-speaking end user, rather than
left implicit in a ratio table. Each rule either fires (appending one flag)
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
        f"FY{y}: alacak %{recv_growth[y] * 100:+.1f} vs gelir %{rev_growth[y] * 100:+.1f}"
        for y in streak_years
    )
    return _flag(
        "RECEIVABLES_OUTPACE",
        "Alacaklar gelirden daha hızlı büyüyor",
        f"Son {len(streak_years)} yılda alacak büyümesi gelir büyümesini geride bıraktı ({breakdown}). "
        "Bu, tahsilatların zayıfladığına veya gelirin erken kaydedildiğine işaret edebilir.",
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
        "Kâr var ama işletme nakit akışı negatif",
        f"FY{latest_fy}: net kâr {ni:,.0f} (pozitif) iken işletme faaliyetlerinden nakit akışı "
        f"{ocf:,.0f} (negatif). Kâr kalitesi düşük olabilir.",
    )


def _check_dilution(metrics: dict) -> Optional[dict]:
    """DILUTION: shares outstanding grew more than 5% year-over-year --
    existing shareholders are being meaningfully diluted."""
    shares_yoy = metrics.get("shares_yoy")
    if shares_yoy is None or shares_yoy <= _DILUTION_THRESHOLD:
        return None

    return _flag(
        "DILUTION",
        "Hisse sayısı hızla artıyor (seyrelme riski)",
        f"Dolaşımdaki hisse sayısı yıllık %{shares_yoy * 100:.1f} arttı "
        f"(eşik %{_DILUTION_THRESHOLD * 100:.0f}). Mevcut ortaklar önemli ölçüde seyreliyor.",
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
        "Hisse bazlı ödemeler gelire göre yüksek",
        f"Hisse bazlı ödemeler (SBC), gelirin %{sbc_revenue * 100:.1f}'i "
        f"(eşik %{_SBC_REVENUE_THRESHOLD * 100:.0f}). Bu, raporlanan kârlılığın "
        "gerçek nakit ekonomisini abartabileceği anlamına gelir.",
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
        "5 yıllık ufukta bu kontrol zorunludur: "
        if horizon == "5y"
        else "Uzun vadeli değerlendirmede önemlidir: "
    )
    return _flag(
        "CYCLICAL_TRAP",
        "Düşük P/E yanıltıcı olabilir (döngüsel tepe riski)",
        f"{horizon_note}FY{latest_fy} net kâr marjı %{latest_margin * 100:.1f}, "
        f"tarihi zirveye çok yakın (zirve %{max_margin * 100:.1f}) ve P/E {pe:.1f} "
        f"(eşik {_CYCLICAL_PE_MAX:.0f} altı). Marjlar döngüsel bir tepeden normalleşirse "
        "bugünkü düşük P/E yanıltıcı olabilir.",
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
