"""Deterministic SIC-code-based sector classification.

Classifies a filer into one of five buckets that the rest of the valuation
engine uses to pick a method (FCF-DCF vs. P/B x ROE, normalized-earnings DCF
variant, multiples primary key): ``"reit"``, ``"financial"``, ``"cyclical"``,
``"growth_unprofitable"``, ``"mature"``. Classification is purely a function
of the SIC code, with a financial-statement override (unprofitable ->
``"growth_unprofitable"``) when the SIC itself doesn't already pin down
``"reit"``/``"financial"``/``"cyclical"``.

Per ``sec_analyzer/valuation/SPEC.md`` Sec.8, when ``sic`` itself is missing
the *caller* (CLI/engine wiring) is responsible for falling back to the
LLM's phase-1 ``sector_type`` proposal instead of calling this function at
all; called in isolation with a missing/unparseable ``sic``, this function
returns ``"mature"`` as its own safe default.
"""

import logging
from typing import List, Optional, Tuple, Union

from sec_analyzer.normalize.normalizer import to_annual_series

logger = logging.getLogger(__name__)

SECTOR_REIT = "reit"
SECTOR_FINANCIAL = "financial"
SECTOR_CYCLICAL = "cyclical"
SECTOR_GROWTH_UNPROFITABLE = "growth_unprofitable"
SECTOR_MATURE = "mature"

#: REIT SIC code -- classified separately from the broader financial range.
_REIT_SIC = 6798

#: SIC range classified as "financial" (banks, insurance, brokers, ...),
#: excluding the REIT code above.
_FINANCIAL_SIC_RANGE = (6000, 6999)

#: Inclusive SIC ranges classified as "cyclical" (commodity-linked or
#: capital-intensive industries whose earnings/multiples swing hard with
#: the cycle): mining/energy, chemicals, metals, autos, shipping/air.
_CYCLICAL_SIC_RANGES = (
    (1000, 1499),  # mining / energy
    (2800, 2899),  # chemicals
    (3310, 3399),  # metals
    (3711, 3716),  # autos
    (4400, 4599),  # shipping / air
)

#: Individual cyclical SIC codes that don't fall inside a clean range:
#: 2911 (petroleum refining), 3559 (special industry machinery), 3674
#: (semiconductors).
_CYCLICAL_SIC_SINGLES = {2911, 3559, 3674}

#: Realized revenue CAGR (5y, falling back to 3y) strictly above this
#: triggers hyper-grower consideration (SPEC.md Sec.1). Exactly 25% does
#: NOT trigger.
_HYPER_GROWTH_CAGR_THRESHOLD = 0.25

#: FCF margin strictly below this counts as "suppressed cash flow" for
#: hyper-grower clause (b). Exactly 5% does NOT trigger.
_HYPER_GROWTH_FCF_MARGIN_THRESHOLD = 0.05

#: R&D + SBC as a fraction of revenue (S&M proxy -- no standalone S&M line
#: exists in the normalized concepts) strictly above this counts as
#: "aggressive growth investment" for hyper-grower clause (c). Exactly 40%
#: does NOT trigger.
_HYPER_GROWTH_OPEX_INTENSITY_THRESHOLD = 0.40


def _to_int_sic(sic: Optional[Union[int, str]]) -> Optional[int]:
    """Parse ``sic`` (int or numeric string) into an int, or ``None``."""
    if sic is None:
        return None
    try:
        return int(str(sic).strip())
    except (TypeError, ValueError):
        return None


def _is_cyclical_sic(sic: int) -> bool:
    if sic in _CYCLICAL_SIC_SINGLES:
        return True
    return any(lo <= sic <= hi for lo, hi in _CYCLICAL_SIC_RANGES)


def classify_sector(sic: Optional[Union[int, str]], normalized: dict, metrics: dict) -> str:
    """Classify a filer into a valuation-method sector bucket.

    Args:
        sic: The filer's SIC code (int or numeric string), typically
            ``submissions["sic"]``. May be ``None``/unparseable.
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts`` (used for
            the ``NetIncome`` override check).
        metrics: The dict returned by
            ``sec_analyzer.normalize.metrics.compute_metrics`` (used for
            ``latest_fy``).

    Returns:
        One of ``"reit"``, ``"financial"``, ``"cyclical"``,
        ``"growth_unprofitable"``, ``"mature"``. Deterministic, never
        raises: SIC 6798 -> reit; SIC 6000-6999 (excl. 6798) -> financial;
        SIC in the cyclical set -> cyclical; else, if the latest fiscal
        year's ``NetIncome`` is negative -> growth_unprofitable; else ->
        mature. An unparseable/missing ``sic`` returns ``"mature"`` (see
        the module docstring for the SIC-missing engine-wiring fallback).
    """
    sic_int = _to_int_sic(sic)
    if sic_int is None:
        return SECTOR_MATURE

    if sic_int == _REIT_SIC:
        return SECTOR_REIT
    if _FINANCIAL_SIC_RANGE[0] <= sic_int <= _FINANCIAL_SIC_RANGE[1]:
        return SECTOR_FINANCIAL
    if _is_cyclical_sic(sic_int):
        return SECTOR_CYCLICAL

    latest_fy = (metrics or {}).get("latest_fy")
    latest_net_income = None
    if latest_fy is not None:
        latest_net_income = to_annual_series(normalized or {}, "NetIncome").get(latest_fy)

    if latest_net_income is not None and latest_net_income < 0:
        return SECTOR_GROWTH_UNPROFITABLE

    return SECTOR_MATURE


def detect_hyper_grower(metrics: dict, ratios: List[dict], normalized: dict) -> Tuple[bool, List[str]]:
    """Detect whether a filer should be valued in "hyper-grower" mode.

    A Reddit-type problem: a hyper-growth, barely/not-yet-profitable
    company's FCF today is suppressed because growth spend is expensed, so
    a standard FCF-DCF grown at a clamped rate systematically undervalues
    it (see ``sec_analyzer/valuation/SPEC.md`` and ``VALUATION.md`` Sec.4a).
    This function only *detects* the condition from financials; it has no
    say in how the engine values the filer once detected.

    Triggered when the realized revenue CAGR (5-year, falling back to
    3-year) is strictly above 25% AND at least one of:
      (a) latest-FY FCF is zero or negative ("FCF negatif veya sıfır");
      (b) latest-FY FCF margin is strictly below 5% ("FCF marjı %5'in
          altında (bastırılmış nakit akışı)");
      (c) R&D + SBC as a fraction of revenue (an S&M proxy -- no
          standalone S&M line exists in the normalized concepts) is
          strictly above 40% ("Ar-Ge + SBC / gelir %40'ı aşıyor (agresif
          büyüme yatırımı)").
    All boundary values (growth exactly 25%, FCF margin exactly 5%, opex
    intensity exactly 40%) do NOT trigger their respective clause.

    Args:
        metrics: The dict returned by
            ``sec_analyzer.normalize.metrics.compute_metrics`` (uses
            ``revenue_cagr_5y``, ``revenue_cagr_3y``, ``latest_fy``,
            ``fcf``, ``rnd_revenue``, ``sbc_revenue``).
        ratios: The list of per-fiscal-year ratio dicts (accepted for a
            uniform call signature alongside ``metrics``/``normalized``;
            not currently consulted -- ``fcf``/margin inputs already come
            from ``metrics``).
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts`` (used to
            look up the latest annual ``Revenue``).

    Returns:
        A ``(triggered, reasons)`` tuple. ``reasons`` is empty when
        ``triggered`` is ``False``. When ``triggered`` is ``True``,
        ``reasons`` starts with a CAGR summary line followed by one entry
        per clause that fired. Never raises: any unexpected shape in
        ``metrics``/``normalized`` degrades to ``(False, [])``.
    """
    try:
        metrics = metrics or {}
        realized_cagr = metrics.get("revenue_cagr_5y")
        if realized_cagr is None:
            realized_cagr = metrics.get("revenue_cagr_3y")

        if realized_cagr is None or realized_cagr <= _HYPER_GROWTH_CAGR_THRESHOLD:
            return False, []

        latest_fy = metrics.get("latest_fy")
        latest_revenue = None
        if latest_fy is not None:
            latest_revenue = to_annual_series(normalized or {}, "Revenue").get(latest_fy)

        fcf = metrics.get("fcf")
        fcf_margin = None
        if fcf is not None and latest_revenue is not None and latest_revenue > 0:
            fcf_margin = fcf / latest_revenue

        opex_intensity = (metrics.get("rnd_revenue") or 0.0) + (metrics.get("sbc_revenue") or 0.0)

        reasons = [f"Gelir CAGR %{realized_cagr * 100:.1f} (>%25)"]

        if fcf is not None and fcf <= 0:
            reasons.append("FCF negatif veya sıfır")
        if fcf_margin is not None and fcf_margin < _HYPER_GROWTH_FCF_MARGIN_THRESHOLD:
            reasons.append("FCF marjı %5'in altında (bastırılmış nakit akışı)")
        if opex_intensity > _HYPER_GROWTH_OPEX_INTENSITY_THRESHOLD:
            reasons.append("Ar-Ge + SBC / gelir %40'ı aşıyor (agresif büyüme yatırımı)")

        if len(reasons) == 1:
            # Growth condition met but none of (a)/(b)/(c) fired.
            return False, []

        return True, reasons
    except Exception:
        logger.warning("detect_hyper_grower: unexpected error computing hyper-grower detection.", exc_info=True)
        return False, []
