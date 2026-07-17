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

from sec_analyzer.normalize.metrics import resolve_fundamental_fy
from sec_analyzer.normalize.normalizer import to_annual_series

logger = logging.getLogger(__name__)

SECTOR_REIT = "reit"
SECTOR_FINANCIAL = "financial"
SECTOR_CYCLICAL = "cyclical"
SECTOR_GROWTH_UNPROFITABLE = "growth_unprofitable"
SECTOR_MATURE = "mature"

#: REIT SIC code -- classified separately from the broader financial range.
_REIT_SIC = 6798

#: Real-estate operator/lessor SIC codes that get the same FFO treatment as
#: REITs (property owners carry the same GAAP real-estate-depreciation
#: distortion). Excludes real-estate agents/managers (6531) and land
#: subdividers/developers (6552), which are asset-light/inventory businesses,
#: not depreciable-property owners. Non-REIT filers here self-correct: the
#: FFO valuation falls back to P/B×ROE when no depreciation series exists.
_REIT_LIKE_SIC_RANGES = ((6500, 6500), (6510, 6519))

#: SIC range classified as "financial" (banks, insurance, brokers, ...),
#: excluding the REIT code and REIT-like real-estate ranges above.
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
#: 2911 (petroleum refining), 3559 (special industry machinery). 3674
#: (semiconductors) used to live here unconditionally but is now handled
#: by a dedicated growth-aware branch in ``classify_sector`` (see
#: ``_SEMICONDUCTOR_SIC`` below) -- a secular-growth semi is no longer
#: force-classified as cyclical.
_CYCLICAL_SIC_SINGLES = {2911, 3559}

#: SIC code for semiconductors -- handled separately from the blanket
#: cyclical set: a commodity/memory-type or unknown-growth semi stays
#: cyclical, but a secular-growth semi (realized revenue CAGR above
#: ``_SEMICONDUCTOR_GROWTH_CAGR_MIN``) falls through to the ordinary
#: profitability-based classification so it can also be picked up by
#: ``detect_hyper_grower`` (SPEC.md Sec.8).
_SEMICONDUCTOR_SIC = 3674

#: Realized revenue CAGR (5y, falling back to 3y) strictly above this
#: means a semiconductor filer is treated as a secular grower rather than
#: a classic through-the-cycle cyclical. Exactly 15% (or unknown/None
#: CAGR) keeps the filer classified as ``"cyclical"``.
_SEMICONDUCTOR_GROWTH_CAGR_MIN = 0.15

#: Realized revenue CAGR (5y, falling back to 3y) strictly above this
#: triggers hyper-grower consideration (SPEC.md Sec.1/Sec.8). Exactly 25%
#: does NOT trigger the strong tier on its own -- it now falls inside the
#: gray zone below (which additionally requires a fired clause and a high
#: P/S).
_HYPER_GROWTH_CAGR_THRESHOLD = 0.25

#: Lower bound (exclusive) of the hyper-grower "gray zone": realized CAGR
#: in ``(_HYPER_GROWTH_CAGR_GRAY_ZONE_MIN, _HYPER_GROWTH_CAGR_THRESHOLD]``
#: can still trigger hyper-grower mode, but only with a fired clause AND a
#: high P/S (SPEC.md Sec.8). Exactly 20% does NOT qualify for the gray
#: zone.
_HYPER_GROWTH_CAGR_GRAY_ZONE_MIN = 0.20

#: Current P/S strictly above this, inside the gray zone, signals that the
#: market is already pricing in strong growth -- required (together with a
#: fired clause) for the gray zone to trigger.
_HYPER_GROWTH_GRAY_ZONE_PS_THRESHOLD = 8.0

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


def _is_reit_like_sic(sic: int) -> bool:
    return any(lo <= sic <= hi for lo, hi in _REIT_LIKE_SIC_RANGES)


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
        raises: SIC 6798 -> reit; SIC 6500 or 6510-6519 (real-estate
        operators/lessors, same GAAP-depreciation distortion as REITs) ->
        reit; SIC 6000-6999 (excl. the reit codes above, so including 6531
        real-estate agents/managers and 6552 land subdividers/developers,
        which stay asset-light/inventory businesses) -> financial;
        SIC 3674 (semiconductors) -> cyclical when the realized revenue
        CAGR (5y, falling back to 3y) is unknown or <= 15%, otherwise falls
        through to the profitability check below like any other SIC; SIC
        in the (remaining) cyclical set -> cyclical; else, if the latest
        fiscal year's ``NetIncome`` is negative -> growth_unprofitable,
        UNLESS the firm is normally profitable (>=2 prior fiscal years of
        ``NetIncome`` data, a profitable majority among them, AND the
        immediately prior year profitable), in which case the loss is
        treated as a one-off (writedown/litigation/tax charge) and the
        firm still classifies as mature; else -> mature. An unparseable/
        missing ``sic`` returns ``"mature"`` (see the module docstring for
        the SIC-missing engine-wiring fallback).
    """
    sic_int = _to_int_sic(sic)
    if sic_int is None:
        return SECTOR_MATURE

    if sic_int == _REIT_SIC or _is_reit_like_sic(sic_int):
        return SECTOR_REIT
    if _FINANCIAL_SIC_RANGE[0] <= sic_int <= _FINANCIAL_SIC_RANGE[1]:
        return SECTOR_FINANCIAL

    if sic_int == _SEMICONDUCTOR_SIC:
        cagr = (metrics or {}).get("revenue_cagr_5y")
        if cagr is None:
            cagr = (metrics or {}).get("revenue_cagr_3y")
        if cagr is None or cagr <= _SEMICONDUCTOR_GROWTH_CAGR_MIN:
            # Commodity/memory-type or unknown-growth semi: keep the
            # through-cycle normalization treatment.
            return SECTOR_CYCLICAL
        # Secular-growth semi: fall through to the profitability check
        # below instead of forcing "cyclical" -- this also lets
        # detect_hyper_grower pick it up independently.
    elif _is_cyclical_sic(sic_int):
        return SECTOR_CYCLICAL

    latest_fy = resolve_fundamental_fy(metrics)
    ni_series = to_annual_series(normalized or {}, "NetIncome")
    latest_net_income = ni_series.get(latest_fy) if latest_fy is not None else None

    if latest_net_income is not None and latest_net_income < 0:
        # A single loss year in an otherwise consistently profitable firm is a
        # one-off (writedown/litigation/tax charge), not structural
        # unprofitability -- keep it "mature" so it isn't penalized with the
        # higher unprofitable discount floor and isn't excluded from the EPV
        # path. Requires a real profitable history to override: >=2 prior
        # years, a profitable majority, AND the immediately prior year
        # profitable.
        prior = [(fy, v) for fy, v in ni_series.items()
                 if latest_fy is not None and fy < latest_fy and v is not None]
        prior_values = [v for _, v in prior]
        prior_year_ni = ni_series.get(max((fy for fy, _ in prior), default=None)) if prior else None
        usually_profitable = (
            len(prior_values) >= 2
            and sum(1 for v in prior_values if v > 0) > sum(1 for v in prior_values if v < 0)
            and prior_year_ni is not None and prior_year_ni > 0
        )
        if not usually_profitable:
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

    Two tiers, both keyed off the realized revenue CAGR (5-year, falling
    back to 3-year):

    - **Strong tier** -- CAGR strictly above 25% AND at least one of:
        (a) latest-FY FCF is zero or negative ("FCF negatif veya sıfır");
        (b) latest-FY FCF margin is strictly below 5% ("FCF marjı %5'in
            altında (bastırılmış nakit akışı)");
        (c) R&D + SBC as a fraction of revenue (an S&M proxy -- no
            standalone S&M line exists in the normalized concepts) is
            strictly above 40% ("Ar-Ge + SBC / gelir %40'ı aşıyor (agresif
            büyüme yatırımı)").
      Exactly 25% CAGR does NOT qualify for the strong tier (see gray zone
      below); FCF margin exactly 5% and opex intensity exactly 40% do NOT
      trigger their respective clause.
    - **Gray zone** -- CAGR strictly above 20% and less-than-or-equal to
      25% (i.e. ``(0.20, 0.25]``; exactly 20% does NOT qualify) AND at
      least one of clauses (a)/(b)/(c) above AND the current P/S is
      strictly above 8.0 ("market is clearly pricing high growth"). This
      narrower tier exists for filers (e.g. fast-growing semiconductors)
      whose realized growth sits just under the strong-tier bar but whose
      valuation already implies the market expects hyper-growth -- without
      the P/S gate, a merely fast-but-not-extreme grower with a modest
      multiple would be swept in too aggressively.
    - CAGR at or below 20% never triggers, regardless of clauses or P/S.

    Args:
        metrics: The dict returned by
            ``sec_analyzer.normalize.metrics.compute_metrics`` (uses
            ``revenue_cagr_5y``, ``revenue_cagr_3y``, ``latest_fy``,
            ``fcf``, ``rnd_revenue``, ``sbc_revenue``, ``ps``).
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
        ``reasons`` starts with a CAGR (and, for the gray zone, P/S)
        summary line followed by one entry per clause that fired. Never
        raises: any unexpected shape in ``metrics``/``normalized``
        degrades to ``(False, [])``.
    """
    try:
        metrics = metrics or {}
        realized_cagr = metrics.get("revenue_cagr_5y")
        if realized_cagr is None:
            realized_cagr = metrics.get("revenue_cagr_3y")

        if realized_cagr is None:
            return False, []

        latest_fy = resolve_fundamental_fy(metrics)
        latest_revenue = None
        if latest_fy is not None:
            latest_revenue = to_annual_series(normalized or {}, "Revenue").get(latest_fy)

        fcf = metrics.get("fcf")
        fcf_margin = None
        if fcf is not None and latest_revenue is not None and latest_revenue > 0:
            fcf_margin = fcf / latest_revenue

        opex_intensity = (metrics.get("rnd_revenue") or 0.0) + (metrics.get("sbc_revenue") or 0.0)

        clause_a = fcf is not None and fcf <= 0
        clause_b = fcf_margin is not None and fcf_margin < _HYPER_GROWTH_FCF_MARGIN_THRESHOLD
        clause_c = opex_intensity > _HYPER_GROWTH_OPEX_INTENSITY_THRESHOLD
        any_clause = clause_a or clause_b or clause_c

        clause_reasons = []
        if clause_a:
            clause_reasons.append("FCF negatif veya sıfır")
        if clause_b:
            clause_reasons.append("FCF marjı %5'in altında (bastırılmış nakit akışı)")
        if clause_c:
            clause_reasons.append("Ar-Ge + SBC / gelir %40'ı aşıyor (agresif büyüme yatırımı)")

        if realized_cagr > _HYPER_GROWTH_CAGR_THRESHOLD:
            if not any_clause:
                return False, []
            reasons = [f"Gelir CAGR %{realized_cagr * 100:.1f} (>%25)"] + clause_reasons
            return True, reasons

        if realized_cagr > _HYPER_GROWTH_CAGR_GRAY_ZONE_MIN:
            ps = metrics.get("ps")
            high_ps = ps is not None and ps > _HYPER_GROWTH_GRAY_ZONE_PS_THRESHOLD
            if not (any_clause and high_ps):
                return False, []
            reasons = [
                f"Gelir CAGR %{realized_cagr * 100:.1f} (gri bölge %20-25) ve yüksek P/S "
                f"({ps:.1f}x) — piyasa güçlü büyüme fiyatlıyor"
            ] + clause_reasons
            return True, reasons

        return False, []
    except Exception:
        logger.warning("detect_hyper_grower: unexpected error computing hyper-grower detection.", exc_info=True)
        return False, []
