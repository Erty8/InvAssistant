"""Orchestrate the deterministic valuation engine.

:func:`run_valuation` is the single entry point the interpret layer's phase
2, the CLI verdict card, the HTML report, and the store all consume. It
wires together every other module in this package (DCF, reverse-DCF,
multiples, Damodaran sector medians, sensitivity, triangulation) around one
already-validated assumption set and returns the ``valuation`` dict
documented in ``sec_analyzer/valuation/SPEC.md`` Sec.11.

This module never raises for missing or malformed *data* -- every
unavailable piece becomes ``None`` plus a Turkish note in the returned
``notes`` list. It can only raise for a genuine programmer error (wrong
argument types entirely outside the documented contract), and even then the
top-level :func:`run_valuation` wraps everything in a catch-all so a bug
here degrades to an empty-but-shaped result instead of crashing the CLI.
"""

import logging
import math
import statistics
from typing import Dict, List, Optional

from sec_analyzer.config import Config
from sec_analyzer.normalize.metrics import resolve_fundamental_fy
from sec_analyzer.normalize.normalizer import to_annual_series
from sec_analyzer.valuation import (
    damodaran, dcf, multiples, reverse_dcf, revenue_dcf, sanity, sector, sensitivity, triangulate,
)
from sec_analyzer.valuation.dcf import dcf_per_share

logger = logging.getLogger(__name__)

_SCENARIO_KEYS = ("bear", "base", "bull")

#: Scenario band half-width (+/-10%), used ONLY as a fallback when a
#: scenario's own 3x3 sensitivity grid (growth_5y +/-2pp x discount_rate
#: +/-1pp -- see ``_dcf_scenario_band``/``_hyper_scenario_band``/
#: ``_pb_roe_scenario_band``) has fewer than ``_MIN_GRID_CELLS_FOR_BAND``
#: usable cells (Sec.4/F3). No longer the primary band-construction method.
_BAND_FRACTION = 0.10

#: Minimum number of usable (non-None) sensitivity-grid cells required to
#: derive a scenario band from the grid; below this, fall back to the flat
#: +/-10% band above.
_MIN_GRID_CELLS_FOR_BAND = 2

#: P/B x ROE fair-P/B REFERENCE band (Sec.8). No longer clamp bounds for
#: :func:`_justified_pb` -- a high-ROE compounder can legitimately warrant a
#: justified P/B above 4, and clamping discarded that signal. ``_build_pb_roe``
#: now only FLAGS a raw ``fair_pb_base`` outside this band (does not clamp
#: it), preserving the contradiction/information for the report layer instead
#: of silently hiding it. ``_PB_CLAMP_HI`` is also still used, unchanged, as
#: EPV's separate advisory over-capitalization threshold (see
#: ``_build_earnings_power``).
_PB_CLAMP_LO = 0.5
_PB_CLAMP_HI = 4.0

#: P/B x ROE per-scenario fair-P/B scaling factors.
_PB_SCENARIO_SCALE = {"bear": 0.8, "base": 1.0, "bull": 1.2}

#: Earnings-power-value (EPV) margin-median sanity guard (Sec.8a): the
#: latest fiscal year's net-income margin must not deviate more than this
#: fraction from the historical margin median before it's distrusted in
#: favor of a margin-median-based normalized figure (protects against a
#: one-off non-operating swing, e.g. a mark-to-market gain/loss).
_EPV_SANITY_DEVIATION = 0.5

#: EPV headline gate thresholds (Sec.8a): the FCF-DCF base band's high end
#: must sit below this fraction of the EPV base per-share value for
#: FCF-DCF to be considered "suppressed" at all.
_EPV_GATE_FCF_RATIO = 0.5

#: Cash-conversion guard: operating cash flow must be at least this
#: fraction of net income for the EPV headline to be trusted (otherwise a
#: suppressed FCF-DCF might instead reflect a genuine earnings-quality
#: problem, not just growth CapEx/SBC).
_EPV_GATE_CASH_BACKED_RATIO = 0.8

#: Investment-driven guard: CapEx must consume at least this fraction of
#: operating cash flow for FCF suppression to be attributed to growth
#: investment rather than something else.
_EPV_GATE_CAPEX_OCF_RATIO = 0.5

#: fcf0 selection: deviation threshold from the 3-year average FCF beyond
#: which the latest-FY figure is distrusted in favor of the average.
_FCF0_DEVIATION_THRESHOLD = 0.50

_SECTORS_WITHOUT_FCF_DCF = ("financial", "reit")

# --- Hyper-grower revenue-first DCF wiring (SPEC.md Sec.3 / VALUATION.md Sec.4a) ---

#: Terminal growth and steady-state (full convergence) year shared by every
#: hyper-grower scenario, deterministic and NOT overridable by
#: ``hyper_growth_extras`` (only per-scenario target margin, steady-state
#: year, probability, and ``tam_usd`` are overridable -- see SPEC Sec.5).
#:
#: WP2: this is now only the FALLBACK value, used when no risk-free rate is
#: available. The actual value used at runtime is
#: ``min(risk_free_rate, sanity._TERMINAL_GROWTH_MAX)``, computed once in
#: ``_run_valuation`` and passed into ``_build_hyper_growth`` as its
#: ``terminal_growth`` parameter -- see that function's docstring. A
#: hyper-grower that actually reaches steady state is, by definition, a
#: mature company at that point, so its terminal growth must not be set
#: LOWER than a mature firm's (``rule_based._terminal_growth_anchor`` applies
#: the identical rule to the assumptions-driven mature/midgrowth path)
#: just because it started out risky -- that risk is already priced into the
#: discount rate and the scenario probabilities, not a third time here.
_HYPER_TERMINAL_GROWTH = 0.025
_HYPER_DEFAULT_STEADY_STATE_YEAR = 10

#: Per-scenario discount rate (fixed; not overridable by extras). Hyper-
#: growers are young/money-losing and structurally the riskiest cohort this
#: engine values, so every scenario carries a risk premium above the plain
#: unprofitable-company floor (:data:`sanity._DISCOUNT_RATE_MIN_UNPROFITABLE`,
#: 10%) -- even bull, which must never dip below that floor (a below-floor
#: bull rate would be internally inconsistent with a mature-company DCF
#: discounting a far riskier cash flow at a lower cost of equity).
_HYPER_DISCOUNT_RATE_BY_SCENARIO = {"bear": 0.14, "base": 0.12, "bull": 0.10}

#: Default prob-weighting used unless ``hyper_growth_extras`` overrides a
#: scenario's probability.
_HYPER_DEFAULT_PROBABILITIES = {"bear": 0.25, "base": 0.50, "bull": 0.25}

#: Deterministic start-growth cap (Sec.3.1). Raised from 0.40 to 0.60 in
#: lockstep with ``sanity._GROWTH_5Y_HARD_MAX`` -- see that constant's
#: comment for the rationale (the TAM-share/implied-multiple arrival flags,
#: not this cap, are the real honesty mechanism for hyper-growth).
_HYPER_START_GROWTH_CAP = 0.60

#: Flag-only reference threshold (NOT a cap) for the STANDARD two-stage DCF
#: built by ``_build_dcf_scenarios``. This was the old hard growth cap before
#: WP5 raised it to ``_HYPER_START_GROWTH_CAP`` (0.60) in lockstep with
#: ``sanity._GROWTH_5Y_HARD_MAX``. Unlike the hyper-grower and mid-growth
#: revenue-first DCF paths (which each have their own arrival-point / TAM-
#: share / implied-revenue-multiple safety net for aggressive growth),
#: the standard two-stage DCF has none -- so a scenario whose ``growth_5y``
#: exceeds this reference threshold is only flagged with a note, never
#: clamped or reweighted, honestly surfacing the missing safety check
#: instead of silently valuing it.
_STANDARD_DCF_HIGH_GROWTH_FLAG = 0.40

#: WP4: reporting/flag threshold for the hyper-grower mature-target FCF
#: margin -- NOT an applied ceiling. ``_hyper_target_base`` no longer clamps
#: its derived value (half the latest-FY gross margin, floored at today's
#: FCF margin) to this number; when the derived value exceeds it, the
#: caller (``_build_hyper_growth``) attaches a Turkish note and a
#: ``target_margin_flag`` instead of silently truncating the real,
#: gross-margin-derived economics of a genuinely high-margin business.
_HYPER_TARGET_BASE_CAP = 0.30

#: Default mature-state FCF-margin ceiling used when the latest fiscal
#: year's gross margin isn't available. Replaces the earlier "15% gross
#: margin fallback -> 7.5% ceiling" rule, which badly understated already-
#: profitable hyper-growers with no gross-margin data in the normalized
#: facts (e.g. Reddit, which has no GrossProfit/CostOfRevenue concept).
_HYPER_TARGET_MARGIN_CEILING_FALLBACK = 0.20

#: Dilution rule cap (Sec.3.2). The former SBC-based term
#: (``sbc_revenue * 0.3``) was removed (F2): SBC is now expensed directly
#: in the FCF margin that feeds the hyper-grower projection, so also
#: inflating dilution by SBC/revenue would double-count the same drag.
_HYPER_DILUTION_CAP = 0.05

#: Reverse-DCF "arrival point" (revenue-multiple) flag thresholds (Sec.3.3):
#: base-scenario revenue_multiple <= 8 -> "makul"; 8 < m <= 15 -> "agresif";
#: m > 15 -> "asiri_agresif".
_HYPER_ARRIVAL_AGGRESSIVE_MULTIPLE = 8
_HYPER_ARRIVAL_EXTREME_MULTIPLE = 15

#: TAM-share arrival flag thresholds (Sec.3.3), used instead of the
#: revenue-multiple thresholds above whenever ``tam_usd`` is known.
_HYPER_TAM_SHARE_AGGRESSIVE = 0.40
_HYPER_TAM_SHARE_INVALID = 0.60

#: CapEx-intensity threshold (CapEx / Revenue) strictly above which a filer
#: is treated as "capex-heavy" and the maintenance/growth CapEx split
#: (Sec.3.6) is applied to its starting FCF margin. Below this, the split is
#: never applied and behavior is byte-for-byte unchanged. Shared by both the
#: hyper-grower path (:func:`_build_hyper_growth`) via
#: :func:`_maintenance_adjusted_margin`.
_CAPEX_HEAVY_INTENSITY_THRESHOLD = 0.30

#: Floor on the maintenance-CapEx proxy, as a fraction of revenue (Sec.3.6,
#: reviewer Finding 2). Current-year D&A understates steady-state
#: maintenance CapEx for a still-ramping asset base (a data-center builder's
#: future depreciation reflects the grown-out fleet, not today's small one),
#: so ``growth_capex = capex - max(d&a, this * revenue)`` never treats more
#: of CapEx as "growth" (relievable) than is defensible.
_MAINTENANCE_CAPEX_MIN_PCT_REVENUE = 0.05

# --- Mature, FCF-suppressed-but-growing revenue-first DCF (VALUATION.md
# Sec.4/4a addendum): a second growth-inclusive alternative to the
# zero-growth EPV anchor (Sec.8a) for mature filers whose FCF is suppressed
# by heavy growth investment while they still have genuine, realized
# top-line growth left (e.g. Amazon) -- as opposed to a truly mature,
# no-longer-growing filer, for which EPV alone remains the right floor. ---

#: Minimum realized revenue CAGR (Sec below, reviewer Finding 2) required to
#: even attempt this method -- below this (or at/below the scenario's own
#: terminal growth, i.e. nothing left to fade), the growth story isn't real
#: enough to model a fade off of, and the engine falls back to EPV/raw
#: FCF-DCF instead.
_MATURE_REV_DCF_MIN_GROWTH = 0.10

#: Flat statutory-tax-rate proxy used only to derive a NOPAT-based mature
#: FCF-margin anchor (see ``_mature_target_fcf_margin``) -- not a real tax
#: calculation, just a conservative stand-in.
_MATURE_TAX_ASSUMPTION = 0.25

#: Haircut applied to the NOPAT-margin anchor approximating the
#: reinvestment drag a mature-but-still-growing filer keeps paying even at
#: "steady state" (working capital, maintenance capex beyond D&A, etc.).
_MATURE_REINVEST_HAIRCUT = 0.85

#: Multiplier applied to the single best historical raw FCF margin
#: ((OCF-CapEx)/Revenue) to derive the hist-anchor ceiling.
_MATURE_HIST_UPLIFT = 1.5

#: WP4: reporting/flag threshold for the mature target FCF margin -- NOT an
#: applied ceiling. ``_mature_target_fcf_margin`` no longer clamps its
#: ``min(nopat, hist_anchor)`` result to this number; the caller
#: (``_build_mature_revenue_dcf``) compares the returned value against this
#: constant itself and attaches a Turkish note plus a ``target_margin_flag``
#: when it's exceeded, instead of silently truncating it.
_MATURE_TARGET_CAP = 0.15

#: Full convergence ("steady state") year for the mature revenue-first
#: DCF's growth/margin fade -- shorter than the hyper-grower default (10)
#: since a mature filer's growth story is closer to already playing out.
#: Must stay <= ``revenue_dcf.HORIZON_YEARS`` (10).
_MATURE_STEADY_STATE_YEAR = 7

#: Per-scenario mature-target-margin scaling factors (mirrors
#: ``_PB_SCENARIO_SCALE``/hyper's own bear/base/bull spread).
_MATURE_TARGET_MARGIN_SCALE = {"bear": 0.7, "base": 1.0, "bull": 1.2}

# --- Mid-growth, loss-making revenue-first DCF (Roadmap Madde 2 / SPEC
# Sec.8d): a revenue-first alternative to a multiples-only headline for
# `growth_unprofitable` filers that grow the top line at a real but
# sub-hyper (12-20%) rate and are not picked up by `detect_hyper_grower`
# (which requires CAGR > 20%). Sits between the mature (Sec.8b) and
# hyper-grower (Sec.3) revenue-first paths: a shorter fade and a lower
# margin ceiling than hyper, but -- unlike the mature path -- a
# gross-margin-derived target (loss-makers have no positive operating/FCF
# margin history to anchor on) and hyper-style dilution/financing-share
# handling (a mid-growth loss-maker still funds burn by issuing equity). ---

#: Realized revenue CAGR floor (inclusive) to even attempt this method. Set
#: at 12% deliberately below `detect_hyper_grower`'s 20% gray-zone floor so
#: the 12-20% loss-makers that fall through hyper detection are still valued
#: by a revenue-first model rather than multiples alone.
_MIDGROWTH_MIN_GROWTH = 0.12

#: WP4: reporting/flag threshold for the mid-growth mature target FCF
#: margin -- NOT an applied ceiling -- sitting, as a reference point, between
#: the mature path's 15% and the hyper path's 30%: a still-unprofitable
#: mid-grower's defensible steady-state margin isn't expected to be modeled
#: as high as a proven hyper-grower's, but when the gross-margin-derived
#: value exceeds this anyway, ``_build_midgrowth_revenue_dcf`` attaches a
#: Turkish note plus ``target_margin_flag`` instead of silently truncating.
_MIDGROWTH_TARGET_CAP = 0.20

#: Full-convergence ("steady state") year for the mid-growth fade -- between
#: mature's 7 and hyper's 10. Must stay <= ``revenue_dcf.HORIZON_YEARS`` (10).
_MIDGROWTH_STEADY_STATE_YEAR = 8


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _round_or_none(value: Optional[float], ndigits: int) -> Optional[float]:
    return None if value is None else round(value, ndigits)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _non_sbc_dilution(
    metrics: dict, normalized: dict, fy: Optional[int]
) -> "tuple[float, Optional[str], float]":
    """Project future share-count dilution net of SBC-driven issuance.

    ``metrics["shares_yoy"]`` is realized share-count growth, which itself
    embeds SBC-driven share issuance. SBC is already expensed as a cost in
    every FCF margin fed into the revenue-first DCF projections (see the
    ``sbc_latest`` subtraction in :func:`_build_hyper_growth` and
    :func:`_build_midgrowth_revenue_dcf`), so also projecting future
    per-share dilution from the raw ``shares_yoy`` would charge the same SBC
    cost twice -- once as a margin drag, once as per-share dilution (a known
    Damodaran caution). When ``market_cap`` is usable, this strips the
    SBC-implied share-issuance rate (``sbc_latest / market_cap``) out of
    ``shares_yoy`` before clamping; without a usable ``market_cap`` it falls
    back to today's raw-``shares_yoy`` behavior unchanged.

    Args:
        metrics: Per-ticker metrics dict; reads ``shares_yoy`` and
            ``market_cap``.
        normalized: Normalized financial-statement dict; reads the ``SBC``
            annual series for ``fy``.
        fy: The fiscal year to read SBC for (typically
            ``resolve_fundamental_fy``'s result); ``None`` degrades
            ``sbc_latest`` to ``0.0``.

    Returns:
        A ``(rate, note, sbc_dilution_excluded)`` tuple. ``rate`` is the
        dilution rate to feed into the DCF (clamped to
        ``[0.0, _HYPER_DILUTION_CAP]``). ``note`` is a Turkish string to
        append to the caller's ``notes`` list when the SBC adjustment
        actually changed something, else ``None``. ``sbc_dilution_excluded``
        is the raw SBC-implied share-issuance rate that was subtracted (0.0
        when not applicable). Never raises.
    """
    shares_yoy = metrics.get("shares_yoy")
    if shares_yoy is None or shares_yoy <= 0:
        return 0.0, None, 0.0

    market_cap = metrics.get("market_cap")
    if not _is_number(market_cap) or market_cap <= 0:
        return _clamp(shares_yoy, 0.0, _HYPER_DILUTION_CAP), None, 0.0

    sbc_latest = to_annual_series(normalized, "SBC").get(fy) if fy is not None else None
    sbc_dilution = (sbc_latest or 0.0) / market_cap
    non_sbc = max(0.0, shares_yoy - sbc_dilution)
    rate = _clamp(non_sbc, 0.0, _HYPER_DILUTION_CAP)

    if sbc_dilution > 0.0 and shares_yoy > 0.0:
        note = (
            "SBC ihraçları marjda gider olarak zaten fiyatlandığı için dilüsyon projeksiyonundan "
            "çıkarıldı (çift sayım önlendi); kalan dilüsyon yalnızca SBC-dışı ihraçları yansıtır."
        )
        return rate, note, sbc_dilution
    return rate, None, 0.0


def _empty_fair_value_range() -> dict:
    return {
        key: {"lo": None, "hi": None, "growth": None, "discount_rate": None, "note": None}
        for key in _SCENARIO_KEYS
    }


def _empty_valuation(sector_type: Optional[str], assumptions: dict) -> dict:
    """A minimal, fully-shaped result used only if ``run_valuation`` hits an
    unexpected internal error -- keeps every downstream consumer (CLI, HTML
    report, store) working against the documented shape even in that case."""
    return {
        "sector_type": sector_type,
        "fcf0": None,
        "fcf0_source": None,
        "dcf": {
            "enabled": False, "disabled_reason": None, "scenarios": None, "normalized_variant": None,
            "high_growth_flag": False,
        },
        "pb_roe": None,
        "ffo": None,
        "earnings_power": None,
        "earnings_power_headline": False,
        "fair_value_range": _empty_fair_value_range(),
        "reverse_dcf": {
            "implied_growth": None, "realized_cagr_5y": None, "realized_label": None, "bracket_status": "no_data",
        },
        "multiples": {
            "history": [],
            "current": {"pe": None, "ps": None, "pfcf": None, "pffo": None},
            "pe_percentile": None,
            "ps_percentile": None,
            "pfcf_percentile": None,
            "pffo_percentile": None,
            "history_years": 0,
            "sector": {"available": False, "industry": None, "pe_median": None, "ps_median": None, "pfcf_median": None},
            "growth_adjusted": _empty_growth_adjusted("peg", "PEG", "P/E", None),
        },
        "sensitivity": None,
        "triangulation": {
            "signals": {"dcf": "veri_yok", "reverse_dcf": "veri_yok", "multiples": "veri_yok"},
            "confidence": "DÜŞÜK",
            "direction": "belirsiz",
        },
        "hyper_growth": False,
        "hyper_growth_detail": None,
        "mature_revenue_headline": False,
        "mature_revenue_detail": None,
        "midgrowth_revenue_headline": False,
        "midgrowth_revenue_detail": None,
        "cyclical_fcfe_headline": False,
        "cyclical_fcfe_detail": None,
        "assumptions": assumptions or {},
        "notes": ["Değerleme motoru beklenmeyen bir hatayla karşılaştı; sonuçlar eksik olabilir."],
    }


def _sbc_adjusted_fcf_by_fy(ratios: list, normalized: dict) -> Dict[int, float]:
    """Per-FY FCF net of SBC (Sec.4/F2): ``fcf_fy - sbc_fy``, treating a
    missing SBC figure as ``0.0``.

    SBC (stock-based compensation) is a non-cash add-back inside operating
    cash flow, so a raw OCF-CapEx FCF figure is inflated by an expense that
    real economic owners bear through dilution -- Damodaran's approach (and
    this engine's) is to treat SBC as a genuine cash expense for valuation
    purposes. This is the single source of truth both for the DCF base FCF
    (``_select_fcf0``) and for the realized FCF CAGR that feeds reverse-DCF
    triangulation (F6, see ``_realized_cagr_from_series``). It intentionally
    leaves ``ratios``' own ``fcf`` values (and therefore the *display*
    metrics -- ``ratios[...]["fcf"]``, the P/FCF multiple) untouched: only
    the cash flow fed into the valuation math is SBC-adjusted.
    """
    sbc_series = to_annual_series(normalized, "SBC")
    result: Dict[int, float] = {}
    for row in (ratios or []):
        fy = row.get("fy")
        raw_fcf = row.get("fcf")
        if fy is None or raw_fcf is None:
            continue
        result[fy] = raw_fcf - (sbc_series.get(fy) or 0.0)
    return result


def _realized_cagr_from_series(
    series_by_fy: Dict[int, float], latest_fy: Optional[int]
) -> "tuple[Optional[float], Optional[str]]":
    """5y CAGR (falling back to 3y) from a per-fiscal-year value series (F6).

    ``(value_t / value_{t-n}) ** (1/n) - 1`` for ``n=5``, falling back to
    ``n=3``, each attempted only when BOTH endpoints (``latest_fy`` and
    ``latest_fy - n``) are present in ``series_by_fy`` AND strictly
    positive -- a CAGR across a sign-flipping or zero endpoint isn't
    meaningful (e.g. a company that swung from FCF-negative to positive
    doesn't have a well-defined "growth rate" over that window).

    Used to build the reverse-DCF's realized-growth reference from the
    SBC-adjusted FCF series (standard mode) so it's apples-to-apples with
    the FCF-implied growth rate reverse-DCF solves for (F6) -- revenue
    CAGR (already computed by ``metrics.compute_metrics``) remains the
    reference in hyper-grower mode, where the reverse-DCF solve is itself
    revenue-based.

    Args:
        series_by_fy: A ``{fy: value}`` per-fiscal-year series (e.g.
            ``_sbc_adjusted_fcf_by_fy``'s output).
        latest_fy: The most recent fiscal year to anchor the window on, or
            ``None``.

    Returns:
        A ``(cagr, label)`` tuple where ``label`` is ``"5y"``/``"3y"``, or
        ``(None, None)`` if neither window is usable.
    """
    if latest_fy is None:
        return None, None
    for n, label in ((5, "5y"), (3, "3y")):
        end = series_by_fy.get(latest_fy)
        start = series_by_fy.get(latest_fy - n)
        if end is not None and start is not None and end > 0 and start > 0:
            return (end / start) ** (1.0 / n) - 1, label
    return None, None


def _select_fcf0(
    metrics: dict, sbc_adjusted_fcf_by_fy: Dict[int, float]
) -> "tuple[Optional[float], Optional[str], Optional[str]]":
    """Select the DCF base-year FCF per SPEC Sec.4 (F2: SBC-adjusted).

    Prefers the latest-FY FCF net of SBC (the "ttm" figure -- see
    ``_sbc_adjusted_fcf_by_fy``). Falls back to the 3-year average
    SBC-adjusted FCF when the ttm figure is missing, non-positive, or
    deviates more than 50% from that average -- UNLESS the trailing 3
    fiscal years form a monotonic ramp (see below), in which case the
    deviation is trusted as genuine structural growth/decline rather than a
    one-off spike, and the latest-FY figure is kept. The 3y-average
    fallback is only reachable when it is itself usable (positive); if it
    isn't, a positive ttm figure is still preferred over giving up
    entirely. Returns ``(fcf0, source, note)`` where ``source`` is
    ``"ttm"``/``"3y_avg"``/``None`` and ``note`` is a Turkish string to
    surface, or ``None``. Both the ttm figure and the 3y window it's
    compared against are SBC-adjusted, so the deviation/monotonic checks
    below compare like with like.

    Monotonic-trend detection: only assessed when all three consecutive
    fiscal years ``latest_fy, latest_fy-1, latest_fy-2`` have a non-None
    (SBC-adjusted) fcf. The 3-point series (oldest -> newest) is
    "monotonic" if it is non-decreasing throughout or non-increasing
    throughout. Fewer than 3 consecutive data points means the trend can't
    be assessed, so it is treated as not monotonic (falls through to the
    deviation rule).
    """
    latest_fy = resolve_fundamental_fy(metrics)
    ttm_fcf = sbc_adjusted_fcf_by_fy.get(latest_fy) if latest_fy is not None else None

    window = [sbc_adjusted_fcf_by_fy.get(latest_fy - i) for i in range(3)] if latest_fy is not None else []
    avg_window = [v for v in window if v is not None]
    avg_fcf = sum(avg_window) / len(avg_window) if avg_window else None

    ttm_usable = ttm_fcf is not None and ttm_fcf > 0
    avg_usable = avg_fcf is not None and avg_fcf > 0

    deviates = False
    if ttm_usable and avg_fcf is not None and avg_fcf != 0:
        deviates = abs(ttm_fcf - avg_fcf) / abs(avg_fcf) > _FCF0_DEVIATION_THRESHOLD

    monotonic = False
    if len(window) == 3 and all(v is not None for v in window):
        # window is [latest_fy, latest_fy-1, latest_fy-2] (newest -> oldest);
        # reverse to oldest -> newest for the trend check.
        oldest_to_newest = list(reversed(window))
        non_decreasing = all(oldest_to_newest[i] <= oldest_to_newest[i + 1] for i in range(2))
        non_increasing = all(oldest_to_newest[i] >= oldest_to_newest[i + 1] for i in range(2))
        monotonic = non_decreasing or non_increasing

    if ttm_usable and not deviates:
        return ttm_fcf, "ttm", None

    if ttm_usable and deviates and monotonic:
        note = (
            "Son yılın FCF'i (SBC düşülmüş) 3 yıllık ortalamadan %50'den fazla saptı; ancak FCF istikrarlı "
            "bir trend izlediği için (tek seferlik bir sıçrama değil, yapısal bir artış/azalış) DCF için "
            "başlangıç FCF (fcf0) olarak yine de son yılın rakamı kullanıldı."
        )
        return ttm_fcf, "ttm", note

    if avg_usable:
        note = (
            "DCF için başlangıç FCF (fcf0) olarak son yılın rakamı yerine 3 yıllık ortalama FCF (SBC düşülmüş) "
            "kullanıldı (son yıl verisi eksik, negatif ya da 3 yıllık ortalamadan %50'den fazla saptı)."
        )
        return avg_fcf, "3y_avg", note

    if ttm_usable:
        # Deviates from the average, but the average itself isn't usable
        # (missing or non-positive) -- keep the positive ttm figure rather
        # than discarding perfectly usable data.
        return ttm_fcf, "ttm", None

    return None, None, "Pozitif bir başlangıç FCF (fcf0) hesaplanamadı; DCF bu şirket için üretilemiyor."


def _band(per_share: float) -> "tuple[float, float]":
    """Fallback scenario band: point estimate +/-10%, rounded to 2 decimals.

    Only used when a scenario's own sensitivity grid (see
    ``_dcf_scenario_band``/``_hyper_scenario_band``/``_pb_roe_scenario_band``)
    doesn't have enough usable cells (Sec.4/F3)."""
    lo = round(per_share * (1 - _BAND_FRACTION), 2)
    hi = round(per_share * (1 + _BAND_FRACTION), 2)
    return lo, hi


def _dcf_scenario_band(
    fcf0: float,
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares: float,
    dilution_rate: float,
    per_share: float,
) -> "tuple[float, float, bool]":
    """Derive one DCF scenario's fair-value band from a local 3x3
    sensitivity grid (Sec.4/F3) instead of a flat +/-10%.

    Grid: ``growth_5y +/- _GROWTH_STEP`` (rows) x ``discount_rate +/-
    _DISCOUNT_RATE_STEP`` (cols), reusing ``sensitivity.py``'s own step
    constants so the headline band and the reported sensitivity matrix
    always move by the same increments; ``terminal_growth`` is held fixed
    at this scenario's own value in every cell. The band is the min/max of
    the grid's usable cells (a cell with ``discount_rate <= terminal_growth``
    or a failed ``dcf_per_share`` call is excluded, not treated as 0).

    Falls back to the flat +/-10% band (:func:`_band`) when fewer than
    :data:`_MIN_GRID_CELLS_FOR_BAND` cells are usable -- a band derived
    from 0 or 1 points isn't meaningfully a "sensitivity" band.

    Returns:
        A ``(lo, hi, used_fallback)`` tuple.
    """
    cells: List[float] = []
    for g in (
        growth_5y - sensitivity._GROWTH_STEP, growth_5y, growth_5y + sensitivity._GROWTH_STEP,
    ):
        for r in (
            discount_rate - sensitivity._DISCOUNT_RATE_STEP, discount_rate, discount_rate + sensitivity._DISCOUNT_RATE_STEP,
        ):
            if r <= terminal_growth:
                continue
            try:
                result = dcf_per_share(fcf0, g, terminal_growth, r, shares, dilution_rate)
            except ValueError:
                continue
            cells.append(result["per_share"])

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _build_dcf_scenarios(
    assumptions: dict, fcf0: Optional[float], shares: Optional[float], dilution_rate: float
) -> "tuple[Optional[dict], List[str], bool]":
    """Run the 3-scenario DCF (Sec.4). Returns ``(scenarios, notes,
    high_growth_flag)`` where ``scenarios`` is ``None`` if ``fcf0``/``shares``
    are unusable at all (nothing to compute), otherwise a dict with all three
    scenario keys present -- an individual scenario whose own assumptions are
    invalid (missing fields, or r <= g_t) becomes ``{"per_share": None, "lo":
    None, "hi": None}`` plus a note, without blocking the other scenarios.
    Each scenario's ``lo``/``hi`` band comes from its own 3x3 sensitivity grid
    (see :func:`_dcf_scenario_band`), falling back to the flat +/-10% band
    with an additional note when the grid degrades (Sec.4/F3).

    ``high_growth_flag`` is ``True`` iff at least one scenario has a valid,
    numeric ``growth_5y`` strictly greater than
    :data:`_STANDARD_DCF_HIGH_GROWTH_FLAG` (0.40) -- this standard two-stage
    DCF path has no arrival-point/implied-revenue-multiple safety net (unlike
    the hyper-grower and mid-growth revenue-first paths), so the flag is a
    reporting-only signal (one Turkish note naming the triggering
    scenario(s), appended to ``notes``); it never changes any computed value
    or which scenario is used."""
    notes: List[str] = []
    if fcf0 is None or not shares or shares <= 0:
        return None, notes, False

    scenarios = {}
    high_growth_keys: List[str] = []
    for key in _SCENARIO_KEYS:
        scenario_assumptions = assumptions.get(key) or {}
        growth_5y = scenario_assumptions.get("growth_5y")
        terminal_growth = scenario_assumptions.get("terminal_growth")
        discount_rate = scenario_assumptions.get("discount_rate")

        if _is_number(growth_5y) and growth_5y > _STANDARD_DCF_HIGH_GROWTH_FLAG:
            high_growth_keys.append(key)

        if not all(_is_number(v) for v in (growth_5y, terminal_growth, discount_rate)):
            scenarios[key] = {"per_share": None, "lo": None, "hi": None}
            notes.append(f"{key.capitalize()} senaryosu için DCF varsayımları eksik veya geçersiz.")
            continue

        try:
            result = dcf_per_share(fcf0, growth_5y, terminal_growth, discount_rate, shares, dilution_rate)
        except ValueError as exc:
            scenarios[key] = {"per_share": None, "lo": None, "hi": None}
            notes.append(f"{key.capitalize()} senaryosu için DCF hesaplanamadı: {exc}")
            continue

        per_share = round(result["per_share"], 2)
        lo, hi, used_fallback = _dcf_scenario_band(
            fcf0, growth_5y, terminal_growth, discount_rate, shares, dilution_rate, per_share
        )
        if used_fallback:
            notes.append(
                f"{key.capitalize()} senaryosu için duyarlılık bandı hesaplanamadı; "
                "nokta tahminin +/-%10'u fallback olarak kullanıldı."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    if high_growth_keys:
        names = [key.capitalize() for key in high_growth_keys]
        if len(names) == 1:
            scenario_phrase = f"{names[0]} senaryosunda"
        else:
            scenario_phrase = f"{', '.join(names[:-1])} ve {names[-1]} senaryolarında"
        notes.append(
            f"{scenario_phrase} 5 yıllık büyüme varsayımı %40'ı aşıyor; standart iki-aşamalı DCF'in "
            "(hyper/orta-büyüme revenue-first modellerinin aksine) bir varış noktası (TAM payı/gelir "
            "çarpanı) güvenlik kontrolü yoktur -- bu senaryoyu revenue-first / reverse-DCF çapraz "
            "kontrolüyle karşılaştırmak faydalı olabilir."
        )

    return scenarios, notes, bool(high_growth_keys)


def _normalized_fcf0(normalized: dict, metrics: dict) -> "tuple[Optional[float], List[str]]":
    """Cyclical normalized-earnings fcf0 (Sec.8): the mean of the best
    ``ceil(N/2)`` FCF margins across the ``N`` available fiscal years
    (mid-to-upper cycle), times the latest fiscal year's revenue.

    A plain median degenerates for deep cyclicals: over a typical ~5-year
    window that includes one catastrophic trough year, the median lands on
    the current (near-trough) year, making the "normalized" variant an
    exact no-op copy of the raw trough-FCF DCF. Averaging the upper half of
    the margin distribution instead (best 3 of a 5-6yr window, best 4 of a
    7yr window) approximates through-cycle earning power rather than
    trough earning power. If that upper-half average margin is itself
    non-positive, the variant is considered not meaningful and this
    returns ``None`` plus a Turkish note rather than a nonsensical
    negative "normalized" valuation.

    Per-year margin (F2): ``(ocf - capex - sbc) / revenue``, treating a
    missing SBC figure as ``0.0`` -- same SBC-as-expense treatment as
    ``_sbc_adjusted_fcf_by_fy``, applied here directly since this variant
    builds its own margin history rather than reusing ``ratios``.
    """
    notes: List[str] = []
    latest_fy = resolve_fundamental_fy(metrics)
    if latest_fy is None:
        return None, notes

    revenue_series = to_annual_series(normalized, "Revenue")
    ocf_series = to_annual_series(normalized, "OperatingCashFlow")
    capex_series = to_annual_series(normalized, "CapEx")
    sbc_series = to_annual_series(normalized, "SBC")

    latest_revenue = revenue_series.get(latest_fy)
    if latest_revenue is None or latest_revenue <= 0:
        notes.append("Döngüsel normalize edilmiş FCF hesaplanamadı: son yılın geliri eksik veya negatif.")
        return None, notes

    margins = []
    for fy, revenue in revenue_series.items():
        if revenue is None or revenue <= 0:
            continue
        ocf = ocf_series.get(fy)
        capex = capex_series.get(fy)
        if ocf is None or capex is None:
            continue
        sbc = sbc_series.get(fy) or 0.0
        margins.append((ocf - capex - sbc) / revenue)

    if not margins:
        notes.append("Döngüsel normalize edilmiş FCF hesaplanamadı: yeterli FCF marjı geçmişi yok.")
        return None, notes

    # Mid-to-upper cycle: average the top ceil(N/2) margins rather than
    # taking the median, so a single trough year can't drag the
    # "normalized" figure down to the raw current-year number.
    k = math.ceil(len(margins) / 2)
    top_margins = sorted(margins, reverse=True)[:k]
    normalized_margin = sum(top_margins) / len(top_margins)

    if normalized_margin <= 0:
        notes.append(
            "Döngüsel normalize edilmiş FCF anlamlı değil: üst yarı (mid/tepe döngü) ortalama FCF "
            "marjı pozitif değil."
        )
        return None, notes

    return normalized_margin * latest_revenue, notes


def _justified_pb(roe: float, discount_rate: float, terminal_growth) -> float:
    """Justified price-to-book = (ROE - g) / (r - g), returned RAW (no
    longer clamped). `r` is the cost of equity, `g` the stable growth rate.
    Degrades to the no-growth form ROE/r when g is missing, negative, or
    would make the denominator non-positive (a guard against a non-positive
    denominator, not a cap on the result -- a high-ROE compounder can
    legitimately warrant a justified P/B above 4, so the raw ratio is
    returned as-is; ``_build_pb_roe`` flags, but does not clamp, a
    ``fair_pb_base`` outside the ``[_PB_CLAMP_LO, _PB_CLAMP_HI]`` reference
    band instead)."""
    g = terminal_growth if _is_number(terminal_growth) else 0.0
    if g < 0 or (discount_rate - g) <= 0:
        g = 0.0
    return (roe - g) / (discount_rate - g)


def _build_pb_roe(
    assumptions: dict, normalized: dict, metrics: dict, ratios: list
) -> "tuple[Optional[dict], List[str]]":
    """P/B x ROE anchor for financial/reit sectors (Sec.8).

    Selects its fiscal year independently from ``metrics["latest_fy"]``:
    some filers (e.g. banks reporting a newer dei cover-page share count
    than their latest 10-K's financial statements -- see JPM) have a
    ``metrics["latest_fy"]`` that is newer than the fiscal year their
    ``StockholdersEquity``/ROE actually cover, which would otherwise make
    this anchor silently unavailable even though perfectly good historical
    equity/ROE data exists. Instead this walks the equity series from the
    newest fiscal year down and picks the first one that also has a ROE
    figure (via ``ratios``), independent of whatever ``metrics`` considers
    "latest". The share count itself still comes from ``metrics["shares"]``
    (the current, point-in-time count) since book value per share should be
    divided by shares outstanding *today*, not shares outstanding as of the
    equity fiscal year.

    The anchor multiple is the justified price-to-book ``(ROE - g) / (r - g)``
    (Damodaran), growth-aware via the base scenario's ``terminal_growth`` --
    degrading to the no-growth ``ROE / r`` form when ``g`` is missing or
    degenerate (see :func:`_justified_pb`). The raw multiple is no longer
    clamped to ``[_PB_CLAMP_LO, _PB_CLAMP_HI]``; instead, a base ``fair_pb``
    outside that reference band appends a Turkish note and sets the returned
    ``justified_pb_flag`` (``"above_reference"``/``"below_reference"``) so
    the report layer can surface the signal instead of it being silently
    clipped.

    Returns:
        A ``(result, notes)`` tuple. ``result`` is ``None`` if the anchor
        can't be computed at all, else a dict with keys ``scenarios`` (per
        the existing per-scenario ``{"per_share", "lo", "hi"}`` shape),
        ``fair_pb`` (the raw, unclamped base justified P/B), and
        ``justified_pb_flag`` (``"above_reference"``, ``"below_reference"``,
        or ``None`` when ``fair_pb`` sits inside the reference band).
    """
    notes: List[str] = []
    shares = metrics.get("shares")
    base_assumptions = assumptions.get("base") or {}
    discount_rate_base = base_assumptions.get("discount_rate")
    terminal_growth_base = base_assumptions.get("terminal_growth")

    if not shares or shares <= 0 or not _is_number(discount_rate_base) or discount_rate_base <= 0:
        notes.append("P/B x ROE çapası hesaplanamadı: eksik veya geçersiz girdi (hisse sayısı/iskonto oranı).")
        return None, notes

    equity_series = to_annual_series(normalized, "StockholdersEquity")
    roe_by_fy = {row.get("fy"): row.get("roe") for row in (ratios or []) if row.get("fy") is not None}

    selected_fy, equity_latest, roe = None, None, None
    for fy in sorted(equity_series, reverse=True):
        eq = equity_series.get(fy)
        candidate_roe = roe_by_fy.get(fy)
        if eq is not None and candidate_roe is not None:
            selected_fy, equity_latest, roe = fy, eq, candidate_roe
            break

    if selected_fy is None:
        notes.append("P/B x ROE çapası hesaplanamadı: ROE veya özkaynak verisi eksik.")
        return None, notes

    latest_fy = resolve_fundamental_fy(metrics)
    if latest_fy is not None and selected_fy != latest_fy:
        notes.append(
            f"P/B x ROE çapası için {selected_fy} mali yılının özkaynak/ROE verisi kullanıldı "
            "(en son mali yılın temel verileriyle hisse sayısı hizalı değildi)."
        )

    fair_pb_base = _justified_pb(roe, discount_rate_base, terminal_growth_base)
    book_value_per_share = equity_latest / shares

    justified_pb_flag = None
    if fair_pb_base > _PB_CLAMP_HI:
        justified_pb_flag = "above_reference"
    elif fair_pb_base < _PB_CLAMP_LO:
        justified_pb_flag = "below_reference"
    if justified_pb_flag is not None:
        notes.append(
            f"Adil P/D (justified P/B) {fair_pb_base:.2f}x olağan [{_PB_CLAMP_LO:.1f}, "
            f"{_PB_CLAMP_HI:.1f}] referans aralığının dışında (ROE %{roe * 100:.1f}, "
            f"iskonto oranı %{discount_rate_base * 100:.1f}); yüksek/düşük ROE bunu meşru "
            "kılabilir -- sabit sınırla kırpılmadı."
        )

    scenarios = {}
    for key, scale in _PB_SCENARIO_SCALE.items():
        per_share = round(fair_pb_base * scale * book_value_per_share, 2)
        lo, hi, used_fallback = _pb_roe_scenario_band(
            roe, discount_rate_base, scale, book_value_per_share, per_share, terminal_growth_base
        )
        if used_fallback:
            notes.append(
                f"{key.capitalize()} senaryosu için P/B x ROE duyarlılık bandı hesaplanamadı; "
                "nokta tahminin +/-%10'u fallback olarak kullanıldı."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    return {"scenarios": scenarios, "fair_pb": fair_pb_base, "justified_pb_flag": justified_pb_flag}, notes


def _pb_roe_scenario_band(
    roe: float,
    discount_rate_base: float,
    scale: float,
    book_value_per_share: float,
    per_share: float,
    terminal_growth=None,
) -> "tuple[float, float, bool]":
    """Derive one P/B x ROE scenario's band from ``discount_rate_base +/-
    _DISCOUNT_RATE_STEP`` (Sec.8/F3): recompute the justified ``fair_pb``
    (:func:`_justified_pb`) at each of the 3 nearby discount rates -- ``g``
    (``terminal_growth``) held fixed across the band, exactly like the DCF
    band holds ``terminal_growth`` fixed -- scale by this scenario's own
    ``scale``/``book_value_per_share``, and take the min/max. Falls back to
    the flat +/-10% band (:func:`_band`) when fewer than
    :data:`_MIN_GRID_CELLS_FOR_BAND` discount-rate points are usable (a
    non-positive discount rate makes ``fair_pb`` meaningless and is
    excluded, not clamped to 0).

    Returns:
        A ``(lo, hi, used_fallback)`` tuple.
    """
    cells: List[float] = []
    for dr in (
        discount_rate_base - sensitivity._DISCOUNT_RATE_STEP,
        discount_rate_base,
        discount_rate_base + sensitivity._DISCOUNT_RATE_STEP,
    ):
        if dr <= 0:
            continue
        fair_pb = _justified_pb(roe, dr, terminal_growth)
        cells.append(round(fair_pb * scale * book_value_per_share, 2))

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _select_latest_ffo(
    normalized: dict, metrics: dict
) -> "tuple[Optional[float], Optional[int]]":
    """Per-FY FFO (funds from operations) series and latest-usable-FY
    selection for REITs (Sec.8/FFO): ``FFO_fy = NetIncome_fy +
    Depreciation_fy - GainOnSaleRealEstate_fy + RealEstateImpairment_fy``.

    This is a PRAGMATIC PROXY for Nareit's standardized FFO, moved closer to
    it (Package 2/P2a) by also removing gains on real-estate sales and
    adding back real-estate impairments WHEN those are tagged. Two gaps
    remain, both silent (default to 0.0, never raise):

    * Total D&A (the cash-flow-statement depreciation/depletion/
      amortization add-back) is used wholesale rather than real-estate-only
      depreciation, since the latter isn't separable from this engine's
      normalized data. This slightly OVERSTATES FFO for a filer with
      meaningful non-real-estate amortization (e.g. intangibles from an
      acquisition), but for a pure-play REIT -- whose D&A is overwhelmingly
      building/property depreciation -- it is a close approximation.
    * ``GainOnSaleRealEstate``/``RealEstateImpairment`` (see
      ``normalize/concepts.py``) are best-effort, real-estate-specific tag
      lists; coverage is partial, so a filer using a tag not in the list
      silently contributes 0.0 for that adjustment (identical to today's
      behavior for filers that don't report these at all).

    Mirrors ``_build_pb_roe``'s FY-selection logic: walks the NetIncome
    series newest -> oldest and picks the first fiscal year that ALSO has a
    Depreciation figure for that same year, rather than requiring both
    series to align with ``metrics``'s own notion of the latest fiscal
    year. Does NOT keep walking past that first fiscal year even if its FFO
    turns out to be <= 0 -- "the latest usable FFO" means the newest fiscal
    year with both concepts present, not the newest fiscal year with a
    positive result. The gain/impairment adjustments are read for that SAME
    selected fiscal year only -- they never affect FY selection, which
    still requires only NetIncome + Depreciation.

    Args:
        normalized: The dict returned by ``normalize_facts`` (reads
            ``NetIncome``/``Depreciation`` annual series, the optional
            ``GainOnSaleRealEstate``/``RealEstateImpairment`` series, and --
            for the per-share division below -- the ``SharesOutstanding``
            annual series).
        metrics: Used as a FALLBACK source for ``shares`` (the current,
            point-in-time count), only when the selected FFO fiscal year's
            own share count is missing from ``SharesOutstanding``. Unlike
            ``_build_pb_roe``, which divides book value (a balance-sheet
            STOCK, measured at a point in time -- "today's book value per
            today's share" is coherent) by the CURRENT share count, FFO is a
            period FLOW, so it must be divided by THAT SAME period's own
            share count to be contemporaneous. Dividing a trailing fiscal
            year's FFO by today's (typically larger, for a REIT that issues
            equity regularly) share count systematically understates FFO
            per share. This also makes the anchor consistent with
            ``multiples.multiples_history``'s ``pffo`` column, which already
            divides by the per-FY share count via
            ``to_annual_series(normalized, "SharesOutstanding").get(fy)``.

    Returns:
        A ``(ffo_per_share, selected_fy)`` tuple. ``ffo_per_share`` is
        ``None`` if no fiscal year has both concepts, the resulting FFO is
        <= 0, or shares outstanding are missing/invalid. ``selected_fy`` is
        the fiscal year FFO was computed from (even when the per-share
        result is ``None`` because shares were invalid), or ``None`` if no
        fiscal year had both concepts.

    Note:
        Dividing by the FFO fiscal year's own share count still leaves this
        a TRAILING (not run-rate/forward) FFO per share: for a serial
        equity issuer whose in-year acquisitions/share issuances weren't
        perfectly accretive, the trailing figure can still over/understate
        the true run-rate. A forward/run-rate FFO (e.g. annualizing the
        most recent partial period, or using next-FY guidance) would be
        the fully-correct fix; that refinement is out of scope here.
    """
    ni_series = to_annual_series(normalized, "NetIncome")
    dep_series = to_annual_series(normalized, "Depreciation")
    gain_series = to_annual_series(normalized, "GainOnSaleRealEstate")
    impair_series = to_annual_series(normalized, "RealEstateImpairment")

    selected_fy, ffo = None, None
    for fy in sorted(ni_series, reverse=True):
        ni = ni_series.get(fy)
        dep = dep_series.get(fy)
        if ni is not None and dep is not None:
            # A us-gaap "GainLoss" element is positive for a realized gain
            # (which already inflated GAAP net income) and negative for a
            # loss, so "- gain" removes a gain and, for a negative value (a
            # loss), adds it back -- both match Nareit's treatment.
            # Impairments are positive expense amounts that already reduced
            # net income, so "+ impair" adds them back. Both default to 0.0
            # when the fiscal year has no matching tag (backward compatible
            # with fixtures/filers that never report these).
            gain = gain_series.get(fy) or 0.0
            impair = impair_series.get(fy) or 0.0
            selected_fy, ffo = fy, ni + dep - gain + impair
            break

    if selected_fy is None or ffo is None or ffo <= 0:
        return None, selected_fy

    shares_series = to_annual_series(normalized, "SharesOutstanding")
    shares = shares_series.get(selected_fy)
    if not shares or shares <= 0:
        # Fall back to the current point-in-time count only when the FFO
        # fiscal year's own share count is missing from the series.
        shares = metrics.get("shares")
    if not shares or shares <= 0:
        return None, selected_fy

    return ffo / shares, selected_fy


def _build_ffo(
    assumptions: dict, normalized: dict, metrics: dict, ratios: list
) -> "tuple[Optional[dict], List[str]]":
    """FFO-based Gordon-growth anchor for REITs (Sec.8/FFO), replacing the
    P/B x ROE anchor (:func:`_build_pb_roe`) for this sector: GAAP real-
    estate depreciation is a huge non-cash charge that depresses both net
    income and book equity, so a P/B x ROE (or P/E) anchor systematically
    understates a REIT's fair value. FFO (funds from operations, see
    :func:`_select_latest_ffo`) adds that depreciation back.

    Method: a Gordon growth model on FFO per share, independently per
    scenario -- ``per_share = ffo_per_share * (1 + g) / (r - g)``, where
    ``r``/``g`` are that scenario's own ``discount_rate``/``terminal_growth``
    (cost of equity / long-run growth). The ``(1 + g) / (r - g)`` factor is
    exactly the scenario's implied fair P/FFO multiple -- no arbitrary
    target-multiple constant is needed, unlike P/B x ROE's ``fair_pb``. A
    scenario is skipped (with a Turkish note, NOT fabricated)
    when its ``r``/``g`` are missing/non-numeric or ``r <= g`` -- Package 1's
    ERP-spread guard makes ``r > g`` the normal case, but this still guards
    defensively rather than dividing by a non-positive spread.

    Args:
        assumptions: The bear/base/bull assumption dict; each scenario's own
            ``discount_rate``/``terminal_growth`` drive its Gordon multiple.
        normalized: The dict returned by ``normalize_facts`` (reads
            ``NetIncome``/``Depreciation`` via :func:`_select_latest_ffo`).
        metrics: Used for ``shares`` (via :func:`_select_latest_ffo``) and to
            resolve the fiscal year via ``resolve_fundamental_fy``.
        ratios: Unused directly (accepted for signature symmetry with
            ``_build_pb_roe``).

    Returns:
        A ``(detail, notes)`` tuple. ``detail`` is ``None`` if no scenario
        was computable (e.g. FFO itself couldn't be built -- the caller
        should then fall back to ``_build_pb_roe``), else a dict with
        ``scenarios`` (bear/base/bull ``{"per_share", "lo", "hi"}`` -- the
        SAME shape as ``_build_pb_roe``'s ``scenarios``, so downstream
        consumption is unchanged), ``ffo_per_share``, and ``implied_pffo``
        (per-scenario Gordon multiple, i.e. the implied fair P/FFO, rounded
        1dp). Never raises.
    """
    notes: List[str] = []
    ffo_per_share, selected_fy = _select_latest_ffo(normalized, metrics)

    if ffo_per_share is None:
        notes.append(
            "FFO çapası hesaplanamadı: net kâr ve amortisman (D&A) verisi aynı mali yılda birlikte mevcut "
            "değil ya da sonuçtaki FFO sıfır/negatif."
        )
        return None, notes

    latest_fy = resolve_fundamental_fy(metrics)
    if latest_fy is not None and selected_fy is not None and selected_fy != latest_fy:
        notes.append(
            f"FFO çapası için {selected_fy} mali yılının net kâr/amortisman verisi kullanıldı "
            "(en son mali yılın temel verileriyle hisse sayısı hizalı değildi)."
        )

    scenarios: Dict[str, dict] = {}
    implied_pffo: Dict[str, float] = {}
    for key in _SCENARIO_KEYS:
        scenario_assumptions = assumptions.get(key) or {}
        r = scenario_assumptions.get("discount_rate")
        g = scenario_assumptions.get("terminal_growth")

        if not _is_number(r) or not _is_number(g) or r <= g:
            notes.append(
                f"{key.capitalize()} senaryosu için FFO Gordon büyüme modeli hesaplanamadı (iskonto oranı/"
                "terminal büyüme eksik ya da iskonto oranı terminal büyümeyi aşmıyor)."
            )
            continue

        gordon_multiple = (1 + g) / (r - g)
        per_share = round(ffo_per_share * gordon_multiple, 2)
        lo, hi, used_fallback = _ffo_scenario_band(ffo_per_share, r, g, per_share)
        if used_fallback:
            notes.append(
                f"{key.capitalize()} senaryosu için FFO duyarlılık bandı hesaplanamadı; nokta tahminin "
                "+/-%10'u fallback olarak kullanıldı."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}
        implied_pffo[key] = round(gordon_multiple, 1)

    if not scenarios:
        return None, notes

    return {"scenarios": scenarios, "ffo_per_share": round(ffo_per_share, 2), "implied_pffo": implied_pffo}, notes


def _ffo_scenario_band(
    ffo_per_share: float, discount_rate: float, terminal_growth: float, per_share: float
) -> "tuple[float, float, bool]":
    """Derive one FFO Gordon-growth scenario's band from ``discount_rate +/-
    _DISCOUNT_RATE_STEP`` (Sec.8/FFO), mirroring :func:`_pb_roe_scenario_band`:
    recompute the Gordon multiple at each of the 3 nearby discount rates
    (terminal growth held fixed at this scenario's own value) and take the
    min/max of the resulting per-share values. Falls back to the flat
    +/-10% band (:func:`_band`) when fewer than
    :data:`_MIN_GRID_CELLS_FOR_BAND` discount-rate points are usable (a rate
    that doesn't clear ``r > g`` is excluded, not clamped).

    Returns:
        A ``(lo, hi, used_fallback)`` tuple.
    """
    cells: List[float] = []
    for r in (
        discount_rate - sensitivity._DISCOUNT_RATE_STEP,
        discount_rate,
        discount_rate + sensitivity._DISCOUNT_RATE_STEP,
    ):
        if r <= terminal_growth:
            continue
        gordon_multiple = (1 + terminal_growth) / (r - terminal_growth)
        cells.append(round(ffo_per_share * gordon_multiple, 2))

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _build_earnings_power(
    assumptions: dict, normalized: dict, metrics: dict, ratios: list
) -> "tuple[Optional[dict], List[str]]":
    """Earnings-power-value (EPV) anchor for mature, FCF-suppressed filers
    (Sec.8a) -- e.g. Amazon, whose free cash flow is depressed by heavy
    growth CapEx and/or stock-based compensation (SBC) even though the
    company is genuinely, cash-flow-backed profitable.

    EPV is Bruce Greenwald's no-growth earnings power: ``normalized net
    income / cost of equity / shares outstanding``. Unlike the FCF-DCF, it
    deliberately has NO growth term (this is a conservative, zero-growth
    floor, not a growth valuation) and NO net-debt bridge (like the rest of
    this engine's FCFE-direct convention, it works directly off levered/
    equity net income rather than an EV-to-equity walk).

    Normalized earnings (mandatory margin-median sanity guard): the latest
    fiscal year's net income is used directly UNLESS it deviates from the
    historical net-margin median (applied to the latest year's revenue) by
    more than :data:`_EPV_SANITY_DEVIATION` -- in that case the margin-
    median-based figure is used instead. This guards against reading a
    one-off non-operating swing (e.g. a large mark-to-market gain/loss, a
    tax one-off, litigation settlement) as if it were sustainable earning
    power; without it, EPV could be wildly distorted by a single unusual
    year the same way FCF-DCF's own ``fcf0`` selection guards against a
    one-off FCF spike (see ``_select_fcf0``).

    An advisory-only note is appended (never altering the computed value)
    when the implied ROE-over-cost-of-equity ratio is very high, warning
    that reading EPV as a floor may be misleading if that return isn't
    sustainable.

    Args:
        assumptions: The phase-1 bear/base/bull assumption dict; only the
            base scenario's ``discount_rate`` (used as the cost of equity)
            is consulted.
        normalized: The dict returned by ``normalize_facts`` (reads
            ``NetIncome``, ``Revenue``, ``StockholdersEquity`` annual
            series).
        metrics: Used for ``shares`` and to resolve the fiscal year via
            ``resolve_fundamental_fy``.
        ratios: Unused directly (accepted for signature symmetry with
            ``_build_pb_roe`` and to allow future ratio-based refinements
            without changing the call site).

    Returns:
        A ``(detail, notes)`` tuple. ``detail`` is ``None`` if EPV can't be
        built (missing shares/discount rate/net income), else a dict with
        ``scenarios`` (bear/base/bull ``{"per_share", "lo", "hi"}``),
        ``per_share`` (the base scenario's point estimate),
        ``normalized_net_income``, ``cost_of_equity``, and
        ``sanity_applied``. Never raises.
    """
    notes: List[str] = []
    shares = metrics.get("shares")
    if not shares or shares <= 0:
        return None, ["Kazanç-gücü çapası hesaplanamadı: geçerli hisse sayısı yok."]

    dr_base = (assumptions.get("base") or {}).get("discount_rate")
    if not _is_number(dr_base) or dr_base <= 0:
        return None, ["Kazanç-gücü çapası hesaplanamadı: geçerli iskonto oranı (cost of equity) yok."]

    fy = resolve_fundamental_fy(metrics)
    ni_series = to_annual_series(normalized, "NetIncome")
    rev_series = to_annual_series(normalized, "Revenue")

    latest_ni = ni_series.get(fy)
    latest_rev = rev_series.get(fy)

    if latest_ni is None or latest_ni <= 0:
        return None, ["Kazanç-gücü çapası hesaplanamadı: son yılın net kârı negatif veya eksik."]

    # --- Normalize earnings (mandatory margin-median sanity guard) ---
    margins = [
        ni_series[y] / rev_series[y]
        for y in ni_series
        if ni_series.get(y) is not None and ni_series[y] > 0
        and rev_series.get(y) is not None and rev_series[y] > 0
    ]

    sanity_applied = False
    if not margins or latest_rev is None or latest_rev <= 0:
        normalized_ni = latest_ni
    else:
        ref_ni = statistics.median(margins) * latest_rev
        if ref_ni > 0 and abs(latest_ni / ref_ni - 1.0) > _EPV_SANITY_DEVIATION:
            normalized_ni = ref_ni
            sanity_applied = True
            notes.append(
                f"Kazanç-gücü tabanı için son yılın net kârı ({latest_ni:,.0f}) geçmiş marj medyanından "
                f"belirgin saptı; tek-seferlik faaliyet-dışı etki olasılığına karşı marj-medyanı bazlı "
                f"normalize kazanç ({ref_ni:,.0f}) kullanıldı."
            )
        else:
            normalized_ni = latest_ni

    # --- Value and scenarios ---
    base_value_per_share = normalized_ni / dr_base / shares

    scenarios = {}
    for key, scale in _PB_SCENARIO_SCALE.items():
        per_share = round(base_value_per_share * scale, 2)
        lo, hi, used_fallback = _epv_scenario_band(normalized_ni, dr_base, scale, shares, per_share)
        if used_fallback:
            notes.append(
                f"{key.capitalize()} senaryosu için kazanç-gücü duyarlılık bandı hesaplanamadı; "
                "nokta tahminin +/-%10'u fallback olarak kullanıldı."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    # --- Over-capitalization advisory note (does not affect the computed value) ---
    equity_series = to_annual_series(normalized, "StockholdersEquity")
    eq = equity_series.get(fy)
    if eq is not None and eq > 0:
        roe = normalized_ni / eq
        if roe / dr_base > _PB_CLAMP_HI:
            notes.append(
                "Kazanç-gücü çapası çok yüksek bir örtük getiri/iskonto oranına dayanıyor; bu getirinin "
                "sürdürülebilirliği belirsizse EPV değerini yukarı-yanlı okumayın."
            )

    return (
        {
            "scenarios": scenarios,
            "per_share": scenarios["base"]["per_share"],
            "normalized_net_income": normalized_ni,
            "cost_of_equity": dr_base,
            "sanity_applied": sanity_applied,
        },
        notes,
    )


def _epv_scenario_band(
    normalized_ni: float, dr_base: float, scale: float, shares: float, per_share: float
) -> "tuple[float, float, bool]":
    """Derive one EPV scenario's band from ``dr_base +/-
    _DISCOUNT_RATE_STEP`` (mirroring :func:`_pb_roe_scenario_band`):
    recompute ``normalized_ni / dr / shares`` at each of the 3 nearby
    discount rates, scale by this scenario's own ``scale``, and take the
    min/max. Falls back to the flat +/-10% band (:func:`_band`) when fewer
    than :data:`_MIN_GRID_CELLS_FOR_BAND` discount-rate points are usable
    (a non-positive discount rate makes the ratio meaningless and is
    excluded, not clamped to 0).

    Returns:
        A ``(lo, hi, used_fallback)`` tuple.
    """
    cells: List[float] = []
    for dr in (dr_base - sensitivity._DISCOUNT_RATE_STEP, dr_base, dr_base + sensitivity._DISCOUNT_RATE_STEP):
        if dr <= 0:
            continue
        cells.append(round(normalized_ni / dr * scale / shares, 2))

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _cyclical_fcfe_scenario_band(
    ni_norm: float,
    roe: float,
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares: float,
    dilution_rate: float,
    per_share: float,
) -> "tuple[float, float, bool]":
    """Derive one cyclical sustainable-growth FCFE scenario's band from
    ``discount_rate +/- _DISCOUNT_RATE_STEP`` (Sec.8e), mirroring
    :func:`_epv_scenario_band`/:func:`_pb_roe_scenario_band`: recompute
    :func:`dcf.fcfe_sustainable_growth_per_share` at each of the 3 nearby
    discount rates, ``growth_5y``/``terminal_growth`` held fixed at this
    scenario's own values and ``terminal_roe=dr`` passed through for each
    nearby rate (the terminal phase fades to that same nearby rate's cost
    of equity, mirroring the base scenario's own convention). Falls back to
    the flat +/-10% band (:func:`_band`) when fewer than
    :data:`_MIN_GRID_CELLS_FOR_BAND` discount-rate points are usable (a rate
    that doesn't clear ``r > terminal_growth``, or a failed call, is
    excluded, not clamped).

    Returns:
        A ``(lo, hi, used_fallback)`` tuple.
    """
    cells: List[float] = []
    for dr in (
        discount_rate - sensitivity._DISCOUNT_RATE_STEP, discount_rate, discount_rate + sensitivity._DISCOUNT_RATE_STEP,
    ):
        if dr <= terminal_growth:
            continue
        try:
            result = dcf.fcfe_sustainable_growth_per_share(
                ni_norm, roe, growth_5y, terminal_growth, dr, shares, dilution_rate, terminal_roe=dr
            )
        except ValueError:
            continue
        cells.append(result["per_share"])

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _build_cyclical_fcfe(
    assumptions: dict, earnings_power: Optional[dict], normalized: dict, metrics: dict,
    shares: Optional[float], dilution_rate: float,
) -> "tuple[Optional[dict], List[str]]":
    """Growth-inclusive sustainable-growth FCFE anchor for capital-intensive
    cyclical filers (SPEC.md Sec.8e) -- e.g. Micron, whose free cash flow is
    suppressed by heavy growth CapEx (fab expansion) every year, so even the
    cycle-mid normalized FCF-DCF (:func:`_normalized_fcf0`) badly
    understates fair value: it charges the entire growth CapEx as a
    permanent cash drain while only booking modest revenue growth.

    This anchor is literally "EPV's normalized earnings, grown with
    reinvestment-funded growth": it reuses
    ``earnings_power["normalized_net_income"]``/``["cost_of_equity"]`` as
    its earnings base/discount rate (so it is guaranteed >= EPV whenever
    ROE > cost of equity -- see :func:`dcf.fcfe_sustainable_growth_per_share`),
    with ROE derived from that same normalized net income divided by
    latest-FY stockholders' equity (the spot balance-sheet snapshot for the
    fiscal year resolved via :func:`resolve_fundamental_fy`). Each
    scenario's own ``discount_rate`` (cost of equity) is also passed
    through as ``terminal_roe`` (Sec.8e addendum), so the terminal/perpetuity
    phase assumes the firm's excess return fades to zero (terminal ROE ==
    cost of equity) even when its near-term ROE is higher.

    Args:
        assumptions: The bear/base/bull assumption dict; each scenario's own
            ``growth_5y``/``terminal_growth``/``discount_rate`` drive its
            FCFE projection (``discount_rate`` doubles as ``terminal_roe``).
        earnings_power: The ``_build_earnings_power`` detail dict (must
            already be built for this sector by the caller), or ``None``.
        normalized: The dict returned by ``normalize_facts`` (reads
            latest-FY ``StockholdersEquity`` to derive spot ROE).
        metrics: Used to resolve the latest fiscal year (via
            ``resolve_fundamental_fy``) that ``StockholdersEquity`` is read
            from.
        shares: Diluted shares outstanding.
        dilution_rate: Annual share-count growth rate (see
            :func:`dcf.fcfe_sustainable_growth_per_share`).

    Returns:
        A ``(detail, notes)`` tuple. ``detail`` is ``None`` if
        ``earnings_power`` is missing/incomplete, shares are invalid,
        equity/ROE can't be resolved or ROE isn't positive, or no scenario
        was computable; else a dict with ``scenarios`` (bear/base/bull
        ``{"per_share", "lo", "hi"}``), ``per_share`` (the base scenario's
        point estimate), ``normalized_net_income``, ``roe``, ``equity``
        (the latest-FY stockholders' equity ``roe`` was derived from),
        ``cost_of_equity``, and ``reinvestment_base`` (the base
        scenario's implied reinvestment rate, for display). Never raises.
        NOTE (WP7): the caller (``_run_valuation``) additionally mutates
        this dict with a ``growth_vs_floor`` key (``"adds"``/``"destroys"``/
        ``None``, see :func:`_growth_vs_floor`) after this function
        returns -- it is not set here.
    """
    notes: List[str] = []
    if not earnings_power or "normalized_net_income" not in earnings_power or "cost_of_equity" not in earnings_power:
        return None, notes
    if not shares or shares <= 0:
        return None, notes

    ni_norm = earnings_power["normalized_net_income"]

    fy = resolve_fundamental_fy(metrics)
    equity = to_annual_series(normalized, "StockholdersEquity").get(fy)
    if equity is None or equity <= 0:
        notes.append("Döngüsel FCFE çapası hesaplanamadı: özkaynak verisi eksik/negatif.")
        return None, notes

    roe = ni_norm / equity
    if roe <= 0:
        notes.append("Döngüsel FCFE çapası hesaplanamadı: normalize edilmiş ROE pozitif değil.")
        return None, notes

    scenarios: Dict[str, dict] = {}
    for key in _SCENARIO_KEYS:
        scenario_assumptions = assumptions.get(key) or {}
        growth_5y = scenario_assumptions.get("growth_5y")
        terminal_growth = scenario_assumptions.get("terminal_growth")
        discount_rate = scenario_assumptions.get("discount_rate")

        if (
            not all(_is_number(v) for v in (growth_5y, terminal_growth, discount_rate))
            or discount_rate <= terminal_growth
        ):
            scenarios[key] = {"per_share": None, "lo": None, "hi": None}
            notes.append(f"{key.capitalize()} senaryosu için döngüsel FCFE varsayımları eksik veya geçersiz.")
            continue

        try:
            result = dcf.fcfe_sustainable_growth_per_share(
                ni_norm, roe, growth_5y, terminal_growth, discount_rate, shares, dilution_rate,
                terminal_roe=discount_rate,
            )
        except ValueError as exc:
            scenarios[key] = {"per_share": None, "lo": None, "hi": None}
            notes.append(f"{key.capitalize()} senaryosu için döngüsel FCFE hesaplanamadı: {exc}")
            continue

        per_share = round(result["per_share"], 2)
        lo, hi, used_fallback = _cyclical_fcfe_scenario_band(
            ni_norm, roe, growth_5y, terminal_growth, discount_rate, shares, dilution_rate, per_share
        )
        if used_fallback:
            notes.append(
                f"{key.capitalize()} senaryosu için döngüsel FCFE duyarlılık bandı hesaplanamadı; "
                "nokta tahminin +/-%10'u fallback olarak kullanıldı."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    if not any(_is_number(cell.get("per_share")) for cell in scenarios.values()):
        return None, notes

    base_growth_5y = (assumptions.get("base") or {}).get("growth_5y")
    reinvestment_base = (
        round(min(base_growth_5y, roe) / roe, 4)
        if _is_number(base_growth_5y) else None
    )

    return (
        {
            "scenarios": scenarios,
            "per_share": scenarios["base"]["per_share"],
            "normalized_net_income": ni_norm,
            "roe": round(roe, 4),
            "equity": equity,
            "cost_of_equity": earnings_power["cost_of_equity"],
            "reinvestment_base": reinvestment_base,
        },
        notes,
    )


def _fcf_dcf_unreliable(
    dcf_scenarios: Optional[dict], earnings_power: Optional[dict], normalized: dict, metrics: dict
) -> "tuple[bool, Optional[str]]":
    """Decide whether the FCF-DCF headline is unreliable enough here to be
    replaced by the EPV headline (Sec.8a).

    This gate exists because a suppressed FCF-DCF band alone is NOT
    sufficient reason to switch to an earnings-power headline: FCF can also
    be low because net income itself is low-quality (i.e. it isn't actually
    backed by cash generation), in which case an NI-based EPV headline
    would be a worse anchor than the (correctly) suppressed FCF-DCF, not a
    better one. So this gate requires ALL of:

    - ``fcf_suppressed``: the FCF-DCF base band's high end is materially
      below the EPV base per-share value (or FCF-DCF wasn't computable at
      all) -- there IS a suppression to correct.
    - ``cash_backed``: operating cash flow is at least
      :data:`_EPV_GATE_CASH_BACKED_RATIO` of net income -- net income is
      actually converting into cash, so it's a trustworthy EPV numerator.
    - ``investment_driven``: CapEx consumes at least
      :data:`_EPV_GATE_CAPEX_OCF_RATIO` of operating cash flow -- the
      suppression is plausibly attributable to heavy growth investment
      (the canonical Amazon story), not some other drag.

    When FCF looks suppressed but the cash-conversion guard fails (NI
    isn't cash-backed), the gate refuses to fire and instead returns an
    earnings-quality warning note -- explicitly surfacing that this is a
    reason for caution, not a silent do-nothing.

    Args:
        dcf_scenarios: The raw FCF-DCF scenario dict (pre-EPV-override), or
            ``None``.
        earnings_power: The ``_build_earnings_power`` detail dict, or
            ``None``.
        normalized: Used to look up ``OperatingCashFlow``/``NetIncome``/
            ``CapEx`` annual series.
        metrics: Used to resolve the fiscal year via
            ``resolve_fundamental_fy``.

    Returns:
        A ``(unreliable, quality_note)`` tuple. ``quality_note`` is a
        Turkish string to surface (only set on the "suppressed but not
        cash-backed" branch), or ``None``. Never raises.
    """
    fy = resolve_fundamental_fy(metrics)
    ocf = to_annual_series(normalized, "OperatingCashFlow").get(fy)
    ni = to_annual_series(normalized, "NetIncome").get(fy)
    capex = to_annual_series(normalized, "CapEx").get(fy)

    epv_base = ((earnings_power or {}).get("scenarios") or {}).get("base", {}).get("per_share")
    if epv_base is None:
        return False, None

    dcf_hi = ((dcf_scenarios or {}).get("base") or {}).get("hi")
    fcf_suppressed = dcf_scenarios is None or dcf_hi is None or dcf_hi < _EPV_GATE_FCF_RATIO * epv_base

    cash_backed = ocf is not None and ni is not None and ni > 0 and ocf >= _EPV_GATE_CASH_BACKED_RATIO * ni
    investment_driven = ocf is not None and ocf > 0 and capex is not None and capex / ocf >= _EPV_GATE_CAPEX_OCF_RATIO

    if fcf_suppressed and cash_backed and investment_driven:
        return True, None
    if fcf_suppressed and not cash_backed:
        return False, (
            "Serbest nakit akışı düşük ve işletme nakit akışı net kârı yeterince desteklemiyor "
            "(OCF < 0.8×net kâr); bu bir kazanç-kalitesi/nakde-çevirme uyarısıdır — manşet değerleme "
            "FCF-DCF'te bırakıldı, kazanç-gücü çapasına geçilmedi."
        )
    return False, None


def _growth_vs_floor(epv_base_ps: Optional[float], growth_base_ps: Optional[float]) -> Optional[str]:
    """Classify a growth-inclusive anchor's base per-share value against the
    zero-growth EPV floor (SPEC.md Sec.8a/8e addendum): ``"destroys"`` when
    the growth-inclusive base is below the EPV floor (ROE < cost of equity
    -- growth is destroying value), ``"adds"`` when it meets or clears the
    floor, or ``None`` when either value is missing/non-numeric (not
    comparable). Never raises.

    Args:
        epv_base_ps: The zero-growth EPV base scenario's per-share value.
        growth_base_ps: The growth-inclusive anchor's (cyclical FCFE or
            mature revenue-first DCF) base scenario's per-share value.

    Returns:
        ``"destroys"``, ``"adds"``, or ``None``.
    """
    if not _is_number(epv_base_ps) or not _is_number(growth_base_ps):
        return None
    return "destroys" if growth_base_ps < epv_base_ps else "adds"


def _hyper_target_base(gross_margin: Optional[float], current_margin: Optional[float]) -> float:
    """``target_base`` (Sec.3.1): the mature-state FCF-margin ceiling --
    half the latest-FY gross margin (WP4: no longer clamped to an absolute
    ceiling -- see ``_HYPER_TARGET_BASE_CAP`` as a reporting-only flag
    threshold the caller compares this function's return value against),
    or a 20% default ceiling when gross margin is unavailable -- floored at
    today's FCF margin whenever the filer is already profitable (a
    currently-profitable hyper-grower must never be modeled as if its
    margin collapses below what it already earns), and capped at gross
    margin when known.

    Args:
        gross_margin: The latest-FY gross margin, already filtered to
            ``None`` unless it is a positive number (callers pass ``gm``,
            not the raw ratio value).
        current_margin: Today's FCF margin (``fcf / latest_revenue``), or
            ``None``/non-positive when the filer isn't currently FCF
            profitable.
    """
    ceiling = gross_margin * 0.5 if gross_margin is not None else _HYPER_TARGET_MARGIN_CEILING_FALLBACK
    if current_margin is not None and current_margin > 0:
        base = max(current_margin, ceiling)
    else:
        base = ceiling
    if gross_margin is not None:
        base = min(base, gross_margin)
    return base


def _hyper_scenario_band(
    revenue0: float,
    start_growth: float,
    terminal_growth: float,
    discount_rate: float,
    current_margin: float,
    target_fcf_margin: float,
    steady_state_year: int,
    shares: float,
    annual_dilution: float,
    financing_shares: float,
    per_share: float,
    mature_discount_rate: Optional[float] = None,
) -> "tuple[float, float, bool]":
    """Derive one hyper-grower scenario's band from a local 3x3 sensitivity
    grid (Sec.3/F3), mirroring :func:`_dcf_scenario_band` but over
    ``revenue_dcf.revenue_first_dcf``: ``start_growth +/- _GROWTH_STEP``
    (rows) x ``discount_rate +/- _DISCOUNT_RATE_STEP`` (cols), everything
    else (``target_fcf_margin``, ``steady_state_year``, ``current_margin``,
    ``annual_dilution``, ``financing_shares``) held fixed at this
    scenario's own values. Falls back to the flat +/-10% band
    (:func:`_band`) when fewer than :data:`_MIN_GRID_CELLS_FOR_BAND` cells
    are usable.

    Args:
        mature_discount_rate: Optional WP3 discount-rate-fade target,
            passed straight through to every grid cell's
            ``revenue_dcf.revenue_first_dcf`` call (each cell still starts
            its OWN row's ``discount_rate +/- _DISCOUNT_RATE_STEP``; only
            the fade's mature target is shared). ``None`` (default) keeps
            the flat, unfaded band exactly as before.

    Returns:
        A ``(lo, hi, used_fallback)`` tuple.
    """
    cells: List[float] = []
    for g in (
        start_growth - sensitivity._GROWTH_STEP, start_growth, start_growth + sensitivity._GROWTH_STEP,
    ):
        for r in (
            discount_rate - sensitivity._DISCOUNT_RATE_STEP, discount_rate, discount_rate + sensitivity._DISCOUNT_RATE_STEP,
        ):
            if r <= terminal_growth:
                continue
            try:
                result = revenue_dcf.revenue_first_dcf(
                    revenue0, g, terminal_growth, r, current_margin, target_fcf_margin,
                    steady_state_year, shares, annual_dilution, financing_shares,
                    mature_discount_rate=mature_discount_rate,
                )
            except ValueError:
                continue
            cells.append(result["per_share"])

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _maintenance_adjusted_margin(
    normalized: dict,
    metrics: dict,
    raw_current_margin: float,
    sector_capex_sales: Optional[float] = None,
) -> "tuple[float, Optional[dict]]":
    """Compute a growth-CapEx-relieved "operating" FCF margin for capex-heavy
    filers (Roadmap Madde 1 / SPEC Sec.3.6).

    A capex-heavy hyper-grower (e.g. a data-center builder like APLD) spends
    CapEx that is many multiples of its maintenance needs; that growth CapEx
    builds future revenue. This returns the margin that would result if only
    *maintenance* CapEx (proxied by D&A, floored at either the sector's own
    Cap Ex/Sales ratio (WP6, Damodaran "Capital Expenditures by Sector") when
    ``sector_capex_sales`` is a usable positive number, or
    :data:`_MAINTENANCE_CAPEX_MIN_PCT_REVENUE` (5%) otherwise) were charged.

    IMPORTANT — this is NOT fed into the headline valuation. A finance review
    showed that relieving growth CapEx from the starting margin while revenue
    still compounds up the growth path books the revenue ramp but charges the
    CapEx funding it *nowhere* -- a one-directional over-valuation (the same
    owner-earnings add-back double-count SPEC Sec.8b rejects). The caller
    (:func:`_build_hyper_growth`) therefore keeps the ACTUAL (unrelieved)
    margin for its headline scenarios (so capex-heavy names still suppress
    honestly) and uses ``ops_margin`` only to compute a separate,
    explicitly-labeled AGGRESSIVE UPSIDE figure.

    The relief is an *additive* correction on top of the caller's
    ``raw_current_margin`` -- ``ops_margin = raw_current_margin + growth_capex
    / revenue``, ``growth_capex = capex - max(d&a, min_pct * revenue)`` --
    keeping it consistent with the caller's own margin base.

    Gate (both must hold, else the raw margin is returned unchanged and the
    returned detail is ``None``):

    * ``capex / revenue > _CAPEX_HEAVY_INTENSITY_THRESHOLD`` (0.30) -- the
      filer is genuinely capex-heavy, not an asset-light software grower.
    * ``capex > maintenance_capex`` -- there IS growth CapEx above the
      (floored) maintenance level to relieve.

    All figures are read for the latest fundamental fiscal year
    (``resolve_fundamental_fy``). Never raises (only reads dict data).

    Args:
        normalized: Normalized fundamentals (``Revenue``/``CapEx``/
            ``Depreciation`` annual series).
        metrics: See ``compute_metrics`` (uses ``resolve_fundamental_fy``).
        raw_current_margin: The caller's own (unrelieved) current FCF margin,
            the base the relief is additive on top of.
        sector_capex_sales: The matched Damodaran sector's Cap Ex/Sales ratio
            (WP6), e.g. ``0.045`` for 4.5% of revenue, or ``None``. When this
            is a usable positive number it REPLACES
            :data:`_MAINTENANCE_CAPEX_MIN_PCT_REVENUE` (5%) as the
            maintenance-CapEx floor's percent-of-revenue term -- a
            data-center/telecom/utility sector with a genuinely higher
            maintenance-capex intensity than the flat 5% default no longer
            has its growth CapEx overstated (and its relieved margin
            understated) by that generic floor. ``None`` or a
            non-positive/non-numeric value keeps the flat 5% default exactly
            as before this parameter existed.

    Returns:
        A ``(ops_margin, capex_normalization)`` tuple. ``capex_normalization``
        is ``None`` when the split was not applied, else a dict with keys
        ``applied`` (always ``True`` when present), ``capex_intensity``,
        ``maintenance_capex`` (the floored proxy), ``growth_capex``,
        ``raw_current_margin``, ``ops_current_margin`` (the caller adds
        ``upside_per_share``/``upside_lo``/``upside_hi``), and
        ``maintenance_capex_floor_note`` (only present, Turkish, when the
        sector floor -- not the 5% default -- actually determined
        ``maintenance_capex``).
    """
    fy = resolve_fundamental_fy(metrics)
    if fy is None:
        return raw_current_margin, None

    revenue = to_annual_series(normalized, "Revenue").get(fy)
    capex = to_annual_series(normalized, "CapEx").get(fy)
    dep = to_annual_series(normalized, "Depreciation").get(fy)

    if revenue is None or revenue <= 0 or capex is None or dep is None or dep <= 0:
        return raw_current_margin, None

    # Finding 2: floor the maintenance-CapEx proxy so current-year D&A (which
    # understates the maintenance burden of a still-ramping asset base)
    # cannot make "growth CapEx" look larger than it defensibly is.
    # WP6: use the sector's own Cap Ex/Sales ratio for that floor's
    # percent-of-revenue term when it's a usable positive number, else keep
    # the flat 5% default -- a data-center/telecom/utility sector with a
    # genuinely higher maintenance-capex intensity no longer has its growth
    # CapEx (and thus the relieved margin) mis-sized by the generic floor.
    used_sector_floor = _is_number(sector_capex_sales) and sector_capex_sales > 0
    maintenance_floor_pct = sector_capex_sales if used_sector_floor else _MAINTENANCE_CAPEX_MIN_PCT_REVENUE
    maintenance_capex = max(dep, maintenance_floor_pct * revenue)

    capex_intensity = capex / revenue
    if not (capex_intensity > _CAPEX_HEAVY_INTENSITY_THRESHOLD and capex > maintenance_capex):
        return raw_current_margin, None

    growth_capex = capex - maintenance_capex
    ops_margin = raw_current_margin + growth_capex / revenue

    capex_normalization = {
        "applied": True,
        "capex_intensity": round(capex_intensity, 4),
        "maintenance_capex": maintenance_capex,
        "growth_capex": growth_capex,
        "raw_current_margin": round(raw_current_margin, 4),
        "ops_current_margin": round(ops_margin, 4),
    }
    # Only note the sector floor when it actually drove maintenance_capex
    # above the D&A proxy (i.e. it was the max()'s winning term) -- a sector
    # floor lower than D&A never changes the outcome and shouldn't claim credit.
    if used_sector_floor and maintenance_floor_pct * revenue > dep:
        capex_normalization["maintenance_capex_floor_note"] = (
            f"Bakım-CapEx tabanı sektör verisine göre %{sector_capex_sales * 100:.1f} olarak alındı "
            "(Damodaran Cap Ex/Sales), varsayılan %5 yerine."
        )
    return ops_margin, capex_normalization


def _build_hyper_growth(
    metrics: dict,
    ratios: list,
    normalized: dict,
    price: Optional[float],
    shares: Optional[float],
    hyper_reasons: List[str],
    extras: Optional[dict],
    terminal_growth: float = _HYPER_TERMINAL_GROWTH,
    mature_discount_rate: Optional[float] = None,
    sector_capex_sales: Optional[float] = None,
) -> "tuple[Optional[dict], List[str]]":
    """Build the hyper-grower revenue-first DCF detail (SPEC.md Sec.3).

    Runs the deterministic bear/base/bull revenue-first DCF scenarios
    (``valuation.revenue_dcf.revenue_first_dcf``), the prob-weighted
    expected value, the reverse-DCF-derived "arrival point" flag, and the
    implied-expectations block, optionally overridden per-scenario by the
    LLM/user-supplied ``hyper_growth_extras`` (target margin, steady-state
    year, probability, TAM). Never raises: any missing/invalid input or
    ``revenue_first_dcf``/bisection failure degrades to ``(None, notes)``
    with a Turkish note explaining why, so the caller can fall back to
    ``hyper_growth = False`` without losing the standard valuation.

    Args:
        metrics: See ``compute_metrics`` (uses ``latest_fy``, ``fcf``,
            ``revenue_cagr_5y``/``_3y``, ``shares_yoy``).
        ratios: Per-FY ratio dicts (uses the latest FY's ``gross_margin``).
        normalized: Used to look up the latest annual ``Revenue`` (this
            year's and the prior year's, for the F4 latest-YoY blend) and
            ``SBC`` (subtracted from today's FCF margin, F2).
        price: Current market price, or ``None`` (implied-expectations
            bisections degrade to ``None`` without it).
        shares: Base share count.
        hyper_reasons: The reason strings from
            ``sector.detect_hyper_grower``, echoed into the output's
            ``"reasons"``.
        extras: The optional ``hyper_growth_extras`` dict (SPEC Sec.5):
            ``{"tam_usd": .., "per_scenario": {"bear"/"base"/"bull":
            {"target_fcf_margin", "steady_state_year", "probability"}}}``.
            ``None`` in pure deterministic (script) mode.
        terminal_growth: The shared terminal-growth anchor (WP2):
            ``min(risk_free_rate, sanity._TERMINAL_GROWTH_MAX)``, computed
            once by the caller (``_run_valuation``) from
            ``damodaran.load_sector_data(...)["risk_free"]``. Defaults to
            :data:`_HYPER_TERMINAL_GROWTH` (2.5%) so existing direct callers
            (and tests) that don't pass it keep the old behavior. Used for
            every revenue-first DCF call in this function -- hyper-growers
            get no separate, lower terminal rate than mature/midgrowth
            filers (see the module-level comment on
            :data:`_HYPER_TERMINAL_GROWTH`).
        mature_discount_rate: The shared mature (steady-state) discount
            rate every scenario's revenue-first DCF fades toward (WP3
            Damodaran fade), computed once by the caller (``_run_valuation``)
            from ``assumptions["base"]["discount_rate"]`` (already
            CAPM-aware and clamped), floored at ``terminal_growth +
            sanity._MIN_ERP_SPREAD``. ``None`` (the default, used by
            existing direct callers/tests) disables the fade entirely --
            every revenue-first DCF call in this function then discounts
            at a flat cohort rate exactly as before this parameter
            existed. Each scenario still STARTS its fade from its own
            14/12/10 (bear/base/bull) cohort rate; only the fade's mature
            TARGET is shared across scenarios.
        sector_capex_sales: The matched Damodaran sector's Cap Ex/Sales ratio
            (WP6), computed once by the caller (``_run_valuation``) from
            ``damodaran.sector_medians(...)["capex_sales"]``, and threaded
            into :func:`_maintenance_adjusted_margin` as its maintenance-
            CapEx floor. ``None`` (the default, used by existing direct
            callers/tests) keeps that floor at the flat 5% default exactly
            as before this parameter existed.

    Returns:
        A ``(detail, notes)`` tuple. ``detail`` matches SPEC Sec.3.4's
        ``hyper_growth_detail`` shape, or ``None`` if the mode couldn't be
        built at all (missing revenue/shares/realized growth, or every
        scenario failed). ``notes`` are Turkish strings the caller should
        fold into the top-level ``notes`` list (also echoed into
        ``detail["notes"]`` when ``detail`` is not ``None``). ``detail``
        also carries ``mature_discount_rate`` (rounded to 4 decimals, or
        ``None`` when the fade is inactive) so the fade target is visible
        to downstream reporting.
    """
    notes: List[str] = []
    try:
        latest_fy = resolve_fundamental_fy(metrics)
        revenue_series = to_annual_series(normalized, "Revenue")
        latest_revenue = revenue_series.get(latest_fy) if latest_fy is not None else None
        if latest_revenue is None or latest_revenue <= 0 or not shares or shares <= 0:
            notes.append(
                "Hiper-büyüme modu tetiklendi ancak revenue-first DCF için gerekli veriler "
                "(son yılın geliri veya hisse sayısı) eksik; standart değerleme kullanılıyor."
            )
            return None, notes

        realized_cagr = metrics.get("revenue_cagr_5y")
        if realized_cagr is None:
            realized_cagr = metrics.get("revenue_cagr_3y")
        if realized_cagr is None:
            notes.append(
                "Hiper-büyüme modu tetiklendi ancak gerçekleşen gelir büyümesi (CAGR) eksik; "
                "standart değerleme kullanılıyor."
            )
            return None, notes

        if terminal_growth != _HYPER_TERMINAL_GROWTH:
            notes.append(
                f"Uçtaki (terminal) büyüme risksiz getiri oranına bağlandı (%{terminal_growth * 100:.1f}, "
                "üst sınır %4); hiper-büyüme kohortu için ayrı düşük terminal oran kullanılmıyor."
            )

        # --- Start-growth anchor (F4): blend the realized multi-year CAGR
        # with the latest single-year YoY growth rather than anchoring on
        # the CAGR alone -- a smoothed 5y/3y CAGR can lag a recent,
        # material deceleration (or acceleration) that a hyper-grower's
        # own latest fiscal year already shows.
        prev_revenue = revenue_series.get(latest_fy - 1) if latest_fy is not None else None
        latest_yoy = None
        if latest_revenue > 0 and prev_revenue is not None and prev_revenue > 0:
            latest_yoy = latest_revenue / prev_revenue - 1

        if latest_yoy is not None:
            growth_anchor = 0.5 * realized_cagr + 0.5 * latest_yoy
            notes.append(
                "Hiper-büyüme başlangıç büyümesi, gerçekleşen 5y/3y CAGR ile son yılın büyümesinin "
                "harmanı olarak hesaplandı."
            )
        else:
            growth_anchor = realized_cagr

        ratio_by_fy = {r["fy"]: r for r in (ratios or []) if r.get("fy") is not None}
        gross_margin = (ratio_by_fy.get(latest_fy) or {}).get("gross_margin")
        gm = gross_margin if (gross_margin is not None and gross_margin > 0) else None

        fcf = metrics.get("fcf")
        sbc_latest = to_annual_series(normalized, "SBC").get(latest_fy) if latest_fy is not None else None
        current_margin = (fcf - (sbc_latest or 0.0)) / latest_revenue if fcf is not None else 0.0

        # --- Maintenance/growth CapEx split (Roadmap Madde 1 / SPEC Sec.3.6):
        # for capex-heavy hyper-growers (data-center builders etc.) compute a
        # growth-CapEx-relieved "operating" margin. This is DELIBERATELY NOT
        # fed into the headline scenarios (the reverse-DCF review showed that
        # relieving growth CapEx while revenue still compounds books the
        # revenue ramp but charges the CapEx funding it nowhere -- a
        # one-directional over-valuation). The headline keeps using today's
        # actual (unrelieved) FCF margin, so capex-heavy names still suppress
        # honestly; the relieved value is reported separately below as an
        # explicitly-labeled AGGRESSIVE UPSIDE (never the headline).
        ops_margin, capex_normalization = _maintenance_adjusted_margin(
            normalized, metrics, current_margin, sector_capex_sales
        )

        # Finding 3: the mature-target floor uses the ACTUAL (unrelieved)
        # current margin, so a relieved margin can never leak into the
        # terminal margin. (current_margin here is already the raw margin.)
        target_base = _hyper_target_base(gm, current_margin)
        if gm is None:
            if current_margin > 0:
                notes.append(
                    "Hiper-büyüme hedef olgun FCF marjı için brüt marj verisi eksik; "
                    f"%{_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f} varsayılan tavan kullanıldı, "
                    f"bugünkü FCF marjına (%{current_margin * 100:.0f}) tabanlandı."
                )
            else:
                notes.append(
                    "Hiper-büyüme hedef olgun FCF marjı için brüt marj verisi eksik; "
                    f"%{_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f} varsayılan tavan kullanıldı."
                )

        # WP4: target_base is no longer clamped to _HYPER_TARGET_BASE_CAP --
        # when the (pre-per-scenario-scaling) base value exceeds that
        # reference threshold, flag it instead of silently truncating a
        # genuinely high-margin business's economics.
        if target_base > _HYPER_TARGET_BASE_CAP:
            notes.append(
                f"Hiper-büyüme hedef olgun FCF marjı %{target_base * 100:.0f}, %30 referans eşiğinin "
                "üzerinde (kaynak: brüt marj × 0.5); yüksek marj varsayımı bilinçlidir — sabit tavanla "
                "kırpılmadı."
            )
            target_margin_flag = "above_reference"
        else:
            target_margin_flag = None

        # SBC is now expensed directly in current_margin/target margins (F2).
        # Projected dilution must therefore exclude the SBC-driven share
        # issuance already embedded in ``shares_yoy`` -- otherwise the same
        # SBC cost is charged twice (once as margin drag, once as per-share
        # dilution). ``_non_sbc_dilution`` nets that out when market_cap is
        # available; only the remaining non-SBC dilution passes through,
        # clamped to ``_HYPER_DILUTION_CAP``.
        annual_dilution, dilution_note, sbc_dilution_excluded = _non_sbc_dilution(
            metrics, normalized, latest_fy
        )
        if dilution_note:
            notes.append(dilution_note)

        extras = extras or {}
        per_scenario_extras = extras.get("per_scenario") or {}
        tam_usd = extras.get("tam_usd")
        if not _is_number(tam_usd) or tam_usd <= 0:
            tam_usd = None

        raw_start_growth = {
            "bear": min(growth_anchor, _HYPER_START_GROWTH_CAP) * 0.6,
            "base": min(growth_anchor, _HYPER_START_GROWTH_CAP),
            "bull": min(growth_anchor * 1.2, _HYPER_START_GROWTH_CAP),
        }
        raw_target = {"bear": target_base * 0.7, "base": target_base, "bull": target_base * 1.2}

        start_growth_by_scenario = {}
        target_by_scenario = {}
        steady_state_by_scenario = {}
        probabilities = {}
        target_margin_overridden = {}

        for key in _SCENARIO_KEYS:
            scenario_extras = per_scenario_extras.get(key) or {}

            target = raw_target[key]
            if gm is not None:
                target = min(target, gm)
            override_target = scenario_extras.get("target_fcf_margin")
            if _is_number(override_target):
                target = override_target
                target_margin_overridden[key] = True
            else:
                target_margin_overridden[key] = False

            steady_state_year = _HYPER_DEFAULT_STEADY_STATE_YEAR
            override_steady = scenario_extras.get("steady_state_year")
            if isinstance(override_steady, int) and not isinstance(override_steady, bool) and override_steady >= 1:
                steady_state_year = override_steady

            prob = _HYPER_DEFAULT_PROBABILITIES[key]
            override_prob = scenario_extras.get("probability")
            if _is_number(override_prob) and 0.0 <= override_prob <= 1.0:
                prob = override_prob

            start_growth_by_scenario[key] = raw_start_growth[key]
            target_by_scenario[key] = target
            steady_state_by_scenario[key] = steady_state_year
            probabilities[key] = prob

        # --- WP3: hyper-grower discount-rate fade (Damodaran fade) --------
        # A revenue-first DCF already fades revenue growth and FCF margin
        # toward mature steady-state values (F4 above), but discounting
        # every year at a fixed cohort rate (14/12/10 bear/base/bull) is
        # internally inconsistent with that: the cash flows mature while
        # the risk price never does, and since most of a hyper-grower's
        # value sits in the far years plus the terminal value, a
        # permanently-elevated rate systematically crushes it. When the
        # caller (`_run_valuation`) supplies a `mature_discount_rate`
        # (derived from the CAPM-aware, already-clamped base assumptions'
        # discount rate), every revenue-first DCF call below fades from its
        # own scenario's cohort rate down to that shared mature rate by
        # each scenario's own `steady_state_year` (`revenue_dcf._discount_
        # path`), and the terminal value discounts at the mature rate
        # (itself a mature-firm perpetuity). `None` (no base discount rate
        # available) leaves every call flat, exactly as before this
        # parameter existed.
        if mature_discount_rate is not None:
            notes.append(
                "Hiper-büyüme iskonto oranı sabit tutulmadı: nakit akışları olgunlaştıkça her senaryonun "
                "kendi kohort iskonto oranından (düşüş: bear %"
                f"{_HYPER_DISCOUNT_RATE_BY_SCENARIO['bear'] * 100:.0f}, baz %"
                f"{_HYPER_DISCOUNT_RATE_BY_SCENARIO['base'] * 100:.0f}, boğa %"
                f"{_HYPER_DISCOUNT_RATE_BY_SCENARIO['bull'] * 100:.0f}) olgun özkaynak maliyetine "
                f"(%{mature_discount_rate * 100:.1f}) doğru kendi durağan-durum yılına (baz senaryoda "
                f"{steady_state_by_scenario['base']}. yıl) kadar lineer olarak indirildi (Damodaran fade)."
            )

        if target_margin_overridden["base"]:
            target_margin_source = "LLM/kullanıcı tarafından sağlanan hedef marj (hyper_growth_extras)"
        else:
            # Recompute the ceiling (not the floored target_base itself) just
            # to phrase the source string correctly -- did today's positive
            # FCF margin actually raise target_base above the ceiling? WP4:
            # this must match _hyper_target_base's own (now uncapped)
            # ceiling, else floored_by_current_margin would be computed
            # against a stale, capped ceiling.
            ceiling = gm * 0.5 if gm is not None else _HYPER_TARGET_MARGIN_CEILING_FALLBACK
            floored_by_current_margin = current_margin > 0 and current_margin > ceiling
            if gm is not None:
                if floored_by_current_margin:
                    target_margin_source = (
                        f"brüt marj %{gm * 100:.0f} × 0.5, bugünkü FCF marjına tabanlanmış"
                    )
                else:
                    target_margin_source = "brüt marj × 0.5"
            else:
                if floored_by_current_margin:
                    target_margin_source = (
                        f"brüt marj yok: %{_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f} varsayılan tavan, "
                        f"bugünkü FCF marjına (%{current_margin * 100:.0f}) tabanlanmış"
                    )
                else:
                    target_margin_source = (
                        f"brüt marj yok: %{_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f} varsayılan tavan"
                    )

        # --- Financing shares: derived from the base scenario's own
        # (financing_shares=0) fcf_path -- cumulative negative-FCF years,
        # undiscounted -- then reused for all three scenarios (Sec.3.2).
        try:
            prelim_base = revenue_dcf.revenue_first_dcf(
                latest_revenue, start_growth_by_scenario["base"], terminal_growth,
                _HYPER_DISCOUNT_RATE_BY_SCENARIO["base"], current_margin, target_by_scenario["base"],
                steady_state_by_scenario["base"], shares, annual_dilution, 0.0,
                mature_discount_rate=mature_discount_rate,
            )
        except ValueError as exc:
            notes.append(f"Hiper-büyüme revenue-first DCF (baz senaryo) hesaplanamadı: {exc}")
            return None, notes

        burn = sum(min(fcf_t, 0.0) for fcf_t in prelim_base["fcf_path"])
        if price is not None and price > 0:
            financing_shares = abs(burn) / price
        else:
            financing_shares = 0.0
            if burn < 0:
                notes.append(
                    "Fiyat eksik olduğu için hiper-büyüme finansman (dilution) hisseleri hesaplanamadı; "
                    "finansman hissesi 0 varsayıldı."
                )

        scenarios_detail = {}
        for key in _SCENARIO_KEYS:
            start_growth = start_growth_by_scenario[key]
            target = target_by_scenario[key]
            steady_state_year = steady_state_by_scenario[key]
            discount_rate = _HYPER_DISCOUNT_RATE_BY_SCENARIO[key]

            try:
                result = revenue_dcf.revenue_first_dcf(
                    latest_revenue, start_growth, terminal_growth, discount_rate, current_margin,
                    target, steady_state_year, shares, annual_dilution, financing_shares,
                    mature_discount_rate=mature_discount_rate,
                )
            except ValueError as exc:
                scenarios_detail[key] = {
                    "per_share": None, "lo": None, "hi": None,
                    "start_growth": round(start_growth, 4), "target_fcf_margin": round(target, 4),
                    "final_year_revenue": None, "revenue_multiple": None,
                }
                notes.append(f"{key.capitalize()} hiper-büyüme senaryosu hesaplanamadı: {exc}")
                continue

            per_share = round(result["per_share"], 2)
            lo, hi, used_fallback = _hyper_scenario_band(
                latest_revenue, start_growth, terminal_growth, discount_rate, current_margin,
                target, steady_state_year, shares, annual_dilution, financing_shares, per_share,
                mature_discount_rate=mature_discount_rate,
            )
            if used_fallback:
                notes.append(
                    f"{key.capitalize()} hiper-büyüme senaryosu için duyarlılık bandı hesaplanamadı; "
                    "nokta tahminin +/-%10'u fallback olarak kullanıldı."
                )
            scenarios_detail[key] = {
                "per_share": per_share, "lo": lo, "hi": hi,
                "start_growth": round(start_growth, 4), "target_fcf_margin": round(target, 4),
                "terminal_growth": round(terminal_growth, 4),
                "final_year_revenue": result["final_year_revenue"], "revenue_multiple": result["revenue_multiple"],
            }

        base_cell = scenarios_detail.get("base")
        if base_cell is None or base_cell.get("revenue_multiple") is None:
            notes.append(
                "Hiper-büyüme baz senaryosu hesaplanamadığı için varış noktası (arrival) bayrağı "
                "belirlenemedi; standart değerleme kullanılıyor."
            )
            return None, notes

        # --- Non-credible negative valuation guard --------------------------
        # For capex-heavy hyper-growers the base scenario's discounted early-
        # year cash burn (revenue x a deeply negative current FCF margin, driven
        # by growth CapEx that is many multiples of revenue) can exceed the
        # positive terminal value, giving a negative equity value -> per_share
        # <= 0. A DCF that values a still-financeable going concern below $0 is
        # not a usable number: keep the mode detected (scenarios stay in the
        # detail for transparency) but flag it suppressed so the caller drops
        # the DCF fair-value range and its triangulation vote instead of
        # publishing a negative band.
        base_per_share = base_cell.get("per_share")
        suppressed = base_per_share is not None and base_per_share <= 0
        suppressed_reason = None
        if suppressed:
            suppressed_reason = (
                "Şirket, gelirinin çok üzerinde büyüme yatırımı (CapEx) yaptığı için bugünkü serbest "
                "nakit akışı marjı aşırı negatif; revenue-first DCF baz senaryosu negatif özkaynak "
                "değeri (hisse başı ≤ $0) üretti. Faal ve sermaye toplayabilen bir şirket için "
                "kullanılabilir bir değer olmadığından DCF manşet aralığı ve üçgenleme oyu devre dışı "
                "bırakıldı; capex yoğunluğu normalleşene dek revenue-first DCF güvenilir değil."
            )
            notes.append(suppressed_reason)

        # --- Aggressive capex-normalized UPSIDE (Sec.3.6) -------------------
        # For a capex-heavy filer, also compute a base-scenario value off the
        # growth-CapEx-relieved margin -- reported as an explicitly-labeled
        # AGGRESSIVE UPSIDE, never the headline. The headline scenarios above
        # already used the actual (unrelieved) margin, so this does not change
        # the published fair value or the suppression decision; it only tells
        # the reader what an optimistic "growth CapEx normalizes" view implies.
        # Uses the same base start-growth / target / discount rate / dilution
        # / financing shares as the headline base scenario -- only the
        # starting margin differs (ops_margin vs current_margin).
        if capex_normalization is not None:
            try:
                up = revenue_dcf.revenue_first_dcf(
                    latest_revenue, start_growth_by_scenario["base"], terminal_growth,
                    _HYPER_DISCOUNT_RATE_BY_SCENARIO["base"], ops_margin, target_by_scenario["base"],
                    steady_state_by_scenario["base"], shares, annual_dilution, financing_shares,
                    mature_discount_rate=mature_discount_rate,
                )
                up_ps = round(up["per_share"], 2)
                up_lo, up_hi, _up_fallback = _hyper_scenario_band(
                    latest_revenue, start_growth_by_scenario["base"], terminal_growth,
                    _HYPER_DISCOUNT_RATE_BY_SCENARIO["base"], ops_margin, target_by_scenario["base"],
                    steady_state_by_scenario["base"], shares, annual_dilution, financing_shares, up_ps,
                    mature_discount_rate=mature_discount_rate,
                )
                capex_normalization["upside_per_share"] = up_ps
                capex_normalization["upside_lo"] = up_lo
                capex_normalization["upside_hi"] = up_hi
            except ValueError:
                capex_normalization["upside_per_share"] = None
                capex_normalization["upside_lo"] = None
                capex_normalization["upside_hi"] = None
            notes.append(
                "CapEx-yoğun hiper-büyüme: bugünkü serbest nakit akışı büyük büyüme CapEx'iyle bastırıldığı "
                "için manşet DCF (ham marjla) güvenilir değil ve devre dışı bırakıldı. Ayrıca büyüme CapEx'i "
                "bakım CapEx'inden (≈ D&A, gelirin en az %5'i tabanıyla) ayrılarak AGRESİF BİR ÜST-SENARYO "
                f"(baz ${capex_normalization.get('upside_per_share')}/hisse) hesaplandı — bu MANŞET DEĞİL, "
                "yalnızca capex normalleşirse ima edilen iyimser değeri gösterir. Not: bu üst-senaryo, geliri "
                "büyütürken o büyümeyi finanse eden CapEx'i tam yansıtmaz, bu yüzden yukarı-yanlıdır."
            )

        # --- Prob-weighted expected value: skip failed scenarios and
        # renormalize the surviving probabilities (Sec.3.3).
        weighted_sum = 0.0
        total_prob = 0.0
        for key in _SCENARIO_KEYS:
            cell = scenarios_detail.get(key) or {}
            if cell.get("per_share") is None:
                continue
            weighted_sum += probabilities[key] * cell["per_share"]
            total_prob += probabilities[key]
        expected_value = round(weighted_sum / total_prob, 2) if total_prob > 0 else None

        # --- Arrival-point flag: revenue-multiple thresholds, overridden by
        # TAM-share thresholds whenever tam_usd is known (Sec.3.3).
        multiple = base_cell["revenue_multiple"]
        if multiple <= _HYPER_ARRIVAL_AGGRESSIVE_MULTIPLE:
            arrival_flag = "makul"
        elif multiple <= _HYPER_ARRIVAL_EXTREME_MULTIPLE:
            arrival_flag = "agresif"
        else:
            arrival_flag = "asiri_agresif"
        if arrival_flag != "makul":
            notes.append(
                f"Hiper-büyüme varış noktası: baz senaryoda gelir 10 yılda {multiple:.1f} katına çıkıyor "
                f"({arrival_flag})."
            )

        tam_share = None
        if tam_usd is not None:
            tam_share = base_cell["final_year_revenue"] / tam_usd
            if tam_share > _HYPER_TAM_SHARE_INVALID:
                arrival_flag = "gecersiz"
                notes.append("Hiper-büyüme varış noktası TAM'ın %60'ını aşıyor; revizyon gerekli.")
            elif tam_share > _HYPER_TAM_SHARE_AGGRESSIVE:
                arrival_flag = "agresif"
                notes.append(f"Hiper-büyüme varış noktası TAM'ın %{tam_share * 100:.0f}'ini kullanıyor (agresif).")
            else:
                arrival_flag = "makul"

        # --- Implied expectations (base discount/margin/steady_state; Sec.3.3).
        base_discount_rate = _HYPER_DISCOUNT_RATE_BY_SCENARIO["base"]
        base_target = target_by_scenario["base"]
        base_steady_state_year = steady_state_by_scenario["base"]
        base_start_growth = start_growth_by_scenario["base"]

        implied_growth = revenue_dcf.implied_start_growth(
            price, latest_revenue, terminal_growth, base_discount_rate, current_margin,
            base_target, base_steady_state_year, shares, annual_dilution, financing_shares,
            mature_discount_rate=mature_discount_rate,
        )
        implied_revenue_10y = None
        implied_revenue_multiple = None
        if implied_growth is not None:
            try:
                implied_projection = revenue_dcf.revenue_first_dcf(
                    latest_revenue, implied_growth, terminal_growth, base_discount_rate, current_margin,
                    base_target, base_steady_state_year, shares, annual_dilution, financing_shares,
                    mature_discount_rate=mature_discount_rate,
                )
                implied_revenue_10y = implied_projection["final_year_revenue"]
                implied_revenue_multiple = implied_projection["revenue_multiple"]
            except ValueError:
                implied_revenue_10y = None
                implied_revenue_multiple = None
        else:
            notes.append(
                "Hiper-büyüme: fiyatın ima ettiği başlangıç büyüme oranı hesaplanamadı "
                "(fiyat, makul büyüme aralığının dışında bir beklenti ima ediyor olabilir)."
            )

        implied_margin = revenue_dcf.implied_target_margin(
            price, latest_revenue, base_start_growth, terminal_growth, base_discount_rate,
            current_margin, base_steady_state_year, shares, annual_dilution, financing_shares,
            mature_discount_rate=mature_discount_rate,
        )

        implied_tam_share = (
            implied_revenue_10y / tam_usd if (implied_revenue_10y is not None and tam_usd is not None) else None
        )

        detail = {
            "reasons": list(hyper_reasons or []),
            "scenarios": scenarios_detail,
            "probabilities": probabilities,
            "expected_value": expected_value,
            "arrival_flag": arrival_flag,
            "tam_usd": tam_usd,
            "implied": {
                "growth": implied_growth,
                "revenue_10y": implied_revenue_10y,
                "revenue_multiple": implied_revenue_multiple,
                "steady_state_margin": implied_margin,
                "tam_share": implied_tam_share,
            },
            "target_margin_source": target_margin_source,
            "target_margin_flag": target_margin_flag,
            "target_margin_pct": round(target_base, 4),
            "capex_normalization": capex_normalization,
            "annual_dilution": round(annual_dilution, 4),
            "sbc_dilution_excluded": round(sbc_dilution_excluded, 4),
            "suppressed": suppressed,
            "suppressed_reason": suppressed_reason,
            "mature_discount_rate": round(mature_discount_rate, 4) if mature_discount_rate is not None else None,
            "notes": list(notes),
        }
        return detail, notes
    except Exception:  # noqa: BLE001 - never let a hyper-grower bug break the standard valuation.
        logger.warning("_build_hyper_growth: unexpected error; degrading to standard valuation.", exc_info=True)
        notes.append("Hiper-büyüme modu beklenmeyen bir hatayla karşılaştı; standart değerleme kullanılıyor.")
        return None, notes


def _mature_current_margin(normalized: dict, metrics: dict) -> float:
    """Current (today's) FCF margin anchor for the mature revenue-first DCF,
    smoothed as the median of the last 3 fiscal years' SBC-adjusted FCF
    margin rather than a single year (reviewer Finding 6): a lone working-
    capital swing in the latest fiscal year shouldn't set the anchor the
    whole fade projection starts from.

    Per-year margin: ``(OCF - CapEx - SBC) / Revenue``, treating a missing
    SBC as ``0.0`` -- same SBC-as-expense convention as
    ``_sbc_adjusted_fcf_by_fy``/``_build_hyper_growth``'s own
    ``current_margin``.

    Returns:
        The median margin across the latest fiscal year and the two prior
        ones (only years with usable revenue/OCF/CapEx data are counted),
        or ``0.0`` (never ``None``) when no fiscal year has usable data --
        ``revenue_first_dcf`` treats ``current_margin`` as a plain
        (possibly zero) starting point, not an optional field.
    """
    fy = resolve_fundamental_fy(metrics)
    if fy is None:
        return 0.0

    revenue_series = to_annual_series(normalized, "Revenue")
    ocf_series = to_annual_series(normalized, "OperatingCashFlow")
    capex_series = to_annual_series(normalized, "CapEx")
    sbc_series = to_annual_series(normalized, "SBC")

    margins = []
    for y in (fy, fy - 1, fy - 2):
        revenue = revenue_series.get(y)
        ocf = ocf_series.get(y)
        capex = capex_series.get(y)
        if revenue is None or revenue <= 0 or ocf is None or capex is None:
            continue
        sbc = sbc_series.get(y) or 0.0
        margins.append((ocf - capex - sbc) / revenue)

    if not margins:
        return 0.0
    return statistics.median(margins)


def _mature_target_fcf_margin(normalized: dict, metrics: dict, ratios: list) -> Optional[float]:
    """Mature-state target FCF margin for the revenue-first DCF (reviewer
    Findings 5-6), the smaller of two independent, data-derived anchors:

    - **op-anchor:** the median of every fiscal year's positive operating
      margin (``OperatingIncome / Revenue``), converted to a NOPAT-based FCF
      margin proxy: ``op_margin * (1 - _MATURE_TAX_ASSUMPTION) *
      _MATURE_REINVEST_HAIRCUT``. ``None`` if no fiscal year has a positive
      operating margin (missing ``OperatingIncome`` data).
    - **hist-anchor:** ``_MATURE_HIST_UPLIFT`` times the single best
      historical raw FCF margin (``(OCF - CapEx) / Revenue``, positive
      years only) -- a ceiling derived from the filer's own best-ever cash
      conversion. ``None`` if no fiscal year has a positive raw FCF margin.

    The target is ``min(nopat, hist_anchor)`` over whichever of
    ``nopat``/``hist_anchor`` are available -- ``None`` only when BOTH are
    unavailable (the method can't be built at all without at least one
    anchor). WP4: this is no longer additionally clamped to
    ``_MATURE_TARGET_CAP`` here -- that constant is now a reporting-only
    flag threshold the caller (``_build_mature_revenue_dcf``) compares this
    function's return value against, attaching a note/flag instead of the
    value being silently truncated. Finally floored at the current
    (SBC-adjusted, 3-year-median) FCF margin via :func:`_mature_current_margin` whenever
    that figure is positive, mirroring ``_hyper_target_base``'s
    current-margin floor: a filer already earning more than the computed
    "mature" ceiling today must never be modeled as if its margin falls.

    Returns:
        The target FCF margin (decimal fraction), or ``None`` if neither
        anchor is available. Never raises (only reads dict/list data).
    """
    revenue_series = to_annual_series(normalized, "Revenue")
    op_income_series = to_annual_series(normalized, "OperatingIncome")
    ocf_series = to_annual_series(normalized, "OperatingCashFlow")
    capex_series = to_annual_series(normalized, "CapEx")

    op_margins = [
        op_income_series[y] / revenue_series[y]
        for y in op_income_series
        if op_income_series.get(y) is not None and op_income_series[y] > 0
        and revenue_series.get(y) is not None and revenue_series[y] > 0
    ]
    nopat = None
    if op_margins:
        op_margin = statistics.median(op_margins)
        nopat = op_margin * (1 - _MATURE_TAX_ASSUMPTION) * _MATURE_REINVEST_HAIRCUT

    hist_margins = [
        (ocf_series[y] - capex_series[y]) / revenue_series[y]
        for y in revenue_series
        if revenue_series.get(y) is not None and revenue_series[y] > 0
        and ocf_series.get(y) is not None and capex_series.get(y) is not None
        and (ocf_series[y] - capex_series[y]) > 0
    ]
    hist_anchor = None
    if hist_margins:
        hist_anchor = max(hist_margins) * _MATURE_HIST_UPLIFT

    if nopat is None and hist_anchor is None:
        return None

    target = min(c for c in (nopat, hist_anchor) if c is not None)

    current_margin = _mature_current_margin(normalized, metrics)
    if current_margin > 0:
        target = max(target, current_margin)
    return target


def _mature_start_growth(metrics: dict, normalized: dict) -> Optional[float]:
    """Blended start-growth anchor for the mature revenue-first DCF,
    mirroring the hyper-grower F4 pattern (``_build_hyper_growth``'s own
    ``growth_anchor``): the realized multi-year revenue CAGR (5y, falling
    back to 3y) blended 50/50 with the latest single fiscal year's revenue
    YoY growth when both are available -- a smoothed CAGR alone can lag a
    recent, material deceleration a mature-but-still-growing filer's latest
    fiscal year already shows.

    Returns:
        The blended (or CAGR-only) start growth rate, or ``None`` if the
        realized CAGR itself is unavailable (the method can't be built at
        all without some realized-growth reference).
    """
    realized = metrics.get("revenue_cagr_5y")
    if realized is None:
        realized = metrics.get("revenue_cagr_3y")
    if realized is None:
        return None

    fy = resolve_fundamental_fy(metrics)
    revenue_series = to_annual_series(normalized, "Revenue")
    latest_revenue = revenue_series.get(fy) if fy is not None else None
    prev_revenue = revenue_series.get(fy - 1) if fy is not None else None

    latest_yoy = None
    if latest_revenue is not None and latest_revenue > 0 and prev_revenue is not None and prev_revenue > 0:
        latest_yoy = latest_revenue / prev_revenue - 1

    if latest_yoy is not None:
        return 0.5 * realized + 0.5 * latest_yoy
    return realized


#: Fallback note used whenever ``_build_mature_revenue_dcf`` bails out early
#: -- always ends the same way so callers/readers know what happens next
#: (falls back to the EPV headline, which is already computed by the time
#: this is attempted, or the raw FCF-DCF if EPV itself isn't available).
_MATURE_FALLBACK_SUFFIX = "kazanç-gücü (EPV) çapası veya ham FCF-DCF kullanılıyor."


def _build_mature_revenue_dcf(
    assumptions: dict, normalized: dict, metrics: dict, ratios: list, price: Optional[float], shares: Optional[float]
) -> "tuple[Optional[dict], List[str]]":
    """Build the mature, FCF-suppressed-but-growing revenue-first DCF detail.

    This is a second growth-inclusive alternative to the zero-growth EPV
    anchor (Sec.8a) for mature filers whose FCF is suppressed by heavy
    growth investment while they still have real, realized top-line growth
    left (the canonical Amazon shape) -- reuses the same revenue-first
    engine as the hyper-grower mode (``revenue_dcf.revenue_first_dcf``,
    ``_hyper_scenario_band``) but with a much shorter fade
    (:data:`_MATURE_STEADY_STATE_YEAR` = 7, not 10) and a data-derived
    mature margin (WP4: no longer clamped to :data:`_MATURE_TARGET_CAP`;
    that 15% is now only a reporting/flag reference threshold, just like the
    hyper-grower path's 30% -- both are flags, not applied ceilings) -- this
    method exists for filers that are already large and already profitable,
    not a hyper-grower still finding its steady-state economics.

    Unlike hyper-grower mode, ``start_growth`` is NOT scaled per scenario:
    it's the same realized growth figure (:func:`_mature_start_growth`) in
    every scenario -- the realized growth itself isn't a per-scenario
    assumption here, only the discount rate and mature target margin are
    (Sec below, reviewer Finding 4). A growth gate (reviewer Finding 2)
    guards against building this at all for filers without genuine realized
    growth: a realized start growth below :data:`_MATURE_REV_DCF_MIN_GROWTH`,
    or at/below the base scenario's own terminal growth rate (nothing left
    to fade), degrades this to ``(None, notes)`` so the caller falls back to
    the EPV/raw-FCF-DCF headline instead of fabricating a growth story that
    isn't there.

    Never raises: any missing/invalid input or
    ``revenue_first_dcf``/``_hyper_scenario_band`` failure degrades to
    ``(None, notes)`` with a Turkish note, mirroring ``_build_hyper_growth``.

    Args:
        assumptions: The phase-1 (already clamped) bear/base/bull assumption
            dict -- only ``discount_rate``/``terminal_growth`` per scenario
            are consulted (the growth story itself comes from realized
            data, not the assumptions pipeline).
        normalized: Used to look up the latest annual ``Revenue``,
            ``OperatingIncome``, ``OperatingCashFlow``, ``CapEx``, ``SBC``.
        metrics: Used for ``revenue_cagr_5y``/``_3y`` and to resolve the
            fiscal year via ``resolve_fundamental_fy``.
        ratios: Accepted for signature symmetry with ``_build_earnings_power``
            (unused directly; the margin anchors are derived from
            ``normalized`` series, not ``ratios``).
        price: Unused here (accepted for signature symmetry / future use --
            the reverse-DCF override that consumes this method's output is
            computed by the caller, not this function).
        shares: Base share count.

    Returns:
        A ``(detail, notes)`` tuple. ``detail`` is ``None`` if the method
        can't be built at all (missing revenue/shares/realized growth, the
        growth gate rejects it, the target margin can't be derived, or
        every scenario failed), else a dict with ``scenarios`` (bear/base/
        bull ``{"per_share", "lo", "hi", "start_growth", "target_fcf_margin",
        "terminal_growth", "discount_rate"}``), ``start_growth``,
        ``target_margin_base``, ``current_margin``, and
        ``steady_state_year``. NOTE (WP7): the caller (``_run_valuation``)
        additionally mutates this dict with a ``growth_vs_floor`` key
        (``"adds"``/``"destroys"``/``None``, see :func:`_growth_vs_floor`)
        after this function returns -- it is not set here.
    """
    notes: List[str] = []
    try:
        fy = resolve_fundamental_fy(metrics)
        revenue_series = to_annual_series(normalized, "Revenue")
        revenue0 = revenue_series.get(fy) if fy is not None else None
        if revenue0 is None or revenue0 <= 0 or not shares or shares <= 0:
            notes.append(
                "Olgun revenue-first DCF için gerekli veriler (son yılın geliri veya hisse sayısı) eksik; "
                f"{_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        start_growth = _mature_start_growth(metrics, normalized)
        if start_growth is None:
            notes.append(
                f"Olgun revenue-first DCF için gerçekleşen gelir büyümesi (CAGR) hesaplanamadı; "
                f"{_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        base_terminal_growth = (assumptions.get("base") or {}).get("terminal_growth")
        if not _is_number(base_terminal_growth):
            notes.append(
                f"Olgun revenue-first DCF için baz terminal büyüme oranı eksik; {_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        # --- Growth gate (reviewer Finding 2): this method models a fading
        # GROWTH path -- it only makes sense for filers with genuine
        # realized growth left to fade. A realized growth rate below
        # _MATURE_REV_DCF_MIN_GROWTH, or at/below the terminal growth rate
        # itself (nothing left to fade), degrades to the EPV/raw-FCF-DCF
        # headline instead of fabricating a growth story that isn't there.
        if start_growth < _MATURE_REV_DCF_MIN_GROWTH or start_growth <= base_terminal_growth:
            notes.append(
                f"Gerçekleşen gelir büyümesi (%{start_growth * 100:.1f}) olgun revenue-first DCF için yetersiz "
                f"(< %{_MATURE_REV_DCF_MIN_GROWTH * 100:.0f} veya terminal büyümenin altında); "
                f"{_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        target_base = _mature_target_fcf_margin(normalized, metrics, ratios)
        if target_base is None:
            notes.append(
                "Olgun revenue-first DCF için hedef olgun FCF marjı hesaplanamadı (operasyon marjı ve "
                f"tarihsel FCF marjı verisi eksik); {_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        # WP4: target_base is no longer clamped to _MATURE_TARGET_CAP -- flag
        # it instead of silently truncating when the NOPAT/historical-FCF
        # anchors genuinely derive a higher mature margin.
        if target_base > _MATURE_TARGET_CAP:
            notes.append(
                f"Olgun hedef FCF marjı %{target_base * 100:.0f}, %15 referans eşiğinin üzerinde (NOPAT ve "
                "tarihsel-FCF çapalarından türedi); bilinçli — sabit tavan uygulanmadı."
            )
            target_margin_flag = "above_reference"
        else:
            target_margin_flag = None

        current_margin = _mature_current_margin(normalized, metrics)
        steady_state_year = _MATURE_STEADY_STATE_YEAR

        scenarios: dict = {}
        for key in _SCENARIO_KEYS:
            scenario_assumptions = assumptions.get(key) or {}
            discount_rate = scenario_assumptions.get("discount_rate")
            terminal_growth = scenario_assumptions.get("terminal_growth")

            if not _is_number(discount_rate) or not _is_number(terminal_growth):
                notes.append(f"{key.capitalize()} senaryosu için olgun revenue-first DCF varsayımları eksik.")
                continue
            if discount_rate <= terminal_growth:
                notes.append(
                    f"{key.capitalize()} senaryosu için iskonto oranı terminal büyüme oranından büyük değil; "
                    "senaryo atlandı."
                )
                continue

            target_margin = target_base * _MATURE_TARGET_MARGIN_SCALE[key]

            try:
                result = revenue_dcf.revenue_first_dcf(
                    revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                    target_margin, steady_state_year, shares, 0.0,
                )
            except ValueError as exc:
                notes.append(f"{key.capitalize()} senaryosu için olgun revenue-first DCF hesaplanamadı: {exc}")
                continue

            per_share = round(result["per_share"], 2)
            lo, hi, used_fallback = _hyper_scenario_band(
                revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                target_margin, steady_state_year, shares, 0.0, 0.0, per_share,
            )
            if used_fallback:
                notes.append(
                    f"{key.capitalize()} senaryosu için duyarlılık bandı hesaplanamadı; "
                    "nokta tahminin +/-%10'u fallback olarak kullanıldı."
                )

            scenarios[key] = {
                "per_share": per_share, "lo": lo, "hi": hi,
                "start_growth": round(start_growth, 4),
                "target_fcf_margin": round(target_margin, 4),
                "terminal_growth": round(terminal_growth, 4),
                "discount_rate": round(discount_rate, 4),
            }

        if not scenarios:
            notes.append(f"Olgun revenue-first DCF hiçbir senaryo için hesaplanamadı; {_MATURE_FALLBACK_SUFFIX}")
            return None, notes

        detail = {
            "scenarios": scenarios,
            "start_growth": round(start_growth, 4),
            "target_margin_base": round(target_base, 4),
            "target_margin_flag": target_margin_flag,
            "current_margin": round(current_margin, 4),
            "steady_state_year": steady_state_year,
        }
        return detail, notes
    except Exception:  # noqa: BLE001 - never let a mature-revenue-DCF bug break the standard valuation.
        logger.warning("_build_mature_revenue_dcf: unexpected error; degrading to standard valuation.", exc_info=True)
        notes.append("Olgun revenue-first DCF beklenmeyen bir hatayla karşılaştı; standart değerleme kullanılıyor.")
        return None, notes


#: Fallback note suffix for ``_build_midgrowth_revenue_dcf`` -- always ends
#: the same way so the reader knows what happens when the method bails out
#: (the filer falls back to the multiples-only headline it had before this
#: method existed).
_MIDGROWTH_FALLBACK_SUFFIX = "çarpan (multiples) bazlı değerlemeye düşülüyor."


def _build_midgrowth_revenue_dcf(
    assumptions: dict, normalized: dict, metrics: dict, ratios: list, price: Optional[float], shares: Optional[float]
) -> "tuple[Optional[dict], List[str]]":
    """Build the mid-growth, loss-making revenue-first DCF detail (Roadmap
    Madde 2 / SPEC Sec.8d).

    For ``growth_unprofitable`` filers that grow the top line at a real but
    sub-hyper rate (realized CAGR in roughly 12-20%) and therefore are NOT
    picked up by ``sector.detect_hyper_grower`` (which needs CAGR > 20%),
    this gives a revenue-first fair-value band instead of leaving them to a
    multiples-only headline. It reuses the same revenue-first engine as the
    hyper-grower / mature paths (``revenue_dcf.revenue_first_dcf``,
    ``_hyper_scenario_band``) with parameters that sit between them:

    * **fade horizon** :data:`_MIDGROWTH_STEADY_STATE_YEAR` (8) -- between
      mature's 7 and hyper's 10.
    * **mature target margin** derived from the gross-margin proxy
      (``_hyper_target_base``); WP4: no longer clamped to
      :data:`_MIDGROWTH_TARGET_CAP` (20%) -- that's now a reporting/flag
      reference threshold, not an applied ceiling, mirroring the hyper and
      mature paths -- the mature path's operating-margin/historical-FCF
      anchors degenerate for a loss-maker with no positive-margin history,
      so this borrows the hyper path's gross-margin construction instead.
    * **discount rate / terminal growth** come from the already-clamped
      per-scenario assumptions (``growth_unprofitable`` is clamped with
      ``is_unprofitable=True``, so the discount rate is already floored at
      10%), NOT the hard-coded hyper rates.
    * **dilution & financing shares** follow the hyper path (a mid-growth
      loss-maker still funds cash burn by issuing equity), unlike the mature
      path which assumes none.

    A **growth gate** (mirroring the mature path) refuses to build the method
    for a realized start growth below :data:`_MIDGROWTH_MIN_GROWTH` or at/below
    the base scenario's own terminal growth (nothing left to fade). A
    **suppression guardrail** (mirroring the hyper path) flags a
    non-credible negative base value (``per_share <= 0``) so the caller drops
    the headline back to multiples rather than publishing a negative band.

    Never raises: any missing/invalid input or ``revenue_first_dcf`` failure
    degrades to ``(None, notes)`` with a Turkish note.

    Args:
        assumptions: The phase-1 (already clamped) bear/base/bull assumption
            dict -- only ``discount_rate``/``terminal_growth`` per scenario
            are consulted (the growth story comes from realized data).
        normalized: Used for ``Revenue`` and (via helpers) ``OperatingCashFlow``/
            ``CapEx``/``Depreciation``/``SBC`` series.
        metrics: Used for ``revenue_cagr_5y``/``_3y``, ``shares_yoy`` and to
            resolve the fiscal year via ``resolve_fundamental_fy``.
        ratios: Per-FY ratio dicts (uses the latest FY's ``gross_margin``).
        price: Current market price (used to convert cumulative cash burn into
            financing shares); ``None`` degrades financing shares to 0.
        shares: Base share count.

    Returns:
        A ``(detail, notes)`` tuple. ``detail`` is ``None`` when the method
        can't be built (missing revenue/shares/growth, growth gate rejects
        it, or every scenario failed), else a dict with ``scenarios``
        (bear/base/bull ``{"per_share", "lo", "hi", "start_growth",
        "target_fcf_margin", "terminal_growth", "discount_rate"}``),
        ``start_growth``, ``target_margin_base``, ``current_margin``,
        ``steady_state_year``, ``annual_dilution``, ``financing_shares``,
        and ``suppressed`` (bool).
    """
    notes: List[str] = []
    try:
        fy = resolve_fundamental_fy(metrics)
        revenue_series = to_annual_series(normalized, "Revenue")
        revenue0 = revenue_series.get(fy) if fy is not None else None
        if revenue0 is None or revenue0 <= 0 or not shares or shares <= 0:
            notes.append(
                "Orta-büyüme revenue-first DCF için gerekli veriler (son yılın geliri veya hisse sayısı) "
                f"eksik; {_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        start_growth = _mature_start_growth(metrics, normalized)
        if start_growth is None:
            notes.append(
                "Orta-büyüme revenue-first DCF için gerçekleşen gelir büyümesi (CAGR) hesaplanamadı; "
                f"{_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        base_terminal_growth = (assumptions.get("base") or {}).get("terminal_growth")
        if not _is_number(base_terminal_growth):
            notes.append(
                f"Orta-büyüme revenue-first DCF için baz terminal büyüme oranı eksik; {_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        # --- Growth gate: this method models a fading GROWTH path, so it
        # only makes sense with a real, still-fading growth rate above the
        # 12% floor (and above terminal growth). Below that, fall back to
        # multiples rather than fabricating a growth story.
        if start_growth < _MIDGROWTH_MIN_GROWTH or start_growth <= base_terminal_growth:
            notes.append(
                f"Gerçekleşen gelir büyümesi (%{start_growth * 100:.1f}) orta-büyüme revenue-first DCF için "
                f"yetersiz (< %{_MIDGROWTH_MIN_GROWTH * 100:.0f} veya terminal büyümenin altında); "
                f"{_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        # --- Target mature FCF margin: gross-margin proxy (hyper path). WP4:
        # no longer clamped to _MIDGROWTH_TARGET_CAP (see the flag check
        # below instead). The mature path's operating-margin/historical-FCF
        # anchors need a positive-margin history a loss-maker doesn't have.
        ratio_by_fy = {r["fy"]: r for r in (ratios or []) if r.get("fy") is not None}
        gross_margin = (ratio_by_fy.get(fy) or {}).get("gross_margin")
        gm = gross_margin if (gross_margin is not None and gross_margin > 0) else None

        # --- Current (starting) FCF margin: 3-year median (loss-makers are
        # negative here). The Sec.3.6 maintenance/growth CapEx relief is
        # deliberately NOT applied here -- it produces a one-directional
        # over-valuation (see _maintenance_adjusted_margin's docstring), and
        # the mid-growth path's whole point is a defensible (not aggressive)
        # value; a capex-heavy mid-grower whose base value suppresses simply
        # falls back to multiples.
        current_margin = _mature_current_margin(normalized, metrics)

        target_base = _hyper_target_base(gm, current_margin)
        if gm is None:
            notes.append(
                "Orta-büyüme revenue-first DCF hedef olgun FCF marjı için brüt marj verisi eksik; "
                f"%{_MIDGROWTH_TARGET_CAP * 100:.0f} tavan kullanıldı."
            )

        # WP4: target_base is no longer clamped to _MIDGROWTH_TARGET_CAP --
        # flag it instead of silently truncating when the gross-margin-
        # derived value genuinely exceeds the reference threshold.
        if target_base > _MIDGROWTH_TARGET_CAP:
            notes.append(
                f"Orta-büyüme hedef olgun FCF marjı %{target_base * 100:.0f}, %20 referans eşiğinin üzerinde "
                "(brüt marj × 0.5 kaynaklı); bilinçli — sabit tavan uygulanmadı."
            )
            target_margin_flag = "above_reference"
        else:
            target_margin_flag = None

        steady_state_year = _MIDGROWTH_STEADY_STATE_YEAR

        # --- Dilution (non-SBC share-count growth only; SBC-driven issuance
        # is excluded because SBC is already expensed in the margin -- see
        # _non_sbc_dilution) and financing shares (fund cumulative burn),
        # mirroring the hyper path -- a mid-growth loss-maker still issues
        # equity.
        annual_dilution, dilution_note, sbc_dilution_excluded = _non_sbc_dilution(
            metrics, normalized, fy
        )
        if dilution_note:
            notes.append(dilution_note)

        base_assumptions = assumptions.get("base") or {}
        base_dr = base_assumptions.get("discount_rate")
        financing_shares = 0.0
        if _is_number(base_dr) and _is_number(base_terminal_growth) and base_dr > base_terminal_growth:
            try:
                prelim_base = revenue_dcf.revenue_first_dcf(
                    revenue0, start_growth, base_terminal_growth, base_dr, current_margin,
                    target_base, steady_state_year, shares, annual_dilution, 0.0,
                )
                burn = sum(min(fcf_t, 0.0) for fcf_t in prelim_base["fcf_path"])
                if price is not None and price > 0:
                    financing_shares = abs(burn) / price
                elif burn < 0:
                    notes.append(
                        "Fiyat eksik olduğu için orta-büyüme finansman (dilution) hisseleri hesaplanamadı; "
                        "finansman hissesi 0 varsayıldı."
                    )
            except ValueError:
                financing_shares = 0.0

        scenarios: dict = {}
        for key in _SCENARIO_KEYS:
            scenario_assumptions = assumptions.get(key) or {}
            discount_rate = scenario_assumptions.get("discount_rate")
            terminal_growth = scenario_assumptions.get("terminal_growth")

            if not _is_number(discount_rate) or not _is_number(terminal_growth):
                notes.append(f"{key.capitalize()} senaryosu için orta-büyüme revenue-first DCF varsayımları eksik.")
                continue
            if discount_rate <= terminal_growth:
                notes.append(
                    f"{key.capitalize()} senaryosu için iskonto oranı terminal büyüme oranından büyük değil; "
                    "senaryo atlandı."
                )
                continue

            target_margin = target_base * _MATURE_TARGET_MARGIN_SCALE[key]

            try:
                result = revenue_dcf.revenue_first_dcf(
                    revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                    target_margin, steady_state_year, shares, annual_dilution, financing_shares,
                )
            except ValueError as exc:
                notes.append(f"{key.capitalize()} senaryosu için orta-büyüme revenue-first DCF hesaplanamadı: {exc}")
                continue

            per_share = round(result["per_share"], 2)
            lo, hi, used_fallback = _hyper_scenario_band(
                revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                target_margin, steady_state_year, shares, annual_dilution, financing_shares, per_share,
            )
            if used_fallback:
                notes.append(
                    f"{key.capitalize()} senaryosu için duyarlılık bandı hesaplanamadı; "
                    "nokta tahminin +/-%10'u fallback olarak kullanıldı."
                )

            scenarios[key] = {
                "per_share": per_share, "lo": lo, "hi": hi,
                "start_growth": round(start_growth, 4),
                "target_fcf_margin": round(target_margin, 4),
                "terminal_growth": round(terminal_growth, 4),
                "discount_rate": round(discount_rate, 4),
            }

        if not scenarios:
            notes.append(
                f"Orta-büyüme revenue-first DCF hiçbir senaryo için hesaplanamadı; {_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        # --- Suppression guardrail (mirrors the hyper path): a non-credible
        # negative base value means the caller should drop back to multiples.
        base_ps = (scenarios.get("base") or {}).get("per_share")
        suppressed = base_ps is not None and base_ps <= 0
        if suppressed:
            notes.append(
                "Orta-büyüme revenue-first DCF baz senaryosu negatif özkaynak değeri (hisse başı ≤ $0) "
                f"üretti; manşet için kullanılabilir değil, {_MIDGROWTH_FALLBACK_SUFFIX}"
            )

        detail = {
            "scenarios": scenarios,
            "start_growth": round(start_growth, 4),
            "target_margin_base": round(target_base, 4),
            "target_margin_flag": target_margin_flag,
            "current_margin": round(current_margin, 4),
            "steady_state_year": steady_state_year,
            "annual_dilution": round(annual_dilution, 4),
            "sbc_dilution_excluded": round(sbc_dilution_excluded, 4),
            "financing_shares": financing_shares,
            "suppressed": suppressed,
        }
        return detail, notes
    except Exception:  # noqa: BLE001 - never let a mid-growth-revenue-DCF bug break the standard valuation.
        logger.warning("_build_midgrowth_revenue_dcf: unexpected error; degrading to standard valuation.", exc_info=True)
        notes.append(
            "Orta-büyüme revenue-first DCF beklenmeyen bir hatayla karşılaştı; standart değerleme kullanılıyor."
        )
        return None, notes


#: Turkish labels for the current-multiple fallback notes, keyed by which
#: multiple was derived.
_MULTIPLE_LABELS = {"pe": "F/K", "ps": "F/S", "pfcf": "F/FCF"}


def _derive_current_multiples(
    normalized: dict, ratios: list, metrics: dict, price: Optional[float]
) -> "tuple[dict, List[str]]":
    """Fill gaps in ``metrics``' current pe/ps/pfcf from the per-FY series.

    ``metrics.compute_metrics`` derives every current-period figure from a
    single ``latest_fy`` (the newest fiscal year across ALL series,
    including ``SharesOutstanding``). When one series has a newer fiscal
    year than another -- e.g. a filer's dei cover-page share count is more
    recent than its latest reported EPS/Revenue/FCF (see JPM) -- that
    mismatch makes ``metrics["pe"|"ps"|"pfcf"]`` all ``None`` even though
    plenty of usable historical fundamentals exist.

    This recovers each multiple independently: for ``pe``, the current
    price divided by the latest fiscal year with a positive EPS; for
    ``ps``/``pfcf``, the current price times ``metrics["shares"]`` divided
    by the latest fiscal year with a positive revenue/FCF (FCF from
    ``ratios``' per-fy figure, falling back to OperatingCashFlow - CapEx,
    mirroring ``metrics.compute_metrics``'s own fcf selection). Only fills
    slots that are still ``None`` in ``metrics`` -- never overrides an
    already-computed value. Never raises; returns ``(current, notes)``
    where ``notes`` describes which fiscal year each derived multiple used.
    """
    current = {"pe": metrics.get("pe"), "ps": metrics.get("ps"), "pfcf": metrics.get("pfcf")}
    notes: List[str] = []
    if price is None:
        return current, notes

    def _note(key: str, fy: int) -> None:
        notes.append(
            f"Güncel {_MULTIPLE_LABELS[key]} oranı en son mali yılın verisiyle hizalanamadığı için "
            f"{fy} mali yılının verisiyle hesaplandı."
        )

    if current["pe"] is None:
        eps_series = to_annual_series(normalized, "EPS")
        for fy in sorted(eps_series, reverse=True):
            eps = eps_series.get(fy)
            if eps is not None and eps > 0:
                current["pe"] = round(price / eps, 4)
                _note("pe", fy)
                break

    shares = metrics.get("shares")
    if shares:
        if current["ps"] is None:
            revenue_series = to_annual_series(normalized, "Revenue")
            for fy in sorted(revenue_series, reverse=True):
                revenue = revenue_series.get(fy)
                if revenue is not None and revenue > 0:
                    current["ps"] = round(price * shares / revenue, 4)
                    _note("ps", fy)
                    break

        if current["pfcf"] is None:
            ocf_series = to_annual_series(normalized, "OperatingCashFlow")
            capex_series = to_annual_series(normalized, "CapEx")
            fcf_by_fy = {row.get("fy"): row.get("fcf") for row in (ratios or []) if row.get("fy") is not None}
            fys = set(ocf_series) | set(capex_series) | set(fcf_by_fy)
            for fy in sorted(fys, reverse=True):
                fcf = fcf_by_fy.get(fy)
                if fcf is None:
                    ocf, capex = ocf_series.get(fy), capex_series.get(fy)
                    fcf = None if ocf is None or capex is None else ocf - capex
                if fcf is not None and fcf > 0:
                    current["pfcf"] = round(price * shares / fcf, 4)
                    _note("pfcf", fy)
                    break

    return current, notes


def _empty_growth_adjusted(metric: str, label: str, raw_label: str, base_growth: Optional[float]) -> dict:
    """A fully-shaped, not-applicable growth-adjusted block (all ratio fields
    ``None``), so every downstream consumer can read the same keys whether or
    not a PEG / growth-adjusted EV/Sales could actually be computed."""
    return {
        "metric": metric,
        "label": label,
        "raw_label": raw_label,
        "value": None,
        "percentile": None,
        "raw_percentile": None,
        "applicable": False,
        "reason": None,
        "base_growth_pct": round(base_growth * 100.0, 1) if _is_number(base_growth) else None,
        "sector_peg": None,
    }


def _sector_peg(sector_medians_result: Optional[dict]) -> Optional[float]:
    """Damodaran sector-median PEG, IF the (optional) reference data carries
    it: a direct ``peg`` column wins, else derived from the sector median
    ``pe`` and expected ``growth`` (decimal fraction) when both are present
    and growth clears the :data:`multiples._PEG_MIN_GROWTH` floor. Returns
    ``None`` whenever the growth/peg columns are absent (the default data
    shape) -- sector PEG is a nice-to-have enrichment (VALUATION.md Sec.7)."""
    if not sector_medians_result:
        return None
    direct = sector_medians_result.get("peg")
    if _is_number(direct) and direct > 0:
        return round(direct, 2)
    sec_pe = sector_medians_result.get("pe")
    sec_growth = sector_medians_result.get("growth")
    if _is_number(sec_pe) and sec_pe > 0 and _is_number(sec_growth) and sec_growth >= multiples._PEG_MIN_GROWTH:
        return round(sec_pe / (sec_growth * 100.0), 2)
    return None


def _build_growth_adjusted(
    history: list,
    current: dict,
    metrics: dict,
    normalized: dict,
    base_growth: Optional[float],
    hyper_growth_active: bool,
    pe_pct: Optional[float],
    sector_medians_result: Optional[dict],
) -> "tuple[dict, Optional[float], Optional[float]]":
    """Build the ``multiples.growth_adjusted`` block (SPEC.md Sec.6).

    Standard mode ranks PEG = current P/E / base growth (in % points), paired
    with the raw P/E percentile. Hyper-grower mode ranks growth-adjusted
    EV/Sales = current EV/Sales / base growth, paired with the raw EV/Sales
    percentile -- P/E is meaningless for these filers, so EV/Sales stands in
    as the raw multiple. The denominator is ALWAYS the assumptions pipeline's
    base ``growth_5y`` (surfaced as ``base_growth_pct``); the ratio is only
    computed when the raw multiple is positive AND base growth clears the 5%
    floor (:data:`multiples._PEG_MIN_GROWTH`) -- otherwise it degrades to
    ``applicable=False`` with a Turkish reason, never a negative/exploded
    figure.

    Returns:
        A ``(block, raw_pair_pct, growth_adj_pct)`` tuple. ``block`` is the
        output dict; ``raw_pair_pct``/``growth_adj_pct`` are the two
        percentiles the triangulation divergence check compares (either may
        be ``None``). Never raises.
    """
    if hyper_growth_active:
        metric, label, raw_label, raw_key = "growth_adj_ps", "Büyüme-ayarlı EV/Satış", "EV/S", "ev_sales"
        market_cap = metrics.get("market_cap")
        net_debt = metrics.get("net_debt")
        ps_current = current.get("ps")
        if ps_current is not None and market_cap and market_cap > 0:
            # EV/Sales = P/S * EV/market_cap = P/S * (1 + net_debt/market_cap).
            raw_current = ps_current * (1.0 + (net_debt or 0.0) / market_cap)
        else:
            raw_current = None
        raw_pct = multiples.percentile_position([h.get("ev_sales") for h in history], raw_current)
    else:
        metric, label, raw_label, raw_key = "peg", "PEG", "P/E", "pe"
        raw_current = current.get("pe")
        raw_pct = pe_pct

    block = _empty_growth_adjusted(metric, label, raw_label, base_growth)
    block["raw_percentile"] = raw_pct
    block["sector_peg"] = _sector_peg(sector_medians_result) if metric == "peg" else None

    ga_value = multiples.growth_adjusted_value(raw_current, base_growth)
    if ga_value is None:
        if not _is_number(base_growth) or base_growth < multiples._PEG_MIN_GROWTH:
            block["reason"] = (
                f"Büyümeye göre ayarlı çarpan ({label}) uygulanamaz: base büyüme %5'in altında "
                "(payda güvenilir değil)."
            )
        elif raw_current is None or raw_current <= 0:
            detail = "TTM kâr pozitif değil (P/E yok)" if metric == "peg" else "EV/Satış hesaplanamadı"
            block["reason"] = f"Büyümeye göre ayarlı çarpan ({label}) uygulanamaz: {detail}."
        else:
            block["reason"] = f"Büyümeye göre ayarlı çarpan ({label}) uygulanamaz."
        return block, raw_pct, None

    revenue_series = to_annual_series(normalized, "Revenue")
    ga_hist = multiples.growth_adjusted_history(history, revenue_series, raw_key)
    ga_pct = multiples.percentile_position(ga_hist, ga_value)

    block["value"] = ga_value
    block["percentile"] = ga_pct
    block["applicable"] = True
    return block, raw_pct, ga_pct


def _format_growth_pct(value: float) -> str:
    """Turkish growth string, e.g. ``0.08 -> "%8 büyüme"`` (Sec.4)."""
    return f"%{value * 100:.0f} büyüme"


def _format_discount_rate_pct(value: float) -> str:
    """Turkish discount-rate string, e.g. ``0.12 -> "%12"`` (Sec.4)."""
    return f"%{value * 100:.0f}"


#: Turkish scenario labels used inside the hyper-grower ``fair_value_range``
#: note (see ``_hyper_scenario_meta``), keyed the same as ``_SCENARIO_KEYS``.
_HYPER_SCENARIO_LABEL = {"bear": "kötümser", "base": "temel", "bull": "iyimser"}


def _hyper_scenario_meta(hyper_growth_detail: Optional[dict]) -> dict:
    """Build the ``fair_value_range`` ``scenario_meta`` override for hyper-
    grower mode (SPEC.md Sec.11): per-scenario ``growth``/``discount_rate``/
    ``note`` strings that reflect the revenue-first DCF's own start-growth,
    discount rate, and mature target FCF margin -- instead of the standard
    clamped assumptions the headline band no longer actually uses once
    hyper-grower mode takes over.

    Any scenario whose cell is missing or lacks ``start_growth``/
    ``target_fcf_margin`` (a failed ``revenue_first_dcf`` call for that
    scenario) is simply omitted, so ``_build_fair_value_range`` falls back
    to the standard assumptions-derived value for that one field/scenario
    rather than fabricating a meta entry. Never raises.
    """
    scenarios = (hyper_growth_detail or {}).get("scenarios") or {}
    meta: dict = {}
    for key in _SCENARIO_KEYS:
        cell = scenarios.get(key) or {}
        start_growth = cell.get("start_growth")
        target_fcf_margin = cell.get("target_fcf_margin")
        if not _is_number(start_growth) or not _is_number(target_fcf_margin):
            continue

        discount_rate = _HYPER_DISCOUNT_RATE_BY_SCENARIO[key]
        scenario_label = _HYPER_SCENARIO_LABEL[key]
        # Terminal growth is the risk-free-derived shared anchor (WP2/LEVER 1),
        # no longer a hardcoded 2.5% -- interpolate the actual value like the
        # mature/midgrowth scenario-meta helpers do.
        terminal_growth = cell.get("terminal_growth")
        terminal_str = f"%{terminal_growth * 100:.1f}" if _is_number(terminal_growth) else "terminal"
        meta[key] = {
            "growth": f"%{start_growth * 100:.0f} başlangıç → {terminal_str} terminale fade",
            "discount_rate": f"%{discount_rate * 100:.0f}",
            "note": (
                f"Hiper-büyüme {scenario_label}: başlangıç büyüme %{start_growth * 100:.0f} "
                f"(10 yılda {terminal_str} terminale fade), olgun FCF marjı %{target_fcf_margin * 100:.0f}, "
                f"iskonto %{discount_rate * 100:.0f}."
            ),
        }
    return meta


def _mature_scenario_meta(mature_revenue_detail: Optional[dict]) -> dict:
    """Build the ``fair_value_range`` ``scenario_meta`` override for the
    mature revenue-first DCF headline (mature, FCF-suppressed-but-growing
    filers whose realized growth clears the gate -- see
    ``_build_mature_revenue_dcf``), mirroring :func:`_hyper_scenario_meta`'s
    structure: per-scenario ``growth``/``discount_rate``/``note`` strings
    that reflect this method's own realized start growth and its
    per-scenario mature target FCF margin/discount rate, instead of the
    standard clamped assumptions the headline band no longer actually uses
    once this mode takes over.

    Any scenario whose cell is missing ``start_growth``/
    ``target_fcf_margin``/``discount_rate`` (a failed ``revenue_first_dcf``
    call for that scenario, or the scenario was skipped) is simply omitted,
    so ``_build_fair_value_range`` falls back to the standard
    assumptions-derived value for that field/scenario. Never raises.
    """
    scenarios = (mature_revenue_detail or {}).get("scenarios") or {}
    meta: dict = {}
    for key in _SCENARIO_KEYS:
        cell = scenarios.get(key) or {}
        start_growth = cell.get("start_growth")
        target_fcf_margin = cell.get("target_fcf_margin")
        discount_rate = cell.get("discount_rate")
        terminal_growth = cell.get("terminal_growth")
        if not _is_number(start_growth) or not _is_number(target_fcf_margin) or not _is_number(discount_rate):
            continue

        scenario_label = _HYPER_SCENARIO_LABEL[key]
        terminal_str = f"%{terminal_growth * 100:.1f}" if _is_number(terminal_growth) else "terminal"
        meta[key] = {
            "growth": (
                f"gerçekleşen büyüme %{start_growth * 100:.1f}, olgun hedef marj %{target_fcf_margin * 100:.1f}"
            ),
            "discount_rate": _format_discount_rate_pct(discount_rate),
            "note": (
                f"Olgun revenue-first DCF {scenario_label}: gerçekleşen büyüme %{start_growth * 100:.1f} "
                f"({_MATURE_STEADY_STATE_YEAR} yılda {terminal_str} terminale fade), olgun FCF marjı "
                f"%{target_fcf_margin * 100:.1f}, iskonto %{discount_rate * 100:.0f}."
            ),
        }
    return meta


def _midgrowth_scenario_meta(midgrowth_revenue_detail: Optional[dict]) -> dict:
    """Build the ``fair_value_range`` ``scenario_meta`` override for the
    mid-growth, loss-making revenue-first DCF headline (SPEC Sec.8d),
    mirroring :func:`_mature_scenario_meta` -- same per-scenario
    ``growth``/``discount_rate``/``note`` shape, but with this method's own
    8-year fade horizon and "orta-büyüme" wording. Any scenario whose cell
    is missing ``start_growth``/``target_fcf_margin``/``discount_rate`` is
    omitted so :func:`_build_fair_value_range` falls back to the standard
    assumptions-derived value for that field/scenario. Never raises.
    """
    scenarios = (midgrowth_revenue_detail or {}).get("scenarios") or {}
    meta: dict = {}
    for key in _SCENARIO_KEYS:
        cell = scenarios.get(key) or {}
        start_growth = cell.get("start_growth")
        target_fcf_margin = cell.get("target_fcf_margin")
        discount_rate = cell.get("discount_rate")
        terminal_growth = cell.get("terminal_growth")
        if not _is_number(start_growth) or not _is_number(target_fcf_margin) or not _is_number(discount_rate):
            continue

        scenario_label = _HYPER_SCENARIO_LABEL[key]
        terminal_str = f"%{terminal_growth * 100:.1f}" if _is_number(terminal_growth) else "terminal"
        meta[key] = {
            "growth": (
                f"gerçekleşen büyüme %{start_growth * 100:.1f}, olgun hedef marj %{target_fcf_margin * 100:.1f}"
            ),
            "discount_rate": _format_discount_rate_pct(discount_rate),
            "note": (
                f"Orta-büyüme revenue-first DCF {scenario_label}: gerçekleşen büyüme %{start_growth * 100:.1f} "
                f"({_MIDGROWTH_STEADY_STATE_YEAR} yılda {terminal_str} terminale fade), olgun FCF marjı "
                f"%{target_fcf_margin * 100:.1f}, iskonto %{discount_rate * 100:.0f}."
            ),
        }
    return meta


def _epv_scenario_meta(earnings_power: Optional[dict]) -> dict:
    """Build the ``fair_value_range`` ``scenario_meta`` override for the
    earnings-power (EPV) headline (Sec.8a), mirroring
    :func:`_hyper_scenario_meta`'s structure: per-scenario ``growth``/
    ``discount_rate``/``note`` strings that reflect the EPV anchor's own
    zero-growth, cost-of-equity-only construction instead of the standard
    (unused, since EPV is now the headline) clamped assumptions.

    Returns an empty dict (so :func:`_build_fair_value_range` falls back to
    the assumptions-derived values) when ``earnings_power`` is ``None`` or
    missing its ``scenarios``/``cost_of_equity``. Never raises.
    """
    if not earnings_power:
        return {}
    scenarios = earnings_power.get("scenarios") or {}
    cost_of_equity = earnings_power.get("cost_of_equity")
    if not scenarios or not _is_number(cost_of_equity):
        return {}

    meta: dict = {}
    for key in _SCENARIO_KEYS:
        if key not in scenarios:
            continue
        scale = _PB_SCENARIO_SCALE.get(key, 1.0)
        meta[key] = {
            "growth": "sıfır büyüme (kazanç gücü çapası)",
            "discount_rate": _format_discount_rate_pct(cost_of_equity),
            "note": (
                f"Kazanç-gücü çapası ({key}): normalize net kâr / özkaynak maliyeti (%{cost_of_equity * 100:.0f}), "
                f"ölçek {scale:.1f}x, sıfır büyüme varsayımıyla (büyüme primi kasıtlı olarak dışlandı)."
            ),
        }
    return meta


def _cyclical_fcfe_scenario_meta(cyclical_fcfe_detail: Optional[dict], assumptions: dict) -> dict:
    """Build the ``fair_value_range`` ``scenario_meta`` override for the
    cyclical sustainable-growth FCFE headline (SPEC.md Sec.8e), mirroring
    :func:`_epv_scenario_meta`'s structure: per-scenario ``growth``/
    ``discount_rate``/``note`` strings that reflect this anchor's own
    growth-inclusive, reinvestment-funded (``b = g / roe``) construction,
    instead of the standard clamped-assumptions description.

    Returns an empty dict (so :func:`_build_fair_value_range` falls back to
    the assumptions-derived values) when ``cyclical_fcfe_detail`` is
    ``None`` or missing its ``scenarios``/``roe``. A scenario missing a
    computed ``per_share`` (its own assumptions were invalid) or a numeric
    ``growth_5y``/``discount_rate`` is simply omitted from the returned
    meta, falling back to the assumptions-derived value for that scenario.
    Never raises.
    """
    if not cyclical_fcfe_detail:
        return {}
    scenarios = cyclical_fcfe_detail.get("scenarios") or {}
    roe = cyclical_fcfe_detail.get("roe")
    if not scenarios or not _is_number(roe):
        return {}

    meta: dict = {}
    for key in _SCENARIO_KEYS:
        cell = scenarios.get(key) or {}
        if not _is_number(cell.get("per_share")):
            continue
        scenario_assumptions = assumptions.get(key) or {}
        growth_5y = scenario_assumptions.get("growth_5y")
        discount_rate = scenario_assumptions.get("discount_rate")
        if not _is_number(growth_5y) or not _is_number(discount_rate):
            continue

        reinvestment_rate = min(growth_5y, roe) / roe
        meta[key] = {
            "growth": f"%{growth_5y * 100:.1f} büyüme (kazanç + sürdürülebilir büyüme)",
            "discount_rate": _format_discount_rate_pct(discount_rate),
            "note": (
                f"Sürdürülebilir-büyüme FCFE çapası ({key}): normalize net kâr büyütülür, büyümeyi fonlamak "
                f"için kârın ~%{reinvestment_rate * 100:.0f}'i (g/ROE, ROE %{roe * 100:.0f}) reinvest edilir, "
                "kalanı iskonto edilir."
            ),
        }
    return meta


def _build_fair_value_range(
    dcf_scenarios: Optional[dict],
    pb_roe: Optional[dict],
    assumptions: dict,
    scenario_meta: Optional[dict] = None,
) -> dict:
    """Build the ``fair_value_range`` shape (Sec.4) from whichever scenario
    source is active: the FCF-DCF scenarios if present, else the P/B x ROE
    (or, for reit, FFO Gordon-growth) scenarios; all-``None`` if neither is
    available.

    Args:
        dcf_scenarios: The active per-share/lo/hi scenario dict (may be the
            hyper-grower revenue-first band, the cyclical normalized
            variant, or the raw FCF-DCF band -- whichever the caller has
            already selected as the headline source), or ``None``.
        pb_roe: The financial/reit anchor dict -- P/B x ROE for `financial`,
            the FFO Gordon-growth anchor for `reit` (or `reit`'s own P/B x
            ROE fallback when FFO couldn't be built) -- used as a fallback
            source when ``dcf_scenarios`` is ``None``. Despite the parameter
            name, the caller passes whichever of the two blocks is active
            for the current sector; both share the same ``{"scenarios":
            {...}}`` shape so this function doesn't need to know which one
            it received.
        assumptions: The standard bear/base/bull assumption dict; used to
            derive ``growth``/``discount_rate``/``note`` for any scenario
            not covered by ``scenario_meta``.
        scenario_meta: Optional override, keyed by scenario, of
            ``{"growth": str, "discount_rate": str, "note": str}`` --
            pre-formatted Turkish strings that should replace the
            assumptions-derived ones for that scenario (used when the
            headline band's own inputs differ from the standard clamped
            assumptions, e.g. hyper-grower mode's revenue-first DCF; see
            SPEC.md Sec.11). ``None`` (the default) keeps the previous
            behavior of always reading from ``assumptions``. A scenario
            missing from ``scenario_meta`` (or with a missing field) falls
            back to the assumptions-derived value for that field.
    """
    source = dcf_scenarios if dcf_scenarios is not None else ((pb_roe or {}).get("scenarios"))
    if source is None:
        return _empty_fair_value_range()

    result = {}
    for key in _SCENARIO_KEYS:
        cell = source.get(key) or {}
        scenario_assumptions = assumptions.get(key) or {}
        growth = scenario_assumptions.get("growth_5y")
        discount_rate = scenario_assumptions.get("discount_rate")

        meta = (scenario_meta or {}).get(key) or {}
        growth_str = meta.get("growth") or (_format_growth_pct(growth) if _is_number(growth) else None)
        discount_rate_str = meta.get("discount_rate") or (
            _format_discount_rate_pct(discount_rate) if _is_number(discount_rate) else None
        )
        note = meta.get("note") or scenario_assumptions.get("story")

        result[key] = {
            "lo": cell.get("lo"),
            "hi": cell.get("hi"),
            "growth": growth_str,
            "discount_rate": discount_rate_str,
            "note": note,
        }
    return result


def run_valuation(
    normalized: dict,
    ratios: list,
    metrics: dict,
    price: Optional[float],
    price_df,
    assumptions: dict,
    sector_type: str,
    damodaran_dir: Optional[str] = None,
    sic_description: Optional[str] = None,
    hyper_growth_extras: Optional[dict] = None,
) -> dict:
    """Run the full deterministic valuation engine (SPEC Sec.11).

    Args:
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.
        ratios: The list returned by
            ``sec_analyzer.normalize.ratios.compute_ratios``.
        metrics: The dict returned by
            ``sec_analyzer.normalize.metrics.compute_metrics``.
        price: Current market price per share, or ``None``.
        price_df: The DataFrame returned by
            ``sec_analyzer.fetch.prices.get_price_history``, or ``None``.
        assumptions: The phase-1 bear/base/bull assumption dict (SPEC
            Sec.2), already run through ``sanity.validate_assumptions`` by
            the caller (this function re-validates defensively and records
            any violations as notes rather than trusting the caller).
        sector_type: One of the ``valuation.sector.classify_sector``
            buckets, resolved by the caller (see that module's docstring
            for the SIC-missing fallback wiring).
        damodaran_dir: Directory holding Damodaran reference CSVs. Defaults
            to ``Config.DAMODARAN_DIR``.
        sic_description: The filer's SEC ``sicDescription`` (from
            ``submissions``), used only to look up Damodaran sector
            medians. Not part of the SPEC Sec.11 signature's *required*
            positional args -- an intentional, backward-compatible
            addition (default ``None``) so sector-median matching can work
            when the caller has it, without breaking any caller that
            doesn't pass it.
        hyper_growth_extras: Optional LLM/user-refined hyper-grower inputs
            (SPEC Sec.5): ``{"tam_usd": .., "per_scenario": {"bear"/"base"/
            "bull": {"target_fcf_margin", "steady_state_year",
            "probability"}}}``. Only consulted when
            ``sector.detect_hyper_grower`` triggers; overrides the
            deterministic target margin/steady-state year/probabilities/
            TAM per scenario. ``None`` (the default) keeps every hyper-
            grower input fully deterministic -- backward compatible with
            every existing caller.

    Returns:
        The ``valuation`` dict documented in SPEC Sec.11. Every
        unavailable piece is ``None`` plus a Turkish note in ``notes``.
        Never raises.
    """
    try:
        return _run_valuation(
            normalized or {}, ratios or [], metrics or {}, price, price_df, assumptions or {},
            sector_type, damodaran_dir, sic_description, hyper_growth_extras,
        )
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("run_valuation() failed unexpectedly; returning a degraded result.")
        return _empty_valuation(sector_type, assumptions)


def _run_valuation(
    normalized: dict, ratios: list, metrics: dict, price: Optional[float], price_df, assumptions: dict,
    sector_type: str, damodaran_dir: Optional[str], sic_description: Optional[str],
    hyper_growth_extras: Optional[dict] = None,
) -> dict:
    notes: List[str] = []

    is_unprofitable = sector_type == "growth_unprofitable"
    for violation in sanity.validate_assumptions(assumptions, is_unprofitable=is_unprofitable):
        notes.append(f"Varsayım uyarısı: {violation}")

    # F5: clamp any out-of-range assumption into a sane set and use THAT set
    # for every downstream calculation (DCF, reverse-DCF, sensitivity,
    # hyper-grower) -- what's shown in the output's "assumptions" key is
    # exactly what gets used, not the raw (possibly out-of-range) phase-1
    # input. validate_assumptions above still ran against the original
    # input, so its notes still describe the pre-clamp violations.
    assumptions, clamp_notes = sanity.clamp_assumptions(assumptions, is_unprofitable=is_unprofitable)
    notes.extend(clamp_notes)

    sbc_adjusted_fcf_by_fy = _sbc_adjusted_fcf_by_fy(ratios, normalized)
    fcf0, fcf0_source, fcf0_note = _select_fcf0(metrics, sbc_adjusted_fcf_by_fy)
    if fcf0_note:
        notes.append(fcf0_note)

    # F2: SBC is now expensed directly into fcf0 (via sbc_adjusted_fcf_by_fy)
    # above, so no further dilution adjustment is layered on top of the
    # standard DCF -- that would double-count the same drag. net_debt stays
    # available in metrics for display only: F1 (FCFE-direct) never
    # subtracts it in the valuation math below.
    dilution_rate = 0.0
    shares = metrics.get("shares")

    # WP2: shared terminal-growth anchor (min(risk_free, 4%) -- Damodaran's
    # practical rule that a perpetuity growth rate shouldn't exceed the
    # risk-free rate; see rule_based._terminal_growth_anchor for the
    # assumptions-driven mature/midgrowth analog of this same rule). Loaded
    # here -- earlier than the multiples-comparison section below that
    # historically did this load -- purely so the hyper-grower revenue-first
    # DCF (built further down) can use it too; `sector_data` is reused as-is
    # by that later section (no second load, no behavior change there: same
    # deterministic local CSV read either way).
    sector_data = damodaran.load_sector_data(damodaran_dir if damodaran_dir is not None else Config.DAMODARAN_DIR)
    risk_free_pct = sector_data.get("risk_free") if sector_data else None
    if _is_number(risk_free_pct):
        terminal_growth_anchor = min(risk_free_pct / 100.0, sanity._TERMINAL_GROWTH_MAX)
    else:
        terminal_growth_anchor = _HYPER_TERMINAL_GROWTH

    # WP6: match the sector medians ONCE, here (rather than down in the
    # multiples-comparison block that historically did this), so the
    # hyper-grower's maintenance-CapEx floor (below) can also use the
    # sector's Cap Ex/Sales ratio -- the multiples-comparison block further
    # down reuses this same `sector_medians_result` instead of recomputing it.
    if sector_data is None:
        notes.append("Damodaran sektör verisi bulunamadı; sektör medyanları gösterilemiyor.")
        sector_medians_result = None
    elif not sic_description:
        notes.append("SIC açıklaması sağlanmadığı için Damodaran sektör medyanları eşleştirilemedi.")
        sector_medians_result = None
    else:
        sector_medians_result = damodaran.sector_medians(sector_data, sic_description)
        if sector_medians_result is None:
            notes.append("Şirketin SIC açıklaması Damodaran sektörleriyle eşleştirilemedi.")

    sector_capex_sales = (sector_medians_result or {}).get("capex_sales")

    # --- Hyper-grower detection (deterministic, from financials; SPEC Sec.1) ---
    # F4: never attempted for financial/reit sectors -- a revenue-margin
    # hyper-DCF doesn't make sense there (P/B x ROE is the method instead).
    if sector_type in _SECTORS_WITHOUT_FCF_DCF:
        is_hyper_grower, hyper_reasons = False, []
    else:
        is_hyper_grower, hyper_reasons = sector.detect_hyper_grower(metrics, ratios, normalized)

    dcf_enabled = sector_type not in _SECTORS_WITHOUT_FCF_DCF
    disabled_reason = None
    if not dcf_enabled:
        if sector_type == "reit":
            disabled_reason = (
                "GYO (REIT) şirketlerde serbest nakit akışı DCF'i güvenilir değildir; bunun yerine FFO "
                "(funds from operations) bazlı Gordon büyüme modeli kullanılıyor (FFO hesaplanamazsa "
                "P/B x ROE çapasına geri dönülür)."
            )
        else:
            disabled_reason = (
                "Finansal şirketlerde serbest nakit akışı DCF'i güvenilir değildir; "
                "bunun yerine P/B x ROE çapası kullanılıyor."
            )

    dcf_scenarios = None
    dcf_high_growth_flag = False
    if dcf_enabled:
        dcf_scenarios, dcf_notes, dcf_high_growth_flag = _build_dcf_scenarios(assumptions, fcf0, shares, dilution_rate)
        notes.extend(dcf_notes)
        if dcf_scenarios is None and fcf0 is not None:
            notes.append("DCF hesaplanamadı: geçerli bir hisse sayısı (shares) yok.")

    normalized_variant = None
    normalized_fcf0 = None
    if sector_type == "cyclical":
        normalized_fcf0, cyclical_notes = _normalized_fcf0(normalized, metrics)
        notes.extend(cyclical_notes)
        if normalized_fcf0 is not None:
            normalized_variant, variant_notes, _normalized_high_growth_flag = _build_dcf_scenarios(
                assumptions, normalized_fcf0, shares, dilution_rate
            )
            notes.extend(variant_notes)

    # --- P/B x ROE (financial) / FFO (reit) anchor (SPEC Sec.8/Sec.8c) ------
    # `financial` keeps the unchanged P/B x ROE anchor. `reit` gets the new
    # FFO-based Gordon-growth anchor instead (GAAP real-estate depreciation
    # depresses both net income and book equity, so P/B x ROE systematically
    # understates a REIT); if FFO can't be built at all (no Depreciation data
    # for any fiscal year that also has NetIncome, or the resulting FFO is
    # <= 0), gracefully fall back to the same P/B x ROE anchor `financial`
    # uses, so there's still a book-based headline/triangulation anchor.
    pb_roe = None
    ffo = None
    if sector_type == "financial":
        pb_roe, pb_notes = _build_pb_roe(assumptions, normalized, metrics, ratios)
        notes.extend(pb_notes)
    elif sector_type == "reit":
        ffo, ffo_notes = _build_ffo(assumptions, normalized, metrics, ratios)
        notes.extend(ffo_notes)
        if ffo is None:
            pb_roe, pb_notes = _build_pb_roe(assumptions, normalized, metrics, ratios)
            notes.extend(pb_notes)
            notes.append(
                "GYO (REIT) için FFO hesaplanamadı; manşet/üçgenleme çapası olarak P/B x ROE'ye "
                "geri dönüldü."
            )

    # The active anchor for THIS sector's headline/triangulation purposes:
    # the FFO block when reit's FFO build succeeded, else pb_roe (which is
    # the reit fallback above, or financial's own anchor, or None for every
    # other sector).
    reit_or_financial_anchor = ffo if (sector_type == "reit" and ffo is not None) else pb_roe

    # --- Hyper-grower revenue-first DCF (SPEC Sec.3) ------------------------
    # Only actually built once detected; any sub-step failure degrades the
    # OUTPUT "hyper_growth" flag back to False (detail None) rather than
    # trusting the raw detection result, so a broken hyper build never costs
    # the standard valuation below.
    #
    # WP3: the hyper-grower revenue-first DCF fades revenue growth and FCF
    # margin toward a mature steady state, but discounting every year at a
    # fixed cohort rate (14/12/10 bear/base/bull) is internally inconsistent
    # with that fade -- Damodaran's standard fix is to fade the discount
    # rate too, from the cohort rate down to a mature cost of equity by the
    # steady-state year. The mature target reused here is the BASE
    # scenario's own discount rate from `assumptions` -- already CAPM-aware
    # (Damodaran sector beta relevered with the firm's own D/E, plus ERP and
    # the risk-free rate) and already run through `sanity.clamp_assumptions`
    # above, so no separate CAPM computation is needed in this module.
    # Floored at `terminal_growth_anchor + sanity._MIN_ERP_SPREAD` (the same
    # minimum equity-risk-premium spread `sanity` enforces elsewhere) so the
    # fade can never collapse the rate to within the ERP-spread guard's
    # forbidden zone above the terminal growth rate. Missing/non-numeric
    # base discount rate -> `None` -> the fade is skipped entirely and every
    # revenue-first DCF call inside `_build_hyper_growth` stays flat, exactly
    # as before this parameter existed.
    base_discount_rate_for_fade = (assumptions.get("base") or {}).get("discount_rate")
    if _is_number(base_discount_rate_for_fade):
        mature_discount_rate = max(base_discount_rate_for_fade, terminal_growth_anchor + sanity._MIN_ERP_SPREAD)
    else:
        mature_discount_rate = None

    hyper_growth_detail = None
    if is_hyper_grower:
        hyper_growth_detail, hyper_notes = _build_hyper_growth(
            metrics, ratios, normalized, price, shares, hyper_reasons, hyper_growth_extras,
            terminal_growth_anchor, mature_discount_rate, sector_capex_sales,
        )
        notes.extend(hyper_notes)
    hyper_growth_active = is_hyper_grower and hyper_growth_detail is not None

    # --- Earnings-power-value (EPV) anchor (SPEC Sec.8a/8e) ------------------
    # Built for mature filers (an alternative headline for genuinely
    # FCF-suppressed-but-profitable companies, e.g. Amazon) AND cyclical
    # filers (SPEC Sec.8e): for cyclical it's both the zero-growth floor AND
    # the earnings base the sustainable-growth FCFE anchor
    # (_build_cyclical_fcfe) grows -- neither built once hyper-grower mode
    # is already active. Gated by _fcf_dcf_unreliable below for both sectors.
    earnings_power = None
    ep_notes: List[str] = []
    if sector_type in ("mature", "cyclical") and not hyper_growth_active:
        earnings_power, ep_notes = _build_earnings_power(assumptions, normalized, metrics, ratios)

    # For cyclical filers, the headline fair-value band and the triangulation
    # DCF signal should reflect through-cycle earning power, not a single
    # (often near-trough) year's FCF. Prefer the normalized-earnings variant
    # when it was successfully computed; otherwise fall back to the raw
    # FCF-DCF band. Both variants remain reported side by side under `dcf`.
    # Hyper-grower mode takes precedence over the cyclical variant, which in
    # turn takes precedence over the raw FCF-DCF band (SPEC Sec.3.5).
    primary_dcf_scenarios = dcf_scenarios
    epv_headline = False
    mature_revenue_headline = False
    mature_revenue_detail = None
    midgrowth_revenue_headline = False
    midgrowth_revenue_detail = None
    cyclical_fcfe_headline = False
    cyclical_fcfe_detail = None
    if hyper_growth_active:
        if hyper_growth_detail.get("suppressed"):
            # Revenue-first DCF produced a non-credible negative base value:
            # empty the headline fair-value range and drop the DCF triangulation
            # vote. The explanatory note was already appended inside
            # _build_hyper_growth.
            primary_dcf_scenarios = None
        else:
            primary_dcf_scenarios = hyper_growth_detail["scenarios"]
            notes.append(
                "Hiper-büyüme modu: manşet aralığı revenue-first DCF'ten (büyüme fade + olgun hedef marj) "
                "alındı; standart FCF-DCF ikincil olarak 'dcf.scenarios'ta."
            )
    elif sector_type == "cyclical":
        # Gate: is the raw (near-trough / capex-suppressed) FCF-DCF unreliable
        # enough to replace with an earnings-based anchor? Same proven gate the
        # mature path uses (FCF suppressed vs EPV, cash-backed, investment-driven).
        unreliable, quality_note = (False, None)
        if earnings_power is not None:
            unreliable, quality_note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics)
            if quality_note:
                notes.append(quality_note)
        if unreliable and earnings_power is not None:
            # Growth-inclusive sustainable-growth FCFE (Sec.8e) vs the zero-growth
            # EPV floor. Headline FCFE only when it clears the EPV floor.
            cyclical_fcfe_detail, cf_notes = _build_cyclical_fcfe(
                assumptions, earnings_power, normalized, metrics, shares, dilution_rate
            )
            epv_base_ps = ((earnings_power.get("scenarios") or {}).get("base") or {}).get("per_share")
            cf_base_ps = (((cyclical_fcfe_detail or {}).get("scenarios") or {}).get("base") or {}).get("per_share")
            cf_beats_floor = (
                cyclical_fcfe_detail is not None and _is_number(cf_base_ps)
                and (not _is_number(epv_base_ps) or cf_base_ps >= epv_base_ps)
            )
            if cyclical_fcfe_detail is not None:
                # Augments the builder's returned dict with a caller-side
                # classification; not produced by _build_cyclical_fcfe itself.
                cyclical_fcfe_detail["growth_vs_floor"] = _growth_vs_floor(epv_base_ps, cf_base_ps)
            if cf_beats_floor:
                primary_dcf_scenarios = cyclical_fcfe_detail["scenarios"]
                cyclical_fcfe_headline = True
                notes.extend(cf_notes)
                notes.append(
                    "Döngüsel + sermaye-yoğun: serbest nakit akışı büyüme yatırımıyla (yüksek CapEx) "
                    "bastırıldığı için manşet, döngü-ortası normalize kazanca sürdürülebilir-büyüme "
                    "(reinvestment=g/ROE) uygulayan bir FCFE çapasına dayandırıldı. Sıfır-büyüme EPV "
                    "tabanı, döngü-ortası FCF-DCF ve ham FCF-DCF ikincil olarak raporlanır."
                )
            else:
                # FCFE couldn't clear the EPV floor (or wasn't buildable): headline
                # the zero-growth EPV floor. Still strictly better than the
                # capex-suppressed raw FCF-DCF.
                notes.extend(cf_notes)
                primary_dcf_scenarios = earnings_power["scenarios"]
                epv_headline = True
                notes.extend(ep_notes)
                notes.append(
                    "Döngüsel + sermaye-yoğun: serbest nakit akışı büyüme yatırımı nedeniyle kazanç "
                    "gücünü yansıtmıyor; manşet sıfır-büyüme kazanç-gücü (EPV) çapasına dayandırıldı. "
                    "Döngü-ortası ve ham FCF-DCF ikincil olarak raporlanır."
                )
                if cyclical_fcfe_detail is not None and _is_number(cf_base_ps) and _is_number(epv_base_ps):
                    notes.append(
                        f"Not: Büyüme-dahil sürdürülebilir-büyüme FCFE de hesaplandı (baz ${cf_base_ps:,.2f}) ancak "
                        f"sıfır-büyüme EPV tabanının (${epv_base_ps:,.2f}) altında kaldığı için manşet EPV'de "
                        "tutuldu; bu, normalize ROE'nin özkaynak maliyetinin ALTINDA olduğunu — yani büyümenin değer "
                        "YARATMADIĞINI (değer sildiğini) — gösterir. Büyüme-dahil FCFE 'cyclical_fcfe_detail' altında "
                        "ikincil olarak raporlanır."
                    )
        elif normalized_variant is not None:
            # FCF is NOT capex-suppressed for this cyclical: keep the existing
            # cycle-mid normalized FCF-DCF headline (unchanged behavior).
            primary_dcf_scenarios = normalized_variant
            notes.append(
                "Döngüsel sektör: manşet makul değer aralığı, tek bir yılın (çoğu zaman dibe yakın) "
                "serbest nakit akışı yerine döngü-ortası normalize edilmiş FCF'e dayandırıldı; "
                "ham dip-FCF DCF senaryoları ayrıca 'dcf.scenarios' altında raporlanıyor."
            )
    elif sector_type == "mature" and earnings_power is not None:
        unreliable, quality_note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics)
        if quality_note:
            notes.append(quality_note)
        if unreliable:
            # Growth-inclusive alternative to the zero-growth EPV floor
            # (VALUATION.md Sec.4/4a addendum): a mature filer whose FCF is
            # suppressed by growth investment but that STILL has genuine,
            # realized top-line growth left (Amazon) gets a revenue-first
            # DCF headline instead of EPV -- only when the growth gate
            # inside _build_mature_revenue_dcf actually clears; otherwise
            # this degrades to the existing EPV-headline behavior below.
            mature_revenue_detail, mr_notes = _build_mature_revenue_dcf(
                assumptions, normalized, metrics, ratios, price, shares
            )
            epv_base_ps = ((earnings_power.get("scenarios") or {}).get("base") or {}).get("per_share")
            mr_base_ps = (
                ((mature_revenue_detail.get("scenarios") or {}).get("base") or {}).get("per_share")
                if mature_revenue_detail is not None else None
            )
            # Guardrail: a growth-inclusive revenue-first value that lands BELOW
            # the zero-growth EPV floor is not a credible growth case -- the
            # defensible mature FCF margin is thinner than the earnings the EPV
            # floor already capitalizes. Keep EPV as the headline and demote the
            # revenue-first band to a secondary cross-check, rather than
            # publishing a growth-inclusive number weaker than the no-growth floor.
            mr_beats_floor = (
                mature_revenue_detail is not None
                and _is_number(mr_base_ps)
                and (not _is_number(epv_base_ps) or mr_base_ps >= epv_base_ps)
            )
            if mature_revenue_detail is not None:
                # Augments the builder's returned dict with a caller-side
                # classification; not produced by _build_mature_revenue_dcf itself.
                mature_revenue_detail["growth_vs_floor"] = _growth_vs_floor(epv_base_ps, mr_base_ps)
            if mr_beats_floor:
                primary_dcf_scenarios = mature_revenue_detail["scenarios"]
                mature_revenue_headline = True
                notes.extend(mr_notes)
                target_pct = mature_revenue_detail.get("target_margin_base")
                target_pct_str = f"%{target_pct * 100:.1f}" if _is_number(target_pct) else "—"
                epv_base_str = f"${epv_base_ps:,.2f}" if _is_number(epv_base_ps) else "—"
                notes.append(
                    "Serbest nakit akışı büyüme yatırımıyla bastırıldığı için manşet, geliri fade eden ve "
                    f"FCF marjını olgun bir hedefe ({target_pct_str}) yakınsayan büyüme-dahil bir revenue-first "
                    f"DCF'e dayandırıldı. Sıfır-büyüme EPV tabanı ({epv_base_str}) ve ham FCF-DCF ikincil "
                    "olarak raporlanır."
                )
            else:
                # Either the growth gate didn't clear / data was missing
                # (mature_revenue_detail is None), OR the revenue-first value came
                # in below the EPV floor (guardrail). Headline the EPV floor.
                notes.extend(mr_notes)
                primary_dcf_scenarios = earnings_power["scenarios"]
                epv_headline = True
                # EPV computation notes (margin-median normalization, over-
                # capitalization advisory, band fallback) only surface when EPV is
                # the headline -- for FCF-DCF-headlined filers they are confusing
                # noise about a value the reader isn't being shown (reviewer F1).
                notes.extend(ep_notes)
                notes.append(
                    "Bu şirkette serbest nakit akışı büyük büyüme yatırımı (yüksek CapEx) nedeniyle kazanç "
                    "gücünü yansıtmıyor; manşet makul değer aralığı sıfır-büyüme kazanç-gücü (EPV) çapasına "
                    "dayandırıldı. Ham FCF-DCF senaryoları ikincil olarak 'dcf.scenarios' altında raporlanıyor. "
                    "NOT: EPV, büyüme primini KASITLI dışlayan muhafazakâr bir tabandır; fiyatın ima ettiği "
                    "büyümeyi ters-DCF ölçer."
                )
                if mature_revenue_detail is not None and _is_number(mr_base_ps) and _is_number(epv_base_ps):
                    notes.append(
                        f"Not: Büyüme-dahil revenue-first DCF de hesaplandı (baz ${mr_base_ps:,.2f}) ancak "
                        f"sıfır-büyüme EPV tabanının (${epv_base_ps:,.2f}) altında kaldığı için manşet EPV'de "
                        "tutuldu; bu, şirketin savunulabilir olgun FCF marjının kapitalize edilen kazancından "
                        "ince olduğunu gösterir. Revenue-first band 'mature_revenue_detail' altında ikincil "
                        "çapraz-kontrol olarak raporlanır."
                    )
    elif sector_type == "growth_unprofitable" and not hyper_growth_active:
        # Mid-growth loss-maker revenue-first DCF (Roadmap Madde 2 / SPEC
        # Sec.8d): a growth_unprofitable filer growing the top line at a
        # real but sub-hyper (12-20%) rate -- one that detect_hyper_grower
        # (CAGR > 20%) doesn't pick up -- gets a revenue-first band instead
        # of a multiples-only headline. If the method can't be built, its
        # growth gate rejects it, or its base value is suppressed (<= $0),
        # primary_dcf_scenarios is left unchanged so the filer keeps its
        # existing raw-FCF-DCF / multiples fallback behavior.
        midgrowth_revenue_detail, mg_notes = _build_midgrowth_revenue_dcf(
            assumptions, normalized, metrics, ratios, price, shares
        )
        mg_base_ps = (
            ((midgrowth_revenue_detail.get("scenarios") or {}).get("base") or {}).get("per_share")
            if midgrowth_revenue_detail is not None else None
        )
        if (
            midgrowth_revenue_detail is not None
            and not midgrowth_revenue_detail.get("suppressed")
            and _is_number(mg_base_ps)
        ):
            primary_dcf_scenarios = midgrowth_revenue_detail["scenarios"]
            midgrowth_revenue_headline = True
            notes.extend(mg_notes)
            target_pct = midgrowth_revenue_detail.get("target_margin_base")
            target_pct_str = f"%{target_pct * 100:.1f}" if _is_number(target_pct) else "—"
            notes.append(
                "Orta-büyüme zarar eden şirket: manşet, geliri fade eden ve FCF marjını olgun bir hedefe "
                f"({target_pct_str}) yakınsayan bir revenue-first DCF'e dayandırıldı (gerçekleşen büyüme "
                "%12-20 bandında, hiper-büyüme eşiğinin altında). Ham FCF-DCF ve çarpanlar ikincil olarak "
                "raporlanır."
            )
        else:
            # Method not built / gate rejected / suppressed: surface its
            # explanatory notes so the reader knows why the headline stayed
            # on multiples, and leave primary_dcf_scenarios untouched.
            notes.extend(mg_notes)

    scenario_meta = None
    if hyper_growth_active and not hyper_growth_detail.get("suppressed"):
        scenario_meta = _hyper_scenario_meta(hyper_growth_detail)
    elif mature_revenue_headline:
        scenario_meta = _mature_scenario_meta(mature_revenue_detail)
    elif midgrowth_revenue_headline:
        scenario_meta = _midgrowth_scenario_meta(midgrowth_revenue_detail)
    elif cyclical_fcfe_headline:
        scenario_meta = _cyclical_fcfe_scenario_meta(cyclical_fcfe_detail, assumptions)
    elif epv_headline:
        scenario_meta = _epv_scenario_meta(earnings_power)
    fair_value_range = _build_fair_value_range(primary_dcf_scenarios, reit_or_financial_anchor, assumptions, scenario_meta)

    # --- Reverse DCF -----------------------------------------------------
    base_assumptions = assumptions.get("base") or {}
    base_growth = base_assumptions.get("growth_5y")
    base_terminal_growth = base_assumptions.get("terminal_growth")
    base_discount_rate = base_assumptions.get("discount_rate")

    implied = None
    bracket_status = "no_data"
    if fcf0 is not None and _is_number(base_terminal_growth) and _is_number(base_discount_rate):
        implied, bracket_status = reverse_dcf.implied_growth_with_status(
            price, fcf0, base_terminal_growth, base_discount_rate, shares, dilution_rate
        )
    # F5: the reverse-DCF bracket (-20%..+60%) can fail to bracket the
    # target price entirely rather than just "not converge" -- distinguish
    # that case (and its direction) from a genuine no-data situation.
    if bracket_status in ("above_bracket", "below_bracket"):
        direction_word = "üzerinde" if bracket_status == "above_bracket" else "altında"
        notes.append(
            f"Fiyat, ters-DCF aralığının (%{reverse_dcf._BRACKET_LO * 100:.0f}.."
            f"%{reverse_dcf._BRACKET_HI * 100:.0f}) {direction_word} bir büyüme ima ediyor."
        )
    elif implied is None:
        notes.append("Ters DCF (fiyatın ima ettiği büyüme) hesaplanamadı.")

    # F6: the reverse-DCF reference growth rate must match what the implied
    # growth rate actually represents -- FCF growth in standard mode (since
    # reverse_dcf.implied_growth_with_status solves over the FCF-DCF), so
    # compare it against the realized FCF CAGR (from the same SBC-adjusted
    # series that feeds fcf0) rather than a revenue CAGR (apples-to-oranges).
    latest_fy = resolve_fundamental_fy(metrics)
    realized_cagr, realized_fcf_label = _realized_cagr_from_series(sbc_adjusted_fcf_by_fy, latest_fy)
    realized_label = f"FCF {realized_fcf_label}" if realized_fcf_label else None

    # These four feed the output "reverse_dcf" dict and the triangulation
    # call below; hyper-grower mode overrides them to the revenue-based
    # pair immediately after, since its reverse-DCF solve
    # (revenue_dcf.implied_start_growth) is itself revenue-based.
    output_implied = implied
    output_realized_cagr = realized_cagr
    output_realized_label = realized_label
    output_bracket_status = bracket_status

    if hyper_growth_active:
        hyper_implied_growth = (hyper_growth_detail.get("implied") or {}).get("growth")
        revenue_cagr = metrics.get("revenue_cagr_5y")
        revenue_cagr_label = "5y" if revenue_cagr is not None else None
        if revenue_cagr is None:
            revenue_cagr = metrics.get("revenue_cagr_3y")
            revenue_cagr_label = "3y" if revenue_cagr is not None else None

        output_implied = hyper_implied_growth
        output_realized_cagr = revenue_cagr
        output_realized_label = f"gelir {revenue_cagr_label}" if revenue_cagr_label else None
        # revenue_dcf.implied_start_growth doesn't expose a bracket-boundary
        # status the way reverse_dcf.implied_growth_with_status does (its
        # bracket is also wider, -20%..+60%); rather than guess the
        # direction, this defaults to "ok" -- a missing hyper implied growth
        # already gets its own note inside _build_hyper_growth.
        output_bracket_status = "ok"
    elif mature_revenue_headline:
        # Mirrors the hyper-grower override immediately above: the mature
        # revenue-first DCF's own reverse-DCF solve
        # (revenue_dcf.implied_start_growth) is itself revenue-based, so the
        # realized-growth reference must be revenue CAGR, not FCF CAGR.
        revenue_cagr = metrics.get("revenue_cagr_5y")
        revenue_cagr_label = "5y" if revenue_cagr is not None else None
        if revenue_cagr is None:
            revenue_cagr = metrics.get("revenue_cagr_3y")
            revenue_cagr_label = "3y" if revenue_cagr is not None else None

        mature_revenue0 = to_annual_series(normalized, "Revenue").get(latest_fy) if latest_fy is not None else None
        mature_implied_growth = revenue_dcf.implied_start_growth(
            price, mature_revenue0, base_terminal_growth, base_discount_rate,
            mature_revenue_detail.get("current_margin"), mature_revenue_detail.get("target_margin_base"),
            mature_revenue_detail.get("steady_state_year"), shares, 0.0,
        )
        if mature_implied_growth is None:
            notes.append(
                "Olgun revenue-first DCF: fiyatın ima ettiği başlangıç büyüme oranı hesaplanamadı "
                "(fiyat, makul büyüme aralığının dışında bir beklenti ima ediyor olabilir)."
            )

        output_implied = mature_implied_growth
        output_realized_cagr = revenue_cagr
        output_realized_label = f"gelir {revenue_cagr_label}" if revenue_cagr_label else None
        # Mirrors the hyper-grower branch: no bracket-boundary status is
        # exposed by revenue_dcf.implied_start_growth, so this defaults to
        # "ok" (a missing implied growth already gets its own note above).
        output_bracket_status = "ok"
    elif midgrowth_revenue_headline:
        # Mirrors the mature-revenue override: the mid-growth revenue-first
        # DCF's own reverse-DCF solve is revenue-based, so the realized
        # reference is revenue CAGR. Uses the SAME base-scenario inputs the
        # headline band was built from (incl. dilution/financing shares) so
        # the implied growth is apples-to-apples with the published scenarios.
        revenue_cagr = metrics.get("revenue_cagr_5y")
        revenue_cagr_label = "5y" if revenue_cagr is not None else None
        if revenue_cagr is None:
            revenue_cagr = metrics.get("revenue_cagr_3y")
            revenue_cagr_label = "3y" if revenue_cagr is not None else None

        mg_revenue0 = to_annual_series(normalized, "Revenue").get(latest_fy) if latest_fy is not None else None
        mg_implied_growth = revenue_dcf.implied_start_growth(
            price, mg_revenue0, base_terminal_growth, base_discount_rate,
            midgrowth_revenue_detail.get("current_margin"), midgrowth_revenue_detail.get("target_margin_base"),
            midgrowth_revenue_detail.get("steady_state_year"), shares,
            midgrowth_revenue_detail.get("annual_dilution") or 0.0,
            midgrowth_revenue_detail.get("financing_shares") or 0.0,
        )
        if mg_implied_growth is None:
            notes.append(
                "Orta-büyüme revenue-first DCF: fiyatın ima ettiği başlangıç büyüme oranı hesaplanamadı "
                "(fiyat, makul büyüme aralığının dışında bir beklenti ima ediyor olabilir)."
            )

        output_implied = mg_implied_growth
        output_realized_cagr = revenue_cagr
        output_realized_label = f"gelir {revenue_cagr_label}" if revenue_cagr_label else None
        output_bracket_status = "ok"

    # --- Multiples ---------------------------------------------------------
    history = multiples.multiples_history(normalized, price_df)
    if price_df is None or getattr(price_df, "empty", True):
        notes.append("Fiyat geçmişi alınamadığı için çarpan tarihçesi hesaplanamadı.")
    current, current_notes = _derive_current_multiples(normalized, ratios, metrics, price)
    notes.extend(current_notes)
    pe_pct = multiples.percentile_position([h["pe"] for h in history], current["pe"])
    ps_pct = multiples.percentile_position([h["ps"] for h in history], current["ps"])
    pfcf_pct = multiples.percentile_position([h["pfcf"] for h in history], current["pfcf"])

    # Current P/FFO (Sec.8/FFO Step 5): price / ffo_per_share, using the same
    # latest-usable FFO as the reit anchor (_select_latest_ffo) -- computed
    # unconditionally (not gated on sector_type) exactly like pe/ps/pfcf
    # above, so it degrades to None wherever Depreciation data is missing
    # instead of requiring extra sector-specific plumbing here.
    ffo_per_share_current, _ = _select_latest_ffo(normalized, metrics)
    current["pffo"] = (
        round(price / ffo_per_share_current, 4)
        if price is not None and ffo_per_share_current is not None and ffo_per_share_current > 0
        else None
    )
    pffo_pct = multiples.percentile_position([h["pffo"] for h in history], current["pffo"])

    # `sector_medians_result` was already computed earlier in this function
    # (WP2/WP6, right after the `sector_data` load, so both the
    # terminal-growth anchor and the hyper-grower's sector-CapEx/Sales floor
    # could use it before this multiples-comparison block runs) -- reused
    # here as-is rather than recomputed.
    sector_info = {
        "available": sector_medians_result is not None,
        "industry": (sector_medians_result or {}).get("industry"),
        "pe_median": (sector_medians_result or {}).get("pe"),
        "ps_median": (sector_medians_result or {}).get("ps"),
        "pfcf_median": (sector_medians_result or {}).get("pfcf"),
    }

    # --- Growth-adjusted multiple (PEG / growth-adjusted EV/Sales) ----------
    # Refines (never replaces) the raw multiples signal by dividing the raw
    # multiple by the assumptions pipeline's base growth (in % points).
    # Standard mode ranks PEG (= current P/E / base growth); hyper-grower
    # mode -- where P/E is meaningless -- ranks growth-adjusted EV/Sales
    # instead (SPEC.md Sec.6, VALUATION.md Sec.7). Denominator is ALWAYS the
    # base growth_5y, surfaced in the output as `base_growth_pct`.
    growth_adjusted, ga_raw_pct, ga_pct = _build_growth_adjusted(
        history, current, metrics, normalized, base_growth, hyper_growth_active,
        pe_pct, sector_medians_result,
    )
    if growth_adjusted.get("reason"):
        notes.append(growth_adjusted["reason"])

    multiples_out = {
        "history": history,
        "current": current,
        "pe_percentile": pe_pct,
        "ps_percentile": ps_pct,
        "pfcf_percentile": pfcf_pct,
        "pffo_percentile": pffo_pct,
        "history_years": len(history),
        "sector": sector_info,
        "growth_adjusted": growth_adjusted,
    }

    # --- Sensitivity (base scenario only) -----------------------------------
    # F3: use whichever fcf0 the headline fair_value_range actually reflects
    # -- for cyclical filers where the normalized-earnings variant became
    # the headline, the reported grid should match it rather than silently
    # describing the raw (often near-trough) fcf0 instead. Hyper-grower mode
    # keeps this matrix's existing FCF-DCF-based behavior unchanged (its own
    # revenue-first sensitivity lives in each hyper scenario's own band, see
    # _hyper_scenario_band) -- this reported "sensitivity" key is always the
    # standard/cyclical FCF-DCF grid, never the hyper one.
    headline_fcf0 = fcf0
    if sector_type == "cyclical" and normalized_variant is not None:
        headline_fcf0 = normalized_fcf0
    sensitivity_out = sensitivity.sensitivity_matrix(base_assumptions, headline_fcf0, shares, dilution_rate)
    if sensitivity_out is None and headline_fcf0 is not None and shares:
        notes.append("Duyarlılık matrisi hesaplanamadı.")

    # F(2026-07 refinement, Fix D): for cyclical filers, `headline_fcf0`
    # above became `normalized_fcf0` (the cycle-mid normalized FCF-DCF base)
    # whenever that variant was computable, so the sensitivity grid reflects
    # THAT base, not the raw/suppressed one -- only reverse-DCF (which always
    # solves over the raw `fcf0`, see the Reverse DCF section) reflects the
    # raw base. The mature-sector EPV note below (no `normalized_variant`
    # concept for mature filers) is unaffected and correctly describes both
    # as the same raw FCF-DCF base.
    if epv_headline and sector_type == "cyclical":
        notes.append(
            "Duyarlılık tablosu döngü-ortası normalize FCF-DCF tabanını, ters-DCF ise ham (baskılanmış) "
            "FCF tabanını yansıtır; ikisi de manşet EPV çapasından farklıdır ve serbest nakit akışının "
            "neden düşük olduğunu gösteren kanıt olarak korunur."
        )
        notes.append(
            "NOT: Bu çapa, kazanç tabanını son temsili (kârlı) yıllardan alır ve şiddetli döngü diplerini "
            "(ör. bir bellek-glut zarar yılı) tekrar etmeyecek istisna olarak DIŞLAR (yapısal re-rating "
            "varsayımı). Dipleri döngünün kalıcı parçası sayan tam-döngü ortalaması, değeri belirgin "
            "biçimde düşürür."
        )
    elif epv_headline:
        notes.append(
            "Duyarlılık tablosu ve ters-DCF, manşet EPV çapasını değil, ikincil (baskılanmış) FCF-DCF "
            "tabanını yansıtır; serbest nakit akışının neden düşük olduğunu gösteren kanıt olarak "
            "korunmuştur."
        )
    elif cyclical_fcfe_headline:
        notes.append(
            "Duyarlılık tablosu döngü-ortası normalize FCF-DCF tabanını, ters-DCF ise ham (baskılanmış) "
            "FCF tabanını yansıtır; ikisi de manşet FCFE çapasından farklıdır ve serbest nakit akışının "
            "neden düşük olduğunu gösteren kanıt olarak korunur."
        )
        notes.append(
            "NOT: Bu çapa, kazanç tabanını son temsili (kârlı) yıllardan alır ve şiddetli döngü diplerini "
            "(ör. bir bellek-glut zarar yılı) tekrar etmeyecek istisna olarak DIŞLAR (yapısal re-rating "
            "varsayımı). Dipleri döngünün kalıcı parçası sayan tam-döngü ortalaması, değeri belirgin "
            "biçimde düşürür."
        )
    elif mature_revenue_headline:
        notes.append(
            "Duyarlılık tablosu, manşet olgun revenue-first DCF'i değil, ikincil (baskılanmış) FCF-DCF "
            "tabanını yansıtır; serbest nakit akışının neden düşük olduğunu gösteren kanıt olarak "
            "korunmuştur."
        )
    elif midgrowth_revenue_headline:
        notes.append(
            "Duyarlılık tablosu, manşet orta-büyüme revenue-first DCF'i değil, ikincil FCF-DCF tabanını "
            "yansıtır; büyüme-fade modeli için standart büyüme±2pp ızgarası uygulanamadığından FCF-DCF "
            "ızgarası kanıt olarak korunmuştur."
        )

    # --- Triangulation -------------------------------------------------------
    base_band = None
    if primary_dcf_scenarios and primary_dcf_scenarios.get("base"):
        base_band = primary_dcf_scenarios["base"]
    elif reit_or_financial_anchor and (reit_or_financial_anchor.get("scenarios") or {}).get("base"):
        base_band = reit_or_financial_anchor["scenarios"]["base"]

    # In hyper-grower mode, base_band above is already the revenue-first
    # DCF's base scenario band; pass its bull scenario band through too so
    # the DCF signal can distinguish "priced for high expectations" from an
    # outright "pahali" (HYPER_SPEC.md Sec.4). Non-hyper filers keep the
    # unchanged 3-way DCF signal (hyper_growth=False, bull_band=None).
    hyper_bull_band = None
    if hyper_growth_active and not hyper_growth_detail.get("suppressed"):
        hyper_bull_band = (hyper_growth_detail.get("scenarios") or {}).get("bull")

    triangulation = triangulate.triangulate(
        price, base_band, output_implied, output_realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type,
        hyper_growth=hyper_growth_active, bull_band=hyper_bull_band, reverse_dcf_status=output_bracket_status,
        raw_growth_pair_pct=ga_raw_pct, growth_adj_pct=ga_pct, earnings_power_headline=epv_headline,
        mature_revenue_headline=mature_revenue_headline, midgrowth_revenue_headline=midgrowth_revenue_headline,
        pffo_pct=pffo_pct, cyclical_fcfe_headline=cyclical_fcfe_headline,
    )

    return {
        "sector_type": sector_type,
        "fcf0": fcf0,
        "fcf0_source": fcf0_source,
        "dcf": {
            "enabled": dcf_enabled,
            "disabled_reason": disabled_reason,
            "scenarios": dcf_scenarios,
            "normalized_variant": normalized_variant,
            "high_growth_flag": dcf_high_growth_flag,
        },
        "pb_roe": pb_roe,
        "ffo": ffo,
        "earnings_power": earnings_power,
        "earnings_power_headline": epv_headline,
        "fair_value_range": fair_value_range,
        "reverse_dcf": {
            "implied_growth": _round_or_none(output_implied, 4),
            "realized_cagr_5y": _round_or_none(output_realized_cagr, 4),
            "realized_label": output_realized_label,
            "bracket_status": output_bracket_status,
        },
        "multiples": multiples_out,
        "sensitivity": sensitivity_out,
        "triangulation": triangulation,
        "hyper_growth": hyper_growth_active,
        "hyper_growth_detail": hyper_growth_detail if hyper_growth_active else None,
        "mature_revenue_headline": mature_revenue_headline,
        "mature_revenue_detail": mature_revenue_detail,
        "midgrowth_revenue_headline": midgrowth_revenue_headline,
        "midgrowth_revenue_detail": midgrowth_revenue_detail,
        "cyclical_fcfe_headline": cyclical_fcfe_headline,
        "cyclical_fcfe_detail": cyclical_fcfe_detail,
        "assumptions": assumptions,
        "notes": notes,
    }
