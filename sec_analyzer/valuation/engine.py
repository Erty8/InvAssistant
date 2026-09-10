"""Orchestrate the deterministic valuation engine.

:func:`run_valuation` is the single entry point the interpret layer's phase
2, the CLI verdict card, the HTML report, and the store all consume. It
wires together every other module in this package (DCF, reverse-DCF,
multiples, Damodaran sector medians, sensitivity, triangulation) around one
already-validated assumption set and returns the ``valuation`` dict
documented in ``sec_analyzer/valuation/SPEC.md`` Sec.11.

This module never raises for missing or malformed *data* -- every
unavailable piece becomes ``None`` plus a note in the returned
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
    cyclical, damodaran, dcf, distress, lbo, multiples, precedent_transactions, reverse_dcf,
    revenue_dcf, sanity, sector, sensitivity, triangulate,
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

#: Sectors for which enterprise value is undefined (SPEC.md Sec.20b). EV adds
#: net debt to market cap to value the whole capital structure; for a
#: deposit-funded lender, borrowing IS the raw material of the business, so
#: EV/EBITDA, EV/EBIT and EV/Sales carry no meaning. REITs are deliberately
#: NOT here -- EV multiples are standard practice for them.
_SECTORS_WITHOUT_EV = ("financial",)

#: note emitted when EV multiples are suppressed (SPEC.md Sec.20b).
_EV_SUPPRESSED_NOTE = (
    "Enterprise value (EV) is undefined for financial institutions -- deposits and borrowing are the "
    "raw material of the business, not a capital-structure adjustment. EV/EBITDA, EV/EBIT, and "
    "EV/Sales multiples were not computed."
)

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
#: caller (``_build_hyper_growth``) attaches a note and a
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
#: base-scenario revenue_multiple <= 8 -> "fair"; 8 < m <= 15 -> "aggressive";
#: m > 15 -> "excessively_aggressive".
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
#: constant itself and attaches a note plus a ``target_margin_flag``
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
#: note plus ``target_margin_flag`` instead of silently truncating.
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
        ``[0.0, _HYPER_DILUTION_CAP]``). ``note`` is a string to
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
            "SBC-driven issuance was excluded from the dilution projection since it's already priced as "
            "an expense in the margin (avoiding double counting); the remaining dilution reflects only "
            "non-SBC issuance."
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
        "rim": None,
        "cycle": None,
        "ffo": None,
        "earnings_power": None,
        "earnings_power_headline": False,
        "fair_value_range": _empty_fair_value_range(),
        "reverse_dcf": {
            "implied_growth": None, "realized_cagr_5y": None, "realized_label": None, "bracket_status": "no_data",
        },
        "multiples": {
            "history": [],
            "current": {
                "pe": None, "ps": None, "pfcf": None, "pffo": None,
                "ev_ebit": None, "ev_ebitda": None, "ptbv": None,
            },
            "pe_percentile": None,
            "ps_percentile": None,
            "pfcf_percentile": None,
            "pffo_percentile": None,
            "ev_ebit_percentile": None,
            "ev_ebitda_percentile": None,
            "ptbv_percentile": None,
            "net_debt_to_ebitda": None,
            "leveraged": False,
            "ev_applicable": True,
            "history_years": 0,
            "sector": {
                "available": False, "industry": None,
                "pe_median": None, "ps_median": None, "pfcf_median": None,
                "ev_ebitda_median": None, "precedent_transactions": None,
                "comparison": {"label": None, "current": None, "median": None, "ratio": None, "bucket": None},
            },
            "growth_adjusted": _empty_growth_adjusted("peg", "PEG", "P/E", None),
        },
        "sensitivity": None,
        "triangulation": {
            "signals": {"dcf": "no_data", "reverse_dcf": "no_data", "multiples": "no_data"},
            "confidence": "LOW",
            "direction": "unclear",
            "divergence": None,
        },
        "hyper_growth": False,
        "hyper_growth_detail": None,
        "mature_revenue_headline": False,
        "mature_revenue_detail": None,
        "midgrowth_revenue_headline": False,
        "midgrowth_revenue_detail": None,
        "cyclical_fcfe_headline": False,
        "cyclical_fcfe_detail": None,
        "altman_z": None,
        "beneish_m": None,
        "merton_dtd": None,
        "lbo_floor_detail": None,
        "method_summary": [],
        "assumptions": assumptions or {},
        "notes": ["The valuation engine encountered an unexpected error; results may be incomplete."],
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
    deviates more than 50% from the average of the PRIOR years in the
    window (``latest_fy-1``/``latest_fy-2``, whichever exist; no prior year
    -> deviation not assessable, never fires) -- UNLESS the trailing 3
    fiscal years form a monotonic ramp (see below), in which case the
    deviation is trusted as genuine structural growth/decline rather than a
    one-off spike, and the latest-FY figure is kept. The reference
    deliberately EXCLUDES the candidate year (SPEC Sec.4, 2026-07 fix):
    including it turned the documented 50% into an effective ~+100%. The
    fallback VALUE stays the inclusive 3-year average. The 3y-average
    fallback is only reachable when it is itself usable (positive); if it
    isn't, a positive ttm figure is still preferred over giving up
    entirely. Returns ``(fcf0, source, note)`` where ``source`` is
    ``"ttm"``/``"3y_avg"``/``None`` and ``note`` is a string to
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

    # SPEC Sec.4 (2026-07 fix): the deviation REFERENCE is the prior years
    # only. Including the candidate year diluted the documented 50% into an
    # effective "latest > sum of the two prior years" (~+100%) -- a reference
    # must not contain the candidate it judges. The fallback VALUE below
    # stays the inclusive 3-year average (smoothing, not exclusion).
    prior_values = [v for v in window[1:] if v is not None]
    prior_avg = sum(prior_values) / len(prior_values) if prior_values else None

    deviates = False
    if ttm_usable and prior_avg is not None and prior_avg != 0:
        deviates = abs(ttm_fcf - prior_avg) / abs(prior_avg) > _FCF0_DEVIATION_THRESHOLD

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
            "The latest year's FCF (net of SBC) deviated more than 50% from the 3-year average; but since "
            "FCF follows a stable trend (a structural rise/fall, not a one-off spike) the latest year's "
            "figure was still used as the starting FCF (fcf0) for the DCF."
        )
        return ttm_fcf, "ttm", note

    if avg_usable:
        note = (
            "The 3-year average FCF (net of SBC) was used as the starting FCF (fcf0) for the DCF instead of "
            "the latest year's figure (the latest year's data is missing, negative, or deviated more than "
            "50% from the 3-year average)."
        )
        return avg_fcf, "3y_avg", note

    if ttm_usable:
        # Deviates from the average, but the average itself isn't usable
        # (missing or non-positive) -- keep the positive ttm figure rather
        # than discarding perfectly usable data.
        return ttm_fcf, "ttm", None

    return None, None, "A positive starting FCF (fcf0) could not be computed; the DCF cannot be produced for this company."


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
    reporting-only signal (one note naming the triggering
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
            notes.append(f"DCF assumptions for the {key.capitalize()} scenario are missing or invalid.")
            continue

        try:
            result = dcf_per_share(fcf0, growth_5y, terminal_growth, discount_rate, shares, dilution_rate)
        except ValueError as exc:
            scenarios[key] = {"per_share": None, "lo": None, "hi": None}
            notes.append(f"DCF could not be computed for the {key.capitalize()} scenario: {exc}")
            continue

        per_share = round(result["per_share"], 2)
        lo, hi, used_fallback = _dcf_scenario_band(
            fcf0, growth_5y, terminal_growth, discount_rate, shares, dilution_rate, per_share
        )
        if used_fallback:
            notes.append(
                f"The sensitivity band for the {key.capitalize()} scenario could not be computed; "
                "used +/-10% of the point estimate as a fallback."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    if high_growth_keys:
        names = [key.capitalize() for key in high_growth_keys]
        if len(names) == 1:
            scenario_phrase = f"in the {names[0]} scenario"
        else:
            scenario_phrase = f"in the {', '.join(names[:-1])} and {names[-1]} scenarios"
        notes.append(
            f"The 5-year growth assumption {scenario_phrase} exceeds 40%; unlike the hyper/mid-growth "
            "revenue-first models, the standard two-stage DCF has no arrival-point (TAM share/revenue "
            "multiple) safety check -- it may help to cross-check this scenario against the revenue-first "
            "/ reverse-DCF."
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
    returns ``None`` plus a note rather than a nonsensical
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
        notes.append("Cyclical normalized FCF could not be computed: the latest year's revenue is missing or negative.")
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
        notes.append("Cyclical normalized FCF could not be computed: not enough FCF margin history.")
        return None, notes

    # Mid-to-upper cycle: average the top ceil(N/2) margins rather than
    # taking the median, so a single trough year can't drag the
    # "normalized" figure down to the raw current-year number.
    k = math.ceil(len(margins) / 2)
    top_margins = sorted(margins, reverse=True)[:k]
    normalized_margin = sum(top_margins) / len(top_margins)

    if normalized_margin <= 0:
        notes.append(
            "Cyclical normalized FCF is not meaningful: the upper-half (mid/peak cycle) average FCF "
            "margin is not positive."
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
    outside that reference band appends a note and sets the returned
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
        notes.append("P/B x ROE anchor could not be computed: missing or invalid input (share count/discount rate).")
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
        notes.append("P/B x ROE anchor could not be computed: ROE or equity data is missing.")
        return None, notes

    latest_fy = resolve_fundamental_fy(metrics)
    if latest_fy is not None and selected_fy != latest_fy:
        notes.append(
            f"Used fiscal year {selected_fy}'s equity/ROE data for the P/B x ROE anchor "
            "(the latest fiscal year's fundamentals weren't aligned with the share count)."
        )

    fair_pb_base = _justified_pb(roe, discount_rate_base, terminal_growth_base)
    book_value_per_share = equity_latest / shares

    # A justified P/B of (ROE - g)/(r - g) is non-positive exactly when ROE <= g
    # (the numerator turns non-positive for a loss-making or sub-growth filer;
    # the denominator is always positive here since discount_rate_base > 0 and
    # _justified_pb degrades g to 0 whenever r - g would be <= 0). A book value
    # per share <= 0 (negative equity) is likewise degenerate. A price-to-book
    # multiple applied to book equity can never make the stock worth zero or
    # negative dollars, so the anchor has NO usable opinion in either case --
    # return None (anchor unavailable) rather than emitting a negative fair
    # value. This is distinct from the Work-Package-5 "don't clamp a
    # legitimately high/low POSITIVE fair_pb" rule below (SPEC.md Sec.8): that
    # rule is about not discarding valid positive signal, not about permitting
    # an economically meaningless non-positive multiple.
    if fair_pb_base <= 0 or book_value_per_share <= 0:
        if fair_pb_base <= 0:
            notes.append(
                f"P/B x ROE anchor could not be computed: justified P/B {fair_pb_base:.2f}x "
                f"is not positive (ROE {roe * 100:.1f}% <= growth {(terminal_growth_base or 0) * 100:.1f}%); "
                "there is no meaningful justified P/B x ROE value for a loss-making company or one earning "
                "below its growth rate."
            )
        else:
            notes.append(
                f"P/B x ROE anchor could not be computed: book value per share {book_value_per_share:.2f} "
                "is not positive (negative equity); the P/B anchor is meaningless."
            )
        return None, notes

    justified_pb_flag = None
    if fair_pb_base > _PB_CLAMP_HI:
        justified_pb_flag = "above_reference"
    elif fair_pb_base < _PB_CLAMP_LO:
        justified_pb_flag = "below_reference"
    if justified_pb_flag is not None:
        notes.append(
            f"Justified P/B {fair_pb_base:.2f}x is outside the usual [{_PB_CLAMP_LO:.1f}, "
            f"{_PB_CLAMP_HI:.1f}] reference band (ROE {roe * 100:.1f}%, "
            f"discount rate {discount_rate_base * 100:.1f}%); a high/low ROE can legitimately justify "
            "this -- not clamped to a fixed bound."
        )

    scenarios = {}
    for key, scale in _PB_SCENARIO_SCALE.items():
        per_share = round(fair_pb_base * scale * book_value_per_share, 2)
        lo, hi, used_fallback = _pb_roe_scenario_band(
            roe, discount_rate_base, scale, book_value_per_share, per_share, terminal_growth_base
        )
        if used_fallback:
            notes.append(
                f"The P/B x ROE sensitivity band for the {key.capitalize()} scenario could not be computed; "
                "used +/-10% of the point estimate as a fallback."
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


def _rim_scenario_band(
    bve0: float,
    ni0: float,
    roe: float,
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares: float,
    per_share: float,
) -> "tuple[float, float, bool]":
    """Derive one RIM scenario's band from ``discount_rate +/-
    _DISCOUNT_RATE_STEP`` (Sec.8f), mirroring
    :func:`_cyclical_fcfe_scenario_band`/:func:`_pb_roe_scenario_band`:
    recompute :func:`dcf.rim_per_share` at each of the 3 nearby discount
    rates (``growth_5y``/``terminal_growth`` held fixed at this scenario's
    own values, ``terminal_roe=dr`` passed through for each nearby rate so
    the terminal phase fades to that same nearby rate), and take the
    min/max. Falls back to the flat +/-10% band (:func:`_band`) when fewer
    than :data:`_MIN_GRID_CELLS_FOR_BAND` discount-rate points are usable (a
    rate that doesn't clear ``r > terminal_growth``, or a failed call, is
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
            result = dcf.rim_per_share(
                bve0, ni0, roe, growth_5y, terminal_growth, dr, shares, terminal_roe=dr
            )
        except ValueError:
            continue
        cells.append(round(result["per_share"], 2))

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


#: Relative gap between TTM and latest-FY net income above which the
#: fiscal-year base is called out as stale (SPEC.md Sec.24c).
_TTM_STALENESS_THRESHOLD = 0.25


def _ttm_staleness_notes(metrics: dict) -> List[str]:
    """Warn when the annual base the anchors use is materially out of date.

    Fires in both directions -- a filer mid-upswing (Micron: TTM net income
    5.9x the latest fiscal year) and one mid-collapse are equally misdescribed
    by a stale annual figure. Reports both P/E readings, since the fiscal-year
    P/E is what the rest of the card shows. Never raises.
    """
    metrics = metrics or {}
    if not metrics.get("ttm_complete"):
        return []
    ratio = metrics.get("ttm_vs_fy_net_income")
    if not _is_number(ratio) or abs(ratio - 1.0) <= _TTM_STALENESS_THRESHOLD:
        return []

    ttm_ni, pe_ttm, pe_fy = metrics.get("ttm_net_income"), metrics.get("pe_ttm"), metrics.get("pe")
    direction = "above" if ratio > 1 else "below"
    note = (
        f"The trailing-12-month (TTM) figure is markedly {direction} the fiscal-year basis the valuation "
        f"anchors use: TTM net income is {_format_usd_short(ttm_ni)} "
        f"({ratio:.1f}x the fiscal year, period ending {metrics.get('ttm_period_end')}). "
        "Every anchor is computed from the annual series, so it doesn't yet see these quarters."
    )
    if _is_number(pe_ttm) and _is_number(pe_fy):
        note += f" P/E is {pe_fy:.1f} on a fiscal-year basis, {pe_ttm:.1f} on a TTM basis."
    return [note]


def _format_usd_short(value: float) -> str:
    """Compact USD magnitude for a note, e.g. ``"$32.8B"``.

    Uses a decimal POINT, matching every other number engine.py formats into
    a note (``{x:.1f}%`` percentages, per-share dollar figures), so a
    single sentence never mixes separators.
    """
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.0f}M"
    return f"${value:.0f}"


def _rim_tangible(ratio_by_fy: dict, selected_fy: Optional[int], key: str) -> Optional[float]:
    """Read a tangible-equity figure for the fiscal year the RIM anchor used.

    Returns ``None`` when the year has no ratio row or no usable value, so a
    missing goodwill/intangibles series degrades to "not reported" rather than
    to a figure describing a different period (SPEC.md Sec.23d).
    """
    value = (ratio_by_fy.get(selected_fy) or {}).get(key)
    return value if _is_number(value) else None


def _build_rim_external_growth(
    bve0: float, ni0: float, roe: float, growth_5y: float,
    terminal_growth: float, discount_rate: float, shares: float,
) -> "tuple[Optional[dict], List[str]]":
    """Externally-funded-growth diagnostic for the RIM anchor (SPEC.md Sec.22b).

    Advisory only: it never headlines ``fair_value_range``, never feeds
    ``primary_dcf_scenarios``, and never enters triangulation -- same
    discipline as the Sec.8g/8j/8k screens. Its job is to quantify what the
    internal-funding cap suppressed, so the reader can see whether the
    discarded growth assumption would have HELPED or HURT.

    Never raises: an invalid input degrades to ``(None, [note])``.
    """
    try:
        result = dcf.rim_external_growth_per_share(
            bve0, ni0, roe, growth_5y, terminal_growth, discount_rate, shares,
            terminal_roe=discount_rate,
        )
    except ValueError as exc:
        return None, [f"The externally-funded-growth scenario could not be computed: {exc}"]

    return (
        {
            "per_share": round(result["per_share"], 2),
            "per_share_internal": round(result["per_share_internal"], 2),
            "value_gap_per_share": round(result["value_gap_per_share"], 2),
            "external_funding_total": result["external_funding_total"],
        },
        [],
    )


def _build_rim(
    assumptions: dict, normalized: dict, metrics: dict, ratios: list
) -> "tuple[Optional[dict], List[str]]":
    """Residual income model (RIM) anchor for the financial sector (SPEC.md
    Sec.8f) -- a growth-fading, multi-period alternative to
    :func:`_build_pb_roe`'s single-period justified-P/B heuristic
    (``(ROE - g) / (r - g)`` applied as one static multiple to today's book
    value forever). RIM instead compounds a 10-year residual-income path
    (:func:`dcf.rim_per_share`, sharing the ``g_eff = min(g, roe)``
    reinvestment cap and ``terminal_roe``-fade convention already used by
    :func:`_build_cyclical_fcfe`), so growth and discount-rate assumptions
    actually FADE over the projection instead of being baked into one
    perpetual multiple.

    This is the primary anchor for ``sector_type == "financial"``; the
    caller (``_run_valuation``) falls back to :func:`_build_pb_roe` when
    this returns ``None`` (e.g. a filer with too little equity/ROE/net-income
    history), exactly mirroring the existing ``reit``-FFO-unavailable
    fallback to P/B x ROE. ``reit`` is unaffected by this function -- it
    keeps using FFO/Gordon-growth (and P/B x ROE as ITS fallback) since GAAP
    real-estate depreciation already makes book value/net income unreliable
    inputs for a REIT's own residual-income compounding.

    Selects its fiscal year the same way :func:`_build_pb_roe` does: walks
    the equity series from the newest fiscal year down and picks the first
    one that ALSO has both a net-income figure (``normalized``) and a ROE
    figure (``ratios``), independent of ``metrics["latest_fy"]`` (see
    :func:`_build_pb_roe`'s docstring for the JPM-style edge case this
    guards against). The share count still comes from ``metrics["shares"]``
    (today's point-in-time count).

    Args:
        assumptions: The phase-1 bear/base/bull assumption dict; each
            scenario's own ``growth_5y``/``terminal_growth``/
            ``discount_rate`` drive its RIM projection (``discount_rate``
            doubles as ``terminal_roe``, same convention as
            :func:`_build_cyclical_fcfe`).
        normalized: The dict returned by ``normalize_facts`` (reads
            ``StockholdersEquity``/``NetIncome`` annual series).
        metrics: Used for ``shares`` and to resolve the fiscal year via
            ``resolve_fundamental_fy`` (for the "which FY was used" note
            only -- FY SELECTION itself walks ``normalized``/``ratios``
            independently, see above).
        ratios: Supplies the per-fiscal-year ``roe`` figure.

    Returns:
        A ``(result, notes)`` tuple. ``result`` is ``None`` if the anchor
        can't be computed at all (missing shares/equity/net income/ROE, or
        no scenario is computable), else a dict with ``scenarios`` (bear/
        base/bull ``{"per_share", "lo", "hi"}``), ``per_share`` (the base
        scenario's point estimate), ``bve0``, ``book_value_per_share``,
        ``normalized_net_income``, and ``roe``. Never raises.
    """
    notes: List[str] = []
    shares = metrics.get("shares")
    if not shares or shares <= 0:
        notes.append("RIM anchor could not be computed: no valid share count.")
        return None, notes

    equity_series = to_annual_series(normalized, "StockholdersEquity")
    ni_series = to_annual_series(normalized, "NetIncome")
    roe_by_fy = {row.get("fy"): row.get("roe") for row in (ratios or []) if row.get("fy") is not None}
    ratio_by_fy = {row.get("fy"): row for row in (ratios or []) if row.get("fy") is not None}

    selected_fy, bve0, ni0, roe = None, None, None, None
    for fy in sorted(equity_series, reverse=True):
        eq = equity_series.get(fy)
        ni = ni_series.get(fy)
        candidate_roe = roe_by_fy.get(fy)
        if eq is not None and eq > 0 and ni is not None and candidate_roe is not None:
            selected_fy, bve0, ni0, roe = fy, eq, ni, candidate_roe
            break

    if selected_fy is None:
        notes.append("RIM anchor could not be computed: equity, net income, or ROE data is missing.")
        return None, notes

    latest_fy = resolve_fundamental_fy(metrics)
    if latest_fy is not None and selected_fy != latest_fy:
        notes.append(
            f"Used fiscal year {selected_fy}'s equity/net income/ROE data for the RIM anchor "
            "(the latest fiscal year's fundamentals weren't aligned with the share count)."
        )

    if roe <= 0:
        notes.append(f"RIM anchor could not be computed: ROE ({roe * 100:.1f}%) is not positive.")
        return None, notes

    book_value_per_share = bve0 / shares

    scenarios: Dict[str, dict] = {}
    growth_capped = False
    effective_growth_5y = None
    assumed_growth_5y = None
    external_growth = None
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
            notes.append(f"RIM assumptions for the {key.capitalize()} scenario are missing or invalid.")
            continue

        try:
            result = dcf.rim_per_share(
                bve0, ni0, roe, growth_5y, terminal_growth, discount_rate, shares,
                terminal_roe=discount_rate,
            )
        except ValueError as exc:
            scenarios[key] = {"per_share": None, "lo": None, "hi": None}
            notes.append(f"RIM could not be computed for the {key.capitalize()} scenario: {exc}")
            continue

        # SPEC.md Sec.22c: the base scenario drives the disclosure, since it
        # is the one the headline fair_value_range reports.
        if key == "base":
            growth_capped = bool(result.get("growth_capped"))
            effective_growth_5y = result.get("effective_growth_5y")
            assumed_growth_5y = growth_5y
            if growth_capped:
                external_growth, external_notes = _build_rim_external_growth(
                    bve0, ni0, roe, growth_5y, terminal_growth, discount_rate, shares,
                )
                notes.extend(external_notes)

        per_share = round(result["per_share"], 2)
        lo, hi, used_fallback = _rim_scenario_band(
            bve0, ni0, roe, growth_5y, terminal_growth, discount_rate, shares, per_share
        )
        if used_fallback:
            notes.append(
                f"The RIM sensitivity band for the {key.capitalize()} scenario could not be computed; "
                "used +/-10% of the point estimate as a fallback."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    if not any(_is_number(cell.get("per_share")) for cell in scenarios.values()):
        return None, notes

    if growth_capped and _is_number(assumed_growth_5y) and _is_number(effective_growth_5y):
        note = (
            f"The RIM growth assumption hit the internal-funding constraint: applied "
            f"{effective_growth_5y * 100:.1f}% growth instead of the assumed {assumed_growth_5y * 100:.1f}% "
            f"(g = b x ROE, ROE {roe * 100:.1f}% -- even reinvesting all of its earnings, the company "
            "can't self-fund growth faster than its ROE)."
        )
        if external_growth:
            note += (
                f" If the growth were assumed to be funded entirely by external equity issuance "
                f"(~{_format_usd_short(external_growth['external_funding_total'])} in new equity over 10 "
                f"years), per-share value would be ${external_growth['per_share']:.2f} instead of "
                f"${external_growth['per_share_internal']:.2f}, since ROE is below the cost of equity -- "
                "growth at this return level destroys value rather than creating it."
            )
        notes.append(note)

    return (
        {
            "scenarios": scenarios,
            "per_share": scenarios["base"]["per_share"],
            "bve0": bve0,
            "book_value_per_share": book_value_per_share,
            "normalized_net_income": ni0,
            "roe": round(roe, 4),
            # SPEC.md Sec.23d -- advisory only, read from the SAME fiscal year
            # the anchor selected, so they cannot describe a different period
            # than `roe`/`bve0`. These never feed the projection: residual
            # income is accounting-invariant, so rebasing the anchor on
            # tangible equity would move the value only through the 10-year
            # truncation (see Sec.23's opening argument).
            "tangible_equity": _rim_tangible(ratio_by_fy, selected_fy, "tangible_equity"),
            "tbv_per_share": (
                _rim_tangible(ratio_by_fy, selected_fy, "tangible_equity") / shares
                if _rim_tangible(ratio_by_fy, selected_fy, "tangible_equity") else None
            ),
            "rotce": _rim_tangible(ratio_by_fy, selected_fy, "rotce"),
            # SPEC.md Sec.22c -- observational/advisory only.
            "growth_capped": growth_capped,
            "effective_growth_5y": effective_growth_5y,
            "assumed_growth_5y": assumed_growth_5y,
            "external_growth": external_growth,
        },
        notes,
    )


_ALTMAN_ZONE_NOTE = {
    "safe": "Altman Z-score is in the safe zone (low bankruptcy risk).",
    "grey": "Altman Z-score is in the grey zone (uncertain bankruptcy risk -- worth monitoring).",
    "distress": "Altman Z-score is in the distress zone (high bankruptcy-risk signal).",
}


def _build_altman_z(normalized: dict, metrics: dict) -> "tuple[Optional[dict], List[str]]":
    """Altman Z-score distress screen (SPEC.md Sec.8g) -- an ADVISORY-ONLY
    bankruptcy-risk overlay. Never headlines ``fair_value_range`` and never
    participates in ``triangulate.triangulate``'s confidence vote; it exists
    purely to flag distress risk alongside the valuation.

    Not called for ``financial``/``reit`` filers (the caller,
    ``_run_valuation``, gates this via :data:`_SECTORS_WITHOUT_FCF_DCF`) --
    the classic Altman model was calibrated on industrial/manufacturing
    balance sheets, and both structural leverage (banks) and GAAP real-estate
    depreciation (REITs) make its ratios not meaningful for those sectors,
    exactly the same rationale the FCF-DCF disablement already documents.

    Args:
        normalized: The dict returned by ``normalize_facts`` (reads
            ``CurrentAssets``/``CurrentLiabilities``/``TotalAssets``/
            ``TotalLiabilities``/``RetainedEarningsAccumulatedDeficit``/
            ``OperatingIncome`` (EBIT proxy)/``Revenue`` annual series).
        metrics: Used to resolve the fiscal year (``resolve_fundamental_fy``)
            and for ``metrics["market_cap"]``.

    Returns:
        A ``(result, notes)`` tuple. ``result`` is ``None`` when the fiscal
        year can't be resolved, working-capital inputs are missing, or
        :func:`distress.altman_z_score` itself returns ``None`` (missing
        input or non-positive total assets/liabilities); else the dict
        :func:`distress.altman_z_score` returns (``z_score``, ``zone``,
        ``components``). Never raises.
    """
    notes: List[str] = []
    fy = resolve_fundamental_fy(metrics)
    if fy is None:
        return None, notes

    current_assets = to_annual_series(normalized, "CurrentAssets").get(fy)
    current_liabilities = to_annual_series(normalized, "CurrentLiabilities").get(fy)
    if current_assets is None or current_liabilities is None:
        notes.append("Altman Z-score could not be computed: current-asset/current-liability data is missing.")
        return None, notes
    working_capital = current_assets - current_liabilities

    total_assets = to_annual_series(normalized, "TotalAssets").get(fy)
    total_liabilities = to_annual_series(normalized, "TotalLiabilities").get(fy)
    retained_earnings = to_annual_series(normalized, "RetainedEarningsAccumulatedDeficit").get(fy)
    ebit = to_annual_series(normalized, "OperatingIncome").get(fy)
    revenue = to_annual_series(normalized, "Revenue").get(fy)
    market_cap = metrics.get("market_cap")

    result = distress.altman_z_score(
        working_capital, total_assets, retained_earnings, ebit, market_cap, total_liabilities, revenue
    )
    if result is None:
        notes.append(
            "Altman Z-score could not be computed: required data is missing, or total assets/liabilities are not positive."
        )
        return None, notes

    notes.append(f"{_ALTMAN_ZONE_NOTE[result['zone']]} (Z={result['z_score']})")
    return result, notes


#: SGI (sales growth index) above which a Beneish manipulation flag is
#: caveated as a possible high-growth artifact (SPEC.md Sec.8j / I2): SGI is
#: revenue_t/revenue_t-1, so 1.40 == ~40% YoY sales growth. Beneish's SGI and
#: DSRI both rise mechanically with fast growth and carry positive
#: coefficients, so genuine hyper-growers (e.g. NVDA) can trip the -1.78
#: threshold without manipulating earnings.
_BENEISH_HIGH_GROWTH_SGI = 1.40

#: Concepts pulled per fiscal year for the Beneish M-score (SPEC.md Sec.8j).
#: ``sga``/``long_term_debt``/``current_liabilities``/``net_income``/
#: ``operating_cash_flow`` gate the full 8-variable vs. 5-variable model
#: choice inside ``distress.beneish_m_score`` itself.
_BENEISH_CONCEPT_MAP = {
    "receivables": "Receivables",
    "revenue": "Revenue",
    "gross_profit": "GrossProfit",
    "current_assets": "CurrentAssets",
    "ppe_gross": "PropertyPlantAndEquipmentGross",
    "total_assets": "TotalAssets",
    "depreciation": "Depreciation",
    "sga": "SellingGeneralAndAdministrativeExpense",
    "long_term_debt": "LongTermDebt",
    "current_liabilities": "CurrentLiabilities",
    "net_income": "NetIncome",
    "operating_cash_flow": "OperatingCashFlow",
}


def _build_beneish_m(normalized: dict, metrics: dict) -> "tuple[Optional[dict], List[str]]":
    """Beneish M-score earnings-manipulation screen (SPEC.md Sec.8j) -- an
    ADVISORY-ONLY overlay, same non-chain-touching contract as
    :func:`_build_altman_z`: never headlines `fair_value_range`, never
    participates in `triangulate.triangulate`'s confidence vote.

    Pulls two consecutive fiscal years (the resolved fundamental FY and the
    year immediately before it) of raw figures and delegates the actual
    8-variable-vs-5-variable model choice and math to
    :func:`distress.beneish_m_score`.

    Args:
        normalized: The dict returned by ``normalize_facts`` (reads every
            concept in :data:`_BENEISH_CONCEPT_MAP`'s annual series).
        metrics: Used to resolve the current fiscal year via
            ``resolve_fundamental_fy``; the prior year is simply
            ``fy - 1``.

    Returns:
        A ``(result, notes)`` tuple. ``result`` is ``None`` when the fiscal
        year can't be resolved, or :func:`distress.beneish_m_score` itself
        returns ``None`` (missing required input for even the 5-variable
        model); else that function's result dict. Never raises.
    """
    notes: List[str] = []
    fy = resolve_fundamental_fy(metrics)
    if fy is None:
        return None, notes
    prior_fy = fy - 1

    series = {key: to_annual_series(normalized, concept) for key, concept in _BENEISH_CONCEPT_MAP.items()}
    current = {key: s.get(fy) for key, s in series.items()}
    prior = {key: s.get(prior_fy) for key, s in series.items()}

    result = distress.beneish_m_score(current, prior)
    if result is None:
        notes.append(
            "Beneish M-score could not be computed: required data for two consecutive fiscal years is missing."
        )
        return None, notes

    partial_note = " (partial -- 5-variable model; SG&A/leverage/accrual data is missing)" if result["partial"] else ""
    flag_note = ""
    if result["flag"]:
        flag_note = " -- possible earnings-manipulation signal"
        # I2 caveat: Beneish systematically OVER-flags fast-growing firms --
        # SGI (sales growth index) and DSRI both rise mechanically with rapid
        # growth and carry positive coefficients, so a high-growth filer can
        # trip the -1.78 threshold without any manipulation. When sales grew
        # aggressively (SGI above the caveat threshold), surface that the
        # flag may be a growth artifact rather than a red flag.
        sgi = (result.get("components") or {}).get("sgi")
        if _is_number(sgi) and sgi > _BENEISH_HIGH_GROWTH_SGI:
            flag_note += (
                f" (NOTE: sales grew rapidly [SGI={sgi:.2f}]; Beneish structurally over-flags fast-growing "
                "companies -- this may be a growth side effect)"
            )
    notes.append(f"Beneish M-score {result['m_score']}{partial_note}{flag_note}.")
    return result, notes


_MERTON_ZONE_NOTE = {
    "safe": "Merton distance-to-default model is in the safe zone.",
    "elevated": "Merton distance-to-default model shows elevated default risk -- worth monitoring.",
    "distress": "Merton distance-to-default model signals high default risk.",
}


def _build_merton_dtd(
    metrics: dict, price_df, risk_free_pct: Optional[float]
) -> "tuple[Optional[dict], List[str]]":
    """Merton distance-to-default (SPEC.md Sec.8k) -- an ADVISORY-ONLY
    overlay, same non-chain-touching contract as `_build_altman_z`/
    `_build_beneish_m`: never headlines `fair_value_range`, never
    participates in `triangulate.triangulate`'s confidence vote.

    Args:
        metrics: Reads `market_cap` (equity value) and `total_debt` (debt
            face-value proxy).
        price_df: Passed to `distress._annualized_volatility` for the
            equity-volatility input (a ~1-year window, NOT the technical/
            momentum subsystem's 20-day `volatility_20d`).
        risk_free_pct: The Damodaran sector risk-free rate (a PERCENTAGE
            number, e.g. `4.5` for 4.5%), already resolved earlier in
            `_run_valuation`. `None` -> anchor unavailable (this engine
            never guesses a risk-free rate for this model).

    Returns:
        A `(result, notes)` tuple. `result` is `None` when `risk_free_pct`
        is missing, `_annualized_volatility` can't produce an equity-vol
        estimate (insufficient price history), or
        `distress.merton_distance_to_default` itself returns `None`
        (missing/degenerate input, or solver non-convergence); else that
        function's result dict. Never raises.
    """
    notes: List[str] = []
    if risk_free_pct is None:
        notes.append("Merton distance-to-default could not be computed: no risk-free rate available.")
        return None, notes

    equity_vol = distress._annualized_volatility(price_df)
    if equity_vol is None:
        notes.append(
            "Merton distance-to-default could not be computed: not enough price history (equity volatility)."
        )
        return None, notes

    result = distress.merton_distance_to_default(
        metrics.get("market_cap"), equity_vol, metrics.get("total_debt"), risk_free_pct / 100.0,
    )
    if result is None:
        notes.append(
            "Merton distance-to-default could not be computed: required data is missing/invalid, or the "
            "numerical solver did not converge."
        )
        return None, notes

    notes.append(
        f"{_MERTON_ZONE_NOTE[result['zone']]} (DD={result['distance_to_default']}, "
        f"PD={result['probability_of_default'] * 100:.2f}%)"
    )
    return result, notes


def _build_lbo_floor(metrics: dict, fcf0: Optional[float]) -> "tuple[Optional[dict], List[str]]":
    """LBO-implied floor value (SPEC.md Sec.8h) -- an ADVISORY-ONLY,
    private-equity-return-based value floor. Never headlines
    ``fair_value_range`` and never participates in
    ``triangulate.triangulate``'s confidence vote (see
    ``lbo.lbo_implied_floor_per_share``'s module docstring for the
    deleveraging-return mechanics).

    Entry multiple is the filer's OWN current EV/EBITDA
    (``metrics["ev_ebitda"]``) -- "could a disciplined financial buyer
    justify paying today's price". Exit multiple is held equal to the entry
    multiple (no multiple-expansion credit). **FCF is held FLAT
    (``fcf_growth=0.0``, F3):** a genuine conservative floor credits NO
    organic growth ANYWHERE -- EBITDA flat, exit multiple flat, and the
    debt-paydown FCF stream flat too. (An earlier version grew the FCF
    sweep at the base scenario's ``growth_5y`` while pinning EBITDA flat,
    which is internally incoherent -- for a high-growth filer the projected
    FCF could exceed EBITDA, over-sweeping the debt and inflating the
    "floor" past any conservative reading.) The LBO floor therefore no
    longer depends on the assumption set at all.

    Args:
        metrics: Reads ``ebitda``, ``ev_ebitda`` (entry/exit multiple),
            ``total_debt``, and ``shares``.
        fcf0: The engine's already-selected base-year FCF (SPEC.md Sec.4's
            ``fcf0`` -- the SAME cash-flow base the standard DCF uses, not a
            separately re-derived figure), swept flat over the hold period.

    Returns:
        A ``(result, notes)`` tuple. ``result`` is ``None`` when any
        required input is missing/degenerate (delegates entirely to
        :func:`lbo.lbo_implied_floor_per_share`'s own guards), else that
        function's result dict. Never raises.
    """
    notes: List[str] = []
    result = lbo.lbo_implied_floor_per_share(
        ebitda=metrics.get("ebitda"),
        entry_multiple=metrics.get("ev_ebitda"),
        existing_debt=metrics.get("total_debt"),
        exit_multiple=metrics.get("ev_ebitda"),
        fcf0=fcf0,
        fcf_growth=0.0,
        shares=metrics.get("shares"),
    )
    if result is None:
        notes.append(
            "LBO anchor could not be computed: EBITDA, EV/EBITDA multiple, total debt, FCF, or share count "
            "is missing/invalid."
        )
        return None, notes

    notes.append(
        f"LBO anchor (informational only, NOT part of the headline): at a {lbo._LBO_TARGET_IRR * 100:.0f}% "
        f"target return, the highest price a disciplined financial buyer could pay today is "
        f"~${result['per_share']:.2f}."
    )
    return result, notes


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
    scenario is skipped (with a note, NOT fabricated)
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
            "FFO anchor could not be computed: net income and depreciation (D&A) data aren't both available "
            "for the same fiscal year, or the resulting FFO is zero/negative."
        )
        return None, notes

    latest_fy = resolve_fundamental_fy(metrics)
    if latest_fy is not None and selected_fy is not None and selected_fy != latest_fy:
        notes.append(
            f"Used fiscal year {selected_fy}'s net income/depreciation data for the FFO anchor "
            "(the latest fiscal year's fundamentals weren't aligned with the share count)."
        )

    scenarios: Dict[str, dict] = {}
    implied_pffo: Dict[str, float] = {}
    for key in _SCENARIO_KEYS:
        scenario_assumptions = assumptions.get(key) or {}
        r = scenario_assumptions.get("discount_rate")
        g = scenario_assumptions.get("terminal_growth")

        if not _is_number(r) or not _is_number(g) or r <= g:
            notes.append(
                f"The FFO Gordon-growth model could not be computed for the {key.capitalize()} scenario "
                "(discount rate/terminal growth is missing, or the discount rate doesn't exceed terminal growth)."
            )
            continue

        gordon_multiple = (1 + g) / (r - g)
        per_share = round(ffo_per_share * gordon_multiple, 2)
        lo, hi, used_fallback = _ffo_scenario_band(ffo_per_share, r, g, per_share)
        if used_fallback:
            notes.append(
                f"The FFO sensitivity band for the {key.capitalize()} scenario could not be computed; "
                "used +/-10% of the point estimate as a fallback."
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
        return None, ["Earnings-power anchor could not be computed: no valid share count."]

    dr_base = (assumptions.get("base") or {}).get("discount_rate")
    if not _is_number(dr_base) or dr_base <= 0:
        return None, ["Earnings-power anchor could not be computed: no valid discount rate (cost of equity)."]

    fy = resolve_fundamental_fy(metrics)
    ni_series = to_annual_series(normalized, "NetIncome")
    rev_series = to_annual_series(normalized, "Revenue")

    latest_ni = ni_series.get(fy)
    latest_rev = rev_series.get(fy)

    if latest_ni is None or latest_ni <= 0:
        return None, ["Earnings-power anchor could not be computed: the latest year's net income is negative or missing."]

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
                f"The latest year's net income ({latest_ni:,.0f}) deviated markedly from the historical margin "
                f"median for the earnings-power base; used a margin-median-based normalized figure "
                f"({ref_ni:,.0f}) instead, against the possibility of a one-off non-operating effect."
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
                f"The earnings-power sensitivity band for the {key.capitalize()} scenario could not be "
                "computed; used +/-10% of the point estimate as a fallback."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    # --- Over-capitalization advisory note (does not affect the computed value) ---
    equity_series = to_annual_series(normalized, "StockholdersEquity")
    eq = equity_series.get(fy)
    if eq is not None and eq > 0:
        roe = normalized_ni / eq
        if roe / dr_base > _PB_CLAMP_HI:
            notes.append(
                "The earnings-power anchor relies on a very high implied return/discount-rate ratio; don't "
                "read the EPV value as upward-biased if that return isn't sustainable."
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
        notes.append("Cyclical FCFE anchor could not be computed: equity data is missing/negative.")
        return None, notes

    roe = ni_norm / equity
    if roe <= 0:
        notes.append("Cyclical FCFE anchor could not be computed: normalized ROE is not positive.")
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
            notes.append(f"Cyclical FCFE assumptions for the {key.capitalize()} scenario are missing or invalid.")
            continue

        try:
            result = dcf.fcfe_sustainable_growth_per_share(
                ni_norm, roe, growth_5y, terminal_growth, discount_rate, shares, dilution_rate,
                terminal_roe=discount_rate,
            )
        except ValueError as exc:
            scenarios[key] = {"per_share": None, "lo": None, "hi": None}
            notes.append(f"Cyclical FCFE could not be computed for the {key.capitalize()} scenario: {exc}")
            continue

        per_share = round(result["per_share"], 2)
        lo, hi, used_fallback = _cyclical_fcfe_scenario_band(
            ni_norm, roe, growth_5y, terminal_growth, discount_rate, shares, dilution_rate, per_share
        )
        if used_fallback:
            notes.append(
                f"The cyclical FCFE sensitivity band for the {key.capitalize()} scenario could not be "
                "computed; used +/-10% of the point estimate as a fallback."
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
        string to surface (only set on the "suppressed but not
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
            "Free cash flow is low and operating cash flow doesn't sufficiently support net income "
            "(OCF < 0.8x net income); this is an earnings-quality/cash-conversion warning -- the headline "
            "valuation was left on FCF-DCF rather than switching to the earnings-power anchor."
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
        ``maintenance_capex_floor_note`` (only present when the
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
            f"The maintenance-CapEx floor was set at {sector_capex_sales * 100:.1f}% based on sector data "
            "(Damodaran Cap Ex/Sales), instead of the default 5%."
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
    with a note explaining why, so the caller can fall back to
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
        scenario failed). ``notes`` are strings the caller should
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
                "Hyper-growth mode was triggered but the data required for the revenue-first DCF "
                "(the latest year's revenue or share count) is missing; using the standard valuation."
            )
            return None, notes

        realized_cagr = metrics.get("revenue_cagr_5y")
        if realized_cagr is None:
            realized_cagr = metrics.get("revenue_cagr_3y")
        if realized_cagr is None:
            notes.append(
                "Hyper-growth mode was triggered but realized revenue growth (CAGR) is missing; using "
                "the standard valuation."
            )
            return None, notes

        if terminal_growth != _HYPER_TERMINAL_GROWTH:
            notes.append(
                f"Terminal growth was tied to the risk-free rate ({terminal_growth * 100:.1f}%, capped at "
                "4%); a separate, lower terminal rate is not used for the hyper-growth cohort."
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
                "The hyper-growth start growth was computed as a blend of the realized 5y/3y CAGR and "
                "the latest year's growth."
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
                    "Gross-margin data is missing for the hyper-growth target mature FCF margin; used the "
                    f"{_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f}% default ceiling, floored at "
                    f"today's FCF margin ({current_margin * 100:.0f}%)."
                )
            else:
                notes.append(
                    "Gross-margin data is missing for the hyper-growth target mature FCF margin; used the "
                    f"{_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f}% default ceiling."
                )

        # WP4: target_base is no longer clamped to _HYPER_TARGET_BASE_CAP --
        # when the (pre-per-scenario-scaling) base value exceeds that
        # reference threshold, flag it instead of silently truncating a
        # genuinely high-margin business's economics.
        if target_base > _HYPER_TARGET_BASE_CAP:
            notes.append(
                f"The hyper-growth target mature FCF margin is {target_base * 100:.0f}%, above the 30% "
                "reference threshold (source: gross margin x 0.5); the high-margin assumption is "
                "deliberate -- not clamped to a fixed ceiling."
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

        # --- Deceleration guard (rule-based start-growth cap) ---------------
        # A scenario's start growth must not exceed the latest realized YoY
        # growth. Assuming a visibly decelerating company first RE-accelerates
        # (e.g. bull's raw ``growth_anchor * 1.2``) before fading to terminal
        # is internally inconsistent -- the fade already models growth rolling
        # over, so re-acceleration on top of it double-counts optimism. A
        # genuine re-acceleration thesis is not forbidden, only made explicit:
        # it enters ONLY as a per-scenario ``start_growth`` override in
        # ``hyper_growth_extras`` (AI mode), and is surfaced with a deviation
        # note (how far above the statistical base it sits) rather than applied
        # silently. When the latest YoY isn't usable, no cap is applied and
        # behavior is unchanged. This also subsumes the "bull compounds both
        # growth AND margin by 1.2" concern in rule-based mode: once bull's
        # start growth is capped at the same realized YoY as base, its extra
        # optimism can only come from the margin lever, not a second growth
        # uplift.
        decel_cap = latest_yoy if (latest_yoy is not None and latest_yoy > 0) else None

        start_growth_by_scenario = {}
        target_by_scenario = {}
        steady_state_by_scenario = {}
        probabilities = {}
        target_margin_overridden = {}
        start_growth_capped_keys: List[str] = []

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

            override_start = scenario_extras.get("start_growth")
            if _is_number(override_start):
                # AI-mode explicit assumption: honored, but a re-acceleration
                # above the statistical base is flagged as a deviation.
                start_growth_by_scenario[key] = override_start
                base_ref = decel_cap if decel_cap is not None else raw_start_growth[key]
                if override_start > base_ref:
                    notes.append(
                        f"The {key.capitalize()} scenario's start growth was set to {override_start * 100:.0f}% "
                        f"by an explicit assumption (hyper_growth_extras); above the statistical base "
                        f"({base_ref * 100:.0f}%) -- the re-acceleration thesis is being priced in deliberately."
                    )
            elif decel_cap is not None and raw_start_growth[key] > decel_cap:
                start_growth_by_scenario[key] = decel_cap
                start_growth_capped_keys.append(key)
            else:
                start_growth_by_scenario[key] = raw_start_growth[key]

            target_by_scenario[key] = target
            steady_state_by_scenario[key] = steady_state_year
            probabilities[key] = prob

        if start_growth_capped_keys:
            names = ", ".join(k.capitalize() for k in start_growth_capped_keys)
            notes.append(
                f"Deceleration guard: the {names} scenario(s)' start growth was capped at the latest "
                f"realized annual growth ({decel_cap * 100:.0f}%) -- a decelerating company was not assumed "
                "to re-accelerate first (re-acceleration can only enter via an explicit assumption in AI mode)."
            )

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
                "The hyper-growth discount rate was not held fixed: as the cash flows mature, each "
                "scenario's own cohort discount rate (bear "
                f"{_HYPER_DISCOUNT_RATE_BY_SCENARIO['bear'] * 100:.0f}%, base "
                f"{_HYPER_DISCOUNT_RATE_BY_SCENARIO['base'] * 100:.0f}%, bull "
                f"{_HYPER_DISCOUNT_RATE_BY_SCENARIO['bull'] * 100:.0f}%) was linearly faded toward the "
                f"mature cost of equity ({mature_discount_rate * 100:.1f}%) by its own steady-state year "
                f"(year {steady_state_by_scenario['base']} in the base scenario) (Damodaran fade)."
            )

        if target_margin_overridden["base"]:
            target_margin_source = "target margin supplied by the LLM/user (hyper_growth_extras)"
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
                        f"gross margin {gm * 100:.0f}% x 0.5, floored at today's FCF margin"
                    )
                else:
                    target_margin_source = "gross margin x 0.5"
            else:
                if floored_by_current_margin:
                    target_margin_source = (
                        f"no gross margin: {_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f}% default ceiling, "
                        f"floored at today's FCF margin ({current_margin * 100:.0f}%)"
                    )
                else:
                    target_margin_source = (
                        f"no gross margin: {_HYPER_TARGET_MARGIN_CEILING_FALLBACK * 100:.0f}% default ceiling"
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
            notes.append(f"Hyper-growth revenue-first DCF (base scenario) could not be computed: {exc}")
            return None, notes

        burn = sum(min(fcf_t, 0.0) for fcf_t in prelim_base["fcf_path"])
        if price is not None and price > 0:
            financing_shares = abs(burn) / price
        else:
            financing_shares = 0.0
            if burn < 0:
                notes.append(
                    "Hyper-growth financing (dilution) shares could not be computed because the price is "
                    "missing; assumed zero financing shares."
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
                notes.append(f"The {key.capitalize()} hyper-growth scenario could not be computed: {exc}")
                continue

            per_share = round(result["per_share"], 2)
            lo, hi, used_fallback = _hyper_scenario_band(
                latest_revenue, start_growth, terminal_growth, discount_rate, current_margin,
                target, steady_state_year, shares, annual_dilution, financing_shares, per_share,
                mature_discount_rate=mature_discount_rate,
            )
            if used_fallback:
                notes.append(
                    f"The sensitivity band for the {key.capitalize()} hyper-growth scenario could not be "
                    "computed; used +/-10% of the point estimate as a fallback."
                )
            scenarios_detail[key] = {
                "per_share": per_share, "lo": lo, "hi": hi,
                "start_growth": round(start_growth, 4), "target_fcf_margin": round(target, 4),
                "terminal_growth": round(terminal_growth, 4),
                "final_year_revenue": result["final_year_revenue"], "revenue_multiple": result["revenue_multiple"],
                # Persisted so a later analysis can measure "model-based
                # surprise" -- realized revenue vs. the revenue this scenario
                # projected for the elapsed time. base_revenue is the year-0
                # anchor (revenue_path[i] is the projected revenue for year i+1).
                "base_revenue": round(float(latest_revenue), 2),
                "steady_state_year": steady_state_year,
                "revenue_path": [round(float(v), 2) for v in result["revenue_path"]],
            }

        base_cell = scenarios_detail.get("base")
        if base_cell is None or base_cell.get("revenue_multiple") is None:
            notes.append(
                "The arrival-point flag could not be determined because the hyper-growth base scenario "
                "could not be computed; using the standard valuation."
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
                "The company invests in growth (CapEx) far in excess of its revenue, so today's free cash "
                "flow margin is severely negative; the revenue-first DCF base scenario produced a negative "
                "equity value (per share <= $0). Since that isn't a usable value for a going, capital-raising "
                "concern, the DCF headline range and its triangulation vote were disabled; the revenue-first "
                "DCF isn't reliable until CapEx intensity normalizes."
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
                "CapEx-heavy hyper-growth: today's free cash flow is suppressed by heavy growth CapEx, so "
                "the headline DCF (at the raw margin) isn't reliable and was disabled. Growth CapEx was also "
                "separated from maintenance CapEx (approx. D&A, floored at 5% of revenue) to compute an "
                f"AGGRESSIVE UPSIDE SCENARIO (base ${capex_normalization.get('upside_per_share')}/share) -- "
                "this is NOT THE HEADLINE, it only shows the optimistic value implied if CapEx normalizes. "
                "Note: this upside scenario doesn't fully reflect the CapEx that funds the revenue growth "
                "itself, so it is upward-biased."
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
            arrival_flag = "fair"
        elif multiple <= _HYPER_ARRIVAL_EXTREME_MULTIPLE:
            arrival_flag = "aggressive"
        else:
            arrival_flag = "excessively_aggressive"
        if arrival_flag != "fair":
            notes.append(
                f"Hyper-growth arrival point: in the base scenario, revenue grows {multiple:.1f}x over 10 years "
                f"({arrival_flag})."
            )

        tam_share = None
        if tam_usd is not None:
            tam_share = base_cell["final_year_revenue"] / tam_usd
            if tam_share > _HYPER_TAM_SHARE_INVALID:
                arrival_flag = "invalid"
                notes.append("Hyper-growth arrival point exceeds 60% of TAM; a revision is needed.")
            elif tam_share > _HYPER_TAM_SHARE_AGGRESSIVE:
                arrival_flag = "aggressive"
                notes.append(f"Hyper-growth arrival point uses {tam_share * 100:.0f}% of TAM (aggressive).")
            else:
                arrival_flag = "fair"

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
                "Hyper-growth: the price-implied start growth rate could not be computed "
                "(the price may imply an expectation outside a plausible growth range)."
            )

        implied_margin = revenue_dcf.implied_target_margin(
            price, latest_revenue, base_start_growth, terminal_growth, base_discount_rate,
            current_margin, base_steady_state_year, shares, annual_dilution, financing_shares,
            mature_discount_rate=mature_discount_rate,
        )

        # Third reverse lens: the flat cost of equity the price implies when
        # growth and margin are held at the model's base assumptions. A large
        # gap vs the model's own cohort/mature discount rate is one face of a
        # model-market divergence (see triangulate.py's governor).
        implied_discount_rate = revenue_dcf.implied_discount_rate(
            price, latest_revenue, base_start_growth, terminal_growth, current_margin,
            base_target, base_steady_state_year, shares, annual_dilution, financing_shares,
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
                "discount_rate": implied_discount_rate,
                "tam_share": implied_tam_share,
            },
            "base_discount_rate": round(base_discount_rate, 4),
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
        notes.append("Hyper-growth mode encountered an unexpected error; using the standard valuation.")
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
_MATURE_FALLBACK_SUFFIX = "using the earnings-power (EPV) anchor or the raw FCF-DCF instead."


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
    ``(None, notes)`` with a note, mirroring ``_build_hyper_growth``.

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
                "Data required for the mature revenue-first DCF (the latest year's revenue or share count) "
                f"is missing; {_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        start_growth = _mature_start_growth(metrics, normalized)
        if start_growth is None:
            notes.append(
                f"Realized revenue growth (CAGR) could not be computed for the mature revenue-first DCF; "
                f"{_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        base_terminal_growth = (assumptions.get("base") or {}).get("terminal_growth")
        if not _is_number(base_terminal_growth):
            notes.append(
                f"The base terminal growth rate is missing for the mature revenue-first DCF; {_MATURE_FALLBACK_SUFFIX}"
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
                f"Realized revenue growth ({start_growth * 100:.1f}%) is insufficient for the mature "
                f"revenue-first DCF (< {_MATURE_REV_DCF_MIN_GROWTH * 100:.0f}% or below terminal growth); "
                f"{_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        target_base = _mature_target_fcf_margin(normalized, metrics, ratios)
        if target_base is None:
            notes.append(
                "The target mature FCF margin could not be computed for the mature revenue-first DCF "
                f"(operating margin and historical FCF margin data are missing); {_MATURE_FALLBACK_SUFFIX}"
            )
            return None, notes

        # WP4: target_base is no longer clamped to _MATURE_TARGET_CAP -- flag
        # it instead of silently truncating when the NOPAT/historical-FCF
        # anchors genuinely derive a higher mature margin.
        if target_base > _MATURE_TARGET_CAP:
            notes.append(
                f"The mature target FCF margin is {target_base * 100:.0f}%, above the 15% reference "
                "threshold (derived from the NOPAT and historical-FCF anchors); deliberate -- not "
                "clamped to a fixed ceiling."
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
                notes.append(f"Mature revenue-first DCF assumptions for the {key.capitalize()} scenario are missing.")
                continue
            if discount_rate <= terminal_growth:
                notes.append(
                    f"The discount rate for the {key.capitalize()} scenario doesn't exceed terminal growth; "
                    "scenario skipped."
                )
                continue

            target_margin = target_base * _MATURE_TARGET_MARGIN_SCALE[key]

            try:
                result = revenue_dcf.revenue_first_dcf(
                    revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                    target_margin, steady_state_year, shares, 0.0,
                )
            except ValueError as exc:
                notes.append(f"Mature revenue-first DCF could not be computed for the {key.capitalize()} scenario: {exc}")
                continue

            per_share = round(result["per_share"], 2)
            lo, hi, used_fallback = _hyper_scenario_band(
                revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                target_margin, steady_state_year, shares, 0.0, 0.0, per_share,
            )
            if used_fallback:
                notes.append(
                    f"The sensitivity band for the {key.capitalize()} scenario could not be computed; "
                    "used +/-10% of the point estimate as a fallback."
                )

            scenarios[key] = {
                "per_share": per_share, "lo": lo, "hi": hi,
                "start_growth": round(start_growth, 4),
                "target_fcf_margin": round(target_margin, 4),
                "terminal_growth": round(terminal_growth, 4),
                "discount_rate": round(discount_rate, 4),
            }

        if not scenarios:
            notes.append(f"The mature revenue-first DCF could not be computed for any scenario; {_MATURE_FALLBACK_SUFFIX}")
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
        notes.append("The mature revenue-first DCF encountered an unexpected error; using the standard valuation.")
        return None, notes


#: Fallback note suffix for ``_build_midgrowth_revenue_dcf`` -- always ends
#: the same way so the reader knows what happens when the method bails out
#: (the filer falls back to the multiples-only headline it had before this
#: method existed).
_MIDGROWTH_FALLBACK_SUFFIX = "falling back to a multiples-based valuation."


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
    degrades to ``(None, notes)`` with a note.

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
                "Data required for the mid-growth revenue-first DCF (the latest year's revenue or share "
                f"count) is missing; {_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        start_growth = _mature_start_growth(metrics, normalized)
        if start_growth is None:
            notes.append(
                "Realized revenue growth (CAGR) could not be computed for the mid-growth revenue-first DCF; "
                f"{_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        base_terminal_growth = (assumptions.get("base") or {}).get("terminal_growth")
        if not _is_number(base_terminal_growth):
            notes.append(
                f"The base terminal growth rate is missing for the mid-growth revenue-first DCF; {_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        # --- Growth gate: this method models a fading GROWTH path, so it
        # only makes sense with a real, still-fading growth rate above the
        # 12% floor (and above terminal growth). Below that, fall back to
        # multiples rather than fabricating a growth story.
        if start_growth < _MIDGROWTH_MIN_GROWTH or start_growth <= base_terminal_growth:
            notes.append(
                f"Realized revenue growth ({start_growth * 100:.1f}%) is insufficient for the mid-growth "
                f"revenue-first DCF (< {_MIDGROWTH_MIN_GROWTH * 100:.0f}% or below terminal growth); "
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
                "Gross-margin data is missing for the mid-growth revenue-first DCF's target mature FCF "
                f"margin; used a {_MIDGROWTH_TARGET_CAP * 100:.0f}% ceiling."
            )

        # WP4: target_base is no longer clamped to _MIDGROWTH_TARGET_CAP --
        # flag it instead of silently truncating when the gross-margin-
        # derived value genuinely exceeds the reference threshold.
        if target_base > _MIDGROWTH_TARGET_CAP:
            notes.append(
                f"The mid-growth target mature FCF margin is {target_base * 100:.0f}%, above the 20% "
                "reference threshold (derived from gross margin x 0.5); deliberate -- not clamped to a "
                "fixed ceiling."
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
                        "Mid-growth financing (dilution) shares could not be computed because the price is "
                        "missing; assumed zero financing shares."
                    )
            except ValueError:
                financing_shares = 0.0

        scenarios: dict = {}
        for key in _SCENARIO_KEYS:
            scenario_assumptions = assumptions.get(key) or {}
            discount_rate = scenario_assumptions.get("discount_rate")
            terminal_growth = scenario_assumptions.get("terminal_growth")

            if not _is_number(discount_rate) or not _is_number(terminal_growth):
                notes.append(f"Mid-growth revenue-first DCF assumptions for the {key.capitalize()} scenario are missing.")
                continue
            if discount_rate <= terminal_growth:
                notes.append(
                    f"The discount rate for the {key.capitalize()} scenario doesn't exceed terminal growth; "
                    "scenario skipped."
                )
                continue

            target_margin = target_base * _MATURE_TARGET_MARGIN_SCALE[key]

            try:
                result = revenue_dcf.revenue_first_dcf(
                    revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                    target_margin, steady_state_year, shares, annual_dilution, financing_shares,
                )
            except ValueError as exc:
                notes.append(f"Mid-growth revenue-first DCF could not be computed for the {key.capitalize()} scenario: {exc}")
                continue

            per_share = round(result["per_share"], 2)
            lo, hi, used_fallback = _hyper_scenario_band(
                revenue0, start_growth, terminal_growth, discount_rate, current_margin,
                target_margin, steady_state_year, shares, annual_dilution, financing_shares, per_share,
            )
            if used_fallback:
                notes.append(
                    f"The sensitivity band for the {key.capitalize()} scenario could not be computed; "
                    "used +/-10% of the point estimate as a fallback."
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
                f"The mid-growth revenue-first DCF could not be computed for any scenario; {_MIDGROWTH_FALLBACK_SUFFIX}"
            )
            return None, notes

        # --- Suppression guardrail (mirrors the hyper path): a non-credible
        # negative base value means the caller should drop back to multiples.
        base_ps = (scenarios.get("base") or {}).get("per_share")
        suppressed = base_ps is not None and base_ps <= 0
        if suppressed:
            notes.append(
                "The mid-growth revenue-first DCF base scenario produced a negative equity value (per share "
                f"<= $0); not usable for the headline, {_MIDGROWTH_FALLBACK_SUFFIX}"
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
            "The mid-growth revenue-first DCF encountered an unexpected error; using the standard valuation."
        )
        return None, notes


#: labels for the current-multiple fallback notes, keyed by which
#: multiple was derived.
_MULTIPLE_LABELS = {
    "pe": "P/E", "ps": "P/S", "pfcf": "P/FCF",
    "ev_ebit": "EV/EBIT", "ev_ebitda": "EV/EBITDA", "ptbv": "P/B",
}


def _derive_current_multiples(
    normalized: dict, ratios: list, metrics: dict, price: Optional[float],
    suppress_ev: bool = False,
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
    current = {
        "pe": metrics.get("pe"), "ps": metrics.get("ps"), "pfcf": metrics.get("pfcf"),
        # SPEC.md Sec.20b: for a sector with no meaningful enterprise value,
        # the EV slots are never seeded and never back-filled below, so no
        # "derived from FYxxxx" note claims an EV multiple that shouldn't exist.
        "ev_ebit": None if suppress_ev else metrics.get("ev_ebit"),
        "ev_ebitda": None if suppress_ev else metrics.get("ev_ebitda"),
        # SPEC.md Sec.23c: reported for every sector, rendered only where it
        # carries information (financial).
        "ptbv": metrics.get("ptbv"),
    }
    notes: List[str] = []
    if price is None:
        return current, notes

    def _note(key: str, fy: int) -> None:
        notes.append(
            f"The current {_MULTIPLE_LABELS[key]} ratio couldn't be aligned with the latest fiscal year's "
            f"data, so it was computed using fiscal year {fy}'s data."
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
        # P/TBV fy-mismatch recovery (SPEC.md Sec.23c), same shape as the
        # pe/ps/pfcf fallbacks: scan back to the latest fiscal year with a
        # positive tangible equity.
        if current["ptbv"] is None:
            tangible_by_fy = {
                row.get("fy"): row.get("tangible_equity")
                for row in (ratios or []) if row.get("fy") is not None
            }
            for fy in sorted(tangible_by_fy, reverse=True):
                tangible = tangible_by_fy.get(fy)
                if tangible is not None and tangible > 0:
                    current["ptbv"] = round(price * shares / tangible, 4)
                    _note("ptbv", fy)
                    break

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

    # EV/EBIT and EV/EBITDA fy-mismatch fallback: metrics already anchors these
    # to latest_fundamental_fy, so a cover-page mismatch doesn't zero them the
    # way it can pe/ps/pfcf; this only recovers the case where that anchor year
    # has a non-positive/missing EBIT(DA), by scanning back to the latest fy
    # with a positive denominator. EV = current market cap + net debt
    # (metrics["ev"], already computed the same way).
    ev = None if suppress_ev else metrics.get("ev")
    if ev is not None:
        if current["ev_ebit"] is None:
            oi_series = to_annual_series(normalized, "OperatingIncome")
            for fy in sorted(oi_series, reverse=True):
                oi = oi_series.get(fy)
                if oi is not None and oi > 0:
                    current["ev_ebit"] = round(ev / oi, 4)
                    _note("ev_ebit", fy)
                    break

        if current["ev_ebitda"] is None:
            oi_series = to_annual_series(normalized, "OperatingIncome")
            dep_series = to_annual_series(normalized, "Depreciation")
            for fy in sorted(set(oi_series) & set(dep_series), reverse=True):
                oi, dep = oi_series.get(fy), dep_series.get(fy)
                if oi is not None and dep is not None and oi + dep > 0:
                    current["ev_ebitda"] = round(ev / (oi + dep), 4)
                    _note("ev_ebitda", fy)
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
    ``applicable=False`` with a reason, never a negative/exploded
    figure.

    Returns:
        A ``(block, raw_pair_pct, growth_adj_pct)`` tuple. ``block`` is the
        output dict; ``raw_pair_pct``/``growth_adj_pct`` are the two
        percentiles the triangulation divergence check compares (either may
        be ``None``). Never raises.
    """
    if hyper_growth_active:
        metric, label, raw_label, raw_key = "growth_adj_ps", "Growth-adjusted EV/Sales", "EV/S", "ev_sales"
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
                f"The growth-adjusted multiple ({label}) isn't applicable: base growth is below 5% "
                "(the denominator isn't reliable)."
            )
        elif raw_current is None or raw_current <= 0:
            detail = "TTM earnings aren't positive (no P/E)" if metric == "peg" else "EV/Sales could not be computed"
            block["reason"] = f"The growth-adjusted multiple ({label}) isn't applicable: {detail}."
        else:
            block["reason"] = f"The growth-adjusted multiple ({label}) isn't applicable."
        return block, raw_pct, None

    revenue_series = to_annual_series(normalized, "Revenue")
    ga_hist = multiples.growth_adjusted_history(history, revenue_series, raw_key)
    ga_pct = multiples.percentile_position(ga_hist, ga_value)

    block["value"] = ga_value
    block["percentile"] = ga_pct
    block["applicable"] = True
    return block, raw_pct, ga_pct


def _format_growth_pct(value: float) -> str:
    """growth string, e.g. ``0.08 -> "8% growth"`` (Sec.4)."""
    return f"{value * 100:.0f}% growth"


def _format_discount_rate_pct(value: float) -> str:
    """discount-rate string, e.g. ``0.12 -> "12%"`` (Sec.4)."""
    return f"{value * 100:.0f}%"


#: scenario labels used inside the hyper-grower ``fair_value_range``
#: note (see ``_hyper_scenario_meta``), keyed the same as ``_SCENARIO_KEYS``.
_HYPER_SCENARIO_LABEL = {"bear": "Pessimistic", "base": "Base", "bull": "Optimistic"}


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
        terminal_str = f"{terminal_growth * 100:.1f}%" if _is_number(terminal_growth) else "terminal"
        meta[key] = {
            "growth": f"{start_growth * 100:.0f}% start -> fading to {terminal_str} terminal",
            "discount_rate": f"{discount_rate * 100:.0f}%",
            "note": (
                f"Hyper-growth {scenario_label}: {start_growth * 100:.0f}% start growth "
                f"(fading to {terminal_str} terminal over 10 years), {target_fcf_margin * 100:.0f}% mature "
                f"FCF margin, {discount_rate * 100:.0f}% discount rate."
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
        terminal_str = f"{terminal_growth * 100:.1f}%" if _is_number(terminal_growth) else "terminal"
        meta[key] = {
            "growth": (
                f"realized growth {start_growth * 100:.1f}%, mature target margin {target_fcf_margin * 100:.1f}%"
            ),
            "discount_rate": _format_discount_rate_pct(discount_rate),
            "note": (
                f"Mature revenue-first DCF {scenario_label}: realized growth {start_growth * 100:.1f}% "
                f"(fading to {terminal_str} terminal over {_MATURE_STEADY_STATE_YEAR} years), mature FCF "
                f"margin {target_fcf_margin * 100:.1f}%, discount rate {discount_rate * 100:.0f}%."
            ),
        }
    return meta


def _midgrowth_scenario_meta(midgrowth_revenue_detail: Optional[dict]) -> dict:
    """Build the ``fair_value_range`` ``scenario_meta`` override for the
    mid-growth, loss-making revenue-first DCF headline (SPEC Sec.8d),
    mirroring :func:`_mature_scenario_meta` -- same per-scenario
    ``growth``/``discount_rate``/``note`` shape, but with this method's own
    8-year fade horizon and "mid-growth" wording. Any scenario whose cell
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
        terminal_str = f"{terminal_growth * 100:.1f}%" if _is_number(terminal_growth) else "terminal"
        meta[key] = {
            "growth": (
                f"realized growth {start_growth * 100:.1f}%, mature target margin {target_fcf_margin * 100:.1f}%"
            ),
            "discount_rate": _format_discount_rate_pct(discount_rate),
            "note": (
                f"Mid-growth revenue-first DCF {scenario_label}: realized growth {start_growth * 100:.1f}% "
                f"(fading to {terminal_str} terminal over {_MIDGROWTH_STEADY_STATE_YEAR} years), mature FCF "
                f"margin {target_fcf_margin * 100:.1f}%, discount rate {discount_rate * 100:.0f}%."
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
            "growth": "zero growth (earnings-power anchor)",
            "discount_rate": _format_discount_rate_pct(cost_of_equity),
            "note": (
                f"Earnings-power anchor ({key}): normalized net income / cost of equity "
                f"({cost_of_equity * 100:.0f}%), {scale:.1f}x scale, with a zero-growth assumption (the "
                "growth premium is deliberately excluded)."
            ),
        }
    return meta


def _rim_scenario_meta(rim_detail: Optional[dict], assumptions: dict) -> dict:
    """Build the ``fair_value_range`` ``scenario_meta`` override for the
    financial-sector RIM headline (SPEC.md Sec.22c), mirroring
    :func:`_cyclical_fcfe_scenario_meta`'s structure.

    Before this existed, ``financial`` had no entry in the ``scenario_meta``
    chain, so its fair-value rows fell through to the raw assumption's
    ``growth_5y`` -- printing "25% growth" for a filer whose RIM had actually
    compounded earnings at its 4.6% ROE, because ``g = b x ROE`` caps growth
    at what retained earnings can fund. The label now reports the EFFECTIVE
    rate and names the assumed one as discarded.

    Returns an empty dict (so :func:`_build_fair_value_range` falls back to
    the assumptions-derived values) when ``rim_detail`` is ``None`` or is
    missing its ``scenarios``/``roe``. Never raises.
    """
    if not rim_detail:
        return {}
    scenarios = rim_detail.get("scenarios") or {}
    roe = rim_detail.get("roe")
    if not scenarios or not _is_number(roe) or roe <= 0:
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

        effective_growth = min(growth_5y, roe)
        if effective_growth < growth_5y:
            growth_str = (
                f"{effective_growth * 100:.1f}% growth "
                f"(the assumed {growth_5y * 100:.1f}% was capped by the internal-funding constraint)"
            )
            note = (
                f"Residual income (RIM) anchor ({key}): book value + the present value of 10 years of "
                f"residual income. Applied the {effective_growth * 100:.1f}% growth ROE can fund instead "
                f"of the assumed {growth_5y * 100:.1f}%; the only factor distinguishing this scenario from "
                f"the others is the cost of equity ({discount_rate * 100:.1f}%)."
            )
        else:
            growth_str = f"{growth_5y * 100:.1f}% growth (earnings + sustainable growth)"
            note = (
                f"Residual income (RIM) anchor ({key}): book value + the present value of 10 years of "
                f"residual income; growth {growth_5y * 100:.1f}%, cost of equity "
                f"{discount_rate * 100:.1f}%, ROE {roe * 100:.1f}%."
            )

        meta[key] = {
            "growth": growth_str,
            "discount_rate": _format_discount_rate_pct(discount_rate),
            "note": note,
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

        # SPEC.md Sec.22c: report the growth ACTUALLY applied. The old label
        # printed the uncapped assumption, so a filer whose ROE binds the cap
        # was told its valuation assumed a growth rate the model discarded.
        effective_growth = min(growth_5y, roe)
        if effective_growth < growth_5y:
            growth_str = (
                f"{effective_growth * 100:.1f}% growth "
                f"(the assumed {growth_5y * 100:.1f}% was capped by the internal-funding constraint)"
            )
        else:
            growth_str = f"{growth_5y * 100:.1f}% growth (earnings + sustainable growth)"

        reinvestment_rate = effective_growth / roe
        meta[key] = {
            "growth": growth_str,
            "discount_rate": _format_discount_rate_pct(discount_rate),
            "note": (
                f"Sustainable-growth FCFE anchor ({key}): normalized net income is grown; to fund that "
                f"growth, ~{reinvestment_rate * 100:.0f}% of earnings (g/ROE, ROE {roe * 100:.0f}%) is "
                "reinvested, and the remainder is discounted."
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
            pre-formatted strings that should replace the
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


def _entry(role: str, key: str, label_tr: str, reason_tr: str) -> dict:
    """One ``method_summary`` row (see :func:`_build_method_summary`)."""
    return {"role": role, "key": key, "label_tr": label_tr, "reason_tr": reason_tr}


def _build_method_summary(
    sector_type: str,
    hyper_growth_active: bool,
    hyper_growth_detail: Optional[dict],
    cyclical_fcfe_headline: bool,
    cyclical_fcfe_detail: Optional[dict],
    epv_headline: bool,
    earnings_power: Optional[dict],
    normalized_variant: Optional[dict],
    dcf_scenarios: Optional[dict],
    mature_revenue_headline: bool,
    mature_revenue_detail: Optional[dict],
    midgrowth_revenue_headline: bool,
    midgrowth_revenue_detail: Optional[dict],
    rim: Optional[dict],
    ffo: Optional[dict],
    output_implied: Optional[float],
    output_bracket_status: str,
    multiples_out: dict,
    altman_z: Optional[dict],
    beneish_m: Optional[dict],
    merton_dtd: Optional[dict],
    lbo_floor_detail: Optional[dict],
    cycle: Optional[dict],
) -> List[dict]:
    """Build the ``method_summary`` output (SPEC.md Sec.8l): a purely
    additive, packaging-only explanation of WHICH valuation method(s)
    ``run_valuation`` used for this filer and WHY -- for the report layer's
    educational "Valuation Methods Used" section.

    This function introduces NO new precedence logic and NO new numeric
    computation: every classification below re-reads the exact same
    flags/detail dicts the headline-selection code (immediately above, in
    ``_run_valuation``) already branched on, in the same order. It is pure
    packaging of an already-made decision, never a second decision-maker.

    Args:
        sector_type: The resolved sector bucket (``classify_sector``).
        hyper_growth_active: Whether the hyper-grower revenue-first DCF is
            active (``is_hyper_grower and hyper_growth_detail is not None``).
        hyper_growth_detail: The hyper-grower detail dict, or ``None``.
        cyclical_fcfe_headline: Whether the cyclical sustainable-growth FCFE
            anchor won the headline (Sec.8e).
        cyclical_fcfe_detail: The cyclical FCFE detail dict, or ``None``.
        epv_headline: Whether the zero-growth EPV anchor won the headline
            (Sec.8a).
        earnings_power: The EPV detail dict (mature/cyclical only), or
            ``None``.
        normalized_variant: The cyclical cycle-mid normalized FCF-DCF
            scenarios, or ``None``.
        dcf_scenarios: The raw (as-reported-FCF) FCF-DCF scenarios, or
            ``None``.
        mature_revenue_headline: Whether the mature revenue-first DCF won
            the headline (Sec.8b).
        mature_revenue_detail: The mature revenue-first DCF detail dict, or
            ``None``.
        midgrowth_revenue_headline: Whether the mid-growth revenue-first DCF
            won the headline (Sec.8d).
        midgrowth_revenue_detail: The mid-growth revenue-first DCF detail
            dict, or ``None``.
        rim: The financial-sector residual income model detail dict, or
            ``None``.
        ffo: The reit-sector FFO Gordon-growth detail dict, or ``None``.
        output_implied: The (possibly hyper/mature/midgrowth-overridden)
            reverse-DCF implied growth rate, or ``None``.
        output_bracket_status: The (possibly overridden) reverse-DCF bracket
            status string.
        multiples_out: The already-assembled ``multiples`` output dict (used
            only to read ``multiples_out["leveraged"]``; not recomputed).
        altman_z: The Altman Z-score detail dict, or ``None``.
        beneish_m: The Beneish M-score detail dict, or ``None``.
        merton_dtd: The Merton distance-to-default detail dict, or ``None``.
        lbo_floor_detail: The LBO-implied floor detail dict, or ``None``.
        cycle: The through-cycle/two-regime detail dict, or ``None``.

    Returns:
        A list of ``{"role", "key", "label_tr", "reason_tr"}`` dicts: exactly
        one ``"headline"`` entry first, then any ``"secondary"`` entries for
        methods computed but demoted, then the ``"cross_check"`` entries
        (reverse-DCF, multiples) when data allows, then ``"advisory"``
        entries for whichever risk/quality screens are non-``None``. Never
        raises -- degrades to ``[]`` on any unexpected error (the caller
        wraps this in its own try/except too, per module discipline).
    """
    summary: List[dict] = []

    # --- Headline (exactly one) + its directly-demoted secondaries ---------
    if hyper_growth_active:
        if hyper_growth_detail and hyper_growth_detail.get("suppressed"):
            summary.append(_entry(
                "headline", "hyper_growth_revenue_dcf", "Hyper-Growth Revenue-First DCF",
                "The base scenario produced a negative equity value because of excessive growth "
                "investment (CapEx); the headline range was left empty since the revenue-first DCF "
                "isn't considered reliable.",
            ))
        else:
            summary.append(_entry(
                "headline", "hyper_growth_revenue_dcf", "Hyper-Growth Revenue-First DCF",
                "Since hyper-growth was detected, the headline is based on a revenue-first DCF that "
                "fades revenue growth and converges to a mature target FCF margin.",
            ))
            if dcf_scenarios is not None:
                summary.append(_entry(
                    "secondary", "raw_fcf_dcf", "Standard FCF-DCF",
                    "The standard two-stage free-cash-flow DCF was reported as a secondary "
                    "cross-check.",
                ))
    elif sector_type == "cyclical":
        if cyclical_fcfe_headline:
            summary.append(_entry(
                "headline", "cyclical_fcfe", "Sustainable-Growth FCFE Anchor",
                "Since free cash flow is suppressed by growth investment (heavy CapEx), the headline "
                "is based on an FCFE anchor that applies sustainable growth (g/ROE) to mid-cycle "
                "normalized earnings.",
            ))
            if earnings_power is not None:
                summary.append(_entry(
                    "secondary", "epv", "Earnings Power Valuation (EPV)",
                    "The zero-growth earnings-power base was reported as a secondary reference "
                    "against which the FCFE anchor is measured.",
                ))
            if normalized_variant is not None:
                summary.append(_entry(
                    "secondary", "cycle_mid_normalized_fcf_dcf", "Mid-Cycle Normalized FCF-DCF",
                    "The DCF based on mid-cycle normalized FCF was reported as a secondary "
                    "cross-check.",
                ))
            if dcf_scenarios is not None:
                summary.append(_entry(
                    "secondary", "raw_fcf_dcf", "Standard FCF-DCF",
                    "The standard FCF-DCF based on the raw (suppressed) year was reported as "
                    "secondary.",
                ))
        elif epv_headline:
            summary.append(_entry(
                "headline", "epv", "Earnings Power Valuation (EPV)",
                "Since the cyclical + capital-intensive structure keeps free cash flow from "
                "reflecting earning power, the headline is based on the zero-growth earnings-power "
                "(EPV) anchor.",
            ))
            if normalized_variant is not None:
                summary.append(_entry(
                    "secondary", "cycle_mid_normalized_fcf_dcf", "Mid-Cycle Normalized FCF-DCF",
                    "The mid-cycle normalized FCF-DCF was reported as secondary.",
                ))
            if dcf_scenarios is not None:
                summary.append(_entry(
                    "secondary", "raw_fcf_dcf", "Standard FCF-DCF",
                    "The raw (suppressed) standard FCF-DCF was reported as secondary.",
                ))
            if cyclical_fcfe_detail is not None:
                summary.append(_entry(
                    "secondary", "cyclical_fcfe", "Sustainable-Growth FCFE Anchor",
                    "The growth-inclusive FCFE was also computed but stayed below the zero-growth "
                    "EPV base, so the headline was kept at EPV; reported as secondary.",
                ))
        elif normalized_variant is not None:
            summary.append(_entry(
                "headline", "cycle_mid_normalized_fcf_dcf", "Mid-Cycle Normalized FCF-DCF",
                "In the cyclical sector, the headline range was anchored on mid-cycle normalized FCF "
                "instead of a single year's (often near-trough) free cash flow.",
            ))
            if dcf_scenarios is not None:
                summary.append(_entry(
                    "secondary", "raw_fcf_dcf", "Standard FCF-DCF",
                    "The raw trough-FCF DCF scenarios were reported as secondary.",
                ))
        else:
            summary.append(_entry(
                "headline", "raw_fcf_dcf", "Standard FCF-DCF",
                "Since a normalized alternative couldn't be computed for the cyclical sector, the "
                "headline is based on the standard free-cash-flow DCF.",
            ))
    elif sector_type == "mature":
        if mature_revenue_headline:
            summary.append(_entry(
                "headline", "mature_revenue_dcf", "Mature Revenue-First DCF",
                "Since free cash flow is suppressed by growth investment, but the realized growth is "
                "real, the headline is based on a growth-inclusive revenue-first DCF that fades revenue.",
            ))
            if earnings_power is not None:
                summary.append(_entry(
                    "secondary", "epv", "Earnings Power Valuation (EPV)",
                    "The zero-growth EPV base was reported as a secondary cross-check.",
                ))
            if dcf_scenarios is not None:
                summary.append(_entry(
                    "secondary", "raw_fcf_dcf", "Standard FCF-DCF",
                    "The raw FCF-DCF scenarios were reported as secondary.",
                ))
        elif epv_headline:
            summary.append(_entry(
                "headline", "epv", "Earnings Power Valuation (EPV)",
                "Since free cash flow doesn't reflect earning power because of heavy growth investment "
                "(high CapEx), the headline is based on the zero-growth earnings-power (EPV) anchor.",
            ))
            if dcf_scenarios is not None:
                summary.append(_entry(
                    "secondary", "raw_fcf_dcf", "Standard FCF-DCF",
                    "The raw FCF-DCF scenarios were reported as secondary.",
                ))
            if mature_revenue_detail is not None:
                summary.append(_entry(
                    "secondary", "mature_revenue_dcf", "Mature Revenue-First DCF",
                    "The growth-inclusive revenue-first DCF was also computed but stayed below the "
                    "zero-growth EPV base, so the headline was kept at EPV; reported as a secondary "
                    "cross-check.",
                ))
        else:
            summary.append(_entry(
                "headline", "raw_fcf_dcf", "Standard FCF-DCF",
                "Since free cash flow reliably reflects earning power, the headline is based on the "
                "standard FCF-DCF.",
            ))
    elif sector_type == "growth_unprofitable":
        if midgrowth_revenue_headline:
            summary.append(_entry(
                "headline", "midgrowth_revenue_dcf", "Mid-Growth Revenue-First DCF",
                "Since realized growth is below the hyper-growth threshold but still real (in the "
                "12-20% band), the headline is based on a revenue-first DCF that fades revenue.",
            ))
            if dcf_scenarios is not None:
                summary.append(_entry(
                    "secondary", "raw_fcf_dcf", "Standard FCF-DCF",
                    "The raw FCF-DCF was reported as secondary.",
                ))
        else:
            summary.append(_entry(
                "headline", "raw_fcf_dcf", "Standard FCF-DCF",
                "Since the loss-making growth company doesn't clear the revenue-first DCF's growth "
                "gate, the headline falls back to the standard FCF-DCF whenever it can be computed.",
            ))
    elif sector_type == "financial":
        if rim is not None:
            summary.append(_entry(
                "headline", "rim", "Residual Income Model (RIM)",
                "Since the free-cash-flow DCF isn't reliable in the financial sector, the headline is "
                "based on the RIM anchor, which combines book value with the present value of residual "
                "income.",
            ))
        else:
            summary.append(_entry(
                "headline", "pb_roe", "P/B x ROE Anchor",
                "Since RIM couldn't be computed in the financial sector, the headline/triangulation "
                "anchor fell back to P/B x ROE.",
            ))
    elif sector_type == "reit":
        if ffo is not None:
            summary.append(_entry(
                "headline", "ffo", "FFO Gordon-Growth Anchor",
                "Since the free-cash-flow DCF isn't reliable for REITs, the headline is based on the "
                "FFO (funds from operations)-based Gordon-growth model.",
            ))
        else:
            summary.append(_entry(
                "headline", "pb_roe", "P/B x ROE Anchor",
                "Since FFO couldn't be computed for the REIT, the headline/triangulation anchor fell "
                "back to P/B x ROE.",
            ))
    else:
        summary.append(_entry(
            "headline", "raw_fcf_dcf", "Standard FCF-DCF",
            "Since no dedicated alternative method is defined for this sector type, the headline is "
            "based on the standard FCF-DCF.",
        ))

    # --- Cross-checks: reverse-DCF + multiples, always attempted ------------
    if output_implied is not None or output_bracket_status != "no_data":
        summary.append(_entry(
            "cross_check", "reverse_dcf", "Reverse DCF (Price-Implied Growth)",
            "The price-implied growth rate is compared against realized growth to test what the "
            "market expects from this company.",
        ))

    leveraged = bool((multiples_out or {}).get("leveraged"))
    if sector_type == "growth_unprofitable":
        summary.append(_entry(
            "cross_check", "multiples", "Multiple Comparison (P/S)",
            "Since P/E is meaningless for loss-making growth companies, P/S (Price/Sales) was used "
            "as the primary multiple.",
        ))
    elif sector_type == "reit":
        summary.append(_entry(
            "cross_check", "multiples", "Multiple Comparison (P/FFO)",
            "Since GAAP depreciation distorts P/E and book value, P/FFO was used as the primary "
            "multiple.",
        ))
    elif leveraged:
        summary.append(_entry(
            "cross_check", "multiples", "Multiple Comparison (EV/EBITDA)",
            "Since the net-debt/EBITDA ratio is high (leveraged), capital-structure-neutral "
            "EV/EBITDA was used as the primary multiple.",
        ))
    else:
        summary.append(_entry(
            "cross_check", "multiples", "Multiple Comparison (P/E)",
            "P/E was used as the standard primary multiple.",
        ))

    # --- Advisory screens: informational only, never headline --------------
    if altman_z is not None:
        summary.append(_entry(
            "advisory", "altman_z", "Altman Z-Score",
            "An informational indicator screening for financial distress/bankruptcy risk; does not "
            "affect the headline valuation.",
        ))
    if beneish_m is not None:
        summary.append(_entry(
            "advisory", "beneish_m", "Beneish M-Score",
            "An informational indicator screening for possible earnings manipulation; does not "
            "affect the headline valuation.",
        ))
    if merton_dtd is not None:
        summary.append(_entry(
            "advisory", "merton_dtd", "Merton Distance-to-Default Model",
            "An informational indicator estimating distance to default from the market price; does "
            "not affect the headline valuation.",
        ))
    if lbo_floor_detail is not None:
        summary.append(_entry(
            "advisory", "lbo_floor_detail", "LBO-Implied Floor Value",
            "Estimates the floor value a leveraged buyout could pay; does not affect the headline "
            "valuation.",
        ))
    if cycle is not None:
        summary.append(_entry(
            "advisory", "cycle", "Cycle Position / Two-Regime Valuation",
            "Shows the company's position in its cycle and a peak/trough two-regime valuation "
            "reading; does not affect the headline valuation.",
        ))

    return summary


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
    as_of=None,
    fred_rate: Optional[dict] = None,
    precedent_transactions_dir: Optional[str] = None,
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
        as_of: Optional point-in-time date (``datetime.date`` or ISO string).
            When set, the Damodaran macro load resolves ERP/risk-free from
            the historical archive (see
            :func:`sec_analyzer.valuation.damodaran.load_sector_data`), the
            resulting ``macro_asof`` provenance is copied into the result,
            and two notes (macro source + static-multiples caveat)
            are appended. ``None`` leaves behavior unchanged.
        fred_rate: Optional historical risk-free dict (from
            :func:`sec_analyzer.fetch.fred.get_risk_free_asof`), forwarded to
            the macro load. Only consulted when ``as_of`` is set.
        precedent_transactions_dir: Directory holding an operator-curated
            precedent-transaction (M&A comps) reference CSV (SPEC.md
            Sec.8i). Defaults to ``Config.PRECEDENT_TRANSACTIONS_DIR``.
            Purely additive/optional -- absent data degrades silently to
            today's behavior (no EV/EBITDA sector-median axis-b comparison
            for leveraged filers).

    Returns:
        The ``valuation`` dict documented in SPEC Sec.11. Every
        unavailable piece is ``None`` plus a note in ``notes``.
        Never raises.
    """
    try:
        return _run_valuation(
            normalized or {}, ratios or [], metrics or {}, price, price_df, assumptions or {},
            sector_type, damodaran_dir, sic_description, hyper_growth_extras,
            as_of=as_of, fred_rate=fred_rate, precedent_transactions_dir=precedent_transactions_dir,
        )
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("run_valuation() failed unexpectedly; returning a degraded result.")
        return _empty_valuation(sector_type, assumptions)


def _run_valuation(
    normalized: dict, ratios: list, metrics: dict, price: Optional[float], price_df, assumptions: dict,
    sector_type: str, damodaran_dir: Optional[str], sic_description: Optional[str],
    hyper_growth_extras: Optional[dict] = None,
    as_of=None,
    fred_rate: Optional[dict] = None,
    precedent_transactions_dir: Optional[str] = None,
) -> dict:
    notes: List[str] = []

    # SPEC.md Sec.19: surface a rejected as-reported revenue basis, so the
    # substituted top line is auditable rather than a silent correction.
    revenue_basis_note = ((normalized or {}).get("revenue_basis") or {}).get("note")
    if revenue_basis_note:
        notes.append(revenue_basis_note)

    # SPEC.md Sec.24c: every anchor below is built on ANNUAL series, so a filer
    # deep into an unreported fiscal year can be valued on a base that the
    # quarterly filings have already overtaken. Say so rather than letting the
    # stale figure pass as current.
    notes.extend(_ttm_staleness_notes(metrics))

    is_unprofitable = sector_type == "growth_unprofitable"
    for violation in sanity.validate_assumptions(assumptions, is_unprofitable=is_unprofitable):
        notes.append(f"Assumption warning: {violation}")

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
    sector_data = damodaran.load_sector_data(
        damodaran_dir if damodaran_dir is not None else Config.DAMODARAN_DIR,
        as_of=as_of,
        fred_rate=fred_rate,
    )
    risk_free_pct = sector_data.get("risk_free") if sector_data else None
    # Macro provenance is emitted on EVERY run, not just as-of ones: the
    # risk-free rate is now read live from FRED (SPEC.md "Risk-free source:
    # live FRED DGS10"), so it moves between runs and the note is what lets a
    # reader reproduce a given fair value. `macro_asof["as_of"]` is None on a
    # live run, which is how the two phrasings are told apart.
    if sector_data and sector_data.get("macro_asof"):
        macro_asof = sector_data["macro_asof"]
        if macro_asof.get("as_of"):
            notes.append(
                f"Historical (as-of) mode: as of {macro_asof['as_of']} -- "
                f"ERP source: {macro_asof['erp_source']}; risk-free source: "
                f"{macro_asof['risk_free_source']}; multiple/beta source: "
                f"{macro_asof.get('multiples_source', 'multiples.csv')}."
            )
        else:
            notes.append(
                f"Macro sources -- risk-free: {macro_asof['risk_free_source']}; "
                f"ERP: {macro_asof['erp_source']}; multiple/beta: "
                f"{macro_asof.get('multiples_source', 'multiples.csv')}."
            )
        # Surface any anachronism warnings (current snapshot substituted for a
        # missing historical one) directly in the valuation notes.
        for warning in macro_asof.get("warnings") or []:
            notes.append(warning)
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
        notes.append("Damodaran sector data not found; sector medians cannot be shown.")
        sector_medians_result = None
    elif not sic_description:
        notes.append("Damodaran sector medians could not be matched because no SIC description was provided.")
        sector_medians_result = None
    else:
        sector_medians_result = damodaran.sector_medians(sector_data, sic_description)
        if sector_medians_result is None:
            notes.append("The company's SIC description could not be matched to a Damodaran sector.")

    sector_capex_sales = (sector_medians_result or {}).get("capex_sales")

    # --- Precedent-transaction (M&A comps) reference data (SPEC.md Sec.8i) --
    # Optional, purely additive: looked up against the SAME industry name
    # sector_medians_result already resolved (no second SIC matcher). Fills a
    # real gap in the multiples-comparison block below -- Damodaran's own
    # multiples.csv carries no EV/EBITDA sector median at all, so a leveraged
    # filer's axis-b comparison was previously always disabled for its
    # primary (EV/EBITDA) multiple; precedent-transaction EV/EBITDA medians
    # can fill that in when the operator has curated the data.
    precedent_deals = precedent_transactions.load_precedent_transactions(
        precedent_transactions_dir if precedent_transactions_dir is not None else Config.PRECEDENT_TRANSACTIONS_DIR
    )
    precedent_medians = precedent_transactions.find_industry_medians(
        precedent_deals, (sector_medians_result or {}).get("industry")
    )

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
                "The free-cash-flow DCF isn't reliable for REITs; the FFO (funds from operations)-based "
                "Gordon-growth model is used instead (falling back to the P/B x ROE anchor if FFO can't "
                "be computed)."
            )
        else:
            disabled_reason = (
                "The free-cash-flow DCF isn't reliable for financial companies; the P/B x ROE anchor is "
                "used instead."
            )

    dcf_scenarios = None
    dcf_high_growth_flag = False
    if dcf_enabled:
        dcf_scenarios, dcf_notes, dcf_high_growth_flag = _build_dcf_scenarios(assumptions, fcf0, shares, dilution_rate)
        notes.extend(dcf_notes)
        if dcf_scenarios is None and fcf0 is not None:
            notes.append("DCF could not be computed: no valid share count.")

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

    # --- RIM (financial) / FFO (reit) anchor (SPEC Sec.8/Sec.8c/Sec.8f) -----
    # `financial` now gets the residual income model (RIM, Sec.8f) as its
    # PRIMARY anchor instead of the single-period P/B x ROE heuristic -- a
    # multi-year fade of growth/discount-rate assumptions rather than one
    # static justified-multiple. If RIM can't be built at all (too little
    # equity/net-income/ROE history, or no scenario is computable), fall
    # back to the original P/B x ROE anchor, so there's still a book-based
    # headline/triangulation anchor. `reit` is UNAFFECTED by RIM -- it keeps
    # its own FFO-based Gordon-growth anchor (GAAP real-estate depreciation
    # depresses both net income and book equity, so neither RIM nor P/B x
    # ROE is a reliable book-value-based signal for a REIT); if FFO can't be
    # built at all (no Depreciation data for any fiscal year that also has
    # NetIncome, or the resulting FFO is <= 0), it falls back to P/B x ROE,
    # exactly as before.
    pb_roe = None
    rim = None
    ffo = None
    if sector_type == "financial":
        rim, rim_notes = _build_rim(assumptions, normalized, metrics, ratios)
        notes.extend(rim_notes)
        if rim is None:
            pb_roe, pb_notes = _build_pb_roe(assumptions, normalized, metrics, ratios)
            notes.extend(pb_notes)
            notes.append(
                "RIM (the earnings-power/equity composite model) could not be computed in the financial "
                "sector; the headline/triangulation anchor fell back to P/B x ROE."
            )
    elif sector_type == "reit":
        ffo, ffo_notes = _build_ffo(assumptions, normalized, metrics, ratios)
        notes.extend(ffo_notes)
        if ffo is None:
            pb_roe, pb_notes = _build_pb_roe(assumptions, normalized, metrics, ratios)
            notes.extend(pb_notes)
            notes.append(
                "FFO could not be computed for the REIT; the headline/triangulation anchor fell back "
                "to P/B x ROE."
            )

    # The active anchor for THIS sector's headline/triangulation purposes:
    # the FFO block when reit's FFO build succeeded, else RIM when financial's
    # RIM build succeeded, else pb_roe (the reit-FFO fallback above, the
    # financial-RIM fallback above, or None for every other sector).
    if sector_type == "reit" and ffo is not None:
        reit_or_financial_anchor = ffo
    elif sector_type == "financial" and rim is not None:
        reit_or_financial_anchor = rim
    else:
        reit_or_financial_anchor = pb_roe

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
                "Hyper-growth mode: the headline range came from the revenue-first DCF (growth fade + "
                "mature target margin); the standard FCF-DCF is secondary, in 'dcf.scenarios'."
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
                    "Cyclical + capital-intensive: since free cash flow is suppressed by growth "
                    "investment (heavy CapEx), the headline is based on an FCFE anchor applying "
                    "sustainable growth (reinvestment=g/ROE) to mid-cycle normalized earnings. The "
                    "zero-growth EPV base, the mid-cycle FCF-DCF, and the raw FCF-DCF are reported "
                    "as secondary."
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
                    "Cyclical + capital-intensive: free cash flow doesn't reflect earning power because "
                    "of growth investment; the headline is based on the zero-growth earnings-power (EPV) "
                    "anchor. The mid-cycle and raw FCF-DCF are reported as secondary."
                )
                if cyclical_fcfe_detail is not None and _is_number(cf_base_ps) and _is_number(epv_base_ps):
                    notes.append(
                        f"Note: the growth-inclusive sustainable-growth FCFE was also computed (base "
                        f"${cf_base_ps:,.2f}) but stayed below the zero-growth EPV base "
                        f"(${epv_base_ps:,.2f}), so the headline was kept at EPV; this shows that normalized "
                        "ROE is BELOW the cost of equity -- i.e. growth is NOT CREATING value (it's "
                        "destroying it). The growth-inclusive FCFE is reported as secondary under "
                        "'cyclical_fcfe_detail'."
                    )
        elif normalized_variant is not None:
            # FCF is NOT capex-suppressed for this cyclical: keep the existing
            # cycle-mid normalized FCF-DCF headline (unchanged behavior).
            primary_dcf_scenarios = normalized_variant
            notes.append(
                "Cyclical sector: the headline fair-value range was anchored on mid-cycle normalized FCF "
                "instead of a single year's (often near-trough) free cash flow; the raw trough-FCF DCF "
                "scenarios are also reported under 'dcf.scenarios'."
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
                target_pct_str = f"{target_pct * 100:.1f}%" if _is_number(target_pct) else "—"
                epv_base_str = f"${epv_base_ps:,.2f}" if _is_number(epv_base_ps) else "—"
                notes.append(
                    "Since free cash flow is suppressed by growth investment, the headline is based on a "
                    f"growth-inclusive revenue-first DCF that fades revenue and converges the FCF margin "
                    f"toward a mature target ({target_pct_str}). The zero-growth EPV base ({epv_base_str}) "
                    "and the raw FCF-DCF are reported as secondary."
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
                    "For this company, free cash flow doesn't reflect earning power because of heavy "
                    "growth investment (high CapEx); the headline fair-value range is based on the "
                    "zero-growth earnings-power (EPV) anchor. The raw FCF-DCF scenarios are reported as "
                    "secondary under 'dcf.scenarios'. NOTE: EPV is a conservative base that DELIBERATELY "
                    "excludes the growth premium; the reverse-DCF measures the price-implied growth."
                )
                if mature_revenue_detail is not None and _is_number(mr_base_ps) and _is_number(epv_base_ps):
                    notes.append(
                        f"Note: the growth-inclusive revenue-first DCF was also computed (base "
                        f"${mr_base_ps:,.2f}) but stayed below the zero-growth EPV base "
                        f"(${epv_base_ps:,.2f}), so the headline was kept at EPV; this shows that the "
                        "company's defensible mature FCF margin is thinner than its capitalized earnings. "
                        "The revenue-first band is reported as a secondary cross-check under "
                        "'mature_revenue_detail'."
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
            target_pct_str = f"{target_pct * 100:.1f}%" if _is_number(target_pct) else "—"
            notes.append(
                "Mid-growth loss-making company: the headline is based on a revenue-first DCF that fades "
                f"revenue and converges the FCF margin toward a mature target ({target_pct_str}) (realized "
                "growth in the 12-20% band, below the hyper-growth threshold). The raw FCF-DCF and "
                "multiples are reported as secondary."
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
    elif sector_type == "financial" and rim is not None:
        # SPEC.md Sec.22c: `financial` was the one headline path with no
        # scenario_meta, so its rows printed the raw (often discarded)
        # assumption growth instead of what RIM actually applied.
        scenario_meta = _rim_scenario_meta(rim, assumptions)
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
        direction_word = "above" if bracket_status == "above_bracket" else "below"
        notes.append(
            f"The price implies growth {direction_word} the reverse-DCF bracket "
            f"({reverse_dcf._BRACKET_LO * 100:.0f}%..{reverse_dcf._BRACKET_HI * 100:.0f}%)."
        )
    elif implied is None:
        notes.append("Reverse DCF (price-implied growth) could not be computed.")

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
        output_realized_label = f"revenue {revenue_cagr_label}" if revenue_cagr_label else None
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
                "Mature revenue-first DCF: the price-implied start growth rate could not be computed "
                "(the price may imply an expectation outside a plausible growth range)."
            )

        output_implied = mature_implied_growth
        output_realized_cagr = revenue_cagr
        output_realized_label = f"revenue {revenue_cagr_label}" if revenue_cagr_label else None
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
                "Mid-growth revenue-first DCF: the price-implied start growth rate could not be computed "
                "(the price may imply an expectation outside a plausible growth range)."
            )

        output_implied = mg_implied_growth
        output_realized_cagr = revenue_cagr
        output_realized_label = f"revenue {revenue_cagr_label}" if revenue_cagr_label else None
        output_bracket_status = "ok"

    # --- Multiples ---------------------------------------------------------
    history = multiples.multiples_history(normalized, price_df)
    if price_df is None or getattr(price_df, "empty", True):
        notes.append("Multiple history could not be computed because price history is unavailable.")
    ev_applicable = sector_type not in _SECTORS_WITHOUT_EV
    current, current_notes = _derive_current_multiples(
        normalized, ratios, metrics, price, suppress_ev=not ev_applicable
    )
    notes.extend(current_notes)
    if not ev_applicable:
        # SPEC.md Sec.20b. Blanking the history too means the percentiles
        # below have nothing to rank against either, so no EV signal can
        # survive by a side route. `metrics` itself is deliberately NOT
        # mutated -- suppression is scoped to this valuation's own output.
        current["ev_sales"] = None
        for row in history:
            for key in ("ev_sales", "ev_ebit", "ev_ebitda"):
                if key in row:
                    row[key] = None
        notes.append(_EV_SUPPRESSED_NOTE)
    pe_pct = multiples.percentile_position([h["pe"] for h in history], current["pe"])
    ps_pct = multiples.percentile_position([h["ps"] for h in history], current["ps"])
    pfcf_pct = multiples.percentile_position([h["pfcf"] for h in history], current["pfcf"])
    ev_ebit_pct = multiples.percentile_position([h.get("ev_ebit") for h in history], current.get("ev_ebit"))
    ev_ebitda_pct = multiples.percentile_position([h.get("ev_ebitda") for h in history], current.get("ev_ebitda"))

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
    # SPEC.md Sec.23c: reported, never promoted -- P/TBV does not enter the
    # sector-axis candidate order and never reaches triangulate.triangulate,
    # so `financial` keeps its existing P/E-primary multiples signal.
    ptbv_pct = multiples.percentile_position([h.get("ptbv") for h in history], current.get("ptbv"))

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
        # SPEC.md Sec.8i: sourced from precedent-transaction data (optional,
        # operator-curated), NOT Damodaran's own multiples.csv, which
        # carries no EV/EBITDA sector median at all.
        "ev_ebitda_median": (precedent_medians or {}).get("ev_ebitda"),
        "precedent_transactions": precedent_medians,
    }

    # Sector-relative multiples axis (VALUATION.md Sec.7 axis-b): the current
    # PRIMARY multiple over its Damodaran sector median. The primary is picked
    # with the SAME sector-type candidate order + first-non-None-percentile
    # rule as triangulate._raw_multiples_signal, so the sector axis and the
    # own-history percentile signal describe the identical multiple. No
    # Damodaran P/FFO median exists, so a reit whose primary is P/FFO yields
    # no comparison (axis disabled, own-history behavior preserved). The
    # `comparison` block is surfaced in the report so the sector standing is
    # explicit, not hidden behind a bare "karisik" signal.
    # Leverage gate (VALUATION.md Sec.2/Sec.7): net debt / EBITDA, both current
    # (from metrics). A filer at/above triangulate._LEVERAGE_EBITDA_RATIO uses
    # EV/EBITDA as its PRIMARY own-history multiple instead of P/E. Net cash
    # (net_debt <= 0) or unusable/non-positive EBITDA -> None (not leveraged).
    _net_debt = metrics.get("net_debt")
    _ebitda = metrics.get("ebitda")
    net_debt_to_ebitda = (
        _net_debt / _ebitda
        if _is_number(_net_debt) and _is_number(_ebitda) and _net_debt > 0 and _ebitda > 0
        else None
    )
    if not ev_applicable:
        # SPEC.md Sec.20b: the leverage gate exists solely to promote
        # EV/EBITDA to primary. With EV undefined there is nothing to promote,
        # and a lender's net debt / EBITDA would trip it on every filer.
        net_debt_to_ebitda = None
    leveraged = net_debt_to_ebitda is not None and net_debt_to_ebitda >= triangulate._LEVERAGE_EBITDA_RATIO

    if sector_type == "growth_unprofitable":
        _ratio_candidates = (
            (ps_pct, "P/S", current.get("ps"), sector_info["ps_median"]),
            (pe_pct, "P/E", current.get("pe"), sector_info["pe_median"]),
            (pfcf_pct, "P/FCF", current.get("pfcf"), sector_info["pfcf_median"]),
        )
    elif sector_type == "reit":
        _ratio_candidates = (
            (pffo_pct, "P/FFO", current.get("pffo"), None),
            (ps_pct, "P/S", current.get("ps"), sector_info["ps_median"]),
        )
    elif leveraged:
        # EV/EBITDA is primary. Damodaran's own multiples.csv carries no
        # EV/EBITDA sector median (SPEC.md Sec.8i note above) -- axis-b here
        # is sourced from precedent-transaction data instead
        # (sector_info["ev_ebitda_median"]), None (axis-b disabled, mirrors
        # reit's P/FFO) when no precedent-transaction data is curated. The
        # P/E fallbacks stay in the list only so a filer with no usable
        # EV/EBITDA history still resolves a primary further down.
        _ratio_candidates = (
            (ev_ebitda_pct, "EV/EBITDA", current.get("ev_ebitda"), sector_info["ev_ebitda_median"]),
            (pe_pct, "P/E", current.get("pe"), sector_info["pe_median"]),
            (ps_pct, "P/S", current.get("ps"), sector_info["ps_median"]),
            (pfcf_pct, "P/FCF", current.get("pfcf"), sector_info["pfcf_median"]),
        )
    else:
        _ratio_candidates = (
            (pe_pct, "P/E", current.get("pe"), sector_info["pe_median"]),
            (ps_pct, "P/S", current.get("ps"), sector_info["ps_median"]),
            (pfcf_pct, "P/FCF", current.get("pfcf"), sector_info["pfcf_median"]),
        )
    _primary = next(
        ((lbl, cur, med) for pct, lbl, cur, med in _ratio_candidates if pct is not None),
        None,
    )
    sector_ratio = None
    sector_info["comparison"] = {
        "label": None, "current": None, "median": None, "ratio": None, "bucket": None,
    }
    if _primary is not None:
        _lbl, _cur, _med = _primary
        if _is_number(_cur) and _is_number(_med) and _cur > 0 and _med > 0:
            sector_ratio = _cur / _med
            sector_info["comparison"] = {
                "label": _lbl,
                "current": round(_cur, 2),
                "median": round(_med, 2),
                "ratio": round(sector_ratio, 2),
                "bucket": triangulate._sector_ratio_bucket(sector_ratio),
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

    # --- Cycle position + two-regime read (SPEC.md Sec.25/26) ---------------
    # Advisory only: never headlines fair_value_range, never feeds
    # primary_dcf_scenarios, never enters triangulation. Same discipline as
    # the Sec.8g/8j/8k screens.
    cycle = None
    if sector_type == "cyclical":
        cycle_stats = cyclical.through_cycle_stats(normalized, ratios, history, metrics)
        cycle = cyclical.two_regime_valuation(cycle_stats, metrics, price)
        if cycle is None and cycle_stats is None:
            notes.append(
                "Cycle position could not be computed: not enough annual net-margin data for a "
                "through-cycle statistic (at least 6 fiscal years are required)."
            )
        elif cycle is not None:
            if cycle["stats"].get("window_short"):
                notes.append(
                    f"The cycle statistic was computed over only {cycle['stats']['years']} fiscal years; "
                    "this window spans one cycle, but WHICH cycle it caught dominates the average. Run "
                    "with `--years 12` for a wider base."
                )
            notes.append(cycle["verdict_sentence"])

    multiples_out = {
        "history": history,
        "current": current,
        "pe_percentile": pe_pct,
        "ps_percentile": ps_pct,
        "pfcf_percentile": pfcf_pct,
        "pffo_percentile": pffo_pct,
        "ev_ebit_percentile": ev_ebit_pct,
        "ev_ebitda_percentile": ev_ebitda_pct,
        "ptbv_percentile": ptbv_pct,
        "net_debt_to_ebitda": (round(net_debt_to_ebitda, 2) if net_debt_to_ebitda is not None else None),
        "leveraged": leveraged,
        "ev_applicable": ev_applicable,
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
        notes.append("The sensitivity matrix could not be computed.")

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
            "The sensitivity table reflects the mid-cycle normalized FCF-DCF base, and the reverse-DCF "
            "reflects the raw (suppressed) FCF base; both differ from the headline EPV anchor and are "
            "kept as evidence showing why free cash flow is low."
        )
        notes.append(
            "NOTE: this anchor takes the earnings base from the most recent representative (profitable) "
            "years and EXCLUDES severe cycle troughs (e.g. a memory-glut loss year) as an exception "
            "unlikely to repeat (a structural re-rating assumption). A full-cycle average that treats "
            "troughs as a permanent part of the cycle would materially lower the value."
        )
    elif epv_headline:
        notes.append(
            "The sensitivity table and reverse-DCF reflect the secondary (suppressed) FCF-DCF base, not "
            "the headline EPV anchor; kept as evidence showing why free cash flow is low."
        )
    elif cyclical_fcfe_headline:
        notes.append(
            "The sensitivity table reflects the mid-cycle normalized FCF-DCF base, and the reverse-DCF "
            "reflects the raw (suppressed) FCF base; both differ from the headline FCFE anchor and are "
            "kept as evidence showing why free cash flow is low."
        )
        notes.append(
            "NOTE: this anchor takes the earnings base from the most recent representative (profitable) "
            "years and EXCLUDES severe cycle troughs (e.g. a memory-glut loss year) as an exception "
            "unlikely to repeat (a structural re-rating assumption). A full-cycle average that treats "
            "troughs as a permanent part of the cycle would materially lower the value."
        )
    elif mature_revenue_headline:
        notes.append(
            "The sensitivity table reflects the secondary (suppressed) FCF-DCF base, not the headline "
            "mature revenue-first DCF; kept as evidence showing why free cash flow is low."
        )
    elif midgrowth_revenue_headline:
        notes.append(
            "The sensitivity table reflects the secondary FCF-DCF base, not the headline mid-growth "
            "revenue-first DCF; the standard growth+/-2pp grid doesn't apply to the growth-fade model, so "
            "the FCF-DCF grid is kept as evidence."
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
        pffo_pct=pffo_pct, cyclical_fcfe_headline=cyclical_fcfe_headline, sector_ratio=sector_ratio,
        ev_ebitda_pct=ev_ebitda_pct, net_debt_to_ebitda=net_debt_to_ebitda,
    )

    # --- Altman Z-score distress screen (SPEC.md Sec.8g) --------------------
    # ADVISORY ONLY: computed independently of everything above, never feeds
    # fair_value_range/primary_dcf_scenarios/triangulation. Not meaningful
    # for financial/reit (same rationale _SECTORS_WITHOUT_FCF_DCF already
    # documents for the FCF-DCF disablement).
    altman_z = None
    if sector_type not in _SECTORS_WITHOUT_FCF_DCF:
        altman_z, altman_notes = _build_altman_z(normalized, metrics)
        notes.extend(altman_notes)

    # --- Beneish M-score earnings-manipulation screen (SPEC.md Sec.8j) ------
    # ADVISORY ONLY, same pattern as altman_z. Computed for every sector
    # (unlike altman_z/lbo_floor_detail, this isn't leverage/EBITDA-based --
    # a bank or REIT's revenue/receivables/gross-margin trend is just as
    # meaningful an earnings-manipulation signal as any other sector's).
    beneish_m, beneish_notes = _build_beneish_m(normalized, metrics)
    notes.extend(beneish_notes)

    # --- Merton distance-to-default (SPEC.md Sec.8k) ------------------------
    # ADVISORY ONLY, same pattern as altman_z/beneish_m. Computed for every
    # sector (like beneish_m) -- unlike altman_z/lbo_floor_detail, this isn't
    # an EBITDA/leverage-ratio model that's structurally wrong for
    # financial/reit; it only needs market cap, total debt, price history,
    # and a risk-free rate, all equally meaningful across sectors.
    merton_dtd, merton_notes = _build_merton_dtd(metrics, price_df, risk_free_pct)
    notes.extend(merton_notes)

    # --- LBO-implied floor value (SPEC.md Sec.8h) ---------------------------
    # ADVISORY ONLY, same non-chain-touching pattern as altman_z above. Not
    # meaningful for financial/reit (EV/EBITDA-based leverage doesn't apply
    # to a bank's regulated capital structure or a REIT's FFO-centric one --
    # same _SECTORS_WITHOUT_FCF_DCF gate as altman_z).
    lbo_floor_detail = None
    if sector_type not in _SECTORS_WITHOUT_FCF_DCF:
        lbo_floor_detail, lbo_notes = _build_lbo_floor(metrics, fcf0)
        notes.extend(lbo_notes)

    # --- Method summary (SPEC.md Sec.8l) ------------------------------------
    # Purely additive packaging of the headline/secondary/cross-check/advisory
    # decisions already made above -- no new precedence logic, no new numeric
    # computation. Isolated in its own try/except (rather than relying on
    # run_valuation's outer catch-all) so a bug here degrades to an empty
    # list instead of discarding the entire, already-computed valuation.
    try:
        method_summary = _build_method_summary(
            sector_type, hyper_growth_active, hyper_growth_detail,
            cyclical_fcfe_headline, cyclical_fcfe_detail,
            epv_headline, earnings_power, normalized_variant, dcf_scenarios,
            mature_revenue_headline, mature_revenue_detail,
            midgrowth_revenue_headline, midgrowth_revenue_detail,
            rim, ffo, output_implied, output_bracket_status, multiples_out,
            altman_z, beneish_m, merton_dtd, lbo_floor_detail, cycle,
        )
    except Exception:  # noqa: BLE001 - method_summary must never crash the CLI
        logger.exception("method_summary derivation failed unexpectedly; degrading to an empty list.")
        method_summary = []

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
        "rim": rim,
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
        "altman_z": altman_z,
        "beneish_m": beneish_m,
        "merton_dtd": merton_dtd,
        "lbo_floor_detail": lbo_floor_detail,
        "cycle": cycle,
        "method_summary": method_summary,
        "assumptions": assumptions,
        "notes": notes,
        # Present on live runs too (``as_of: None`` inside marks them) so the
        # risk-free observation behind a stored verdict is always recoverable.
        **(
            {"macro_asof": sector_data["macro_asof"]}
            if sector_data and sector_data.get("macro_asof")
            else {}
        ),
    }
