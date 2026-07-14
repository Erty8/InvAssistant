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
from typing import Dict, List, Optional

from sec_analyzer.config import Config
from sec_analyzer.normalize.normalizer import to_annual_series
from sec_analyzer.valuation import damodaran, multiples, reverse_dcf, revenue_dcf, sanity, sector, sensitivity, triangulate
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

#: P/B x ROE fair-P/B clamp bounds (Sec.8).
_PB_CLAMP_LO = 0.5
_PB_CLAMP_HI = 4.0

#: P/B x ROE per-scenario fair-P/B scaling factors.
_PB_SCENARIO_SCALE = {"bear": 0.8, "base": 1.0, "bull": 1.2}

#: fcf0 selection: deviation threshold from the 3-year average FCF beyond
#: which the latest-FY figure is distrusted in favor of the average.
_FCF0_DEVIATION_THRESHOLD = 0.50

_SECTORS_WITHOUT_FCF_DCF = ("financial", "reit")

# --- Hyper-grower revenue-first DCF wiring (SPEC.md Sec.3 / VALUATION.md Sec.4a) ---

#: Terminal growth and steady-state (full convergence) year shared by every
#: hyper-grower scenario, deterministic and NOT overridable by
#: ``hyper_growth_extras`` (only per-scenario target margin, steady-state
#: year, probability, and ``tam_usd`` are overridable -- see SPEC Sec.5).
_HYPER_TERMINAL_GROWTH = 0.025
_HYPER_DEFAULT_STEADY_STATE_YEAR = 10

#: Per-scenario discount rate (fixed; not overridable by extras).
_HYPER_DISCOUNT_RATE_BY_SCENARIO = {"bear": 0.12, "base": 0.10, "bull": 0.09}

#: Default prob-weighting used unless ``hyper_growth_extras`` overrides a
#: scenario's probability.
_HYPER_DEFAULT_PROBABILITIES = {"bear": 0.25, "base": 0.50, "bull": 0.25}

#: Deterministic start-growth cap and mature-target-margin cap (Sec.3.1).
_HYPER_START_GROWTH_CAP = 0.40
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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _round_or_none(value: Optional[float], ndigits: int) -> Optional[float]:
    return None if value is None else round(value, ndigits)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
        "dcf": {"enabled": False, "disabled_reason": None, "scenarios": None, "normalized_variant": None},
        "pb_roe": None,
        "fair_value_range": _empty_fair_value_range(),
        "reverse_dcf": {
            "implied_growth": None, "realized_cagr_5y": None, "realized_label": None, "bracket_status": "no_data",
        },
        "multiples": {
            "history": [],
            "current": {"pe": None, "ps": None, "pfcf": None},
            "pe_percentile": None,
            "ps_percentile": None,
            "pfcf_percentile": None,
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
    latest_fy = metrics.get("latest_fy")
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
) -> "tuple[Optional[dict], List[str]]":
    """Run the 3-scenario DCF (Sec.4). Returns ``(scenarios, notes)`` where
    ``scenarios`` is ``None`` if ``fcf0``/``shares`` are unusable at all
    (nothing to compute), otherwise a dict with all three scenario keys
    present -- an individual scenario whose own assumptions are invalid
    (missing fields, or r <= g_t) becomes ``{"per_share": None, "lo": None,
    "hi": None}`` plus a note, without blocking the other scenarios. Each
    scenario's ``lo``/``hi`` band comes from its own 3x3 sensitivity grid
    (see :func:`_dcf_scenario_band`), falling back to the flat +/-10% band
    with an additional note when the grid degrades (Sec.4/F3)."""
    notes: List[str] = []
    if fcf0 is None or not shares or shares <= 0:
        return None, notes

    scenarios = {}
    for key in _SCENARIO_KEYS:
        scenario_assumptions = assumptions.get(key) or {}
        growth_5y = scenario_assumptions.get("growth_5y")
        terminal_growth = scenario_assumptions.get("terminal_growth")
        discount_rate = scenario_assumptions.get("discount_rate")

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

    return scenarios, notes


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
    latest_fy = metrics.get("latest_fy")
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
    """
    notes: List[str] = []
    shares = metrics.get("shares")
    base_assumptions = assumptions.get("base") or {}
    discount_rate_base = base_assumptions.get("discount_rate")

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

    latest_fy = metrics.get("latest_fy")
    if latest_fy is not None and selected_fy != latest_fy:
        notes.append(
            f"P/B x ROE çapası için {selected_fy} mali yılının özkaynak/ROE verisi kullanıldı "
            "(en son mali yılın temel verileriyle hisse sayısı hizalı değildi)."
        )

    fair_pb_base = _clamp(roe / discount_rate_base, _PB_CLAMP_LO, _PB_CLAMP_HI)
    book_value_per_share = equity_latest / shares

    scenarios = {}
    for key, scale in _PB_SCENARIO_SCALE.items():
        per_share = round(fair_pb_base * scale * book_value_per_share, 2)
        lo, hi, used_fallback = _pb_roe_scenario_band(roe, discount_rate_base, scale, book_value_per_share, per_share)
        if used_fallback:
            notes.append(
                f"{key.capitalize()} senaryosu için P/B x ROE duyarlılık bandı hesaplanamadı; "
                "nokta tahminin +/-%10'u fallback olarak kullanıldı."
            )
        scenarios[key] = {"per_share": per_share, "lo": lo, "hi": hi}

    return {"scenarios": scenarios}, notes


def _pb_roe_scenario_band(
    roe: float, discount_rate_base: float, scale: float, book_value_per_share: float, per_share: float
) -> "tuple[float, float, bool]":
    """Derive one P/B x ROE scenario's band from ``discount_rate_base +/-
    _DISCOUNT_RATE_STEP`` (Sec.8/F3): recompute the clamped ``fair_pb`` at
    each of the 3 nearby discount rates, scale by this scenario's own
    ``scale``/``book_value_per_share``, and take the min/max. Falls back to
    the flat +/-10% band (:func:`_band`) when fewer than
    :data:`_MIN_GRID_CELLS_FOR_BAND` discount-rate points are usable (a
    non-positive discount rate makes ``roe / dr`` meaningless and is
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
        fair_pb = _clamp(roe / dr, _PB_CLAMP_LO, _PB_CLAMP_HI)
        cells.append(round(fair_pb * scale * book_value_per_share, 2))

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _hyper_target_base(gross_margin: Optional[float], current_margin: Optional[float]) -> float:
    """``target_base`` (Sec.3.1): the mature-state FCF-margin ceiling --
    half the latest-FY gross margin (capped at 30%), or a 20% default
    ceiling when gross margin is unavailable -- floored at today's FCF
    margin whenever the filer is already profitable (a currently-
    profitable hyper-grower must never be modeled as if its margin
    collapses below what it already earns), and capped at gross margin
    when known.

    Args:
        gross_margin: The latest-FY gross margin, already filtered to
            ``None`` unless it is a positive number (callers pass ``gm``,
            not the raw ratio value).
        current_margin: Today's FCF margin (``fcf / latest_revenue``), or
            ``None``/non-positive when the filer isn't currently FCF
            profitable.
    """
    ceiling = (
        min(gross_margin * 0.5, _HYPER_TARGET_BASE_CAP)
        if gross_margin is not None
        else _HYPER_TARGET_MARGIN_CEILING_FALLBACK
    )
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
                )
            except ValueError:
                continue
            cells.append(result["per_share"])

    if len(cells) < _MIN_GRID_CELLS_FOR_BAND:
        lo, hi = _band(per_share)
        return lo, hi, True
    return round(min(cells), 2), round(max(cells), 2), False


def _build_hyper_growth(
    metrics: dict,
    ratios: list,
    normalized: dict,
    price: Optional[float],
    shares: Optional[float],
    hyper_reasons: List[str],
    extras: Optional[dict],
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

    Returns:
        A ``(detail, notes)`` tuple. ``detail`` matches SPEC Sec.3.4's
        ``hyper_growth_detail`` shape, or ``None`` if the mode couldn't be
        built at all (missing revenue/shares/realized growth, or every
        scenario failed). ``notes`` are Turkish strings the caller should
        fold into the top-level ``notes`` list (also echoed into
        ``detail["notes"]`` when ``detail`` is not ``None``).
    """
    notes: List[str] = []
    try:
        latest_fy = metrics.get("latest_fy")
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

        # SBC is now expensed directly in current_margin/target margins
        # (F2), so dilution here is share-count growth only -- no separate
        # SBC/revenue term (that would double-count the same drag).
        shares_yoy = metrics.get("shares_yoy")
        annual_dilution = _clamp(
            shares_yoy if (shares_yoy is not None and shares_yoy > 0) else 0.0,
            0.0, _HYPER_DILUTION_CAP,
        )

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

        if target_margin_overridden["base"]:
            target_margin_source = "LLM/kullanıcı tarafından sağlanan hedef marj (hyper_growth_extras)"
        else:
            # Recompute the ceiling (not the floored target_base itself) just
            # to phrase the source string correctly -- did today's positive
            # FCF margin actually raise target_base above the ceiling?
            ceiling = (
                min(gm * 0.5, _HYPER_TARGET_BASE_CAP) if gm is not None else _HYPER_TARGET_MARGIN_CEILING_FALLBACK
            )
            floored_by_current_margin = current_margin > 0 and current_margin > ceiling
            if gm is not None:
                if floored_by_current_margin:
                    target_margin_source = (
                        f"brüt marj %{gm * 100:.0f} × 0.5 (tavan %30), bugünkü FCF marjına tabanlanmış"
                    )
                else:
                    target_margin_source = "brüt marj × 0.5 (tavan %30)"
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
                latest_revenue, start_growth_by_scenario["base"], _HYPER_TERMINAL_GROWTH,
                _HYPER_DISCOUNT_RATE_BY_SCENARIO["base"], current_margin, target_by_scenario["base"],
                steady_state_by_scenario["base"], shares, annual_dilution, 0.0,
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
                    latest_revenue, start_growth, _HYPER_TERMINAL_GROWTH, discount_rate, current_margin,
                    target, steady_state_year, shares, annual_dilution, financing_shares,
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
                latest_revenue, start_growth, _HYPER_TERMINAL_GROWTH, discount_rate, current_margin,
                target, steady_state_year, shares, annual_dilution, financing_shares, per_share,
            )
            if used_fallback:
                notes.append(
                    f"{key.capitalize()} hiper-büyüme senaryosu için duyarlılık bandı hesaplanamadı; "
                    "nokta tahminin +/-%10'u fallback olarak kullanıldı."
                )
            scenarios_detail[key] = {
                "per_share": per_share, "lo": lo, "hi": hi,
                "start_growth": round(start_growth, 4), "target_fcf_margin": round(target, 4),
                "final_year_revenue": result["final_year_revenue"], "revenue_multiple": result["revenue_multiple"],
            }

        base_cell = scenarios_detail.get("base")
        if base_cell is None or base_cell.get("revenue_multiple") is None:
            notes.append(
                "Hiper-büyüme baz senaryosu hesaplanamadığı için varış noktası (arrival) bayrağı "
                "belirlenemedi; standart değerleme kullanılıyor."
            )
            return None, notes

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
            price, latest_revenue, _HYPER_TERMINAL_GROWTH, base_discount_rate, current_margin,
            base_target, base_steady_state_year, shares, annual_dilution, financing_shares,
        )
        implied_revenue_10y = None
        implied_revenue_multiple = None
        if implied_growth is not None:
            try:
                implied_projection = revenue_dcf.revenue_first_dcf(
                    latest_revenue, implied_growth, _HYPER_TERMINAL_GROWTH, base_discount_rate, current_margin,
                    base_target, base_steady_state_year, shares, annual_dilution, financing_shares,
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
            price, latest_revenue, base_start_growth, _HYPER_TERMINAL_GROWTH, base_discount_rate,
            current_margin, base_steady_state_year, shares, annual_dilution, financing_shares,
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
            "notes": list(notes),
        }
        return detail, notes
    except Exception:  # noqa: BLE001 - never let a hyper-grower bug break the standard valuation.
        logger.warning("_build_hyper_growth: unexpected error; degrading to standard valuation.", exc_info=True)
        notes.append("Hiper-büyüme modu beklenmeyen bir hatayla karşılaştı; standart değerleme kullanılıyor.")
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
        meta[key] = {
            "growth": f"%{start_growth * 100:.0f} başlangıç → %2.5 terminale fade",
            "discount_rate": f"%{discount_rate * 100:.0f}",
            "note": (
                f"Hiper-büyüme {scenario_label}: başlangıç büyüme %{start_growth * 100:.0f} "
                f"(10 yılda %2.5 terminale fade), olgun FCF marjı %{target_fcf_margin * 100:.0f}, "
                f"iskonto %{discount_rate * 100:.0f}."
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
    scenarios; all-``None`` if neither is available.

    Args:
        dcf_scenarios: The active per-share/lo/hi scenario dict (may be the
            hyper-grower revenue-first band, the cyclical normalized
            variant, or the raw FCF-DCF band -- whichever the caller has
            already selected as the headline source), or ``None``.
        pb_roe: The P/B x ROE anchor dict, used as a fallback source when
            ``dcf_scenarios`` is ``None``.
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
        disabled_reason = (
            "Finansal/GYO şirketlerde serbest nakit akışı DCF'i güvenilir değildir; "
            "bunun yerine P/B x ROE çapası kullanılıyor."
        )

    dcf_scenarios = None
    if dcf_enabled:
        dcf_scenarios, dcf_notes = _build_dcf_scenarios(assumptions, fcf0, shares, dilution_rate)
        notes.extend(dcf_notes)
        if dcf_scenarios is None and fcf0 is not None:
            notes.append("DCF hesaplanamadı: geçerli bir hisse sayısı (shares) yok.")

    normalized_variant = None
    normalized_fcf0 = None
    if sector_type == "cyclical":
        normalized_fcf0, cyclical_notes = _normalized_fcf0(normalized, metrics)
        notes.extend(cyclical_notes)
        if normalized_fcf0 is not None:
            normalized_variant, variant_notes = _build_dcf_scenarios(
                assumptions, normalized_fcf0, shares, dilution_rate
            )
            notes.extend(variant_notes)

    pb_roe = None
    if sector_type in _SECTORS_WITHOUT_FCF_DCF:
        pb_roe, pb_notes = _build_pb_roe(assumptions, normalized, metrics, ratios)
        notes.extend(pb_notes)

    # --- Hyper-grower revenue-first DCF (SPEC Sec.3) ------------------------
    # Only actually built once detected; any sub-step failure degrades the
    # OUTPUT "hyper_growth" flag back to False (detail None) rather than
    # trusting the raw detection result, so a broken hyper build never costs
    # the standard valuation below.
    hyper_growth_detail = None
    if is_hyper_grower:
        hyper_growth_detail, hyper_notes = _build_hyper_growth(
            metrics, ratios, normalized, price, shares, hyper_reasons, hyper_growth_extras,
        )
        notes.extend(hyper_notes)
    hyper_growth_active = is_hyper_grower and hyper_growth_detail is not None

    # For cyclical filers, the headline fair-value band and the triangulation
    # DCF signal should reflect through-cycle earning power, not a single
    # (often near-trough) year's FCF. Prefer the normalized-earnings variant
    # when it was successfully computed; otherwise fall back to the raw
    # FCF-DCF band. Both variants remain reported side by side under `dcf`.
    # Hyper-grower mode takes precedence over the cyclical variant, which in
    # turn takes precedence over the raw FCF-DCF band (SPEC Sec.3.5).
    primary_dcf_scenarios = dcf_scenarios
    if hyper_growth_active:
        primary_dcf_scenarios = hyper_growth_detail["scenarios"]
        notes.append(
            "Hiper-büyüme modu: manşet aralığı revenue-first DCF'ten (büyüme fade + olgun hedef marj) "
            "alındı; standart FCF-DCF ikincil olarak 'dcf.scenarios'ta."
        )
    elif sector_type == "cyclical" and normalized_variant is not None:
        primary_dcf_scenarios = normalized_variant
        notes.append(
            "Döngüsel sektör: manşet makul değer aralığı, tek bir yılın (çoğu zaman dibe yakın) "
            "serbest nakit akışı yerine döngü-ortası normalize edilmiş FCF'e dayandırıldı; "
            "ham dip-FCF DCF senaryoları ayrıca 'dcf.scenarios' altında raporlanıyor."
        )

    hyper_scenario_meta = _hyper_scenario_meta(hyper_growth_detail) if hyper_growth_active else None
    fair_value_range = _build_fair_value_range(primary_dcf_scenarios, pb_roe, assumptions, hyper_scenario_meta)

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
    # F5: the reverse-DCF bracket (-20%..+40%) can fail to bracket the
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
    latest_fy = metrics.get("latest_fy")
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

    # --- Multiples ---------------------------------------------------------
    history = multiples.multiples_history(normalized, price_df)
    if price_df is None or getattr(price_df, "empty", True):
        notes.append("Fiyat geçmişi alınamadığı için çarpan tarihçesi hesaplanamadı.")
    current, current_notes = _derive_current_multiples(normalized, ratios, metrics, price)
    notes.extend(current_notes)
    pe_pct = multiples.percentile_position([h["pe"] for h in history], current["pe"])
    ps_pct = multiples.percentile_position([h["ps"] for h in history], current["ps"])
    pfcf_pct = multiples.percentile_position([h["pfcf"] for h in history], current["pfcf"])

    sector_data = damodaran.load_sector_data(damodaran_dir if damodaran_dir is not None else Config.DAMODARAN_DIR)
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

    # --- Triangulation -------------------------------------------------------
    base_band = None
    if primary_dcf_scenarios and primary_dcf_scenarios.get("base"):
        base_band = primary_dcf_scenarios["base"]
    elif pb_roe and (pb_roe.get("scenarios") or {}).get("base"):
        base_band = pb_roe["scenarios"]["base"]

    # In hyper-grower mode, base_band above is already the revenue-first
    # DCF's base scenario band; pass its bull scenario band through too so
    # the DCF signal can distinguish "priced for high expectations" from an
    # outright "pahali" (HYPER_SPEC.md Sec.4). Non-hyper filers keep the
    # unchanged 3-way DCF signal (hyper_growth=False, bull_band=None).
    hyper_bull_band = None
    if hyper_growth_active:
        hyper_bull_band = (hyper_growth_detail.get("scenarios") or {}).get("bull")

    triangulation = triangulate.triangulate(
        price, base_band, output_implied, output_realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type,
        hyper_growth=hyper_growth_active, bull_band=hyper_bull_band, reverse_dcf_status=output_bracket_status,
        raw_growth_pair_pct=ga_raw_pct, growth_adj_pct=ga_pct,
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
        },
        "pb_roe": pb_roe,
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
        "assumptions": assumptions,
        "notes": notes,
    }
