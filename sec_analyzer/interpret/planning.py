"""Deterministic phase-2 post-processing: scenario returns, entry plan,
stop-adding signals, and thesis-anchor metric.

This module implements the mechanical structures required by
``sec_analyzer/METODOLOJI.md`` sections 4-7 (scenario table, phased entry
plan, stop-adding signals, thesis-validation metric). Every function here is
pure, ``None``-safe, and computed entirely by plain arithmetic over already
-computed inputs (:mod:`sec_analyzer.valuation.engine`'s ``valuation`` dict,
:mod:`sec_analyzer.technical.indicators`'s technical dict, the ``ratios``
list, and the earnings-catalyst estimate) -- no LLM, no network access, no
randomness. Given the same inputs, every function here always returns the
same output, and :mod:`sec_analyzer.interpret.analyzer` injects the results
uniformly for every provider (LLM or ``"script"``) exactly the way
``fair_value_range``/``confidence`` are already injected in
``_postprocess_phase2_result`` -- no provider, including the LLMs, computes
these fields itself.

Design goals, matching the rest of ``sec_analyzer.interpret``:

* **Never raise.** Every public function wraps its body in a try/except that
  logs and returns the documented degraded shape (``None``/``[]``/a
  "could not be computed" sentence) rather than letting an exception
  propagate to the CLI.
* **Fully mechanical.** Trigger levels, invalidation, and target anchors are
  derived only from numbers already present in ``valuation``/``technical`` --
  nothing here invents a level that isn't traceable to one of those inputs.
* **Turkish user-facing strings; English code/docstrings.** Trigger text,
  verdict-style labels, and rationale sentences are Turkish, per the rest of
  the ``sec_analyzer.interpret`` package.

The entry plan (:func:`compute_entry_plan`) is two-directional per
METODOLOJI.md Sec.1 item 5: each tranche carries a ``kind`` of ``"dip"``
(buy-the-dip, triggered by a daily close below a level) or ``"breakout"``
(uptrend-confirmation, triggered by a daily close above a reclaimed/broken
level). Dip tranches share one structural invalidation level; breakout
tranches each carry their own failed-breakout invalidation (their own
trigger level, less a buffer). The defensive R:R-monotonicity check is
scoped to consecutive dip-kind tranches only, since only they share a
common invalidation and are expected to move monotonically as price falls.

Accepted design tradeoff (deliberate, not a bug): sizing is unified across
both kinds -- one ~100%-summing, price-descending weight ladder covering
every selected tranche regardless of ``kind`` (the cheapest tranche, dip or
breakout, always gets the largest weight; see :data:`_SIZE_WEIGHT_EXPONENT`).
This is a value-accumulation posture: the plan is sized as if the position
will be built up gradually as price falls, not as if any single directional
move (a pure breakout rally, or a pure dip) will ever deploy the full 100%.
Two consequences follow, and both are intentional: (1) the largest
allocation can land on the lowest-R:R (deepest-dip) tranche rather than the
best risk/reward one, since size tracks "how cheap" rather than "how good";
and (2) the "lower-priced -> higher R:R" monotonicity expectation described
above is scoped to the dip ladder specifically (which shares one structural
stop) -- breakout tranches use their own tighter, per-tranche stops and are
not comparable to dip tranches (or to each other) on that R:R scale.

:func:`select_thesis_metric`'s ``quarterly_check`` (METODOLOJI.md Sec.7) is
the one exception to this module's "no I/O" rule: it reads/writes a small
``thesis_anchors`` table (:mod:`sec_analyzer.store.thesis_anchors`) to
persist the *day-1* direction of the chosen anchor metric, so later runs
check the metric against the ORIGINAL thesis direction rather than
re-deriving a direction from whatever the trend happens to be today (which
would make the invalidation check tautological -- a metric can't invalidate
a thesis it is itself redefining every run). The persisted state is read
back deterministically (never wall-clock-driven; only a stored
``established_at`` audit timestamp uses the clock, and it is never read back
into the comparison logic), and every DB call is wrapped the same
never-raise way as the rest of this module.
"""

import logging
from typing import Dict, List, Optional

from sec_analyzer.normalize.normalizer import quarterly_ratio_series, to_quarterly_series
from sec_analyzer.signals.momentum import (
    _MARGIN_TREND_DEADBAND_PP,
    _REV_ACCEL_DEADBAND_PP,
    _fcf_margin_series,
    _yoy_growth_series,
)
from sec_analyzer.store import thesis_anchors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants (all named per the house style; tune here, not inline).
# ---------------------------------------------------------------------------

#: Half-width of the price zone drawn around each mechanical trigger level
#: (e.g. 0.015 = +/-1.5%).
_ENTRY_ZONE_BAND_PCT = 0.015

#: How far below the lowest structural reference (bear.lo / low_52w, or the
#: lowest kept tranche level as a last resort) the invalidation level sits.
_INVALIDATION_BUFFER_PCT = 0.05

#: Round-trip transaction cost (commission, both legs) folded into every
#: R:R calculation, per METODOLOJI.md Sec.2.
_ROUND_TRIP_COST_PCT = 0.002

#: How close the current price must be to the invalidation level (as a
#: fraction above it) before the NEAR_INVALIDATION stop-adding signal fires.
_NEAR_INVALIDATION_BUFFER_PCT = 0.03

#: Two candidate trigger levels within this relative distance of each other
#: are treated as the same level and one is dropped -- which one depends on
#: the pass: the descending (dip) pass keeps the higher of the two, while
#: the ascending (breakout) pass keeps the lower, nearest-to-price one (see
#: :func:`_dedupe_descending`/:func:`_dedupe_ascending`).
_DEDUPE_THRESHOLD_PCT = 0.02

#: Entry-plan tranche count bounds. The lower bound is a target, not a hard
#: guarantee -- see :func:`compute_entry_plan`'s docstring for the
#: degraded-data case where fewer than 3 distinct mechanical levels exist.
_MIN_ENTRY_TRANCHES = 3
_MAX_ENTRY_TRANCHES = 5

#: Exponent controlling how aggressively tranche size grows toward the
#: cheaper (lower-priced) tranches. ``1.0`` = linear ascending weights
#: (tranche i's raw weight is ``i ** exponent`` for i = 1..N, so tranche N,
#: the cheapest, always gets the largest weight).
_SIZE_WEIGHT_EXPONENT = 1.0

# ---------------------------------------------------------------------------
# Entry-plan educational annotations ("why" strings). Purely additive: these
# label WHICH mechanical level fired / won, they never influence selection,
# sizing, invalidation, or target math above. Kept as named constants (not
# inline literals) so :data:`_TRIGGER_REASON_BY_SOURCE` can be tested against
# the exact label set actually produced by :func:`_collect_entry_candidates`.
# ---------------------------------------------------------------------------

#: Dip candidate source labels, in :func:`_collect_entry_candidates`'s own
#: order. Referenced from that function so the labels are defined once.
_DIP_SOURCE_LABELS = (
    "bear scenario lower band",
    "base scenario lower band",
    "base scenario upper band",
    "bull scenario upper band",
    "52-week low",
    "SMA50 support",
    "SMA200 support",
)

#: Breakout candidate source labels (excluding the per-zone ``"resistance /
#: prior-high breakout"`` label, which is attached once per resistance zone
#: rather than being a single fixed-position constant -- it is listed
#: separately in :data:`_BREAKOUT_RESISTANCE_SOURCE_LABEL`).
_BREAKOUT_SOURCE_LABELS = (
    "SMA50 pullback",
    "SMA200 pullback",
    "52-week high breakout",
)

#: The English label attached to every above-price ``resistance_levels``
#: zone candidate (one label shared by all zones, since any number of zones
#: can be present).
_BREAKOUT_RESISTANCE_SOURCE_LABEL = "resistance / prior-high breakout"

#: One sentence per ``source`` label explaining WHY that *kind* of
#: level is a meaningful reference point -- the technical-analysis or
#: valuation reasoning behind the level, not the specific number (the
#: number is already in ``trigger``/``price_zone``). Looked up with
#: ``.get(source, "")`` at the call site (never raises on an unknown
#: label, per this module's never-raise discipline) -- but
#: ``test_planning.py`` asserts every label in :data:`_DIP_SOURCE_LABELS`/
#: :data:`_BREAKOUT_SOURCE_LABELS`/:data:`_BREAKOUT_RESISTANCE_SOURCE_LABEL`
#: has an entry here, so a missing reason fails loudly in tests rather than
#: silently in the report.
_TRIGGER_REASON_BY_SOURCE: Dict[str, str] = {
    "bear scenario lower band": (
        "Not a chart level: this is the lower band of the valuation engine's pessimistic (bear) "
        "scenario -- the low end of the DCF/multiple-based fair-value estimate."
    ),
    "base scenario lower band": (
        "Also not technical -- this is the lower band of the valuation engine's base scenario; "
        "it marks the point where price has sagged below the fair-value estimate."
    ),
    "base scenario upper band": (
        "The upper band of the valuation engine's base scenario; not a technical resistance "
        "level, but the upper end of the fair-value range."
    ),
    "bull scenario upper band": (
        "The upper band of the valuation engine's optimistic (bull) scenario; appearing here as "
        "a dip tranche means price is still below even this optimistic scenario -- again, a "
        "valuation-derived level, not a technical one."
    ),
    "52-week low": (
        "The lowest close of the past 52 weeks; a level other market participants have actually "
        "tested, so breaking below it is not an arbitrary number but a sign that the trading "
        "range has genuinely shifted."
    ),
    "SMA50 support": (
        "The 50-day simple moving average; a dynamic support/resistance line watched by a large "
        "number of market participants, so price's reaction here is partly self-fulfilling."
    ),
    "SMA200 support": (
        "The 200-day simple moving average; considered the reference point for the long-term "
        "trend, and since it is widely watched, reactions here are also partly self-fulfilling."
    ),
    "SMA50 pullback": (
        "Price reclaiming the 50-day moving average to the upside; a move above this "
        "widely-watched dynamic level is taken as confirmation that the short-term trend has "
        "turned back up."
    ),
    "SMA200 pullback": (
        "Price reclaiming the 200-day moving average to the upside; a move above this long-term "
        "trend reference is a widely-watched confirmation that the primary trend's direction has "
        "changed."
    ),
    "resistance / prior-high breakout": (
        "A level where sellers previously overpowered buyers; an upside break of this resistance "
        "shows the supply-demand balance has now turned in buyers' favor."
    ),
    "52-week high breakout": (
        "The highest close of the past 52 weeks; breaking above it is a sign the trading range "
        "has moved into new territory -- a genuine regime change."
    ),
}

#: Sentence for a dip tranche's ``invalidation_reason``: which of the
#: three structural-floor candidates (bear.lo / low_52w / the lowest kept dip
#: level) actually bound for this plan, plus the buffer rule itself. See
#: :func:`_resolve_invalidation`.
_DIP_INVALIDATION_REASON_TEMPLATE = (
    "This level sits {buffer_pct:.0f}% below the {source}-based floor; it is the SHARED "
    "structural invalidation level for ALL accumulation (dip) tranches across the plan -- not "
    "just one tranche, the entire thesis is questioned here."
)

#: Sentence for a breakout tranche's ``invalidation_reason``: unlike
#: dip tranches, each breakout tranche has its own invalidation (a buffer
#: below its own trigger level), so a failed breakout only invalidates that
#: one tranche.
_BREAKOUT_INVALIDATION_REASON = (
    "Unlike dip tranches, each breakout tranche's invalidation level is its own: this tranche's "
    "invalidation is a buffer {buffer_pct:.0f}% below its own trigger level. A failed breakout "
    "invalidates only this tranche, not the whole plan."
).format(buffer_pct=_INVALIDATION_BUFFER_PCT * 100)

#: Sentence for ``target_reason`` when the shared upside target came
#: from bull.hi (see :func:`_resolve_target`).
_TARGET_REASON_BULL = (
    "The target is based on the bull scenario's upper band (falling back to the base scenario's "
    "upper band if unavailable); this is an output of the DCF/multiple-based valuation engine, "
    "not a technical level."
)

#: Sentence for ``target_reason`` when bull.hi was unavailable and
#: the shared upside target fell back to base.hi.
_TARGET_REASON_BASE_FALLBACK = (
    "Since the bull scenario's upper band is unavailable, the target is based on the base "
    "scenario's upper band; this too is an output of the DCF/multiple-based valuation engine, "
    "not a technical level."
)

#: Sentence for ``target_reason`` when neither bull.hi nor base.hi
#: is available (``target`` is ``None``).
_TARGET_REASON_UNAVAILABLE = (
    "The target level could not be computed: neither the bull nor the base scenario's upper "
    "band is available."
)

#: Sentence for ``size_reason``, repeated on every tranche (mirrors
#: how ``target``/``target_reason`` are already repeated): paraphrases the
#: module docstring's "Accepted design tradeoff" ascending-weight-toward-
#: cheaper-tranche rule for an end-user reader.
_SIZE_REASON_TR = (
    "The position is sized to be built up gradually as price falls; so the cheapest tranche -- "
    "whether dip or breakout -- always gets the largest share of the intended position."
)

#: Absolute change (in the metric's own units, e.g. 0.01 = 1 percentage
#: point for a margin/growth-rate metric) below which a thesis metric's
#: year-over-year move is read as "flat" rather than
#: improving/deteriorating.
_TREND_FLAT_THRESHOLD = 0.01

#: Position thresholds (fraction of the trough->peak range) that bucket the
#: thesis metric's current value into a qualitative "where in the cycle"
#: descriptor. These match the report's cycle-position bar zones; the raw
#: ``position`` float is always returned too, so the template can place the
#: marker precisely regardless of these buckets.
_CYCLE_NEAR_TROUGH = 0.2
_CYCLE_NEAR_PEAK = 0.8

#: METODOLOJI.md Sec.7's invalidation rule, appended to every thesis-metric
#: rationale regardless of sector.
_THESIS_INVALIDATION_RULE_TR = (
    "METODOLOJI §7 rule: if this metric shows the opposite of the thesis for two consecutive "
    "quarters, the thesis is considered invalidated, and this is stated explicitly."
)

#: Appended to the rationale, in ADDITION to :data:`_THESIS_INVALIDATION_RULE_TR`
#: above, only when :func:`_classify_quarterly_series` actually finds the
#: metric moving against the established (day-1) thesis direction for 2+
#: consecutive quarters -- METODOLOJI.md Sec.7 requires this be stated
#: EXPLICITLY ("bu açıkça söylenir"), not merely left implied by the
#: `quarterly_check` data. Kept in Turkish per CLAUDE.md's user-facing-string
#: rule; the rest of this module's rationale/reason strings are English in
#: the current working tree, a pre-existing discrepancy outside this
#: function's scope (see this change's final report).
_THESIS_INVALIDATION_TRIGGERED_TR = (
    "UYARI: METODOLOJI §7 kuralı gereği -- bu metrik son çeyreklerde art arda tezin aksi yönünde "
    "hareket etti; tez GEÇERSİZ sayılır ve bu açıkça belirtilir."
)

#: Most-recent quarters considered by the quarterly-invalidation check
#: (METODOLOJI.md Sec.7). Generous enough to show a short sparkline of
#: recent quarters without dragging in ancient ones.
_QUARTERLY_CHECK_WINDOW = 8

#: Below this many usable quarters in the window, no quarterly-invalidation
#: check is attempted -- fewer can't even produce the two comparison deltas
#: needed to populate a 2-quarter against-thesis streak.
_QUARTERLY_CHECK_MIN_QUARTERS = 3

#: Per-anchor-metric-key deadband (percentage points) for the quarter-over-
#: quarter with/against-thesis classification below -- mirrors momentum.py's
#: own noise-guard constants
#: (:data:`sec_analyzer.signals.momentum._MARGIN_TREND_DEADBAND_PP` for
#: margin/ROE-shaped ratios, :data:`sec_analyzer.signals.momentum.
#: _REV_ACCEL_DEADBAND_PP` for the growth-rate anchor) rather than reacting
#: to any nonzero quarter-over-quarter move.
_QUARTERLY_CHECK_DEADBAND_PP: Dict[str, float] = {
    "net_margin": _MARGIN_TREND_DEADBAND_PP,
    "gross_margin": _MARGIN_TREND_DEADBAND_PP,
    "fcf_margin": _MARGIN_TREND_DEADBAND_PP,
    "roe": _MARGIN_TREND_DEADBAND_PP,
    "yoy_revenue_growth": _REV_ACCEL_DEADBAND_PP,
}

#: Sector-type -> ordered list of (ratio-row key, display name)
#: candidates for the thesis anchor metric. The first candidate with a
#: computable latest-fiscal-year value in ``ratios`` wins; if none do, the
#: first candidate's name is still reported with ``latest_value=None``.
_SECTOR_METRIC_CANDIDATES: Dict[Optional[str], List["tuple[str, str]"]] = {
    "mature": [("net_margin", "Net Profit Margin"), ("roe", "Return on Equity (ROE)")],
    "growth_unprofitable": [("yoy_revenue_growth", "Annual Revenue Growth (YoY)")],
    "financial": [("roe", "Return on Equity (ROE, NIM proxy)")],
    "reit": [("fcf_margin", "FCF Margin (FFO proxy)")],
    "cyclical": [("gross_margin", "Gross Profit Margin"), ("net_margin", "Net Profit Margin")],
}

#: Default candidate list used for ``None``/unrecognized ``sector_type``.
_DEFAULT_METRIC_CANDIDATES: List["tuple[str, str]"] = [("net_margin", "Net Profit Margin")]

#: For a ratio-row key that has no meaningful multi-year series in
#: ``ratios`` on its own (only ``yoy_revenue_growth`` today), an ordered
#: list of ``metrics`` dict keys to try as a single-point fallback value
#: (no year-over-year trend is derivable from a single ``metrics`` figure).
_METRICS_FALLBACK_FOR_RATIO_KEY: Dict[str, "tuple[str, ...]"] = {
    "yoy_revenue_growth": ("revenue_cagr_5y", "revenue_cagr_3y"),
}

#: One sentence per sector_type explaining why the chosen metric
#: anchors the thesis. Combined with :data:`_THESIS_INVALIDATION_RULE_TR`.
_RATIONALE_BY_SECTOR: Dict[Optional[str], str] = {
    "mature": (
        "In mature sectors, the health of the thesis is seen in the stability of the profit "
        "margin; therefore net profit margin (or ROE if not computable) is tracked as the single "
        "anchor metric."
    ),
    "growth_unprofitable": (
        "In growth stories that are not yet profitable, the thesis rests on revenue growth "
        "re-accelerating; therefore annual revenue growth is tracked as the single anchor metric."
    ),
    "financial": (
        "In financial institutions, return on equity (ROE) is tracked as the single anchor "
        "metric because it is an indirect indicator of net interest margin (NIM)."
    ),
    "reit": (
        "In REITs, the thesis rests on the trajectory of FCF margin, the closest computable "
        "proxy for FFO (funds from operations)."
    ),
    "cyclical": (
        "In cyclical companies, the thesis rests on the margin's trajectory relative to its "
        "mid-cycle level; therefore gross (or net if not computable) profit margin is tracked as "
        "the single anchor metric."
    ),
    None: "Since the sector classification is undetermined, net profit margin is tracked as the default single anchor metric.",
}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def _format_pct(value: float) -> str:
    """Render a decimal-fraction ratio/growth rate as a percent string with
    one decimal, e.g. ``0.234 -> "23.4%"``."""
    return f"{value * 100:.1f}%"


def _classify_trend(diff: float) -> str:
    """Classify a year-over-year change in a "higher is better" metric
    (margin, ROE, or growth rate) into a trend label."""
    if abs(diff) < _TREND_FLAT_THRESHOLD:
        return "flat"
    return "improving" if diff > 0 else "deteriorating"


def _empty_scenario_returns() -> dict:
    return {key: {"ret_lo_pct": None, "ret_hi_pct": None} for key in ("bear", "base", "bull")}


def compute_scenario_returns(fair_value_range: Optional[dict], price: Optional[float]) -> dict:
    """Compute the current-price-relative return for each fair-value band edge.

    METODOLOJI.md Sec.1 item 4 ("Senaryo tablosu"): every scenario row needs
    the percentage return from the current price to each band edge, so the
    report can show both the price target and the implied return.

    Args:
        fair_value_range: ``valuation["fair_value_range"]`` (``{"bear":
            {"lo", "hi", ...}, "base": {...}, "bull": {...}}``), or ``None``.
            Passed by reference from the caller's ``valuation`` dict --
            this function never mutates it.
        price: Current market price per share, or ``None``.

    Returns:
        ``{"bear": {"ret_lo_pct": float|None, "ret_hi_pct": float|None},
        "base": {...}, "bull": {...}}`` -- always all three scenario keys,
        even when every value degrades to ``None`` (missing price, missing
        band, or a non-positive price). Never raises.
    """
    try:
        return _compute_scenario_returns(fair_value_range or {}, price)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("compute_scenario_returns() failed unexpectedly; returning an all-None result.")
        return _empty_scenario_returns()


def _compute_scenario_returns(fair_value_range: dict, price: Optional[float]) -> dict:
    result = {}
    price_usable = price is not None and price > 0
    for key in ("bear", "base", "bull"):
        band = fair_value_range.get(key) or {}
        lo, hi = band.get("lo"), band.get("hi")
        ret_lo_pct = round((lo / price - 1) * 100, 1) if price_usable and lo is not None else None
        ret_hi_pct = round((hi / price - 1) * 100, 1) if price_usable and hi is not None else None
        result[key] = {"ret_lo_pct": ret_lo_pct, "ret_hi_pct": ret_hi_pct}
    return result


def _collect_entry_candidates(
    valuation: dict, technical: Optional[dict], price: float
) -> "tuple[List[dict], List[dict]]":
    """Gather every mechanical trigger-level candidate named in the spec,
    split into the two directional kinds METODOLOJI.md Sec.1 item 5 requires.

    Dip candidates (``kind="dip"``, level <= ``price``): the valuation
    fair-value band's bear.lo/base.lo/base.hi/bull.hi, and the technical
    read's low_52w/sma50/sma200.

    Breakout candidates (``kind="breakout"``, level > ``price``): sma50/
    sma200 when above price (an uptrend-confirmation reclaim), each
    ``resistance_levels`` zone's price when above price (a resistance/
    prior-swing-high breakout), and high_52w when above price (a 52-week-
    high breakout) -- unless an above-price ``resistance_levels`` zone is
    itself the 52-week high (``zone["is_52w_high"]``), in which case
    high_52w is skipped to avoid double-counting the same "new highs" event
    as two near-identical candidates. ``base.hi``/``bull.hi`` are
    intentionally never used as breakout triggers -- ``bull.hi`` is the
    shared upside target (see :func:`_resolve_target`), and ``base.hi`` is
    excluded by product decision.

    Every candidate is tagged with a short ``source`` label (used
    verbatim in the trigger sentence, both kinds, so the reader can see
    where each mechanical level came from). Missing/non-numeric values
    are simply omitted from both lists.

    Returns:
        ``(dip_candidates, breakout_candidates)``, each a list of
        ``{"source": str, "level": float, "kind": "dip"|"breakout"}``.
    """
    fvr = valuation.get("fair_value_range") or {}
    bear, base, bull = fvr.get("bear") or {}, fvr.get("base") or {}, fvr.get("bull") or {}
    technical = technical or {}

    dip_raw = list(
        zip(
            _DIP_SOURCE_LABELS,
            (
                bear.get("lo"),
                base.get("lo"),
                base.get("hi"),
                bull.get("hi"),
                technical.get("low_52w"),
                technical.get("sma50"),
                technical.get("sma200"),
            ),
        )
    )
    dip = [
        {"source": label, "level": float(value), "kind": "dip"}
        for label, value in dip_raw
        if isinstance(value, (int, float)) and value <= price
    ]

    breakout_raw = [
        (_BREAKOUT_SOURCE_LABELS[0], technical.get("sma50")),
        (_BREAKOUT_SOURCE_LABELS[1], technical.get("sma200")),
    ]
    resistance_zones = technical.get("resistance_levels") or []
    for zone in resistance_zones:
        breakout_raw.append((_BREAKOUT_RESISTANCE_SOURCE_LABEL, (zone or {}).get("price")))

    # Avoid double-counting: a resistance zone can itself BE the 52-week
    # high (zone["is_52w_high"]), in which case adding high_52w separately
    # would produce two near-identical "new highs" breakout candidates that
    # only the dedupe threshold would (unreliably) collapse. Prefer the
    # resistance zone and skip the separate high_52w candidate whenever any
    # above-price resistance zone already is the 52-week high.
    resistance_is_52w_high = any(
        (zone or {}).get("is_52w_high") and isinstance((zone or {}).get("price"), (int, float)) and (zone or {}).get("price") > price
        for zone in resistance_zones
    )
    if not resistance_is_52w_high:
        breakout_raw.append((_BREAKOUT_SOURCE_LABELS[2], technical.get("high_52w")))

    breakout = [
        {"source": label, "level": float(value), "kind": "breakout"}
        for label, value in breakout_raw
        if isinstance(value, (int, float)) and value > price
    ]

    return dip, breakout


def _dedupe_by_level(candidates: "List[dict]", descending: bool) -> "List[dict]":
    """Sort ``candidates`` by ``level`` (descending or ascending) and drop
    any that land within :data:`_DEDUPE_THRESHOLD_PCT` of the
    previously-kept level."""
    ordered = sorted(candidates, key=lambda c: c["level"], reverse=descending)
    kept: "List[dict]" = []
    for cand in ordered:
        prev = kept[-1]["level"] if kept else None
        if prev is not None and prev != 0 and abs(cand["level"] - prev) / abs(prev) < _DEDUPE_THRESHOLD_PCT:
            continue
        kept.append(cand)
    return kept


def _dedupe_descending(candidates: "List[dict]") -> "List[dict]":
    """Dedupe dip candidates, nearest-below-price (highest level) first."""
    return _dedupe_by_level(candidates, descending=True)


def _dedupe_ascending(candidates: "List[dict]") -> "List[dict]":
    """Dedupe breakout candidates, nearest-above-price (lowest level) first."""
    return _dedupe_by_level(candidates, descending=False)


def _select_tranche_candidates(dip: "List[dict]", breakout: "List[dict]") -> "List[dict]":
    """Pick up to :data:`_MAX_ENTRY_TRANCHES` candidates total, keeping both
    directional sides represented whenever both have candidates.

    - Only one side has candidates: take up to the cap from that side
      (preserves the dip-only behavior from before breakout tranches
      existed).
    - Both sides have candidates and together fit within the cap: keep all
      of them.
    - Both sides have candidates and together exceed the cap: guarantee one
      slot per side (nearest to price on each side), then fill the
      remaining slots by alternating sides, each time taking that side's
      next nearest-to-price candidate, so neither side crowds out the
      other.
    """
    dip_sorted = _dedupe_descending(dip)  # nearest-below-price first
    breakout_sorted = _dedupe_ascending(breakout)  # nearest-above-price first

    if not dip_sorted and not breakout_sorted:
        return []
    if not dip_sorted or not breakout_sorted:
        side = dip_sorted or breakout_sorted
        return side[:_MAX_ENTRY_TRANCHES]
    if len(dip_sorted) + len(breakout_sorted) <= _MAX_ENTRY_TRANCHES:
        return dip_sorted + breakout_sorted

    selected = [breakout_sorted[0], dip_sorted[0]]
    bi, di = 1, 1
    take_breakout = True
    while len(selected) < _MAX_ENTRY_TRANCHES:
        if take_breakout and bi < len(breakout_sorted):
            selected.append(breakout_sorted[bi])
            bi += 1
        elif not take_breakout and di < len(dip_sorted):
            selected.append(dip_sorted[di])
            di += 1
        elif bi < len(breakout_sorted):
            selected.append(breakout_sorted[bi])
            bi += 1
        elif di < len(dip_sorted):
            selected.append(dip_sorted[di])
            di += 1
        else:
            break
        take_breakout = not take_breakout
    return selected


#: Source label used for ``target_reason``/tuple return when
#: :func:`_resolve_target` used ``bull.hi``.
_TARGET_SOURCE_BULL = "bull scenario upper band"

#: Source label used when :func:`_resolve_target` fell back to ``base.hi``.
_TARGET_SOURCE_BASE = "base scenario upper band"

#: Source label used when :func:`_resolve_invalidation`'s binding floor
#: candidate was ``bear.lo``.
_INVALIDATION_SOURCE_BEAR_LO = "bear scenario lower band"

#: Source label used when the binding floor candidate was ``low_52w``.
_INVALIDATION_SOURCE_LOW_52W = "52-week low"

#: Source label used when the binding floor candidate was the lowest kept
#: dip tranche's own level (no bear.lo/low_52w value sits at or below it).
_INVALIDATION_SOURCE_LOWEST_DIP = "lowest accumulation tranche"


def _resolve_target(valuation: dict) -> "tuple[Optional[float], Optional[str]]":
    """Upside anchor for entry-plan targets: prefer bull.hi, else base.hi.

    Returns:
        ``(target, source_label)`` -- ``source_label`` is
        :data:`_TARGET_SOURCE_BULL`/:data:`_TARGET_SOURCE_BASE` naming which
        branch actually fired for this ``valuation`` input, or ``(None,
        None)`` when neither band edge is available.
    """
    fvr = valuation.get("fair_value_range") or {}
    bull_hi = (fvr.get("bull") or {}).get("hi")
    if bull_hi is not None:
        return bull_hi, _TARGET_SOURCE_BULL
    base_hi = (fvr.get("base") or {}).get("hi")
    if base_hi is not None:
        return base_hi, _TARGET_SOURCE_BASE
    return None, None


def _resolve_invalidation(
    valuation: dict, technical: Optional[dict], lowest_kept_level: float
) -> "tuple[float, str]":
    """Shared structural invalidation level for dip-kind tranches: a buffer
    below the lowest of bear.lo/low_52w/``lowest_kept_level`` (the lowest
    kept *dip* tranche's level). ``lowest_kept_level`` is always included in
    the floor -- not just used as a last resort -- so this always sits
    strictly below every dip tranche's price zone by construction, even
    when a dip level (e.g. an sma200/base_lo level) falls below
    bear.lo/low_52w. Breakout tranches use their own per-tranche failed-
    breakout invalidation instead (see :func:`_compute_entry_plan`).

    Returns:
        ``(level, source_label)`` -- ``source_label`` is one of
        :data:`_INVALIDATION_SOURCE_BEAR_LO`/
        :data:`_INVALIDATION_SOURCE_LOW_52W`/
        :data:`_INVALIDATION_SOURCE_LOWEST_DIP`, naming which of the three
        candidates was the actual (lowest) binding floor for this plan. On a
        tie, the first-listed candidate in ``(bear_lo, low_52w,
        lowest_kept_level)`` order wins, matching this function's own
        evaluation order.
    """
    fvr = valuation.get("fair_value_range") or {}
    bear_lo = (fvr.get("bear") or {}).get("lo")
    low_52w = (technical or {}).get("low_52w")
    candidates = [
        (_INVALIDATION_SOURCE_BEAR_LO, bear_lo),
        (_INVALIDATION_SOURCE_LOW_52W, low_52w),
        (_INVALIDATION_SOURCE_LOWEST_DIP, lowest_kept_level),
    ]
    usable = [(label, value) for label, value in candidates if value is not None]
    source_label, base_level = min(usable, key=lambda pair: pair[1])
    return round(base_level * (1 - _INVALIDATION_BUFFER_PCT), 2), source_label


def _support_confluence_note(level: float, technical: Optional[dict]) -> Optional[str]:
    """Note when a dip tranche's trigger level lands on/inside one of
    the technical read's ``support_levels`` zones.

    Dip candidates come from the valuation band (plus low_52w/SMA50/SMA200),
    NOT from ``support_levels`` -- that asymmetry is deliberate (dip levels
    stay value-anchored; see :func:`_collect_entry_candidates`). This note is
    the bridge between the two report sections: when a value-derived level
    happens to coincide with a swing-tested support zone, say so, so the
    reader doesn't wonder why the technical card and the entry plan show
    near-identical-but-different numbers.

    A level "coincides" with a zone when it falls inside the zone's
    ``low``/``high`` band widened by :data:`_DEDUPE_THRESHOLD_PCT` on each
    side (the same tolerance used to call two candidate levels "the same").
    Purely additive: selection, sizing, invalidation, and R:R are untouched.

    Returns the note for the first (nearest-to-price) matching zone, or
    ``None`` when there is no match / no usable zone data.
    """
    zones = (technical or {}).get("support_levels") or []
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        zone_lo, zone_hi = zone.get("low"), zone.get("high")
        if not isinstance(zone_lo, (int, float)) or not isinstance(zone_hi, (int, float)):
            continue
        if zone_lo * (1 - _DEDUPE_THRESHOLD_PCT) <= level <= zone_hi * (1 + _DEDUPE_THRESHOLD_PCT):
            return (
                f"Overlaps with a technical support zone ({zone_lo:.2f}-{zone_hi:.2f} USD)."
            )
    return None


#: Stabilization precondition appended to dip tranches when momentum flags a
#: falling knife (cheap fundamentals + negative price momentum). It gates the
#: TIMING of a dip entry, not its price level -- the tranche's trigger/size are
#: unchanged; the reader is told to wait for a momentum turn before acting.
_STABILIZATION_NOTE = (
    "Stabilization condition: momentum is negative (falling knife) -- do not buy immediately "
    "when the dip trigger fires; wait for RSI to reclaim above 30 or for MACD upward-crossover "
    "confirmation."
)


def apply_stabilization_condition(entry_plan: Optional[list], active: bool) -> Optional[list]:
    """Append the falling-knife stabilization precondition to every dip
    tranche's ``note`` when ``active`` (a cheap + negative-momentum
    cross-signal fired). Pure and defensive: returns ``entry_plan`` unchanged
    when inactive, empty, or not a list, and never overwrites an existing note
    (it appends). Only the timing note changes -- trigger levels, sizes and
    targets are untouched.
    """
    if not active or not isinstance(entry_plan, list):
        return entry_plan
    for tranche in entry_plan:
        if not isinstance(tranche, dict) or tranche.get("kind") != "dip":
            continue
        existing = tranche.get("note")
        tranche["note"] = f"{existing} {_STABILIZATION_NOTE}" if existing else _STABILIZATION_NOTE
    return entry_plan


def compute_entry_plan(valuation: Optional[dict], technical: Optional[dict], price: Optional[float]) -> list:
    """Build the mechanical, tranche-based scale-in plan (METODOLOJI.md
    Sec.1 item 5, "Phased entry plan") -- two directional tranche kinds,
    unified into one plan.

    Two candidate sets are collected (see :func:`_collect_entry_candidates`):
    **dip** candidates (``kind="dip"``, level <= price -- from the
    fair-value band's ``bear.lo``/``base.lo``/``base.hi``/``bull.hi`` and
    the technical read's ``low_52w``/``sma50``/``sma200``) and **breakout**
    candidates (``kind="breakout"``, level > price -- ``sma50``/``sma200``
    reclaims, ``resistance_levels`` breakouts, and a ``high_52w`` breakout).
    Each side is deduplicated independently when two of its own levels sit
    within :data:`_DEDUPE_THRESHOLD_PCT` of each other. When both sides
    have candidates, a balanced subset of up to :data:`_MAX_ENTRY_TRANCHES`
    is selected guaranteeing at least one tranche per side (see
    :func:`_select_tranche_candidates`); when only one side has candidates,
    up to the cap is taken from that side alone (today's dip-only
    behavior). The final list is ordered by descending price (breakout
    tranches on top, dip tranches below) and numbered top-to-bottom.

    A single upside target applies to every tranche (see
    :func:`_resolve_target`). Invalidation is per-tranche: dip tranches
    share one structural invalidation level (see
    :func:`_resolve_invalidation`, scoped to the kept dip levels only);
    each breakout tranche carries its own failed-breakout invalidation --
    a buffer below *that tranche's own* trigger level, since a daily close
    back below a reclaimed/broken level voids that setup specifically, not
    the whole plan. R:R is computed per tranche against its own
    invalidation, and is only set when both risk and reward are positive
    (a breakout tranche whose entry is at/above the shared target has no
    reward, so its ``rr`` is ``None``).

    Because dip tranches share one fixed invalidation/target while their
    entry price decreases from tranche to tranche, dip-side R:R is
    mathematically guaranteed to be non-decreasing as price decreases
    (lower entry -> larger reward, smaller risk). The explicit post-hoc
    monotonicity check below is scoped to consecutive dip-kind tranches
    only (breakout tranches each have their own invalidation, so no such
    guarantee -- and no such check -- applies to them).

    Args:
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation`, or
            ``None``.
        technical: The merged indicators + verdict dict from
            :mod:`sec_analyzer.technical`, or ``None``.
        price: Current market price per share, or ``None``.

    Returns:
        A list of 1-5 tranche dicts (target 3-5; see below for the
        degraded case), ordered by descending ``price_zone`` level (the
        highest-priced tranche -- a breakout tranche when present -- comes
        first)::

            {
              "n": int,                          # 1-based order
              "trigger": str,                     # daily-close-only condition
              "price_zone": {"lo": float, "hi": float},
              "size_pct": float,                   # % of intended full position
              "invalidation": float,               # daily-close level; thesis void below it
              "target": float|None,                # upside anchor; None if neither
                                                    # base.hi nor bull.hi is available
              "rr": float|None,                    # reward:risk, 1dp; None if no
                                                    # positive reward or risk
              "note": str|None,                     # "Above model" for an
                                                    # above-target breakout, or a
                                                    # support-zone confluence note
                                                    # for a dip level that lands on
                                                    # a technical support zone
              "kind": str,                          # "dip" or "breakout"
              "trigger_reason": str,                 # WHY this *kind* of
                                                    # level (named in "source",
                                                    # not the specific number) is
                                                    # a meaningful TA/valuation
                                                    # reference point; "" only if
                                                    # a new source label is ever
                                                    # added without a lookup entry
                                                    # (see _TRIGGER_REASON_BY_SOURCE)
              "invalidation_reason": str,            # for a dip tranche,
                                                    # names which of bear.lo/
                                                    # low_52w/lowest-kept-dip-level
                                                    # is the actual binding floor
                                                    # and states the shared-stop
                                                    # rule; for a breakout tranche,
                                                    # explains its own per-tranche
                                                    # failed-breakout stop -- both
                                                    # state the buffer percentage
              "target_reason": str,                  # same value on
                                                    # every tranche in a plan --
                                                    # names whether the shared
                                                    # target came from bull.hi or
                                                    # the base.hi fallback (or
                                                    # neither, if target is None)
              "size_reason": str,                    # same value on
                                                    # every tranche in a plan --
                                                    # explains the ascending-
                                                    # weight-toward-cheaper-
                                                    # tranche sizing rule
            }

        ``[]`` if ``price`` is missing/non-positive, or if neither side
        yields any usable candidate. If usable candidates exist but fewer
        than 3 distinct levels survive filtering/deduplication/selection,
        this returns fewer than 3 tranches rather than fabricating extra
        levels not traceable to the inputs above -- see the module
        docstring's "fully mechanical" design goal. Never raises.
    """
    try:
        return _compute_entry_plan(valuation or {}, technical, price)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("compute_entry_plan() failed unexpectedly; returning an empty plan.")
        return []


def _compute_entry_plan(valuation: dict, technical: Optional[dict], price: Optional[float]) -> list:
    if price is None or price <= 0:
        return []

    dip_candidates, breakout_candidates = _collect_entry_candidates(valuation, technical, price)
    if not dip_candidates and not breakout_candidates:
        return []

    selected = _select_tranche_candidates(dip_candidates, breakout_candidates)
    if not selected:
        return []

    ordered = sorted(selected, key=lambda c: -c["level"])
    n = len(ordered)
    target, target_source = _resolve_target(valuation)
    if target_source == _TARGET_SOURCE_BULL:
        target_reason = _TARGET_REASON_BULL
    elif target_source == _TARGET_SOURCE_BASE:
        target_reason = _TARGET_REASON_BASE_FALLBACK
    else:
        target_reason = _TARGET_REASON_UNAVAILABLE

    dip_levels = [c["level"] for c in ordered if c["kind"] == "dip"]
    if dip_levels:
        dip_invalidation, dip_invalidation_source = _resolve_invalidation(valuation, technical, min(dip_levels))
        dip_invalidation_reason = _DIP_INVALIDATION_REASON_TEMPLATE.format(
            source=dip_invalidation_source, buffer_pct=_INVALIDATION_BUFFER_PCT * 100
        )
    else:
        dip_invalidation, dip_invalidation_reason = None, ""

    weights = [(idx + 1) ** _SIZE_WEIGHT_EXPONENT for idx in range(n)]
    weight_sum = sum(weights)
    size_pcts = [round(w / weight_sum * 100, 1) for w in weights]

    tranches = []
    for idx, cand in enumerate(ordered):
        level, kind, source = cand["level"], cand["kind"], cand["source"]
        lo = round(level * (1 - _ENTRY_ZONE_BAND_PCT), 2)
        hi = round(level * (1 + _ENTRY_ZONE_BAND_PCT), 2)
        entry = round((lo + hi) / 2, 2)

        if kind == "dip":
            invalidation = dip_invalidation
            invalidation_reason = dip_invalidation_reason
            trigger = (
                f"If the daily close falls below {level:.2f} USD (zone "
                f"{lo:.2f}-{hi:.2f} USD -- {source}); an intraday touch does not count as a "
                "trigger."
            )
        else:
            invalidation = round(level * (1 - _INVALIDATION_BUFFER_PCT), 2)
            invalidation_reason = _BREAKOUT_INVALIDATION_REASON
            trigger = (
                f"If the daily close rises above {level:.2f} USD (upside confirmation "
                f"-- {source}); an intraday touch does not count as a trigger."
            )

        rr = None
        if target is not None and invalidation is not None:
            reward = target * (1 - _ROUND_TRIP_COST_PCT) - entry * (1 + _ROUND_TRIP_COST_PCT)
            risk = entry * (1 + _ROUND_TRIP_COST_PCT) - invalidation
            if risk > 0 and reward > 0:
                rr = round(reward / risk, 1)

        note = None
        if kind == "dip":
            # Bridge to the technical card: flag when this value-derived dip
            # level coincides with a swing-tested support zone (see
            # _support_confluence_note). Informational only.
            note = _support_confluence_note(level, technical)
        if kind == "breakout" and target is not None and entry >= target:
            # Product decision: keep above-target breakout tranches (they're
            # trend-following adds, not value-anchored entries) but mark them,
            # since rr has no meaningful value-anchored reward to report.
            note = (
                f"Above model: the trigger level is above the model's bull target "
                f"({target:.2f} USD); value-anchored R:R is undefined -- this is a "
                "trend-following entry only."
            )

        tranches.append(
            {
                "n": idx + 1,
                "trigger": trigger,
                "price_zone": {"lo": lo, "hi": hi},
                "size_pct": size_pcts[idx],
                "invalidation": invalidation,
                "target": target,
                "rr": rr,
                "note": note,
                "kind": kind,
                "trigger_reason": _TRIGGER_REASON_BY_SOURCE.get(source, ""),
                "invalidation_reason": invalidation_reason,
                "target_reason": target_reason,
                "size_reason": _SIZE_REASON_TR,
            }
        )

    # Defensive monotonicity check (METODOLOJI.md Sec.1 item 5): as price
    # decreases from tranche to tranche, dip-side R:R should never decrease.
    # Given a fixed target/invalidation this is guaranteed by construction
    # among dip tranches (see the docstring), but flag rather than silently
    # reorder if it somehow doesn't hold. Breakout tranches each have their
    # own invalidation, so no such guarantee -- and no such check -- applies
    # to them or to a dip/breakout pair.
    for idx in range(len(tranches) - 1):
        cur, nxt = tranches[idx], tranches[idx + 1]
        if cur["kind"] != "dip" or nxt["kind"] != "dip":
            continue
        rr_cur, rr_next = cur["rr"], nxt["rr"]
        if rr_cur is not None and rr_next is not None and rr_next < rr_cur:
            nxt["note"] = (
                "R:R order is reversed: this tranche offers a lower R:R than the previous "
                "(higher-priced) tranche; the plan should be mechanically reviewed."
            )

    return tranches


def compute_stop_adding(
    valuation: Optional[dict],
    technical: Optional[dict],
    red_flags: Optional[list],
    entry_plan: Optional[list],
    catalyst: Optional[dict],
) -> list:
    """Determine mechanical "do not open a new tranche" signals (METODOLOJI.md
    Sec.1 item 6, "Stop-adding sinyalleri").

    Concentration-limit signals are out of scope (no ``PROFIL.md``
    portfolio-position schema exists yet) -- only the mechanical,
    filing/price-derived signals below are checked.

    Args:
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation`, or
            ``None``.
        technical: The merged indicators + verdict dict from
            :mod:`sec_analyzer.technical`, or ``None``. The current price
            used for the price-based signals below comes from
            ``technical["price"]`` -- this function has no separate price
            argument, so if the current price is only known via ``metrics``
            (not ``technical``), the price-dependent signals are simply
            skipped rather than fabricating a price from elsewhere.
        red_flags: The list of ``{"code", "message", "detail"}`` dicts from
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``/``[]``.
        entry_plan: The list returned by :func:`compute_entry_plan` (used
            for its structural invalidation floor -- the minimum
            ``invalidation`` among dip-kind tranches; a tranche with no
            ``"kind"`` key is treated as dip for backward compatibility.
            If the plan has no dip-kind tranches at all -- e.g. a
            breakout-only plan, whose per-tranche failed-breakout stops
            sit just below price by construction and are not a structural
            floor -- the NEAR_INVALIDATION signal is skipped entirely
            rather than falling back to a non-dip invalidation), or
            ``None``/``[]``.
        catalyst: The ``{"estimate_date", "label", "based_on"}`` dict from
            :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
            ``None``.

    Returns:
        A list of ``{"code": str, "message": str}`` dicts, one per
        triggered signal, in this fixed check order:
        ``"BELOW_BEAR_FLOOR"``, ``"NEAR_INVALIDATION"``,
        ``"HIGH_UNCERTAINTY"``, ``"ACTIVE_RED_FLAG"`` (one summarized entry
        for all active red flags), ``"BINARY_CATALYST_NEAR"``. ``[]`` if
        none apply. Never raises.
    """
    try:
        return _compute_stop_adding(valuation or {}, technical, red_flags, entry_plan or [], catalyst)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("compute_stop_adding() failed unexpectedly; returning an empty signal list.")
        return []


#: A catalyst this many days out (or nearer) counts as "upcoming" for the
#: stop-adding signal. Matches the HTML report's own earnings-proximity badge
#: window, so the two surfaces agree on what "near" means. SPEC.md Sec.21b.
_CATALYST_NEAR_DAYS = 21


def _catalyst_is_near(catalyst: dict) -> bool:
    """Whether a catalyst is close enough to warrant the stop-adding signal.

    A quarter published a day ago is not an upcoming catalyst, and neither is
    one a full quarter away -- the signal used to fire on the mere PRESENCE
    of a label. Uses ``days_until``/``recently_reported`` as computed by
    ``fetch.filings.estimate_next_earnings`` against its own reference date,
    so this stays deterministic in as-of mode (no wall-clock read here).

    A catalyst dict without ``days_until`` (older or hand-built) keeps the
    previous unconditional behavior rather than silently losing the signal.
    """
    if catalyst.get("recently_reported"):
        return False
    days_until = catalyst.get("days_until")
    if not isinstance(days_until, int):
        return True
    return 0 <= days_until <= _CATALYST_NEAR_DAYS


def _compute_stop_adding(
    valuation: dict,
    technical: Optional[dict],
    red_flags: Optional[list],
    entry_plan: list,
    catalyst: Optional[dict],
) -> list:
    signals: List[dict] = []
    price = (technical or {}).get("price")

    bear_lo = ((valuation.get("fair_value_range") or {}).get("bear") or {}).get("lo")
    if price is not None and bear_lo is not None and price < bear_lo:
        signals.append(
            {
                "code": "BELOW_BEAR_FLOOR",
                "message": (
                    f"Current price ({price:.2f} USD) is below the pessimistic (bear) scenario "
                    f"floor ({bear_lo:.2f} USD); the fundamental floor may have broken down."
                ),
            }
        )

    if entry_plan:
        # Backward-compat: a tranche dict with no "kind" key at all (older
        # callers) is treated as dip, preserving pre-two-directional
        # behavior. A breakout-only plan's per-tranche failed-breakout
        # stops sit just below the current price by construction, so
        # falling back to "min across all tranches" when there are no dip
        # tranches would fire this signal spuriously -- skip it instead.
        dip_invalidations = [
            t.get("invalidation")
            for t in entry_plan
            if t.get("kind", "dip") == "dip" and t.get("invalidation") is not None
        ]
        invalidation = min(dip_invalidations) if dip_invalidations else None
        if price is not None and invalidation is not None:
            threshold = invalidation * (1 + _NEAR_INVALIDATION_BUFFER_PCT)
            if price <= threshold:
                signals.append(
                    {
                        "code": "NEAR_INVALIDATION",
                        "message": (
                            f"Price ({price:.2f} USD) is close to the entry plan's invalidation "
                            f"level ({invalidation:.2f} USD); opening a new tranche is risky."
                        ),
                    }
                )

    if (valuation.get("sensitivity") or {}).get("high_uncertainty"):
        signals.append(
            {
                "code": "HIGH_UNCERTAINTY",
                "message": (
                    "The sensitivity matrix shows high uncertainty (band width is more than 60% "
                    "of the base cell); caution is warranted when adding to the position."
                ),
            }
        )

    flag_messages = [f.get("message") for f in (red_flags or []) if f.get("message")]
    if flag_messages:
        signals.append(
            {
                "code": "ACTIVE_RED_FLAG",
                "message": "Caution is warranted due to active red flag(s): " + "; ".join(flag_messages),
            }
        )

    if catalyst and catalyst.get("label") and _catalyst_is_near(catalyst):
        signals.append(
            {
                "code": "BINARY_CATALYST_NEAR",
                "message": (
                    f"Upcoming binary catalyst: {catalyst['label']}; adding to the position "
                    "without a trigger before the catalyst date is not recommended."
                ),
            }
        )

    return signals


def _degraded_thesis_metric() -> dict:
    return {
        "name": _DEFAULT_METRIC_CANDIDATES[0][1],
        "latest_value": None,
        "trend": None,
        "rationale": "The thesis-validation metric could not be determined due to an internal error.",
        "cycle": None,
        "quarterly_check": None,
    }


def _compute_cycle_position(
    fy_map: Dict[int, float], current_fy: int, is_cyclical: bool
) -> Optional[dict]:
    """Locate the anchor metric's latest value inside its own multi-year
    trough->peak range, so the report can visualize where the business sits
    in its cycle (METODOLOJI.md §7; mirrors the ``CYCLICAL_TRAP`` red flag's
    "latest margin vs historical peak" idea in
    :mod:`sec_analyzer.normalize.red_flags`).

    Args:
        fy_map: ``{fiscal_year: value}`` for the chosen anchor metric, with
            values in the metric's own units (decimal fractions for the
            margin/ROE/growth candidates). Built by
            :func:`_select_thesis_metric` from the ``ratios`` series.
        current_fy: The latest fiscal year in ``fy_map`` (the "you are here"
            point).
        is_cyclical: Whether ``sector_type == "cyclical"`` -- only affects
            the display terminology the template chooses ("cycle" vs
            "historical range"), never the numbers.

    Returns:
        ``{"low", "high", "current", "position", "low_fy", "high_fy",
        "current_fy", "n_years", "is_cyclical", "series"}`` -- or ``None``
        when a position can't be placed (fewer than two fiscal years, or a
        perfectly flat series where trough == peak, which would make the
        0..1 position undefined). ``position`` is ``(current - low) /
        (high - low)`` clamped to ``[0, 1]``. ``series`` is the full annual
        series ``[{"fy": int, "value": float}, ...]`` sorted ascending by
        fiscal year, so the report can draw a level sparkline of the
        metric's trajectory alongside the positional bar.
    """
    if len(fy_map) < 2:
        return None

    low_fy = min(fy_map, key=lambda fy: fy_map[fy])
    high_fy = max(fy_map, key=lambda fy: fy_map[fy])
    low = fy_map[low_fy]
    high = fy_map[high_fy]
    if high <= low:  # flat series -> no meaningful trough/peak spread
        return None

    current = fy_map[current_fy]
    position = _clamp((current - low) / (high - low), 0.0, 1.0)
    return {
        "low": low,
        "high": high,
        "current": current,
        "position": round(position, 3),
        "low_fy": low_fy,
        "high_fy": high_fy,
        "current_fy": current_fy,
        "n_years": len(fy_map),
        "is_cyclical": is_cyclical,
        "series": [{"fy": fy, "value": fy_map[fy]} for fy in sorted(fy_map)],
    }


def _quarterly_series_for_metric_key(metric_key: str, normalized: dict) -> List[dict]:
    """Build the quarterly ``[{"period_end": str, "value": float}, ...]``
    series (percent units, ascending) for one of :data:`_SECTOR_METRIC_
    CANDIDATES`'s anchor-metric keys.

    Reuses the exact quarterly machinery already built for the momentum
    layer rather than re-deriving it: :func:`sec_analyzer.normalize.
    normalizer.quarterly_ratio_series` for the plain numerator/denominator
    ratios (net/gross margin, ROE), momentum.py's own
    :func:`sec_analyzer.signals.momentum._fcf_margin_series` for FCF margin
    (OCF-CapEx isn't a single-concept ratio, so it doesn't fit the
    numerator/denominator shape), and momentum.py's :func:`sec_analyzer.
    signals.momentum._yoy_growth_series` (re-keyed from its own ``yoy_pct``
    to ``value`` for a uniform shape here) for the growth-story anchor.

    Returns ``[]`` for an unrecognized key or missing inputs -- every
    concept lookup underneath already degrades to ``[]`` on its own, so this
    never raises.
    """
    if metric_key == "net_margin":
        return quarterly_ratio_series(normalized, "NetIncome", "Revenue")
    if metric_key == "gross_margin":
        return quarterly_ratio_series(normalized, "GrossProfit", "Revenue")
    if metric_key == "roe":
        return quarterly_ratio_series(normalized, "NetIncome", "StockholdersEquity")
    if metric_key == "fcf_margin":
        return _fcf_margin_series(normalized)
    if metric_key == "yoy_revenue_growth":
        quarters = to_quarterly_series(normalized, "Revenue")
        return [
            {"period_end": q["period_end"], "value": q["yoy_pct"]}
            for q in _yoy_growth_series(quarters)
        ]
    return []


def _establish_or_load_anchor(
    cik, metric_key: str, trend: Optional[str], established_fy: Optional[int]
) -> Optional[dict]:
    """Load the day-1 thesis anchor for ``(cik, metric_key)``, establishing
    (or re-establishing) it first when needed.

    The anchor's ``direction`` is (re)written only when no anchor row exists
    yet for this ``cik``, or the stored anchor's ``metric_key`` no longer
    matches ``metric_key`` (e.g. a sector reclassification swapped the
    anchor metric, per this function's design question 1). In either case
    the new direction is whatever THIS run's annual ``trend`` is; if that is
    ``None``/``"flat"`` (no prior fiscal year yet, or no clear direction),
    establishment is deferred -- there is nothing yet to anchor on -- and
    this returns ``None`` until a later run has a real
    improving/deteriorating trend to seed it with.

    Once established for a given ``(cik, metric_key)`` pair, the anchor's
    ``direction`` never changes on subsequent runs even if the metric's own
    annual trend later reverses -- that reversal is exactly what the
    quarterly check is meant to catch, so overwriting the anchor on every
    run would make the check tautological. Never raises: both underlying
    calls (:func:`sec_analyzer.store.thesis_anchors.get_anchor`/
    :func:`~sec_analyzer.store.thesis_anchors.set_anchor`) are themselves
    never-raise and degrade to ``None`` on any DB failure.
    """
    anchor = thesis_anchors.get_anchor(cik)
    if anchor is not None and anchor.get("metric_key") == metric_key:
        return anchor
    if trend not in ("improving", "deteriorating"):
        return None
    return thesis_anchors.set_anchor(
        cik=cik, metric_key=metric_key, direction=trend, established_fy=established_fy
    )


def _classify_quarterly_series(
    metric_key: str,
    series: List[dict],
    anchor_direction: str,
    anchor_established_fy: Optional[int],
) -> Optional[dict]:
    """Classify each quarter in the most recent
    :data:`_QUARTERLY_CHECK_WINDOW` of ``series`` as with/against/neutral
    relative to ``anchor_direction``, and count the trailing consecutive
    against-thesis streak (METODOLOJI.md Sec.7).

    Each quarter is compared only to the PRIOR quarter in the window (a
    quarter-over-quarter delta -- not a trend-window mean like
    :func:`_classify_trend`/:func:`sec_analyzer.signals.momentum.
    _classify_trend`) against a per-metric deadband (see
    :data:`_QUARTERLY_CHECK_DEADBAND_PP`): a delta inside the deadband is
    neutral (``against_thesis=None``) and neither extends nor breaks the
    streak, matching momentum.py's own noise-tolerance philosophy. The
    earliest quarter in the window has no prior quarter to compare against
    (even when an earlier quarter exists outside the window) and is always
    neutral for that reason. The trailing streak walks backward from the
    most recent quarter, counting consecutive against-thesis quarters and
    skipping neutral ones, stopping at the first with-thesis quarter (or the
    start of the window).

    Returns ``None`` when fewer than :data:`_QUARTERLY_CHECK_MIN_QUARTERS`
    usable quarters are available -- too few to populate even a 2-quarter
    streak. Never raises on well-formed input; the caller
    (:func:`_compute_quarterly_check`) is wrapped so any unexpected error
    here degrades to ``quarterly_check=None`` rather than blanking the rest
    of the thesis-metric card.
    """
    window = [q for q in series if q.get("period_end") is not None and q.get("value") is not None]
    window = window[-_QUARTERLY_CHECK_WINDOW:]
    if len(window) < _QUARTERLY_CHECK_MIN_QUARTERS:
        return None

    deadband = _QUARTERLY_CHECK_DEADBAND_PP.get(metric_key, _MARGIN_TREND_DEADBAND_PP)
    thesis_is_up = anchor_direction == "improving"

    quarters: List[dict] = []
    prev_value: Optional[float] = None
    for point in window:
        value = float(point["value"])
        against: Optional[bool] = None
        if prev_value is not None:
            diff = value - prev_value
            if abs(diff) >= deadband:
                against = (diff > 0) != thesis_is_up
        quarters.append({"period_end": point["period_end"], "value": value, "against_thesis": against})
        prev_value = value

    consecutive_against = 0
    for q in reversed(quarters):
        flag = q["against_thesis"]
        if flag is True:
            consecutive_against += 1
        elif flag is None:
            continue
        else:  # False: a with-thesis quarter breaks the against-thesis streak
            break

    return {
        "anchor_direction": anchor_direction,
        "anchor_established_fy": anchor_established_fy,
        "quarters": quarters,
        "consecutive_against": consecutive_against,
        "invalidated": consecutive_against >= 2,
    }


def _compute_quarterly_check(
    normalized: dict, metric_key: str, trend: Optional[str], established_fy: Optional[int]
) -> Optional[dict]:
    """Establish/load the ``(cik, metric_key)`` anchor and classify the
    metric's recent quarterly series against it (METODOLOJI.md Sec.7).

    Returns ``None`` when ``normalized`` carries no ``cik`` (nothing to key
    the anchor on), when no anchor could be established/loaded yet (see
    :func:`_establish_or_load_anchor` -- notably, a brand-new ``(cik,
    metric_key)`` pair whose current annual ``trend`` is ``None``/``"flat"``
    has no direction to anchor on yet), or when the metric's quarterly
    series is missing/too short (see :func:`_classify_quarterly_series`).
    """
    cik = normalized.get("cik") if isinstance(normalized, dict) else None
    if cik is None:
        return None

    anchor = _establish_or_load_anchor(cik, metric_key, trend, established_fy)
    if anchor is None:
        return None

    series = _quarterly_series_for_metric_key(metric_key, normalized)
    if not series:
        return None

    return _classify_quarterly_series(
        metric_key, series, anchor.get("direction"), anchor.get("established_fy")
    )


def select_thesis_metric(
    sector_type: Optional[str],
    ratios: Optional[list],
    metrics: Optional[dict],
    normalized: Optional[dict] = None,
) -> dict:
    """Select the single anchor metric that validates (or invalidates) the
    investment thesis (METODOLOJI.md Sec.1 item 7, "Thesis-validation metric").

    Args:
        sector_type: One of ``valuation.sector.classify_sector``'s buckets
            (``"mature"``, ``"growth_unprofitable"``, ``"financial"``,
            ``"reit"``, ``"cyclical"``), or ``None``/unrecognized (falls
            back to net margin).
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`, or
            ``None``. Supplies the per-fiscal-year values used for
            ``latest_value``/``trend``.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None``. Used only as a single-point fallback (no derivable
            trend) when ``ratios`` has no usable series for the chosen
            metric -- currently only wired for the
            ``growth_unprofitable`` metric (revenue CAGR).
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`, or
            ``None``. Optional and defaulted so existing call sites that
            can't supply it keep working (they simply get
            ``quarterly_check=None``, same as a missing ``cycle``). Supplies
            the raw facts (and ``cik``, for the persisted anchor) needed to
            build the anchor metric's quarterly series for
            ``quarterly_check``.

    Returns:
        ``{"name": str, "latest_value": str|None, "trend": str|None,
        "rationale": str, "cycle": dict|None, "quarterly_check": dict|None}``.
        ``latest_value`` is a formatted percent string (e.g. ``"23.4%"``)
        read from the latest available fiscal year, never fabricated --
        ``None`` if the chosen metric isn't computable from the given
        inputs, in which case ``rationale`` says so explicitly. ``trend`` is
        ``"improving"``/``"deteriorating"``/``"flat"``, or ``None`` if no
        prior fiscal year's value is available to compare against.
        ``cycle`` locates the latest value inside the metric's own
        multi-year trough->peak range (see :func:`_compute_cycle_position`
        for its shape), or ``None`` when fewer than two fiscal years exist
        or the series is perfectly flat -- and always ``None`` for the
        single-point ``metrics`` fallback, which has no series.

        ``quarterly_check`` (METODOLOJI.md Sec.7's real, code-checked
        quarterly-invalidation rule -- see :func:`_compute_quarterly_check`)
        is ``{"anchor_direction": "improving"|"deteriorating",
        "anchor_established_fy": int|None, "quarters": [{"period_end": str,
        "value": float, "against_thesis": bool|None}, ...],
        "consecutive_against": int, "invalidated": bool}``, or ``None``
        when: ``normalized`` wasn't supplied; the chosen metric came from
        the single-point ``metrics`` fallback (no series to check, same
        reason ``cycle`` is ``None`` there); ``normalized`` carries no
        ``cik``; no day-1 anchor could be established yet (a brand-new
        ``(cik, metric)`` pair whose current annual trend is
        ``None``/``"flat"`` has no direction to anchor on); or fewer than
        ~3 usable quarters exist for the metric. When ``invalidated`` is
        ``True``, ``rationale`` states this explicitly (in addition to the
        general Sec.7 rule sentence always appended -- see
        :data:`_THESIS_INVALIDATION_TRIGGERED_TR`).

        Never raises.
    """
    try:
        return _select_thesis_metric(sector_type, ratios or [], metrics or {}, normalized)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("select_thesis_metric() failed unexpectedly; returning a degraded result.")
        return _degraded_thesis_metric()


def _select_thesis_metric(
    sector_type: Optional[str], ratios: list, metrics: dict, normalized: Optional[dict]
) -> dict:
    candidates = _SECTOR_METRIC_CANDIDATES.get(sector_type, _DEFAULT_METRIC_CANDIDATES)

    chosen_name = candidates[0][1]
    chosen_key: Optional[str] = None
    latest_value: Optional[str] = None
    trend: Optional[str] = None
    cycle: Optional[dict] = None
    established_candidate_fy: Optional[int] = None

    for key, name in candidates:
        fy_map = {
            r["fy"]: r.get(key)
            for r in ratios
            if r.get("fy") is not None and r.get(key) is not None
        }
        if fy_map:
            latest_fy = max(fy_map)
            value = fy_map[latest_fy]
            chosen_name = name
            chosen_key = key
            latest_value = _format_pct(value)
            established_candidate_fy = latest_fy

            earlier_fys = [fy for fy in fy_map if fy < latest_fy]
            if earlier_fys:
                prior_fy = max(earlier_fys)
                trend = _classify_trend(value - fy_map[prior_fy])

            # Locate the latest value inside the metric's own trough->peak
            # range so the report can show where the business sits in its
            # cycle. Only derivable from a real multi-year series, so it is
            # skipped for the single-point metrics fallback below.
            cycle = _compute_cycle_position(fy_map, latest_fy, sector_type == "cyclical")
            break

        fallback_keys = _METRICS_FALLBACK_FOR_RATIO_KEY.get(key)
        if fallback_keys:
            fallback_value = next(
                (metrics.get(fb_key) for fb_key in fallback_keys if metrics.get(fb_key) is not None),
                None,
            )
            if fallback_value is not None:
                chosen_name = name
                latest_value = _format_pct(fallback_value)
                trend = None  # a single-point metrics figure has no derivable year-over-year trend.
                # chosen_key intentionally left None: the single-point metrics
                # fallback has no quarterly series to check against
                # (METODOLOJI.md Sec.7's quarterly check needs a real
                # multi-quarter series -- same reason `cycle` stays None here).
                break

    # METODOLOJI.md Sec.7's quarterly-invalidation check: only attempted when
    # a real annual series backed the choice above (chosen_key is set) and
    # the caller supplied normalized facts. Wrapped locally so a bug here
    # degrades only this one field, not the rest of the thesis-metric card.
    quarterly_check: Optional[dict] = None
    if chosen_key is not None and normalized is not None:
        try:
            quarterly_check = _compute_quarterly_check(normalized, chosen_key, trend, established_candidate_fy)
        except Exception:  # noqa: BLE001 - a bug here must not blank the rest of the thesis card
            logger.exception(
                "Quarterly-invalidation check failed for metric '%s'; omitting quarterly_check.",
                chosen_key,
            )
            quarterly_check = None

    rationale = f"{_RATIONALE_BY_SECTOR.get(sector_type, _RATIONALE_BY_SECTOR[None])} {_THESIS_INVALIDATION_RULE_TR}"
    if latest_value is None:
        rationale += " This metric could not be computed from the available data."
    if quarterly_check and quarterly_check.get("invalidated"):
        rationale += f" {_THESIS_INVALIDATION_TRIGGERED_TR}"

    return {
        "name": chosen_name,
        "latest_value": latest_value,
        "trend": trend,
        "rationale": rationale,
        "cycle": cycle,
        "quarterly_check": quarterly_check,
    }
