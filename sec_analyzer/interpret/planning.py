"""Deterministic phase-2 post-processing: scenario returns, entry plan,
stop-adding signals, and thesis-anchor metric.

This module implements the mechanical structures required by
``sec_analyzer/METODOLOJI.md`` sections 4-7 (senaryo tablosu, kademeli giriş
planı, stop-adding sinyalleri, tez doğrulama metriği). Every function here is
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
"""

import logging
from typing import Dict, List, Optional

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

#: Absolute change (in the metric's own units, e.g. 0.01 = 1 percentage
#: point for a margin/growth-rate metric) below which a thesis metric's
#: year-over-year move is read as "yatay" (flat) rather than
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
    "METODOLOJI §7 kuralı: bu metrik iki ardışık çeyrek boyunca tezin aksini "
    "gösterirse tez geçersiz sayılır ve bu açıkça belirtilir."
)

#: Sector-type -> ordered list of (ratio-row key, Turkish display name)
#: candidates for the thesis anchor metric. The first candidate with a
#: computable latest-fiscal-year value in ``ratios`` wins; if none do, the
#: first candidate's name is still reported with ``latest_value=None``.
_SECTOR_METRIC_CANDIDATES: Dict[Optional[str], List["tuple[str, str]"]] = {
    "mature": [("net_margin", "Net Kâr Marjı"), ("roe", "Özkaynak Getirisi (ROE)")],
    "growth_unprofitable": [("yoy_revenue_growth", "Yıllık Gelir Büyümesi (YoY)")],
    "financial": [("roe", "Özkaynak Getirisi (ROE, NIM proxy)")],
    "reit": [("fcf_margin", "FCF Marjı (FFO proxy)")],
    "cyclical": [("gross_margin", "Brüt Kâr Marjı"), ("net_margin", "Net Kâr Marjı")],
}

#: Default candidate list used for ``None``/unrecognized ``sector_type``.
_DEFAULT_METRIC_CANDIDATES: List["tuple[str, str]"] = [("net_margin", "Net Kâr Marjı")]

#: For a ratio-row key that has no meaningful multi-year series in
#: ``ratios`` on its own (only ``yoy_revenue_growth`` today), an ordered
#: list of ``metrics`` dict keys to try as a single-point fallback value
#: (no year-over-year trend is derivable from a single ``metrics`` figure).
_METRICS_FALLBACK_FOR_RATIO_KEY: Dict[str, "tuple[str, ...]"] = {
    "yoy_revenue_growth": ("revenue_cagr_5y", "revenue_cagr_3y"),
}

#: One Turkish sentence per sector_type explaining why the chosen metric
#: anchors the thesis. Combined with :data:`_THESIS_INVALIDATION_RULE_TR`.
_RATIONALE_BY_SECTOR: Dict[Optional[str], str] = {
    "mature": (
        "Olgun sektörlerde tezin sağlığı kâr marjının istikrarında görülür; bu yüzden net kâr marjı "
        "(hesaplanamıyorsa ROE) tek çapa metrik olarak izlenir."
    ),
    "growth_unprofitable": (
        "Henüz kâr etmeyen büyüme hikayelerinde tez, gelir büyümesinin yeniden hızlanmasına "
        "(re-acceleration) dayanır; bu yüzden yıllık gelir büyümesi tek çapa metrik olarak izlenir."
    ),
    "financial": (
        "Finansal kuruluşlarda özkaynak getirisi (ROE), net faiz marjının (NIM) dolaylı bir göstergesi "
        "olduğu için tek çapa metrik olarak izlenir."
    ),
    "reit": (
        "GYO'larda tez, FFO'ya (fonlardan operasyon) en yakın hesaplanabilir gösterge olan FCF marjının "
        "seyrine dayanır."
    ),
    "cyclical": (
        "Döngüsel şirketlerde tez, marjın orta-döngü (mid-cycle) seviyesine göre seyrine dayanır; bu "
        "yüzden brüt (hesaplanamıyorsa net) kâr marjı tek çapa metrik olarak izlenir."
    ),
    None: "Sektör sınıflandırması belirsiz olduğundan varsayılan olarak net kâr marjı tek çapa metrik olarak izlenir.",
}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def _format_pct(value: float) -> str:
    """Render a decimal-fraction ratio/growth rate as a Turkish percent
    string with one decimal, e.g. ``0.234 -> "%23.4"``."""
    return f"%{value * 100:.1f}"


def _classify_trend(diff: float) -> str:
    """Classify a year-over-year change in a "higher is better" metric
    (margin, ROE, or growth rate) into a Turkish trend label."""
    if abs(diff) < _TREND_FLAT_THRESHOLD:
        return "yatay"
    return "iyileşiyor" if diff > 0 else "bozuluyor"


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

    Every candidate is tagged with a short Turkish ``source`` label (used
    verbatim in the breakout trigger sentence). Missing/non-numeric values
    are simply omitted from both lists.

    Returns:
        ``(dip_candidates, breakout_candidates)``, each a list of
        ``{"source": str, "level": float, "kind": "dip"|"breakout"}``.
    """
    fvr = valuation.get("fair_value_range") or {}
    bear, base, bull = fvr.get("bear") or {}, fvr.get("base") or {}, fvr.get("bull") or {}
    technical = technical or {}

    dip_raw = [
        ("bear_lo", bear.get("lo")),
        ("base_lo", base.get("lo")),
        ("base_hi", base.get("hi")),
        ("bull_hi", bull.get("hi")),
        ("low_52w", technical.get("low_52w")),
        ("sma50", technical.get("sma50")),
        ("sma200", technical.get("sma200")),
    ]
    dip = [
        {"source": label, "level": float(value), "kind": "dip"}
        for label, value in dip_raw
        if isinstance(value, (int, float)) and value <= price
    ]

    breakout_raw = [
        ("SMA50 geri alımı", technical.get("sma50")),
        ("SMA200 geri alımı", technical.get("sma200")),
    ]
    resistance_zones = technical.get("resistance_levels") or []
    for zone in resistance_zones:
        breakout_raw.append(("direnç/önceki zirve kırılımı", (zone or {}).get("price")))

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
        breakout_raw.append(("52 hafta zirve kırılımı", technical.get("high_52w")))

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


def _resolve_target(valuation: dict) -> Optional[float]:
    """Upside anchor for entry-plan targets: prefer bull.hi, else base.hi."""
    fvr = valuation.get("fair_value_range") or {}
    bull_hi = (fvr.get("bull") or {}).get("hi")
    if bull_hi is not None:
        return bull_hi
    return (fvr.get("base") or {}).get("hi")


def _resolve_invalidation(valuation: dict, technical: Optional[dict], lowest_kept_level: float) -> float:
    """Shared structural invalidation level for dip-kind tranches: a buffer
    below the lowest of bear.lo/low_52w/``lowest_kept_level`` (the lowest
    kept *dip* tranche's level). ``lowest_kept_level`` is always included in
    the floor -- not just used as a last resort -- so this always sits
    strictly below every dip tranche's price zone by construction, even
    when a dip level (e.g. an sma200/base_lo level) falls below
    bear.lo/low_52w. Breakout tranches use their own per-tranche failed-
    breakout invalidation instead (see :func:`_compute_entry_plan`)."""
    fvr = valuation.get("fair_value_range") or {}
    bear_lo = (fvr.get("bear") or {}).get("lo")
    low_52w = (technical or {}).get("low_52w")
    sources = [v for v in (bear_lo, low_52w) if v is not None]
    base_level = min([*sources, lowest_kept_level])
    return round(base_level * (1 - _INVALIDATION_BUFFER_PCT), 2)


#: Stabilization precondition appended to dip tranches when momentum flags a
#: falling knife (cheap fundamentals + negative price momentum). It gates the
#: TIMING of a dip entry, not its price level -- the tranche's trigger/size are
#: unchanged; the reader is told to wait for a momentum turn before acting.
_STABILIZATION_NOTE = (
    "Stabilizasyon koşulu: momentum negatif (düşen bıçak) — dip tetiği geldiğinde "
    "hemen alma; RSI'nin 30 üstünü geri alması veya MACD yukarı kesişim teyidi beklenmeli."
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
    Sec.1 item 5, "Kademeli giriş planı") -- two directional tranche kinds,
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
              "trigger": str,                     # Turkish, daily-close-only condition
              "price_zone": {"lo": float, "hi": float},
              "size_pct": float,                   # % of intended full position
              "invalidation": float,               # daily-close level; thesis void below it
              "target": float|None,                # upside anchor; None if neither
                                                    # base.hi nor bull.hi is available
              "rr": float|None,                    # reward:risk, 1dp; None if no
                                                    # positive reward or risk
              "note": str|None,
              "kind": str,                          # "dip" or "breakout"
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
    target = _resolve_target(valuation)

    dip_levels = [c["level"] for c in ordered if c["kind"] == "dip"]
    dip_invalidation = _resolve_invalidation(valuation, technical, min(dip_levels)) if dip_levels else None

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
            trigger = (
                f"Günlük kapanış {level:.2f} USD seviyesinin altına inerse (bölge "
                f"{lo:.2f}-{hi:.2f} USD); gün içi dokunuş tetik saymaz."
            )
        else:
            invalidation = round(level * (1 - _INVALIDATION_BUFFER_PCT), 2)
            trigger = (
                f"Günlük kapanış {level:.2f} USD seviyesinin üzerine çıkarsa (yükseliş teyidi "
                f"— {source}); gün içi dokunuş tetik saymaz."
            )

        rr = None
        if target is not None and invalidation is not None:
            reward = target * (1 - _ROUND_TRIP_COST_PCT) - entry * (1 + _ROUND_TRIP_COST_PCT)
            risk = entry * (1 + _ROUND_TRIP_COST_PCT) - invalidation
            if risk > 0 and reward > 0:
                rr = round(reward / risk, 1)

        note = None
        if kind == "breakout" and target is not None and entry >= target:
            # Product decision: keep above-target breakout tranches (they're
            # trend-following adds, not value-anchored entries) but mark them,
            # since rr has no meaningful value-anchored reward to report.
            note = (
                f"Model üstü: tetik seviyesi model bull hedefinin ({target:.2f} USD) üzerinde; "
                "değer-çapalı R:R tanımsız -- yalnızca trend-takip girişi."
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
                "R:R sırası ters: bu tranche, bir önceki (daha yüksek fiyatlı) tranche'dan "
                "daha düşük R:R sunuyor; plan mekanik olarak yeniden gözden geçirilmeli."
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
        A list of ``{"code": str, "message": str}`` dicts (Turkish
        messages), one per triggered signal, in this fixed check order:
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
                    f"Güncel fiyat ({price:.2f} USD), kötümser (bear) senaryo tabanının "
                    f"({bear_lo:.2f} USD) altında; fundamental taban kırılmış olabilir."
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
                            f"Fiyat ({price:.2f} USD), giriş planının invalidation seviyesine "
                            f"({invalidation:.2f} USD) yakın; yeni tranche açmak riskli."
                        ),
                    }
                )

    if (valuation.get("sensitivity") or {}).get("high_uncertainty"):
        signals.append(
            {
                "code": "HIGH_UNCERTAINTY",
                "message": (
                    "Duyarlılık matrisi yüksek belirsizlik gösteriyor (bant genişliği baz hücrenin "
                    "%60'ından fazla); pozisyon büyütmede temkinli olunmalı."
                ),
            }
        )

    flag_messages = [f.get("message") for f in (red_flags or []) if f.get("message")]
    if flag_messages:
        signals.append(
            {
                "code": "ACTIVE_RED_FLAG",
                "message": "Aktif red flag(lar) nedeniyle temkinli olunmalı: " + "; ".join(flag_messages),
            }
        )

    if catalyst and catalyst.get("label"):
        signals.append(
            {
                "code": "BINARY_CATALYST_NEAR",
                "message": (
                    f"Yaklaşan binary katalizör: {catalyst['label']}; katalizör tarihinden önce "
                    "tetiksiz pozisyon büyütülmesi önerilmez."
                ),
            }
        )

    return signals


def _degraded_thesis_metric() -> dict:
    return {
        "name": _DEFAULT_METRIC_CANDIDATES[0][1],
        "latest_value": None,
        "trend": None,
        "rationale": "Bir iç hata nedeniyle tez doğrulama metriği belirlenemedi.",
        "cycle": None,
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
            the display terminology the template chooses ("döngü" vs
            "geçmiş aralık"), never the numbers.

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


def select_thesis_metric(sector_type: Optional[str], ratios: Optional[list], metrics: Optional[dict]) -> dict:
    """Select the single anchor metric that validates (or invalidates) the
    investment thesis (METODOLOJI.md Sec.1 item 7, "Tez doğrulama metriği").

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

    Returns:
        ``{"name": str, "latest_value": str|None, "trend": str|None,
        "rationale": str, "cycle": dict|None}``. ``latest_value`` is a
        formatted Turkish percent string (e.g. ``"%23.4"``) read from the
        latest available fiscal year, never fabricated -- ``None`` if the
        chosen metric isn't computable from the given inputs, in which case
        ``rationale`` says so explicitly. ``trend`` is
        ``"iyileşiyor"``/``"bozuluyor"``/``"yatay"``, or ``None`` if no
        prior fiscal year's value is available to compare against.
        ``cycle`` locates the latest value inside the metric's own
        multi-year trough->peak range (see :func:`_compute_cycle_position`
        for its shape), or ``None`` when fewer than two fiscal years exist
        or the series is perfectly flat -- and always ``None`` for the
        single-point ``metrics`` fallback, which has no series. Never
        raises.
    """
    try:
        return _select_thesis_metric(sector_type, ratios or [], metrics or {})
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("select_thesis_metric() failed unexpectedly; returning a degraded result.")
        return _degraded_thesis_metric()


def _select_thesis_metric(sector_type: Optional[str], ratios: list, metrics: dict) -> dict:
    candidates = _SECTOR_METRIC_CANDIDATES.get(sector_type, _DEFAULT_METRIC_CANDIDATES)

    chosen_name = candidates[0][1]
    latest_value: Optional[str] = None
    trend: Optional[str] = None
    cycle: Optional[dict] = None

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
            latest_value = _format_pct(value)

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
                break

    rationale = f"{_RATIONALE_BY_SECTOR.get(sector_type, _RATIONALE_BY_SECTOR[None])} {_THESIS_INVALIDATION_RULE_TR}"
    if latest_value is None:
        rationale += " Mevcut veriyle bu metrik hesaplanamadı."

    return {
        "name": chosen_name,
        "latest_value": latest_value,
        "trend": trend,
        "rationale": rationale,
        "cycle": cycle,
    }
