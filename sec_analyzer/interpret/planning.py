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
#: are treated as the same level (only the higher one is kept).
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


def _collect_entry_candidates(valuation: dict, technical: Optional[dict]) -> "List[tuple[str, float]]":
    """Gather every mechanical trigger-level candidate named in the spec:
    the valuation fair-value band's bear.lo/base.lo/base.hi/bull.hi, and
    the technical read's low_52w/sma50/sma200. Missing/non-numeric values
    are simply omitted."""
    fvr = valuation.get("fair_value_range") or {}
    bear, base, bull = fvr.get("bear") or {}, fvr.get("base") or {}, fvr.get("bull") or {}
    technical = technical or {}

    raw = [
        ("bear_lo", bear.get("lo")),
        ("base_lo", base.get("lo")),
        ("base_hi", base.get("hi")),
        ("bull_hi", bull.get("hi")),
        ("low_52w", technical.get("low_52w")),
        ("sma50", technical.get("sma50")),
        ("sma200", technical.get("sma200")),
    ]
    return [(label, value) for label, value in raw if isinstance(value, (int, float))]


def _dedupe_descending(candidates: "List[tuple[str, float]]") -> "List[tuple[str, float]]":
    """Sort candidates by descending level and drop any that land within
    :data:`_DEDUPE_THRESHOLD_PCT` of the previously-kept (higher) level."""
    ordered = sorted(candidates, key=lambda pair: -pair[1])
    kept: "List[tuple[str, float]]" = []
    for label, value in ordered:
        if kept and kept[-1][1] != 0 and abs(value - kept[-1][1]) / abs(kept[-1][1]) < _DEDUPE_THRESHOLD_PCT:
            continue
        kept.append((label, value))
    return kept


def _resolve_target(valuation: dict) -> Optional[float]:
    """Upside anchor for entry-plan targets: prefer bull.hi, else base.hi."""
    fvr = valuation.get("fair_value_range") or {}
    bull_hi = (fvr.get("bull") or {}).get("hi")
    if bull_hi is not None:
        return bull_hi
    return (fvr.get("base") or {}).get("hi")


def _resolve_invalidation(valuation: dict, technical: Optional[dict], lowest_kept_level: float) -> float:
    """Invalidation level: a buffer below the lower of bear.lo/low_52w, or
    (if neither is available) below the lowest kept tranche level itself --
    always strictly below every tranche's price zone by construction."""
    fvr = valuation.get("fair_value_range") or {}
    bear_lo = (fvr.get("bear") or {}).get("lo")
    low_52w = (technical or {}).get("low_52w")
    sources = [v for v in (bear_lo, low_52w) if v is not None]
    base_level = min(sources) if sources else lowest_kept_level
    return round(base_level * (1 - _INVALIDATION_BUFFER_PCT), 2)


def compute_entry_plan(valuation: Optional[dict], technical: Optional[dict], price: Optional[float]) -> list:
    """Build the mechanical, tranche-based scale-in plan (METODOLOJI.md
    Sec.1 item 5, "Kademeli giriş planı").

    Candidate trigger levels are pulled only from already-computed figures
    -- the fair-value band's ``bear.lo``/``base.lo``/``base.hi``/``bull.hi``
    and the technical read's ``low_52w``/``sma50``/``sma200`` -- filtered to
    levels at-or-below the current price, deduplicated when two levels sit
    within :data:`_DEDUPE_THRESHOLD_PCT` of each other, then sorted
    descending and capped at :data:`_MAX_ENTRY_TRANCHES`. A single
    invalidation level and a single upside target apply to every tranche
    (see :func:`_resolve_invalidation`/:func:`_resolve_target`); because
    both are fixed while the entry price decreases from tranche to tranche,
    R:R is mathematically guaranteed to be non-decreasing as price
    decreases (lower entry -> larger reward, smaller risk) -- the explicit
    post-hoc monotonicity check below exists to flag it defensively rather
    than because it's expected to ever fire.

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
        first tranche triggers first, nearest the current price on the way
        down)::

            {
              "n": int,                          # 1-based order
              "trigger": str,                     # Turkish, daily-close-only condition
              "price_zone": {"lo": float, "hi": float},
              "size_pct": float,                   # % of intended full position
              "invalidation": float,               # daily-close level; thesis void below it
              "target": float|None,                # upside anchor; None if neither
                                                    # base.hi nor bull.hi is available
              "rr": float|None,                    # reward:risk, 1dp
              "note": str|None,
            }

        ``[]`` if ``price`` is missing/non-positive, or if neither the
        valuation fair-value band nor the technical levels yield any usable
        candidate at or below the current price. If usable candidates exist
        but fewer than 3 distinct levels survive filtering/deduplication,
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

    candidates = _collect_entry_candidates(valuation, technical)
    if not candidates:
        return []

    below_price = [(label, value) for label, value in candidates if value <= price]
    if not below_price:
        return []

    kept = _dedupe_descending(below_price)[:_MAX_ENTRY_TRANCHES]
    if not kept:
        return []

    n = len(kept)
    target = _resolve_target(valuation)
    invalidation = _resolve_invalidation(valuation, technical, kept[-1][1])

    weights = [(idx + 1) ** _SIZE_WEIGHT_EXPONENT for idx in range(n)]
    weight_sum = sum(weights)
    size_pcts = [round(w / weight_sum * 100, 1) for w in weights]

    tranches = []
    for idx, (_label, level) in enumerate(kept):
        lo = round(level * (1 - _ENTRY_ZONE_BAND_PCT), 2)
        hi = round(level * (1 + _ENTRY_ZONE_BAND_PCT), 2)
        entry = round((lo + hi) / 2, 2)

        rr = None
        if target is not None:
            reward = target * (1 - _ROUND_TRIP_COST_PCT) - entry * (1 + _ROUND_TRIP_COST_PCT)
            risk = entry * (1 + _ROUND_TRIP_COST_PCT) - invalidation
            if risk > 0:
                rr = round(reward / risk, 1)

        tranches.append(
            {
                "n": idx + 1,
                "trigger": (
                    f"Günlük kapanış {level:.2f} USD seviyesinin altına inerse (bölge "
                    f"{lo:.2f}-{hi:.2f} USD); gün içi dokunuş tetik saymaz."
                ),
                "price_zone": {"lo": lo, "hi": hi},
                "size_pct": size_pcts[idx],
                "invalidation": invalidation,
                "target": target,
                "rr": rr,
                "note": None,
            }
        )

    # Defensive monotonicity check (METODOLOJI.md Sec.1 item 5): as price
    # decreases from tranche to tranche, R:R should never decrease. Given
    # a fixed target/invalidation this is guaranteed by construction (see
    # the docstring), but flag rather than silently reorder if it somehow
    # doesn't hold.
    for idx in range(len(tranches) - 1):
        rr_cur, rr_next = tranches[idx]["rr"], tranches[idx + 1]["rr"]
        if rr_cur is not None and rr_next is not None and rr_next < rr_cur:
            tranches[idx + 1]["note"] = (
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
            for its shared ``invalidation`` level), or ``None``/``[]``.
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
        invalidation = entry_plan[-1].get("invalidation")
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
        "rationale": str}``. ``latest_value`` is a formatted Turkish
        percent string (e.g. ``"%23.4"``) read from the latest available
        fiscal year, never fabricated -- ``None`` if the chosen metric
        isn't computable from the given inputs, in which case
        ``rationale`` says so explicitly. ``trend`` is
        ``"iyileşiyor"``/``"bozuluyor"``/``"yatay"``, or ``None`` if no
        prior fiscal year's value is available to compare against. Never
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

    return {"name": chosen_name, "latest_value": latest_value, "trend": trend, "rationale": rationale}
