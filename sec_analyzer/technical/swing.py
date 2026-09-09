"""Composite *swing-trade* score from the technical-indicator set.

This is the scoring core of the S&P 500 swing screener (see
``sec_analyzer/screener/SWING_SPEC.md``, the binding contract for this
feature). Where :mod:`sec_analyzer.technical.momentum` answers "is this
stock's price trending up over weeks-to-months", this module answers a
narrower, more tactical question: "is *right now* a buyable moment for a
multi-day-to-multi-week swing trade" -- is the trend up, is price sitting in
a sane (not extended) entry zone, is there a fresh trigger, is it beating the
market, do buyers show up, and is the nearest risk/reward favourable.

Design, mirroring :mod:`sec_analyzer.technical.momentum` and the rest of the
technical layer:

* **Pure & None-safe** -- takes the flat ``indicators`` dict (the merged
  ``compute_indicators`` + ``relative_strength`` output) and returns a small
  dict of JSON-native scalars, or ``None`` when not a single component is
  available. Never raises.
* **One-directional** -- this is a long-only swing-setup score (SWING_SPEC.md
  Sec.0 non-goals): there is no short/bearish ranking, only "how good a long
  swing setup is this, right now".
* **Weights are tunable** -- the component weights and thresholds are module
  constants, matching the calibration-friendly style of ``momentum.py``.
* **Deterministic** -- same ``indicators`` in, same score dict out; no
  wall-clock dependence anywhere in the math.
* **Coverage floor** -- because this score exists to rank ~500 names against
  each other, a ticker whose surviving components carry too little combined
  weight is rejected outright (``None``) rather than renormalized into an
  extreme, non-comparable score (SWING_SPEC.md Sec.3.0).
"""

import logging
import math

logger = logging.getLogger(__name__)

# --- Composite-score weights (SWING_SPEC.md Sec.3.1; must be positive,
# renormalized over whichever components are actually available for a given
# ticker, identical convention to momentum.py). Trend and setup quality carry
# the most weight (is the timeframe right, is the entry not extended); the
# trigger nudges timing; relative strength/volume/risk-reward are smaller
# confirmations.
_WEIGHTS = {
    "trend": 0.25,
    "setup": 0.25,
    "trigger": 0.20,
    "rel_strength": 0.15,
    "volume": 0.10,
    "risk_reward": 0.05,
}

# --- Coverage floor (SWING_SPEC.md Sec.3.0, ranking integrity). This feature
# exists to rank ~500 tickers cross-sectionally, so a score renormalized over
# a tiny surviving component set (e.g. only a Bollinger squeeze computable)
# would read as extreme and would not be comparable to a score built on the
# full component set -- a real observed failure scored such a ticker 97 and
# topped the ranking on almost no evidence. Below this raw (pre-renormalize)
# summed weight, the ticker is rejected outright (returns ``None``) rather
# than scored on too little evidence.
_MIN_COVERAGE_WEIGHT = 0.60

#: Display labels for each component (screener table / card readout).
_COMPONENT_LABELS = {
    "trend": "Trend",
    "setup": "Setup quality",
    "trigger": "Trigger",
    "rel_strength": "Relative strength",
    "volume": "Volume",
    "risk_reward": "Risk/Reward",
}

# --- Trend sub-score scales (SWING_SPEC.md Sec.3.2): SMA slope % over their
# respective lookbacks that reads as full strength, and the distance from the
# 50-day that reads as full strength/weakness.
_SMA50_SLOPE_FULL = 5.0
_SMA200_SLOPE_FULL = 3.0
_DIST_SMA50_TREND_FULL = 10.0

# --- Setup sub-score: extension/pullback piecewise-linear zone (Sec.3.3a).
# ``ext`` is ``dist_sma50_pct``: a swing entry wants price at or slightly
# below the 50-day, not far above (extended) or far below (broken down).
_EXT_BUY_LO = -6.0
_EXT_BUY_HI = 2.0
_EXT_EXTENDED_HI = 30.0
_EXT_BREAKDOWN_LO = -30.0

# --- Setup sub-score: Bollinger squeeze base-tightness scale (Sec.3.3b) --
# the percentile denominator below which a non-``active`` squeeze still gets
# partial credit for a tightening base.
_SQUEEZE_PERCENTILE_FULL = 50.0

# --- Setup sub-score: 52-week-high proximity piecewise-linear zone
# (Sec.3.3c). ``d = dist_52w_high_pct`` is <= 0; 0 is at the high, the mid
# point reads neutral, the floor reads full-negative.
_HIGH_PROX_MID_PCT = -15.0
_HIGH_PROX_FLOOR_PCT = -35.0

# --- Relative-strength sub-score scales (Sec.3.5): 3-month/1-month
# outperformance vs. SPY (percentage points) that reads as full strength.
_RS_3M_FULL = 20.0
_RS_1M_FULL = 10.0

# --- Volume sub-score (Sec.3.6): up/down-volume-ratio log base (2.0 -> full
# strength at a 2x ratio, matching momentum.py's convention).
_UVR_LOG_BASE = 2.0

# --- Trigger sub-score (Sec.3.4): MACD histogram-only fallback magnitude
# (used only when there's no fresh cross to read).
_MACD_HIST_ONLY_MAGNITUDE = 0.4

# --- Risk/reward sub-score (Sec.3.7): log base such that a 3:1 reward:risk
# reads as full strength, 1:1 as neutral, 1:3 as full weakness.
_RR_LOG_BASE = 3.0

# --- Setup classification thresholds (Sec.3.8). First-match-wins against
# the raw indicators and the already-computed trend sub-score ``T``.
_BREAKOUT_DIST_HIGH_PCT = -3.0
_BREAKOUT_TREND_MIN = 0.2
_PULLBACK_TREND_MIN = 0.2
_PULLBACK_DIST_SMA50_LO = -12.0
_PULLBACK_DIST_SMA50_HI = 2.0
_SQUEEZE_SETUP_TREND_MIN = 0.0
_OVERSOLD_RSI_MAX = 40.0
_CONTINUATION_TREND_MIN = 0.4

# --- Badge thresholds (Sec.3.9).
_VOLUME_BADGE_REL_MIN = 1.5
_CAPITULATION_BADGE_MAX_BARS_AGO = 5

# --- Grade-label score bands (Sec.3.10), on the 0-100 display score.
_SCORE_STRONG_OPPORTUNITY = 75
_SCORE_OPPORTUNITY = 60
_SCORE_NEUTRAL = 45
_SCORE_WEAK = 30

# --- Trade-level construction (Sec.3.11).
_STOP_ATR_MULT = 2.0
_TARGET_ATR_MULT = 3.0
_STOP_SUPPORT_BUFFER = 0.99   # stop sits 1% below the nearest support level.
_TARGET_MIN_GAP = 1.02        # resistance must be >= 2% above price to count.

# Volatility floor on the stop (Sec.3.11): a stop closer to price than this
# many ATRs is inside the stock's own daily noise range -- ordinary chop can
# sweep it before the setup has had a chance to work, and it inflates `rr`
# into fantasy values (observed on a live scan: a stop 1.6% away against a
# 3.1% ATR read as an rr of 17). The support-based stop from the max() above
# is only trusted when it clears this floor; otherwise the 2xATR stop stands.
_STOP_MIN_ATR_MULT = 1.0


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _trend_subscore(ind: dict) -> "float | None":
    """Trend sub-score in [-1, 1] (SWING_SPEC.md Sec.3.2): average of
    whichever of SMA50/SMA200 slope, SMA50-vs-SMA200 state, and distance from
    the 50-day are computable. ``None`` if none are available."""
    subs = []
    s50 = ind.get("sma50_slope_pct")
    if _is_num(s50):
        subs.append(_clamp(float(s50) / _SMA50_SLOPE_FULL))
    s200 = ind.get("sma200_slope_pct")
    if _is_num(s200):
        subs.append(_clamp(float(s200) / _SMA200_SLOPE_FULL))
    above = ind.get("sma50_above_sma200")
    if above is True:
        subs.append(1.0)
    elif above is False:
        subs.append(-1.0)
    dist50 = ind.get("dist_sma50_pct")
    if _is_num(dist50):
        subs.append(_clamp(float(dist50) / _DIST_SMA50_TREND_FULL))
    if not subs:
        return None
    return sum(subs) / len(subs)


def _extension_component(ext: float) -> float:
    """Extension/pullback-quality piecewise-linear score (Sec.3.3a): a swing
    entry wants price at or slightly below the 50-day (the buy zone), not far
    extended above it or badly broken down below it."""
    if _EXT_BUY_LO <= ext <= _EXT_BUY_HI:
        return 1.0
    if _EXT_BUY_HI < ext <= _EXT_EXTENDED_HI:
        span = _EXT_EXTENDED_HI - _EXT_BUY_HI
        return _clamp(1.0 - (ext - _EXT_BUY_HI) * (2.0 / span))
    if _EXT_BREAKDOWN_LO <= ext < _EXT_BUY_LO:
        span = _EXT_BUY_LO - _EXT_BREAKDOWN_LO
        return _clamp(-1.0 + (ext - _EXT_BREAKDOWN_LO) * (2.0 / span))
    return -1.0


def _squeeze_component(bb_squeeze) -> "float | None":
    """Base-tightness score from ``bb_squeeze`` (Sec.3.3b): a live squeeze is
    full strength; otherwise a tight-but-not-yet-``active`` percentile gets
    partial credit. ``None`` if ``bb_squeeze`` is missing/unusable."""
    if not isinstance(bb_squeeze, dict):
        return None
    if bb_squeeze.get("active") is True:
        return 1.0
    percentile = bb_squeeze.get("percentile")
    if _is_num(percentile):
        return _clamp((_SQUEEZE_PERCENTILE_FULL - float(percentile)) / _SQUEEZE_PERCENTILE_FULL)
    return None


def _high_proximity_component(d: float) -> float:
    """52-week-high proximity piecewise-linear score (Sec.3.3c): at the high
    is full strength, decaying to neutral then full weakness the further
    below it price sits. ``d`` is expected <= 0."""
    if d >= 0:
        return 1.0
    if d >= _HIGH_PROX_MID_PCT:
        return _clamp(1.0 + d / abs(_HIGH_PROX_MID_PCT))
    if d >= _HIGH_PROX_FLOOR_PCT:
        span = _HIGH_PROX_FLOOR_PCT - _HIGH_PROX_MID_PCT
        return _clamp((d - _HIGH_PROX_MID_PCT) / span * -1.0)
    return -1.0


def _setup_subscore(ind: dict) -> "float | None":
    """Swing-specific "is this a buyable position" sub-score in [-1, 1]
    (SWING_SPEC.md Sec.3.3): average of extension/pullback quality, base
    tightness, and 52-week-high proximity, whichever are computable."""
    subs = []
    ext = ind.get("dist_sma50_pct")
    if _is_num(ext):
        subs.append(_extension_component(float(ext)))
    squeeze_sub = _squeeze_component(ind.get("bb_squeeze"))
    if squeeze_sub is not None:
        subs.append(squeeze_sub)
    dist_high = ind.get("dist_52w_high_pct")
    if _is_num(dist_high):
        subs.append(_high_proximity_component(float(dist_high)))
    if not subs:
        return None
    return sum(subs) / len(subs)


def _trigger_subscore(ind: dict) -> "float | None":
    """Fresh-actionable-turn sub-score in [-1, 1] (SWING_SPEC.md Sec.3.4):
    average of an RSI reclaim, a MACD read (fresh cross dominates, else
    histogram sign), and an RSI/price divergence, whichever fire. Note that
    ``volume_climax`` deliberately does not score here -- it is badge-only
    (Sec.3.9)."""
    subs = []
    rsi_reclaim = ind.get("rsi_reclaim")
    if rsi_reclaim == "bullish":
        subs.append(1.0)
    elif rsi_reclaim == "bearish":
        subs.append(-1.0)

    macd_cross = ind.get("macd_cross")
    macd_hist = ind.get("macd_hist")
    if macd_cross == "bullish":
        subs.append(1.0)
    elif macd_cross == "bearish":
        subs.append(-1.0)
    elif _is_num(macd_hist):
        mh = float(macd_hist)
        if mh > 0:
            subs.append(_MACD_HIST_ONLY_MAGNITUDE)
        elif mh < 0:
            subs.append(-_MACD_HIST_ONLY_MAGNITUDE)
        else:
            subs.append(0.0)

    rsi_divergence = ind.get("rsi_divergence")
    if rsi_divergence == "bullish":
        subs.append(1.0)
    elif rsi_divergence == "bearish":
        subs.append(-1.0)

    if not subs:
        return None
    return sum(subs) / len(subs)


def _rel_strength_subscore(ind: dict) -> "float | None":
    """Relative-strength-vs-SPY sub-score in [-1, 1] (SWING_SPEC.md Sec.3.5)
    from ``indicators["relative_strength"]``'s 3-month and 1-month
    out/under-performance. ``None`` if the dict is missing or neither field
    is numeric."""
    rs = ind.get("relative_strength")
    if not isinstance(rs, dict):
        return None
    subs = []
    rs_3m = rs.get("rs_3m_pct")
    if _is_num(rs_3m):
        subs.append(_clamp(float(rs_3m) / _RS_3M_FULL))
    rs_1m = rs.get("rs_1m_pct")
    if _is_num(rs_1m):
        subs.append(_clamp(float(rs_1m) / _RS_1M_FULL))
    if not subs:
        return None
    return sum(subs) / len(subs)


def _volume_subscore(ind: dict) -> "float | None":
    """Volume-confirmation sub-score in [-1, 1] (SWING_SPEC.md Sec.3.6): the
    up/down-volume ratio on a log scale, plus the OBV trend direction."""
    subs = []
    uvr = ind.get("updown_volume_ratio")
    if _is_num(uvr) and uvr > 0:
        subs.append(_clamp(math.log(float(uvr)) / math.log(_UVR_LOG_BASE)))
    obv_trend = ind.get("obv_trend")
    if obv_trend == "up":
        subs.append(1.0)
    elif obv_trend == "flat":
        subs.append(0.0)
    elif obv_trend == "down":
        subs.append(-1.0)
    if not subs:
        return None
    return sum(subs) / len(subs)


def _risk_reward_subscore(ind: dict) -> "float | None":
    """Nearest-support/resistance risk:reward sub-score in [-1, 1]
    (SWING_SPEC.md Sec.3.7) on a log scale (3:1 -> +1, 1:1 -> 0, 1:3 -> -1).
    ``None`` unless ``price``, a ``nearest_support`` below it, and a
    ``nearest_resistance`` above it are all available, with positive risk."""
    price = ind.get("price")
    support = ind.get("nearest_support")
    resistance = ind.get("nearest_resistance")
    if not (_is_num(price) and _is_num(support) and _is_num(resistance)):
        return None
    price, support, resistance = float(price), float(support), float(resistance)
    if not (support < price < resistance):
        return None
    risk = price - support
    reward = resistance - price
    if risk <= 0 or reward <= 0:
        return None
    rr = reward / risk
    return _clamp(math.log(rr) / math.log(_RR_LOG_BASE))


def _classify_setup(ind: dict, trend_sub: "float | None") -> str:
    """Deterministic, first-match-wins setup label (SWING_SPEC.md Sec.3.8),
    evaluated against the raw indicators and the already-computed ``trend``
    sub-score (a ``None`` trend sub-score is treated as ``0.0``). A
    comparison against a missing/``None`` field simply fails that rule rather
    than raising."""
    t = trend_sub if trend_sub is not None else 0.0

    dist_high = ind.get("dist_52w_high_pct")
    if _is_num(dist_high) and dist_high >= _BREAKOUT_DIST_HIGH_PCT and t >= _BREAKOUT_TREND_MIN:
        return "BREAKOUT"

    dist50 = ind.get("dist_sma50_pct")
    if t >= _PULLBACK_TREND_MIN and _is_num(dist50) and _PULLBACK_DIST_SMA50_LO <= dist50 <= _PULLBACK_DIST_SMA50_HI:
        return "TREND PULLBACK"

    squeeze = ind.get("bb_squeeze")
    if isinstance(squeeze, dict) and squeeze.get("active") is True and t >= _SQUEEZE_SETUP_TREND_MIN:
        return "SQUEEZE"

    rsi_reclaim = ind.get("rsi_reclaim")
    rsi14 = ind.get("rsi14")
    rsi_divergence = ind.get("rsi_divergence")
    if rsi_reclaim == "bullish" or (_is_num(rsi14) and rsi14 < _OVERSOLD_RSI_MAX and rsi_divergence == "bullish"):
        return "OVERSOLD BOUNCE"

    if t >= _CONTINUATION_TREND_MIN:
        return "MOMENTUM CONTINUATION"

    return "NO SETUP"


def _badges(ind: dict) -> "list[str]":
    """Short badge strings (SWING_SPEC.md Sec.3.9), emitted in this
    fixed order whenever their condition holds. Possibly empty."""
    badges = []

    rel_volume = ind.get("rel_volume")
    if _is_num(rel_volume) and rel_volume >= _VOLUME_BADGE_REL_MIN:
        badges.append("VOLUME")

    climax = ind.get("volume_climax")
    if (
        isinstance(climax, dict)
        and climax.get("detected")
        and climax.get("direction") == "down"
        and _is_num(climax.get("bars_ago"))
        and climax["bars_ago"] <= _CAPITULATION_BADGE_MAX_BARS_AGO
    ):
        badges.append("CAPITULATION")

    squeeze = ind.get("bb_squeeze")
    if isinstance(squeeze, dict) and squeeze.get("active") is True:
        badges.append("SQUEEZE")

    if ind.get("golden_cross") is True:
        badges.append("GOLDEN CROSS")

    if ind.get("rsi_divergence") == "bullish":
        badges.append("RSI DIVERGENCE")

    return badges


def _label_for_score(score: int) -> str:
    """Map the 0-100 display ``score`` to its grade label (5 bands,
    SWING_SPEC.md Sec.3.10)."""
    if score >= _SCORE_STRONG_OPPORTUNITY:
        return "STRONG OPPORTUNITY"
    if score >= _SCORE_OPPORTUNITY:
        return "OPPORTUNITY"
    if score >= _SCORE_NEUTRAL:
        return "NEUTRAL"
    if score >= _SCORE_WEAK:
        return "WEAK"
    return "AVOID"


def _trade_levels(ind: dict) -> dict:
    """Entry/stop/target/R:R trade levels (SWING_SPEC.md Sec.3.11).

    Only computed when ``price`` is available; each field is independently
    ``None`` when its own inputs are missing. All prices rounded to 2dp.

    The stop is ``max(base_stop, support_stop)`` (2xATR vs. a 1%-buffered
    nearest support), then subject to a volatility floor: if the resulting
    stop sits closer to price than ``_STOP_MIN_ATR_MULT`` ATRs, the
    support-based pick is discarded in favour of the 2xATR stop, since a
    sub-ATR stop is inside ordinary daily noise range and would otherwise
    inflate ``rr``. The floor only applies when ``atr14`` is available.

    Returns:
        ``{"entry", "stop", "stop_pct", "target", "target_pct", "rr"}``, all
        ``None`` when ``price`` itself is unavailable.
    """
    price = ind.get("price")
    if not _is_num(price) or price <= 0:
        return {"entry": None, "stop": None, "stop_pct": None, "target": None, "target_pct": None, "rr": None}
    price = float(price)

    atr = ind.get("atr14")
    atr = float(atr) if _is_num(atr) and atr > 0 else None

    support = ind.get("nearest_support")
    support = float(support) if _is_num(support) else None

    base_stop = price - _STOP_ATR_MULT * atr if atr is not None else None
    if support is not None and support < price:
        support_stop = support * _STOP_SUPPORT_BUFFER
        stop = max(base_stop, support_stop) if base_stop is not None else support_stop
    else:
        stop = base_stop

    # Volatility floor: a support-based stop that sits closer than 1 ATR is
    # inside daily noise range, so it's discarded in favour of the 2xATR
    # stop. When atr is unavailable there is nothing to floor against, so the
    # structural stop stands unchanged.
    if stop is not None and atr is not None and price - stop < _STOP_MIN_ATR_MULT * atr:
        stop = base_stop

    if stop is not None and not (0 < stop < price):
        stop = None

    resistance = ind.get("nearest_resistance")
    resistance = float(resistance) if _is_num(resistance) else None
    if resistance is not None and resistance >= price * _TARGET_MIN_GAP:
        target = resistance
    elif atr is not None:
        target = price + _TARGET_ATR_MULT * atr
    else:
        target = None

    rr = None
    if stop is not None and target is not None:
        risk = price - stop
        if risk > 0:
            rr = round((target - price) / risk, 2)

    return {
        "entry": round(price, 2),
        "stop": round(stop, 2) if stop is not None else None,
        "stop_pct": round((stop / price - 1) * 100, 1) if stop is not None else None,
        "target": round(target, 2) if target is not None else None,
        "target_pct": round((target / price - 1) * 100, 1) if target is not None else None,
        "rr": rr,
    }


def _build_summary(score: int, label: str, setup: str, components: "list[dict]") -> str:
    """One-line readout (SWING_SPEC.md Sec.3.12): the score/label/setup
    headline, plus the strongest positive and strongest negative driver
    components (by ``points``), when present."""
    head = f"{score}/100 {label.lower()} — {setup.lower()}"
    ordered = sorted(components, key=lambda c: c["points"])
    strongest = ordered[-1] if ordered else None
    weakest = ordered[0] if ordered else None
    tail = ""
    if strongest and strongest["points"] > 0:
        tail += f"; strongest: {strongest['label']}"
    if weakest and weakest["points"] < 0 and weakest is not strongest:
        tail += f", weakest: {weakest['label']}"
    return head + tail


def compute_swing_score(indicators: "dict | None") -> "dict | None":
    """Fold the technical-indicator set into one long-only swing-setup score.

    Args:
        indicators: The flat dict from
            :func:`sec_analyzer.technical.indicators.compute_indicators`,
            optionally merged with the ``relative_strength`` dict from
            :func:`sec_analyzer.technical.indicators.relative_strength`
            (the SPY cross-check). No other input is read. Every field may
            be ``None``/absent.

    Returns:
        ``None`` if ``indicators`` is not a dict, not a single component
        sub-score could be computed, or the coverage floor (Sec.3.0) rejects
        it -- i.e. ``price`` is not a positive number, ``trend`` or ``setup``
        is not computable, or the summed raw weight of the computable
        components is below ``_MIN_COVERAGE_WEIGHT``. This keeps the
        cross-sectional ranking honest: a score built on too little evidence
        is not comparable to one built on the full component set, so it is
        rejected rather than returned as a degraded dict. Otherwise a dict of
        JSON-native scalars:

        * ``score``: 0-100 display score (``50`` == neutral).
        * ``s``: the raw ``[-1, 1]`` score, 3dp.
        * ``label``: grade label (5 bands, Sec.3.10).
        * ``setup``: setup classification (Sec.3.8).
        * ``badges``: list of short badge strings (Sec.3.9), possibly
          empty.
        * ``components``: list of ``{key, label, sub, weight, points}`` for
          the contributing components only, in ``_WEIGHTS`` declaration
          order (``points`` sum to ``score - 50`` up to rounding).
        * ``entry`` / ``stop`` / ``stop_pct`` / ``target`` / ``target_pct`` /
          ``rr``: trade levels (Sec.3.11), each independently ``None`` when
          its inputs are missing.
        * ``summary``: a one-line readout (Sec.3.12).

    Never raises: any unexpected failure is logged and treated as "not
    computable" (returns ``None``), matching the rest of the technical layer
    and the project's "analysis layer never crashes the CLI" rule.
    """
    if not isinstance(indicators, dict):
        return None

    try:
        builders = {
            "trend": _trend_subscore,
            "setup": _setup_subscore,
            "trigger": _trigger_subscore,
            "rel_strength": _rel_strength_subscore,
            "volume": _volume_subscore,
            "risk_reward": _risk_reward_subscore,
        }

        raw = {}
        for key, fn in builders.items():
            sub = fn(indicators)
            if sub is not None:
                raw[key] = sub
        if not raw:
            return None

        total_w = sum(_WEIGHTS[k] for k in raw)

        # Coverage floor (SWING_SPEC.md Sec.3.0): runs after the sub-scores
        # are built and before renormalization. A ticker resting on too
        # little evidence is rejected outright rather than scored -- this is
        # a normal outcome for short-history/degraded-data names, not an
        # error, so it is logged at debug level and routed by the caller
        # (screener/swing_scan.py) into `skipped` with a "yetersiz veri"
        # reason.
        price = indicators.get("price")
        if (
            not (_is_num(price) and price > 0)
            or "trend" not in raw
            or "setup" not in raw
            or total_w < _MIN_COVERAGE_WEIGHT
        ):
            logger.debug(
                "compute_swing_score: rejected by coverage floor "
                "(price=%r, trend=%s, setup=%s, total_w=%.3f, floor=%.2f)",
                price, "trend" in raw, "setup" in raw, total_w, _MIN_COVERAGE_WEIGHT,
            )
            return None

        s = 0.0
        components = []
        for key, sub in raw.items():
            eff_w = _WEIGHTS[key] / total_w
            s += eff_w * sub
            components.append({
                "key": key,
                "label": _COMPONENT_LABELS[key],
                "sub": round(sub, 3),
                "weight": round(eff_w, 3),
                "points": round(eff_w * sub * 50.0, 1),
            })
        s = _clamp(s)
        score = int(round(50 + s * 50))
        score = max(0, min(100, score))

        label = _label_for_score(score)
        setup = _classify_setup(indicators, raw.get("trend"))
        badges = _badges(indicators)
        levels = _trade_levels(indicators)
        summary = _build_summary(score, label, setup, components)

        return {
            "score": score,
            "s": round(s, 3),
            "label": label,
            "setup": setup,
            "badges": badges,
            "components": components,
            "entry": levels["entry"],
            "stop": levels["stop"],
            "stop_pct": levels["stop_pct"],
            "target": levels["target"],
            "target_pct": levels["target_pct"],
            "rr": levels["rr"],
            "summary": summary,
        }
    except Exception:  # noqa: BLE001 - the swing score must never crash a scan
        logger.warning("compute_swing_score failed", exc_info=True)
        return None
