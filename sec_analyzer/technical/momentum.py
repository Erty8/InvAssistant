"""Composite *price* momentum score from the technical-indicator set.

This is the single, deterministic source of truth for the price-momentum
synthesis that used to live as ad-hoc JavaScript in the HTML report. It folds
the multi-horizon returns, relative strength, trend quality, volume
confirmation and oscillators computed by
:mod:`sec_analyzer.technical.indicators` into one ``[-1, 1]`` score ``S`` (and
a ``0-100`` display score), so the report, the terminal card and the backtest
all read the same number instead of each re-deriving it.

Design, mirroring the rest of the technical layer:

* **Pure & None-safe** -- takes the flat ``indicators`` dict (the merged
  ``compute_indicators`` + relative-strength output) and returns a small dict
  of JSON-native scalars, or ``None`` when not a single component is available.
  Never raises.
* **Context, not verdict** -- this score feeds the report's MOMENTUM row and
  the entry-timing narrative; it is deliberately kept out of the fair-value /
  valuation path (value convergence and trend continuation are different
  theses).
* **Weights are tunable** -- the component weights and thresholds are module
  constants so the backtest can calibrate them without touching logic.

Also hosts the SIC-code -> sector-ETF map used to compute *sector*-relative
strength (e.g. a semiconductor name benchmarked against SMH, not just SPY),
mirroring the SIC-range tables in :mod:`sec_analyzer.valuation.sector`.
"""

import logging
import math

logger = logging.getLogger(__name__)

# --- Composite-score weights (must be positive; renormalized over whichever
# components are actually available for a given ticker). Returns dominate the
# direction call; relative strength is the second pillar; oscillators only
# nudge. Tuned as starting values -- the backtest layer calibrates these.
_WEIGHTS = {
    "returns": 0.40,
    "rel_strength": 0.25,
    "trend": 0.20,
    "volume": 0.10,
    "oscillator": 0.05,
}

#: Turkish display labels for each component (report/terminal driver readout).
_COMPONENT_LABELS = {
    "returns": "Getiri (3a/6a/12-1)",
    "rel_strength": "Relatif güç",
    "trend": "Trend kalitesi",
    "volume": "Hacim teyidi",
    "oscillator": "Osilatör (RSI/MACD)",
}

#: Direction/label band edges on the [-1, 1] score S (kept identical to the
#: prior JS thresholds so the report reads the same).
_S_STRONG = 0.5
_S_MILD = 0.15

#: How many horizon-sigma of return maps to a full-strength (+/-1) returns
#: sub-score when volatility normalization is available.
_RETURN_SIGMA_FULL = 1.5

#: Fallback fixed scales (in return %) when 20-day volatility is unavailable,
#: per return horizon -- roughly the move that reads as full strength.
_RETURN_FALLBACK_SCALE = {"return_3m_pct": 40.0, "return_6m_pct": 60.0, "mom_12_1_pct": 80.0}

#: Within-component weights for the multi-horizon returns blend (shorter
#: horizon carries more weight for a swing-oriented read).
_RETURN_SUBWEIGHTS = {"return_3m_pct": 0.5, "return_6m_pct": 0.3, "mom_12_1_pct": 0.2}

#: Approximate horizon length in months for each return key (for annualized
#: -> horizon volatility scaling).
_RETURN_MONTHS = {"return_3m_pct": 3.0, "return_6m_pct": 6.0, "mom_12_1_pct": 11.0}

#: Relative outperformance (in percentage points, 3-month) that maps to a
#: full-strength relative-strength sub-score.
_RS_FULL_PP = 20.0

#: Trend-quality sub-score scales: SMA slope % over ~1 month that reads as
#: full strength, and the distance-below-52w-high (%) that maps to a zero /
#: full-negative proximity score.
_SLOPE50_FULL = 5.0
_SLOPE200_FULL = 3.0
_HIGH_PROX_ZERO_PCT = 25.0   # -25% below the high -> proximity sub-score 0
_HIGH_PROX_FLOOR_PCT = 50.0  # -50% or worse -> proximity sub-score -1

#: Acceleration deadband (percentage points) on the latest-month pace vs. the
#: pace implied by the prior two months -- below this reads as "steady".
_ACCEL_DEADBAND = 3.0


# --- SIC-code -> sector-ETF map (for sector-relative strength) --------------
# First-match-wins. Singles are checked before ranges so a semiconductor
# (3674) benchmarks against SMH rather than the broad-tech XLK its neighbours
# use, and petroleum refining (2911) lands in energy. Coverage is intentionally
# broad-brush: a miss simply falls back to SPY-only relative strength, so
# perfect GICS fidelity is not required. Mirrors the SIC-range table style in
# valuation/sector.py.
_SIC_ETF_SINGLES = {
    3674: "SMH",   # semiconductors
    2911: "XLE",   # petroleum refining
    6798: "XLRE",  # REIT
}

_SIC_ETF_RANGES = (
    (1000, 1299, "XLB"),   # metal / mineral mining -> materials
    (1300, 1399, "XLE"),   # oil & gas extraction -> energy
    (1400, 1499, "XLB"),   # nonmetallic mining -> materials
    (2000, 2199, "XLP"),   # food & beverage -> consumer staples
    (2200, 2399, "XLY"),   # textiles / apparel -> consumer discretionary
    (2400, 2599, "XLI"),   # lumber / furniture -> industrials
    (2600, 2699, "XLB"),   # paper -> materials
    (2700, 2799, "XLC"),   # publishing -> communication services
    (2800, 2829, "XLB"),   # industrial chemicals -> materials
    (2830, 2836, "XLV"),   # pharma / biotech -> health care
    (2840, 2899, "XLB"),   # chemicals -> materials
    (2900, 2999, "XLE"),   # petroleum & coal -> energy
    (3000, 3299, "XLB"),   # rubber / plastics / stone-clay-glass -> materials
    (3300, 3399, "XLB"),   # primary metals -> materials
    (3400, 3569, "XLI"),   # fabricated metal / machinery -> industrials
    (3570, 3579, "XLK"),   # computers & office equipment -> technology
    (3600, 3673, "XLK"),   # electronic equipment -> technology
    (3675, 3699, "XLK"),   # electronic components (ex-semis) -> technology
    (3700, 3716, "XLY"),   # motor vehicles -> consumer discretionary
    (3717, 3799, "XLI"),   # aerospace / transport equipment -> industrials
    (3800, 3839, "XLI"),   # measuring / control instruments -> industrials
    (3840, 3859, "XLV"),   # medical devices -> health care
    (3860, 3999, "XLK"),   # photographic / misc manufacturing -> technology
    (4000, 4499, "XLI"),   # rail / trucking / water transport -> industrials
    (4500, 4599, "XLI"),   # air transport -> industrials
    (4600, 4699, "XLE"),   # pipelines -> energy
    (4700, 4799, "XLI"),   # transportation services -> industrials
    (4800, 4899, "XLC"),   # telecommunications -> communication services
    (4900, 4999, "XLU"),   # utilities
    (5000, 5199, "XLI"),   # wholesale trade -> industrials
    (5200, 5999, "XLY"),   # retail -> consumer discretionary
    (6000, 6499, "XLF"),   # banks / insurance / finance
    (6500, 6599, "XLRE"),  # real estate operators -> real estate
    (6700, 6799, "XLF"),   # holding / investment offices -> finance
    (7000, 7299, "XLY"),   # hotels / personal services -> consumer discretionary
    (7300, 7369, "XLI"),   # business services -> industrials
    (7370, 7379, "XLK"),   # software / IT services -> technology
    (7380, 7399, "XLI"),   # misc business services -> industrials
    (7400, 7999, "XLC"),   # entertainment / recreation -> communication services
    (8000, 8099, "XLV"),   # health services -> health care
    (8100, 8999, "XLI"),   # misc professional services -> industrials
)


def sector_etf_for_sic(sic) -> "str | None":
    """Map a SIC code to a sector-ETF symbol for sector-relative strength, or
    ``None`` if the code is missing/unparseable or unmapped (caller then falls
    back to SPY-only relative strength). Never raises."""
    if sic is None:
        return None
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    if code in _SIC_ETF_SINGLES:
        return _SIC_ETF_SINGLES[code]
    for lo, hi, etf in _SIC_ETF_RANGES:
        if lo <= code <= hi:
            return etf
    return None


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _returns_subscore(ind: dict) -> "float | None":
    """Volatility-normalized multi-horizon returns sub-score in [-1, 1].

    Each horizon's return is scaled by that horizon's expected volatility
    (annualized 20-day vol projected to the horizon), so a given % move counts
    for more on a calm stock than on a wild one -- comparable across tickers.
    Falls back to a fixed per-horizon scale when volatility is unavailable."""
    vol_annual = ind.get("volatility_20d")
    vol_annual = float(vol_annual) if _is_num(vol_annual) and vol_annual > 0 else None
    num, wsum = 0.0, 0.0
    for key, subw in _RETURN_SUBWEIGHTS.items():
        r = ind.get(key)
        if not _is_num(r):
            continue
        r = float(r)
        if vol_annual is not None:
            months = _RETURN_MONTHS[key]
            horizon_vol_pct = vol_annual * 100.0 * math.sqrt(months / 12.0)
            denom = _RETURN_SIGMA_FULL * horizon_vol_pct
            sub = _clamp(r / denom) if denom > 0 else 0.0
        else:
            sub = _clamp(r / _RETURN_FALLBACK_SCALE[key])
        num += sub * subw
        wsum += subw
    if wsum == 0:
        return None
    return num / wsum


def _rel_strength_subscore(ind: dict) -> "float | None":
    """Relative-strength sub-score in [-1, 1] from the 3-month out/under-
    performance vs. SPY and (if available) the sector ETF, equally weighted.
    Positive means the stock beat its benchmark(s)."""
    subs = []
    for key in ("relative_strength", "relative_strength_sector"):
        rs = ind.get(key)
        if isinstance(rs, dict) and _is_num(rs.get("rs_3m_pct")):
            subs.append(_clamp(float(rs["rs_3m_pct"]) / _RS_FULL_PP))
    if not subs:
        return None
    return sum(subs) / len(subs)


def _trend_subscore(ind: dict) -> "float | None":
    """Trend-quality sub-score in [-1, 1]: are the moving averages themselves
    rising (slope) and is price near its 52-week high (proximity)? Independent
    of the raw returns, so a smoothly-trending name scores higher than an
    equally-up but choppy one."""
    subs = []
    s50 = ind.get("sma50_slope_pct")
    if _is_num(s50):
        subs.append(_clamp(float(s50) / _SLOPE50_FULL))
    s200 = ind.get("sma200_slope_pct")
    if _is_num(s200):
        subs.append(_clamp(float(s200) / _SLOPE200_FULL))
    dist_high = ind.get("dist_52w_high_pct")
    if _is_num(dist_high):
        # dist_high <= 0 (at/below high). 0 -> +1 (at high), -_HIGH_PROX_ZERO -> 0,
        # -_HIGH_PROX_FLOOR or worse -> -1. Linear between.
        d = float(dist_high)
        prox = 1.0 + d / _HIGH_PROX_ZERO_PCT
        subs.append(_clamp(prox))
    if not subs:
        return None
    return sum(subs) / len(subs)


def _volume_subscore(ind: dict) -> "float | None":
    """Volume-confirmation sub-score in [-1, 1] from the up/down volume ratio
    on a log scale (ratio 2.0 -> +1 accumulation, 0.5 -> -1 distribution)."""
    uvr = ind.get("updown_volume_ratio")
    if not _is_num(uvr) or uvr <= 0:
        return None
    return _clamp(math.log(float(uvr)) / math.log(2.0))


def _oscillator_subscore(ind: dict) -> "float | None":
    """Oscillator sub-score in [-1, 1] from RSI (centered on 50) and MACD
    (fresh cross dominates, else histogram sign). Small weight -- a nudge, not
    a driver."""
    subs = []
    rsi = ind.get("rsi14")
    if _is_num(rsi):
        subs.append(_clamp((float(rsi) - 50.0) / 40.0))
    mc = ind.get("macd_cross")
    mh = ind.get("macd_hist")
    if mc == "bullish":
        subs.append(1.0)
    elif mc == "bearish":
        subs.append(-1.0)
    elif _is_num(mh):
        subs.append(0.6 if mh > 0 else (-0.6 if mh < 0 else 0.0))
    if not subs:
        return None
    return sum(subs) / len(subs)


def _acceleration(ind: dict, direction: str) -> "str | None":
    """Turkish acceleration word from the latest-month pace vs. the pace
    implied by the two months before it, interpreted in the direction of the
    prevailing momentum. Mirrors the prior JS logic (deadband + direction
    mapping). ``None`` when the 1m/3m returns aren't both available."""
    r1 = ind.get("return_1m_pct")
    r3 = ind.get("return_3m_pct")
    if not (_is_num(r1) and _is_num(r3)):
        return None
    r1, r3 = float(r1), float(r3)
    pace_earlier = (r3 - r1) / 2.0
    diff = r1 - pace_earlier
    if diff > _ACCEL_DEADBAND:
        accel = "up"
    elif diff < -_ACCEL_DEADBAND:
        accel = "down"
    else:
        accel = "steady"
    if direction == "down":
        return "hızlanıyor" if accel == "down" else ("yavaşlıyor" if accel == "up" else "sabit")
    # up and flat share the same literal reading of the pace change.
    return "hızlanıyor" if accel == "up" else ("yavaşlıyor" if accel == "down" else "sabit")


def _label_for(s: float) -> "tuple[str, str, str]":
    """Map the score ``s`` to (Turkish label, direction, arrow)."""
    if s >= _S_STRONG:
        return "GÜÇLÜ YUKARI MOMENTUM", "up", "↑"
    if s >= _S_MILD:
        return "YUKARI MOMENTUM", "up", "↑"
    if s >= -_S_MILD:
        return "YATAY MOMENTUM", "flat", "→"
    if s > -_S_STRONG:
        return "AŞAĞI MOMENTUM", "down", "↓"
    return "GÜÇLÜ AŞAĞI MOMENTUM", "down", "↓"


def compute_price_momentum(indicators: "dict | None") -> "dict | None":
    """Fold the technical-indicator set into one composite price-momentum read.

    Args:
        indicators: The flat dict from
            :func:`sec_analyzer.technical.indicators.compute_indicators`,
            merged with the relative-strength dicts (``relative_strength`` /
            ``relative_strength_sector``) as assembled by the CLI/web layer.

    Returns:
        A dict of JSON-native scalars, or ``None`` if not a single component
        could be computed:

        * ``score``: 0-100 display score (``50`` == neutral).
        * ``s``: the raw ``[-1, 1]`` score.
        * ``direction``: ``"up"`` / ``"flat"`` / ``"down"``.
        * ``label``: Turkish momentum label (5 bands).
        * ``arrow``: ``"↑"`` / ``"→"`` / ``"↓"``.
        * ``accel``: ``"hızlanıyor"`` / ``"sabit"`` / ``"yavaşlıyor"`` /
          ``None`` -- whether the move is speeding up or cooling off.
        * ``meter_pos``: 0-100 marker position (same as ``score``, kept as a
          distinct key for the report meter).
        * ``components``: list of ``{key, label, sub, weight, points}`` for the
          contributing components (``points`` sum to ``score - 50``), so the
          report can show what pushed the score up/down.
        * ``summary``: a one-line Turkish readout.
    """
    if not isinstance(indicators, dict):
        return None

    builders = {
        "returns": _returns_subscore,
        "rel_strength": _rel_strength_subscore,
        "trend": _trend_subscore,
        "volume": _volume_subscore,
        "oscillator": _oscillator_subscore,
    }

    raw = {}
    for key, fn in builders.items():
        sub = fn(indicators)
        if sub is not None:
            raw[key] = sub
    if not raw:
        return None

    total_w = sum(_WEIGHTS[k] for k in raw)
    if total_w <= 0:
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

    label, direction, arrow = _label_for(s)
    accel = _acceleration(indicators, direction)

    summary = _build_summary(score, label, accel, components)

    return {
        "score": score,
        "s": round(s, 3),
        "direction": direction,
        "label": label,
        "arrow": arrow,
        "accel": accel,
        "meter_pos": score,
        "components": components,
        "summary": summary,
    }


def _build_summary(score: int, label: str, accel: "str | None", components: list) -> str:
    """One-line Turkish readout: score + label (+ acceleration), then the
    strongest positive and strongest negative driver."""
    head = f"{score}/100 {label.lower()}"
    if accel:
        head += f", {accel}"
    ordered = sorted(components, key=lambda c: c["points"])
    strongest = ordered[-1] if ordered else None
    weakest = ordered[0] if ordered else None
    tail_parts = []
    if strongest and strongest["points"] > 0:
        tail_parts.append(f"en güçlü: {strongest['label'].lower()}")
    if weakest and weakest["points"] < 0 and weakest is not strongest:
        tail_parts.append(f"en zayıf: {weakest['label'].lower()}")
    if tail_parts:
        return head + "; " + ", ".join(tail_parts) + "."
    return head + "."
