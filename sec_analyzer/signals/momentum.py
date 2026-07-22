"""Fundamental- and verdict-momentum signal layer (deterministic, no LLM).

Mirrors :mod:`sec_analyzer.signals.events`: pure, defensive functions that
never raise, return small Turkish-labelled dicts, and are wired into the CLI
*after* the fetch/interpret steps rather than through the LLM payload. These
are **context** signals -- they feed the report's MOMENTUM row and the
entry-timing narrative; they never enter the fair-value computation.

Two layers live here:

* :func:`compute_fundamental_momentum` -- the *second derivative* of the
  business: is quarterly revenue growth accelerating or decelerating, are
  margins trending up or down, and does the latest realized revenue beat or
  miss what a prior stored analysis projected ("model-based surprise").
* :func:`compute_verdict_momentum` -- the trajectory of the model's own
  fair-value / price ratio across successive stored analyses: is the model
  finding the name progressively cheaper (a convergence opportunity) or is
  fair value eroding toward the price (a weakening thesis)?
"""

import logging
from datetime import datetime

from sec_analyzer.normalize.normalizer import (
    latest_annual_value,
    to_quarterly_series,
)

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"

#: Minimum quarter-over-year-ago gap (days) for a YoY pairing, and the
#: tolerance around a 1-year gap.
_YOY_GAP_DAYS = 365
_YOY_GAP_TOL = 55

#: Deadbands (percentage points) for the trend classifications, so noise
#: reads as "steady" rather than flip-flopping.
_REV_ACCEL_DEADBAND_PP = 2.0
_MARGIN_TREND_DEADBAND_PP = 1.0
_SURPRISE_DEADBAND_PP = 2.0

#: How many recent YoY / margin points a trend classification looks at.
_TREND_WINDOW = 4

#: Composite label thresholds over the weighted sub-signal score.
_LABEL_POS = 2
_LABEL_NEG = -2

_LABEL_POSITIVE = "POZİTİF"
_LABEL_NEUTRAL = "NÖTR"
_LABEL_NEGATIVE = "NEGATİF"


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], _DATE_FMT)
    except (ValueError, TypeError):
        return None


def _classify_trend(values, deadband):
    """Classify a short ascending numeric series as rising/steady/falling by
    comparing the mean of its recent half against the mean of its earlier half,
    with a deadband. Returns ``+1`` / ``0`` / ``-1`` and the signed magnitude
    (recent - earlier), or ``(None, None)`` if there are fewer than two points."""
    window = [v for v in values[-_TREND_WINDOW:] if v is not None]
    if len(window) < 2:
        return None, None
    half = len(window) // 2
    earlier = window[:half] if half > 0 else window[:1]
    recent = window[half:]
    diff = (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))
    if diff > deadband:
        return 1, diff
    if diff < -deadband:
        return -1, diff
    return 0, diff


def _classify_accel(values):
    """Robust revenue-growth acceleration from a short ascending YoY series.

    Quarterly YoY is noisy, so a bare half-mean comparison can be flipped by a
    single quarter. This adds two guards on top of :func:`_classify_trend`:

    1. **Latest-quarter agreement** -- an "accelerating" call requires the most
       recent quarter's YoY to actually be higher than the prior quarter's
       (mirrored for decelerating). This blocks a series whose mean drifted up
       but whose latest print turned down (``[36, 48, 63.5, 35.7]``) from
       reading as "hızlanıyor".
    2. **Two-quarter confirmation** -- ``confirmed`` is ``True`` only when the
       last *two* consecutive transitions agree with the direction; a move
       resting on a single quarter is returned unconfirmed (the caller marks it
       "teyit bekliyor" and weights it half).

    Returns ``(sign, confirmed)`` where ``sign`` is ``+1``/``0``/``-1`` (or
    ``None`` with too little data) and ``confirmed`` is a bool.
    """
    window = [v for v in values[-_TREND_WINDOW:] if v is not None]
    if len(window) < 2:
        return None, False
    sign, _ = _classify_trend(window, _REV_ACCEL_DEADBAND_PP)
    if not sign:  # None or 0
        return sign, False
    # Guard 1: the latest quarter must move in the claimed direction.
    last_delta = window[-1] - window[-2]
    if sign > 0 and last_delta <= 0:
        return 0, False
    if sign < 0 and last_delta >= 0:
        return 0, False
    # Guard 2: confirmed only if the last two transitions both agree.
    confirmed = False
    if len(window) >= 3:
        d1 = window[-1] - window[-2]
        d2 = window[-2] - window[-3]
        confirmed = (d1 * sign > 0) and (d2 * sign > 0)
    return sign, confirmed


def _yoy_growth_series(quarters):
    """Year-over-year growth (%) for each quarter that has a ~1-year-earlier
    counterpart, ascending. ``quarters`` is the ascending
    ``[{period_end, value}]`` list from :func:`to_quarterly_series`. Pairs a
    quarter with the earlier quarter closest to a 365-day gap (tolerant of
    missing quarters). Skips pairs whose base is non-positive."""
    dated = []
    for q in quarters:
        d = _parse_date(q.get("period_end"))
        v = q.get("value")
        if d is not None and v is not None:
            dated.append((d, float(v), q.get("period_end")))
    out = []
    for i, (d_i, v_i, pe_i) in enumerate(dated):
        best = None
        best_gap = None
        for j in range(i):
            gap = (d_i - dated[j][0]).days
            if abs(gap - _YOY_GAP_DAYS) <= _YOY_GAP_TOL:
                score = abs(gap - _YOY_GAP_DAYS)
                if best_gap is None or score < best_gap:
                    best, best_gap = dated[j], score
        if best is None:
            continue
        base = best[1]
        if base <= 0:
            continue
        out.append({"period_end": pe_i, "yoy_pct": round((v_i / base - 1.0) * 100.0, 1)})
    return out


def _aligned(series_a, series_b):
    """Inner-join two ascending ``[{period_end, value}]`` series on
    ``period_end``, returning ascending ``[(period_end, value_a, value_b)]``."""
    by_pe = {q["period_end"]: q["value"] for q in series_b if q.get("period_end") is not None}
    out = []
    for q in series_a:
        pe = q.get("period_end")
        if pe in by_pe and q.get("value") is not None and by_pe[pe] is not None:
            out.append((pe, float(q["value"]), float(by_pe[pe])))
    return out


def _margin_series(normalized, numerator_concept):
    """Quarterly margin (%) series = ``numerator / Revenue`` aligned by quarter,
    ascending ``[{period_end, value}]``; ``[]`` if either series is missing."""
    rev = to_quarterly_series(normalized, "Revenue")
    num = to_quarterly_series(normalized, numerator_concept)
    out = []
    for pe, num_v, rev_v in _aligned(num, rev):
        if rev_v > 0:
            out.append({"period_end": pe, "value": round(num_v / rev_v * 100.0, 2)})
    return out


def _fcf_margin_series(normalized):
    """Quarterly FCF-margin (%) series = ``(OperatingCashFlow - CapEx)/Revenue``
    aligned by quarter, ascending; ``[]`` if inputs are missing. CapEx is a
    positive outflow (subtracted), matching the annual FCF convention."""
    rev = to_quarterly_series(normalized, "Revenue")
    ocf = to_quarterly_series(normalized, "OperatingCashFlow")
    capex = to_quarterly_series(normalized, "CapEx")
    if not rev or not ocf or not capex:
        return []
    capex_by_pe = {q["period_end"]: q["value"] for q in capex}
    fcf = []
    for q in ocf:
        pe = q.get("period_end")
        if pe in capex_by_pe and q.get("value") is not None and capex_by_pe[pe] is not None:
            fcf.append({"period_end": pe, "value": float(q["value"]) - float(capex_by_pe[pe])})
    out = []
    for pe, fcf_v, rev_v in _aligned(fcf, rev):
        if rev_v > 0:
            out.append({"period_end": pe, "value": round(fcf_v / rev_v * 100.0, 2)})
    return out


def _trend_word(sign, improving="iyileşiyor", worsening="bozuluyor", flat="sabit"):
    if sign is None:
        return None
    return improving if sign > 0 else (worsening if sign < 0 else flat)


def _accel_word(sign):
    if sign is None:
        return None
    return "hızlanıyor" if sign > 0 else ("yavaşlıyor" if sign < 0 else "sabit")


def _ttm_revenue(normalized):
    """Trailing-twelve-month revenue from the last four true quarters, or the
    latest annual value if fewer than four quarters are available."""
    quarters = to_quarterly_series(normalized, "Revenue")
    if len(quarters) >= 4:
        return sum(float(q["value"]) for q in quarters[-4:]), quarters[-1].get("period_end")
    annual = latest_annual_value(normalized, "Revenue")
    if annual is not None:
        return float(annual), None
    return None, None


def _interpolate_path(base_revenue, revenue_path, elapsed_years):
    """Log-linear interpolation of an annual revenue path at ``elapsed_years``.

    ``base_revenue`` is the year-0 anchor; ``revenue_path[k]`` is the projected
    revenue for year ``k+1``. Returns the implied revenue at the (possibly
    fractional) elapsed point, or ``None`` if it can't be bracketed / a value
    is non-positive."""
    if base_revenue is None or base_revenue <= 0 or not revenue_path:
        return None
    if elapsed_years <= 0:
        return float(base_revenue)
    anchors = [float(base_revenue)] + [float(v) for v in revenue_path]
    k = int(elapsed_years)
    frac = elapsed_years - k
    if k >= len(anchors) - 1:
        return anchors[-1]
    v_k, v_k1 = anchors[k], anchors[k + 1]
    if v_k <= 0 or v_k1 <= 0:
        return None
    return v_k * (v_k1 / v_k) ** frac


def _model_surprise(normalized, prior_verdict):
    """Realized-revenue vs. prior-model-projection surprise ("model-based
    beat/miss"). ``prior_verdict`` is ``{"valuation": <dict>, "ref_date":
    "YYYY-MM-DD"}`` from the most recent prior live analysis. Returns
    ``{"surprise_pct", "direction", "basis"}`` or ``None`` when the prior
    analysis has no revenue projection (e.g. it wasn't a revenue-first
    hyper-grower) or the data can't be assembled -- degrades cleanly for old
    stored verdicts that predate revenue-path persistence."""
    if not isinstance(prior_verdict, dict):
        return None
    valuation = prior_verdict.get("valuation")
    prior_date = _parse_date(prior_verdict.get("ref_date"))
    if not isinstance(valuation, dict) or prior_date is None:
        return None
    scenarios = (((valuation.get("hyper_growth_detail") or {}).get("scenarios")) or {})
    base = scenarios.get("base") if isinstance(scenarios, dict) else None
    if not isinstance(base, dict):
        return None
    base_revenue = base.get("base_revenue")
    revenue_path = base.get("revenue_path")
    if base_revenue is None or not revenue_path:
        return None

    actual, actual_pe = _ttm_revenue(normalized)
    if actual is None or actual <= 0:
        return None
    # Determinism: anchor the elapsed time on the latest quarter's period_end
    # (a filed fact), never wall-clock. If there's no dated quarter (TTM fell
    # back to the annual value), skip the surprise rather than guess a date.
    actual_date = _parse_date(actual_pe)
    if actual_date is None:
        return None
    elapsed_years = (actual_date - prior_date).days / 365.25
    if elapsed_years <= 0:
        return None
    implied = _interpolate_path(base_revenue, revenue_path, elapsed_years)
    if implied is None or implied <= 0:
        return None

    surprise_pct = round((actual / implied - 1.0) * 100.0, 1)
    if surprise_pct > _SURPRISE_DEADBAND_PP:
        direction = "beat"
    elif surprise_pct < -_SURPRISE_DEADBAND_PP:
        direction = "miss"
    else:
        direction = "inline"
    basis = (
        f"Gerçekleşen gelir, {prior_date.date().isoformat()} tarihli modelin baz "
        f"senaryosunun ima ettiği seviyeye göre %{surprise_pct:+.1f}."
    )
    return {"surprise_pct": surprise_pct, "direction": direction, "basis": basis}


def compute_fundamental_momentum(normalized: dict, prior_verdict: "dict | None" = None) -> "dict | None":
    """Fundamental momentum: growth acceleration, margin trend, model surprise.

    Args:
        normalized: The normalized-facts dict (annual + quarterly buckets)
            from :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        prior_verdict: Optional ``{"valuation": <dict>, "ref_date": str}`` for
            the most recent prior *live* analysis, used for the model-based
            surprise. ``None`` disables that sub-signal.

    Returns:
        A dict of JSON-native values, or ``None`` when quarterly fundamentals
        are too sparse to say anything (e.g. banks with no Revenue concept):

        * ``label``: ``"POZİTİF"`` / ``"NÖTR"`` / ``"NEGATİF"``.
        * ``s``: continuous score in ``[-1, 1]`` (the fundamental-momentum axis
          of the report's price x fundamental quadrant).
        * ``score``: ``0-100`` display score (``50`` == neutral).
        * ``revenue_accel``: ``{"word", "confirmed", "latest_yoy_pct",
          "yoy_series"}`` or ``None`` -- is quarterly YoY revenue growth
          speeding up or slowing? ``confirmed`` is ``False`` for a
          single-quarter move (weighted half, flagged "teyit bekliyor").
        * ``margin_trend``: ``{"gross", "fcf"}`` trend words (or ``None`` each).
        * ``model_surprise``: see :func:`_model_surprise`, or ``None``.
        * ``detail``: a one-line Turkish readout.
    """
    if not isinstance(normalized, dict):
        return None

    quarters = to_quarterly_series(normalized, "Revenue")
    yoy_series = _yoy_growth_series(quarters)

    revenue_accel = None
    accel_sign = None
    accel_confirmed = False
    if yoy_series:
        yoy_values = [y["yoy_pct"] for y in yoy_series]
        accel_sign, accel_confirmed = _classify_accel(yoy_values)
        revenue_accel = {
            "word": _accel_word(accel_sign),
            "confirmed": accel_confirmed,
            "latest_yoy_pct": yoy_values[-1],
            "yoy_series": yoy_series[-_TREND_WINDOW:],
        }

    gross_series = _margin_series(normalized, "GrossProfit")
    gross_sign, _ = _classify_trend([m["value"] for m in gross_series], _MARGIN_TREND_DEADBAND_PP)
    fcf_series = _fcf_margin_series(normalized)
    fcf_sign, _ = _classify_trend([m["value"] for m in fcf_series], _MARGIN_TREND_DEADBAND_PP)
    margin_trend = {"gross": _trend_word(gross_sign), "fcf": _trend_word(fcf_sign)}

    model_surprise = _model_surprise(normalized, prior_verdict)

    # Nothing usable at all -> None (report renders "—").
    if revenue_accel is None and gross_sign is None and fcf_sign is None and model_surprise is None:
        return None

    # Composite: acceleration is weighted double (it is the headline signal).
    # Max magnitude is 5 (accel ±2, gross ±1, fcf ±1, surprise ±1), used to
    # normalize into a continuous [-1, 1] score `s` for the report's price x
    # fundamental momentum quadrant, and a 0-100 display `score`.
    raw = 0
    if accel_sign is not None:
        # Acceleration is the headline signal (double weight) -- but only when
        # confirmed by two consecutive quarters. A single-quarter move counts
        # single-weight, so a noisy one-off can't drive the label on its own.
        raw += (2 if accel_confirmed else 1) * accel_sign
    if gross_sign is not None:
        raw += gross_sign
    if fcf_sign is not None:
        raw += fcf_sign
    if model_surprise is not None:
        raw += {"beat": 1, "miss": -1, "inline": 0}[model_surprise["direction"]]

    if raw >= _LABEL_POS:
        label = _LABEL_POSITIVE
    elif raw <= _LABEL_NEG:
        label = _LABEL_NEGATIVE
    else:
        label = _LABEL_NEUTRAL

    s = max(-1.0, min(1.0, raw / 5.0))
    score = int(round(50 + s * 50))

    detail = _build_fundamental_detail(revenue_accel, margin_trend, model_surprise)

    return {
        "label": label,
        "s": round(s, 3),
        "score": score,
        "revenue_accel": revenue_accel,
        "margin_trend": margin_trend,
        "model_surprise": model_surprise,
        "detail": detail,
    }


#: Relative deadband on the FV/price ratio trajectory (5%) below which the
#: verdict-momentum reads as flat.
_VERDICT_RATIO_DEADBAND = 0.05


def compute_verdict_momentum(history: "list | None") -> "dict | None":
    """The trajectory of the model's own fair-value / price ratio across
    successive stored *live* analyses.

    Args:
        history: The list of scalar verdict rows from
            :func:`sec_analyzer.store.database.load_verdicts` (call it with
            ``live_only=True``). Each row needs ``analyzed_at``, ``price`` and
            the base band columns ``fv_base_lo``/``fv_base_hi``.

    Returns:
        A dict, or ``None`` if fewer than two dated points carry a usable
        FV(base mid)/price ratio:

        * ``label``: ``"POZİTİF"`` / ``"NÖTR"`` / ``"NEGATİF"``.
        * ``direction``: ``"up"`` / ``"flat"`` / ``"down"`` (ratio trajectory).
        * ``series``: ascending ``[{date, fv, price, ratio}]``.
        * ``detail``: a one-line Turkish reading of the trajectory.
    """
    if not history:
        return None
    points = []
    for row in history:
        if not isinstance(row, dict):
            continue
        date = row.get("analyzed_at")
        price = row.get("price")
        lo = row.get("fv_base_lo")
        hi = row.get("fv_base_hi")
        if date is None or not _is_positive(price) or lo is None or hi is None:
            continue
        fv_mid = (float(lo) + float(hi)) / 2.0
        if fv_mid <= 0:
            continue
        points.append({
            "date": str(date)[:10],
            "fv": round(fv_mid, 2),
            "price": round(float(price), 2),
            "ratio": round(fv_mid / float(price), 3),
        })
    if len(points) < 2:
        return None
    points.sort(key=lambda p: p["date"])
    # Dedupe consecutive same-date points (keep the latest per date).
    dedup = {}
    for p in points:
        dedup[p["date"]] = p
    series = [dedup[d] for d in sorted(dedup)]
    if len(series) < 2:
        return None

    first, last = series[0], series[-1]
    ratio_change = last["ratio"] / first["ratio"] - 1.0 if first["ratio"] > 0 else 0.0
    price_down = last["price"] < first["price"]
    fv_down = last["fv"] < first["fv"]

    if ratio_change > _VERDICT_RATIO_DEADBAND:
        label, direction = _LABEL_POSITIVE, "up"
        if price_down:
            detail = "Model bu ismi giderek daha ucuz buluyor: FV/fiyat oranı yükseliyor, fiyat geriliyor (yakınsama fırsatı)."
        else:
            detail = "FV/fiyat oranı yükseliyor: adil değer fiyattan daha hızlı artıyor."
    elif ratio_change < -_VERDICT_RATIO_DEADBAND:
        direction = "down"
        if fv_down:
            label = _LABEL_NEGATIVE
            detail = "Tez zayıflıyor: adil değer fiyata doğru eriyor (FV/fiyat oranı düşüyor)."
        else:
            label = _LABEL_NEUTRAL
            detail = "Yakınsama gerçekleşiyor: fiyat adil değere yaklaşıyor (FV/fiyat oranı düşüyor)."
    else:
        label, direction = _LABEL_NEUTRAL, "flat"
        detail = "FV/fiyat oranı yatay: model görüşü zaman içinde stabil."

    return {"label": label, "direction": direction, "series": series, "detail": detail}


def _is_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


#: Price-momentum label -> tier on a [-2, +2] axis for the composite verdict.
_PRICE_TIER = {
    "GÜÇLÜ YUKARI MOMENTUM": 2,
    "YUKARI MOMENTUM": 1,
    "YATAY MOMENTUM": 0,
    "AŞAĞI MOMENTUM": -1,
    "GÜÇLÜ AŞAĞI MOMENTUM": -2,
}

_MOMENTUM_STRONG_POS = "GÜÇLÜ+"


def synthesize_momentum(
    price_momentum: "dict | None",
    fundamental_momentum: "dict | None",
    verdict_momentum: "dict | None",
    fundamental_verdict: "str | None",
) -> "dict | None":
    """Combine the three momentum layers into one context read + cross-signals.

    Price momentum is the main axis; the fundamental label nudges it one notch,
    and the verdict-momentum trajectory a lighter half-notch. The cross-signals
    are the actionable readings from intersecting momentum with the fundamental
    (value) verdict -- e.g. cheap-but-falling (falling knife), expensive-but-
    surging (the profile-guardrail case), cheap-and-accelerating (the strongest
    combination).

    Returns ``None`` only when there is no momentum information at all.
    """
    if price_momentum is None and fundamental_momentum is None and verdict_momentum is None:
        return None

    price_tier = 0
    if isinstance(price_momentum, dict):
        price_tier = _PRICE_TIER.get(price_momentum.get("label"), 0)

    t = float(price_tier)
    if isinstance(fundamental_momentum, dict):
        fl = fundamental_momentum.get("label")
        t += 1 if fl == _LABEL_POSITIVE else (-1 if fl == _LABEL_NEGATIVE else 0)
    if isinstance(verdict_momentum, dict):
        vl = verdict_momentum.get("label")
        t += 0.5 if vl == _LABEL_POSITIVE else (-0.5 if vl == _LABEL_NEGATIVE else 0)

    if t >= 2:
        verdict = _MOMENTUM_STRONG_POS
    elif t >= 0.5:
        verdict = _LABEL_POSITIVE
    elif t > -0.5:
        verdict = _LABEL_NEUTRAL
    else:
        verdict = _LABEL_NEGATIVE

    cross_signals = _cross_signals(price_momentum, fundamental_momentum, price_tier, fundamental_verdict)
    falling_knife = any(c["type"] == "falling_knife" for c in cross_signals)

    return {
        "verdict": verdict,
        "price": price_momentum,
        "fundamental": fundamental_momentum,
        "verdict_trend": verdict_momentum,
        "cross_signals": cross_signals,
        "falling_knife": falling_knife,
    }


def _cross_signals(price_momentum, fundamental_momentum, price_tier, fundamental_verdict):
    """The value x momentum cross-readings, each ``{type, severity, text}``."""
    signals = []
    fv = (fundamental_verdict or "").upper()
    is_cheap = "UCUZ" in fv
    is_expensive = "PAHALI" in fv
    price_dir = price_momentum.get("direction") if isinstance(price_momentum, dict) else None
    fund_label = fundamental_momentum.get("label") if isinstance(fundamental_momentum, dict) else None
    model_beat = (
        isinstance(fundamental_momentum, dict)
        and (fundamental_momentum.get("model_surprise") or {}).get("direction") == "beat"
    )

    # 1. Cheap + falling price momentum -> falling-knife warning.
    if is_cheap and price_dir == "down":
        signals.append({
            "type": "falling_knife",
            "severity": "warn",
            "text": (
                "Fundamental UCUZ ama fiyat momentumu NEGATİF: düşen bıçak riski. "
                "Kademeli giriş dip tranche'larına stabilizasyon koşulu eklendi."
            ),
        })

    # 2. Expensive + strong-up momentum -> profile guardrail.
    if is_expensive and price_tier >= 2:
        signals.append({
            "type": "profile_guardrail",
            "severity": "warn",
            "text": (
                "Fundamental PAHALI ama momentum GÜÇLÜ+: momentum cazibesi yüksek, "
                "değerleme tetiği yok — plan dışı alım riski (profil zaafı)."
            ),
        })

    # 3. Cheap + positive fundamental momentum -> strongest combination.
    if is_cheap and fund_label == _LABEL_POSITIVE:
        extra = " (üst üste model-beat)" if model_beat else ""
        signals.append({
            "type": "strong_combo",
            "severity": "good",
            "text": (
                f"Fundamental UCUZ + fundamental momentum POZİTİF{extra}: en güçlü kombinasyon. "
                "Tranche planını öne çekmek için gerekçe olabilir."
            ),
        })

    return signals


def _build_fundamental_detail(revenue_accel, margin_trend, model_surprise) -> str:
    parts = []
    if revenue_accel and revenue_accel.get("word"):
        # Explicitly "çeyreklik" (quarterly) so it never reads as contradicting
        # the thesis card's *annual* "Yıllık Gelir Büyümesi (YoY)" -- the two
        # legitimately differ (an annual rate can decelerate while the latest
        # quarters reaccelerate). Momentum is deliberately the timelier read.
        caveat = "" if revenue_accel.get("confirmed", True) else " — tek çeyrek, teyit bekliyor"
        parts.append(
            f"Çeyreklik gelir büyümesi {revenue_accel['word']} "
            f"(son çeyrek YoY %{revenue_accel['latest_yoy_pct']:+.1f}{caveat})"
        )
    gross = (margin_trend or {}).get("gross")
    if gross:
        parts.append(f"brüt marj {gross}")
    fcf = (margin_trend or {}).get("fcf")
    if fcf:
        parts.append(f"FCF marjı {fcf}")
    if model_surprise:
        word = {"beat": "modeli aştı", "miss": "modelin altında", "inline": "modelle uyumlu"}[model_surprise["direction"]]
        parts.append(f"gerçekleşen gelir {word} (%{model_surprise['surprise_pct']:+.1f})")
    if not parts:
        return "Fundamental momentum için yeterli çeyreklik veri yok."
    return "; ".join(parts) + "."
