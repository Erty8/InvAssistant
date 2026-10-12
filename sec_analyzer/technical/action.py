"""Single technical *state* read, folded from the indicator set.

The technical layer produces many overlapping reads -- the RSI verdict, the
trend regime, the composite momentum score, the reversal checklist, support/
resistance levels. This module folds them into one plain-language state
("UPTREND — STRETCHED", "DOWNTREND — NO TURN", ...) with the key levels and
reasons, deterministically, from numbers that are already computed.

It is a description, not a signal: the pre-registered validation study
(``backtest/ACTION_STUDY_SPEC.md`` §6) found no predictive edge for forward
returns, so the display label and ``evidence_note`` say so. It answers three
questions in order:

1. **Trend** -- where is price relative to SMA50/SMA200 (the same regime the
   report's Technical card shows)?
2. **Fuel** -- in an uptrend, how many independent exhaustion signs have
   fired (extended far above SMA50, overbought RSI, bearish divergence,
   slowing momentum, distribution, buying climax)? Few signs means the move
   still has room; several means it is stretched.
3. **Turn** -- outside an uptrend, how many independent bullish reversal
   confirmations are present (the same checklist the report's "Reversal
   Confirmation" block uses, with the same horizon tuning)?

The answer is one state label plus a short read of the entry picture and the
holding picture, the key trigger/stop levels, and the reasons behind it, so
the label is never a black box.

Design, mirroring the rest of the technical layer: pure, None-safe,
deterministic, never raises (returns ``None`` when there isn't enough data).
This is timing context only -- it never feeds the valuation path.
"""

import logging

from sec_analyzer.technical.swing import _trade_levels

logger = logging.getLogger(__name__)

# --- Internal state codes (stable for the backtest; never shown to the user) --
ACTION_BUY = "BUY"
ACTION_HOLD = "HOLD"
ACTION_WAIT_PULLBACK = "WAIT FOR PULLBACK"
ACTION_WAIT = "WAIT"
ACTION_REDUCE = "REDUCE"
ACTION_AVOID = "AVOID"

_TONE = {
    ACTION_BUY: "green",
    ACTION_HOLD: "green",
    ACTION_WAIT_PULLBACK: "amber",
    ACTION_WAIT: "amber",
    ACTION_REDUCE: "red",
    ACTION_AVOID: "red",
}

#: Display colour follows the trend STATE, not an implied trade: a turn
#: forming inside a downtrend is amber, not green.
_LABEL_TONE = {
    "UPTREND — ROOM TO RUN": "green",
    "UPTREND — MOMENTUM FADING": "amber",
    "UPTREND — STRETCHED": "amber",
    "PULLBACK — BOUNCE CONFIRMING": "green",
    "PULLBACK — NO BOUNCE YET": "amber",
    "UPTREND BREAKING DOWN": "red",
    "RECOVERY ATTEMPT": "amber",
    "DOWNTREND — TURN CONFIRMING": "amber",
    "DOWNTREND — SELLING EXHAUSTED": "red",
    "DOWNTREND — EARLY TURN SIGNAL": "red",
    "DOWNTREND — NO TURN": "red",
}

#: Result of the pre-registered validation study (ACTION_STUDY_SPEC.md §6),
#: shown next to every read so the state is never mistaken for a signal.
EVIDENCE_NOTE = (
    "Backtest (126 S&P 500 names, 2015–2026): these states showed no predictive edge for "
    "10–63 day returns. Use them to read trend and risk levels, not as buy/sell signals."
)

# --- Exhaustion ("fuel") thresholds -----------------------------------------
#: Price this many ATRs above SMA50 reads as extended (stretched away from
#: its trend line); the % fallback applies when ATR is unavailable.
_EXTENDED_ATR_MULT = 3.0
_EXTENDED_FALLBACK_PCT = 15.0
#: RSI level that counts as overbought (same line as technical.verdict).
_RSI_OVERBOUGHT = 70.0
#: A buying climax only counts as exhaustion when it is this recent.
_CLIMAX_MAX_BARS_AGO = 5
#: Fired exhaustion signs at/above which fuel reads "low" (move stretched).
_FUEL_LOW_MIN = 2

# --- Reversal confirmation tuning, mirroring the report's
# swingHorizonTuning() so the Python action and the report's checklist agree.
_REVERSAL_TUNING = {
    "3m": {"vol_min": 1.3, "confirm": 1},
    "1y": {"vol_min": 1.5, "confirm": 2},
    "5y": {"vol_min": 1.8, "confirm": 3},
}


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _money(value) -> str:
    return f"${value:,.2f}"


def _check(key: str, label: str, fired: bool) -> dict:
    return {"key": key, "label": label, "fired": bool(fired)}


def _regime(ind: dict) -> "str | None":
    """``"up"`` (above both SMAs), ``"down"`` (below both), ``"transition"``
    (between them), or ``None`` without price/SMA data. Identical rule to the
    report's trend-regime badge."""
    price, s50, s200 = ind.get("price"), ind.get("sma50"), ind.get("sma200")
    if not _is_num(price):
        return None
    above50 = price >= s50 if _is_num(s50) else None
    above200 = price >= s200 if _is_num(s200) else None
    if above50 is None and above200 is None:
        return None
    if above50 is True and above200 is True:
        return "up"
    if above50 is False and above200 is False:
        return "down"
    return "transition"


def _fuel_checks(ind: dict, momentum: "dict | None") -> "list[dict]":
    """Exhaustion signs for an uptrend. A fired check means the move is
    getting stretched; the count drives the fuel level."""
    checks = []
    price, s50, atr = ind.get("price"), ind.get("sma50"), ind.get("atr14")
    if _is_num(price) and _is_num(s50) and _is_num(atr) and atr > 0:
        ext_atr = (price - s50) / atr
        checks.append(_check(
            "extended",
            f"Price {ext_atr:.1f}× ATR above SMA50 (stretched at ≥{_EXTENDED_ATR_MULT:.0f}×)",
            ext_atr >= _EXTENDED_ATR_MULT,
        ))
    elif _is_num(ind.get("dist_sma50_pct")):
        d = float(ind["dist_sma50_pct"])
        checks.append(_check(
            "extended",
            f"Price {d:+.1f}% vs SMA50 (stretched at ≥{_EXTENDED_FALLBACK_PCT:.0f}%)",
            d >= _EXTENDED_FALLBACK_PCT,
        ))

    rsi = ind.get("rsi14")
    if _is_num(rsi):
        checks.append(_check("rsi_overbought", f"RSI {rsi:.0f} (overbought above {_RSI_OVERBOUGHT:.0f})",
                             rsi > _RSI_OVERBOUGHT))

    checks.append(_check("bearish_divergence", "Bearish RSI divergence (new price high, RSI lower)",
                         ind.get("rsi_divergence") == "bearish"))

    accel = momentum.get("accel") if isinstance(momentum, dict) else None
    if accel:
        checks.append(_check("momentum_slowing", f"Momentum {accel}", accel == "slowing"))

    obv = ind.get("obv_trend")
    if obv:
        checks.append(_check("distribution", "OBV distribution (money flowing out)", obv == "down"))

    climax = ind.get("volume_climax")
    climax_fired = (
        isinstance(climax, dict) and climax.get("detected") and climax.get("direction") == "up"
        and _is_num(climax.get("bars_ago")) and climax["bars_ago"] <= _CLIMAX_MAX_BARS_AGO
    )
    checks.append(_check("buying_climax", "Buying climax (huge-volume up bar, last 5 sessions)", climax_fired))
    return checks


def _reversal_checks(ind: dict, horizon: str) -> "list[dict]":
    """Bullish reversal confirmations -- the same rows the report's
    "Reversal Confirmation" checklist counts for a falling trend."""
    tuning = _REVERSAL_TUNING.get(horizon, _REVERSAL_TUNING["1y"])
    checks = [
        _check("rsi_reclaim", "RSI reclaimed 30 after being oversold", ind.get("rsi_reclaim") == "bullish"),
        _check("macd_cross", "Fresh bullish MACD crossover", ind.get("macd_cross") == "bullish"),
        _check("obv_accumulation", "OBV accumulation (money flowing in)", ind.get("obv_trend") == "up"),
    ]
    rv = ind.get("rel_volume")
    if _is_num(rv):
        checks.append(_check("volume", f"Volume {rv:.1f}× average (participation at ≥{tuning['vol_min']}×)",
                             rv >= tuning["vol_min"]))
    climax = ind.get("volume_climax")
    if isinstance(climax, dict) and climax.get("detected") and climax.get("direction") == "down":
        checks.append(_check("capitulation", "Capitulation bar (selling exhaustion)", True))
    if ind.get("rsi_divergence") == "bullish":
        checks.append(_check("bullish_divergence", "Bullish RSI divergence (new price low, RSI higher)", True))
    return checks


#: Reversal checks that reflect price/oscillator actually turning. Volume,
#: OBV and capitulation show exhaustion or participation, not a turn, so a
#: BUY on a reversal needs at least one of these.
_PRICE_TURN_KEYS = frozenset({"rsi_reclaim", "macd_cross", "bullish_divergence"})


def _fuel_level(fired: int) -> "tuple[str, str]":
    if fired == 0:
        return "strong", "Plenty of room — no exhaustion signs"
    if fired < _FUEL_LOW_MIN:
        return "moderate", "Some room — one exhaustion sign"
    return "low", f"Stretched — {fired} exhaustion signs"


def compute_technical_action(indicators: "dict | None", horizon: str = "1y") -> "dict | None":
    """Fold the technical indicators into one actionable read.

    Args:
        indicators: The merged technical dict (``compute_indicators`` output
            plus ``momentum`` from
            :func:`sec_analyzer.technical.momentum.compute_price_momentum`).
        horizon: ``"3m"`` / ``"1y"`` / ``"5y"``; only tunes how many reversal
            confirmations are required and adds a timing-only caveat at 5y.

    Returns:
        ``None`` if the trend regime can't be determined, otherwise a dict:

        * ``action``: internal state code, one of BUY / HOLD / WAIT FOR
          PULLBACK / WAIT / REDUCE / AVOID. Stable for the backtest; NOT
          display text -- the validation study found no predictive edge.
        * ``label``: the display label, a trend-state description such as
          "UPTREND — STRETCHED"; ``tone``: ``"green"`` / ``"amber"`` /
          ``"red"`` following that state.
        * ``evidence_note``: the backtest caveat to show with the read.
        * ``headline``: one-line state, e.g. "Uptrend intact — momentum has room".
        * ``regime``: ``"up"`` / ``"down"`` / ``"transition"``.
        * ``for_new`` / ``for_holder``: guidance with and without a position.
        * ``fuel``: ``{level, label, fired, total, checks}`` in an uptrend,
          else ``None``.
        * ``reversal``: ``{confirmed, needed, total, checks}`` outside an
          uptrend, else ``None``.
        * ``trigger`` / ``stop``: key price levels (``None`` if unavailable);
          ``trigger_label`` says what the trigger level means.
        * ``reasons``: short strings explaining the call.
        * ``horizon_note``: timing caveat for the 5y horizon, else ``None``.
    """
    if not isinstance(indicators, dict):
        return None
    try:
        return _compute(indicators, horizon)
    except Exception:  # noqa: BLE001 - analysis layer must never crash the caller
        logger.warning("compute_technical_action failed", exc_info=True)
        return None


def _compute(ind: dict, horizon: str) -> "dict | None":
    regime = _regime(ind)
    if regime is None:
        return None

    momentum = ind.get("momentum") if isinstance(ind.get("momentum"), dict) else None
    mom_dir = momentum.get("direction") if momentum else None
    mom_score = momentum.get("score") if momentum else None

    price = float(ind["price"])
    s50 = float(ind["sma50"]) if _is_num(ind.get("sma50")) else None
    s200 = float(ind["sma200"]) if _is_num(ind.get("sma200")) else None
    res = float(ind["nearest_resistance"]) if _is_num(ind.get("nearest_resistance")) else None
    macd_hist = ind.get("macd_hist")
    levels = _trade_levels(ind)
    stop = levels.get("stop")

    reasons = []
    if mom_score is not None and momentum.get("label"):
        reasons.append(f"Momentum {mom_score}/100 ({momentum['label'].lower()})")

    fuel = None
    reversal = None
    trigger = None
    trigger_label = None

    if regime == "up":
        reasons.insert(0, "Price above SMA50 and SMA200 (uptrend)")
        checks = _fuel_checks(ind, momentum)
        fired = sum(1 for c in checks if c["fired"])
        level, fuel_label = _fuel_level(fired)
        fuel = {"level": level, "label": fuel_label, "fired": fired, "total": len(checks), "checks": checks}
        reasons.append(f"Exhaustion signs: {fired}/{len(checks)}")
        trigger = s50
        trigger_label = "Trend support (SMA50)"
        stop_txt = f"a stop near {_money(stop)}" if stop is not None else "a stop below SMA50"
        s50_txt = f"SMA50 ({_money(s50)})" if s50 is not None else "SMA50"
        if level == "low":
            action = ACTION_WAIT_PULLBACK
            label = "UPTREND — STRETCHED"
            headline = "Uptrend, but price is far from its trend line"
            for_new = (f"An entry here sits far above {s50_txt}, so a trend-based stop is wide; "
                       "a pullback would shorten it.")
            for_holder = f"Trend intact; {stop_txt} marks where the move would be in question."
        elif mom_dir == "up":
            action = ACTION_BUY
            label = "UPTREND — ROOM TO RUN"
            headline = "Uptrend intact, momentum rising, few exhaustion signs"
            for_new = f"Trend and momentum agree; risk can be defined with {stop_txt}."
            for_holder = f"Trend intact; {stop_txt} is the level to trail."
        else:
            action = ACTION_HOLD
            label = "UPTREND — MOMENTUM FADING"
            headline = "Still an uptrend, but momentum has flattened"
            for_new = "Trend is up but there is no fresh push behind it."
            for_holder = f"The trend holds while price stays above {s50_txt}."
    else:
        tuning = _REVERSAL_TUNING.get(horizon, _REVERSAL_TUNING["1y"])
        checks = _reversal_checks(ind, horizon)
        confirmed = sum(1 for c in checks if c["fired"])
        needed = tuning["confirm"]
        price_turn = any(c["fired"] and c["key"] in _PRICE_TURN_KEYS for c in checks)
        turn_confirmed = confirmed >= needed and price_turn
        reversal = {"confirmed": confirmed, "needed": needed, "total": len(checks),
                    "price_turn": price_turn, "checks": checks}
        reasons.append(f"Reversal confirmations: {confirmed}/{len(checks)} (need {needed})")
        if confirmed >= needed and not price_turn:
            reasons.append("No price turn yet (RSI/MACD/divergence) — volume alone shows exhaustion, not a reversal")

        if regime == "down":
            reasons.insert(0, "Price below SMA50 and SMA200 (downtrend)")
            trigger = res if res is not None else s50
            trigger_label = "Reversal trigger (close above)"
            trig_txt = f"a daily close above {_money(trigger)}" if trigger is not None else "a close back above resistance"
            stop_note = f" A stop near {_money(stop)} defines the risk." if stop is not None else ""
            if turn_confirmed:
                action = ACTION_BUY
                label = "DOWNTREND — TURN CONFIRMING"
                headline = "Downtrend, but reversal signals are lining up"
                for_new = f"Still below both averages, so any entry is against the trend.{stop_note}"
                for_holder = f"{trig_txt[0].upper() + trig_txt[1:]} would complete the turn."
            elif confirmed >= needed:
                action = ACTION_WAIT
                label = "DOWNTREND — SELLING EXHAUSTED"
                headline = "Downtrend — selling looks exhausted, but price hasn't turned"
                for_new = f"No RSI/MACD turn yet; {trig_txt} would be the first price confirmation."
                for_holder = "Exhaustion without a price turn often precedes further chop or lows."
            elif confirmed > 0:
                action = ACTION_WAIT
                label = "DOWNTREND — EARLY TURN SIGNAL"
                headline = "Downtrend with an early reversal signal"
                for_new = f"One signal is not a turn; {trig_txt} would add confirmation."
                for_holder = "The downtrend is still in place; the turn is unconfirmed."
            else:
                action = ACTION_AVOID
                label = "DOWNTREND — NO TURN"
                headline = "Downtrend — no signs of a turn"
                for_new = "No reversal evidence; entries here are against an intact downtrend."
                for_holder = "The downtrend is intact; review the position against your stop."
        else:
            # Transition: price between the two averages.
            below50 = s50 is not None and price < s50
            if s200 is None:
                reasons.insert(0, "Not enough history for SMA200; trend read from SMA50 only")
            if below50:
                if s200 is not None:
                    reasons.insert(0, "Price below SMA50 but above SMA200 (pullback in a longer uptrend)")
                trigger = s50
                trigger_label = "Trend repair (close above SMA50)"
                s200_txt = f"SMA200 ({_money(s200)})" if s200 is not None else "the longer-term average"
                if turn_confirmed and mom_dir != "down":
                    action = ACTION_BUY
                    label = "PULLBACK — BOUNCE CONFIRMING"
                    headline = "Pullback in an uptrend, bounce signals lining up"
                    for_new = f"Pullback within a longer uptrend; {s200_txt} is the structural stop."
                    for_holder = f"The longer-term uptrend holds while price stays above {s200_txt}."
                elif mom_dir == "down" and _is_num(macd_hist) and macd_hist < 0:
                    action = ACTION_REDUCE
                    label = "UPTREND BREAKING DOWN"
                    headline = "Price lost SMA50 with falling momentum"
                    for_new = f"A close back above SMA50 ({_money(s50)}) would repair the trend."
                    for_holder = f"A close below {s200_txt} would end the long-term uptrend."
                else:
                    action = ACTION_WAIT
                    label = "PULLBACK — NO BOUNCE YET"
                    headline = "Pullback in an uptrend, no bounce yet"
                    for_new = f"No bounce signal yet; a close back above SMA50 ({_money(s50)}) would show one."
                    for_holder = f"The longer-term uptrend holds while price stays above {s200_txt}."
            else:
                if s200 is not None:
                    reasons.insert(0, "Price above SMA50 but below SMA200 (recovery attempt)")
                trigger = s200
                trigger_label = "Uptrend confirmation (close above SMA200)"
                trig_txt = f"a daily close above SMA200 ({_money(s200)})" if s200 is not None else "a close above SMA200"
                action = ACTION_WAIT
                label = "RECOVERY ATTEMPT"
                if mom_dir == "up":
                    headline = "Recovery attempt with rising momentum"
                else:
                    headline = "Recovery attempt, momentum unconvincing"
                for_new = f"Not yet a confirmed uptrend; {trig_txt} would confirm one."
                for_holder = (f"The recovery holds while price stays above SMA50 ({_money(s50)})." if s50 is not None
                              else "The recovery is unconfirmed.")

    horizon_note = None
    if horizon == "5y":
        horizon_note = ("At a 5-year horizon this is only an entry-timing aid; the long-term "
                        "decision rests on the valuation.")

    return {
        "action": action,
        "label": label,
        "tone": _LABEL_TONE.get(label, _TONE[action]),
        "headline": headline,
        "evidence_note": EVIDENCE_NOTE,
        "regime": regime,
        "for_new": for_new,
        "for_holder": for_holder,
        "fuel": fuel,
        "reversal": reversal,
        "trigger": round(trigger, 2) if trigger is not None else None,
        "trigger_label": trigger_label if trigger is not None else None,
        "stop": stop,
        "reasons": reasons,
        "horizon_note": horizon_note,
    }
