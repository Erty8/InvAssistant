"""Rule-based (no-LLM) interpretation of technical indicators.

Mirrors the spirit of :mod:`sec_analyzer.interpret.rule_based`: a fixed,
deterministic, fully auditable set of rules over the numbers computed by
:mod:`sec_analyzer.technical.indicators`, with no network access and no
language model involved. Output text is written in English, matching the
rest of the application's verdict-facing output.
"""

import logging

logger = logging.getLogger(__name__)

#: RSI thresholds for the overbought/oversold verdict rule.
_RSI_OVERBOUGHT = 70
_RSI_OVERSOLD = 30

_VERDICT_OVERBOUGHT = "OVERBOUGHT"
_VERDICT_OVERSOLD = "OVERSOLD"
_VERDICT_NEUTRAL = "NEUTRAL"


def _format_signed_pct(value: float) -> str:
    """Format a percentage the way used across this app's verdicts: sign
    first, then a literal ``%``, then the magnitude, e.g. ``+%12`` for +12%
    or ``-%7`` for -7%."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}%{abs(value):.0f}"


def _verdict_detail(indicators: dict, verdict: str) -> str:
    """Build the compact ``verdict_detail`` string, e.g. ``"RSI 74, SMA50 +%12"``."""
    rsi14 = indicators.get("rsi14")
    parts = [f"RSI {rsi14:.0f}"]

    dist_sma50_pct = indicators.get("dist_sma50_pct")
    if dist_sma50_pct is not None:
        parts.append(f"SMA50 {_format_signed_pct(dist_sma50_pct)}")

    return ", ".join(parts)


def _momentum_sentence(indicators: dict) -> "str | None":
    """One-line momentum lead built from the composite score
    (``indicators['momentum']``, attached by the CLI/web layer), or ``None``
    if the score isn't available. Keeps the narrative in sync with the report's
    MOMENTUM row without re-deriving the score here."""
    momentum = indicators.get("momentum")
    if not isinstance(momentum, dict):
        return None
    label = momentum.get("label")
    score = momentum.get("score")
    if not label or score is None:
        return None
    accel = momentum.get("accel")
    tail = f", momentum {accel}" if accel else ""
    return f"Composite momentum {score}/100 ({label.lower()}){tail}."


def _horizon_summary_3m(indicators: dict) -> str:
    """Momentum-framed narrative for a 3-month horizon: the composite momentum
    score leads (when available), followed by RSI, SMA50 distance, 20d
    volatility, and 52w range position."""
    rsi14 = indicators.get("rsi14")
    dist_sma50_pct = indicators.get("dist_sma50_pct")
    volatility_20d = indicators.get("volatility_20d")
    range_position_pct = indicators.get("range_position_pct")

    sentences = []

    momentum_lead = _momentum_sentence(indicators)
    if momentum_lead:
        sentences.append(momentum_lead)

    if rsi14 is None:
        sentences.append("There isn't enough price history for RSI, so a short-term momentum signal cannot be generated.")
    else:
        momentum = "strong upward momentum" if rsi14 > _RSI_OVERBOUGHT else (
            "strong downward momentum" if rsi14 < _RSI_OVERSOLD else "balanced momentum"
        )
        sentences.append(f"RSI at {rsi14:.1f} shows {momentum}.")

    if dist_sma50_pct is not None:
        direction = "above" if dist_sma50_pct >= 0 else "below"
        sentences.append(f"Price is trading {_format_signed_pct(dist_sma50_pct)} {direction} the 50-day average.")

    if volatility_20d is not None:
        sentences.append(f"The latest 20-day annualized volatility is approximately %{volatility_20d * 100:.0f}.")

    if range_position_pct is not None:
        sentences.append(f"Price is positioned at the %{range_position_pct:.0f} level of the 52-week range.")

    return " ".join(sentences)


def _horizon_summary_1y(indicators: dict) -> str:
    """Balanced narrative for a 1-year horizon: RSI, both SMAs,
    golden/death cross, and 52w range position."""
    rsi14 = indicators.get("rsi14")
    sma50_above_sma200 = indicators.get("sma50_above_sma200")
    dist_sma200_pct = indicators.get("dist_sma200_pct")
    golden_cross = indicators.get("golden_cross")
    death_cross = indicators.get("death_cross")
    range_position_pct = indicators.get("range_position_pct")

    sentences = []

    if rsi14 is None:
        sentences.append("There isn't enough price history for RSI.")
    else:
        sentences.append(f"RSI is at {rsi14:.1f}.")

    if sma50_above_sma200 is not None:
        state = "above (SMA50 > SMA200)" if sma50_above_sma200 else "below (SMA50 < SMA200)"
        sentences.append(f"The 50-day average is {state} the 200-day average.")
    if dist_sma200_pct is not None:
        direction = "above" if dist_sma200_pct >= 0 else "below"
        sentences.append(f"Price is {_format_signed_pct(dist_sma200_pct)} {direction} the 200-day average.")

    if golden_cross:
        sentences.append("A golden cross occurred in the last 60 trading days.")
    elif death_cross:
        sentences.append("A death cross occurred in the last 60 trading days.")

    if range_position_pct is not None:
        sentences.append(f"Price is at the %{range_position_pct:.0f} level of the 52-week range.")

    return " ".join(sentences)


def _horizon_summary_5y(indicators: dict) -> str:
    """5-year narrative: explicitly notes RSI is not decision-relevant at
    this horizon, and frames the SMA200 trend only as an entry-timing note."""
    dist_sma200_pct = indicators.get("dist_sma200_pct")
    sma50_above_sma200 = indicators.get("sma50_above_sma200")

    sentences = [
        "At a 5-year horizon, short-term momentum indicators like RSI are not decisive for the decision."
    ]

    momentum_note = _momentum_sentence(indicators)
    if momentum_note:
        sentences.append(momentum_note + " This is only a timing footnote for the long-term thesis.")

    if dist_sma200_pct is not None:
        direction = "above" if dist_sma200_pct >= 0 else "below"
        sentences.append(
            f"Price is currently {_format_signed_pct(dist_sma200_pct)} {direction} SMA200; "
            "this should be treated not as a signal for the long-term thesis but only as "
            "a possible entry-timing note."
        )
    elif sma50_above_sma200 is not None:
        state = "above" if sma50_above_sma200 else "below"
        sentences.append(
            f"SMA50 is currently {state} SMA200; this too is not a signal for the "
            "long-term thesis, only a possible entry-timing note."
        )
    else:
        sentences.append(
            "There isn't enough price history for SMA200, so a technical reference "
            "for entry timing cannot be generated."
        )

    return " ".join(sentences)


def _horizon_summary(indicators: dict, horizon: str) -> str:
    """Dispatch to the horizon-specific narrative builder."""
    if horizon == "3m":
        return _horizon_summary_3m(indicators)
    if horizon == "5y":
        return _horizon_summary_5y(indicators)
    # Default / "1y": balanced view.
    return _horizon_summary_1y(indicators)


def _entry(role: str, key: str, label_tr: str, reason_tr: str) -> dict:
    """One ``method_summary`` row (see :func:`_build_method_summary`)."""
    return {"role": role, "key": key, "label_tr": label_tr, "reason_tr": reason_tr}


def _build_method_summary(indicators: dict, horizon: str) -> list:
    """Build the ``method_summary`` output: a purely additive, packaging-only
    explanation of WHICH technical indicators/methods :func:`technical_verdict`
    leaned on for this ``horizon`` and WHY -- the technical-analysis analog of
    :func:`sec_analyzer.valuation.engine._build_method_summary`, for the same
    "Valuation Methods Used"-style educational card on the report.

    This function introduces NO new thresholds and re-derives nothing: the
    headline choice below is exactly :func:`_horizon_summary`'s own dispatch
    (mirrored, not recomputed), and every other row only checks whether an
    already-computed indicator is present on ``indicators``. It is pure
    packaging of decisions made elsewhere, never a second decision-maker.

    Args:
        indicators: The dict returned by
            :func:`sec_analyzer.technical.indicators.compute_indicators`,
            merged with ``momentum`` / ``relative_strength`` /
            ``relative_strength_sector`` / ``bb_squeeze`` as assembled by the
            caller (mirrors :func:`technical_verdict`'s own ``indicators``
            contract).
        horizon: One of ``"3m"``, ``"1y"``, ``"5y"``; anything else falls
            back to the ``"1y"`` framing, exactly like ``_horizon_summary``.

    Returns:
        A list of ``{"role", "key", "label_tr", "reason_tr"}`` dicts: exactly
        one ``"headline"`` entry first (the horizon-selected framing), then a
        ``"cross_check"`` entry for the composite momentum when it was
        computed, then ``"advisory"`` entries for whichever secondary
        indicators (MA cross, relative strength, sector-relative strength,
        Bollinger squeeze) are actually present. Never raises -- degrades to
        ``[]`` on any unexpected error, per this module's discipline.
    """
    try:
        summary: list = []

        # --- Headline (exactly one), mirroring _horizon_summary's dispatch --
        if horizon == "3m":
            summary.append(_entry(
                "headline", "short_horizon_momentum", "Short-Term Momentum Indicators",
                "At the 3-month horizon, composite price momentum, RSI (overbought/"
                "oversold), distance from SMA50, short-term (20-day) volatility, and "
                "52-week range position are the most decisive indicators for the decision.",
            ))
        elif horizon == "5y":
            summary.append(_entry(
                "headline", "long_horizon_entry_timing", "Long-Term Entry Timing",
                "At the 5-year horizon, short-term oscillators like RSI are NOT decisive "
                "for the decision; only the price's distance from SMA200 is used, not as "
                "a signal but as a possible entry-timing footnote.",
            ))
        else:
            # Default / "1y": balanced view.
            summary.append(_entry(
                "headline", "medium_horizon_trend", "Medium-Term Trend Indicators",
                "At the 1-year horizon, RSI, the SMA50-SMA200 relationship (including "
                "golden/death cross), price's distance from SMA200, and 52-week range "
                "position provide a balanced trend reading.",
            ))

        # --- Always-on cross-check: composite momentum, when computed -------
        if isinstance(indicators.get("momentum"), dict):
            summary.append(_entry(
                "cross_check", "composite_momentum", "Composite Price Momentum",
                "Provides a cross-check that confirms the horizon-specific reading by "
                "combining 5 weighted components -- returns, relative strength, trend "
                "quality, volume confirmation, and oscillator -- into a single "
                "independent score.",
            ))

        # --- Advisory context, only when the underlying indicator fired -----
        if indicators.get("golden_cross") is True or indicators.get("death_cross") is True:
            summary.append(_entry(
                "advisory", "ma_cross", "Golden/Death Cross",
                "SMA50 crossing above (golden) or below (death) SMA200 is a classic "
                "long-term trend-reversal signal; when triggered, it is shown for "
                "informational purposes only.",
            ))

        if isinstance(indicators.get("relative_strength"), dict):
            summary.append(_entry(
                "advisory", "relative_strength", "Relative Strength (vs. SPY)",
                "Shows the stock's relative performance against the broad market (SPY); "
                "it is contextual information and does not determine the verdict on its own.",
            ))

        if isinstance(indicators.get("relative_strength_sector"), dict):
            summary.append(_entry(
                "advisory", "relative_strength_sector", "Sector-Relative Strength",
                "Shows the stock's relative performance against its own sector ETF; a "
                "narrower and more relevant contextual measure than the comparison "
                "against SPY.",
            ))

        bb_squeeze = indicators.get("bb_squeeze")
        if isinstance(bb_squeeze, dict) and bb_squeeze.get("active") is True:
            summary.append(_entry(
                "advisory", "bb_squeeze", "Bollinger Squeeze",
                "A volatility squeeze usually precedes an expansion move; it doesn't "
                "indicate direction, so it's shown for informational purposes only.",
            ))

        return summary
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("_build_method_summary() failed unexpectedly; returning an empty list.")
        return []


def technical_verdict(indicators: dict, horizon: str = "1y") -> dict:
    """Derive a rule-based technical verdict from a computed indicator set.

    Verdict rule (deterministic, no exceptions):

    * RSI > 70 **and** price > SMA50 -> ``"OVERBOUGHT"``.
    * RSI < 30 -> ``"OVERSOLD"``.
    * Otherwise -> ``"NEUTRAL"``.
    * If ``rsi14`` is missing (insufficient price history) -> ``"NEUTRAL"``
      with ``verdict_detail == "insufficient data"``.

    Args:
        indicators: The dict returned by
            :func:`sec_analyzer.technical.indicators.compute_indicators`.
        horizon: One of ``"3m"``, ``"1y"``, ``"5y"``. Controls only the
            narrative framing of ``horizon_summary``, not the verdict rule
            itself. Unrecognized values fall back to the ``"1y"`` framing.

    Returns:
        A dict with exactly these keys: ``verdict``, ``verdict_detail``,
        ``horizon_summary``, ``horizon``, ``method_summary``. ``method_summary``
        is a list of ``{"role", "key", "label_tr", "reason_tr"}`` dicts (see
        :func:`_build_method_summary`) explaining which technical
        indicators/methods were used for this horizon and why -- purely
        additive packaging over the same decisions ``horizon_summary`` already
        encodes in prose; degrades to ``[]`` rather than raising.
    """
    rsi14 = indicators.get("rsi14")

    if rsi14 is None:
        return {
            "verdict": _VERDICT_NEUTRAL,
            "verdict_detail": "insufficient data",
            "horizon_summary": _horizon_summary(indicators, horizon),
            "horizon": horizon,
            "method_summary": _build_method_summary(indicators, horizon),
        }

    price = indicators.get("price")
    sma50 = indicators.get("sma50")

    if rsi14 > _RSI_OVERBOUGHT and price is not None and sma50 is not None and price > sma50:
        verdict = _VERDICT_OVERBOUGHT
    elif rsi14 < _RSI_OVERSOLD:
        verdict = _VERDICT_OVERSOLD
    else:
        verdict = _VERDICT_NEUTRAL

    return {
        "verdict": verdict,
        "verdict_detail": _verdict_detail(indicators, verdict),
        "horizon_summary": _horizon_summary(indicators, horizon),
        "horizon": horizon,
        "method_summary": _build_method_summary(indicators, horizon),
    }
