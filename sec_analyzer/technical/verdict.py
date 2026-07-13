"""Rule-based (no-LLM) interpretation of technical indicators.

Mirrors the spirit of :mod:`sec_analyzer.interpret.rule_based`: a fixed,
deterministic, fully auditable set of rules over the numbers computed by
:mod:`sec_analyzer.technical.indicators`, with no network access and no
language model involved. Output text is written in Turkish, matching the
rest of the application's verdict-facing output.
"""

import logging

logger = logging.getLogger(__name__)

#: RSI thresholds for the overbought/oversold verdict rule.
_RSI_OVERBOUGHT = 70
_RSI_OVERSOLD = 30

_VERDICT_OVERBOUGHT = "AŞIRI ALIM"
_VERDICT_OVERSOLD = "AŞIRI SATIM"
_VERDICT_NEUTRAL = "NÖTR"


def _format_signed_pct(value: float) -> str:
    """Format a percentage the Turkish way used across this app's verdicts:
    sign first, then a literal ``%``, then the magnitude, e.g. ``+%12`` for
    +12% or ``-%7`` for -7%."""
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


def _horizon_summary_3m(indicators: dict) -> str:
    """Momentum-framed narrative for a 3-month horizon: RSI, SMA50
    distance, 20d volatility, and 52w range position."""
    rsi14 = indicators.get("rsi14")
    dist_sma50_pct = indicators.get("dist_sma50_pct")
    volatility_20d = indicators.get("volatility_20d")
    range_position_pct = indicators.get("range_position_pct")

    sentences = []

    if rsi14 is None:
        sentences.append("RSI için yeterli fiyat geçmişi bulunmuyor, bu nedenle kısa vadeli momentum sinyali üretilemiyor.")
    else:
        momentum = "güçlü yukarı momentum" if rsi14 > _RSI_OVERBOUGHT else (
            "güçlü aşağı momentum" if rsi14 < _RSI_OVERSOLD else "dengeli momentum"
        )
        sentences.append(f"RSI {rsi14:.1f} ile {momentum} gösteriyor.")

    if dist_sma50_pct is not None:
        yon = "üzerinde" if dist_sma50_pct >= 0 else "altında"
        sentences.append(f"Fiyat, 50 günlük ortalamanın {_format_signed_pct(dist_sma50_pct)} {yon} seyrediyor.")

    if volatility_20d is not None:
        sentences.append(f"Son 20 günlük yıllıklandırılmış volatilite yaklaşık %{volatility_20d * 100:.0f}.")

    if range_position_pct is not None:
        sentences.append(f"Fiyat, 52 haftalık aralığın %{range_position_pct:.0f} seviyesinde konumlanıyor.")

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
        sentences.append("RSI için yeterli fiyat geçmişi bulunmuyor.")
    else:
        sentences.append(f"RSI {rsi14:.1f} seviyesinde.")

    if sma50_above_sma200 is not None:
        konum = "üzerinde (SMA50 > SMA200)" if sma50_above_sma200 else "altında (SMA50 < SMA200)"
        sentences.append(f"50 günlük ortalama, 200 günlük ortalamanın {konum}.")
    if dist_sma200_pct is not None:
        yon = "üzerinde" if dist_sma200_pct >= 0 else "altında"
        sentences.append(f"Fiyat, 200 günlük ortalamanın {_format_signed_pct(dist_sma200_pct)} {yon}.")

    if golden_cross:
        sentences.append("Son 60 işlem gününde bir altın kesişim (golden cross) oluştu.")
    elif death_cross:
        sentences.append("Son 60 işlem gününde bir ölüm kesişimi (death cross) oluştu.")

    if range_position_pct is not None:
        sentences.append(f"Fiyat, 52 haftalık aralığın %{range_position_pct:.0f} seviyesinde.")

    return " ".join(sentences)


def _horizon_summary_5y(indicators: dict) -> str:
    """5-year narrative: explicitly notes RSI is not decision-relevant at
    this horizon, and frames the SMA200 trend only as an entry-timing note."""
    dist_sma200_pct = indicators.get("dist_sma200_pct")
    sma50_above_sma200 = indicators.get("sma50_above_sma200")

    sentences = [
        "5 yıllık ufukta RSI gibi kısa vadeli momentum göstergeleri karar açısından belirleyici değildir."
    ]

    if dist_sma200_pct is not None:
        yon = "üzerinde" if dist_sma200_pct >= 0 else "altında"
        sentences.append(
            f"Fiyat şu anda SMA200'ün {_format_signed_pct(dist_sma200_pct)} {yon}; "
            "bu, uzun vadeli tez için bir sinyal değil, yalnızca olası bir giriş "
            "zamanlaması notu olarak değerlendirilmelidir."
        )
    elif sma50_above_sma200 is not None:
        konum = "üzerinde" if sma50_above_sma200 else "altında"
        sentences.append(
            f"SMA50 şu anda SMA200'ün {konum}; bu da uzun vadeli tez için bir "
            "sinyal değil, yalnızca olası bir giriş zamanlaması notudur."
        )
    else:
        sentences.append(
            "SMA200 için yeterli fiyat geçmişi bulunmuyor, bu nedenle giriş "
            "zamanlaması için bir teknik referans üretilemiyor."
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


def technical_verdict(indicators: dict, horizon: str = "1y") -> dict:
    """Derive a rule-based technical verdict from a computed indicator set.

    Verdict rule (deterministic, no exceptions):

    * RSI > 70 **and** price > SMA50 -> ``"AŞIRI ALIM"`` (overbought).
    * RSI < 30 -> ``"AŞIRI SATIM"`` (oversold).
    * Otherwise -> ``"NÖTR"`` (neutral).
    * If ``rsi14`` is missing (insufficient price history) -> ``"NÖTR"``
      with ``verdict_detail == "yetersiz veri"``.

    Args:
        indicators: The dict returned by
            :func:`sec_analyzer.technical.indicators.compute_indicators`.
        horizon: One of ``"3m"``, ``"1y"``, ``"5y"``. Controls only the
            narrative framing of ``horizon_summary``, not the verdict rule
            itself. Unrecognized values fall back to the ``"1y"`` framing.

    Returns:
        A dict with exactly these keys: ``verdict``, ``verdict_detail``,
        ``horizon_summary``, ``horizon``.
    """
    rsi14 = indicators.get("rsi14")

    if rsi14 is None:
        return {
            "verdict": _VERDICT_NEUTRAL,
            "verdict_detail": "yetersiz veri",
            "horizon_summary": _horizon_summary(indicators, horizon),
            "horizon": horizon,
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
    }
