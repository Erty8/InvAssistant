"""Combine DCF, reverse-DCF, and multiples signals into one triangulated view.

Each of the three methods independently votes "ucuz" (cheap) / "makul"
(fair) / "pahali" (expensive) / "veri_yok" (no data); this module doesn't
recompute any valuation number, it only classifies each method's existing
output and looks for agreement across methods as a simple, transparent
confidence signal (rather than a black-box weighted score).

In hyper-grower mode (HYPER_SPEC.md Sec.4, ``hyper_growth=True``) the DCF
vote gains a 4th value, "yuksek_beklenti" ("priced for high expectations"):
above the base band but at/below the bull band, distinct from an outright
"pahali" (only above the bull band).
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SIGNAL_CHEAP = "ucuz"
SIGNAL_FAIR = "makul"
SIGNAL_EXPENSIVE = "pahali"
SIGNAL_NO_DATA = "veri_yok"

#: Multiples-only signal: the raw multiple's own historical percentile and
#: the growth-adjusted multiple's percentile (PEG in standard mode,
#: growth-adjusted EV/Sales in hyper-grower mode) land in DIFFERENT
#: directional buckets (e.g. raw expensive, growth-adjusted fair). Surfaces
#: the disagreement instead of hiding it behind the raw percentile alone
#: (VALUATION.md Sec.7). Never emitted unless BOTH percentiles are present.
SIGNAL_MIXED = "karisik"

#: Hyper-grower-only DCF signal (HYPER_SPEC.md Sec.4): price sits above the
#: base band but at/below the bull band -- "priced for high expectations",
#: distinct from an outright "pahali" (only above the bull band).
SIGNAL_HIGH_EXPECTATION = "yuksek_beklenti"

#: Reverse-DCF: implied growth more than this many percentage points above/
#: below the reference growth rate flips the signal to expensive/cheap.
_REVERSE_DCF_MARGIN = 0.03

#: Multiples percentile thresholds.
_PERCENTILE_EXPENSIVE = 70
_PERCENTILE_CHEAP = 30

CONFIDENCE_HIGH = "YÜKSEK"
CONFIDENCE_MEDIUM = "ORTA"
CONFIDENCE_LOW = "DÜŞÜK"

_DIRECTION_UNCLEAR = "belirsiz"

#: Turkish display labels for each signal value, used in rationale sentences.
_SIGNAL_LABEL_TR = {
    SIGNAL_CHEAP: "ucuz",
    SIGNAL_FAIR: "makul",
    SIGNAL_EXPENSIVE: "pahalı",
    SIGNAL_HIGH_EXPECTATION: "yüksek beklenti",
    SIGNAL_MIXED: "karışık",
    SIGNAL_NO_DATA: "veri yok",
}


def _percentile_bucket(pct: Optional[float]) -> Optional[str]:
    """Map a 0..100 multiple percentile to its directional bucket, or
    ``None`` if the percentile itself is ``None``. Same thresholds the raw
    multiples signal uses: ``> 70`` expensive, ``< 30`` cheap, else fair."""
    if pct is None:
        return None
    if pct > _PERCENTILE_EXPENSIVE:
        return SIGNAL_EXPENSIVE
    if pct < _PERCENTILE_CHEAP:
        return SIGNAL_CHEAP
    return SIGNAL_FAIR


def _bucket_word_tr(pct: Optional[float]) -> str:
    """Turkish phrase describing where a percentile sits, for the mixed-
    signal rationale sentence (e.g. ``"pahalı tarafta"``)."""
    bucket = _percentile_bucket(pct)
    if bucket == SIGNAL_EXPENSIVE:
        return "pahalı tarafta"
    if bucket == SIGNAL_CHEAP:
        return "ucuz tarafta"
    return "tarihsel ortasında"


def _money(value: Optional[float]) -> str:
    """Format a price/band value as ``$`` + thousands-separated number with
    up to 2 decimals, trimming trailing zeros. ``None`` -> ``"—"``."""
    if value is None:
        return "—"
    text = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"${text}"


def _pct(value: Optional[float]) -> str:
    """Format a fraction (e.g. ``0.185``) as a Turkish-style percent with 1
    decimal, e.g. ``"%18.5"``. ``None`` -> ``"—"``."""
    if value is None:
        return "—"
    return f"%{value * 100:.1f}"


def _percentile(value: Optional[float]) -> str:
    """Format an already-0..100 percentile as ``"%73"`` (0 decimals).
    ``None`` -> ``"—"``."""
    if value is None:
        return "—"
    return f"%{value:.0f}"


def _dcf_signal(
    price: Optional[float],
    dcf_base_band: Optional[dict],
    hyper_growth: bool = False,
    bull_band: Optional[dict] = None,
) -> str:
    """DCF (or P/B x ROE, if that's what's passed as ``dcf_base_band``)
    signal: price below the base band -> cheap, above -> expensive, inside
    -> fair; missing price or band -> no data.

    When ``hyper_growth`` is true and a usable ``bull_band`` (with a numeric
    ``"hi"``) is supplied, the "above the base band" case splits in two per
    HYPER_SPEC.md Sec.4: at/below the bull band's high end is
    :data:`SIGNAL_HIGH_EXPECTATION` (priced for an aggressive-but-plausible
    bull case), only above it is :data:`SIGNAL_EXPENSIVE`. Non-hyper calls
    (or hyper calls missing a usable bull band) keep the plain 3-way logic
    unchanged.
    """
    if price is None or not dcf_base_band:
        return SIGNAL_NO_DATA
    lo = dcf_base_band.get("lo")
    hi = dcf_base_band.get("hi")
    if lo is None or hi is None:
        return SIGNAL_NO_DATA

    bull_hi = bull_band.get("hi") if (hyper_growth and bull_band) else None
    if bull_hi is not None:
        if price < lo:
            return SIGNAL_CHEAP
        if price <= hi:
            return SIGNAL_FAIR
        if price <= bull_hi:
            return SIGNAL_HIGH_EXPECTATION
        return SIGNAL_EXPENSIVE

    if price < lo:
        return SIGNAL_CHEAP
    if price > hi:
        return SIGNAL_EXPENSIVE
    return SIGNAL_FAIR


def _dcf_rationale(
    price: Optional[float],
    dcf_base_band: Optional[dict],
    hyper_growth: bool = False,
    bull_band: Optional[dict] = None,
) -> str:
    """Turkish display sentence explaining the DCF signal, mirroring
    :func:`_dcf_signal`'s branches exactly (same inputs, same thresholds)."""
    if price is None or not dcf_base_band:
        return "Fiyat ya da baz değerleme aralığı yok."
    lo = dcf_base_band.get("lo")
    hi = dcf_base_band.get("hi")
    if lo is None or hi is None:
        return "Fiyat ya da baz değerleme aralığı yok."

    bull_hi = bull_band.get("hi") if (hyper_growth and bull_band) else None
    if bull_hi is not None:
        if price < lo:
            return f"Fiyat {_money(price)}, baz değerleme aralığının ({_money(lo)}–{_money(hi)}) altında → ucuz."
        if price <= hi:
            return f"Fiyat {_money(price)}, baz değerleme aralığı ({_money(lo)}–{_money(hi)}) içinde → makul."
        if price <= bull_hi:
            return (
                f"Fiyat {_money(price)}, baz aralığın ({_money(lo)}–{_money(hi)}) üzerinde ama boğa senaryosu "
                f"({_money(bull_hi)}) sınırında → yüksek beklenti."
            )
        return f"Fiyat {_money(price)}, baz değerleme aralığının ({_money(lo)}–{_money(hi)}) üzerinde → pahalı."

    if price < lo:
        return f"Fiyat {_money(price)}, baz değerleme aralığının ({_money(lo)}–{_money(hi)}) altında → ucuz."
    if price > hi:
        return f"Fiyat {_money(price)}, baz değerleme aralığının ({_money(lo)}–{_money(hi)}) üzerinde → pahalı."
    return f"Fiyat {_money(price)}, baz değerleme aralığı ({_money(lo)}–{_money(hi)}) içinde → makul."


def _reverse_dcf_signal(
    implied_growth: Optional[float],
    realized_cagr: Optional[float],
    base_growth: Optional[float],
    reverse_dcf_status: Optional[str] = None,
) -> str:
    """Reverse-DCF signal: compares the growth rate the price implies
    against a reference growth rate (realized CAGR if available, else the
    base-scenario assumed growth).

    When ``reverse_dcf_status`` is ``"above_bracket"``/``"below_bracket"``
    (see ``reverse_dcf.implied_growth_with_status``), the price implies a
    growth rate outside the bisection bracket entirely -- there is no
    numeric ``implied_growth`` to compare, but the direction is still known
    (a price the model can't reach even at its most optimistic/pessimistic
    growth is definitionally expensive/cheap), so this returns
    :data:`SIGNAL_EXPENSIVE`/:data:`SIGNAL_CHEAP` directly without needing
    ``implied_growth`` or a reference growth rate at all. Any other status
    (``None``/``"ok"``/``"no_data"``) falls through to the original
    implied-vs-reference comparison, unchanged.
    """
    if reverse_dcf_status == "above_bracket":
        return SIGNAL_EXPENSIVE
    if reverse_dcf_status == "below_bracket":
        return SIGNAL_CHEAP
    if implied_growth is None:
        return SIGNAL_NO_DATA
    reference = realized_cagr if realized_cagr is not None else base_growth
    if reference is None:
        return SIGNAL_NO_DATA
    if implied_growth > reference + _REVERSE_DCF_MARGIN:
        return SIGNAL_EXPENSIVE
    if implied_growth < reference - _REVERSE_DCF_MARGIN:
        return SIGNAL_CHEAP
    return SIGNAL_FAIR


def _reverse_dcf_rationale(
    implied_growth: Optional[float],
    realized_cagr: Optional[float],
    base_growth: Optional[float],
    reverse_dcf_status: Optional[str] = None,
) -> str:
    """Turkish display sentence explaining the reverse-DCF signal, mirroring
    :func:`_reverse_dcf_signal`'s branches exactly (same inputs, same
    thresholds)."""
    if reverse_dcf_status == "above_bracket":
        return "Fiyat, modelin ulaşabileceği en iyimser büyümenin bile üzerinde bir beklenti ima ediyor → pahalı."
    if reverse_dcf_status == "below_bracket":
        return "Fiyat, modelin en kötümser büyüme senaryosunun bile altında bir beklenti ima ediyor → ucuz."
    if implied_growth is None:
        return "Fiyatın ima ettiği büyüme hesaplanamadı."
    reference = realized_cagr if realized_cagr is not None else base_growth
    if reference is None:
        return "Fiyatın ima ettiği büyüme hesaplanamadı."
    ref_word = "gerçekleşen büyüme" if realized_cagr is not None else "varsayılan büyüme"
    if implied_growth > reference + _REVERSE_DCF_MARGIN:
        return (
            f"Fiyat {_pct(implied_growth)} büyüme ima ediyor; {ref_word} {_pct(reference)} — piyasa daha fazla "
            "büyüme fiyatlıyor → pahalı."
        )
    if implied_growth < reference - _REVERSE_DCF_MARGIN:
        return (
            f"Fiyat {_pct(implied_growth)} büyüme ima ediyor; {ref_word} {_pct(reference)} — piyasa daha az "
            "büyüme fiyatlıyor → ucuz."
        )
    return f"Fiyatın ima ettiği büyüme ({_pct(implied_growth)}) {ref_word} ({_pct(reference)}) ile uyumlu → makul."


def _raw_multiples_signal(
    pe_pct: Optional[float], ps_pct: Optional[float], pfcf_pct: Optional[float], sector_type: Optional[str]
) -> str:
    """Raw multiples signal: uses the primary percentile (P/E, falling back
    to P/S then P/FCF; for growth_unprofitable filers P/S is primary since
    P/E is usually meaningless there). This is the pre-growth-adjustment
    signal -- :func:`_multiples_signal` layers the growth-adjusted (PEG /
    EV-Sales) divergence check on top of it."""
    if sector_type == "growth_unprofitable":
        candidates = (ps_pct, pe_pct, pfcf_pct)
    else:
        candidates = (pe_pct, ps_pct, pfcf_pct)

    pct = next((p for p in candidates if p is not None), None)
    return _percentile_bucket(pct) or SIGNAL_NO_DATA


def _multiples_signal(
    pe_pct: Optional[float],
    ps_pct: Optional[float],
    pfcf_pct: Optional[float],
    sector_type: Optional[str],
    raw_growth_pair_pct: Optional[float] = None,
    growth_adj_pct: Optional[float] = None,
) -> str:
    """Multiples signal, two-component (VALUATION.md Sec.7).

    Starts from the raw primary-percentile signal (see
    :func:`_raw_multiples_signal`). When BOTH the raw multiple's own
    percentile (``raw_growth_pair_pct`` -- P/E in standard mode, EV/Sales in
    hyper-grower mode; the multiple the growth-adjusted figure is derived
    from) and the growth-adjusted percentile (``growth_adj_pct``) are
    present and fall in DIFFERENT directional buckets, returns
    :data:`SIGNAL_MIXED` -- the raw multiple and its growth-normalized
    counterpart disagree on direction. When they agree, or either is
    missing, the raw signal is returned unchanged (existing behavior).
    """
    raw = _raw_multiples_signal(pe_pct, ps_pct, pfcf_pct, sector_type)
    if raw == SIGNAL_NO_DATA:
        return raw
    if raw_growth_pair_pct is None or growth_adj_pct is None:
        return raw
    if _percentile_bucket(raw_growth_pair_pct) != _percentile_bucket(growth_adj_pct):
        return SIGNAL_MIXED
    return raw


def _multiples_rationale(
    pe_pct: Optional[float],
    ps_pct: Optional[float],
    pfcf_pct: Optional[float],
    sector_type: Optional[str],
    raw_growth_pair_pct: Optional[float] = None,
    growth_adj_pct: Optional[float] = None,
) -> str:
    """Turkish display sentence explaining the multiples signal, mirroring
    :func:`_multiples_signal`'s branches exactly (same inputs, thresholds)."""
    if sector_type == "growth_unprofitable":
        candidates = ((ps_pct, "P/S"), (pe_pct, "P/E"), (pfcf_pct, "P/FCF"))
    else:
        candidates = ((pe_pct, "P/E"), (ps_pct, "P/S"), (pfcf_pct, "P/FCF"))

    pct, label = next(((p, lbl) for p, lbl in candidates if p is not None), (None, None))
    if pct is None:
        return "Çarpan persentili hesaplanamadı."

    # Growth-adjusted divergence takes precedence over the plain sentence,
    # matching _multiples_signal's SIGNAL_MIXED branch.
    if (
        raw_growth_pair_pct is not None
        and growth_adj_pct is not None
        and _percentile_bucket(raw_growth_pair_pct) != _percentile_bucket(growth_adj_pct)
    ):
        return (
            f"Ham çarpan persentili {_percentile(raw_growth_pair_pct)} ({_bucket_word_tr(raw_growth_pair_pct)}) "
            f"ama büyümeye göre normalize edilince (persentil {_percentile(growth_adj_pct)}, "
            f"{_bucket_word_tr(growth_adj_pct)}) ayrışıyor → karışık sinyal."
        )

    if pct > _PERCENTILE_EXPENSIVE:
        return f"{label} persentili {_percentile(pct)} (>70) — kendi tarihsel aralığına göre pahalı."
    if pct < _PERCENTILE_CHEAP:
        return f"{label} persentili {_percentile(pct)} (<30) — kendi tarihsel aralığına göre ucuz."
    return f"{label} persentili {_percentile(pct)} — tarihsel aralığın ortalarında → makul."


def triangulate(
    price: Optional[float],
    dcf_base_band: Optional[dict],
    implied_growth: Optional[float],
    realized_cagr: Optional[float],
    base_growth: Optional[float],
    pe_pct: Optional[float],
    ps_pct: Optional[float],
    pfcf_pct: Optional[float],
    sector_type: Optional[str],
    hyper_growth: bool = False,
    bull_band: Optional[dict] = None,
    reverse_dcf_status: Optional[str] = None,
    raw_growth_pair_pct: Optional[float] = None,
    growth_adj_pct: Optional[float] = None,
) -> dict:
    """Combine the three method signals into one confidence + direction view.

    Args:
        price: Current market price per share.
        dcf_base_band: The base scenario's ``{"lo", "hi"}`` band (DCF, or
            the P/B x ROE anchor's base band for financial/reit sectors).
        implied_growth: The reverse-DCF implied growth rate, or ``None``.
        realized_cagr: The realized revenue CAGR to compare against, or
            ``None`` (falls back to ``base_growth``).
        base_growth: The base-scenario assumed ``growth_5y``, used as the
            reverse-DCF reference when ``realized_cagr`` is unavailable.
        pe_pct, ps_pct, pfcf_pct: Current-multiple percentile positions (see
            ``multiples.percentile_position``).
        sector_type: One of the ``sector.classify_sector`` buckets; changes
            which multiple is primary for ``growth_unprofitable`` filers.
        hyper_growth: Whether the filer is in hyper-grower revenue-first DCF
            mode (HYPER_SPEC.md Sec.4). When true (and ``bull_band`` is
            usable), the DCF signal gains a 4th value,
            :data:`SIGNAL_HIGH_EXPECTATION`, for prices above the base band
            but at/below the bull band -- reverse-DCF remains the primary
            interpretive lens in this mode; the DCF signal here is only
            reclassified, not reweighted. Defaults to ``False`` (unchanged
            3-way DCF signal).
        bull_band: The bull scenario's ``{"lo", "hi"}`` band, required
            (alongside ``hyper_growth=True``) for the 4-way DCF signal;
            ignored otherwise. ``None`` by default.
        reverse_dcf_status: The reverse-DCF bracket status (see
            ``reverse_dcf.implied_growth_with_status``): ``"above_bracket"``
            forces the reverse-DCF signal to :data:`SIGNAL_EXPENSIVE`,
            ``"below_bracket"`` forces :data:`SIGNAL_CHEAP`, even when
            ``implied_growth`` is ``None`` (there was no root in the
            bracket to report a number for). ``None`` (the default,
            matching every existing caller) preserves the original
            implied-vs-reference comparison.
        raw_growth_pair_pct: The historical percentile of the raw multiple
            the growth-adjusted figure is derived from (P/E in standard
            mode, EV/Sales in hyper-grower mode). Paired with
            ``growth_adj_pct`` to detect a growth-adjustment divergence
            (VALUATION.md Sec.7). ``None`` (default) disables the check.
        growth_adj_pct: The historical percentile of the growth-adjusted
            multiple (PEG / growth-adjusted EV/Sales). When it and
            ``raw_growth_pair_pct`` fall in different directional buckets,
            the multiples signal becomes :data:`SIGNAL_MIXED`. ``None``
            (default) disables the check, preserving the raw signal.

    Returns:
        ``{"signals": {"dcf", "reverse_dcf", "multiples"}, "confidence":
        "YÜKSEK"|"ORTA"|"DÜŞÜK", "direction": <majority signal or
        "belirsiz">, "rationale": {"dcf", "reverse_dcf", "multiples",
        "confidence"}}``. Confidence: all three (non-"veri_yok") signals
        agree -> YÜKSEK; exactly two agree -> ORTA; otherwise (scattered,
        or 2+ signals are "veri_yok") -> DÜŞÜK. ``direction`` can surface
        :data:`SIGNAL_HIGH_EXPECTATION` in hyper-grower mode exactly like any
        other signal value, via the same majority/agreement counting.
        ``rationale`` holds one display-ready Turkish sentence per method
        (explaining the signal it gave, with the underlying numbers already
        formatted in) plus one explaining why the overall ``confidence``
        came out as it did; on the exception-fallback path these are
        generic "veri yok" sentences. Never raises.
    """
    try:
        return _triangulate(
            price, dcf_base_band, implied_growth, realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type,
            hyper_growth, bull_band, reverse_dcf_status, raw_growth_pair_pct, growth_adj_pct,
        )
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("triangulate() failed unexpectedly; returning a no-data result.")
        return {
            "signals": {"dcf": SIGNAL_NO_DATA, "reverse_dcf": SIGNAL_NO_DATA, "multiples": SIGNAL_NO_DATA},
            "confidence": CONFIDENCE_LOW,
            "direction": _DIRECTION_UNCLEAR,
            "rationale": {
                "dcf": "Veri yok.",
                "reverse_dcf": "Veri yok.",
                "multiples": "Veri yok.",
                "confidence": "Veri yok.",
            },
        }


def _triangulate(
    price, dcf_base_band, implied_growth, realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type,
    hyper_growth=False, bull_band=None, reverse_dcf_status=None, raw_growth_pair_pct=None, growth_adj_pct=None,
) -> dict:
    signals = {
        "dcf": _dcf_signal(price, dcf_base_band, hyper_growth, bull_band),
        "reverse_dcf": _reverse_dcf_signal(implied_growth, realized_cagr, base_growth, reverse_dcf_status),
        "multiples": _multiples_signal(pe_pct, ps_pct, pfcf_pct, sector_type, raw_growth_pair_pct, growth_adj_pct),
    }
    rationale = {
        "dcf": _dcf_rationale(price, dcf_base_band, hyper_growth, bull_band),
        "reverse_dcf": _reverse_dcf_rationale(implied_growth, realized_cagr, base_growth, reverse_dcf_status),
        "multiples": _multiples_rationale(pe_pct, ps_pct, pfcf_pct, sector_type, raw_growth_pair_pct, growth_adj_pct),
    }

    substantive = [s for s in signals.values() if s != SIGNAL_NO_DATA]
    no_data_count = len(signals) - len(substantive)

    counts: Dict[str, int] = {}
    for s in substantive:
        counts[s] = counts.get(s, 0) + 1
    majority_signal, majority_count = (max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0))

    if no_data_count >= 2:
        confidence = CONFIDENCE_LOW
        direction = _DIRECTION_UNCLEAR
        rationale["confidence"] = (
            "İki veya daha fazla yöntemde veri yok; sağlam bir karşılaştırma yapılamadığı için güven düşük."
        )
    elif len(substantive) == 3 and majority_count == 3:
        confidence = CONFIDENCE_HIGH
        direction = majority_signal
        rationale["confidence"] = (
            f"Üç yöntem de aynı yönü ({_SIGNAL_LABEL_TR.get(direction, direction)}) gösteriyor; güven yüksek."
        )
    elif majority_count == 2:
        confidence = CONFIDENCE_MEDIUM
        direction = majority_signal
        rationale["confidence"] = (
            f"Üç yöntemden ikisi {_SIGNAL_LABEL_TR.get(majority_signal, majority_signal)} diyor, biri ayrışıyor; "
            "güven orta."
        )
    else:
        confidence = CONFIDENCE_LOW
        direction = _DIRECTION_UNCLEAR
        rationale["confidence"] = "Yöntemler birbirinden farklı sinyaller veriyor; ortak bir yön olmadığı için güven düşük."

    return {"signals": signals, "confidence": confidence, "direction": direction, "rationale": rationale}
