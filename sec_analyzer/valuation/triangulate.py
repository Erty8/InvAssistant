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

#: Leverage gate (VALUATION.md Sec.2/Sec.7): a non-financial/non-reit filer is
#: treated as "leveraged" -- EV/EBITDA becomes the PRIMARY own-history multiple
#: ahead of P/E -- when its net-debt-to-EBITDA ratio is at or above this. 1.0x
#: means more net debt than a full year of EBITDA. EV/EBIT(DA) is
#: capital-structure-neutral (numerator adds net debt, denominator is
#: pre-interest), so it ranks a leveraged filer against its own history without
#: the leverage distortion P/E carries; below this ratio the P/E-primary path
#: is preserved unchanged.
_LEVERAGE_EBITDA_RATIO = 1.0

#: Sector-relative multiple band (VALUATION.md Sec.7 axis-b). The current
#: primary multiple divided by its Damodaran sector median: above the
#: expensive bound reads as "expensive vs sector", below the (reciprocal)
#: cheap bound as "cheap vs sector", between as "in line with the median".
#: Geometric-symmetric band (0.80 == 1/1.25) so a multiple 25% above and one
#: 20% below the median are mirror cases. Threshold is sector-RELATIVE, never
#: an absolute multiple value -- consistent with the "no absolute PEG
#: threshold" rule (VALUATION.md Sec.7).
_SECTOR_RATIO_EXPENSIVE = 1.25
_SECTOR_RATIO_CHEAP = 0.80

CONFIDENCE_HIGH = "YÜKSEK"
CONFIDENCE_MEDIUM = "ORTA"
CONFIDENCE_LOW = "DÜŞÜK"

_DIRECTION_UNCLEAR = "belirsiz"

#: Model–market divergence governor (the "expectations discipline" backstop).
#: When the base fair-value band sits a large multiple away from the price,
#: the three method signals are NOT independent confirmation -- they all read
#: off the same assumption set, so a unanimous "cheap"/"expensive" is one
#: assumption reflected in three mirrors, not three witnesses. In that case
#: the honest reading is "the model and the market disagree about this
#: company's entire future", not a high-confidence verdict.
#:
#: The trigger is symmetric but its ACTION is not (yet):
#:  * UP-side (model says deeply cheap): ``base_lo > price * _DIVERGENCE_UP_FACTOR``
#:    -- confidence is floored to :data:`CONFIDENCE_LOW` now, and the payload
#:    carries ``action="verdict"`` so the report/interpret layer can restate
#:    the headline as an explicit divergence rather than "cheap".
#:  * DOWN-side (model says deeply expensive): ``base_hi < price *
#:    _DIVERGENCE_DOWN_FACTOR`` -- ``action="log_only"``: the payload is
#:    recorded but confidence/direction are left untouched, so the coming
#:    low-side calibration pass inherits "which names, which paths" data
#:    without any verdict changing today.
#: Anchored on the base band's NEAR edge (lo for up, hi for down) so a merely
#: wide band can't trip it on optimism alone. Threshold 2.0/0.5 is a starting
#: point, tuned by how many names in the calibration basket trip it.
_DIVERGENCE_UP_FACTOR = 2.0
_DIVERGENCE_DOWN_FACTOR = 0.5

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


def _sector_ratio_bucket(ratio: Optional[float]) -> Optional[str]:
    """Map a ``current primary multiple / sector median`` ratio to its
    directional bucket (VALUATION.md Sec.7 axis-b), or ``None`` when the
    ratio is absent/non-positive (no usable sector median). ``> 1.25``
    expensive vs sector, ``< 0.80`` cheap vs sector, else in line."""
    if ratio is None or ratio <= 0:
        return None
    if ratio > _SECTOR_RATIO_EXPENSIVE:
        return SIGNAL_EXPENSIVE
    if ratio < _SECTOR_RATIO_CHEAP:
        return SIGNAL_CHEAP
    return SIGNAL_FAIR


def _sector_word_tr(bucket: Optional[str]) -> str:
    """Turkish phrase for a sector-relative bucket, for the mixed-signal
    rationale sentence (e.g. ``"pahalı"``)."""
    if bucket == SIGNAL_EXPENSIVE:
        return "pahalı"
    if bucket == SIGNAL_CHEAP:
        return "ucuz"
    return "medyanla uyumlu"


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
    pe_pct: Optional[float],
    ps_pct: Optional[float],
    pfcf_pct: Optional[float],
    sector_type: Optional[str],
    pffo_pct: Optional[float] = None,
    ev_ebitda_pct: Optional[float] = None,
    leveraged: bool = False,
) -> str:
    """Raw multiples signal: uses the primary percentile (P/E, falling back
    to P/S then P/FCF; for growth_unprofitable filers P/S is primary since
    P/E is usually meaningless there; for reit, P/FFO is primary, falling
    back to P/S -- P/E is meaningless for REITs since GAAP depreciation
    depresses net income, see SPEC.md Sec.8/FFO). This is the
    pre-growth-adjustment signal -- :func:`_multiples_signal` layers the
    growth-adjusted (PEG / EV-Sales) divergence check on top of it.

    For a ``leveraged`` mature/cyclical filer (net debt / EBITDA at or above
    :data:`_LEVERAGE_EBITDA_RATIO`), EV/EBITDA becomes the PRIMARY multiple
    ahead of P/E (VALUATION.md Sec.2/Sec.7) -- capital-structure-neutral, so
    it isn't distorted by the leverage that inflates/deflates P/E. Falls back
    to the P/E-primary order when ``ev_ebitda_pct`` is ``None`` (no usable EV
    history). ``leveraged`` is ignored for growth_unprofitable/reit (their
    primary is unchanged)."""
    if sector_type == "growth_unprofitable":
        candidates = (ps_pct, pe_pct, pfcf_pct)
    elif sector_type == "reit":
        candidates = (pffo_pct, ps_pct)
    elif leveraged:
        candidates = (ev_ebitda_pct, pe_pct, ps_pct, pfcf_pct)
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
    pffo_pct: Optional[float] = None,
    sector_ratio: Optional[float] = None,
    ev_ebitda_pct: Optional[float] = None,
    leveraged: bool = False,
) -> str:
    """Multiples signal, positioned on TWO axes (VALUATION.md Sec.7).

    Starts from the raw primary-percentile signal (see
    :func:`_raw_multiples_signal`; for reit, P/FFO is primary rather than
    P/E, see SPEC.md Sec.8/FFO; for a ``leveraged`` filer, EV/EBITDA is
    primary rather than P/E, see :data:`_LEVERAGE_EBITDA_RATIO`) -- the
    company's position against its OWN multiple history. Two independent
    divergence checks can override the raw signal to :data:`SIGNAL_MIXED`, in
    precedence order:

    1. **Growth-adjusted (axis-a refinement, highest precedence):** when BOTH
       the raw multiple's own percentile (``raw_growth_pair_pct`` -- P/E in
       standard mode, EV/Sales in hyper-grower mode) and the growth-adjusted
       percentile (``growth_adj_pct``) are present and fall in DIFFERENT
       directional buckets, the raw multiple and its growth-normalized
       counterpart disagree.
    2. **Sector-relative (axis-b):** when a usable ``sector_ratio`` (current
       primary multiple / Damodaran sector median) is present and its bucket
       (see :func:`_sector_ratio_bucket`) disagrees with the own-history
       bucket, the company is cheap/expensive against its own past but the
       opposite against peers.

    When neither divergence fires (they agree, or the relevant inputs are
    missing), the raw own-history signal is returned unchanged -- so callers
    that pass neither the growth-adjusted pair nor a sector ratio keep the
    exact pre-existing behavior.
    """
    raw = _raw_multiples_signal(pe_pct, ps_pct, pfcf_pct, sector_type, pffo_pct, ev_ebitda_pct, leveraged)
    if raw == SIGNAL_NO_DATA:
        return raw
    # When EV/EBITDA is the primary (leveraged filer), the P/E-based PEG
    # divergence axis is deliberately skipped -- we've decided P/E is
    # unreliable for this filer, so a P/E-vs-PEG disagreement shouldn't
    # override the EV/EBITDA own-history read. (The sector axis-b is naturally
    # off too: no Damodaran EV/EBITDA median exists, so the engine passes
    # sector_ratio=None here.)
    ev_primary = leveraged and ev_ebitda_pct is not None
    # 1. Growth-adjusted divergence takes precedence (most informative).
    if (
        not ev_primary
        and raw_growth_pair_pct is not None
        and growth_adj_pct is not None
        and _percentile_bucket(raw_growth_pair_pct) != _percentile_bucket(growth_adj_pct)
    ):
        return SIGNAL_MIXED
    # 2. Own-history vs sector-median divergence (only when both defined).
    sector_bucket = _sector_ratio_bucket(sector_ratio)
    if sector_bucket is not None and sector_bucket != raw:
        return SIGNAL_MIXED
    return raw


def _multiples_rationale(
    pe_pct: Optional[float],
    ps_pct: Optional[float],
    pfcf_pct: Optional[float],
    sector_type: Optional[str],
    raw_growth_pair_pct: Optional[float] = None,
    growth_adj_pct: Optional[float] = None,
    pffo_pct: Optional[float] = None,
    sector_ratio: Optional[float] = None,
    ev_ebitda_pct: Optional[float] = None,
    leveraged: bool = False,
) -> str:
    """Turkish display sentence explaining the multiples signal, mirroring
    :func:`_multiples_signal`'s branches exactly (same inputs, thresholds,
    precedence)."""
    ev_primary = leveraged and ev_ebitda_pct is not None
    if sector_type == "growth_unprofitable":
        candidates = ((ps_pct, "P/S"), (pe_pct, "P/E"), (pfcf_pct, "P/FCF"))
    elif sector_type == "reit":
        candidates = ((pffo_pct, "P/FFO"), (ps_pct, "P/S"))
    elif leveraged:
        candidates = ((ev_ebitda_pct, "FD/FAVÖK"), (pe_pct, "P/E"), (ps_pct, "P/S"), (pfcf_pct, "P/FCF"))
    else:
        candidates = ((pe_pct, "P/E"), (ps_pct, "P/S"), (pfcf_pct, "P/FCF"))

    pct, label = next(((p, lbl) for p, lbl in candidates if p is not None), (None, None))
    if pct is None:
        return "Çarpan persentili hesaplanamadı."

    # 1. Growth-adjusted divergence takes precedence over the plain sentence,
    # matching _multiples_signal's first SIGNAL_MIXED branch (skipped when
    # EV/EBITDA is primary -- see _multiples_signal for why).
    if (
        not ev_primary
        and raw_growth_pair_pct is not None
        and growth_adj_pct is not None
        and _percentile_bucket(raw_growth_pair_pct) != _percentile_bucket(growth_adj_pct)
    ):
        return (
            f"Ham çarpan persentili {_percentile(raw_growth_pair_pct)} ({_bucket_word_tr(raw_growth_pair_pct)}) "
            f"ama büyümeye göre normalize edilince (persentil {_percentile(growth_adj_pct)}, "
            f"{_bucket_word_tr(growth_adj_pct)}) ayrışıyor → karışık sinyal."
        )

    # 2. Own-history vs sector-median divergence (second SIGNAL_MIXED branch).
    own_bucket = _percentile_bucket(pct)
    sector_bucket = _sector_ratio_bucket(sector_ratio)
    if sector_bucket is not None and sector_bucket != own_bucket:
        return (
            f"{label} persentili {_percentile(pct)} ({_bucket_word_tr(pct)}) "
            f"ama sektör medyanına göre {_sector_word_tr(sector_bucket)} → karışık sinyal."
        )

    # Own-history and sector agree (or no sector data): plain sentence, with a
    # sector-confirmation clause appended when a sector median is available.
    if sector_bucket == SIGNAL_EXPENSIVE:
        sector_suffix = " Sektör medyanına göre de pahalı."
    elif sector_bucket == SIGNAL_CHEAP:
        sector_suffix = " Sektör medyanına göre de ucuz."
    elif sector_bucket == SIGNAL_FAIR:
        sector_suffix = " Sektör medyanıyla da uyumlu."
    else:
        sector_suffix = ""

    if pct > _PERCENTILE_EXPENSIVE:
        return f"{label} persentili {_percentile(pct)} (>70) — kendi tarihsel aralığına göre pahalı.{sector_suffix}"
    if pct < _PERCENTILE_CHEAP:
        return f"{label} persentili {_percentile(pct)} (<30) — kendi tarihsel aralığına göre ucuz.{sector_suffix}"
    return f"{label} persentili {_percentile(pct)} — tarihsel aralığın ortalarında → makul.{sector_suffix}"


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
    earnings_power_headline: bool = False,
    mature_revenue_headline: bool = False,
    midgrowth_revenue_headline: bool = False,
    pffo_pct: Optional[float] = None,
    cyclical_fcfe_headline: bool = False,
    sector_ratio: Optional[float] = None,
    ev_ebitda_pct: Optional[float] = None,
    net_debt_to_ebitda: Optional[float] = None,
) -> dict:
    """Combine the three method signals into one confidence + direction view.

    Args:
        price: Current market price per share.
        dcf_base_band: The base scenario's ``{"lo", "hi"}`` band (DCF, or
            the P/B x ROE anchor's base band for the financial sector, or
            the FFO Gordon-growth anchor's base band for the reit sector --
            see SPEC.md Sec.8/FFO).
        implied_growth: The reverse-DCF implied growth rate, or ``None``.
        realized_cagr: The realized revenue CAGR to compare against, or
            ``None`` (falls back to ``base_growth``).
        base_growth: The base-scenario assumed ``growth_5y``, used as the
            reverse-DCF reference when ``realized_cagr`` is unavailable.
        pe_pct, ps_pct, pfcf_pct: Current-multiple percentile positions (see
            ``multiples.percentile_position``).
        sector_type: One of the ``sector.classify_sector`` buckets; changes
            which multiple is primary for ``growth_unprofitable`` (P/S) and
            ``reit`` (P/FFO, via ``pffo_pct``) filers.
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
        earnings_power_headline: Whether the headline fair-value range came
            from the earnings-power-value (EPV) anchor rather than the
            FCF-DCF (SPEC.md Sec.8a -- mature, FCF-suppressed-but-profitable
            filers like Amazon). When true and confidence would otherwise
            come out :data:`CONFIDENCE_HIGH`, it's capped at
            :data:`CONFIDENCE_MEDIUM`: DCF and multiples both ultimately
            derive from the same underlying earnings signal in this mode,
            so three-way agreement is weaker evidence than it is when the
            DCF leg is an independent FCF-based estimate. ``False`` (the
            default) preserves existing behavior.
        mature_revenue_headline: Whether the headline fair-value range came
            from the mature, FCF-suppressed-but-growing revenue-first DCF
            (VALUATION.md Sec.4/4a addendum -- e.g. Amazon-shaped filers
            whose realized growth clears the growth gate; see
            ``engine._build_mature_revenue_dcf``) rather than the raw
            FCF-DCF or the EPV anchor. Same :data:`CONFIDENCE_MEDIUM` cap as
            ``earnings_power_headline`` when confidence would otherwise come
            out :data:`CONFIDENCE_HIGH`: the DCF leg and the reverse-DCF leg
            both derive from this same revenue-first model in this mode, so
            they aren't independent evidence of one another. ``False`` (the
            default) preserves existing behavior. Mutually exclusive with
            ``earnings_power_headline`` in practice (``engine.py`` never
            sets both), but if both were ever true, the
            ``earnings_power_headline`` cap message takes precedence.
        midgrowth_revenue_headline: Whether the headline fair-value range came
            from the mid-growth, loss-making revenue-first DCF (SPEC.md
            Sec.8d -- ``growth_unprofitable`` filers growing 12-20% that
            ``detect_hyper_grower`` doesn't pick up; see
            ``engine._build_midgrowth_revenue_dcf``). Same
            :data:`CONFIDENCE_MEDIUM` cap as ``mature_revenue_headline`` for
            the same reason: the DCF leg and its reverse-DCF leg both derive
            from one revenue-first model, so they aren't independent
            evidence. ``False`` (the default) preserves existing behavior.
            Mutually exclusive in practice with the two headline flags above.
        cyclical_fcfe_headline: Whether the headline fair-value range came
            from the cyclical sustainable-growth FCFE anchor (SPEC.md
            Sec.8e -- capital-intensive cyclical filers, e.g. Micron, whose
            FCF-DCF is suppressed by heavy growth CapEx; see
            ``engine._build_cyclical_fcfe``) rather than the raw FCF-DCF,
            the cycle-mid normalized FCF-DCF, or the EPV anchor. Same
            :data:`CONFIDENCE_MEDIUM` cap as the other headline overrides
            for the same reason: the DCF leg here derives from the same
            normalized-earnings anchor the reverse-DCF/multiples legs are
            ultimately compared against, so three-way agreement is weaker
            evidence than it is for an independent FCF-based estimate.
            ``False`` (the default) preserves existing behavior. Mutually
            exclusive in practice with the other headline flags above.
        pffo_pct: The current P/FFO's historical percentile (see
            ``multiples.percentile_position`` over ``multiples_history``'s
            ``pffo`` column). Primary multiples-signal candidate for
            ``sector_type == "reit"`` (SPEC.md Sec.8/FFO), falling back to
            ``ps_pct`` when ``None``; ignored for every other sector.
            ``None`` (the default) preserves existing behavior for callers
            that don't pass it.
        sector_ratio: The current primary multiple divided by its Damodaran
            sector median (VALUATION.md Sec.7 axis-b) -- the SAME primary
            multiple the own-history percentile signal uses (P/E, or P/S for
            ``growth_unprofitable``, or P/FFO for ``reit`` when a sector
            median exists for it). ``> 1.25`` reads as expensive vs sector,
            ``< 0.80`` as cheap; when its bucket disagrees with the
            own-history bucket the multiples signal becomes
            :data:`SIGNAL_MIXED` (second-precedence, after the
            growth-adjusted check). ``None`` (the default) -- no usable
            sector median -- disables the axis and preserves the pure
            own-history behavior.
        ev_ebitda_pct: The current EV/EBITDA's historical percentile (see
            ``multiples.percentile_position`` over ``multiples_history``'s
            ``ev_ebitda`` column). Becomes the PRIMARY own-history multiple
            (ahead of P/E) when ``net_debt_to_ebitda`` marks the filer as
            leveraged and this is not ``None`` (VALUATION.md Sec.2/Sec.7);
            ignored for growth_unprofitable/reit and for non-leveraged filers.
            ``None`` (the default) preserves P/E-primary behavior.
        net_debt_to_ebitda: The filer's current net-debt-to-EBITDA ratio, or
            ``None`` when it can't be derived (missing/non-positive EBITDA or
            net debt). ``>= _LEVERAGE_EBITDA_RATIO`` (1.0) flips the multiples
            primary to EV/EBITDA per above. ``None`` (the default) -- not
            leveraged / unknown -- preserves P/E-primary behavior.

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
            earnings_power_headline, mature_revenue_headline, midgrowth_revenue_headline, pffo_pct,
            cyclical_fcfe_headline, sector_ratio, ev_ebitda_pct, net_debt_to_ebitda,
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
            "divergence": None,
        }


def _triangulate(
    price, dcf_base_band, implied_growth, realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type,
    hyper_growth=False, bull_band=None, reverse_dcf_status=None, raw_growth_pair_pct=None, growth_adj_pct=None,
    earnings_power_headline=False, mature_revenue_headline=False, midgrowth_revenue_headline=False, pffo_pct=None,
    cyclical_fcfe_headline=False, sector_ratio=None, ev_ebitda_pct=None, net_debt_to_ebitda=None,
) -> dict:
    leveraged = net_debt_to_ebitda is not None and net_debt_to_ebitda >= _LEVERAGE_EBITDA_RATIO
    signals = {
        "dcf": _dcf_signal(price, dcf_base_band, hyper_growth, bull_band),
        "reverse_dcf": _reverse_dcf_signal(implied_growth, realized_cagr, base_growth, reverse_dcf_status),
        "multiples": _multiples_signal(
            pe_pct, ps_pct, pfcf_pct, sector_type, raw_growth_pair_pct, growth_adj_pct, pffo_pct,
            sector_ratio=sector_ratio, ev_ebitda_pct=ev_ebitda_pct, leveraged=leveraged,
        ),
    }
    rationale = {
        "dcf": _dcf_rationale(price, dcf_base_band, hyper_growth, bull_band),
        "reverse_dcf": _reverse_dcf_rationale(implied_growth, realized_cagr, base_growth, reverse_dcf_status),
        "multiples": _multiples_rationale(
            pe_pct, ps_pct, pfcf_pct, sector_type, raw_growth_pair_pct, growth_adj_pct, pffo_pct,
            sector_ratio=sector_ratio, ev_ebitda_pct=ev_ebitda_pct, leveraged=leveraged,
        ),
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

    if earnings_power_headline and confidence == CONFIDENCE_HIGH:
        confidence = CONFIDENCE_MEDIUM
        rationale["confidence"] += (
            " (Serbest nakit akışı çapası güvenilmez olduğu için manşet kazanç-gücüne dayanıyor; DCF ve "
            "çarpan bacakları aynı kazanç sinyalinin türevi olduğundan güven en fazla ORTA ile sınırlandı.)"
        )
    elif mature_revenue_headline and confidence == CONFIDENCE_HIGH:
        confidence = CONFIDENCE_MEDIUM
        rationale["confidence"] += (
            " (Manşet, olgun revenue-first DCF'e dayanıyor; bu yöntemin ters-DCF'i de aynı modelden "
            "türediği için bağımsız bir kanıt değil, güven en fazla ORTA ile sınırlandı.)"
        )
    elif midgrowth_revenue_headline and confidence == CONFIDENCE_HIGH:
        confidence = CONFIDENCE_MEDIUM
        rationale["confidence"] += (
            " (Manşet, orta-büyüme revenue-first DCF'e dayanıyor; bu yöntemin ters-DCF'i de aynı modelden "
            "türediği için bağımsız bir kanıt değil, güven en fazla ORTA ile sınırlandı.)"
        )
    elif cyclical_fcfe_headline and confidence == CONFIDENCE_HIGH:
        confidence = CONFIDENCE_MEDIUM
        rationale["confidence"] += (
            " (Manşet, kazanç-tabanlı FCFE çapasından geldiği için — DCF ve çarpanlar bastırılmış FCF'i "
            "yansıtır — güven ORTA'ya sınırlandı.)"
        )

    # --- Model–market divergence governor (expectations discipline) --------
    # Runs LAST, after every headline confidence cap, so it takes precedence.
    divergence = _divergence(price, dcf_base_band)
    if divergence is not None and divergence["action"] == "verdict":
        confidence = CONFIDENCE_LOW
        rationale["confidence"] = (
            f"Model-piyasa ayrışması: baz değerleme aralığının alt ucu ({_money(divergence['band_edge'])}) "
            f"fiyatın ({_money(price)}) {divergence['factor']:.1f} katı. Üç yöntem de aynı büyüme/marj "
            "varsayımından beslendiği için oybirliği bağımsız bir doğrulama değil — aynı varsayımın üç "
            "aynada yansımasıdır. Model büyümenin sürdüğünü, piyasa bittiğini fiyatlıyor; hakem önümüzdeki "
            "çeyreklerin gerçekleşen büyümesidir. Güven düşük."
        )

    return {
        "signals": signals,
        "confidence": confidence,
        "direction": direction,
        "rationale": rationale,
        "divergence": divergence,
    }


def _divergence(price: Optional[float], dcf_base_band: Optional[dict]) -> Optional[dict]:
    """Detect a model–market divergence from the base band vs. price (the
    governor; see :data:`_DIVERGENCE_UP_FACTOR`).

    Returns ``None`` when price/band are unusable or the band sits within the
    normal range of the price. Otherwise a dict:
    ``{"direction": "ucuz"|"pahali", "action": "verdict"|"log_only",
    "factor": <band_edge / price>, "band_edge": <the triggering lo or hi>}``.
    The UP-side ("ucuz") returns ``action="verdict"``; the DOWN-side
    ("pahali") returns ``action="log_only"`` (recorded but not yet acted on).
    """
    if price is None or price <= 0 or not dcf_base_band:
        return None
    lo = dcf_base_band.get("lo")
    hi = dcf_base_band.get("hi")
    if lo is not None and lo > price * _DIVERGENCE_UP_FACTOR:
        return {"direction": SIGNAL_CHEAP, "action": "verdict", "factor": lo / price, "band_edge": lo}
    if hi is not None and hi < price * _DIVERGENCE_DOWN_FACTOR:
        return {"direction": SIGNAL_EXPENSIVE, "action": "log_only", "factor": hi / price, "band_edge": hi}
    return None
