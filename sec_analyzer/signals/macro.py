"""Summarize a FRED macro panel into a valuation-relevant context.

:func:`sec_analyzer.fetch.fred.get_macro_panel` returns raw per-series
observations (level, date, self-anchored percentile) for a small panel of
yield-curve and credit-spread series. This module turns that raw panel into
the handful of derived numbers a human (or the report layer) actually wants
to read -- curve slope, real rate, credit regime -- plus a short list of
deterministic notes, without ever touching the network or disk itself.

This is the macro-oriented sibling of :mod:`sec_analyzer.signals.events`:
pure, deterministic, and fully defensive. Missing series never raise; they
just make the fields that depend on them ``None``. As with ``events.py``,
:func:`build_macro_context` is a never-raising public wrapper around a
``_``-prefixed implementation, and :func:`summarize_macro` is the one-line
formatter for the future "Macro:" row of the verdict card.

Percentile fields (``risk_free_percentile``, ``ig_spread_percentile``,
``hy_spread_percentile``) and :data:`_CREDIT_TIGHT_PERCENTILE` /
:data:`_CREDIT_LOOSE_PERCENTILE` all rank a value within its OWN recent
history rather than against an invented absolute threshold -- see
:func:`sec_analyzer.fetch.fred.get_series_asof`'s docstring for why a fixed
basis-point cutoff would be an unjustified parameter this project's ROADMAP
forbids.
"""

import logging
from typing import List, Optional

from sec_analyzer.fetch.fred import (
    SERIES_BAA10Y,
    SERIES_DGS2,
    SERIES_DGS10,
    SERIES_DGS30,
    SERIES_HY_OAS,
    SERIES_T10YIE,
)

logger = logging.getLogger(__name__)

#: Credit-regime cut points, expressed as percentiles of the spread's OWN
#: trailing history (see module docstring) -- quartile boundaries of the
#: series' own recent range, NOT a judgment about what basis-point level
#: counts as "wide". A high percentile means the spread is expensive
#: relative to its own recent past (tight/stressed financing); a low
#: percentile means it is cheap relative to its own recent past
#: (loose/complacent pricing of credit risk).
_CREDIT_TIGHT_PERCENTILE = 75.0
_CREDIT_LOOSE_PERCENTILE = 25.0

#: A terminal-growth vs. breakeven-inflation gap larger than this (in
#: percentage points) is called out as "real growth forever" in rule 4;
#: below it (but still >= 0) the gap is treated as noise, not a note-worthy
#: assumption. This ONLY drives an informational note -- it never changes
#: the terminal growth rate the valuation engine actually uses (see
#: :func:`build_macro_context`).
_TERMINAL_VS_INFLATION_DEADBAND_PCT = 1.0

#: A risk-free-rate percentile at or beyond either end of its own 10y range
#: is worth calling out in rule 5, since the engine's discount rate is built
#: on top of this level.
_RISK_FREE_EXTREME_PERCENTILE_HIGH = 80.0
_RISK_FREE_EXTREME_PERCENTILE_LOW = 20.0


def _fmt_pct(value: Optional[float], decimals: int = 2) -> str:
    """Format a percent-like number.

    Trailing zeros are dropped (``3.90`` -> ``"3.9"``, ``4.21`` -> ``"4.21"``)
    since the notes in this module quote numbers the way a person would
    write them, not padded to a fixed width. Returns ``"—"`` for ``None`` so
    a missing value never renders as the literal string ``"None"``.
    """
    if value is None:
        return "—"
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_percentile_int(value: Optional[float]) -> Optional[str]:
    """Render a percentile as a whole-number string (``62.5`` -> ``"62"``).

    Notes/summary text quotes percentiles without decimals (e.g. "62nd
    percentile"); the underlying context field keeps 1-decimal precision.
    Returns ``None`` for ``None`` input.
    """
    if value is None:
        return None
    return str(int(round(value)))


def _round(value: Optional[float], decimals: int) -> Optional[float]:
    """``round`` that passes ``None`` through instead of raising."""
    if value is None:
        return None
    return round(value, decimals)


def _ordinal(value: str) -> str:
    """Append the English ordinal suffix to a whole-number string (``"62"``
    -> ``"62nd"``). Used to render :func:`_fmt_percentile_int`'s output in
    the notes/summary text below."""
    n = int(value)
    if 10 <= abs(n) % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(abs(n) % 10, "th")
    return f"{value}{suffix}"


def _credit_regime(hy_percentile: Optional[float], ig_percentile: Optional[float]) -> Optional[str]:
    """Classify credit conditions from a spread's self-anchored percentile.

    Uses ``hy_percentile`` when available, falling back to
    ``ig_percentile`` (high-yield is the more sensitive/leading gauge, but
    investment-grade is better than nothing). Returns ``None`` when neither
    is available.

    A HIGH percentile means the spread is wide relative to its own recent
    history, i.e. credit is expensive/stressed -> ``"tight"``. A LOW
    percentile means it is narrow relative to its own recent history, i.e.
    credit risk is being priced cheaply -> ``"loose"``.
    """
    percentile = hy_percentile if hy_percentile is not None else ig_percentile
    if percentile is None:
        return None
    if percentile >= _CREDIT_TIGHT_PERCENTILE:
        return "tight"
    if percentile <= _CREDIT_LOOSE_PERCENTILE:
        return "loose"
    return "normal"


def _build_notes(ctx: dict) -> List[str]:
    """Build the deterministic observation list for ``ctx``.

    Each rule is independent and self-contained; a rule fires purely off the
    fields already computed in ``ctx`` (never re-reads the raw panel). Order
    matches the rule numbering in this module's design: curve inversion,
    credit-regime tight, credit-regime loose, terminal-growth cross-check
    (either direction), risk-free-rate extreme percentile.
    """
    notes: List[str] = []

    # 1. Inverted yield curve -- widely cited, poorly timed; keep the hedge.
    if ctx.get("curve_inverted"):
        notes.append(
            "Yield curve is inverted (10y-2y: {slope} pts); historically "
            "considered a recession signal, though its timing is unreliable.".format(
                slope=_fmt_pct(ctx.get("curve_slope_pct"))
            )
        )

    # 2 & 3. Credit regime.
    regime = ctx.get("credit_regime")
    if regime == "tight":
        hy_pctl = _fmt_percentile_int(ctx.get("hy_spread_percentile"))
        pctl_clause = f" ({_ordinal(hy_pctl)} percentile)" if hy_pctl is not None else ""
        notes.append(
            "Credit spreads are tight: HY spread {hy}%{pctl} — refinancing "
            "conditions are tightening, which hits leveraged companies first.".format(
                hy=_fmt_pct(ctx.get("hy_spread_pct")), pctl=pctl_clause
            )
        )
    elif regime == "loose":
        hy_pctl = _fmt_percentile_int(ctx.get("hy_spread_percentile"))
        pctl_clause = f" ({_ordinal(hy_pctl)} percentile)" if hy_pctl is not None else ""
        notes.append(
            "Credit spreads are loose: HY spread {hy}%{pctl} sits at the low end "
            "of its own recent range — credit risk is being priced cheaply.".format(
                hy=_fmt_pct(ctx.get("hy_spread_pct")), pctl=pctl_clause
            )
        )

    # 4. Terminal growth vs. breakeven inflation -- informational cross-check
    # ONLY. This note never feeds back into the terminal growth the engine
    # uses; SPEC.md's min(risk_free, 4%) anchor rule is untouched.
    diff = ctx.get("terminal_vs_inflation_pct")
    terminal_pct = ctx.get("terminal_growth_pct")
    if diff is not None and terminal_pct is not None:
        if diff > _TERMINAL_VS_INFLATION_DEADBAND_PCT:
            notes.append(
                "Terminal growth ({tg}%) is notably above the inflation the market "
                "is pricing in ({be}%) — assuming real growth forever.".format(
                    tg=_fmt_pct(terminal_pct), be=_fmt_pct(ctx.get("breakeven_inflation_pct"))
                )
            )
        elif diff < 0:
            notes.append(
                "Terminal growth ({tg}%) is below expected inflation ({be}%) "
                "— assuming a perpetuity that shrinks in real terms.".format(
                    tg=_fmt_pct(terminal_pct), be=_fmt_pct(ctx.get("breakeven_inflation_pct"))
                )
            )
        # 0 <= diff <= deadband: no note -- treated as noise.

    # 5. Risk-free rate at an extreme of its own 10y range.
    rf_pctl = ctx.get("risk_free_percentile")
    if rf_pctl is not None and (
        rf_pctl >= _RISK_FREE_EXTREME_PERCENTILE_HIGH or rf_pctl <= _RISK_FREE_EXTREME_PERCENTILE_LOW
    ):
        side = "top" if rf_pctl >= _RISK_FREE_EXTREME_PERCENTILE_HIGH else "bottom"
        notes.append(
            "10-year yield at {rf}% sits at the {side} of its own 10-year range "
            "({pctl} percentile) — the model's discount rate is built on top of "
            "this level.".format(
                rf=_fmt_pct(ctx.get("risk_free_pct")),
                side=side,
                pctl=_ordinal(_fmt_percentile_int(rf_pctl)),
            )
        )

    return notes


def build_macro_context(panel: Optional[dict], terminal_growth: Optional[float] = None) -> Optional[dict]:
    """Derive a valuation-relevant macro context from a FRED panel.

    Pure and deterministic: no network, no disk I/O. ``panel`` is exactly
    what :func:`sec_analyzer.fetch.fred.get_macro_panel` returns --
    ``{series_id: observation_dict_or_None}``.

    Args:
        panel: The FRED macro panel (see above). ``None``/empty is tolerated.
        terminal_growth: Optional terminal/perpetuity growth rate as a
            decimal fraction (e.g. ``0.04`` for 4%), for an informational
            cross-check against priced-in inflation (rule 4 of
            :func:`_build_notes`). This is purely diagnostic: it is only
            ECHOED BACK (as ``terminal_growth_pct``) and compared against
            breakeven inflation for a note. It never changes, overrides, or
            feeds back into the terminal growth rate the valuation engine
            actually uses -- that remains governed exclusively by
            ``valuation/SPEC.md``'s ``min(risk_free, 4%)`` anchor rule.

    Returns:
        A dict (see module-level fields below), or ``None`` when ``panel``
        is empty or every series in it is missing. Never raises.

        Fields: ``risk_free_pct``, ``risk_free_date``, ``risk_free_percentile``,
        ``curve_slope_pct`` (DGS10-DGS2, percentage points), ``curve_inverted``,
        ``long_slope_pct`` (DGS30-DGS10), ``breakeven_inflation_pct`` (T10YIE),
        ``real_rate_pct`` (DGS10-T10YIE), ``ig_spread_pct`` (BAA10Y),
        ``ig_spread_percentile``, ``hy_spread_pct`` (BAMLH0A0HYM2),
        ``hy_spread_percentile``, ``credit_regime`` (``"tight"``/``"normal"``/
        ``"loose"``/``None``), ``terminal_growth_pct``,
        ``terminal_vs_inflation_pct``, ``notes`` (list of sentences),
        ``as_of`` (newest observation date across the panel),
        ``series_available``/``series_missing`` (lists of series ids), and
        ``providers`` -- the sorted set of distinct ``"provider"`` values
        (``"FRED"``/``"Treasury"``) actually present in ``panel``'s non-``None``
        entries. Normally ``["FRED"]``; becomes (partly) ``["Treasury"]`` when
        :mod:`sec_analyzer.fetch.fred` fell back for one or more series (see
        that module's docstring -- ``BAA10Y``/``BAMLH0A0HYM2`` have no
        Treasury substitute and stay ``None`` rather than get a fabricated
        proxy, regardless of ``providers``).
        Percent values are rounded to 2 decimals, percentiles to 1. Every
        field is ``None`` when its inputs are missing -- never ``0`` as a
        stand-in.
    """
    try:
        return _build_macro_context(panel or {}, terminal_growth)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("build_macro_context() failed unexpectedly; returning None.")
        return None


def _build_macro_context(panel: dict, terminal_growth: Optional[float]) -> Optional[dict]:
    if not panel or all(v is None for v in panel.values()):
        return None

    dgs10 = panel.get(SERIES_DGS10)
    dgs2 = panel.get(SERIES_DGS2)
    dgs30 = panel.get(SERIES_DGS30)
    t10yie = panel.get(SERIES_T10YIE)
    baa10y = panel.get(SERIES_BAA10Y)
    hy = panel.get(SERIES_HY_OAS)

    risk_free_pct = _round(dgs10["value_pct"], 2) if dgs10 else None
    risk_free_date = dgs10.get("date") if dgs10 else None
    risk_free_percentile = _round(dgs10.get("percentile"), 1) if dgs10 else None

    curve_slope_pct = (
        _round(dgs10["value_pct"] - dgs2["value_pct"], 2) if dgs10 and dgs2 else None
    )
    curve_inverted = curve_slope_pct < 0 if curve_slope_pct is not None else None

    long_slope_pct = (
        _round(dgs30["value_pct"] - dgs10["value_pct"], 2) if dgs30 and dgs10 else None
    )

    breakeven_inflation_pct = _round(t10yie["value_pct"], 2) if t10yie else None

    real_rate_pct = (
        _round(dgs10["value_pct"] - t10yie["value_pct"], 2) if dgs10 and t10yie else None
    )

    ig_spread_pct = _round(baa10y["value_pct"], 2) if baa10y else None
    ig_spread_percentile = _round(baa10y.get("percentile"), 1) if baa10y else None

    hy_spread_pct = _round(hy["value_pct"], 2) if hy else None
    hy_spread_percentile = _round(hy.get("percentile"), 1) if hy else None

    credit_regime = _credit_regime(hy_spread_percentile, ig_spread_percentile)

    terminal_growth_pct = _round(terminal_growth * 100.0, 2) if terminal_growth is not None else None
    terminal_vs_inflation_pct = (
        _round(terminal_growth_pct - breakeven_inflation_pct, 2)
        if terminal_growth_pct is not None and breakeven_inflation_pct is not None
        else None
    )

    dates = [v.get("date") for v in panel.values() if v and v.get("date")]
    as_of = max(dates) if dates else None

    series_available = [series_id for series_id, v in panel.items() if v is not None]
    series_missing = [series_id for series_id, v in panel.items() if v is None]
    providers = sorted({v.get("provider") for v in panel.values() if v and v.get("provider")})

    ctx = {
        "risk_free_pct": risk_free_pct,
        "risk_free_date": risk_free_date,
        "risk_free_percentile": risk_free_percentile,
        "curve_slope_pct": curve_slope_pct,
        "curve_inverted": curve_inverted,
        "long_slope_pct": long_slope_pct,
        "breakeven_inflation_pct": breakeven_inflation_pct,
        "real_rate_pct": real_rate_pct,
        "ig_spread_pct": ig_spread_pct,
        "ig_spread_percentile": ig_spread_percentile,
        "hy_spread_pct": hy_spread_pct,
        "hy_spread_percentile": hy_spread_percentile,
        "credit_regime": credit_regime,
        "terminal_growth_pct": terminal_growth_pct,
        "terminal_vs_inflation_pct": terminal_vs_inflation_pct,
        "as_of": as_of,
        "series_available": series_available,
        "series_missing": series_missing,
        "providers": providers,
    }
    ctx["notes"] = _build_notes(ctx)
    return ctx


def summarize_macro(context: Optional[dict]) -> str:
    """Render a compact one-line summary of a macro context.

    Suitable for the verdict card's future "Macro:" row, e.g.::

        "10y 4.21% (62nd percentile of 10y) · curve +0.35 · HY spread 3.18%
        (18th percentile, loose)"

    When ``context["providers"]`` shows the FRED-with-Treasury-fallback
    chain (:mod:`sec_analyzer.fetch.fred`) actually used Treasury for one or
    more series, a trailing ``" (via Treasury)"`` is appended so a fallback
    is never silently invisible in the summary line. Nothing is appended in
    the normal (FRED-only) case, so the everyday line stays uncluttered.

    Returns ``"—"`` for ``None``/empty ``context``. Mirrors
    :func:`sec_analyzer.signals.events.summarize_events`'s defensiveness:
    never emits ``"None"``/``"nan"``, never raises.
    """
    if not context:
        return "—"

    try:
        parts: List[str] = []

        risk_free_pct = context.get("risk_free_pct")
        if risk_free_pct is not None:
            piece = f"10y {_fmt_pct(risk_free_pct)}%"
            pctl = _fmt_percentile_int(context.get("risk_free_percentile"))
            if pctl is not None:
                piece += f" ({_ordinal(pctl)} percentile of 10y)"
            parts.append(piece)

        curve_slope_pct = context.get("curve_slope_pct")
        if curve_slope_pct is not None:
            sign = "+" if curve_slope_pct >= 0 else ""
            parts.append(f"curve {sign}{_fmt_pct(curve_slope_pct)}")

        hy_spread_pct = context.get("hy_spread_pct")
        ig_spread_pct = context.get("ig_spread_pct")
        if hy_spread_pct is not None:
            piece = f"HY spread {_fmt_pct(hy_spread_pct)}%"
            extras = []
            pctl = _fmt_percentile_int(context.get("hy_spread_percentile"))
            if pctl is not None:
                extras.append(f"{_ordinal(pctl)} percentile")
            if context.get("credit_regime") is not None:
                extras.append(context["credit_regime"])
            if extras:
                piece += f" ({', '.join(extras)})"
            parts.append(piece)
        elif ig_spread_pct is not None:
            piece = f"IG spread {_fmt_pct(ig_spread_pct)}%"
            extras = []
            pctl = _fmt_percentile_int(context.get("ig_spread_percentile"))
            if pctl is not None:
                extras.append(f"{_ordinal(pctl)} percentile")
            if context.get("credit_regime") is not None:
                extras.append(context["credit_regime"])
            if extras:
                piece += f" ({', '.join(extras)})"
            parts.append(piece)

        if not parts:
            return "—"

        line = " · ".join(parts)
        if "Treasury" in (context.get("providers") or []):
            line += " (via Treasury)"
        return line
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("summarize_macro() failed unexpectedly; returning placeholder.")
        return "—"
