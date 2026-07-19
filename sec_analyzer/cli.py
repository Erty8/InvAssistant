"""Command-line entry point for sec_analyzer.

Two subcommands:

* ``fetch TICKER`` -- resolve the ticker to a CIK, pull SEC XBRL company
  facts, normalize them, compute ratios, print both, and persist everything
  to the local SQLite database.
* ``analyze TICKER`` -- everything ``fetch`` does, plus price/technical data,
  valuation metrics, red flags, an earnings-date estimate, and a full
  fundamental+technical interpretation (fair-value range, verdicts,
  cyclicality, and a summary) from a selectable backend: a deterministic
  script-based (no-AI) analyzer (default), a local Ollama/Gemma model, or the
  hosted Anthropic Claude API. The result is printed as a compact Turkish-language
  verdict card and, optionally, saved as a standalone HTML report.

Usage::

    python -m sec_analyzer.cli fetch AAPL --years 5
    python -m sec_analyzer.cli analyze AAPL
    python -m sec_analyzer.cli analyze AAPL --horizon 5y --provider script
    python -m sec_analyzer.cli analyze AAPL --html

Only the official SEC EDGAR API and (for ``analyze`` with the ``ollama`` or
``anthropic`` providers) an LLM API are used for financial statement data --
no third-party finance data libraries beyond the optional Stooq/yfinance
price-history fetch used to power the technical-analysis layer.
"""

import argparse
import json
import logging
import sys
from datetime import date
from typing import List, Optional, Tuple

import requests

from sec_analyzer.calibrate import (
    DEFAULT_TICKERS,
    print_calibration_table,
    run_calibration,
    save_calibration_snapshot,
    summarize_ratios,
)
from sec_analyzer.config import Config, ConfigError
from sec_analyzer.fetch.analyst import get_analyst_targets
from sec_analyzer.fetch.companyfacts import get_company_facts, get_submissions
from sec_analyzer.fetch.filings import estimate_next_earnings
from sec_analyzer.fetch.prices import PriceDataError, get_price_history, latest_price
from sec_analyzer.fetch.tickers import resolve_cik
from sec_analyzer.http_client import SecHttpClient
from sec_analyzer.interpret.analyzer import interpret
from sec_analyzer.normalize.metrics import compute_metrics
from sec_analyzer.normalize.normalizer import format_table, normalize_facts
from sec_analyzer.normalize.ratios import compute_ratios
from sec_analyzer.normalize.red_flags import detect_red_flags
from sec_analyzer.report.generator import generate_report
from sec_analyzer.signals.events import detect_events, summarize_events
from sec_analyzer.store.database import save_normalized, save_prices, save_verdict
from sec_analyzer.technical.indicators import compute_indicators
from sec_analyzer.technical.verdict import technical_verdict

logger = logging.getLogger(__name__)

#: Column width used when rendering the ratios table in _print_ratios.
_RATIO_COL_WIDTH = 14

#: Placeholder shown for any missing/None value in the terminal verdict card.
_DASH = "—"

#: Label column width for the verdict card's aligned "Label: value" lines
#: (e.g. "Fundamental:", "Teknik:", ...) -- chosen so every label lines up
#: regardless of its own length.
_CARD_LABEL_WIDTH = 13


def _print_ratios(ratios: List[dict]) -> None:
    """Render the per-fiscal-year ratio list as a compact aligned table.

    Net margin and the two YoY growth ratios are shown as percentages; ROE
    and the current ratio are shown as plain decimals. Missing (``None``)
    values are rendered as ``-``.

    Args:
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
    """
    if not ratios:
        print("No ratios available (insufficient annual data).")
        return

    def as_pct(value) -> str:
        return f"{value * 100:.1f}%" if value is not None else "-"

    def as_dec(value) -> str:
        return f"{value:.2f}" if value is not None else "-"

    headers = ["FY", "Net Margin", "ROE", "Current Ratio", "Rev YoY", "NI YoY"]
    print("".join(h.rjust(_RATIO_COL_WIDTH) for h in headers))
    print("-" * (_RATIO_COL_WIDTH * len(headers)))

    for row in ratios:
        cells = [
            str(row.get("fy", "-")),
            as_pct(row.get("net_margin")),
            as_dec(row.get("roe")),
            as_dec(row.get("current_ratio")),
            as_pct(row.get("yoy_revenue_growth")),
            as_pct(row.get("yoy_net_income_growth")),
        ]
        print("".join(c.rjust(_RATIO_COL_WIDTH) for c in cells))


def _fetch_normalize_store(args: argparse.Namespace) -> Tuple[str, str, dict, List[dict]]:
    """Resolve, fetch, normalize, print, and persist financials for a ticker.

    Shared by both ``fetch`` and ``analyze`` so the two subcommands stay in
    sync and ``analyze`` doesn't duplicate any of this logic.

    Args:
        args: Parsed CLI arguments; must have ``ticker``, ``years``, and
            ``no_cache`` attributes.

    Returns:
        ``(cik, name, normalized, ratios)``.
    """
    client = SecHttpClient()

    cik, name = resolve_cik(args.ticker, client, no_cache=args.no_cache)
    logger.info("Resolved ticker %s -> CIK %s (%s)", args.ticker, cik, name)

    facts = get_company_facts(cik, client, no_cache=args.no_cache)
    normalized = normalize_facts(facts, years=args.years)
    ratios = compute_ratios(normalized)

    print(format_table(normalized))
    print()
    _print_ratios(ratios)

    save_normalized(args.ticker, cik, name, normalized, ratios, db_path=Config.DB_PATH)
    print(f"\nSaved to database: {Config.DB_PATH}")

    return cik, name, normalized, ratios


def cmd_fetch(args: argparse.Namespace) -> None:
    """Handle the ``fetch`` subcommand: fetch, normalize, and store only."""
    _fetch_normalize_store(args)


def _fetch_price_and_technical(
    ticker: str, horizon: str, no_cache: bool
):
    """Fetch price history and derive the merged technical indicators/verdict.

    Fully graceful: if price data can't be obtained (Stooq and yfinance both
    fail, or the ticker simply has too little history), this logs a warning
    and returns ``(None, None, None, None)`` rather than raising -- the
    fundamental side of ``analyze`` must keep working even when there's no
    usable price data at all.

    Returns:
        ``(price, as_of, technical, price_df)`` where ``technical`` is the
        merged ``{**indicators, **technical_verdict_result}`` dict expected
        by :func:`sec_analyzer.interpret.analyzer.interpret`, and
        ``price_df`` is the raw OHLCV DataFrame (kept so the caller can
        persist it without re-fetching). All four are ``None`` if price data
        is unavailable.
    """
    try:
        price_df, source = get_price_history(ticker, no_cache=no_cache)
        price, as_of = latest_price(price_df)
        indicators = compute_indicators(price_df)
        verdict_result = technical_verdict(indicators, horizon)
        technical = {**indicators, **verdict_result}
        logger.info(
            "Price data for %s from %s: %.2f as of %s", ticker, source, price, as_of
        )
        return price, as_of, technical, price_df
    except PriceDataError as exc:
        logger.warning("Price data unavailable for %s: %s", ticker, exc)
        return None, None, None, None


def _fetch_analyst_targets(ticker: str, no_cache: bool) -> Optional[dict]:
    """Best-effort fetch of consensus analyst price targets; never raises.

    Display-only cross-check (see :mod:`sec_analyzer.fetch.analyst`) -- never
    feeds the valuation engine. Any failure is logged and swallowed so it
    never blocks the rest of ``analyze``.

    Returns:
        The dict returned by
        :func:`sec_analyzer.fetch.analyst.get_analyst_targets`, or ``None``
        if unavailable or the fetch fails for any reason.
    """
    try:
        return get_analyst_targets(ticker, no_cache=no_cache)
    except Exception:  # noqa: BLE001 - a display-only cross-check must never be fatal
        logger.warning("Could not fetch analyst targets for %s", ticker, exc_info=True)
        return None


def _fetch_submissions(cik: str, ticker: str, no_cache: bool) -> Optional[dict]:
    """Best-effort fetch of a filer's raw SEC submissions document; never raises.

    Fetched exactly once per ``analyze`` run (SPEC.md Sec.13) and reused for
    both the next-earnings catalyst estimate (:func:`_fetch_catalyst`) and
    SIC-based sector classification (passed straight through to
    :func:`sec_analyzer.interpret.analyzer.interpret` as ``submissions=``).

    Returns:
        The dict returned by
        :func:`sec_analyzer.fetch.companyfacts.get_submissions`, or ``None``
        if the fetch fails for any reason.
    """
    try:
        client = SecHttpClient()
        return get_submissions(cik, client, no_cache=no_cache)
    except Exception:  # noqa: BLE001 - submissions are best-effort, never fatal
        logger.warning("Could not fetch SEC submissions for %s", ticker, exc_info=True)
        return None


def _fetch_catalyst(submissions: Optional[dict], ticker: str) -> Optional[dict]:
    """Best-effort next-earnings estimate from already-fetched submissions; never raises.

    Args:
        submissions: The dict returned by :func:`_fetch_submissions`, or
            ``None``.
        ticker: Stock ticker symbol, used only for the warning log message.

    Returns:
        The dict returned by
        :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
        ``None`` if ``submissions`` is unavailable or the estimate itself
        fails for any reason.
    """
    if not submissions:
        return None
    try:
        return estimate_next_earnings(submissions)
    except Exception:  # noqa: BLE001 - a catalyst estimate is a nice-to-have, never fatal
        logger.warning("Could not estimate next earnings date for %s", ticker, exc_info=True)
        return None


#: Lookback window (days) for the recent-8-K event signal, and the minimum
#: severity worth surfacing. Routine "info" filings (earnings releases, votes,
#: Reg FD, exhibit-only 8-Ks) are dropped so the card/report only shows events
#: a human should actually weigh against the numbers.
_EVENTS_LOOKBACK_DAYS = 365
_EVENTS_MIN_SEVERITY = "warning"
_EVENTS_MAX = 8


def _detect_filing_events(submissions: Optional[dict]) -> List[dict]:
    """Best-effort recent 8-K event signal from already-fetched submissions.

    Reuses the ``submissions`` document :func:`_fetch_submissions` fetched
    once for this run (no extra network, no document download, no LLM). Only
    warning/critical events within the last year are surfaced. ``detect_events``
    is itself never-raises, so this simply returns an empty list when there's
    nothing to show.
    """
    if not submissions:
        return []
    return detect_events(
        submissions,
        lookback_days=_EVENTS_LOOKBACK_DAYS,
        min_severity=_EVENTS_MIN_SEVERITY,
        max_events=_EVENTS_MAX,
    )


def _save_price_rows(cik: str, price_df) -> None:
    """Convert a price-history DataFrame to row dicts and persist them.

    Never raises: a failure to persist price history must not prevent the
    rest of ``analyze`` (interpretation, verdict card, HTML report) from
    completing.
    """
    try:
        rows = [
            {
                "date": row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }
            for _, row in price_df.reset_index().iterrows()
        ]
        save_prices(cik, rows, db_path=Config.DB_PATH)
    except Exception:  # noqa: BLE001 - persistence failure must not be fatal
        logger.warning("Failed to save price history for CIK %s", cik, exc_info=True)


def _fmt_money(value) -> str:
    """Format a number as a dollar amount for the terminal verdict card.

    No decimals for a whole number (``"$115"``), 2 decimals otherwise
    (``"$128.40"``). Returns :data:`_DASH` for ``None``/unparseable input.
    """
    if value is None:
        return _DASH
    try:
        value = float(value)
    except (TypeError, ValueError):
        return _DASH
    if value == int(value):
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def _dr_suffix(value) -> str:
    """Append a trailing ``" dr"`` to a discount-rate string like ``"%12"``
    -> ``"%12 dr"``, unless it already reads that way. ``discount_rate`` (and
    ``growth``) values are always already-formatted, human-readable Turkish
    strings produced upstream (e.g. by
    :mod:`sec_analyzer.interpret.rule_based`) -- this never reformats the
    number itself, only appends the unit label. Returns :data:`_DASH` for
    ``None``.
    """
    if value is None:
        return _DASH
    text = str(value)
    lowered = text.lower()
    if "dr" in lowered or "iskonto" in lowered:
        return text
    return f"{text} dr"


def _strip_leading_dollar(text: str) -> str:
    """Drop a leading ``"$"`` from an already-formatted money string."""
    return text[1:] if text.startswith("$") else text


def _scenario_line(label: str, scenario: Optional[dict]) -> str:
    """Render one bear/bull scenario, e.g. ``"bear $70–95 (%8 büyüme, %12 dr)"``."""
    scenario = scenario or {}
    lo, hi = scenario.get("lo"), scenario.get("hi")
    if lo is None or hi is None:
        range_text = _DASH
    else:
        range_text = f"{_fmt_money(lo)}–{_strip_leading_dollar(_fmt_money(hi))}"
    growth = scenario.get("growth") or _DASH
    dr = _dr_suffix(scenario.get("discount_rate"))
    return f"{label} {range_text} ({growth}, {dr})"


def _analyst_line(analyst: Optional[dict], price) -> Optional[str]:
    """Render the display-only consensus analyst-target line for the verdict
    card, e.g. ``"Analist:     $128 ort (34 analist) · +%12 · aralık $95–$160"``.

    This is a reference cross-check only (see
    :mod:`sec_analyzer.fetch.analyst`) -- it never feeds the valuation
    engine. ``None`` when ``analyst`` is falsy or has no usable
    ``target_mean``.

    Args:
        analyst: The dict returned by
            :func:`sec_analyzer.fetch.analyst.get_analyst_targets`, or
            ``None``.
        price: The latest market price per share (used only to compute the
            upside vs. the consensus mean), or ``None``.
    """
    if not analyst or analyst.get("target_mean") is None:
        return None

    target_mean = analyst["target_mean"]
    parts = [f"{_fmt_money(target_mean)} ort"]

    num_analysts = analyst.get("num_analysts")
    if num_analysts is not None:
        parts.append(f"({num_analysts} analist)")

    try:
        price_is_positive = price is not None and float(price) > 0
    except (TypeError, ValueError):
        price_is_positive = False
    if price_is_positive:
        upside = (target_mean / float(price) - 1) * 100
        parts.append(f"· {_signed_pct_tr(upside)}")

    target_low, target_high = analyst.get("target_low"), analyst.get("target_high")
    if target_low is not None and target_high is not None:
        parts.append(f"· aralık {_fmt_money(target_low)}–{_fmt_money(target_high)}")

    label = "Analist:".ljust(_CARD_LABEL_WIDTH)
    return f"{label}{' '.join(parts)}"


def _valuation_method_label(valuation: dict) -> str:
    """Return the fair-value method label for the card's "Fair Value" line.

    Checked in order, depending on which anchor actually produced the
    headline fair-value band for this filer (SPEC.md Sec.8/8e/11):

    - ``"Revenue-DCF (hiper-büyüme)"``: the hyper-grower revenue-first DCF is
      the headline (``valuation["hyper_growth"]`` truthy and its
      ``hyper_growth_detail`` present and not suppressed -- SPEC.md Sec.3.5,
      which takes precedence over every anchor below).
    - ``"FCFE (kazanç+büyüme)"``: the cyclical sustainable-growth FCFE
      anchor is the headline (``valuation["cyclical_fcfe_headline"]`` --
      SPEC.md Sec.8e, e.g. Micron-shaped capital-intensive cyclicals whose
      FCF-DCF is suppressed by growth CapEx).
    - ``"EPV"``: the zero-growth earnings-power-value anchor is the headline
      (``valuation["earnings_power_headline"]`` -- SPEC.md Sec.8a, e.g.
      Amazon-shaped mature filers, or a cyclical whose FCFE anchor couldn't
      clear the EPV floor).
    - ``"Revenue-DCF"``: a revenue-first DCF is the headline
      (``valuation["mature_revenue_headline"]`` or
      ``["midgrowth_revenue_headline"]``).
    - ``"DCF"``: the FCF-based DCF is enabled (``valuation["dcf"]["enabled"]``
      is truthy) -- the common case for non-financial, non-REIT sectors when
      none of the anchors above headlined.
    - ``"FFO"``: the DCF is disabled and ``valuation["ffo"]`` is a populated
      FFO-based Gordon growth block (has a ``"scenarios"`` key) -- REIT/GYO
      filers valued via FFO multiples instead of a cash-flow DCF.
    - ``"P/B×ROE"``: the DCF is disabled and there's no populated FFO block --
      financial (banks/insurers) filers valued via the P/B x ROE anchor, or a
      REIT that fell back to the P/B x ROE anchor without a populated FFO
      block.
    """
    valuation = valuation or {}
    # Hyper-grower revenue-first DCF takes precedence over every other anchor
    # (SPEC.md Sec.3.5). It's the headline whenever it was detected and its
    # detail wasn't suppressed; the standard FCF-DCF is still computed as a
    # secondary (so ``dcf.enabled`` stays truthy) -- without this check a
    # hyper-headlined filer (e.g. Reddit) would mislabel as "DCF".
    hyper_detail = valuation.get("hyper_growth_detail") or {}
    if valuation.get("hyper_growth") and hyper_detail and not hyper_detail.get("suppressed"):
        return "Revenue-DCF (hiper-büyüme)"
    if valuation.get("cyclical_fcfe_headline"):
        return "FCFE (kazanç+büyüme)"
    if valuation.get("earnings_power_headline"):
        return "EPV"
    if valuation.get("mature_revenue_headline") or valuation.get("midgrowth_revenue_headline"):
        return "Revenue-DCF"
    dcf = valuation.get("dcf") or {}
    if dcf.get("enabled", True):
        return "DCF"
    ffo = valuation.get("ffo")
    if isinstance(ffo, dict) and "scenarios" in ffo:
        return "FFO"
    return "P/B×ROE"


def _format_pct_signed(value) -> str:
    """Render a decimal-fraction growth rate as a whole-percent Turkish
    string, e.g. ``0.19 -> "%19"``, ``-0.05 -> "%-5"``. Returns :data:`_DASH`
    for ``None``/non-numeric input."""
    if value is None:
        return _DASH
    try:
        return f"%{float(value) * 100:.0f}"
    except (TypeError, ValueError):
        return _DASH


def _reverse_dcf_line(result: dict, valuation: dict) -> str:
    """Render the "Reverse DCF:" card line (SPEC.md Sec.13): the price-
    implied growth rate versus the filer's realized revenue CAGR.

    Prefers the phase-2 commentary's own ``result["reverse_dcf_comment"]``
    when present -- it already narrates the same
    ``valuation["reverse_dcf"]`` numbers in prose. Falls back to building
    the line directly from those numbers otherwise (e.g. the ``"script"``
    provider, or a commentary that left the field empty).
    """
    label = "Reverse DCF:".ljust(_CARD_LABEL_WIDTH)
    comment = result.get("reverse_dcf_comment")
    if comment:
        return f"{label}{comment}"

    reverse_dcf = valuation.get("reverse_dcf") or {}
    implied = reverse_dcf.get("implied_growth")
    if implied is None:
        return f"{label}{_DASH}"

    realized_label = reverse_dcf.get("realized_label") or _DASH
    realized_text = _format_pct_signed(reverse_dcf.get("realized_cagr_5y"))
    return (
        f"{label}fiyat 10y {_format_pct_signed(implied)} CAGR ima ediyor "
        f"(gerçekleşen {realized_label}: {realized_text})"
    )


def _multiples_line(valuation: dict) -> str:
    """Render the "Multiples:" card line (SPEC.md Sec.13).

    When a growth-adjusted multiple (PEG in standard mode, growth-adjusted
    EV/Sales in hyper-grower mode) is applicable, renders the compact
    two-component form -- raw percentile · growth-adjusted value (percentile)
    -- appending "→ karışık sinyal" when the triangulation multiples signal
    is ``"karisik"`` (the two components disagree on direction, VALUATION.md
    Sec.7), e.g. ``"P/E 88. pctile · PEG 1.4 (45. pctile) → karışık
    sinyal"``. When the growth-adjusted figure is not applicable, falls back
    to the original descriptive form (``"P/E kendi Ny medyanının N.
    yüzdeliğinde"``). ``"veri yetersiz"`` when no raw percentile is usable
    (fewer than 5 years of price-backed history).

    For ``valuation["sector_type"] == "reit"``, P/E and P/FCF (and the
    P/E-derived PEG) are meaningless -- GAAP depreciation distorts REIT
    earnings the same way it distorts book equity (SPEC.md Sec.8c), which is
    why REITs get their own FFO-based anchor. The reit primary multiple is
    P/FFO, falling back to P/S (mirroring
    ``triangulate._raw_multiples_signal``'s reit candidate order); no
    growth-adjusted component is ever rendered for reit."""
    label = "Multiples:".ljust(_CARD_LABEL_WIDTH)
    multiples = valuation.get("multiples") or {}
    history_years = multiples.get("history_years") or 0
    is_reit = valuation.get("sector_type") == "reit"
    growth_adjusted = {} if is_reit else (multiples.get("growth_adjusted") or {})
    multiples_signal = ((valuation.get("triangulation") or {}).get("signals") or {}).get("multiples")

    # Primary raw multiple. Reit uses P/FFO -> P/S only (P/E and P/FCF are
    # meaningless for REITs); everything else keeps the P/E -> P/S -> P/FCF
    # fallback order.
    primary = None
    if is_reit:
        primary_candidates = (
            ("P/FFO", multiples.get("pffo_percentile")),
            ("P/S", multiples.get("ps_percentile")),
        )
    else:
        primary_candidates = (
            ("P/E", multiples.get("pe_percentile")),
            ("P/S", multiples.get("ps_percentile")),
            ("P/FCF", multiples.get("pfcf_percentile")),
        )
    for name, percentile in primary_candidates:
        if percentile is not None:
            primary = (name, percentile)
            break

    if growth_adjusted.get("applicable"):
        raw_name = growth_adjusted.get("raw_label")
        raw_pct = growth_adjusted.get("raw_percentile")
        if raw_pct is None and primary is not None:
            raw_name, raw_pct = primary

        parts = []
        if raw_pct is not None:
            parts.append(f"{raw_name} {raw_pct:.0f}. pctile")

        ga_label = growth_adjusted.get("label") or "PEG"
        ga_value = growth_adjusted.get("value")
        ga_pct = growth_adjusted.get("percentile")
        if ga_value is not None and ga_pct is not None:
            parts.append(f"{ga_label} {ga_value:.2f} ({ga_pct:.0f}. pctile)")
        elif ga_value is not None:
            parts.append(f"{ga_label} {ga_value:.2f}")

        if parts:
            line = " · ".join(parts)
            if multiples_signal == "karisik":
                line += " → karışık sinyal"
            return f"{label}{line}"

    if primary is not None:
        name, percentile = primary
        return f"{label}{name} kendi {history_years}y medyanının {percentile:.0f}. yüzdeliğinde"

    return f"{label}veri yetersiz"


def _triangulation_line(valuation: dict) -> str:
    """Render the "Üçgenleme:" card line (SPEC.md Sec.13): each of the three
    valuation methods' cheap/fair/expensive direction signal, plus a
    confidence-derived closing phrase, from ``valuation["triangulation"]``.
    ``"YÜKSEK"`` confidence reads as "yön net" (clear direction), ``"DÜŞÜK"``
    as "yön karışık" (mixed signals); anything else (typically ``"ORTA"``)
    falls back to the triangulation's own majority ``direction`` word."""
    label = "Üçgenleme:".ljust(_CARD_LABEL_WIDTH)
    triangulation = valuation.get("triangulation") or {}
    signals = triangulation.get("signals") or {}
    confidence = triangulation.get("confidence")

    dcf_signal = signals.get("dcf") or _DASH
    reverse_dcf_signal = signals.get("reverse_dcf") or _DASH
    multiples_signal = signals.get("multiples") or _DASH

    if confidence == "YÜKSEK":
        suffix = "yön net"
    elif confidence == "DÜŞÜK":
        suffix = "yön karışık"
    else:
        suffix = triangulation.get("direction") or _DASH

    return f"{label}DCF {dcf_signal} · rDCF {reverse_dcf_signal} · multiples {multiples_signal} → {suffix}"


def _sensitivity_line(valuation: dict) -> str:
    """Render the "Duyarlılık:" card line (SPEC.md Sec.13): the full price
    range spanned by the base-scenario 3x3 growth/discount-rate sensitivity
    matrix, from ``valuation["sensitivity"]``, appending a "yüksek
    belirsizlik" flag when that matrix's own ``high_uncertainty`` bit is
    set (its (hi-lo)/base-cell spread exceeds 60%)."""
    label = "Duyarlılık:".ljust(_CARD_LABEL_WIDTH)
    sensitivity = valuation.get("sensitivity") or {}
    lo, hi = sensitivity.get("lo"), sensitivity.get("hi")
    if lo is None or hi is None:
        return f"{label}{_DASH}"

    range_text = f"{_fmt_money(lo)}–{_fmt_money(hi)}"
    line = f"{label}base {range_text} (g±2pp, r±1pp)"
    if sensitivity.get("high_uncertainty"):
        line += " — yüksek belirsizlik"
    return line


def _signed_pct_tr(value) -> str:
    """Turkish signed-percent, sign-first then ``%``: ``4 -> "+%4"``,
    ``-7.5 -> "-%8"`` (0 decimals, matching the compact card style)."""
    if value is None:
        return _DASH
    sign = "+" if value >= 0 else "-"
    return f"{sign}%{abs(value):.0f}"


def _momentum_line(technical: dict) -> Optional[str]:
    """Compact momentum sub-line for the technical card, e.g.
    ``"Momentum:  1a +%4 · 3a +%13 · 6a -%3 · Trend: yükseliş (GC) · 52h %68"``.
    ``None`` when no momentum figure is available."""
    parts: List[str] = []
    for label, key in (("1a", "return_1m_pct"), ("3a", "return_3m_pct"), ("6a", "return_6m_pct")):
        value = technical.get(key)
        if value is not None:
            parts.append(f"{label} {_signed_pct_tr(value)}")

    above = technical.get("sma50_above_sma200")
    if above is True:
        parts.append("Trend: yükseliş" + (" (GC)" if technical.get("golden_cross") else ""))
    elif above is False:
        parts.append("Trend: düşüş" + (" (DC)" if technical.get("death_cross") else ""))

    range_position = technical.get("range_position_pct")
    if range_position is not None:
        parts.append(f"52h %{range_position:.0f}")

    if not parts:
        return None
    return "Momentum:  " + " · ".join(parts)


def _rsi_divergence_line(technical: dict) -> Optional[str]:
    """Explanatory RSI-divergence sub-line (a reversal warning worth spelling
    out), or ``None`` when there's no divergence."""
    detail = technical.get("rsi_divergence_detail")
    if not detail:
        return None
    price_prev, price_last = _fmt_money(detail.get("price_prev")), _fmt_money(detail.get("price_last"))
    rsi_prev, rsi_last = detail.get("rsi_prev"), detail.get("rsi_last")
    if detail.get("type") == "bearish":
        return (
            f"RSI uyumsuzluğu (ayı): fiyat {price_prev}→{price_last} daha yüksek zirve ama "
            f"RSI {rsi_prev:.0f}→{rsi_last:.0f} düştü — momentum teyit etmiyor."
        )
    return (
        f"RSI uyumsuzluğu (boğa): fiyat {price_prev}→{price_last} daha düşük dip ama "
        f"RSI {rsi_prev:.0f}→{rsi_last:.0f} yükseldi — satış baskısı azalıyor."
    )


def _signal_volume_line(technical: dict) -> Optional[str]:
    """Compact MACD + volume sub-line, e.g.
    ``"MACD/Hacim:  MACD boğa (kesişim ↑) · Hacim 1.3× · OBV ↑"``. ``None``
    when neither MACD nor volume signals are available."""
    parts: List[str] = []

    macd_hist = technical.get("macd_hist")
    if macd_hist is not None:
        state = "boğa" if macd_hist > 0 else ("ayı" if macd_hist < 0 else "nötr")
        segment = f"MACD {state}"
        cross = technical.get("macd_cross")
        if cross == "bullish":
            segment += " (kesişim ↑)"
        elif cross == "bearish":
            segment += " (kesişim ↓)"
        parts.append(segment)

    rel_volume = technical.get("rel_volume")
    if rel_volume is not None:
        parts.append(f"Hacim {rel_volume:.1f}×")

    obv_trend = technical.get("obv_trend")
    if obv_trend:
        parts.append({"up": "OBV ↑", "down": "OBV ↓", "flat": "OBV yatay"}.get(obv_trend, f"OBV {obv_trend}"))

    if not parts:
        return None
    return "MACD/Hacim:  " + " · ".join(parts)


def _support_resistance_line(technical: dict) -> Optional[str]:
    """Compact support/resistance sub-line for the technical card, e.g.
    ``"Destek/Direnç:  Direnç $108 (+%8), $118 (+%18) | Destek $92.50 (-%8)"``.
    Shows up to two nearest levels per side. ``None`` when none exist."""
    def _range(z: dict) -> str:
        lo, hi = z.get("low"), z.get("high")
        if lo is not None and hi is not None and (hi - lo) > 0.01:
            return f"{_fmt_money(lo)}–{_fmt_money(hi)}"
        return _fmt_money(z.get("price") if z.get("price") is not None else lo)

    def _one(z: dict) -> str:
        touches = z.get("touches") or 0
        parts = []
        if z.get("is_52w_high"):
            parts.append("52h zirve")
        elif z.get("is_52w_low"):
            parts.append("52h dip")
        if touches >= 1:
            parts.append(f"{touches}×")
        if z.get("fib"):
            parts.append(f"Fib {z['fib']}")
        note = " + ".join(parts) or "seviye"
        return f"{_range(z)} ({_signed_pct_tr(z.get('dist_pct'))} · {note})"

    def _levels(items: list) -> str:
        return ", ".join(_one(z) for z in items[:2])

    resistances = technical.get("resistance_levels") or []
    supports = technical.get("support_levels") or []
    segments: List[str] = []
    if resistances:
        segments.append(f"Direnç {_levels(resistances)}")
    if supports:
        segments.append(f"Destek {_levels(supports)}")
    if not segments:
        return None
    return "Destek/Direnç:  " + " | ".join(segments)


def _print_verdict_card(
    ticker: str,
    horizon: str,
    result: dict,
    metrics: Optional[dict] = None,
    flags: Optional[List[dict]] = None,
    technical: Optional[dict] = None,
    analyst: Optional[dict] = None,
) -> None:
    """Print the compact, Turkish-language terminal verdict card.

    This is the default (non-``--verbose``) output of ``analyze`` -- it
    replaces the old raw-JSON dump, which is now only shown behind
    ``--verbose``. Every field is rendered defensively (:data:`_DASH` for
    anything missing), and an ``{"error": ...}`` result renders a minimal
    card (ticker/horizon/date/price plus the error) instead of crashing.

    When ``result["valuation"]`` (the dict from
    :func:`sec_analyzer.valuation.engine.run_valuation`, SPEC.md Sec.13) is
    present, the "Fair Value" line gains a method label (``"DCF"``, ``"FFO"``,
    or ``"P/B×ROE"`` -- see :func:`_valuation_method_label`) and a "Güven:"
    confidence suffix, and four extra lines
    follow the bear/bull line: "Reverse DCF:", "Multiples:", "Üçgenleme:",
    and "Duyarlılık:". Without a ``"valuation"`` key (e.g. an older stored
    result, or a phase-2 provider failure that still reached this function)
    the card renders exactly as it did before those additions.

    Args:
        ticker: Stock ticker symbol as typed by the user (rendered upper-case).
        horizon: One of ``"3m"``, ``"1y"``, ``"5y"``.
        result: The dict returned by
            :func:`sec_analyzer.interpret.analyzer.interpret` (success or
            error shape).
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`; its
            ``"price"`` field is used for the "Fiyat:" line.
        flags: The list returned by
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, used
            as a fallback "Red flags:" line when ``result`` has no
            ``red_flags_comment`` (e.g. an error result).
        analyst: The dict returned by
            :func:`sec_analyzer.fetch.analyst.get_analyst_targets`, or
            ``None``. Display-only consensus cross-check, rendered as an
            "Analist:" line right after the bear/bull scenario line (see
            :func:`_analyst_line`); never feeds the valuation engine.

    The "Olaylar:" line summarizes ``result["events"]`` (the recent 8-K event
    list attached in ``cmd_analyze`` via
    :func:`sec_analyzer.signals.events.detect_events`) using
    :func:`sec_analyzer.signals.events.summarize_events`; it reads ``"yok"``
    when there are no recent warning/critical events.
    """
    result = result or {}
    metrics = metrics or {}
    flags = flags or []

    ticker_label = str(ticker).upper()
    header = f"{ticker_label} — Vade: {horizon} — {date.today().isoformat()}"

    print()
    print(header)
    print("─" * len(header))
    print(f"Fiyat: {_fmt_money(metrics.get('price'))}")

    if "error" in result:
        print(f"Analiz kullanılamıyor ({result['error']}): {result.get('summary', _DASH)}")
        return

    fv = result.get("fair_value_range") or {}
    base = fv.get("base") or {}
    valuation = result.get("valuation")

    if base.get("lo") is not None and base.get("hi") is not None:
        base_range = f"{_fmt_money(base['lo'])}–{_fmt_money(base['hi'])}"
    else:
        base_range = _DASH

    if valuation:
        method_label = _valuation_method_label(valuation)
        confidence = result.get("confidence") or _DASH
        print(f"Fair Value (base, {method_label}): {base_range}   Güven: {confidence}")
    else:
        print(f"Fair Value (base): {base_range}")
    print(f"  {_scenario_line('bear', fv.get('bear'))} | {_scenario_line('bull', fv.get('bull'))}")

    analyst_line = _analyst_line(analyst, metrics.get('price'))
    if analyst_line:
        print(f"  {analyst_line}")

    if valuation:
        print(_reverse_dcf_line(result, valuation))
        print(_multiples_line(valuation))
        print(_triangulation_line(valuation))
        print(_sensitivity_line(valuation))

    print(f"{'Fundamental:'.ljust(_CARD_LABEL_WIDTH)}{result.get('fundamental_verdict') or _DASH}")

    technical = technical or {}
    technical_verdict = result.get("technical_verdict") or technical.get("verdict") or _DASH
    verdict_detail = technical.get("verdict_detail")
    technical_line = technical_verdict
    if verdict_detail and verdict_detail != "yetersiz veri":
        technical_line = f"{technical_verdict} ({verdict_detail})"
    print(f"{'Teknik:'.ljust(_CARD_LABEL_WIDTH)}{technical_line}")

    momentum_line = _momentum_line(technical)
    if momentum_line:
        print(f"  {momentum_line}")
    signal_line = _signal_volume_line(technical)
    if signal_line:
        print(f"  {signal_line}")
    divergence_line = _rsi_divergence_line(technical)
    if divergence_line:
        print(f"  {divergence_line}")
    sr_line = _support_resistance_line(technical)
    if sr_line:
        print(f"  {sr_line}")

    profile = result.get("profile_fit") or {}
    profile_verdict = profile.get("verdict") or _DASH
    profile_reason = profile.get("reason")
    profile_line = f"{profile_verdict} — {profile_reason}" if profile_reason else profile_verdict
    print(f"{'Profil:'.ljust(_CARD_LABEL_WIDTH)}{profile_line}")

    if "red_flags_comment" in result:
        red_flags_line = result.get("red_flags_comment") or "yok"
    elif flags:
        red_flags_line = "; ".join(f.get("message", "") for f in flags if f.get("message")) or "yok"
    else:
        red_flags_line = "yok"
    print(f"{'Red flags:'.ljust(_CARD_LABEL_WIDTH)}{red_flags_line}")

    events_line = summarize_events(result.get("events") or [])
    print(f"{'Olaylar:'.ljust(_CARD_LABEL_WIDTH)}{events_line}")

    print(f"{'Katalizör:'.ljust(_CARD_LABEL_WIDTH)}{result.get('catalyst') or _DASH}")

    summary = result.get("summary")
    if summary:
        print(f"Özet: {summary}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the ``analyze`` subcommand: fetch/normalize/store, then a full
    fundamental + technical interpretation, printed as a verdict card (and,
    with ``--html``, saved as a standalone HTML report)."""
    horizon = getattr(args, "horizon", None) or "1y"
    cik, _name, normalized, ratios = _fetch_normalize_store(args)

    provider = getattr(args, "provider", None) or Config.ANALYZER_PROVIDER
    print(f"\nRunning {provider} analysis (horizon={horizon})...")

    price, as_of, technical, price_df = _fetch_price_and_technical(
        args.ticker, horizon, args.no_cache
    )
    analyst = _fetch_analyst_targets(args.ticker, args.no_cache)

    metrics = compute_metrics(normalized, ratios, price)
    flags = detect_red_flags(normalized, ratios, metrics, horizon)
    submissions = _fetch_submissions(cik, args.ticker, args.no_cache)
    catalyst = _fetch_catalyst(submissions, args.ticker)
    events = _detect_filing_events(submissions)

    if price_df is not None:
        _save_price_rows(cik, price_df)

    result = interpret(
        normalized,
        ratios,
        provider=provider,
        horizon=horizon,
        metrics=metrics,
        technical=technical,
        red_flags=flags,
        catalyst=catalyst,
        submissions=submissions,
        price_df=price_df,
    )

    # Recent 8-K events are deterministic filing-metadata facts, not model
    # output, so they're attached to the result post-interpret (like the
    # numbers the card renders directly) rather than routed through the LLM
    # payload. This also persists them with the verdict and carries them into
    # the HTML report, which reads result.events.
    if isinstance(result, dict):
        result["events"] = events

    if "error" in result:
        print(
            f"\nWARNING: analysis unavailable ({result['error']}): "
            f"{result.get('summary', 'no further details')}",
            file=sys.stderr,
        )
        if "raw" in result:
            logger.debug("Raw model output that failed to parse: %s", result["raw"])
    else:
        try:
            save_verdict(
                args.ticker, cik, horizon, provider, price, result,
                db_path=Config.DB_PATH, valuation=result.get("valuation"),
            )
        except Exception:  # noqa: BLE001 - persistence failure must not be fatal
            logger.warning("Failed to save verdict for %s", args.ticker, exc_info=True)

    _print_verdict_card(args.ticker, horizon, result, metrics, flags, technical, analyst=analyst)

    if getattr(args, "verbose", False):
        print("\n" + json.dumps(result, indent=2, ensure_ascii=False))

    if getattr(args, "html", False):
        try:
            report_path = generate_report(
                args.ticker,
                horizon,
                result,
                metrics=metrics,
                technical=technical,
                flags=flags,
                price=price,
                as_of=as_of,
                analyst=analyst,
            )
            print(f"\nHTML report saved to: {report_path}")
        except Exception as exc:  # noqa: BLE001 - a report-writing failure must not crash analyze
            logger.exception("Failed to generate HTML report for %s", args.ticker)
            print(f"\nWARNING: failed to generate HTML report: {exc}", file=sys.stderr)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Handle the ``calibrate`` subcommand: run the headless script-provider
    pipeline over a ticker basket and report the fair-value/price ratio
    distribution (normalization Work Package 0 -- see
    :mod:`sec_analyzer.calibrate`).

    Prints the per-ticker table, a readable summary of the ratio
    distribution, and the path of the saved JSON snapshot.
    """
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if getattr(args, "tickers", None)
        else DEFAULT_TICKERS
    )

    rows = run_calibration(tickers, years=args.years, no_cache=args.no_cache)
    print()
    print_calibration_table(rows)

    summary = summarize_ratios(rows)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    path = save_calibration_snapshot(args.label, rows, summary)
    if path:
        print(f"\nSaved calibration snapshot to: {path}")
    else:
        print("\nWARNING: failed to save calibration snapshot.", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``argparse`` parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="sec_analyzer",
        description=(
            "Fetch, normalize, store, and (optionally) interpret SEC EDGAR "
            "financial data for a public company."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared options for every subcommand: the ticker, plus fetch/store knobs
    # and logging verbosity.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("ticker", metavar="TICKER", help="Stock ticker symbol, e.g. AAPL")
    common.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of most-recent fiscal years to retain (default: 5).",
    )
    common.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk raw JSON cache and re-fetch from SEC EDGAR.",
    )
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )
    verbosity.add_argument(
        "--quiet", action="store_true", help="Only log WARNING and above."
    )

    fetch_parser = subparsers.add_parser(
        "fetch",
        parents=[common],
        help="Fetch, normalize, and store SEC financials for TICKER.",
    )
    fetch_parser.set_defaults(func=cmd_fetch)

    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[common],
        help="Fetch/normalize/store, then run a full fundamental + technical analysis for TICKER.",
    )
    analyze_parser.add_argument(
        "--horizon",
        choices=["3m", "1y", "5y"],
        default="1y",
        help=(
            "Investment horizon; controls the fundamental/technical weighting "
            "and the framing of the verdict (default: 1y). See "
            "Config.HORIZON_WEIGHTS."
        ),
    )
    analyze_parser.add_argument(
        "--html",
        action="store_true",
        help="Also save a standalone HTML verdict-card report (see Config.REPORTS_DIR).",
    )
    analyze_parser.add_argument(
        "--provider",
        choices=["ollama", "gemma", "anthropic", "script"],
        default=None,
        help=(
            "Analysis backend (default: ANALYZER_PROVIDER env, i.e. 'script'). "
            "script = deterministic rule-based analysis, no AI/LLM required."
        ),
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help=(
            "Run the headless script-provider pipeline over a ticker basket "
            "and report the fair-value/price ratio distribution (normalization "
            "measurement tool; see sec_analyzer.calibrate)."
        ),
    )
    calibrate_parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help=(
            "Comma-separated ticker list, e.g. 'AAPL,MSFT'. "
            "Default: sec_analyzer.calibrate.DEFAULT_TICKERS (~28-ticker basket)."
        ),
    )
    calibrate_parser.add_argument(
        "--label",
        type=str,
        default="run",
        help="Short label used in the saved snapshot's filename (default: 'run').",
    )
    calibrate_parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of most-recent fiscal years to retain (default: 5).",
    )
    calibrate_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk raw JSON/price caches and re-fetch.",
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)

    return parser


def main() -> None:
    """CLI entry point: parse arguments, configure logging, dispatch."""
    # Windows consoles frequently default to a legacy code page that can't
    # represent Turkish characters or the box-drawing characters used in the
    # verdict card; force UTF-8 with a lossy fallback rather than crashing
    # mid-report on an encode error. Not every stream supports reconfigure()
    # (e.g. when stdout has been redirected to certain non-TTY targets), so
    # this is best-effort.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - purely a best-effort console fix-up
        pass

    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"SEC EDGAR request failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
