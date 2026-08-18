"""Command-line entry point for sec_analyzer.

Per-ticker subcommands:

* ``fetch TICKER`` -- resolve the ticker to a CIK, pull SEC XBRL company
  facts, normalize them, compute ratios, print both, and persist everything
  to the local SQLite database.
* ``analyze TICKER`` -- everything ``fetch`` does, plus price/technical data,
  valuation metrics, red flags, an earnings-date estimate, recent 8-K events,
  SEC Form 4 insider activity, and a full fundamental+technical interpretation
  (fair-value range, verdicts, cyclicality, and a summary) from a selectable
  backend: the local Claude Code CLI (`claude -p`, subscription billing;
  default), a local Ollama/Gemma model, or a deterministic script-based
  (no-AI) analyzer. The result is printed as a compact Turkish-language
  verdict card and, optionally, saved as a standalone HTML report.

Portfolio-level subcommands, both of which read back what has already been
analyzed rather than running a new valuation:

* ``overview`` -- network-free dashboard over every ticker with a stored
  verdict: sector heat map, per-name fair-value gap, staleness, upcoming
  earnings, and verdict drift.
* ``preearnings`` -- which watchlist names report soon, and what the stored
  analysis currently says about each of them.

Usage::

    python -m sec_analyzer.cli fetch AAPL --years 5
    python -m sec_analyzer.cli analyze AAPL
    python -m sec_analyzer.cli analyze AAPL --horizon 5y --provider script
    python -m sec_analyzer.cli analyze AAPL --html
    python -m sec_analyzer.cli overview
    python -m sec_analyzer.cli preearnings --within 14

Only the official SEC EDGAR API and (for ``analyze`` with the ``claude_code``
or ``ollama`` providers) a local LLM are used for financial statement data --
no third-party finance data libraries beyond the optional yfinance
price-history fetch used to power the technical-analysis layer.
"""

import argparse
import copy
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
from sec_analyzer.fetch.earnings import get_earnings_history
from sec_analyzer.fetch.filings import estimate_next_earnings
from sec_analyzer.fetch.insider import get_insider_transactions
from sec_analyzer.fetch.fred import get_macro_panel, get_risk_free_asof
from sec_analyzer.fetch.prices import (
    PriceDataError,
    drop_unsettled_bars,
    get_price_history,
    is_total_return_basis,
    latest_price,
    slice_asof,
)
from sec_analyzer.fetch.tickers import resolve_cik
from sec_analyzer.http_client import SecHttpClient
from sec_analyzer.interpret.analyzer import build_script_phase1, interpret, propose_assumptions
from sec_analyzer.normalize.metrics import compute_metrics, resolve_fundamental_fy
from sec_analyzer.normalize.normalizer import format_table, normalize_facts
from sec_analyzer.normalize.ratios import compute_ratios
from sec_analyzer.normalize.red_flags import detect_red_flags
from sec_analyzer.report.financials import serialize_financials
from sec_analyzer.report.generator import generate_report
from sec_analyzer.interpret import planning, rule_based
from sec_analyzer.screener.overview import build_overview, sector_type_label
from sec_analyzer.screener.peers import (
    build_peer_snapshot,
    metrics_from_normalized,
    rank_against_peers,
    resolve_sector,
)
from sec_analyzer.screener.preearnings import (
    DEFAULT_WITHIN_DAYS,
    format_surprise_pct,
    scan_preearnings,
)
from sec_analyzer.screener.swing_scan import DEFAULT_MAX_WORKERS, scan_swing
from sec_analyzer.screener.universe import load_universe, normalize_index
from sec_analyzer.signals.events import detect_events, summarize_events
from sec_analyzer.signals.insider import detect_insider_activity, summarize_insider
from sec_analyzer.signals.macro import build_macro_context, summarize_macro
from sec_analyzer.store import assumptions as assumptions_store
from sec_analyzer.valuation import damodaran
from sec_analyzer.valuation.capm import compute_cost_of_equity
from sec_analyzer.valuation.sanity import clamp_assumptions, validate_assumptions
from sec_analyzer.valuation.sector import SECTOR_FINANCIAL, _is_deposit_funded, classify_sector
from sec_analyzer.signals.momentum import (
    compute_fundamental_momentum,
    compute_verdict_momentum,
    synthesize_momentum,
)
from sec_analyzer.store.database import (
    load_latest_peer_snapshot,
    load_latest_verdicts,
    load_prior_live_verdict,
    load_verdicts,
    load_watchlist_tickers,
    save_normalized,
    save_peer_snapshot,
    save_prices,
    save_swing_scan,
    save_verdict,
)
from sec_analyzer.technical.indicators import compute_indicators, relative_strength
from sec_analyzer.technical.momentum import compute_price_momentum, sector_etf_for_sic
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


def _print_ratios(ratios: List[dict], sector_type: Optional[str] = None) -> None:
    """Render the per-fiscal-year ratio list as a compact aligned table.

    Net margin and the two YoY growth ratios are shown as percentages; ROE
    and the current ratio are shown as plain decimals. Missing (``None``)
    values are rendered as ``-``.

    Args:
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
        sector_type: One of
            :func:`sec_analyzer.valuation.sector.classify_sector`'s buckets,
            or ``None``. ``"financial"`` drops the Current Ratio column,
            which is undefined for a filer that does not classify its
            balance sheet by maturity (SPEC.md Sec.20c); every other value
            keeps the existing table unchanged.
    """
    if not ratios:
        print("No ratios available (insufficient annual data).")
        return

    def as_pct(value) -> str:
        return f"{value * 100:.1f}%" if value is not None else "-"

    def as_dec(value) -> str:
        return f"{value:.2f}" if value is not None else "-"

    show_current_ratio = sector_type != "financial"

    headers = ["FY", "Net Margin", "ROE"]
    if show_current_ratio:
        headers.append("Current Ratio")
    headers += ["Rev YoY", "NI YoY"]
    print("".join(h.rjust(_RATIO_COL_WIDTH) for h in headers))
    print("-" * (_RATIO_COL_WIDTH * len(headers)))

    for row in ratios:
        cells = [
            str(row.get("fy", "-")),
            as_pct(row.get("net_margin")),
            as_dec(row.get("roe")),
        ]
        if show_current_ratio:
            cells.append(as_dec(row.get("current_ratio")))
        cells += [
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
            ``no_cache`` attributes. An optional ``as_of`` attribute
            (``datetime.date`` or ``None``) enables point-in-time mode:
            only facts filed on/before that date survive normalization, and
            the truncated slice is NOT persisted to the database (the
            financials table holds the current-view upsert).

    Returns:
        ``(cik, name, normalized, ratios)``.
    """
    client = SecHttpClient()
    as_of = getattr(args, "as_of", None)

    cik, name = resolve_cik(args.ticker, client, no_cache=args.no_cache)
    logger.info("Resolved ticker %s -> CIK %s (%s)", args.ticker, cik, name)

    facts = get_company_facts(cik, client, no_cache=args.no_cache)
    normalized = normalize_facts(facts, years=args.years, as_of=as_of)
    ratios = compute_ratios(normalized)

    print(format_table(normalized))
    print()
    # This function has no SIC in scope (submissions are fetched later, only
    # by `analyze`), but the deposit test alone is enough to know the current
    # ratio is meaningless here, and it needs no extra network call.
    _print_ratios(
        ratios,
        sector_type=SECTOR_FINANCIAL if _is_deposit_funded(normalized, {}) else None,
    )

    if as_of is not None:
        print(
            f"\nNot: geçmiş tarih (as-of {as_of.isoformat()}) modunda finansallar "
            "veritabanına yazılmaz."
        )
    else:
        save_normalized(args.ticker, cik, name, normalized, ratios, db_path=Config.DB_PATH)
        print(f"\nSaved to database: {Config.DB_PATH}")

    return cik, name, normalized, ratios


def cmd_fetch(args: argparse.Namespace) -> None:
    """Handle the ``fetch`` subcommand: fetch, normalize, and store only."""
    _fetch_normalize_store(args)


def _fetch_price_and_technical(
    ticker: str, horizon: str, no_cache: bool, as_of=None
):
    """Fetch price history and derive the merged technical indicators/verdict.

    Fully graceful: if price data can't be obtained (yfinance fails, or the
    ticker simply has too little history), this logs a warning
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
        # prefer_live only when reporting on today: an as-of run slices the
        # live bar back off anyway, so fetching it would be wasted work.
        price_df, source = get_price_history(
            ticker, no_cache=no_cache, prefer_live=as_of is None
        )
        if as_of is not None:
            price_df = slice_asof(price_df, as_of)
            if price_df.empty:
                logger.warning(
                    "No price data for %s on/before as-of %s; skipping technical.",
                    ticker, as_of,
                )
                return None, None, None, None
        price, as_of_date = latest_price(price_df)
        indicators = compute_indicators(price_df)
        # Relative strength (vs. SPY) is attached BEFORE the momentum synthesis
        # so it feeds the composite score; the momentum dict is then attached
        # BEFORE technical_verdict so the horizon narrative can lead with it.
        # Sector-relative strength (which needs the SIC from submissions) is
        # folded in later by the caller, which then recomputes momentum.
        indicators["relative_strength"] = _fetch_relative_strength(ticker, price_df, no_cache, as_of)
        indicators["momentum"] = compute_price_momentum(indicators)
        verdict_result = technical_verdict(indicators, horizon)
        technical = {**indicators, **verdict_result}
        # Provenance of the price series these indicators were computed from,
        # carried here rather than widening this function's return tuple
        # (four call sites plus test fakes). The report names the source
        # actually used instead of a hardcoded "Stooq".
        technical["price_source"] = source
        logger.info(
            "Price data for %s from %s: %.2f as of %s", ticker, source, price, as_of_date
        )
        return price, as_of_date, technical, price_df
    except PriceDataError as exc:
        logger.warning("Price data unavailable for %s: %s", ticker, exc)
        return None, None, None, None


#: Benchmark ticker for the relative-strength (RS) cross-check.
_RS_BENCHMARK = "SPY"


def _fetch_relative_strength(
    ticker: str, price_df, no_cache: bool, as_of=None, benchmark: str = _RS_BENCHMARK
) -> Optional[dict]:
    """Best-effort price relative strength vs. ``benchmark`` (default
    :data:`_RS_BENCHMARK`); never raises.

    Fetches the benchmark's (cached) price history and compares returns via
    :func:`sec_analyzer.technical.indicators.relative_strength`. Returns
    ``None`` when the ticker *is* the benchmark, price data is unavailable, or
    anything fails -- RS is a display-only cross-check and must never block the
    rest of ``analyze``. When ``as_of`` is set the benchmark frame is sliced
    to the same cutoff so the comparison stays point-in-time.
    """
    if str(ticker).strip().upper() == benchmark:
        return None
    try:
        # Matches the stock's own fetch: comparing a live bar against a
        # settled benchmark bar would skew relative strength by a session.
        bench_df, _ = get_price_history(
            benchmark, no_cache=no_cache, prefer_live=as_of is None
        )
        if as_of is not None:
            bench_df = slice_asof(bench_df, as_of)
        return relative_strength(price_df["Close"], bench_df["Close"], benchmark=benchmark)
    except Exception:  # noqa: BLE001 - display-only cross-check, never fatal
        logger.warning("Could not compute relative strength for %s vs %s", ticker, benchmark, exc_info=True)
        return None


def _attach_momentum(result, ticker, normalized, technical, as_of):
    """Compute and attach the three-layer momentum *context* to ``result``.

    Combines the composite price momentum (already on ``technical``), the
    fundamental momentum (quarterly growth acceleration + margin trend + model-
    based surprise), and the verdict momentum (FV/price trajectory across prior
    stored live analyses) into ``result["momentum"]``, and applies the falling-
    knife stabilization note to the dip tranches when the cross-signal fires.

    Deterministic and never fatal (best-effort context, like events). In as-of /
    backtest mode the verdict-momentum and model-surprise sub-signals are
    skipped -- only the point-in-time price momentum is used -- so a historical
    run cannot borrow forward information from later stored verdicts.
    """
    try:
        prior = None
        verdict_hist = None
        if as_of is None:
            prior_v = load_prior_live_verdict(ticker, db_path=Config.DB_PATH)
            if prior_v:
                prior = {"valuation": prior_v.get("valuation"), "ref_date": prior_v.get("analyzed_at")}
            verdict_hist = load_verdicts(ticker, db_path=Config.DB_PATH, live_only=True)
        fundamental_m = compute_fundamental_momentum(normalized, prior)
        verdict_m = compute_verdict_momentum(verdict_hist)
        price_m = (technical or {}).get("momentum")
        momentum = synthesize_momentum(price_m, fundamental_m, verdict_m, result.get("fundamental_verdict"))
        if momentum is not None:
            result["momentum"] = momentum
            planning.apply_stabilization_condition(result.get("entry_plan"), momentum.get("falling_knife"))
    except Exception:  # noqa: BLE001 - momentum is display-only context, never fatal
        logger.warning("Could not attach momentum context for %s", ticker, exc_info=True)


def _enrich_sector_momentum(ticker, technical, price_df, submissions, no_cache, as_of):
    """Fold sector-relative strength into ``technical`` and recompute the
    composite momentum score with it.

    Sector-relative strength needs the ticker's SIC code, which only becomes
    available once ``submissions`` are fetched (after the initial technical
    pass), so this runs as a second enrichment step. Best-effort and never
    fatal: a missing/unmapped SIC, missing price frame, or any failure just
    leaves the SPY-only momentum already on ``technical`` in place.
    """
    if not isinstance(technical, dict) or price_df is None:
        return
    try:
        sic = submissions.get("sic") if isinstance(submissions, dict) else None
        etf = sector_etf_for_sic(sic)
        if etf and str(ticker).strip().upper() != etf:
            sector_rs = _fetch_relative_strength(ticker, price_df, no_cache, as_of, benchmark=etf)
            if sector_rs is not None:
                technical["relative_strength_sector"] = sector_rs
        # Recompute momentum so the sector-relative component is included.
        technical["momentum"] = compute_price_momentum(technical)
    except Exception:  # noqa: BLE001 - display-only enrichment, never fatal
        logger.warning("Could not enrich sector momentum for %s", ticker, exc_info=True)


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


def _fetch_earnings_history(ticker: str, no_cache: bool) -> Optional[dict]:
    """Best-effort fetch of recent quarterly EPS beat/miss history; never raises.

    Display-only cross-check (see :mod:`sec_analyzer.fetch.earnings`) -- never
    feeds the valuation engine; shown only in the HTML report's "Bilanço" tab.
    Any failure is logged and swallowed so it never blocks the rest of
    ``analyze``.

    Returns:
        The dict returned by
        :func:`sec_analyzer.fetch.earnings.get_earnings_history`, or ``None``
        if unavailable or the fetch fails for any reason.
    """
    try:
        return get_earnings_history(ticker, no_cache=no_cache)
    except Exception:  # noqa: BLE001 - a display-only cross-check must never be fatal
        logger.warning("Could not fetch earnings history for %s", ticker, exc_info=True)
        return None


def _fetch_risk_free_asof(as_of, no_cache: bool) -> Optional[dict]:
    """Best-effort risk-free rate (FRED DGS10); never raises.

    ``as_of=None`` (an ordinary live run) asks for the LATEST observation;
    an as-of date asks for the last observation on/before it.

    Live runs used to skip FRED entirely and let the risk-free rate fall
    through to the archived ``data/damodaran/erp.csv`` value -- which meant a
    backtest of 2022 was priced off the real 2022 yield while *today's*
    analysis was priced off a hand-refreshed CSV that could be a year stale.
    Since the risk-free rate feeds both the CAPM cost of equity and the
    terminal-growth anchor (SPEC.md "Terminal-growth anchor"), that stale
    input moved every fair value the engine produced. Preferring the live
    series here changes no RULE -- only the freshness of its input -- and
    `damodaran.load_sector_data` already ranks ``fred_rate`` above the CSV,
    so a FRED outage still degrades to the archived value exactly as before.
    """
    try:
        return get_risk_free_asof(as_of, no_cache=no_cache)
    except Exception:  # noqa: BLE001 - macro fetch must never be fatal
        logger.warning(
            "Could not fetch FRED risk-free rate (as_of=%s); falling back to "
            "the archived erp.csv value.", as_of, exc_info=True,
        )
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


def _fetch_catalyst(submissions: Optional[dict], ticker: str, as_of=None) -> Optional[dict]:
    """Best-effort next-earnings estimate from already-fetched submissions; never raises.

    Args:
        submissions: The dict returned by :func:`_fetch_submissions`, or
            ``None``.
        ticker: Stock ticker symbol, used only for the warning log message.
        as_of: Optional point-in-time reference date; forwarded as
            ``estimate_next_earnings(today=as_of)`` so the projection walks
            forward from that date and ignores filings dated after it.

    Returns:
        The dict returned by
        :func:`sec_analyzer.fetch.filings.estimate_next_earnings`, or
        ``None`` if ``submissions`` is unavailable or the estimate itself
        fails for any reason.
    """
    if not submissions:
        return None
    try:
        return estimate_next_earnings(submissions, today=as_of)
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


def _detect_filing_events(submissions: Optional[dict], as_of=None) -> List[dict]:
    """Best-effort recent 8-K event signal from already-fetched submissions.

    Reuses the ``submissions`` document :func:`_fetch_submissions` fetched
    once for this run (no extra network, no document download, no LLM). Only
    warning/critical events within the last year are surfaced. ``detect_events``
    is itself never-raises, so this simply returns an empty list when there's
    nothing to show. When ``as_of`` is set it is the reference date for the
    lookback window, and filings dated after it are excluded.
    """
    if not submissions:
        return []
    return detect_events(
        submissions,
        lookback_days=_EVENTS_LOOKBACK_DAYS,
        min_severity=_EVENTS_MIN_SEVERITY,
        max_events=_EVENTS_MAX,
        today=as_of,
    )


#: Lookback window (days) for the SEC Form 4 insider-activity signal, and the
#: cap on how many Form 4 documents one run will download. Six months is long
#: enough for a cluster of buys to form and short enough that last year's
#: compensation cycle doesn't drown out this quarter's conviction.
_INSIDER_LOOKBACK_DAYS = 180
_INSIDER_MAX_FILINGS = 40


def _fetch_insider_activity(
    cik: str, ticker: str, submissions: Optional[dict], no_cache: bool, as_of=None
) -> Optional[dict]:
    """Best-effort SEC Form 4 insider-activity signal; never raises.

    Reuses the ``submissions`` document :func:`_fetch_submissions` already
    fetched for this run to locate the recent Form 4 filings, then downloads
    and parses each one's raw ownership XML (cached per accession, since a
    filed document is immutable -- see :mod:`sec_analyzer.fetch.insider`).
    Deterministic and LLM-free; context only, never an input to the fair
    value. ``as_of`` is the point-in-time reference date, so a backtest run
    only sees filings that existed on that date.
    """
    if not submissions:
        return None
    try:
        client = SecHttpClient()
        fetched = get_insider_transactions(
            cik,
            submissions,
            client,
            lookback_days=_INSIDER_LOOKBACK_DAYS,
            max_filings=_INSIDER_MAX_FILINGS,
            no_cache=no_cache,
            today=as_of,
        )
        return detect_insider_activity(
            fetched, lookback_days=_INSIDER_LOOKBACK_DAYS, today=as_of
        )
    except Exception:  # noqa: BLE001 - insider context is display-only, never fatal
        logger.warning("Could not fetch insider activity for %s", ticker, exc_info=True)
        return None


def _base_terminal_growth(result: dict) -> Optional[float]:
    """Pull the base scenario's terminal growth out of a valuation result.

    Only used to give the macro layer something to cross-check against the
    market's priced inflation expectation. Returns ``None`` for any shape it
    doesn't recognize -- the cross-check is then simply omitted.
    """
    try:
        base = ((result.get("valuation") or {}).get("assumptions") or {}).get("base") or {}
        value = base.get("terminal_growth")
        return float(value) if isinstance(value, (int, float)) else None
    except Exception:  # noqa: BLE001 - a display cross-check must never be fatal
        return None


def _attach_macro(result: dict, as_of, no_cache: bool) -> None:
    """Attach the rates/credit macro context to ``result``; never raises.

    Context only -- SPEC.md's rules are untouched by anything here. The one
    place macro data DOES reach the numbers is the risk-free rate, and that
    goes through `_fetch_risk_free_asof` -> `damodaran.load_sector_data`
    entirely separately from this block. Attached post-``interpret`` (like
    events/insider/momentum) so it never enters an LLM payload.
    """
    try:
        panel = get_macro_panel(as_of=as_of, no_cache=no_cache)
        context = build_macro_context(panel, terminal_growth=_base_terminal_growth(result))
        if context is not None:
            result["macro"] = context
    except Exception:  # noqa: BLE001 - macro context is display-only, never fatal
        logger.warning("Could not build macro context", exc_info=True)


def _attach_peers(
    result: dict, ticker: str, cik: str, submissions: Optional[dict],
    normalized: dict, ratios: list, metrics: dict, as_of=None,
) -> None:
    """Attach the cross-sectional peer comparison to ``result``; never raises.

    Reads the most recent stored peer snapshot (built offline by
    ``cli peers --refresh``) -- no network here, so an analysis never waits on
    a dozen multi-megabyte Frames downloads. When no snapshot has been built
    yet, the comparison is simply absent.

    **Point-in-time guard.** A peer snapshot is keyed on a fiscal year, so an
    as-of run must not be ranked against a snapshot built from filings that
    did not exist on the cutoff date -- that is hindsight leakage of exactly
    the kind the backtest layer is designed to avoid. In as-of mode we
    therefore look for a snapshot of the last fiscal year that had completed
    by the cutoff, and attach nothing when there isn't one, rather than
    silently falling back to the newest snapshot.

    Context only: this never feeds the fair value, the triangulation signals
    or ``sector_ratio`` (see ``screener/peers.py`` and SPEC.md Sec.6/7, which
    are deliberately NOT amended for this layer).
    """
    try:
        if as_of is not None:
            snapshot = load_latest_peer_snapshot(
                db_path=Config.DB_PATH, year=as_of.year - 1
            )
        else:
            snapshot = load_latest_peer_snapshot(db_path=Config.DB_PATH)
        if not snapshot:
            return
        sic = (submissions or {}).get("sic")
        sector = resolve_sector(ticker=ticker, cik=cik, sic=sic, snapshot=snapshot)
        if not sector:
            return
        own = metrics_from_normalized(normalized, ratios, metrics)
        ranked = rank_against_peers(own, sector, snapshot)
        if ranked is not None:
            result["peers"] = ranked
    except Exception:  # noqa: BLE001 - peer context is display-only, never fatal
        logger.warning("Could not build peer comparison for %s", ticker, exc_info=True)


def _save_price_rows(cik: str, price_df) -> None:
    """Convert a price-history DataFrame to row dicts and persist them.

    Never raises: a failure to persist price history must not prevent the
    rest of ``analyze`` (interpretation, verdict card, HTML report) from
    completing.

    Bars for a session still in progress are not stored: the same reasoning as
    the price cache (see ``prices.drop_unsettled_bars``) applies to the
    ``prices`` table, which ``load_latest_stored_price`` reads as a settled
    close.
    """
    try:
        price_df = drop_unsettled_bars(price_df)
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


def _fmt_swing_price(value) -> str:
    """Format a number as a dollar amount for the swing-scan table.

    Unlike :func:`_fmt_money` (which drops decimals for whole numbers in the
    terminal verdict card by deliberate, pre-existing choice), the swing
    table's Fiyat/Giriş/Stop/Hedef columns must always show exactly 2
    decimals so the column stays visually aligned across rows (e.g.
    ``"$153.00"``, never ``"$153"``). Returns :data:`_DASH` for
    ``None``/unparseable input.
    """
    if value is None:
        return _DASH
    try:
        value = float(value)
    except (TypeError, ValueError):
        return _DASH
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
    - ``"RIM"``: the DCF is disabled, there's no populated FFO block, and
      ``valuation["rim"]`` is a populated residual income model block (has a
      ``"scenarios"`` key) -- financial (banks/insurers) filers valued via
      the multi-period RIM anchor (SPEC.md Sec.8f).
    - ``"P/B×ROE"``: the DCF is disabled and there's no populated FFO or RIM
      block -- financial filers whose RIM build failed (falls back to the
      P/B x ROE anchor), or a REIT that fell back to the P/B x ROE anchor
      without a populated FFO block.
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
    rim = valuation.get("rim")
    if isinstance(rim, dict) and "scenarios" in rim:
        return "RIM"
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
    multiples_signal = ((valuation.get("triangulation") or {}).get("signals") or {}).get("multiples")

    # Leverage-primary (SPEC.md Sec.6/Sec.10): a leveraged non-reit filer's
    # primary own-history multiple is EV/EBITDA (FD/FAVÖK) ahead of P/E, and
    # the P/E-based PEG line is suppressed -- mirrors triangulate's signal.
    ev_primary = (
        not is_reit
        and bool(multiples.get("leveraged"))
        and multiples.get("ev_ebitda_percentile") is not None
    )
    growth_adjusted = {} if (is_reit or ev_primary) else (multiples.get("growth_adjusted") or {})

    # Primary raw multiple. Reit uses P/FFO -> P/S only (P/E and P/FCF are
    # meaningless for REITs); a leveraged filer leads with FD/FAVÖK; everything
    # else keeps the P/E -> P/S -> P/FCF fallback order.
    primary = None
    if is_reit:
        primary_candidates = (
            ("P/FFO", multiples.get("pffo_percentile")),
            ("P/S", multiples.get("ps_percentile")),
        )
    elif ev_primary:
        primary_candidates = (
            ("FD/FAVÖK", multiples.get("ev_ebitda_percentile")),
            ("P/E", multiples.get("pe_percentile")),
            ("P/S", multiples.get("ps_percentile")),
            ("P/FCF", multiples.get("pfcf_percentile")),
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


def _ev_multiples_line(valuation: dict) -> Optional[str]:
    """Render the informational "EV çarpanları:" card line: current EV/EBITDA
    (FD/FAVÖK) and EV/EBIT (FD/FVÖK), each with its own historical percentile
    when enough price-backed history exists (SPEC.md Sec.6). These are
    capital-structure-neutral earnings multiples reported alongside P/E; the
    verdict/triangulation still keys off P/E. Returns ``None`` (line omitted)
    when neither current EV multiple is available."""
    multiples = valuation.get("multiples") or {}
    current = multiples.get("current") or {}
    fragments = []
    for name, cur_key, pct_key in (
        ("FD/FAVÖK", "ev_ebitda", "ev_ebitda_percentile"),
        ("FD/FVÖK", "ev_ebit", "ev_ebit_percentile"),
    ):
        value = current.get(cur_key)
        if value is None:
            continue
        pct = multiples.get(pct_key)
        if pct is not None:
            fragments.append(f"{name} {value:.1f}× ({pct:.0f}. pctile)")
        else:
            fragments.append(f"{name} {value:.1f}×")

    if not fragments:
        return None
    label = "FD çarpanı:".ljust(_CARD_LABEL_WIDTH)
    return f"{label}{' · '.join(fragments)}"


_CYCLE_FLAG_LABELS = {
    "peak_cycle_pe_trap": "tepe-döngü F/K tuzağı",
    "peak_annualization": "tepe-çeyrek yıllıklaştırma",
    "regime_change_premium": "rejim-değişimi primi",
}


def _cycle_lines(valuation: dict) -> List[str]:
    """Render the cyclical cycle-position / two-regime block (SPEC.md Sec.26e).

    Advisory: this never moves the headline fair value. It answers a
    different question -- not "what is it worth" but "what does today's price
    force you to believe about the cycle". Returns ``[]`` for any filer
    without a ``cycle`` block (every non-cyclical sector, and cyclicals with
    too little history)."""
    cycle = valuation.get("cycle")
    if not cycle:
        return []
    stats = cycle.get("stats") or {}
    lines: List[str] = []

    def _pct(value):
        return f"%{value * 100:.1f}" if isinstance(value, (int, float)) else _DASH

    percentile = stats.get("margin_percentile")
    percentile_txt = f"{percentile:.0f}. pctile" if isinstance(percentile, (int, float)) else _DASH
    position = (
        f"marj {_pct(stats.get('margin_current'))} "
        f"({stats.get('margin_current_basis', '?')}, {percentile_txt}) · "
        f"ort {_pct(stats.get('margin_mean'))} · dip {_pct(stats.get('margin_trough'))} "
        f"/ tepe {_pct(stats.get('margin_peak'))} [{stats.get('years')}y]"
    )
    lines.append(f"{'Döngü:'.ljust(_CARD_LABEL_WIDTH)}{position}")

    if all(isinstance(stats.get(k), (int, float)) for k in ("pb_current", "pb_median", "pb_peak")):
        lines.append(
            f"{''.ljust(_CARD_LABEL_WIDTH)}P/B {stats['pb_current']:.2f} "
            f"(tarihsel dip {stats['pb_trough']:.2f} · medyan {stats['pb_median']:.2f} "
            f"· tepe {stats['pb_peak']:.2f})"
        )

    def _band(regime, label):
        if not regime:
            return None
        parts = [
            f"{name} ${regime[key]:,.0f}"
            for name, key in (("düşük", "low"), ("merkez", "center"), ("yüksek", "high"))
            if isinstance(regime.get(key), (int, float))
        ]
        return f"{label}: {' · '.join(parts)}" if parts else None

    for text in (_band(cycle.get("regime_a"), "Rejim A (döngü ortalaması)"),
                 _band(cycle.get("regime_b"), "Rejim B (yapısal kırılım)")):
        if text:
            lines.append(f"{''.ljust(_CARD_LABEL_WIDTH)}{text}")

    blend = cycle.get("blend") or {}
    if blend:
        cells = " · ".join(f"p={p} ${v:,.0f}" for p, v in sorted(blend.items()))
        lines.append(f"{''.ljust(_CARD_LABEL_WIDTH)}Harman: {cells}")

    lines.append(f"{''.ljust(_CARD_LABEL_WIDTH)}{cycle.get('verdict_sentence', '')}")

    fired = [label for key, label in _CYCLE_FLAG_LABELS.items() if (cycle.get("flags") or {}).get(key)]
    if fired:
        lines.append(f"{''.ljust(_CARD_LABEL_WIDTH)}Bayraklar: {', '.join(fired)}")
    return lines


def _tangible_multiples_line(valuation: dict) -> Optional[str]:
    """Render the informational "Maddi çarpan:" card line for a financial
    filer: current P/TBV (F/MDD -- fiyat / maddi defter değeri) with its own
    historical percentile, plus ROTCE (SPEC.md Sec.23e).

    Occupies the slot :func:`_ev_multiples_line` fills for other sectors and
    that Sec.20b deliberately emptied for this one: enterprise value is
    undefined for a deposit funder, while P/TBV is that sector's own
    convention (P/B is not comparable across filers carrying different
    goodwill loads). Reported alongside P/E; the verdict/triangulation still
    keys off P/E. Returns ``None`` (line omitted) for any other sector, or
    when no P/TBV figure exists."""
    if (valuation.get("sector_type")) != "financial":
        return None
    multiples = valuation.get("multiples") or {}
    ptbv = (multiples.get("current") or {}).get("ptbv")
    if ptbv is None:
        return None

    pct = multiples.get("ptbv_percentile")
    fragment = f"F/MDD {ptbv:.2f}×"
    if pct is not None:
        fragment += f" ({pct:.0f}. pctile)"
    fragments = [fragment]

    rotce = (valuation.get("rim") or {}).get("rotce")
    if rotce is not None:
        fragments.append(f"ROTCE %{rotce * 100:.1f}")

    # 12 chars, so it still leaves a separating space at _CARD_LABEL_WIDTH
    # (13) -- and it parallels _ev_multiples_line's "FD çarpanı:".
    label = "MDD çarpanı:".ljust(_CARD_LABEL_WIDTH)
    return f"{label}{' · '.join(fragments)}"


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


#: Turkish display label for each Altman-Z zone (SPEC.md Sec.8g).
_ALTMAN_ZONE_LABEL_TR = {"safe": "güvenli", "grey": "gri", "distress": "sıkıntı"}


def _distress_flags_line(valuation: dict) -> Optional[str]:
    """Render the "Risk skoru:" card line (SPEC.md Sec.8g/8j/8k): a
    single line summarizing whichever ADVISORY distress/earnings-quality
    screens (Altman Z-score, Beneish M-score, Merton distance-to-default)
    were computable for this filer.

    These screens never affect ``fair_value_range`` or the triangulation
    confidence (SPEC.md Sec.8g/8j/8k are explicit about this) -- this line
    exists purely to surface bankruptcy-risk/earnings-manipulation signals
    alongside the valuation, the same way "Red flags:" surfaces
    ``detect_red_flags`` findings without touching the fair-value math.

    A single shared line (rather than one bespoke line per screen) is
    deliberate: all three screens are advisory-only and share the same
    "is it present, what's its headline number/zone" shape, so one function
    reading all three ``valuation`` keys avoids near-duplicate wiring per
    screen as Beneish (Sec.8j) and Merton (Sec.8k) land.

    Returns ``None`` (renders nothing) when none of the three screens
    produced a result -- e.g. ``financial``/``reit`` sectors, where Altman Z
    isn't computed at all (SPEC.md Sec.8g), or a filer missing the
    underlying balance-sheet data for all three.
    """
    valuation = valuation or {}
    parts = []

    altman = valuation.get("altman_z")
    if isinstance(altman, dict) and altman.get("z_score") is not None:
        zone = _ALTMAN_ZONE_LABEL_TR.get(altman.get("zone"), altman.get("zone"))
        parts.append(f"Altman Z {altman['z_score']:.2f} ({zone})")

    beneish = valuation.get("beneish_m")
    if isinstance(beneish, dict) and beneish.get("m_score") is not None:
        suffix = " [kısmi]" if beneish.get("partial") else ""
        if beneish.get("flag"):
            # I2: caveat the flag as a likely growth artifact when sales grew
            # fast (Beneish over-flags hyper-growers) rather than presenting it
            # as a manipulation signal on the compact card.
            sgi = (beneish.get("components") or {}).get("sgi")
            if isinstance(sgi, (int, float)) and sgi > 1.40:
                suffix += " — işaretlendi (olasılıkla hızlı büyüme yan etkisi)"
            else:
                suffix += " — olası manipülasyon sinyali"
        parts.append(f"Beneish M {beneish['m_score']:.2f}{suffix}")

    merton = valuation.get("merton_dtd")
    if isinstance(merton, dict) and merton.get("distance_to_default") is not None:
        pd_pct = merton.get("probability_of_default")
        pd_text = f", PD %{pd_pct * 100:.1f}" if pd_pct is not None else ""
        parts.append(f"Merton DD {merton['distance_to_default']:.2f}{pd_text}")

    if not parts:
        return None
    return f"{'Risk skoru:'.ljust(_CARD_LABEL_WIDTH)}{' · '.join(parts)}"


def _lbo_floor_line(valuation: dict) -> Optional[str]:
    """Render the "LBO çapası:" card line (SPEC.md Sec.8h): the LBO-implied
    per-share value floor -- the highest price a disciplined financial
    (private-equity) buyer could justify today, purely from debt-paydown
    ("deleveraging") returns, with NO organic growth or multiple-expansion
    credit. ADVISORY ONLY: informational, never affects ``fair_value_range``
    or the triangulation confidence.

    Returns ``None`` (renders nothing) when ``valuation["lbo_floor_detail"]``
    is absent -- e.g. ``financial``/``reit`` sectors (never computed), or
    missing EBITDA/EV-EBITDA/debt/FCF/share data.
    """
    detail = valuation.get("lbo_floor_detail")
    if not isinstance(detail, dict) or detail.get("per_share") is None:
        return None
    label = "LBO çapası:".ljust(_CARD_LABEL_WIDTH)
    return f"{label}{_fmt_money(detail['per_share'])} (bilgi amaçlı, manşete girmez)"


def _signed_pct_tr(value) -> str:
    """Turkish signed-percent, sign-first then ``%``: ``4 -> "+%4"``,
    ``-7.5 -> "-%8"`` (0 decimals, matching the compact card style)."""
    if value is None:
        return _DASH
    sign = "+" if value >= 0 else "-"
    return f"{sign}%{abs(value):.0f}"


def _momentum_line(technical: dict) -> Optional[str]:
    """Compact returns/trend sub-line for the technical card, e.g.
    ``"Getiriler:  1a +%4 · 3a +%13 · 6a -%3 · Trend: yükseliş (GC) · 52h %68"``.
    ``None`` when no figure is available. (The composite momentum synthesis is
    a separate top-level ``Momentum:`` row -- see :func:`_momentum_synthesis_text`.)"""
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
    return "Getiriler:  " + " · ".join(parts)


#: Severity -> glyph for the momentum cross-signal sub-lines.
_CROSS_ICON = {"warn": "⚠", "good": "✓", "info": "•"}


def _momentum_synthesis_text(momentum: dict) -> str:
    """Top-level ``Momentum:`` row text from ``result["momentum"]``: the
    composite verdict followed by the price / fundamental / verdict-trend
    sub-reads, e.g. ``"POZİTİF · fiyat: YUKARI MOMENTUM (68/100, hızlanıyor) ·
    fundamental: POZİTİF · verdict: yakınsama"``."""
    parts: List[str] = []
    price = momentum.get("price")
    if isinstance(price, dict) and price.get("label"):
        seg = f"fiyat: {price['label']} ({price.get('score')}/100"
        seg += f", {price['accel']}" if price.get("accel") else ""
        seg += ")"
        parts.append(seg)
    fundamental = momentum.get("fundamental")
    if isinstance(fundamental, dict) and fundamental.get("label"):
        parts.append(f"fundamental: {fundamental['label']}")
    verdict_trend = momentum.get("verdict_trend")
    if isinstance(verdict_trend, dict) and verdict_trend.get("label"):
        parts.append(f"verdict: {verdict_trend['label']}")
    head = momentum.get("verdict") or _DASH
    return f"{head} · " + " · ".join(parts) if parts else head


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
    as_of=None,
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
    if as_of is not None:
        print(
            f"AS-OF {as_of.isoformat()} — geçmiş veri, hindsight içermez "
            "(analist konsensüsü gösterilmiyor)"
        )
        macro_asof = (result.get("valuation") or {}).get("macro_asof") if isinstance(result, dict) else None
        if macro_asof:
            print(
                f"  Makro: ERP {macro_asof.get('erp_source')} · risksiz faiz "
                f"{macro_asof.get('risk_free_source')} · çarpan/beta "
                f"{macro_asof.get('multiples_source', 'multiples.csv')}"
            )
        leak = result.get("hindsight_leak_risk") if isinstance(result, dict) else None
        if leak:
            print(f"  ⚠ {leak}")
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
        ev_multiples_line = _ev_multiples_line(valuation)
        if ev_multiples_line:
            print(ev_multiples_line)
        # SPEC.md Sec.23e: the same slot, for the sector where EV is undefined.
        tangible_multiples_line = _tangible_multiples_line(valuation)
        if tangible_multiples_line:
            print(tangible_multiples_line)
        for cycle_line in _cycle_lines(valuation):
            print(cycle_line)
        print(_triangulation_line(valuation))
        print(_sensitivity_line(valuation))
        distress_line = _distress_flags_line(valuation)
        if distress_line:
            print(distress_line)
        lbo_line = _lbo_floor_line(valuation)
        if lbo_line:
            print(lbo_line)

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

    # Top-level momentum synthesis row (price + fundamental + verdict momentum)
    # with any value x momentum cross-signals as indented sub-lines.
    momentum = result.get("momentum")
    if isinstance(momentum, dict):
        print(f"{'Momentum:'.ljust(_CARD_LABEL_WIDTH)}{_momentum_synthesis_text(momentum)}")
        for cross in momentum.get("cross_signals") or []:
            icon = _CROSS_ICON.get(cross.get("severity"), "•")
            print(f"  {icon} {cross.get('text', '')}")

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

    insider = result.get("insider")
    print(f"{'İçeriden:'.ljust(_CARD_LABEL_WIDTH)}{summarize_insider(insider)}")
    if isinstance(insider, dict) and insider.get("note"):
        print(f"  {insider['note']}")

    peers = result.get("peers")
    if isinstance(peers, dict):
        print(f"{'Emsal:'.ljust(_CARD_LABEL_WIDTH)}{peers.get('note') or _DASH}")

    macro = result.get("macro")
    if isinstance(macro, dict):
        print(f"{'Makro:'.ljust(_CARD_LABEL_WIDTH)}{summarize_macro(macro)}")
        for note in macro.get("notes") or []:
            print(f"  {note}")

    print(f"{'Katalizör:'.ljust(_CARD_LABEL_WIDTH)}{result.get('catalyst') or _DASH}")

    summary = result.get("summary")
    if summary:
        print(f"Özet: {summary}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the ``analyze`` subcommand: fetch/normalize/store, then a full
    fundamental + technical interpretation, printed as a verdict card (and,
    with ``--html``, saved as a standalone HTML report)."""
    horizon = getattr(args, "horizon", None) or "1y"
    as_of = getattr(args, "as_of", None)
    cik, _name, normalized, ratios = _fetch_normalize_store(args)

    # As-of default is no-AI: an LLM's training data can carry post-as_of
    # knowledge, so unless the user *explicitly* asks for an AI provider,
    # historical runs use the deterministic script engine. An explicit
    # --provider still works but the result carries a hindsight-leak label.
    explicit_provider = getattr(args, "provider", None)
    if getattr(args, "no_ai", False):
        # --no-ai wins over everything (explicit --provider, LLM_BACKEND, as-of).
        provider = "script"
    elif as_of is not None and explicit_provider is None:
        provider = "script"
    else:
        provider = explicit_provider or Config.ANALYZER_PROVIDER
    if as_of is not None and provider != "script":
        print(
            f"\nUYARI: as-of modunda AI sağlayıcı ({provider}) kullanılıyor — "
            "sonuç 'hindsight sızıntısı riski' etiketi taşıyacak.",
            file=sys.stderr,
        )
    print(f"\nRunning {provider} analysis (horizon={horizon})...")

    # Point-in-time no-data guard: if the as-of cutoff predates every filed
    # fact, there's nothing to analyze -- print a minimal Turkish card and
    # stop before the rest of the pipeline (which would otherwise divide by
    # missing fundamentals). Never crashes the CLI.
    if as_of is not None and not any((normalized.get("annual") or {}).values()):
        result = {
            "error": "as_of_no_data",
            "summary": (
                f"{as_of.isoformat()} tarihi itibarıyla dosyalanmış SEC verisi "
                "bulunamadı; şirketin ilk dosyalaması bu tarihten sonra olabilir."
            ),
        }
        _print_verdict_card(args.ticker, horizon, result, {}, [], None, as_of=as_of)
        return

    price, price_as_of, technical, price_df = _fetch_price_and_technical(
        args.ticker, horizon, args.no_cache, as_of
    )
    # Analyst consensus (yfinance) is undated and cannot be made point-in-time,
    # so it is suppressed entirely in as-of mode (SPEC: as-of contract).
    analyst = None if as_of is not None else _fetch_analyst_targets(args.ticker, args.no_cache)
    fred_rate = _fetch_risk_free_asof(as_of, args.no_cache)

    metrics = compute_metrics(normalized, ratios, price)
    flags = detect_red_flags(normalized, ratios, metrics, horizon)
    submissions = _fetch_submissions(cik, args.ticker, args.no_cache)
    catalyst = _fetch_catalyst(submissions, args.ticker, as_of)
    events = _detect_filing_events(submissions, as_of)
    insider = _fetch_insider_activity(cik, args.ticker, submissions, args.no_cache, as_of)
    # Now that the SIC is known, fold in sector-relative strength and recompute
    # the composite momentum score with it (SPY-only momentum is already set).
    _enrich_sector_momentum(args.ticker, technical, price_df, submissions, args.no_cache, as_of)

    # A degraded-source frame (split-adjusted only) must not overwrite rows in
    # a table read as a total-return series, nor reach the valuation layer.
    # Technicals keep the full frame -- see prices.is_total_return_basis.
    price_source = (technical or {}).get("price_source")
    valuation_price_df = price_df if is_total_return_basis(price_source) else None
    if valuation_price_df is None and price_df is not None:
        logger.warning(
            "Price source %r is not a total-return series for %s; historical multiples "
            "and price persistence are skipped. Technicals are unaffected.",
            price_source, args.ticker,
        )

    if valuation_price_df is not None and as_of is None:
        _save_price_rows(cik, valuation_price_df)

    # Resolve where phase-1 assumptions come from (frozen cache / deterministic
    # script / legacy live LLM) per --assumptions (ASSUMPTIONS_CACHE_SPEC.md).
    override, assum_note, assum_stop, verdict_provider = _resolve_analyze_phase1(
        args, cik, normalized, ratios, metrics, submissions, as_of, fred_rate
    )
    if assum_note:
        print(assum_note)
    if assum_stop:
        result = {"error": "assumptions_frozen_unavailable", "summary": assum_note}
        _print_verdict_card(args.ticker, horizon, result, metrics, flags, technical, as_of=as_of)
        return

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
        price_df=valuation_price_df,
        as_of=as_of,
        fred_rate=fred_rate,
        phase1_override=override,
    )

    # Recent 8-K events are deterministic filing-metadata facts, not model
    # output, so they're attached to the result post-interpret (like the
    # numbers the card renders directly) rather than routed through the LLM
    # payload. This also persists them with the verdict and carries them into
    # the HTML report, which reads result.events.
    if isinstance(result, dict):
        result["events"] = events
        # Insider (Form 4) activity is the same kind of fact as `events`: SEC
        # filing metadata classified deterministically, attached after
        # interpret() so it never enters the LLM payload or the fair value.
        if insider is not None:
            result["insider"] = insider
        if as_of is not None:
            result["as_of"] = as_of.isoformat()
        # Momentum context layer (price + fundamental + verdict momentum),
        # attached post-interpret like events -- never routed through the LLM
        # and never part of the fair-value computation. Computed before
        # save_verdict so the persisted verdict carries the momentum label.
        if "error" not in result:
            _attach_momentum(result, args.ticker, normalized, technical, as_of)
            _attach_macro(result, as_of, args.no_cache)
            _attach_peers(
                result, args.ticker, cik, submissions, normalized, ratios, metrics,
                as_of=as_of,
            )

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
                args.ticker, cik, horizon, verdict_provider or provider, price, result,
                db_path=Config.DB_PATH, valuation=result.get("valuation"),
                as_of=as_of.isoformat() if as_of is not None else None,
                catalyst_date=(catalyst or {}).get("estimate_date"),
                sic=(submissions or {}).get("sic"),
            )
        except Exception:  # noqa: BLE001 - persistence failure must not be fatal
            logger.warning("Failed to save verdict for %s", args.ticker, exc_info=True)

    _print_verdict_card(args.ticker, horizon, result, metrics, flags, technical, analyst=analyst, as_of=as_of)

    if getattr(args, "verbose", False):
        print("\n" + json.dumps(result, indent=2, ensure_ascii=False))

    if getattr(args, "html", False):
        # The earnings beat/miss history is display-only and only shown in the
        # HTML report's "Bilanço" tab, so it's fetched here (not on plain-text
        # runs) and, like the analyst consensus, suppressed in as-of mode
        # because it is undated and cannot be made point-in-time.
        earnings = None if as_of is not None else _fetch_earnings_history(args.ticker, args.no_cache)
        try:
            report_path = generate_report(
                args.ticker,
                horizon,
                result,
                metrics=metrics,
                technical=technical,
                flags=flags,
                price=price,
                as_of=price_as_of,
                price_source=(technical or {}).get("price_source"),
                analyst=analyst,
                analysis_as_of=as_of.isoformat() if as_of is not None else None,
                financials=serialize_financials(normalized, ratios),
                earnings=earnings,
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

    as_of = getattr(args, "as_of", None)
    rows = run_calibration(tickers, years=args.years, no_cache=args.no_cache, as_of=as_of)
    print()
    if as_of is not None:
        print(f"As-of (geçmiş tarih) modu: {as_of.isoformat()}")
    print_calibration_table(rows)

    summary = summarize_ratios(rows)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    path = save_calibration_snapshot(
        args.label, rows, summary, as_of=as_of.isoformat() if as_of is not None else None
    )
    if path:
        print(f"\nSaved calibration snapshot to: {path}")
    else:
        print("\nWARNING: failed to save calibration snapshot.", file=sys.stderr)


#: Column widths for the swing-scan terminal table (SWING_SPEC.md Sec.8).
_SWING_COL_WIDTHS = [4, 8, 6, 14, 22, 10, 10, 10, 10, 6]

#: How often (in scanned tickers) a progress line is logged during `swing`,
#: so a cold-cache 500-ticker scan doesn't flood the log with one line per
#: ticker but still shows it's making progress.
_SWING_PROGRESS_STEP = 25


def _print_swing_table(result: dict, top: int) -> None:
    """Print the ranked plain-text swing-scan table (SWING_SPEC.md Sec.8):
    rank, ticker, score, label, setup, price, entry/stop/target, R:R, plus a
    one-line summary of skipped tickers."""
    rows = result.get("rows") or []
    index_label = result.get("universe_label") or result.get("universe") or _DASH
    print(
        f"\nSwing Tarama — {index_label} — {result.get('generated_at') or _DASH} "
        f"(fiyat tarihi: {result.get('price_as_of') or _DASH}) — "
        f"{result.get('count', 0)}/{result.get('total', 0)} hisse tarandı"
    )

    headers = ["#", "Hisse", "Skor", "Etiket", "Kurulum", "Fiyat", "Giriş", "Stop", "Hedef", "R:R"]
    print("".join(h.ljust(w) for h, w in zip(headers, _SWING_COL_WIDTHS)))
    print("-" * sum(_SWING_COL_WIDTHS))

    for rank, row in enumerate(rows[:top], start=1):
        rr = row.get("rr")
        cells = [
            str(rank),
            str(row.get("ticker") or _DASH),
            str(row.get("score") if row.get("score") is not None else _DASH),
            str(row.get("label") or _DASH),
            str(row.get("setup") or _DASH),
            _fmt_swing_price(row.get("price")),
            _fmt_swing_price(row.get("entry")),
            _fmt_swing_price(row.get("stop")),
            _fmt_swing_price(row.get("target")),
            f"{rr:.2f}" if rr is not None else _DASH,
        ]
        print("".join(c.ljust(w) for c, w in zip(cells, _SWING_COL_WIDTHS)))

    skipped = result.get("skipped") or []
    if skipped:
        sample = ", ".join(str(s.get("ticker") or "?") for s in skipped[:5])
        more = f" (+{len(skipped) - 5} diğer)" if len(skipped) > 5 else ""
        print(f"\nAtlanan: {len(skipped)} hisse — {sample}{more}")
    else:
        print("\nAtlanan: yok")


def cmd_swing(args: argparse.Namespace) -> None:
    """Handle the ``swing`` subcommand: scan the S&P 500 (or a subset, via
    ``--limit``) for long-only swing-trade setups (SWING_SPEC.md Sec.8),
    print a ranked table, and persist the result unless ``--no-save``.

    Never raises out to the user: a failed scan prints a Turkish error line
    to stderr and exits with status 1, so a warmed-cache/network hiccup
    can't crash the CLI (CLAUDE.md: analysis layer never crashes the CLI).
    """
    index = normalize_index(getattr(args, "index", None))

    tickers = None
    if args.limit is not None:
        try:
            universe = load_universe(index=index)
        except OSError as exc:
            print(f"Evren dosyası okunamadı: {exc}", file=sys.stderr)
            sys.exit(1)
            return
        tickers = [row["ticker"] for row in universe[: args.limit]]

    def _progress(done: int, total: int, ticker: str) -> None:
        if done % _SWING_PROGRESS_STEP == 0 or done == total:
            logger.info("Swing tarama ilerlemesi: %d/%d (son: %s)", done, total, ticker)

    try:
        result = scan_swing(
            tickers=tickers,
            no_cache=args.no_cache,
            max_workers=args.workers,
            progress_cb=_progress,
            index=index,
        )
    except Exception as exc:  # noqa: BLE001 - a failed scan must not crash the CLI
        logger.exception("Swing tarama başarısız oldu")
        print(f"Swing tarama başarısız: {exc}", file=sys.stderr)
        sys.exit(1)
        return

    _print_swing_table(result, args.top)

    if not args.no_save:
        scan_id = save_swing_scan(result, db_path=Config.DB_PATH)
        if scan_id:
            print(f"\nTarama kaydedildi (id {scan_id}): {Config.DB_PATH}")
        else:
            print("\nWARNING: swing tarama veritabanına kaydedilemedi.", file=sys.stderr)


#: Column widths for the swing-study decile terminal table (SWING_STUDY_SPEC.md
#: Sec.5): Kova, n, ortalama fazla getiri %, medyan, isabet %, ortalama skor.
_SWING_STUDY_DECILE_COL_WIDTHS = [8, 8, 16, 12, 10, 12]


def _fmt_swing_study_pct(value) -> str:
    return _DASH if value is None else f"%{value:.2f}"


def _print_swing_study_report(result: dict) -> None:
    """Print the swing-study terminal report (SWING_STUDY_SPEC.md Sec.5):
    header, one decile table per horizon, top-minus-bottom spread,
    monotonicity, the per-setup table, then limitations and the disclaimer.
    """
    index_label = result.get("index_label") or result.get("index") or _DASH
    print(
        f"\n=== Swing skor kova (decile) çalışması — {index_label} — "
        f"{result.get('start') or _DASH} .. {result.get('end') or _DASH} ==="
    )
    print(
        f"Rebalance tarihi: {result.get('rebalance_dates', 0)} · "
        f"Gözlem: {result.get('observations', 0)} · "
        f"Taranan hisse: {result.get('tickers_scanned', 0)} · "
        f"Atlanan: {len(result.get('skipped') or [])} · "
        f"Skor-yok: {result.get('dropped_no_score', 0)} · "
        f"Vade-verisi-yok: {result.get('dropped_no_forward', 0)}"
    )

    deciles = result.get("deciles") or {}
    spread = result.get("spread") or {}
    monotonicity = result.get("monotonicity") or {}

    for horizon_key in sorted(deciles, key=lambda k: int(k)):
        print(f"\n[{horizon_key} işlem günü ileri getiri]")
        headers = ["Kova", "n", "Ort. fazla getiri", "Medyan", "İsabet %", "Ort. skor"]
        print("".join(h.ljust(w) for h, w in zip(headers, _SWING_STUDY_DECILE_COL_WIDTHS)))
        print("-" * sum(_SWING_STUDY_DECILE_COL_WIDTHS))
        for row in deciles.get(horizon_key) or []:
            flag = "  ⚠ yetersiz örneklem" if row.get("low_sample") else ""
            cells = [
                str(row.get("decile")),
                str(row.get("n")),
                _fmt_swing_study_pct(row.get("mean_excess_pct")),
                _fmt_swing_study_pct(row.get("median_excess_pct")),
                f"%{row['hit_rate_pct']:.1f}" if row.get("hit_rate_pct") is not None else _DASH,
                f"{row['mean_score']:.1f}" if row.get("mean_score") is not None else _DASH,
            ]
            print("".join(c.ljust(w) for c, w in zip(cells, _SWING_STUDY_DECILE_COL_WIDTHS)) + flag)

        sp = spread.get(horizon_key) or {}
        ci_low, ci_high = sp.get("ci_low"), sp.get("ci_high")
        ci_str = (
            f" (bootstrap %95 GA: {_fmt_swing_study_pct(ci_low)} .. {_fmt_swing_study_pct(ci_high)})"
            if ci_low is not None and ci_high is not None else ""
        )
        print(f"Üst-alt kova farkı: {_fmt_swing_study_pct(sp.get('mean_excess_pct'))}{ci_str}")
        print(f"Monotonluk (Spearman rho): {monotonicity.get(horizon_key, 0.0):.3f}")

    by_setup = result.get("by_setup") or {}
    if by_setup:
        print("\n[Kurulum türüne göre]")
        print(f"{'Kurulum':<24}{'n':>5}{'Fazla 10g':>12}{'Fazla 21g':>12}{'İsabet 10g':>12}{'İsabet 21g':>12}")
        print("-" * 77)
        for setup, row in sorted(by_setup.items()):
            hit_10 = row.get("hit_rate_10_pct")
            hit_21 = row.get("hit_rate_21_pct")
            hit_10_str = _DASH if hit_10 is None else f"%{hit_10:.1f}"
            hit_21_str = _DASH if hit_21 is None else f"%{hit_21:.1f}"
            print(
                f"{setup:<24}{row.get('n', 0):>5}"
                f"{_fmt_swing_study_pct(row.get('mean_excess_10_pct')):>12}"
                f"{_fmt_swing_study_pct(row.get('mean_excess_21_pct')):>12}"
                f"{hit_10_str:>12}{hit_21_str:>12}"
            )

    print("\n[Bilinen kısıtlar]")
    for line in result.get("limitations") or []:
        print(f"  - {line}")
    print(f"\n{result.get('disclaimer') or ''}")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Handle the ``backtest`` subcommand group: ``run``/``evaluate``/``report``.

    ``run`` executes the (ticker x date) as-of grid (deterministic, no-AI),
    persists verdicts, and evaluates forward outcomes. ``evaluate`` (re)computes
    outcomes for all stored verdicts. ``report`` prints the hit-rate /
    calibration / divergence tables and writes an HTML report. Each carries the
    backtest sample disclaimer.
    """
    from sec_analyzer.backtest import BACKTEST_DISCLAIMER
    from sec_analyzer.backtest.outcomes import evaluate_outcomes
    from sec_analyzer.backtest.report import (
        build_report_data,
        render_terminal,
        write_html_report,
    )
    from sec_analyzer.backtest.runner import parse_dates, read_tickers_file, run_backtest

    action = getattr(args, "backtest_action", None)

    if action == "run":
        tickers = read_tickers_file(args.tickers_file)
        if not tickers:
            print(f"No tickers found in {args.tickers_file}.", file=sys.stderr)
            return
        try:
            dates = parse_dates(args.dates)
        except ValueError as exc:
            print(f"Invalid --dates: {exc}", file=sys.stderr)
            return
        if not dates:
            print("No dates given to --dates.", file=sys.stderr)
            return
        print(
            f"Backtest grid: {len(tickers)} ticker × {len(dates)} tarih "
            f"= {len(tickers) * len(dates)} hücre (no-AI, script)."
        )
        tally = run_backtest(
            tickers, dates, years=args.years, no_cache=args.no_cache, db_path=Config.DB_PATH
        )
        print(
            f"\nBitti: {tally['ok']} ok, {tally['no_data']} veri-yok, "
            f"{tally['error']} hata (of {tally['cells']} hücre)."
        )
        if tally.get("outcomes"):
            o = tally["outcomes"]
            print(
                f"Outcomes: {o['evaluated']} değerlendirildi, "
                f"{o['skipped_immature']} vadesi dolmadı, {o['skipped_no_data']} veri-yok."
            )
        print(f"\n{BACKTEST_DISCLAIMER}")
        return

    if action == "evaluate":
        summary = evaluate_outcomes(db_path=Config.DB_PATH, no_cache=args.no_cache)
        print(
            f"Outcomes: {summary['evaluated']} değerlendirildi, "
            f"{summary['skipped_immature']} vadesi dolmadı, "
            f"{summary['skipped_no_data']} veri-yok (of {summary['verdicts_seen']} verdict)."
        )
        print(f"\n{BACKTEST_DISCLAIMER}")
        return

    if action == "report":
        data = build_report_data(db_path=Config.DB_PATH)
        print(render_terminal(data))
        generated_on = date.today().isoformat()
        path = write_html_report(data, generated_on)
        print(f"\nHTML backtest raporu: {path}")
        return

    if action == "swing":
        from sec_analyzer.backtest.swing_study import run_swing_study
        from sec_analyzer.screener.universe import normalize_index
        from sec_analyzer.store.database import save_swing_study

        index = normalize_index(getattr(args, "index", None))

        def _progress(done: int, total: int, ticker: str) -> None:
            if done % _SWING_PROGRESS_STEP == 0 or done == total:
                logger.info("Swing çalışması ilerlemesi: %d/%d (son: %s)", done, total, ticker)

        try:
            result = run_swing_study(
                index=index,
                start=args.start,
                end=args.end,
                max_workers=args.workers,
                progress_cb=_progress,
            )
        except Exception as exc:  # noqa: BLE001 - a failed study must not crash the CLI
            logger.exception("Swing skor çalışması başarısız oldu")
            print(f"Swing skor çalışması başarısız: {exc}", file=sys.stderr)
            sys.exit(1)
            return

        _print_swing_study_report(result)

        if not args.no_save:
            study_id = save_swing_study(result, db_path=Config.DB_PATH)
            if study_id:
                print(f"\nÇalışma kaydedildi (id {study_id}): {Config.DB_PATH}")
            else:
                print("\nWARNING: swing çalışması veritabanına kaydedilemedi.", file=sys.stderr)
        return

    print("Usage: backtest {run|evaluate|report|swing} ...", file=sys.stderr)


def _parse_as_of(value: str) -> date:
    """argparse ``type`` for ``--as-of``: parse an ISO date, reject the future.

    A point-in-time cutoff after today is meaningless (there's no "future
    knowledge" to restrict to) and almost always a typo, so it's rejected up
    front rather than silently behaving like a live run.
    """
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid as-of date {value!r}; expected YYYY-MM-DD."
        )
    if parsed > date.today():
        raise argparse.ArgumentTypeError(
            f"as-of date {value} is in the future; expected a past date."
        )
    return parsed


# --- assumptions {propose|show|edit|freeze} (ASSUMPTIONS_CACHE_SPEC.md) ---

#: Scenario keys and numeric fields an `assumptions edit --set` path may target.
_ASSUMPTION_SCENARIOS = ("bear", "base", "bull")
_ASSUMPTION_NUMERIC_FIELDS = ("growth_5y", "terminal_growth", "discount_rate")
_ASSUMPTION_FIELDS = _ASSUMPTION_NUMERIC_FIELDS + ("story",)


def _default_model_for(provider: Optional[str]) -> Optional[str]:
    """The configured default model name for an LLM provider (``None`` for script).

    ``claude_code`` returns ``None`` here (its model is resolved per-phase in
    the dispatcher, or by the CLI itself, not from a single default)."""
    p = (provider or "").lower()
    if p in ("ollama", "gemma"):
        return Config.OLLAMA_MODEL
    return None


def _fmt_frac_pct(value) -> str:
    """Format a decimal fraction (0.105) as a Turkish percent string ('%10.5')."""
    if not isinstance(value, (int, float)):
        return _DASH
    return f"%{value * 100:.1f}"


def _assumptions_resolve_inputs(args: argparse.Namespace):
    """Shared fetch/normalize/metrics + CAPM resolution for `assumptions propose`.

    Mirrors exactly what :func:`sec_analyzer.interpret.analyzer.interpret`
    feeds phase 1 today (sector hint, CAPM cost of equity, global risk-free),
    so a cached proposal is comparable to the legacy live path. Returns
    ``(cik, name, normalized, ratios, metrics, sector_hint, capm,
    risk_free_pct)``.
    """
    cik, name, normalized, ratios = _fetch_normalize_store(args)
    metrics = compute_metrics(normalized, ratios, None)
    submissions = _fetch_submissions(cik, args.ticker, args.no_cache)
    sic = (submissions or {}).get("sic")
    sic_description = (submissions or {}).get("sicDescription")
    sector_hint = classify_sector(sic, normalized, metrics) if sic is not None else None
    sector_data = damodaran.load_sector_data(Config.DAMODARAN_DIR)
    capm = compute_cost_of_equity(
        sector_data, sic_description, metrics,
        is_unprofitable=(sector_hint == "growth_unprofitable"),
    )
    risk_free_pct = sector_data.get("risk_free") if sector_data else None
    return cik, name, normalized, ratios, metrics, sector_hint, capm, risk_free_pct


def _print_assumptions_review(
    ticker: str,
    set_id: Optional[int],
    sector_type: str,
    source_provider: str,
    source_model: Optional[str],
    clamped: dict,
    script_baseline: Optional[dict],
    sanity_notes: List[str],
) -> None:
    """Render the propose/edit review card: proposed vs. script baseline.

    Shows each scenario's growth_5y / terminal_growth / discount_rate side by
    side with the deterministic CAPM/CAGR baseline and the per-field delta (in
    percentage points), then the base story and any clamp/sanity notes. Large
    divergence is information, not an error.
    """
    model_str = f" ({source_model})" if source_model else ""
    print(f"\n=== {ticker} — Varsayım Önerisi (taslak #{set_id}) ===")
    print(f"Sektör tipi: {sector_type}   |   Kaynak: {source_provider}{model_str}")
    print(f"\n{'Senaryo':<6} {'Alan':<16} {'Öneri':>9} {'Script':>9} {'Δ':>9}")
    print("-" * 52)
    field_labels = {
        "growth_5y": "büyüme 5y",
        "terminal_growth": "terminal",
        "discount_rate": "iskonto",
    }
    for scenario in _ASSUMPTION_SCENARIOS:
        prop = (clamped or {}).get(scenario) or {}
        base = (script_baseline or {}).get(scenario) or {}
        for field in _ASSUMPTION_NUMERIC_FIELDS:
            pv = prop.get(field)
            bv = base.get(field)
            if isinstance(pv, (int, float)) and isinstance(bv, (int, float)):
                delta = f"{(pv - bv) * 100:+.1f}pp"
            else:
                delta = _DASH
            print(
                f"{scenario:<6} {field_labels[field]:<16} "
                f"{_fmt_frac_pct(pv):>9} {_fmt_frac_pct(bv):>9} {delta:>9}"
            )
    base_story = ((clamped or {}).get("base") or {}).get("story")
    if base_story:
        print(f"\nHikaye (base): {base_story}")
    if sanity_notes:
        print("\nClamp/sanity notları:")
        for note in sanity_notes:
            print(f"  - {note}")


def _cmd_assumptions_propose(args: argparse.Namespace) -> None:
    """`assumptions propose TICKER` — the only step that may call an LLM.

    Proposes a phase-1 assumption set (via the chosen provider), clamps it,
    builds the deterministic script baseline for side-by-side review, prints
    the review card, and stores the result as a DRAFT (replacing any prior
    unreviewed draft). Never freezes automatically.
    """
    provider = getattr(args, "provider", None) or Config.ANALYZER_PROVIDER
    (
        cik, _name, normalized, ratios, metrics, sector_hint, capm, risk_free_pct
    ) = _assumptions_resolve_inputs(args)

    print(f"\n'{provider}' ile varsayım önerisi hazırlanıyor (phase 1)...")
    phase1 = propose_assumptions(
        normalized, ratios, metrics, sector_hint=sector_hint,
        provider=provider, model=getattr(args, "model", None),
        capm=capm, risk_free_pct=risk_free_pct,
    )
    actual_provider = phase1.get("_provider") or provider
    sector_type = sector_hint or phase1.get("sector_type") or "mature"
    is_unprofitable = sector_type == "growth_unprofitable"

    raw_assumptions = phase1["assumptions"]
    clamped, clamp_notes = clamp_assumptions(raw_assumptions, is_unprofitable=is_unprofitable)
    script_baseline = rule_based.default_assumptions(
        metrics, sector_type, capm=capm, risk_free_pct=risk_free_pct
    )

    if actual_provider == "claude_code":
        # propose is a phase-1 (assumptions) step -> the strong model.
        source_model = getattr(args, "model", None) or Config.CLAUDE_CODE_MODEL_ASSUMPTIONS
    elif actual_provider in ("ollama", "gemma"):
        source_model = getattr(args, "model", None) or _default_model_for(actual_provider)
    else:
        source_model = None

    payload = {
        "fundamental_fy": resolve_fundamental_fy(metrics),
        "facts_fingerprint": assumptions_store.fingerprint_annual(normalized),
        "source_provider": actual_provider,
        "source_model": source_model,
        "sector_type": sector_type,
        "assumptions": clamped,
        "hyper_extras": phase1.get("hyper_growth_extras"),
        "script_baseline": script_baseline,
        "sanity_notes": clamp_notes,
    }
    set_id = assumptions_store.save_draft(cik, args.ticker, payload, db_path=Config.DB_PATH)

    _print_assumptions_review(
        args.ticker, set_id, sector_type, actual_provider, source_model,
        clamped, script_baseline, clamp_notes,
    )
    if set_id is None:
        print("\nUYARI: taslak veritabanına kaydedilemedi (log'a bakın).", file=sys.stderr)
        return
    print(
        f"\nGözden geçir, sonra dondur:  assumptions freeze {args.ticker}"
        f"\nDüzeltmek için:              assumptions edit {args.ticker} --set base.growth_5y=0.12"
    )


def _freshness_label(cik: str, row: Optional[dict], args: argparse.Namespace) -> str:
    """Turkish freshness label for a stored set vs. the current fundamentals."""
    if row is None:
        return _DASH
    try:
        client = SecHttpClient()
        facts = get_company_facts(cik, client, no_cache=args.no_cache)
        normalized = normalize_facts(facts, years=args.years, as_of=None)
        ratios = compute_ratios(normalized)
        metrics = compute_metrics(normalized, ratios, None)
    except Exception:  # noqa: BLE001 - freshness is best-effort; never crash `show`
        logger.warning("Could not fetch current fundamentals for freshness check", exc_info=True)
        return "bilinmiyor (finansallar getirilemedi)"
    if assumptions_store.is_fresh(row, normalized, metrics):
        return "GÜNCEL"
    return "BAYAT — yeni dosyalama var; yeniden 'propose' önerilir"


def _cmd_assumptions_show(args: argparse.Namespace) -> None:
    """`assumptions show TICKER` — print the draft, frozen set, and history."""
    client = SecHttpClient()
    cik, _name = resolve_cik(args.ticker, client, no_cache=args.no_cache)
    draft = assumptions_store.load_active(cik, assumptions_store.STATUS_DRAFT, db_path=Config.DB_PATH)
    frozen = assumptions_store.load_active(cik, assumptions_store.STATUS_FROZEN, db_path=Config.DB_PATH)
    history = assumptions_store.load_history(cik, db_path=Config.DB_PATH)

    print(f"\n=== {args.ticker} — Varsayım Setleri ===")

    if frozen is not None:
        freshness = _freshness_label(cik, frozen, args)
        model_str = f" ({frozen.get('source_model')})" if frozen.get("source_model") else ""
        print(
            f"\nDONDURULMUŞ (aktif) #{frozen['id']}  |  {freshness}"
            f"\n  Kaynak: {frozen.get('source_provider')}{model_str}"
            f"  |  önerilme: {frozen.get('proposed_at')}  |  dondurma: {frozen.get('frozen_at')}"
            f"\n  Sektör: {frozen.get('sector_type')}  |  fundamental FY: {frozen.get('fundamental_fy')}"
        )
        if frozen.get("review_note"):
            print(f"  Not: {frozen['review_note']}")
        _print_assumptions_review(
            args.ticker, frozen["id"], frozen.get("sector_type"),
            frozen.get("source_provider"), frozen.get("source_model"),
            frozen.get("assumptions"), frozen.get("script_baseline"),
            frozen.get("sanity_notes") or [],
        )
    else:
        print("\nDondurulmuş set yok.")

    if draft is not None:
        print(
            f"\nTASLAK #{draft['id']} (henüz dondurulmadı)  |  "
            f"kaynak: {draft.get('source_provider')}  |  önerilme: {draft.get('proposed_at')}"
        )
        print(f"  Dondurmak için: assumptions freeze {args.ticker}")

    superseded = [r for r in history if r.get("status") == assumptions_store.STATUS_SUPERSEDED]
    if superseded:
        print("\nGeçmiş (supersede edilmiş):")
        for r in superseded:
            print(
                f"  #{r['id']}  {r.get('source_provider')}  "
                f"dondurma: {r.get('frozen_at')} → supersede: {r.get('superseded_at')}"
            )


def _cmd_assumptions_edit(args: argparse.Namespace) -> None:
    """`assumptions edit TICKER --set PATH=VALUE ...` — edit the draft in place."""
    client = SecHttpClient()
    cik, _name = resolve_cik(args.ticker, client, no_cache=args.no_cache)
    draft = assumptions_store.load_active(cik, assumptions_store.STATUS_DRAFT, db_path=Config.DB_PATH)
    if draft is None:
        print(
            f"'{args.ticker}' için taslak yok. Önce: assumptions propose {args.ticker}",
            file=sys.stderr,
        )
        return

    assumptions = copy.deepcopy(draft.get("assumptions") or {})
    for expr in args.set or []:
        path, sep, raw_value = expr.partition("=")
        if not sep:
            print(f"Geçersiz --set '{expr}'; beklenen biçim PATH=VALUE.", file=sys.stderr)
            return
        scenario, _, field = path.strip().partition(".")
        if scenario not in _ASSUMPTION_SCENARIOS or field not in _ASSUMPTION_FIELDS:
            print(
                f"Geçersiz yol '{path}'. Senaryo ∈ {_ASSUMPTION_SCENARIOS}, "
                f"alan ∈ {_ASSUMPTION_FIELDS}.",
                file=sys.stderr,
            )
            return
        if field == "story":
            value = raw_value
        else:
            try:
                value = float(raw_value)
            except ValueError:
                print(f"'{path}' için sayısal değer beklendi (ondalık kesir), '{raw_value}' verildi.", file=sys.stderr)
                return
        assumptions.setdefault(scenario, {})[field] = value

    updated = assumptions_store.update_draft(
        cik, assumptions, sector_type=None, review_note=getattr(args, "note", None),
        db_path=Config.DB_PATH,
    )
    if updated is None:
        print("UYARI: taslak güncellenemedi (log'a bakın).", file=sys.stderr)
        return
    _print_assumptions_review(
        args.ticker, updated["id"], updated.get("sector_type"),
        updated.get("source_provider"), updated.get("source_model"),
        updated.get("assumptions"), updated.get("script_baseline"),
        updated.get("sanity_notes") or [],
    )
    print(f"\nDondurmak için: assumptions freeze {args.ticker}")


def _cmd_assumptions_freeze(args: argparse.Namespace) -> None:
    """`assumptions freeze TICKER` — promote the draft to the active frozen set."""
    client = SecHttpClient()
    cik, _name = resolve_cik(args.ticker, client, no_cache=args.no_cache)
    draft = assumptions_store.load_active(cik, assumptions_store.STATUS_DRAFT, db_path=Config.DB_PATH)
    if draft is None:
        print(
            f"'{args.ticker}' için dondurulacak taslak yok. Önce: assumptions propose {args.ticker}",
            file=sys.stderr,
        )
        return

    # Final defense-in-depth guard: the draft is already clamped (propose/edit),
    # so this should pass -- but refuse to freeze a set that somehow violates
    # the sanity bounds rather than persist a bad artifact.
    violations = validate_assumptions(
        draft.get("assumptions") or {},
        is_unprofitable=(draft.get("sector_type") == "growth_unprofitable"),
    )
    if violations:
        print("Dondurulamadı — sanity ihlalleri var:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(f"Düzeltin: assumptions edit {args.ticker} --set ...", file=sys.stderr)
        return

    frozen = assumptions_store.freeze_draft(cik, getattr(args, "note", None), db_path=Config.DB_PATH)
    if frozen is None:
        print("UYARI: dondurma başarısız (log'a bakın).", file=sys.stderr)
        return
    print(
        f"\n{args.ticker} için varsayım seti donduruldu (#{frozen['id']}, "
        f"{frozen.get('frozen_at')}).\n"
        f"Artık 'analyze {args.ticker}' (varsayılan --assumptions auto) bu seti kullanır."
    )


def _resolve_analyze_phase1(
    args: argparse.Namespace,
    cik: str,
    normalized: dict,
    ratios: List[dict],
    metrics: dict,
    submissions: Optional[dict],
    as_of,
    fred_rate: Optional[dict],
):
    """Resolve the phase-1 assumption source for `analyze` (Sec.4 policy).

    Returns ``(phase1_override, note, stop, verdict_provider)``:

    * ``phase1_override``: a phase-1-result dict to pass to ``interpret``
      (from a fresh frozen set, or a deterministic script build), or ``None``
      to let ``interpret`` run its ordinary phase-1 (legacy ``llm`` mode, and
      every as-of case).
    * ``note``: a Turkish line to print explaining what happened, or ``None``.
    * ``stop``: ``True`` only for strict ``--assumptions frozen`` when no fresh
      frozen set exists -- the caller prints an error card and does not analyze.
    * ``verdict_provider``: ``"cached:<src>"`` when a frozen set is used (so the
      verdict's provider column shows cached provenance at a glance), else
      ``None`` (use the phase-2 provider).
    """
    strategy = getattr(args, "assumptions", "auto") or "auto"
    sic = (submissions or {}).get("sic")
    sic_description = (submissions or {}).get("sicDescription")
    sector_hint = classify_sector(sic, normalized, metrics) if sic is not None else None

    # As-of mode never consults the cache: a set proposed today would leak
    # post-cutoff knowledge into a point-in-time analysis. auto/frozen degrade
    # to the deterministic path already forced upstream (provider -> script).
    if as_of is not None:
        note = None
        if strategy in ("auto", "frozen"):
            note = (
                "as-of modunda dondurulmuş varsayım seti kullanılmaz (hindsight "
                "riski); deterministik varsayımlarla devam edildi."
            )
        return None, note, False, None

    if strategy == "llm":
        return None, None, False, None

    if strategy == "script":
        override = build_script_phase1(
            normalized, ratios, metrics, sector_hint, sic_description, fred_rate=fred_rate
        )
        return override, None, False, None

    # auto / frozen: try the frozen set first.
    frozen = assumptions_store.load_active(
        cik, assumptions_store.STATUS_FROZEN, db_path=Config.DB_PATH
    )
    if frozen is not None and assumptions_store.is_fresh(frozen, normalized, metrics):
        src = frozen.get("source_provider")
        override = {
            "assumptions": frozen.get("assumptions"),
            "sector_type": frozen.get("sector_type"),
            "hyper_growth_extras": frozen.get("hyper_extras"),
            "_provider": f"cached:{src}",
            "_assumption_set_id": frozen.get("id"),
        }
        note = f"Dondurulmuş varsayım seti #{frozen.get('id')} kullanıldı (kaynak: {src})."
        return override, note, False, f"cached:{src}"

    reason = "yok" if frozen is None else "bayat (yeni dosyalama var)"
    if strategy == "frozen":
        note = (
            f"--assumptions frozen: kullanılabilir güncel dondurulmuş set {reason}; "
            f"analiz durduruldu. Öneri: assumptions propose {args.ticker}"
        )
        return None, note, True, None

    # auto fallback -> deterministic script assumptions.
    override = build_script_phase1(
        normalized, ratios, metrics, sector_hint, sic_description, fred_rate=fred_rate
    )
    note = (
        f"Dondurulmuş varsayım seti {reason}; deterministik (script) varsayımlar "
        f"kullanıldı. Öneri: assumptions propose {args.ticker}"
    )
    return override, note, False, None


def cmd_assumptions(args: argparse.Namespace) -> None:
    """Dispatch the ``assumptions`` subcommand group."""
    action = getattr(args, "assumptions_action", None)
    if action == "propose":
        _cmd_assumptions_propose(args)
    elif action == "show":
        _cmd_assumptions_show(args)
    elif action == "edit":
        _cmd_assumptions_edit(args)
    elif action == "freeze":
        _cmd_assumptions_freeze(args)
    else:
        print("Usage: assumptions {propose|show|edit|freeze} TICKER ...", file=sys.stderr)


#: Column widths for the portfolio-overview terminal table: ticker, sector,
#: verdict, price, fair value (base band mid), gap %, confidence, momentum,
#: insider, age. The verdict column is sized for the longest label the engine
#: actually emits ("MODEL-PİYASA AYRIŞMASI"); the valuation route is not a
#: column here because it already has its own summary table above.
_OVERVIEW_COL_WIDTHS = [8, 19, 24, 11, 11, 9, 8, 10, 12, 5]

#: Column widths for the pre-earnings briefing tables: ticker, date, days,
#: verdict, gap %, momentum, beat record, verdict age. The verdict column is
#: sized for the longest label the engine emits ("MODEL-PİYASA AYRIŞMASI").
_PREEARNINGS_COL_WIDTHS = [8, 13, 7, 24, 10, 10, 9, 6]

#: Cap on how many per-name observation blocks the briefing prints before
#: summarizing the rest. A full watchlist run can match dozens of names, and a
#: briefing nobody reads to the end is not a briefing.
_PREEARNINGS_MAX_NOTE_BLOCKS = 20


def _fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    """Render a percentage with an explicit sign, or the em-dash placeholder."""
    if value is None:
        return _DASH
    return f"{value:+.{decimals}f}%"


def _fmt_price(value: Optional[float]) -> str:
    """Render a price, or the em-dash placeholder for a missing one."""
    return _DASH if value is None else f"{value:,.2f}"


def _print_overview_table(overview: dict) -> None:
    """Print the portfolio overview: the sector heat map, then one row per
    analyzed ticker, then the stale / upcoming-earnings / drift call-outs.

    Everything here is read back from stored verdicts -- no network, no
    valuation is re-run, so the numbers are exactly what the last analysis of
    each name produced.
    """
    rows = overview.get("rows") or []
    print(
        f"\nPortföy Genel Bakış — {overview.get('today') or _DASH} — "
        f"{overview.get('count', 0)} hisse "
        f"(medyan makul-değer/fiyat farkı: {_fmt_pct(overview.get('median_fv_vs_price_pct'))})"
    )

    sectors = overview.get("sectors") or []
    if sectors:
        print("\n[Sektör ısı haritası]")
        print(f"{'Sektör':<28}{'n':>4}{'Medyan fark':>14}{'Ucuz':>7}{'Makul':>7}{'Pahalı':>8}{'Bayat':>7}")
        print("-" * 75)
        for sector in sectors:
            print(
                f"{str(sector.get('label') or _DASH)[:27]:<28}"
                f"{sector.get('n', 0):>4}"
                f"{_fmt_pct(sector.get('median_fv_vs_price_pct')):>14}"
                f"{sector.get('cheap_n', 0):>7}"
                f"{sector.get('fair_n', 0):>7}"
                f"{sector.get('expensive_n', 0):>8}"
                f"{sector.get('stale_n', 0):>7}"
            )

    routes = overview.get("routes") or []
    if routes:
        print("\n[Değerleme yoluna göre]")
        print(f"{'Yol':<22}{'n':>4}{'Medyan fark':>14}{'Ucuz':>7}{'Makul':>7}{'Pahalı':>8}")
        print("-" * 62)
        for route in routes:
            print(
                f"{str(route.get('label') or _DASH)[:21]:<22}"
                f"{route.get('n', 0):>4}"
                f"{_fmt_pct(route.get('median_fv_vs_price_pct')):>14}"
                f"{route.get('cheap_n', 0):>7}"
                f"{route.get('fair_n', 0):>7}"
                f"{route.get('expensive_n', 0):>8}"
            )

    if not rows:
        print("\nKayıtlı analiz bulunamadı — önce `analyze TICKER` çalıştırın.")
        return

    print("\n[Hisseler — makul değer / fiyat farkına göre]")
    headers = ["Hisse", "Sektör", "Karar", "Fiyat", "Makul", "Fark", "Güven", "Momentum", "İçeriden", "Yaş"]
    print("".join(h.ljust(w) for h, w in zip(headers, _OVERVIEW_COL_WIDTHS)))
    print("-" * sum(_OVERVIEW_COL_WIDTHS))
    for row in rows:
        age = row.get("age_days")
        cells = [
            str(row.get("ticker") or _DASH),
            str(row.get("sector") or "Sınıflandırılmamış")[:18],
            str(row.get("fundamental_verdict") or _DASH)[:23],
            _fmt_price(row.get("price")),
            _fmt_price(row.get("fv_base_mid")),
            _fmt_pct(row.get("fv_vs_price_pct")),
            str(row.get("confidence") or _DASH),
            str(row.get("momentum_verdict") or _DASH)[:9],
            str(row.get("insider_verdict") or _DASH)[:11],
            (f"{age}g" if age is not None else _DASH),
        ]
        print("".join(c.ljust(w) for c, w in zip(cells, _OVERVIEW_COL_WIDTHS)))

    upcoming = overview.get("upcoming_earnings") or []
    if upcoming:
        window = overview.get("earnings_window_days")
        print(f"\n[Yaklaşan bilançolar — {window} gün]")
        for row in upcoming:
            print(
                f"  {str(row.get('ticker') or _DASH):<8}"
                f"{row.get('catalyst_date') or _DASH}  "
                f"({row.get('days_until_earnings')} gün) — "
                f"{row.get('fundamental_verdict') or _DASH}"
            )

    # The engine's own verdict asks WHERE IN THE BAND the price sits (inside
    # the base band -> MAKUL); this dashboard's bucket asks HOW FAR the band's
    # MIDPOINT is from the price. On a wide band those two can disagree while
    # both being right, which is why the disagreement is worth naming: it
    # isolates the names whose midpoint leans hard one way but whose band is
    # too wide for the engine to commit.
    drift = overview.get("drift") or []
    if drift:
        print("\n[Karar ile band ortası arasında gerilim]")
        print("  (fiyat baz bandın içinde kaldığı için karar MAKUL; band geniş olduğundan orta nokta uzakta)")
        for row in drift[:10]:
            lo, hi = row.get("fv_base_lo"), row.get("fv_base_hi")
            band = f"{_fmt_price(lo)}–{_fmt_price(hi)}" if lo is not None and hi is not None else _DASH
            print(
                f"  {str(row.get('ticker') or _DASH):<8}"
                f"karar: {str(row.get('fundamental_verdict') or _DASH):<9}"
                f"band ortası: {str(row.get('value_bucket') or _DASH):<8}"
                f"fark: {_fmt_pct(row.get('fv_vs_price_pct')):<9}"
                f"band: {band:<22}"
                f"güven: {row.get('confidence') or _DASH}"
            )

    stale = overview.get("stale") or []
    if stale:
        names = ", ".join(str(r.get("ticker") or "?") for r in stale[:12])
        more = f" (+{len(stale) - 12} diğer)" if len(stale) > 12 else ""
        print(f"\nBayat analiz ({overview.get('stale_days')} günden eski): {len(stale)} — {names}{more}")


def cmd_overview(args: argparse.Namespace) -> None:
    """Handle the ``overview`` subcommand: a network-free dashboard over every
    ticker that has a stored verdict -- sector heat map, per-name fair-value
    gap, staleness, upcoming earnings, and verdict drift.

    Reads only the local database, so it is instant and works offline. Never
    raises out to the user (CLAUDE.md: the analysis layer never crashes the CLI).
    """
    try:
        rows = load_latest_verdicts(db_path=Config.DB_PATH)
    except Exception as exc:  # noqa: BLE001 - a read failure must not crash the CLI
        logger.exception("Portföy genel bakış verisi okunamadı")
        print(f"Genel bakış okunamadı: {exc}", file=sys.stderr)
        sys.exit(1)
        return

    overview = build_overview(
        rows,
        stale_days=args.stale_days,
        earnings_window_days=args.earnings_window,
    )
    _print_overview_table(overview)

    if getattr(args, "json", False):
        print("\n" + json.dumps(overview, indent=2, ensure_ascii=False))


def _preearnings_row_cells(row: dict) -> List[str]:
    """Render one briefing row's cells, shared by both briefing tables."""
    age = row.get("verdict_age_days")
    beat = row.get("beat_count")
    surprises = row.get("surprises") or []
    return [
        str(row.get("ticker") or _DASH),
        str(row.get("earnings_date") or _DASH),
        (_DASH if row.get("just_reported") else str(row.get("days_until", _DASH))),
        str(row.get("verdict") or _DASH)[:23],
        _fmt_pct(row.get("fv_vs_price_pct")),
        str(row.get("momentum_verdict") or _DASH)[:9],
        (f"{beat}/{len(surprises)}" if beat is not None and surprises else _DASH),
        (f"{age}g" if age is not None else _DASH),
    ]


def _print_preearnings_table(rows: List[dict], date_header: str) -> None:
    """Print one briefing table with ``date_header`` naming what its date
    column means -- the two sections below carry different dates (an upcoming
    estimate vs. the estimate for the quarter AFTER the one just published),
    and one shared header would misstate the second."""
    headers = ["Hisse", date_header, "Gün", "Karar", "Fark", "Momentum", "Sürpriz", "Yaş"]
    print("".join(h.ljust(w) for h, w in zip(headers, _PREEARNINGS_COL_WIDTHS)))
    print("-" * sum(_PREEARNINGS_COL_WIDTHS))
    for row in rows:
        print("".join(c.ljust(w) for c, w in zip(_preearnings_row_cells(row), _PREEARNINGS_COL_WIDTHS)))


def _print_preearnings_briefing(result: dict, include_all: bool = False) -> None:
    """Print the pre-earnings briefing, then the per-name observations.

    Upcoming and just-reported names are printed as two separate tables. They
    are different states -- one is a catalyst still ahead, the other a print
    that already landed -- and during earnings season the just-reported set is
    large enough to bury the names actually about to report. Their date columns
    also mean different things: for a just-reported name the estimate shown is
    for the quarter AFTER the one just published, which can be months out.
    """
    rows = result.get("rows") or []
    reported = [r for r in rows if r.get("just_reported")]
    upcoming = [r for r in rows if not r.get("just_reported")]

    # With --all the window is not applied, so naming it in the header would
    # contradict the rows below it (names months out under a "14 gün" title).
    window_text = "tüm takvim" if include_all else f"önümüzdeki {result.get('within_days')} gün"
    print(
        f"\nBilanço Öncesi Brifing — {result.get('today') or _DASH} — "
        f"{window_text} — "
        f"{result.get('count', 0)}/{result.get('requested', 0)} hisse"
    )

    if not rows:
        print("\nBu pencerede bilanço açıklayacak isim yok.")

    if upcoming:
        print(f"\n[Yaklaşan bilançolar — {len(upcoming)} hisse]")
        _print_preearnings_table(upcoming, "Bilanço")
    elif rows:
        print("\n[Yaklaşan bilançolar] yok — aşağıdaki isimler bilançosunu yeni açıkladı.")

    if reported:
        print(f"\n[Yeni açıklananlar — {len(reported)} hisse · tarih sütunu SONRAKİ çeyreğin tahmini]")
        _print_preearnings_table(reported, "Sonraki")

    # Observations: upcoming names first -- they are the ones a reader can
    # still act on before the print.
    ordered = upcoming + reported
    for row in ordered[:_PREEARNINGS_MAX_NOTE_BLOCKS]:
        notes = row.get("notes") or []
        if not notes:
            continue
        print(f"\n{row.get('ticker')} — {row.get('catalyst_label') or _DASH}")
        for note in notes:
            print(f"  • {note}")
        surprises = row.get("surprises") or []
        if surprises:
            cells = " · ".join(
                f"{q.get('period') or _DASH}: {format_surprise_pct(q.get('surprise_pct'))}"
                for q in surprises
            )
            print(f"  Geçmiş sürprizler: {cells}")
    if len(ordered) > _PREEARNINGS_MAX_NOTE_BLOCKS:
        print(
            f"\n(+{len(ordered) - _PREEARNINGS_MAX_NOTE_BLOCKS} hissenin notları gösterilmedi — "
            "daraltmak için --within kullanın veya tam veri için --json)"
        )

    skipped = result.get("skipped") or []
    if skipped:
        print(f"\nAtlanan: {len(skipped)} hisse")
        for entry in skipped[:10]:
            print(f"  {entry.get('ticker') or '?'}: {entry.get('reason') or _DASH}")
        if len(skipped) > 10:
            print(f"  (+{len(skipped) - 10} diğer)")


def cmd_preearnings(args: argparse.Namespace) -> None:
    """Handle the ``preearnings`` subcommand: which watchlist names report
    soon, and what the stored analysis currently says about them.

    The watchlist defaults to every ticker with a stored verdict (i.e. what
    the user has actually analyzed); ``--tickers``/``--tickers-file`` override
    it. Never raises out to the user.
    """
    tickers: List[str] = []
    if getattr(args, "tickers", None):
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif getattr(args, "tickers_file", None):
        try:
            with open(args.tickers_file, "r", encoding="utf-8") as handle:
                tickers = [
                    line.strip().upper()
                    for line in handle
                    if line.strip() and not line.strip().startswith("#")
                ]
        except OSError as exc:
            print(f"İzleme listesi dosyası okunamadı: {exc}", file=sys.stderr)
            sys.exit(1)
            return
    else:
        tickers = load_watchlist_tickers(db_path=Config.DB_PATH)
        if not tickers:
            print(
                "İzleme listesi boş: kayıtlı analiz yok. Önce `analyze TICKER` "
                "çalıştırın veya --tickers / --tickers-file verin.",
                file=sys.stderr,
            )
            sys.exit(1)
            return

    def _progress(done: int, total: int, ticker: str) -> None:
        logger.info("Bilanço brifingi ilerlemesi: %d/%d (son: %s)", done, total, ticker)

    include_all = getattr(args, "all", False)
    result = scan_preearnings(
        tickers,
        within_days=args.within,
        no_cache=args.no_cache,
        db_path=Config.DB_PATH,
        progress_cb=_progress,
        include_all=include_all,
    )
    _print_preearnings_briefing(result, include_all=include_all)

    if getattr(args, "json", False):
        print("\n" + json.dumps(result, indent=2, ensure_ascii=False))


#: Metrics shown, in this order, by the `peers` command's sector table.
_PEER_TABLE_METRICS = [
    ("net_margin", "Net marj"),
    ("operating_margin", "Faaliyet marjı"),
    ("gross_margin", "Brüt marj"),
    ("fcf_margin", "FCF marjı"),
    ("roe", "ROE"),
    ("roa", "ROA"),
    ("debt_to_equity", "Borç/Özkaynak"),
    ("revenue_growth", "Gelir büyümesi"),
]


def _fmt_ratio(value: Optional[float]) -> str:
    """Render a decimal-fraction metric as a Turkish-style percentage."""
    if value is None:
        return _DASH
    return f"%{value * 100:.1f}".replace(".", ",")


def _print_peer_snapshot(snapshot: dict) -> None:
    """Print a built/stored peer snapshot: coverage, then per-sector medians."""
    print(
        f"\nEmsal Kesiti — {snapshot.get('year')} mali yılı — "
        f"{snapshot.get('covered', 0)}/{snapshot.get('universe_size', 0)} şirket "
        f"({', '.join(snapshot.get('indexes') or []) or _DASH})"
    )
    missing = snapshot.get("frames_missing") or []
    if missing:
        print(f"Çekilemeyen frame'ler: {', '.join(str(m) for m in missing)}")

    sectors = snapshot.get("sectors") or {}
    if not sectors:
        print("\nSektör verisi yok.")
        return

    print(f"\n{'Sektör':<26}{'n':>4}" + "".join(f"{label:>16}" for _k, label in _PEER_TABLE_METRICS))
    print("-" * (30 + 16 * len(_PEER_TABLE_METRICS)))
    for sector in sorted(sectors):
        data = sectors[sector] or {}
        metrics = data.get("metrics") or {}
        cells = "".join(
            f"{_fmt_ratio((metrics.get(key) or {}).get('median')):>16}"
            for key, _label in _PEER_TABLE_METRICS
        )
        print(f"{str(sector)[:25]:<26}{data.get('n', 0):>4}{cells}")

    print(
        "\nNot: bu kesit bağlam katmanıdır — makul değer hesabına, üçgenlemeye "
        "veya sektör çarpanı sinyaline GİRMEZ."
    )


def cmd_peers(args: argparse.Namespace) -> None:
    """Handle the ``peers`` subcommand: build or show the peer cross-section.

    Without ``--refresh`` this only reads the stored snapshot (instant,
    offline). With ``--refresh`` it pulls the SEC XBRL Frames documents for
    the requested fiscal year and rebuilds — a dozen multi-megabyte downloads,
    which is exactly why the analyze path never does it inline.

    Never raises out to the user.
    """
    year = getattr(args, "year", None)
    if getattr(args, "refresh", False):
        if year is None:
            # Default to the last COMPLETED calendar year: the current year's
            # annual frames are still filling up as filers report, so a
            # snapshot built from them would compare a handful of early
            # filers against each other rather than a sector.
            year = date.today().year - 1
        print(f"{year} mali yılı için emsal kesiti oluşturuluyor (SEC Frames)...")
        try:
            snapshot = build_peer_snapshot(year, SecHttpClient(), no_cache=args.no_cache)
        except Exception as exc:  # noqa: BLE001 - a failed build must not crash the CLI
            logger.exception("Emsal kesiti oluşturulamadı")
            print(f"Emsal kesiti oluşturulamadı: {exc}", file=sys.stderr)
            sys.exit(1)
            return
        _print_peer_snapshot(snapshot)
        if not args.no_save:
            snapshot_id = save_peer_snapshot(snapshot, db_path=Config.DB_PATH)
            if snapshot_id:
                print(f"\nKesit kaydedildi (id {snapshot_id}): {Config.DB_PATH}")
            else:
                print("\nWARNING: emsal kesiti veritabanına kaydedilemedi.", file=sys.stderr)
        return

    snapshot = load_latest_peer_snapshot(db_path=Config.DB_PATH, year=year)
    if not snapshot:
        print(
            "Kayıtlı emsal kesiti yok. Oluşturmak için: "
            "python -m sec_analyzer.cli peers --refresh",
            file=sys.stderr,
        )
        sys.exit(1)
        return
    _print_peer_snapshot(snapshot)

    if getattr(args, "json", False):
        print("\n" + json.dumps(snapshot, indent=2, ensure_ascii=False))


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
        default=12,
        help=(
            "Number of most-recent fiscal years to retain (default: 12). "
            "Wide enough to span a full commodity cycle -- a shorter window "
            "makes through-cycle statistics depend on WHICH cycle it caught."
        ),
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
        choices=["claude_code", "ollama", "gemma", "script"],
        default=None,
        help=(
            "Analysis backend (default: resolved from LLM_BACKEND/ANALYZER_PROVIDER "
            "env). claude_code = local `claude -p` subprocess (subscription "
            "billing); ollama = local Gemma; script = deterministic rule-based "
            "(no AI). See --no-ai."
        ),
    )
    analyze_parser.add_argument(
        "--no-ai",
        action="store_true",
        dest="no_ai",
        help=(
            "Force the deterministic rule-based analyzer (no LLM). Takes "
            "priority over --provider and LLM_BACKEND."
        ),
    )
    analyze_parser.add_argument(
        "--assumptions",
        choices=["auto", "frozen", "script", "llm"],
        default="auto",
        help=(
            "Phase-1 assumption source (ASSUMPTIONS_CACHE_SPEC.md). "
            "auto (default): use the frozen set if fresh, else deterministic "
            "script. frozen: require a fresh frozen set (else stop). script: "
            "force deterministic assumptions. llm: legacy per-run LLM proposal. "
            "In as-of mode the cache is never consulted."
        ),
    )
    analyze_parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        type=_parse_as_of,
        default=None,
        dest="as_of",
        help=(
            "Point-in-time mode: analyze TICKER using only data knowable on "
            "this past date -- SEC facts filed on/before it, prices up to it, "
            "and archived ERP/risk-free macro. Analyst consensus is suppressed "
            "and financials are not written to the database."
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
        default=12,
        help=(
            "Number of most-recent fiscal years to retain (default: 12). "
            "Wide enough to span a full commodity cycle -- a shorter window "
            "makes through-cycle statistics depend on WHICH cycle it caught."
        ),
    )
    calibrate_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk raw JSON/price caches and re-fetch.",
    )
    calibrate_parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        type=_parse_as_of,
        default=None,
        dest="as_of",
        help=(
            "Point-in-time mode: run the whole basket as of this past date "
            "(e.g. --as-of 2021-11-19 --label peak2021 vs --as-of 2022-10-14 "
            "--label trough2022) to separate engine conservatism from the "
            "market regime. See sec_analyzer.calibrate."
        ),
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)

    # --- swing (S&P 500 swing-trade screener; SWING_SPEC.md Sec.8) ---
    swing_parser = subparsers.add_parser(
        "swing",
        help=(
            "Scan the S&P 500 (or a subset) for long-only swing-trade setups "
            "and print a ranked table (see sec_analyzer.screener.swing_scan)."
        ),
    )
    swing_parser.add_argument(
        "--index",
        type=str,
        default="sp500",
        help=(
            "Universe to scan: sp500 (default) or ndx (Nasdaq 100). "
            "Case-insensitive; an unrecognized value falls back to sp500."
        ),
    )
    swing_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Scan only the first N universe tickers (alphabetical), for a quick smoke run.",
    )
    swing_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk price cache and re-fetch for every ticker (including SPY).",
    )
    swing_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Thread-pool size for the per-ticker fan-out (default: {DEFAULT_MAX_WORKERS}).",
    )
    swing_parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of top-ranked rows to print (default: 25).",
    )
    swing_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the scan result to the database.",
    )
    swing_parser.set_defaults(func=cmd_swing)

    # --- overview (portfolio dashboard over stored verdicts) ---
    overview_parser = subparsers.add_parser(
        "overview",
        help=(
            "Network-free dashboard over every ticker with a stored verdict: "
            "sector heat map, fair-value gap per name, staleness, upcoming "
            "earnings, and verdict drift (see sec_analyzer.screener.overview)."
        ),
    )
    overview_parser.add_argument(
        "--stale-days",
        type=int,
        default=90,
        help="Flag a stored verdict older than this many days as stale (default: 90).",
    )
    overview_parser.add_argument(
        "--earnings-window",
        type=int,
        default=21,
        help="Surface names whose stored earnings estimate falls within this many days (default: 21).",
    )
    overview_parser.add_argument(
        "--json", action="store_true", help="Also print the full overview payload as JSON.",
    )
    overview_parser.set_defaults(func=cmd_overview)

    # --- preearnings (pre-earnings briefing for the watchlist) ---
    preearnings_parser = subparsers.add_parser(
        "preearnings",
        help=(
            "Which watchlist names report soon, and what the stored analysis "
            "says about them (see sec_analyzer.screener.preearnings)."
        ),
    )
    preearnings_group = preearnings_parser.add_mutually_exclusive_group()
    preearnings_group.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers. Defaults to every ticker with a stored verdict.",
    )
    preearnings_group.add_argument(
        "--tickers-file",
        type=str,
        default=None,
        dest="tickers_file",
        help="Path to a watchlist file (one ticker per line; '#' comments allowed).",
    )
    preearnings_parser.add_argument(
        "--within",
        type=int,
        default=DEFAULT_WITHIN_DAYS,
        help=f"Only names reporting within this many days (default: {DEFAULT_WITHIN_DAYS}).",
    )
    preearnings_parser.add_argument(
        "--all",
        action="store_true",
        help="Keep every name regardless of how far out its earnings date is (full calendar).",
    )
    preearnings_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk SEC/price caches and re-fetch.",
    )
    preearnings_parser.add_argument(
        "--json", action="store_true", help="Also print the full briefing payload as JSON.",
    )
    preearnings_parser.set_defaults(func=cmd_preearnings)

    # --- peers (cross-sectional peer snapshot from SEC XBRL Frames) ---
    peers_parser = subparsers.add_parser(
        "peers",
        help=(
            "Build or show the cross-sectional peer snapshot: per-sector "
            "medians of filing-derived metrics across the bundled index "
            "universe, from SEC's XBRL Frames API (see "
            "sec_analyzer.screener.peers). Context layer -- never feeds the "
            "fair value."
        ),
    )
    peers_parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Rebuild from SEC Frames (a dozen multi-MB downloads). Without "
            "this, the stored snapshot is shown from the database."
        ),
    )
    peers_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Fiscal year to build/show. Defaults to the last completed "
            "calendar year when refreshing, and to the most recent stored "
            "snapshot of any year otherwise."
        ),
    )
    peers_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk Frames cache and re-fetch every document.",
    )
    peers_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the rebuilt snapshot to the database.",
    )
    peers_parser.add_argument(
        "--json", action="store_true", help="Also print the full snapshot as JSON.",
    )
    peers_parser.set_defaults(func=cmd_peers)

    # --- backtest {run|evaluate|report} ---
    backtest_parser = subparsers.add_parser(
        "backtest",
        help=(
            "Evaluation (not optimization) tool: run an as-of grid, evaluate "
            "forward outcomes, and report hit-rate/calibration/divergence. "
            "See ROADMAP.md 'Backtest — tasarım ilkesi'."
        ),
    )
    backtest_sub = backtest_parser.add_subparsers(dest="backtest_action", required=True)

    bt_run = backtest_sub.add_parser(
        "run", help="Run the (ticker x date) as-of grid, persist verdicts, evaluate outcomes."
    )
    bt_run.add_argument(
        "--tickers-file", required=True,
        help="Path to a watchlist file (one ticker per line; '#' comments allowed).",
    )
    bt_run.add_argument(
        "--dates", required=True,
        help="Comma-separated as-of dates, e.g. '2020-06-30,2022-06-30,2023-12-31'.",
    )
    bt_run.add_argument("--years", type=int, default=12, help="Fiscal-year window (default 12).")
    bt_run.add_argument(
        "--no-cache", action="store_true", help="Bypass raw JSON/price caches and re-fetch.",
    )

    bt_eval = backtest_sub.add_parser(
        "evaluate", help="(Re)evaluate forward outcomes for all stored verdicts (idempotent).",
    )
    bt_eval.add_argument(
        "--no-cache", action="store_true", help="Bypass the price cache and re-fetch.",
    )

    backtest_sub.add_parser(
        "report", help="Print hit-rate/calibration/divergence tables and write an HTML report.",
    )

    bt_swing = backtest_sub.add_parser(
        "swing",
        help=(
            "Stage 1 swing backtest: score-decile forward-return study -- does a "
            "higher swing score predict a higher forward return? "
            "See sec_analyzer/backtest/SWING_STUDY_SPEC.md."
        ),
    )
    bt_swing.add_argument(
        "--index", type=str, default="sp500",
        help="Universe to study: sp500 (default) or ndx. Case-insensitive.",
    )
    bt_swing.add_argument("--start", type=str, default=None, help="ISO start date (default: earliest usable).")
    bt_swing.add_argument("--end", type=str, default=None, help="ISO end date (default: latest available bar).")
    bt_swing.add_argument(
        "--workers", type=int, default=None,
        help="Process/thread-pool size (default: min(8, cpu_count)).",
    )
    bt_swing.add_argument(
        "--no-save", action="store_true", help="Do not persist the study result to the database.",
    )

    backtest_parser.set_defaults(func=cmd_backtest)

    # --- assumptions {propose|show|edit|freeze} ---
    assumptions_parser = subparsers.add_parser(
        "assumptions",
        help=(
            "Manage the per-filing frozen phase-1 assumption set used by "
            "`analyze` (ASSUMPTIONS_CACHE_SPEC.md). propose (LLM, offline) → "
            "review/edit → freeze; analyze then reads the frozen set with no "
            "LLM call, so the fair value is reproducible."
        ),
    )
    assumptions_sub = assumptions_parser.add_subparsers(dest="assumptions_action", required=True)

    as_propose = assumptions_sub.add_parser(
        "propose", parents=[common],
        help="Propose a draft assumption set for TICKER via an LLM (the only LLM step).",
    )
    as_propose.add_argument(
        "--provider", choices=["claude_code", "ollama", "gemma", "script"], default=None,
        help="Phase-1 backend (default: resolved from LLM_BACKEND env). 'script' caches the deterministic baseline.",
    )
    as_propose.add_argument(
        "--model", default=None,
        help="Model ID/name override for the chosen provider (recorded as the set's source_model).",
    )

    assumptions_sub.add_parser(
        "show", parents=[common],
        help="Show TICKER's draft/frozen/superseded assumption sets and freshness.",
    )

    as_edit = assumptions_sub.add_parser(
        "edit", parents=[common],
        help="Edit TICKER's draft in place (re-validated/re-clamped, marked manual).",
    )
    as_edit.add_argument(
        "--set", action="append", metavar="PATH=VALUE", dest="set",
        help=(
            "Override one field; repeatable. PATH is "
            "{bear,base,bull}.{growth_5y,terminal_growth,discount_rate,story}. "
            "Rates are decimal fractions, e.g. --set base.discount_rate=0.11."
        ),
    )
    as_edit.add_argument("--note", default=None, help="Analyst note stored on the draft.")

    as_freeze = assumptions_sub.add_parser(
        "freeze", parents=[common],
        help="Freeze TICKER's draft as the active set, superseding any prior frozen set.",
    )
    as_freeze.add_argument("--note", default=None, help="Analyst note stored on the frozen set.")

    assumptions_parser.set_defaults(func=cmd_assumptions)

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
