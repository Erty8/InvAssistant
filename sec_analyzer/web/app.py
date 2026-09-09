"""Small Flask web UI for the sec_analyzer package.

Serves a single-page, vanilla-JS front end that lets a user type a stock
ticker, fetch its earnings/financials straight from SEC EDGAR, and
optionally run a fundamental analysis using a selectable backend: the local
Claude Code CLI (`claude -p`, subscription billing), a local Ollama/Gemma
model, or a deterministic script-based (no-AI) analyzer.

This module is a thin HTTP wrapper around the existing fetch/normalize/
store/interpret pipeline (see ``sec_analyzer.cli`` for the equivalent CLI
flow) -- it does not reimplement any of that logic.

Run it with::

    python -m sec_analyzer.web.app

Then open http://127.0.0.1:5050 in a browser.

Before starting the server, ``SEC_USER_AGENT`` must be set (typically via a
``.env`` file in the working directory) -- SEC EDGAR requires every request
to identify a real requester. See ``sec_analyzer.config.Config.get_user_agent``
for details. If it's missing, the API routes return a clear 400 error rather
than crashing.
"""

import logging
import re
import threading
import time
from datetime import date, datetime, timezone
from html import escape
from typing import Optional, Tuple

from flask import Flask, jsonify, request

from sec_analyzer.config import Config, ConfigError
from sec_analyzer.fetch.analyst import get_analyst_targets
from sec_analyzer.fetch.companyfacts import get_company_facts, get_submissions
from sec_analyzer.fetch.earnings import get_earnings_history
from sec_analyzer.fetch.filings import estimate_next_earnings
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
from sec_analyzer.cli import (
    _attach_macro,
    _attach_momentum,
    _attach_peers,
    _enrich_sector_momentum,
    _fetch_insider_activity,
    _fetch_risk_free_asof,
)
from sec_analyzer.interpret.analyzer import interpret
from sec_analyzer.normalize.metrics import compute_metrics
from sec_analyzer.normalize.normalizer import normalize_facts
from sec_analyzer.normalize.ratios import compute_ratios
from sec_analyzer.normalize.red_flags import detect_red_flags
from sec_analyzer.report.financials import serialize_financials
from sec_analyzer.report.generator import (
    render_history_page,
    render_overview_page,
    render_report_html,
    render_search_page,
    render_swing_page,
)
from sec_analyzer.screener.overview import build_overview
from sec_analyzer.screener.swing_scan import DEFAULT_MAX_WORKERS, scan_swing
from sec_analyzer.screener.universe import UNIVERSES, load_universe, normalize_index, universe_label
from sec_analyzer.store.database import (
    load_latest_stored_price,
    load_latest_swing_scan,
    load_latest_verdicts,
    load_verdicts,
    save_normalized,
    save_prices,
    save_swing_scan,
    save_verdict,
)
from sec_analyzer.technical.indicators import compute_indicators, relative_strength
from sec_analyzer.technical.verdict import technical_verdict

logger = logging.getLogger(__name__)

#: Selectable investment horizons shown in the UI, as (value, label) pairs.
_HORIZONS = [
    ("3m", "3 months"),
    ("1y", "1 year"),
    ("5y", "5 years"),
]

app = Flask(__name__)

#: Selectable analysis providers shown in the UI, as (value, label) pairs.
_PROVIDERS = [
    ("script", "Script (no AI · deterministic)"),
    ("ollama", "Gemma (local · Ollama)"),
    ("claude_code", "Claude Code (subscription · claude -p)"),
]


#: The financials-serialization contract (payload shape + concept ordering)
#: now lives in :mod:`sec_analyzer.report.financials` so the CLI's
#: ``analyze --html`` report and this web UI share one source of truth. Kept
#: as a module-level alias here so the existing call sites (and the tests that
#: patch/inspect them) read unchanged.
_serialize_financials = serialize_financials


#: Guards `_swing_state` so exactly one swing-trade screener scan can run at
#: a time, regardless of index (SWING_SPEC.md Sec.6 -- the bottleneck is CPU,
#: so scans across different indexes still serialize). A cold-cache
#: full-universe scan takes minutes, so it runs in a background daemon
#: thread; this lock + dict is the single source of truth both the
#: scan-starting route and the progress-polling route read/write. `index`/
#: `index_label` name the index the running (or last-completed) scan
#: belongs to, so the UI can tell whether the progress it sees belongs to
#: the index it is displaying.
_swing_lock = threading.Lock()
_swing_state = {
    "running": False,
    "done": 0,
    "total": 0,
    "index": None,
    "index_label": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


#: Ordered stage table for the `POST /api/analyze` progress feature (see
#: `_report_stage` below and `GET /api/analyze/progress`). Keys are English
#: snake_case identifiers used internally and in the polling route's JSON
#: contract; labels are the English text shown to the user while that stage
#: is the active one. Order matters -- it is also the order the pipeline
#: actually executes in, which is what lets a stage's index double as a
#: "done"/"active"/"pending" cursor.
_ANALYZE_STAGES = [
    ("resolve", "Resolving company identity (CIK)"),
    ("facts", "Downloading SEC financial data"),
    ("normalize", "Normalizing financials, computing ratios"),
    ("price", "Fetching price history and technical indicators"),
    ("filings", "Computing SEC filing history and catalyst calendar"),
    ("macro", "Fetching macro data (risk-free rate)"),
    ("market", "Fetching analyst consensus and earnings history"),
    ("valuation", "Running valuation engine (DCF, multiples, triangulation)"),
    ("context", "Adding momentum, insider activity, and peer companies"),
    ("save", "Saving result"),
]
_ANALYZE_STAGE_LABELS = dict(_ANALYZE_STAGES)
_ANALYZE_STAGE_INDEX = {key: i for i, (key, _label) in enumerate(_ANALYZE_STAGES)}


#: Guards `_analyze_progress`, the per-job progress-tracking dict for
#: `POST /api/analyze`. Unlike `_swing_state` (one shared scan at a time),
#: analyze requests run concurrently -- Flask's dev server is threaded by
#: default (see `main`) -- so progress is keyed by a client-supplied job id
#: rather than a single global record. Each record: `stage` (current stage
#: key), `stage_index` (int cursor into `_ANALYZE_STAGES`), `started_at`/
#: `updated_at` (`time.monotonic()` floats -- UI telemetry only, never fed
#: into any analysis/valuation result), `done` (bool), `error`
#: (Optional[str]), `ticker` (str, diagnostics only).
_analyze_lock = threading.Lock()
_analyze_progress: dict = {}

#: Per-thread binding of "the job id this request is tracking", set by
#: `_progress_begin` and cleared by `_progress_finish`. Each Flask request
#: runs on its own thread, so this is how `_report_stage` -- called from deep
#: inside `_run_pipeline`/`_run_full_pipeline`, which must keep their exact
#: existing signatures for `/api/financials` and the tests that monkeypatch
#: them -- finds "the job for this request" without threading a job id
#: through every pipeline helper.
_analyze_job_local = threading.local()

#: Bounds for `_analyze_progress` pruning (see `_prune_analyze_progress`): a
#: long-lived server process must not accumulate one record per analyze
#: request forever just because a client abandoned a job without polling it
#: to completion. Age and count are independent caps -- age catches jobs left
#: around after their client stopped polling, count bounds memory even if
#: every job is recent but numerous.
_ANALYZE_PROGRESS_MAX_AGE_SECONDS = 15 * 60
_ANALYZE_PROGRESS_MAX_RECORDS = 20

#: Format for the client-supplied `job` id (`POST /api/analyze`'s JSON field
#: and `GET /api/analyze/progress`'s query param). Validated defensively
#: since it's used as a dict key and echoed back in a JSON response.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _prune_analyze_progress() -> None:
    """Drop stale and/or excess records from `_analyze_progress`.

    Must be called with `_analyze_lock` already held. Called once per new
    job registration (from `_progress_begin`), not on every `_report_stage`
    call -- this is O(n) bookkeeping, not something that belongs on the hot
    path of a request already fetching SEC/price/market data.
    """
    now = time.monotonic()
    stale_ids = [
        job_id for job_id, record in _analyze_progress.items()
        if now - record["updated_at"] > _ANALYZE_PROGRESS_MAX_AGE_SECONDS
    ]
    for job_id in stale_ids:
        del _analyze_progress[job_id]

    if len(_analyze_progress) >= _ANALYZE_PROGRESS_MAX_RECORDS:
        # Evict the oldest-updated records first, keeping just enough room
        # (`_ANALYZE_PROGRESS_MAX_RECORDS - 1`) for the new record the caller
        # is about to insert to fit under the cap.
        oldest_first = sorted(_analyze_progress.items(), key=lambda kv: kv[1]["updated_at"])
        overflow = len(oldest_first) - _ANALYZE_PROGRESS_MAX_RECORDS + 1
        for job_id, _record in oldest_first[:overflow]:
            del _analyze_progress[job_id]


def _progress_begin(job_id: str, ticker: str) -> None:
    """Register a fresh progress record for `job_id` and bind it to this thread.

    Called once, at the start of `api_analyze`, only when the request supplied
    a `job` id that passed `_JOB_ID_RE` -- requests that don't opt in never
    touch `_analyze_progress` at all. `ticker` is stored for diagnostics only;
    the polling route's response does not echo it.
    """
    with _analyze_lock:
        _prune_analyze_progress()
        now = time.monotonic()
        _analyze_progress[job_id] = {
            "stage": _ANALYZE_STAGES[0][0],
            "stage_index": 0,
            "started_at": now,
            "updated_at": now,
            "done": False,
            "error": None,
            "ticker": ticker,
        }
    _analyze_job_local.job_id = job_id


def _report_stage(key: str) -> None:
    """Record that the current thread's tracked job has entered stage `key`.

    Called from true stage boundaries inside `_run_pipeline`,
    `_run_full_pipeline`, and `api_analyze` (right before each stage's work
    starts). Deliberately a silent no-op -- never raises, per this file's
    "never let analysis-layer code crash" convention -- whenever there is
    nothing to report to: no job bound to this thread (e.g. `/api/financials`
    calling `_run_pipeline` directly, or an `/api/analyze` request with no/
    invalid `job`), the bound job already pruned from `_analyze_progress`, or
    an unrecognized `key`. This is what lets `_run_pipeline`/
    `_run_full_pipeline` call it unconditionally without changing behavior
    for callers that never asked for progress tracking.
    """
    try:
        job_id = getattr(_analyze_job_local, "job_id", None)
        if job_id is None or key not in _ANALYZE_STAGE_INDEX:
            return
        with _analyze_lock:
            record = _analyze_progress.get(job_id)
            if record is None:
                return
            record["stage"] = key
            record["stage_index"] = _ANALYZE_STAGE_INDEX[key]
            record["updated_at"] = time.monotonic()
    except Exception:  # noqa: BLE001 - progress reporting must never break the pipeline
        logger.warning("Failed to report analyze progress stage %r", key, exc_info=True)


def _progress_finish(error: Optional[str] = None) -> None:
    """Mark the current thread's tracked job done and unbind it from the thread.

    Called from `api_analyze`'s `finally` clause, so it always runs on every
    exit path (success, 400/404/500, or an unexpected exception) once a job
    has been registered. A silent no-op -- never raises -- when no job is
    bound to this thread (the common case: most requests don't pass `job`).
    """
    try:
        job_id = getattr(_analyze_job_local, "job_id", None)
        if job_id is not None:
            with _analyze_lock:
                record = _analyze_progress.get(job_id)
                if record is not None:
                    record["done"] = True
                    record["updated_at"] = time.monotonic()
                    if error is not None:
                        record["error"] = error
    except Exception:  # noqa: BLE001 - progress reporting must never break the pipeline
        logger.warning("Failed to finalize analyze progress", exc_info=True)
    finally:
        _analyze_job_local.job_id = None


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601-seconds string, e.g. "2026-07-27T09:12:03Z".

    Matches the format `scan_swing`'s own `generated_at` uses -- the only
    wall-clock value anywhere in the swing-screener feature (see
    SWING_SPEC.md Sec.4), used here purely for scan-progress bookkeeping.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _swing_progress_cb(done: int, total: int, ticker: str) -> None:
    """Update the shared swing-scan progress state after each ticker.

    Called from the scan's background worker thread; guarded by
    `_swing_lock` since `GET /api/swing/status` reads the same dict
    concurrently from Flask request threads.
    """
    with _swing_lock:
        _swing_state["done"] = done
        _swing_state["total"] = total


def _run_swing_scan_worker(
    tickers: Optional[list], no_cache: bool, max_workers: int, index: str,
) -> None:
    """Background-thread body for `POST /api/swing/scan`.

    Runs the full scan for `index` (already normalized by the caller) and
    persists it via `save_swing_scan` on success. The `finally` clause always
    clears `_swing_state["running"]`, so a scan that raises (it shouldn't --
    `scan_swing` itself never raises per its own contract -- but this is the
    last line of defense) can never wedge the "exactly one scan at a time"
    guard shut forever.
    """
    try:
        result = scan_swing(
            tickers=tickers, no_cache=no_cache, max_workers=max_workers,
            progress_cb=_swing_progress_cb, index=index,
        )
        save_swing_scan(result, db_path=Config.DB_PATH)
    except Exception as exc:  # noqa: BLE001 - never let a scan failure crash a daemon thread silently
        logger.exception("Swing scan failed")
        with _swing_lock:
            _swing_state["error"] = str(exc)
    finally:
        with _swing_lock:
            _swing_state["running"] = False
            _swing_state["finished_at"] = _utc_now_iso()


def _run_pipeline(ticker: str, years: int, no_cache: bool, as_of=None) -> Tuple[str, str, dict, list]:
    """Resolve, fetch, normalize, compute ratios for, and persist a ticker.

    Shared by both API routes so the fetch/normalize/store logic (and its
    error behavior) stays identical whether or not the caller goes on to
    request an LLM analysis.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        years: Number of most-recent fiscal years to retain.
        no_cache: When True, bypass the on-disk raw JSON cache.

    Returns:
        ``(cik, name, normalized, ratios)``.

    Raises:
        ConfigError: If required configuration (e.g. SEC_USER_AGENT) is missing.
        ValueError: If the ticker cannot be resolved to a CIK.
        Exception: Any other failure (network errors, etc.) propagates as-is
            for the caller to log and translate into a 500 response.
    """
    client = SecHttpClient()

    _report_stage("resolve")
    cik, name = resolve_cik(ticker, client, no_cache=no_cache)
    _report_stage("facts")
    facts = get_company_facts(cik, client, no_cache=no_cache)
    _report_stage("normalize")
    normalized = normalize_facts(facts, years=years, as_of=as_of)
    ratios = compute_ratios(normalized)

    # In as-of mode the normalized slice is a truncated historical view; the
    # financials table holds the current-view upsert, so don't overwrite it
    # (mirrors sec_analyzer.cli._fetch_normalize_store).
    if as_of is None:
        save_normalized(ticker, cik, name, normalized, ratios, db_path=Config.DB_PATH)

    return cik, name, normalized, ratios


def _fetch_price_and_technical(ticker: str, horizon: str, no_cache: bool, as_of=None):
    """Fetch price history and derive the merged technical indicators/verdict.

    Mirrors ``sec_analyzer.cli._fetch_price_and_technical`` so the web UI's
    ``/api/analyze`` pipeline behaves identically to the CLI's ``analyze``
    command. Fully graceful: if price data can't be obtained, this logs a
    warning and returns ``(None, None, None, None)`` rather than raising --
    the fundamental side of the pipeline must keep working even with no
    usable price data at all.

    Returns:
        ``(price, as_of, technical, price_df)``, where ``technical`` is the
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
                    "No price data for %s on/before as-of %s; skipping technical.", ticker, as_of
                )
                return None, None, None, None
        price, price_as_of = latest_price(price_df)
        indicators = compute_indicators(price_df)
        verdict_result = technical_verdict(indicators, horizon)
        technical = {**indicators, **verdict_result}
        technical["relative_strength"] = _fetch_relative_strength(ticker, price_df, no_cache, as_of)
        # Provenance of the price series, so the total-return guard and the
        # report's source label see the real source (mirrors the CLI).
        technical["price_source"] = source
        logger.info("Price data for %s from %s: %.2f as of %s", ticker, source, price, price_as_of)
        return price, price_as_of, technical, price_df
    except PriceDataError as exc:
        logger.warning("Price data unavailable for %s: %s", ticker, exc)
        return None, None, None, None


#: Benchmark ticker for the relative-strength (RS) cross-check (mirrors
#: ``sec_analyzer.cli._RS_BENCHMARK``).
_RS_BENCHMARK = "SPY"


def _fetch_relative_strength(ticker: str, price_df, no_cache: bool, as_of=None) -> Optional[dict]:
    """Best-effort price relative strength vs. :data:`_RS_BENCHMARK`; never
    raises. Mirrors ``sec_analyzer.cli._fetch_relative_strength`` so the web UI
    matches the CLI. ``None`` when the ticker is the benchmark or anything
    fails (display-only cross-check). When ``as_of`` is set the benchmark frame
    is sliced to the same cutoff so the comparison stays point-in-time."""
    if str(ticker).strip().upper() == _RS_BENCHMARK:
        return None
    try:
        # Matches the stock's own fetch: comparing a live bar against a
        # settled benchmark bar would skew relative strength by a session.
        bench_df, _ = get_price_history(
            _RS_BENCHMARK, no_cache=no_cache, prefer_live=as_of is None
        )
        if as_of is not None:
            bench_df = slice_asof(bench_df, as_of)
        return relative_strength(price_df["Close"], bench_df["Close"], benchmark=_RS_BENCHMARK)
    except Exception:  # noqa: BLE001 - display-only cross-check, never fatal
        logger.warning("Could not compute relative strength for %s", ticker, exc_info=True)
        return None


def _fetch_submissions(cik: str, ticker: str, no_cache: bool) -> Optional[dict]:
    """Best-effort fetch of a filer's raw SEC submissions document; never raises.

    Mirrors ``sec_analyzer.cli._fetch_submissions``: fetched exactly once per
    request (SPEC.md Sec.13) and reused for both the next-earnings catalyst
    estimate (:func:`_fetch_catalyst`) and SIC-based sector classification
    (passed straight through to
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
    """Best-effort next-earnings estimate from already-fetched submissions;
    never raises (see the CLI's equivalent helper for the rationale).

    Args:
        submissions: The dict returned by :func:`_fetch_submissions`, or
            ``None``.
        ticker: Stock ticker symbol, used only for the warning log message.
        as_of: Optional point-in-time reference date; forwarded as
            ``estimate_next_earnings(today=as_of)``.
    """
    if not submissions:
        return None
    try:
        return estimate_next_earnings(submissions, today=as_of)
    except Exception:  # noqa: BLE001 - a catalyst estimate is a nice-to-have, never fatal
        logger.warning("Could not estimate next earnings date for %s", ticker, exc_info=True)
        return None


def _attach_insider(analysis: dict, cik: str, ticker: str, submissions, no_cache: bool, as_of) -> None:
    """Attach the SEC Form 4 insider-activity signal to ``analysis``.

    Mirrors the CLI's post-``interpret`` attachment of ``events``/``momentum``:
    deterministic filing-derived context, never routed through the LLM and
    never an input to the fair value. Best-effort -- the shared helper it
    delegates to already swallows every failure -- so a missing signal simply
    leaves ``analysis["insider"]`` absent.
    """
    insider = _fetch_insider_activity(cik, ticker, submissions, no_cache, as_of)
    if insider is not None:
        analysis["insider"] = insider


def _fetch_analyst_targets(ticker: str, no_cache: bool) -> Optional[dict]:
    """Best-effort fetch of consensus analyst price targets; never raises.

    Mirrors ``sec_analyzer.cli._fetch_analyst_targets``. Display-only
    cross-check (see ``sec_analyzer.fetch.analyst``) -- never feeds the
    valuation engine, and a failure here must never fail the request.

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

    Mirrors :func:`_fetch_analyst_targets`. Display-only cross-check (see
    ``sec_analyzer.fetch.earnings``) -- never feeds the valuation engine, and a
    failure here must never fail the request.

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


def _save_price_rows(cik: str, price_df) -> None:
    """Convert a price-history DataFrame to row dicts and persist them.

    Never raises: a failure to persist price history must not prevent the
    rest of the request (interpretation, JSON response) from completing.

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


def _run_full_pipeline(
    ticker: str, years: int, no_cache: bool, horizon: str, as_of=None
) -> Tuple[
    str, str, dict, list, dict, Optional[dict], list, Optional[dict], Optional[float],
    Optional[dict], object, Optional[dict],
]:
    """Extend ``_run_pipeline`` with the price/technical/metrics/red-flags/
    submissions/catalyst steps used by both the CLI's ``analyze`` command and
    this module's ``/api/analyze``/``/report`` routes, so all three stay in
    sync. Mirrors ``sec_analyzer.cli.cmd_analyze``: SEC submissions are
    fetched exactly once and reused for both the catalyst estimate and (by
    the caller, via the returned value) SIC-based sector classification.

    Note: the display-only consensus analyst-target cross-check (see
    ``sec_analyzer.fetch.analyst``) is deliberately NOT part of this tuple --
    it's fetched separately (:func:`_fetch_analyst_targets`) by each route,
    so this shared helper's return shape (and the tests that stub it) stay
    unchanged.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        years: Number of most-recent fiscal years to retain.
        no_cache: When True, bypass the on-disk raw JSON cache.
        horizon: Investment horizon ("3m", "1y", or "5y") used to weight and
            frame the technical verdict and red-flag commentary.

    Returns:
        ``(cik, name, normalized, ratios, metrics, technical, flags,
        catalyst, price, submissions, price_df, fred_rate)``. ``submissions``,
        ``price_df``, and (in as-of mode) ``fred_rate`` are meant to be
        threaded straight into
        :func:`sec_analyzer.interpret.analyzer.interpret` as
        ``submissions=``/``price_df=``/``fred_rate=`` so the web UI gets the
        same full deterministic valuation the CLI does. ``fred_rate`` carries
        the latest DGS10 observation on a live run and the as-of one on a
        historical run; it is ``None`` only when FRED itself is unreachable,
        in which case the risk-free rate degrades to the archived
        ``erp.csv`` value.

    Raises:
        Same as ``_run_pipeline`` -- financials fetch/normalize/store
        failures propagate for the caller to translate into an HTTP error.
        Everything past that point (price, technical, submissions, catalyst)
        is best-effort and never raises.
    """
    cik, name, normalized, ratios = _run_pipeline(ticker, years, no_cache, as_of)

    _report_stage("price")
    price, _price_as_of, technical, price_df = _fetch_price_and_technical(
        ticker, horizon, no_cache, as_of
    )

    metrics = compute_metrics(normalized, ratios, price)
    flags = detect_red_flags(normalized, ratios, metrics, horizon)
    _report_stage("filings")
    submissions = _fetch_submissions(cik, ticker, no_cache)
    catalyst = _fetch_catalyst(submissions, ticker, as_of)
    # Live runs take the latest DGS10 observation, as-of runs the one on/before
    # the cutoff -- mirrors the CLI's `_fetch_risk_free_asof` (see its docstring
    # for why a live run must not fall through to the archived erp.csv value).
    _report_stage("macro")
    fred_rate = _fetch_risk_free_asof(as_of, no_cache)
    # Fold sector-relative strength into `technical` and recompute the composite
    # momentum score now that the SIC is known (mirrors the CLI).
    _enrich_sector_momentum(ticker, technical, price_df, submissions, no_cache, as_of)

    # A degraded-source frame (split-adjusted only) must not overwrite rows in
    # a table read as a total-return series, nor reach the valuation layer.
    # Technicals keep the full frame -- see prices.is_total_return_basis.
    # `price_df` is returned as None in that case so every downstream consumer
    # of this tuple inherits the guard (mirrors the CLI).
    price_source = (technical or {}).get("price_source")
    if price_df is not None and not is_total_return_basis(price_source):
        logger.warning(
            "Price source %r is not a total-return series for %s; historical multiples "
            "and price persistence are skipped. Technicals are unaffected.",
            price_source, ticker,
        )
        price_df = None

    # In as-of mode the sliced price frame is a historical subset; don't
    # persist it over the current-view prices table (mirrors the CLI).
    if price_df is not None and as_of is None:
        _save_price_rows(cik, price_df)

    return (
        cik, name, normalized, ratios, metrics, technical, flags, catalyst, price,
        submissions, price_df, fred_rate,
    )


def _bool_param(value: Optional[str]) -> bool:
    """Parse a query-string boolean parameter (``"true"``/``"false"``, etc.)."""
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_as_of_param(value) -> Tuple[Optional[date], Optional[str]]:
    """Validate an optional as-of date value from a request.

    Returns ``(parsed_date, None)`` on success (``(None, None)`` when the
    value is blank/absent), or ``(None, error_message)`` when the value is not
    a valid past ISO date -- the caller turns the message into a 400.
    """
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None, f"invalid as_of {text!r}; expected YYYY-MM-DD."
    if parsed > date.today():
        return None, f"as_of {text} is in the future; expected a past date."
    return parsed, None


@app.route("/")
def index():
    """Render the interactive "Verdict Terminal" search page.

    Serves the same self-contained ``template.html`` shell used by
    ``GET /report`` (see ``sec_analyzer.report.generator``), but with a
    ``mode: "search"`` payload: the template's client-side script renders a
    live ticker/horizon/provider search box instead of a baked result, and
    POSTs to ``/api/analyze`` on submit.
    """
    return render_search_page(
        _HORIZONS, _PROVIDERS, "1y", Config.ANALYZER_PROVIDER, Config.OLLAMA_MODEL
    )


@app.route("/history", methods=["GET"])
def history():
    """Render the verdict-history screen for a ticker.

    Reads the append-only ``verdicts`` table (via
    :func:`sec_analyzer.store.database.load_verdicts`) and the latest stored
    price (:func:`sec_analyzer.store.database.load_latest_stored_price`) and
    renders them through the shared ``template.html`` shell in
    ``mode: "history"`` -- no network, no analysis, just stored data.

    Query params:
        ticker: Stock ticker symbol (required).
    """
    ticker = (request.args.get("ticker") or "").strip()
    if not ticker:
        return _error_page("Query parameter 'ticker' is required."), 400
    try:
        rows = load_verdicts(ticker, db_path=Config.DB_PATH)
        current_price = load_latest_stored_price(ticker, db_path=Config.DB_PATH)
        return render_history_page(ticker, rows, current_price=current_price)
    except Exception:  # noqa: BLE001 - last-resort guard, render a page not a stack trace
        logger.exception("Unexpected error rendering history for %s", ticker)
        return _error_page("An unexpected error occurred while loading the analysis history."), 500


#: Bounds for the two ``/overview`` tuning knobs, so a hand-edited query
#: string can't ask for a nonsensical window (a negative staleness threshold
#: would mark every row stale; an unbounded one would mark none).
_OVERVIEW_STALE_DAYS_DEFAULT = 90
_OVERVIEW_EARNINGS_WINDOW_DEFAULT = 21
_OVERVIEW_MAX_DAYS = 3650


def _int_param(value, default: int, minimum: int = 1, maximum: int = _OVERVIEW_MAX_DAYS) -> int:
    """Parse an optional integer query parameter, clamped into a sane range.

    An absent, blank, or unparseable value falls back to ``default`` rather
    than erroring: these are display-tuning knobs on a dashboard, not inputs
    to a computation, so a bad value should degrade to the default view.
    """
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


@app.route("/overview", methods=["GET"])
def overview():
    """Render the portfolio-overview dashboard over every stored verdict.

    Network-free and computation-free: reads one row per ticker (that
    ticker's most recent live verdict) via
    :func:`sec_analyzer.store.database.load_latest_verdicts`, enriches and
    groups them with :func:`sec_analyzer.screener.overview.build_overview`,
    and renders the result through the shared ``template.html`` shell in
    ``mode: "overview"``. No valuation is re-run, so every number shown is
    exactly what that name's last analysis produced.

    Query params:
        stale_days: Flag a verdict older than this many days as stale
            (default 90).
        earnings_window: Surface names whose stored earnings estimate falls
            within this many days (default 21).
    """
    stale_days = _int_param(request.args.get("stale_days"), _OVERVIEW_STALE_DAYS_DEFAULT)
    earnings_window = _int_param(
        request.args.get("earnings_window"), _OVERVIEW_EARNINGS_WINDOW_DEFAULT
    )
    try:
        rows = load_latest_verdicts(db_path=Config.DB_PATH)
        payload = build_overview(
            rows, stale_days=stale_days, earnings_window_days=earnings_window
        )
        return render_overview_page(payload)
    except Exception:  # noqa: BLE001 - last-resort guard, render a page not a stack trace
        logger.exception("Unexpected error rendering portfolio overview")
        return _error_page("An unexpected error occurred while loading the portfolio overview."), 500


@app.route("/api/overview", methods=["GET"])
def api_overview():
    """JSON counterpart to :func:`overview` -- same payload, no HTML shell."""
    stale_days = _int_param(request.args.get("stale_days"), _OVERVIEW_STALE_DAYS_DEFAULT)
    earnings_window = _int_param(
        request.args.get("earnings_window"), _OVERVIEW_EARNINGS_WINDOW_DEFAULT
    )
    try:
        rows = load_latest_verdicts(db_path=Config.DB_PATH)
        payload = build_overview(
            rows, stale_days=stale_days, earnings_window_days=earnings_window
        )
        return jsonify({"ok": True, "overview": payload})
    except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
        logger.exception("Unexpected error building portfolio overview")
        return jsonify(
            {"ok": False, "error": "An unexpected error occurred while building the portfolio overview."}
        ), 500


@app.route("/swing", methods=["GET"])
def swing():
    """Render the swing-trade screener page for one index.

    Query params:
        index: Universe code (default ``SP500``), resolved through
            :func:`sec_analyzer.screener.universe.normalize_index` so a
            missing/bogus value degrades to the S&P 500 rather than erroring.

    Loads the most recently persisted scan for that index (via
    :func:`sec_analyzer.store.database.load_latest_swing_scan`, or ``None``
    if that index has never been scanned) and renders it through the shared
    ``template.html`` shell in ``mode: "swing"``, alongside the full index
    selector built from :data:`sec_analyzer.screener.universe.UNIVERSES`
    (code + label only -- the CSV paths never reach the client). No scan
    computation happens on this route -- scans are started via
    ``POST /api/swing/scan`` and this page's client-side script polls their
    progress and re-renders in place (see SWING_SPEC.md Sec.6-7).
    """
    index = normalize_index(request.args.get("index"))
    try:
        scan = load_latest_swing_scan(db_path=Config.DB_PATH, universe=index)
        indexes = [(code, label) for code, (label, _path) in UNIVERSES.items()]
        return render_swing_page(scan, index=index, indexes=indexes)
    except Exception:  # noqa: BLE001 - last-resort guard, render a page not a stack trace
        logger.exception("Unexpected error rendering swing screener page")
        return _error_page("An unexpected error occurred while loading the swing screener page."), 500


@app.route("/api/financials", methods=["GET"])
def api_financials():
    """Fetch, normalize, store, and return a ticker's SEC financials.

    Query params:
        ticker: Stock ticker symbol (required).
        years: Number of most-recent fiscal years to retain (default 12).
        no_cache: "true"/"false" -- bypass the on-disk raw JSON cache.
    """
    ticker = (request.args.get("ticker") or "").strip()
    if not ticker:
        return jsonify({"ok": False, "error": "Query parameter 'ticker' is required."}), 400

    try:
        years = int(request.args.get("years", 12))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "'years' must be an integer."}), 400

    no_cache = _bool_param(request.args.get("no_cache"))

    try:
        cik, name, normalized, ratios = _run_pipeline(ticker, years, no_cache)
        payload = _serialize_financials(normalized, ratios)
        return jsonify({"ok": True, **payload, "cik": cik, "name": name})

    except ConfigError as exc:
        logger.error("Configuration error while fetching %s: %s", ticker, exc)
        return jsonify({"ok": False, "error": str(exc)}), 400

    except ValueError as exc:
        logger.info("Ticker resolution failed for %s: %s", ticker, exc)
        return jsonify({"ok": False, "error": str(exc)}), 404

    except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
        logger.exception("Unexpected error fetching financials for %s", ticker)
        return jsonify(
            {"ok": False, "error": "An unexpected server error occurred while fetching financials."}
        ), 500


#: Valid ``horizon`` values accepted by ``/api/analyze``; anything else
#: (missing, malformed, or unrecognized) falls back to "1y".
_VALID_HORIZONS = ("3m", "1y", "5y")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Fetch/normalize/store a ticker's financials, then run a full
    fundamental + technical analysis.

    JSON body:
        ticker: Stock ticker symbol (required).
        years: Number of most-recent fiscal years to retain (default 5).
        horizon: Investment horizon -- "3m", "1y", or "5y" (default "1y").
            Controls the fundamental/technical weighting and the framing of
            the verdict; see ``Config.HORIZON_WEIGHTS``.
        provider: Analysis provider to use ("claude_code", "ollama", or
            "script"); defaults to ``Config.ANALYZER_PROVIDER`` when
            omitted/None.
        no_cache: Bypass the on-disk raw JSON cache (default False).
        job: Optional client-generated progress-tracking id, matching
            ``^[A-Za-z0-9_-]{1,64}$``. When present and valid, this request's
            stage-by-stage progress is recorded under that id and can be
            polled via ``GET /api/analyze/progress``. Missing or malformed
            values are silently treated as "no progress tracking requested"
            -- this is purely additive UI plumbing, so it must never change
            this route's behavior or response shape.

    The financials pipeline uses the same error handling as
    ``/api/financials``. Price/technical data is fetched best-effort -- if
    it's unavailable, the fundamental analysis still runs (with
    ``technical: null`` in the response) rather than failing the whole
    request. SEC submissions are also fetched once (best-effort) and passed
    into ``interpret(...)`` as ``submissions=``/``price_df=``, exactly like
    the CLI's ``analyze`` command, so ``analysis`` carries the full
    deterministic valuation engine output (``analysis["valuation"]``,
    ``analysis["confidence"]``, ``analysis["reverse_dcf_comment"]``) rather
    than a degraded fallback. Once financials succeed, ``interpret(...)`` is
    called and its result -- success or error dict -- is passed through
    under the ``analysis`` key with a 200 status, since a failed/degraded
    analysis is not itself a request failure.

    Response gains, alongside the usual financials payload and ``analysis``:
        technical: merged indicators + technical-verdict dict, or ``null``.
        metrics: valuation/quality metrics dict (see
            ``sec_analyzer.normalize.metrics.compute_metrics``).
        red_flags: list of ``{"code", "message", "detail"}`` dicts.
        catalyst: ``{"estimate_date", "label", "based_on"}`` dict, or ``null``.
        analyst: display-only consensus analyst-target dict (see
            ``sec_analyzer.fetch.analyst.get_analyst_targets``), or ``null``.
            Best-effort, never feeds ``interpret(...)`` or the valuation
            engine -- shown purely as a reference cross-check.
    """
    body = request.get_json(silent=True) or {}

    ticker = str(body.get("ticker") or "").strip()
    if not ticker:
        return jsonify({"ok": False, "error": "JSON field 'ticker' is required."}), 400

    # `job` is validated up front, same as every other body field, but
    # registration happens only after we know `ticker` (used purely for the
    # record's diagnostic `ticker` field -- `_run_pipeline` resolves the
    # canonical name/CIK later, which is exactly why progress is keyed by a
    # caller-supplied job id rather than the resolved identity). An invalid
    # or absent id just means this request opts out of progress tracking, so
    # `job_id` stays None and every `_report_stage`/`_progress_finish` call
    # below is already a documented no-op in that case.
    raw_job = body.get("job")
    job_id = str(raw_job).strip() if isinstance(raw_job, str) and raw_job.strip() else None
    if job_id is not None and not _JOB_ID_RE.match(job_id):
        job_id = None
    if job_id is not None:
        _progress_begin(job_id, ticker)

    # Everything past this point is wrapped so `_progress_finish()` always
    # runs -- on the 400s below, on the 404/500s around `_run_full_pipeline`,
    # and on the final 200 -- even though `_progress_finish` itself is a
    # no-op when `job_id` is None. `progress_error` is set on the paths that
    # represent an actual request failure; a degraded/error `analysis` dict
    # on an otherwise-successful 200 is deliberately NOT treated as a
    # progress-tracking failure (see the route's docstring: that's not a
    # request failure either).
    progress_error: Optional[str] = None
    try:
        try:
            years = int(body.get("years", 12))
        except (TypeError, ValueError):
            progress_error = "'years' must be an integer."
            return jsonify({"ok": False, "error": progress_error}), 400

        horizon = str(body.get("horizon") or "1y").strip().lower()
        if horizon not in _VALID_HORIZONS:
            horizon = "1y"

        provider = body.get("provider") or None
        no_cache = bool(body.get("no_cache", False))

        as_of, as_of_error = _parse_as_of_param(body.get("as_of"))
        if as_of_error:
            progress_error = as_of_error
            return jsonify({"ok": False, "error": as_of_error}), 400

        try:
            (
                cik, name, normalized, ratios, metrics, technical, flags, catalyst, price,
                submissions, price_df, fred_rate,
            ) = _run_full_pipeline(ticker, years, no_cache, horizon, as_of)

        except ConfigError as exc:
            logger.error("Configuration error while analyzing %s: %s", ticker, exc)
            progress_error = str(exc)
            return jsonify({"ok": False, "error": progress_error}), 400

        except ValueError as exc:
            logger.info("Ticker resolution failed for %s: %s", ticker, exc)
            progress_error = str(exc)
            return jsonify({"ok": False, "error": progress_error}), 404

        except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
            logger.exception("Unexpected error fetching financials for %s", ticker)
            progress_error = "An unexpected server error occurred while fetching financials."
            return jsonify({"ok": False, "error": progress_error}), 500

        # Analyst consensus (yfinance) is undated and cannot be point-in-time, so
        # it is suppressed in as-of mode (mirrors the CLI's as-of contract). The
        # earnings-surprise (beat/miss) history is likewise undated and display-only,
        # so it is suppressed in as-of mode for the same reason.
        _report_stage("market")
        analyst = None if as_of is not None else _fetch_analyst_targets(ticker, no_cache)
        earnings = None if as_of is not None else _fetch_earnings_history(ticker, no_cache)

        logger.info(
            "Running %s analysis for %s (horizon=%s%s)",
            provider or Config.ANALYZER_PROVIDER, ticker, horizon,
            f", as_of={as_of.isoformat()}" if as_of is not None else "",
        )
        _report_stage("valuation")
        analysis = interpret(
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
            as_of=as_of,
            fred_rate=fred_rate,
        )

        if isinstance(analysis, dict) and as_of is not None:
            analysis["as_of"] = as_of.isoformat()

        if isinstance(analysis, dict) and "error" not in analysis:
            _report_stage("context")
            _attach_momentum(analysis, ticker, normalized, technical, as_of)
            _attach_insider(analysis, cik, ticker, submissions, no_cache, as_of)
            _attach_macro(analysis, as_of, no_cache)
            _attach_peers(
                analysis, ticker, cik, submissions, normalized, ratios, metrics, as_of=as_of
            )
            try:
                _report_stage("save")
                save_verdict(
                    ticker, cik, horizon, provider or Config.ANALYZER_PROVIDER, price, analysis,
                    db_path=Config.DB_PATH, valuation=analysis.get("valuation"),
                    as_of=as_of.isoformat() if as_of is not None else None,
                    catalyst_date=(catalyst or {}).get("estimate_date"),
                    sic=(submissions or {}).get("sic"),
                )
            except Exception:  # noqa: BLE001 - persistence failure must not fail the request
                logger.warning("Failed to save verdict for %s", ticker, exc_info=True)

        payload = _serialize_financials(normalized, ratios)
        return jsonify({
            "ok": True,
            **payload,
            "cik": cik,
            "name": name,
            "analysis": analysis,
            "technical": technical,
            "metrics": metrics,
            "red_flags": flags,
            "catalyst": catalyst,
            "analyst": analyst,
            "earnings": earnings,
            "as_of": as_of.isoformat() if as_of is not None else None,
        })
    finally:
        if job_id is not None:
            _progress_finish(error=progress_error)


@app.route("/api/analyze/progress", methods=["GET"])
def api_analyze_progress():
    """Report a running/finished ``POST /api/analyze`` request's progress.

    This works with no extra plumbing between the two routes because
    ``main``'s ``app.run(...)`` leaves Flask's ``threaded`` default (True):
    the analyze POST keeps executing in its own request thread while this
    GET is served concurrently on another, and both threads read/write the
    same `_analyze_progress` dict under `_analyze_lock`.

    Query params:
        job: The same id the client passed as ``POST /api/analyze``'s JSON
            ``job`` field (required; must match ``^[A-Za-z0-9_-]{1,64}$``).

    Returns ``200 {"ok": true, "job", "found", "done", "stage", "stage_index",
    "total", "label", "elapsed", "error", "stages"}``, where ``stages`` is
    the full :data:`_ANALYZE_STAGES` table annotated with each stage's
    ``state`` ("done"/"active"/"pending"). An unrecognized/expired job id is
    NOT an error -- it returns ``found: false`` with every stage "pending",
    since the client may start polling before ``POST /api/analyze``'s
    `_progress_begin` call has run, or the record may already have been
    pruned after the job finished (see `_prune_analyze_progress`). Only a
    missing/malformed `job` param itself is a 400.
    """
    job_id = (request.args.get("job") or "").strip()
    if not _JOB_ID_RE.match(job_id):
        return jsonify(
            {"ok": False, "error": "Query parameter 'job' is required and must be a valid job id."}
        ), 400

    with _analyze_lock:
        record = _analyze_progress.get(job_id)
        record = dict(record) if record is not None else None

    if record is None:
        return jsonify({
            "ok": True,
            "job": job_id,
            "found": False,
            "done": False,
            "stage": None,
            "stage_index": 0,
            "total": len(_ANALYZE_STAGES),
            "label": None,
            "elapsed": 0,
            "error": None,
            "stages": [
                {"key": key, "label": label, "state": "pending"} for key, label in _ANALYZE_STAGES
            ],
        })

    stage_index = record["stage_index"]
    done = record["done"]
    error = record["error"]

    stages = []
    for i, (key, label) in enumerate(_ANALYZE_STAGES):
        # When the job is done with no error, every stage reads as "done".
        # When it's done WITH an error, leave the stage it failed on (still
        # `stage_index` -- a failure never advances the stage) as "active"
        # rather than "done", so the UI can point at exactly where the run
        # broke instead of showing a false all-green completion.
        if done and error is None:
            state = "done"
        elif i < stage_index:
            state = "done"
        elif i == stage_index:
            state = "active"
        else:
            state = "pending"
        stages.append({"key": key, "label": label, "state": state})

    return jsonify({
        "ok": True,
        "job": job_id,
        "found": True,
        "done": done,
        "stage": record["stage"],
        "stage_index": stage_index,
        "total": len(_ANALYZE_STAGES),
        "label": _ANALYZE_STAGE_LABELS.get(record["stage"]),
        # UI telemetry only (elapsed wall-clock seconds since the job began) --
        # never fed into any analysis/valuation result.
        "elapsed": round(time.monotonic() - record["started_at"], 1),
        "error": error,
        "stages": stages,
    })


@app.route("/api/swing/scan", methods=["POST"])
def api_swing_scan():
    """Start a background swing-trade screener scan for one index.

    JSON body (all optional):
        no_cache: Bypass the on-disk price cache (default False).
        limit: Scan only the first N universe tickers (a quick smoke run);
            omitted/``None`` scans the full universe.
        index: Universe code (default ``SP500``), resolved through
            :func:`sec_analyzer.screener.universe.normalize_index`.

    Exactly one scan may run at a time, **across all indexes** -- the
    bottleneck is CPU, not the index (SWING_SPEC.md Sec.6/Sec.10). Starts a
    ``threading.Thread(daemon=True)`` and returns immediately with
    ``202 {"ok": true, "status": "running", "total": N, "index": <code>}``.
    If a scan is already running (for this index or another one), returns
    ``409 {"ok": false, "error": "A scan for <label> is already running."}``
    naming the index actually running, without starting a second one.
    """
    body = request.get_json(silent=True) or {}
    no_cache = bool(body.get("no_cache", False))
    limit = body.get("limit")
    index = normalize_index(body.get("index"))

    with _swing_lock:
        if _swing_state["running"]:
            running_label = _swing_state["index_label"] or universe_label(_swing_state["index"])
            return jsonify(
                {"ok": False, "error": f"A scan for {running_label} is already running."}
            ), 409

        try:
            universe = load_universe(index=index)
        except OSError:
            logger.exception("Could not load %s universe for swing scan", index)
            return jsonify({"ok": False, "error": "Could not load the ticker universe."}), 500

        tickers = None
        if limit is not None:
            try:
                limit_n = max(int(limit), 0)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "'limit' must be an integer."}), 400
            tickers = [row["ticker"] for row in universe[:limit_n]]

        total = len(tickers) if tickers is not None else len(universe)

        _swing_state["running"] = True
        _swing_state["done"] = 0
        _swing_state["total"] = total
        _swing_state["index"] = index
        _swing_state["index_label"] = universe_label(index)
        _swing_state["started_at"] = _utc_now_iso()
        _swing_state["finished_at"] = None
        _swing_state["error"] = None

    thread = threading.Thread(
        target=_run_swing_scan_worker,
        args=(tickers, no_cache, DEFAULT_MAX_WORKERS, index),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "status": "running", "total": total, "index": index}), 202


@app.route("/api/swing/status", methods=["GET"])
def api_swing_status():
    """Report the current/most-recent swing scan's progress.

    Always ``200`` -- the client polls this every 2s while ``running`` is
    true (SWING_SPEC.md Sec.6), whether or not a scan has ever been started.
    ``index``/``index_label`` name the index the running (or last-completed)
    scan belongs to, so the client can tell whether the progress/result it
    sees belongs to the index it currently has displayed.
    """
    with _swing_lock:
        state = dict(_swing_state)
    return jsonify({
        "ok": True,
        "running": state["running"],
        "done": state["done"],
        "total": state["total"],
        "index": state["index"],
        "index_label": state["index_label"],
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "error": state["error"],
    })


@app.route("/api/swing/results", methods=["GET"])
def api_swing_results():
    """Return the latest persisted swing scan for one index.

    Query params:
        index: Universe code (default ``SP500``), resolved through
            :func:`sec_analyzer.screener.universe.normalize_index`.

    Returns ``{"ok": true, "index": <code>, "scan": <SWING_SPEC.md Sec.4
    dict or null>}`` -- ``null`` when that index has never been scanned.
    """
    index = normalize_index(request.args.get("index"))
    try:
        scan = load_latest_swing_scan(db_path=Config.DB_PATH, universe=index)
        return jsonify({"ok": True, "index": index, "scan": scan})
    except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
        logger.exception("Unexpected error loading latest swing scan for %s", index)
        return jsonify(
            {"ok": False, "error": "An unexpected error occurred while loading the scan results."}
        ), 500


#: Shell used by ``_error_page`` for ``/report`` failures -- a small,
#: self-contained (no external resources) HTML page in the same dark
#: palette as ``sec_analyzer.report.template``, so an error looks like a
#: degraded report rather than a bare Flask error response.
_ERROR_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Report Error</title>
<style>
  html, body {{
    margin: 0; padding: 0;
    background: #0d1420; color: #e7ecf5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.45;
  }}
  .page {{ max-width: 560px; margin: 0 auto; padding: 48px 16px; }}
  .card {{
    background: #111b2b; border: 1px solid #223349; border-radius: 14px;
    padding: 22px 20px;
  }}
  h1 {{ font-size: 1rem; color: #ff6b5e; margin: 0 0 10px; }}
  p {{ margin: 0; font-size: 0.9rem; color: #e7ecf5; }}
</style>
</head>
<body>
<div class="page"><div class="card">
  <h1>Report could not be generated</h1>
  <p>{message}</p>
</div></div>
</body>
</html>"""


def _error_page(message: str) -> str:
    """Render a minimal, styled, self-contained HTML error page.

    Used by ``GET /report`` so a pipeline failure (bad ticker, missing
    config, unexpected error) renders as a small HTML page rather than a
    JSON error body or a bare stack trace -- the route is meant to be
    opened directly in a browser or embedded in an ``<iframe>``, where a
    JSON response would just show as raw text.
    """
    return _ERROR_PAGE_TEMPLATE.format(message=escape(str(message)))


@app.route("/report", methods=["GET"])
def report():
    """Run the full fundamental + technical analysis pipeline for a ticker
    and return the standalone HTML verdict-card report -- the same report
    ``sec_analyzer.cli``'s ``analyze --html`` flag writes to disk (see
    ``sec_analyzer.report.generator.render_report_html``) -- rendered live
    as the response body.

    This is the "type a ticker, see the exact report card" counterpart to
    ``/api/analyze``: where that route returns JSON, this one returns a
    complete, self-contained HTML page (the same unified "Verdict Terminal"
    template the ``/`` search page renders client-side, only with the
    analysis baked in server-side) suitable for opening directly in a
    browser tab.

    Query params:
        ticker: Stock ticker symbol (required).
        horizon: Investment horizon -- "3m", "1y", or "5y" (default "1y").
        provider: Analysis provider -- "claude_code", "ollama", or "script";
            defaults to ``Config.ANALYZER_PROVIDER`` when omitted/blank.
        years: Number of most-recent fiscal years to retain (default 12,
            wider than ``/api/financials``'/``/api/analyze``'s default 5 so
            the valuation engine's multiples-percentile history has more to
            work with).
        no_cache: Bypass the on-disk raw JSON cache (default False).

    Returns:
        On success, a ``200`` response with ``Content-Type: text/html``
        whose body is the rendered verdict-card report. On any failure
        (missing ticker, bad ticker, missing SEC_USER_AGENT config, or an
        unexpected error), an HTML error page (via :func:`_error_page`)
        with a ``400``/``404``/``500`` status -- never a JSON body, and
        never a bare Python traceback.
    """
    ticker = (request.args.get("ticker") or "").strip()
    if not ticker:
        return _error_page("Query parameter 'ticker' is required."), 400

    try:
        years = int(request.args.get("years", 12))
    except (TypeError, ValueError):
        return _error_page("'years' must be an integer."), 400

    horizon = (request.args.get("horizon") or "1y").strip().lower()
    if horizon not in _VALID_HORIZONS:
        horizon = "1y"

    provider = request.args.get("provider") or None
    no_cache = _bool_param(request.args.get("no_cache"))

    as_of, as_of_error = _parse_as_of_param(request.args.get("as_of"))
    if as_of_error:
        return _error_page(as_of_error), 400

    try:
        (
            cik, name, normalized, ratios, metrics, technical, flags, catalyst, price,
            submissions, price_df, fred_rate,
        ) = _run_full_pipeline(ticker, years, no_cache, horizon, as_of)

    except ConfigError as exc:
        logger.error("Configuration error while generating report for %s: %s", ticker, exc)
        return _error_page(str(exc)), 400

    except ValueError as exc:
        logger.info("Ticker resolution failed for %s: %s", ticker, exc)
        return _error_page(str(exc)), 404

    except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
        logger.exception("Unexpected error generating report for %s", ticker)
        return _error_page(
            "An unexpected server error occurred while generating the report."
        ), 500

    # Analyst consensus and earnings-surprise history are both undated; suppress
    # them in as-of mode (as-of contract).
    analyst = None if as_of is not None else _fetch_analyst_targets(ticker, no_cache)
    earnings = None if as_of is not None else _fetch_earnings_history(ticker, no_cache)

    resolved_provider = provider or Config.ANALYZER_PROVIDER
    logger.info("Running %s analysis for %s report (horizon=%s)", resolved_provider, ticker, horizon)
    analysis = interpret(
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
        as_of=as_of,
        fred_rate=fred_rate,
    )

    if isinstance(analysis, dict) and as_of is not None:
        analysis["as_of"] = as_of.isoformat()

    if isinstance(analysis, dict) and "error" not in analysis:
        _attach_momentum(analysis, ticker, normalized, technical, as_of)
        _attach_insider(analysis, cik, ticker, submissions, no_cache, as_of)
        _attach_macro(analysis, as_of, no_cache)
        _attach_peers(
            analysis, ticker, cik, submissions, normalized, ratios, metrics, as_of=as_of
        )
        try:
            save_verdict(
                ticker, cik, horizon, resolved_provider, price, analysis,
                db_path=Config.DB_PATH, valuation=analysis.get("valuation"),
                as_of=as_of.isoformat() if as_of is not None else None,
                catalyst_date=(catalyst or {}).get("estimate_date"),
                sic=(submissions or {}).get("sic"),
            )
        except Exception:  # noqa: BLE001 - persistence failure must not fail the request
            logger.warning("Failed to save verdict for %s", ticker, exc_info=True)

    price_as_of = technical.get("as_of") if technical else None
    html = render_report_html(
        ticker, horizon, analysis,
        metrics=metrics, technical=technical, flags=flags, price=price, as_of=price_as_of,
        entity_name=name, analyst=analyst,
        analysis_as_of=as_of.isoformat() if as_of is not None else None,
        financials=_serialize_financials(normalized, ratios),
        earnings=earnings,
    )
    return html


def main() -> None:
    """Configure logging and start the development server on 127.0.0.1:5050."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
