"""Scan orchestration for the multi-index swing-trade screener.

Fetches (or loads from cache) price history for every constituent of a
bundled universe (S&P 500 or Nasdaq-100, see
``sec_analyzer/screener/universe.py``), derives the technical-indicator set
and swing score for each, and assembles one ranked scan result -- the
payload the CLI prints and the web tab renders (see
``sec_analyzer/screener/SWING_SPEC.md`` Sec.4).

This module is purely an I/O + fan-out orchestrator: all of the actual
scoring math lives in :mod:`sec_analyzer.technical.swing`
(``compute_swing_score``), which this module treats as a black box.

Concurrency primitive: ``compute_indicators`` is pure-Python CPU work
(pivot/support-resistance loops, RSI divergence scan) costing roughly one
second per ticker, so the fan-out is CPU-bound, not network-bound -- a
``ThreadPoolExecutor`` cannot parallelize it (the GIL serializes pure-Python
bytecode across threads), so a full 500-name scan on threads runs at
essentially one core's speed regardless of worker count (see
``sec_analyzer/backtest/SWING_STUDY_SPEC.md`` Sec.6, the binding contract for
this module's process-pool fan-out). ``scan_swing`` therefore fans out
across a ``concurrent.futures.ProcessPoolExecutor`` by default:

- Worker functions are module-level and picklable (:func:`_run_one_in_process`);
  only small dicts/bools cross the process boundary. Each worker process
  reads its own ticker's price frame from the on-disk cache via
  :func:`sec_analyzer.fetch.prices.get_price_history` -- no ``DataFrame`` (or
  the SPY benchmark ``Series``) is ever pickled.
- The SPY benchmark frame is fetched at most once *per worker process*, via
  a lazy module-level cache (:data:`_worker_spy_cache`), not passed in per
  task.
- ``progress_cb`` is not picklable, so progress is always emitted from the
  parent process as futures complete -- identical for both the process and
  thread paths.
- If a process pool cannot start at all (restricted environment, frozen app,
  ``OSError``/``NotImplementedError``/``RuntimeError`` such as
  ``BrokenProcessPool``), this module logs a warning and falls back to the
  original ``ThreadPoolExecutor`` path; results are identical either way.
  ``use_processes=False`` forces the thread path explicitly (used by tests
  and available to any caller).
- For very small fan-outs (see :data:`_PROCESS_POOL_MIN_TICKERS`), threads
  are used even when ``use_processes`` is true: spawning worker processes
  has real startup/IPC overhead that a handful of tickers (a CLI smoke run,
  a unit test, an ``--limit`` request) cannot recoup, so there is nothing to
  gain -- and a stubbed-out price/universe layer (as the unit tests use)
  necessarily lives only in the parent process's memory, which only the
  thread path can see.
"""

import concurrent.futures
import logging
import os
from collections import Counter
from datetime import datetime, timezone

from sec_analyzer.fetch.prices import get_price_history
from sec_analyzer.screener.universe import DEFAULT_INDEX, load_universe, normalize_index, price_symbol, universe_label
from sec_analyzer.technical.indicators import compute_indicators, relative_strength
from sec_analyzer.technical.swing import compute_swing_score

logger = logging.getLogger(__name__)

#: Benchmark ticker fetched once per scan (per process, for the process-pool
#: path) and reused for every constituent's relative-strength cross-check
#: (SWING_SPEC.md Sec.4 step 2).
BENCHMARK = "SPY"

#: Default worker-pool size for :func:`scan_swing`. The fan-out is CPU-bound
#: (see module docstring), so the sensible default is the machine's core
#: count -- never more workers than there are cores to run them on -- rather
#: than a large network-style pool. Falls back to 8 on the rare platform
#: where ``os.cpu_count()`` can't tell.
DEFAULT_MAX_WORKERS = os.cpu_count() or 8

#: Below this many tickers, use threads even when ``use_processes`` is true
#: (see module docstring): process-pool startup/IPC overhead isn't worth it
#: for a handful of tickers, and it's also the only path a caller's
#: in-process stub (as the unit tests use for the price/universe layer) can
#: reach. Comfortably below any real scan (Nasdaq-100 is 103, S&P 500 is
#: 503) and comfortably above the largest fixture used in this package's
#: tests.
_PROCESS_POOL_MIN_TICKERS = 20


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scan_one(entry: dict, spy_close, no_cache: bool) -> dict:
    """Fetch price history and compute the full swing row for one constituent.

    Args:
        entry: A universe row, ``{"ticker", "name", "sector", "cik"}``
            (``name``/``sector``/``cik`` may be ``None`` for a ticker not
            found in the bundled universe -- see :func:`scan_swing`).
        spy_close: The SPY benchmark's ``Close`` series, or ``None`` when the
            benchmark fetch failed for this scan (relative strength then
            simply doesn't get attached, and the ``rel_strength`` swing
            component drops out for every ticker).
        no_cache: Forwarded to :func:`sec_analyzer.fetch.prices.get_price_history`.

    Returns:
        The per-row dict shape from SWING_SPEC.md Sec.4.

    Raises:
        Exception: Any failure (bad ticker, price-fetch failure, unusable
            history, or a swing score that couldn't be computed at all) is
            allowed to propagate -- the caller (:func:`_run_one`) catches it
            and records the ticker in ``skipped`` instead of failing the
            whole scan.
    """
    ticker = entry["ticker"]
    symbol = price_symbol(ticker)

    df, _source = get_price_history(symbol, no_cache=no_cache)
    indicators = compute_indicators(df)
    if spy_close is not None:
        indicators["relative_strength"] = relative_strength(df["Close"], spy_close, benchmark=BENCHMARK)

    swing = compute_swing_score(indicators)
    if swing is None:
        raise ValueError("Swing skoru hesaplanamadı (yetersiz veri).")

    price = indicators.get("price")
    atr14 = indicators.get("atr14")
    atr_pct = None
    if _is_num(price) and price > 0 and _is_num(atr14):
        atr_pct = round(float(atr14) / float(price) * 100.0, 1)

    rel_strength = indicators.get("relative_strength") or {}

    return {
        "ticker": ticker,
        "name": entry.get("name"),
        "sector": entry.get("sector"),
        "price": price,
        "as_of": indicators.get("as_of"),
        "change_1d_pct": indicators.get("change_1d_pct"),
        "rsi14": indicators.get("rsi14"),
        "dist_sma50_pct": indicators.get("dist_sma50_pct"),
        "dist_52w_high_pct": indicators.get("dist_52w_high_pct"),
        "rs_3m_pct": rel_strength.get("rs_3m_pct"),
        "atr_pct": atr_pct,
        "score": swing["score"],
        "s": swing["s"],
        "label": swing["label"],
        "setup": swing["setup"],
        "badges": swing["badges"],
        "entry": swing["entry"],
        "stop": swing["stop"],
        "stop_pct": swing["stop_pct"],
        "target": swing["target"],
        "target_pct": swing["target_pct"],
        "rr": swing["rr"],
        "summary": swing["summary"],
        "components": swing["components"],
    }


def _run_one(entry: dict, spy_close, no_cache: bool) -> "tuple[str, dict]":
    """Run :func:`_scan_one` for one ticker, converting any failure into a
    ``skip`` result instead of letting it propagate (SWING_SPEC.md Sec.4 step
    4 -- one bad ticker must never fail a whole scan).

    Shared by both the thread-pool and process-pool fan-outs; the only
    difference between the two paths is how ``spy_close`` is obtained (see
    :func:`_run_one_in_process`).

    Returns:
        ``("ok", row_dict)`` on success, ``("skip", {"ticker", "reason"})``
        on any failure.
    """
    try:
        return "ok", _scan_one(entry, spy_close, no_cache)
    except Exception as exc:  # noqa: BLE001 - one bad ticker must never fail the scan
        logger.warning("Swing scan failed for %s", entry.get("ticker"), exc_info=True)
        return "skip", {"ticker": entry.get("ticker"), "reason": str(exc)}


#: Per-process lazy cache for the SPY benchmark ``Close`` series, keyed by
#: ``no_cache``. Populated at most once per worker process by
#: :func:`_worker_spy_close` -- never pickled across the process boundary,
#: exactly as required by SWING_STUDY_SPEC.md Sec.6 ("loaded once per worker
#: process ... not passed per task"). Each ``ProcessPoolExecutor`` worker
#: process gets its own copy of this module (and this dict) on import, so
#: there is no cross-process sharing to worry about.
_worker_spy_cache: "dict[bool, object]" = {}


def _worker_spy_close(no_cache: bool):
    """Return this worker process's cached SPY ``Close`` series, fetching it
    (once) on first use. Mirrors :func:`_fetch_benchmark`'s non-fatal
    posture: a fetch failure caches (and returns) ``None``, which simply
    disables the ``rel_strength`` component for every ticker this worker
    handles, exactly like the thread-pool path."""
    key = bool(no_cache)
    if key not in _worker_spy_cache:
        _worker_spy_cache[key] = _fetch_benchmark(no_cache)
    return _worker_spy_cache[key]


def _run_one_in_process(entry: dict, no_cache: bool) -> "tuple[str, dict]":
    """Module-level, picklable per-ticker task for the process-pool fan-out.

    Only ``entry`` (a small dict of strings) and ``no_cache`` (a bool) cross
    the process boundary. The SPY benchmark series is obtained from this
    worker process's own lazy cache (:func:`_worker_spy_close`) rather than
    being passed in, and the ticker's own price frame is read straight from
    the on-disk cache by :func:`_scan_one` via ``get_price_history`` -- so no
    ``DataFrame`` or ``Series`` is ever pickled between processes.

    Args:
        entry: A universe row, see :func:`_scan_one`.
        no_cache: Forwarded to the price-history fetch (both the ticker's own
            history and, on first use in this process, the SPY benchmark).

    Returns:
        Same shape as :func:`_run_one`.
    """
    spy_close = _worker_spy_close(no_cache)
    return _run_one(entry, spy_close, no_cache)


def _resolve_entries(tickers: "list[str] | None", index: str = DEFAULT_INDEX) -> "list[dict]":
    """Resolve the universe rows to scan (SWING_SPEC.md Sec.4 step 1).

    With ``tickers is None``, scans the whole bundled universe for ``index``.
    Otherwise filters to the given tickers (de-duplicated, order preserved),
    looking each up in that universe for its ``name``/``sector``/``cik`` --
    an unknown ticker is still scanned, just with those fields ``None``.

    Args:
        tickers: Optional ticker subset to restrict the scan to.
        index: Already-normalized universe code (see
            :func:`sec_analyzer.screener.universe.normalize_index`).
    """
    universe = load_universe(index=index)
    if tickers is None:
        return universe

    by_ticker = {row["ticker"]: row for row in universe}
    seen = set()
    entries = []
    for raw in tickers:
        if not raw:
            continue
        t = str(raw).strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        entries.append(by_ticker.get(t, {"ticker": t, "name": None, "sector": None, "cik": None}))
    return entries


def _fetch_benchmark(no_cache: bool):
    """Fetch the SPY benchmark ``Close`` series once, or ``None`` on failure
    (SWING_SPEC.md Sec.4 step 2 -- a benchmark-fetch failure degrades every
    row's relative-strength component rather than failing the scan)."""
    try:
        spy_df, _source = get_price_history(BENCHMARK, no_cache=no_cache)
        return spy_df["Close"]
    except Exception:  # noqa: BLE001 - benchmark fetch failure must not fail the scan
        logger.warning(
            "Could not fetch %s benchmark; relative strength disabled for this scan", BENCHMARK, exc_info=True
        )
        return None


def _emit_progress(progress_cb, done: int, total: int, ticker) -> None:
    """Call ``progress_cb(done, total, ticker)`` from the parent process,
    swallowing any exception it raises (SWING_SPEC.md Sec.4 step 5 -- a
    caller's broken callback must never break the scan). A no-op when
    ``progress_cb`` is ``None``."""
    if progress_cb is None:
        return
    try:
        progress_cb(done, total, ticker)
    except Exception:  # noqa: BLE001 - a broken progress callback must not break the scan
        logger.warning("swing scan progress_cb failed", exc_info=True)


def _scan_via_threads(
    entries: "list[dict]", total: int, no_cache: bool, max_workers: int, progress_cb
) -> "tuple[list[dict], list[dict]]":
    """Fan out over a ``ThreadPoolExecutor`` -- the original concurrency
    primitive, kept as the guaranteed-available fallback (and as the path
    used for small fan-outs, see :data:`_PROCESS_POOL_MIN_TICKERS`).

    The SPY benchmark is fetched once here, in the parent thread, and shared
    (via closure) with every task -- safe because threads share process
    memory, unlike the process-pool path.
    """
    spy_close = _fetch_benchmark(no_cache)

    rows: "list[dict]" = []
    skipped: "list[dict]" = []
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {executor.submit(_run_one, entry, spy_close, no_cache): entry for entry in entries}
        for future in concurrent.futures.as_completed(future_to_entry):
            entry = future_to_entry[future]
            status, payload = future.result()
            if status == "ok":
                rows.append(payload)
            else:
                skipped.append(payload)
            done += 1
            _emit_progress(progress_cb, done, total, entry.get("ticker"))

    return rows, skipped


def _scan_via_processes(
    entries: "list[dict]", total: int, no_cache: bool, max_workers: int, progress_cb
) -> "tuple[list[dict], list[dict]]":
    """Fan out over a ``ProcessPoolExecutor`` so ``compute_indicators``'s
    CPU-bound work actually parallelizes across cores (see module docstring
    and ``SWING_STUDY_SPEC.md`` Sec.6).

    Only ``entry`` dicts and ``no_cache`` are submitted per task (see
    :func:`_run_one_in_process`); the SPY benchmark is fetched independently
    by each worker process, lazily, at most once.

    Raises:
        OSError, NotImplementedError, RuntimeError: If the process pool
            cannot start or a worker process dies (``BrokenProcessPool`` is a
            ``RuntimeError``). The caller (:func:`scan_swing`) catches these
            and falls back to :func:`_scan_via_threads`.
    """
    rows: "list[dict]" = []
    skipped: "list[dict]" = []
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {executor.submit(_run_one_in_process, entry, no_cache): entry for entry in entries}
        for future in concurrent.futures.as_completed(future_to_entry):
            entry = future_to_entry[future]
            status, payload = future.result()
            if status == "ok":
                rows.append(payload)
            else:
                skipped.append(payload)
            done += 1
            _emit_progress(progress_cb, done, total, entry.get("ticker"))

    return rows, skipped


def scan_swing(
    tickers: "list[str] | None" = None,
    no_cache: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress_cb=None,
    index: str = DEFAULT_INDEX,
    use_processes: bool = True,
) -> dict:
    """Scan a bundled universe (or a subset) and rank every constituent's swing score.

    Args:
        tickers: When given, restricts the scan to these tickers (unknown
            ones are still scanned, with ``name``/``sector`` = ``None``).
            ``None`` (default) scans the full bundled universe.
        no_cache: Bypass the on-disk price cache and re-fetch for every
            ticker (including the SPY benchmark).
        max_workers: Worker-pool size for the per-ticker fan-out (default
            :data:`DEFAULT_MAX_WORKERS`, the machine's core count).
        progress_cb: Optional ``progress_cb(done: int, total: int, ticker:
            str)``, called from the parent process after each ticker
            completes (success or skip), for both the process-pool and
            thread-pool paths. Wrapped in its own try/except -- a broken
            callback cannot break the scan.
        index: Which bundled universe to scan (SWING_SPEC.md Sec.1),
            resolved through
            :func:`sec_analyzer.screener.universe.normalize_index`. Defaults
            to the S&P 500 so existing calls are unaffected.
        use_processes: Fan out over a ``ProcessPoolExecutor`` (default
            ``True``) so the CPU-bound indicator computation actually
            parallelizes across cores. Falls back to a
            ``ThreadPoolExecutor`` automatically when a process pool cannot
            start, and (regardless of this flag) for fan-outs smaller than
            :data:`_PROCESS_POOL_MIN_TICKERS`. Pass ``False`` to force the
            thread path explicitly (used by this package's tests, whose
            stubbed price/universe layer only the thread path -- sharing the
            parent's process memory -- can see).

    Returns:
        ``{"generated_at", "price_as_of", "universe", "universe_label",
        "count", "total", "skipped", "rows"}`` -- see SWING_SPEC.md Sec.4.
        ``rows`` is sorted score DESC, then ticker ASC (the only
        deterministic ordering guarantee; ``generated_at`` is the sole
        wall-clock value and no score depends on it). This ordering, and
        every score, is identical regardless of which fan-out path ran.

    One bad ticker (a raised exception of any kind, including
    :class:`sec_analyzer.fetch.prices.PriceDataError`) never fails the whole
    scan -- it is logged at warning level and recorded in ``skipped`` as
    ``{"ticker", "reason"}`` instead.
    """
    resolved_index = normalize_index(index)
    entries = _resolve_entries(tickers, index=resolved_index)
    total = len(entries)

    rows: "list[dict]" = []
    skipped: "list[dict]" = []
    ran_via_process = False

    if use_processes and total >= _PROCESS_POOL_MIN_TICKERS:
        try:
            rows, skipped = _scan_via_processes(entries, total, no_cache, max_workers, progress_cb)
            ran_via_process = True
        except (OSError, NotImplementedError, RuntimeError):
            logger.warning(
                "Process pool unavailable for swing scan (%d tickers); falling back to thread pool",
                total,
                exc_info=True,
            )

    if not ran_via_process:
        rows, skipped = _scan_via_threads(entries, total, no_cache, max_workers, progress_cb)

    rows.sort(key=lambda r: (-r["score"], r["ticker"]))
    skipped.sort(key=lambda s: s.get("ticker") or "")

    as_of_values = [r["as_of"] for r in rows if r.get("as_of")]
    price_as_of = Counter(as_of_values).most_common(1)[0][0] if as_of_values else None

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "generated_at": generated_at,
        "price_as_of": price_as_of,
        "universe": resolved_index,
        "universe_label": universe_label(resolved_index),
        "count": len(rows),
        "total": total,
        "skipped": skipped,
        "rows": rows,
    }
