"""Stage 1 of the swing backtest: a score-decile forward-return study.

Binding contract: ``sec_analyzer/backtest/SWING_STUDY_SPEC.md``. This module
answers one question, with the fewest assumptions possible -- does a higher
``technical.swing.compute_swing_score`` actually predict a higher forward
return? It is a *cross-sectional ranking* study (no entries, no stops, no
equity curve): for each monthly rebalance date, every scoreable ticker in the
universe is bucketed into a score decile (assigned *within that date*, never
pooled across dates -- SWING_STUDY_SPEC.md Sec.2), and the forward return of
each decile is measured at +10 and +21 trading days against the same-window
SPY return.

Point-in-time correctness (SWING_STUDY_SPEC.md Sec.3) is the part that
silently invalidates everything if it's wrong:

* For rebalance date ``d``, the indicator set is computed on
  ``slice_asof(price_df, d)`` -- never the full frame -- so nothing past
  ``d`` leaks into the 52-week high, SMA200, support/resistance, or the
  Bollinger percentile. The SPY benchmark is sliced the same way for the
  ``relative_strength`` cross-check.
* The score is derived from bar ``d``'s close, so the forward window starts
  at the *next* bar in that ticker's own price series (not calendar ``d+1``,
  which may not be a trading day for every name): ``fwd_h_pct`` is measured
  from close(bar ``d`` position + 1) to close(bar ``d`` position + 1 + ``h``).

Per SWING_STUDY_SPEC.md Sec.6, the parallelization unit is one ticker across
all rebalance dates: each worker loads that ticker's price frame once from
the on-disk cache (:func:`sec_analyzer.fetch.prices.get_price_history`) and
slices it per date, so inter-process traffic stays to small dicts. A
``concurrent.futures.ProcessPoolExecutor`` is used (this is CPU-bound
``compute_indicators`` work, so threads cannot parallelize it under the
GIL); a pool that cannot start falls back to a thread pool with a logged
warning, producing identical results either way since both paths execute the
exact same module-level worker function.

Per ROADMAP.md's "Backtest -- tasarım ilkesi" and ``sec_analyzer.backtest``:
this is an EVALUATION tool, never an OPTIMIZATION tool. The swing weights in
``technical/swing.py`` must never be tuned against this study's output.
"""

import concurrent.futures
import logging
import math
import os
import random
import statistics
from collections import defaultdict
from datetime import date, datetime, timezone

import pandas as pd

from sec_analyzer.backtest import BACKTEST_DISCLAIMER
from sec_analyzer.fetch.prices import get_price_history, slice_asof
from sec_analyzer.screener.universe import load_universe, normalize_index, price_symbol, universe_label
from sec_analyzer.technical.indicators import compute_indicators, relative_strength
from sec_analyzer.technical.swing import compute_swing_score

logger = logging.getLogger(__name__)

#: Forward return horizons, in trading days (SWING_STUDY_SPEC.md Sec.2).
#: Fixed by the spec -- every configured horizon is always reported; a
#: caller may narrow the date range or universe but must not add/drop these.
FORWARD_HORIZONS = (10, 21)

#: Cross-sectional decile count, assigned WITHIN each rebalance date.
DECILES = 10

#: Benchmark ticker for relative-strength and excess-return measurement.
BENCHMARK = "SPY"

#: Below this many pooled observations, a decile cell is flagged
#: ``low_sample`` rather than silently reported as meaningful (extends the
#: existing ``backtest report`` "yetersiz örneklem" n<10 convention to n<30,
#: since this study's cells are naturally larger and noisier).
_LOW_SAMPLE_MIN = 30

#: Fixed-seed bootstrap for the top-minus-bottom decile spread's confidence
#: interval (SWING_STUDY_SPEC.md Sec.4) -- same inputs must always produce
#: the same bounds.
_BOOTSTRAP_SEED = 0
_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_CI_LOW_PCT = 0.025
_BOOTSTRAP_CI_HIGH_PCT = 0.975

#: Default process/thread-pool size when the caller doesn't specify one.
_DEFAULT_MAX_WORKERS = 8

#: Known limitations that MUST be surfaced in every study result
#: (SWING_STUDY_SPEC.md Sec.1). Survivorship bias is listed first/dominant:
#: the universe is TODAY's index membership, not point-in-time membership,
#: so every return figure here is optimistic.
_LIMITATIONS = [
    "Hayatta kalma yanlılığı (baskın etki): evren bugünkü endeks üyeliğini "
    "yansıtır, tarihsel (point-in-time) üyeliği değil. Endeksten çıkarılmış "
    "veya iflas etmiş şirketler örneklemde yok, bu yüzden buradaki her getiri "
    "rakamı iyimser yönde sapmalıdır.",
    "Ücretsiz fiyat kaynağı: endeksten düşürülmüş/iflas etmiş hisselerin "
    "yfinance'te geçmiş verisi hiç yok, bu da hayatta kalma "
    "yanlılığını güçlendirir.",
    "Maliyetsiz getiriler: komisyon, kayma (slippage) veya borçlanma "
    "maliyeti yok -- raporlanan getiriler brüttür.",
    "Çakışan gözlemler: ardışık rebalance tarihleri aynı fiyat barlarını "
    "paylaşır, yani gözlemler birbirinden istatistiksel olarak bağımsız "
    "değildir; herhangi bir güven aralığı gösterge niteliğindedir, biçimsel "
    "bir anlamlılık testi değildir.",
]


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _close_on_or_before(df, target) -> "float | None":
    """Last ``Close`` in ``df`` dated on/before ``target``, or ``None``.

    Local re-implementation of the same lookup used by
    ``backtest.outcomes._close_on_or_before`` (this project's convention is
    to re-implement small private helpers locally rather than import
    another module's privates -- see ``technical/swing.py``'s ``_clamp``).
    """
    if df is None or df.empty:
        return None
    cutoff = pd.Timestamp(target)
    sliced = df.loc[df.index <= cutoff]
    if sliced.empty:
        return None
    return float(sliced.iloc[-1]["Close"])


def _month_end_trading_dates(index, start, end) -> "list":
    """The last trading day of each calendar month in ``index``, filtered.

    Args:
        index: An ascending ``DatetimeIndex`` (a price frame's trading
            calendar -- the SPY benchmark's, so it's long and reliable).
        start: Inclusive lower bound (``datetime.date`` or ``None`` for no
            lower bound).
        end: Inclusive upper bound (``datetime.date`` or ``None`` for no
            upper bound).

    Returns:
        Ascending list of ``datetime.date`` objects, one per calendar month
        (the monthly rebalance frequency fixed by SWING_STUDY_SPEC.md Sec.2).
    """
    last_of_month = {}
    for ts in index:
        key = (ts.year, ts.month)
        last_of_month[key] = ts  # ascending iteration -> last write wins
    result = []
    for ts in sorted(last_of_month.values()):
        d = ts.date()
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Benchmark: loaded once per process via a lazy module-level cache
# (SWING_STUDY_SPEC.md Sec.6) -- each worker process gets its own fresh copy
# of this module (and thus this dict) on spawn, so "once per process" falls
# out naturally rather than needing explicit process-identity bookkeeping.
# ---------------------------------------------------------------------------
_benchmark_cache: dict = {}


def _get_benchmark_frame():
    """Lazily fetch and cache the SPY benchmark frame for this process.

    Returns ``None`` (cached) if the fetch fails -- every relative-strength
    and excess-return computation then degrades gracefully (``None``
    components / fields) rather than raising.
    """
    if "df" not in _benchmark_cache:
        try:
            df, _source = get_price_history(BENCHMARK)
            _benchmark_cache["df"] = df
        except Exception:  # noqa: BLE001 - a benchmark fetch failure must not abort the study
            logger.warning("swing_study: could not fetch %s benchmark", BENCHMARK, exc_info=True)
            _benchmark_cache["df"] = None
    return _benchmark_cache["df"]


def _score_ticker_task(task: dict) -> dict:
    """Score one ticker across every rebalance date (the per-worker unit).

    Module-level and picklable (SWING_STUDY_SPEC.md Sec.6): this is the exact
    function submitted to either a ``ProcessPoolExecutor`` or, on fallback, a
    ``ThreadPoolExecutor`` -- both paths execute this same code, so their
    results are identical by construction.

    Args:
        task: ``{"ticker": str, "dates": list[str]}`` -- ``dates`` are ISO
            rebalance dates, small and picklable.

    Returns:
        ``{"ticker", "status": "ok"|"skip", "reason": str|None,
        "observations": list[dict], "dropped_no_score": int,
        "dropped_no_forward": int}``. Never raises: any per-ticker failure
        (price fetch, or any per-date failure) is caught and converted into
        ``status="skip"`` or a per-date drop, mirroring
        ``screener.swing_scan``'s "one bad ticker never sinks the batch"
        posture.
    """
    ticker = task["ticker"]
    dates_iso = task["dates"]

    try:
        df, _source = get_price_history(price_symbol(ticker))
    except Exception as exc:  # noqa: BLE001 - one bad ticker must never abort the study
        logger.warning("swing_study: price fetch failed for %s", ticker, exc_info=True)
        return {
            "ticker": ticker, "status": "skip", "reason": f"Fiyat verisi alınamadı: {exc}",
            "observations": [], "dropped_no_score": 0, "dropped_no_forward": 0,
        }

    spy_df = _get_benchmark_frame()
    n = len(df)
    observations = []
    dropped_no_score = 0
    dropped_no_forward = 0

    for d_iso in dates_iso:
        try:
            d = date.fromisoformat(d_iso)
            sliced = slice_asof(df, d)
            if sliced.empty:
                dropped_no_score += 1
                continue

            indicators = compute_indicators(sliced)
            if spy_df is not None:
                spy_sliced = slice_asof(spy_df, d)
                spy_close = spy_sliced["Close"] if not spy_sliced.empty else None
                indicators["relative_strength"] = relative_strength(
                    sliced["Close"], spy_close, benchmark=BENCHMARK
                )

            swing = compute_swing_score(indicators)
            if swing is None:
                dropped_no_score += 1
                continue

            pos = df.index.get_loc(sliced.index[-1])
            if isinstance(pos, slice):  # defensive: duplicate index, shouldn't happen
                pos = pos.stop - 1

            record = {
                "ticker": ticker, "date": d_iso,
                "score": swing["score"], "setup": swing["setup"],
            }
            any_horizon_ok = False
            for h in FORWARD_HORIZONS:
                start_pos = pos + 1
                end_pos = pos + 1 + h
                fwd_key, bench_key, excess_key = f"fwd_{h}_pct", f"bench_{h}_pct", f"excess_{h}_pct"
                if end_pos >= n:
                    record[fwd_key] = None
                    record[bench_key] = None
                    record[excess_key] = None
                    dropped_no_forward += 1
                    continue
                close_start = float(df["Close"].iloc[start_pos])
                close_end = float(df["Close"].iloc[end_pos])
                if close_start <= 0:
                    record[fwd_key] = None
                    record[bench_key] = None
                    record[excess_key] = None
                    dropped_no_forward += 1
                    continue
                fwd_pct = (close_end / close_start - 1.0) * 100.0
                record[fwd_key] = round(fwd_pct, 3)

                bench_pct = None
                if spy_df is not None:
                    bench_start = _close_on_or_before(spy_df, df.index[start_pos])
                    bench_end = _close_on_or_before(spy_df, df.index[end_pos])
                    if bench_start and bench_start > 0 and bench_end is not None:
                        bench_pct = (bench_end / bench_start - 1.0) * 100.0
                record[bench_key] = round(bench_pct, 3) if bench_pct is not None else None
                record[excess_key] = round(fwd_pct - bench_pct, 3) if bench_pct is not None else None
                any_horizon_ok = True

            if any_horizon_ok:
                observations.append(record)
        except Exception:  # noqa: BLE001 - one bad date must never abort the ticker
            logger.warning("swing_study: failed to score %s at %s", ticker, d_iso, exc_info=True)
            dropped_no_score += 1

    return {
        "ticker": ticker, "status": "ok", "reason": None,
        "observations": observations,
        "dropped_no_score": dropped_no_score,
        "dropped_no_forward": dropped_no_forward,
    }


def _make_executor(max_workers: int):
    """A ``ProcessPoolExecutor``, or a ``ThreadPoolExecutor`` on fallback.

    SWING_STUDY_SPEC.md Sec.6: this study's per-date indicator computation is
    CPU-bound pure-Python work, so a thread pool cannot parallelize it (GIL).
    A process pool that cannot start (restricted environment, frozen app)
    falls back to the thread path with a logged warning -- both execute the
    identical :func:`_score_ticker_task`, so results are identical either way.
    """
    try:
        return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers), "process"
    except Exception:  # noqa: BLE001 - environment may not support process pools
        logger.warning(
            "swing_study: ProcessPoolExecutor unavailable; falling back to a thread pool", exc_info=True
        )
        return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers), "thread"


def _spearman_rho(xs: "list[float]", ys: "list[float]") -> float:
    """Spearman rank correlation of ``xs``/``ys`` (stdlib-only; no scipy).

    Computed as the Pearson correlation of average ranks (ties averaged),
    the standard definition of Spearman's rho. Returns ``0.0`` if fewer than
    2 points or either series has zero rank-variance (degenerate case).
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0

    def _ranks(values: "list[float]") -> "list[float]":
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((r - mean_rx) ** 2 for r in rx)
    var_y = sum((r - mean_ry) ** 2 for r in ry)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _bootstrap_spread_ci(top_values: "list[float]", bottom_values: "list[float]"):
    """Fixed-seed bootstrap CI for ``mean(top) - mean(bottom)``.

    ``random.Random(_BOOTSTRAP_SEED)``, ``_BOOTSTRAP_RESAMPLES`` resamples
    (SWING_STUDY_SPEC.md Sec.4) -- deterministic: identical inputs always
    produce identical bounds. Returns ``(ci_low, ci_high)``, or
    ``(None, None)`` if either side is empty.
    """
    if not top_values or not bottom_values:
        return None, None
    rng = random.Random(_BOOTSTRAP_SEED)
    n_top, n_bottom = len(top_values), len(bottom_values)
    diffs = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        top_sample = [top_values[rng.randrange(n_top)] for _ in range(n_top)]
        bottom_sample = [bottom_values[rng.randrange(n_bottom)] for _ in range(n_bottom)]
        diffs.append(statistics.mean(top_sample) - statistics.mean(bottom_sample))
    diffs.sort()
    lo_idx = int(_BOOTSTRAP_CI_LOW_PCT * _BOOTSTRAP_RESAMPLES)
    hi_idx = min(_BOOTSTRAP_RESAMPLES - 1, int(_BOOTSTRAP_CI_HIGH_PCT * _BOOTSTRAP_RESAMPLES))
    return diffs[lo_idx], diffs[hi_idx]


def _assign_deciles(all_observations: "list[dict]") -> "list[dict]":
    """Attach a ``decile`` (1..``DECILES``) to each observation, assigned
    WITHIN its own rebalance date (SWING_STUDY_SPEC.md Sec.2/invariant 3):
    ranking pooled across dates would blend market regimes.

    Ties broken by ticker (ascending) for determinism. Returns a new list of
    shallow copies; the input is left unmodified.
    """
    by_date = defaultdict(list)
    for o in all_observations:
        by_date[o["date"]].append(o)

    out = []
    for date_key in sorted(by_date):
        group = by_date[date_key]
        n = len(group)
        ordered = sorted(group, key=lambda o: (o["score"], o["ticker"]))
        for rank, o in enumerate(ordered):
            decile = min(DECILES, (rank * DECILES) // n + 1)
            o2 = dict(o)
            o2["decile"] = decile
            out.append(o2)
    return out


def _build_decile_tables(obs_with_decile: "list[dict]"):
    """Per-horizon decile table, top-minus-bottom spread, and monotonicity.

    Returns ``(deciles, spread, monotonicity)`` -- see ``run_swing_study``'s
    return-value docstring for the shape of each.
    """
    deciles: dict = {}
    spread: dict = {}
    monotonicity: dict = {}

    for h in FORWARD_HORIZONS:
        excess_key = f"excess_{h}_pct"
        per_decile = defaultdict(list)
        for o in obs_with_decile:
            if o.get(excess_key) is not None:
                per_decile[o["decile"]].append(o)

        entries = []
        for d_ in range(1, DECILES + 1):
            members = per_decile.get(d_)
            if not members:
                continue
            excesses = [m[excess_key] for m in members]
            scores = [m["score"] for m in members]
            m_n = len(members)
            entries.append({
                "decile": d_,
                "n": m_n,
                "mean_excess_pct": round(statistics.mean(excesses), 3),
                "median_excess_pct": round(statistics.median(excesses), 3),
                "hit_rate_pct": round(100.0 * sum(1 for e in excesses if e > 0) / m_n, 1),
                "mean_score": round(statistics.mean(scores), 2),
                "low_sample": m_n < _LOW_SAMPLE_MIN,
            })
        deciles[str(h)] = entries

        top, bottom = per_decile.get(DECILES), per_decile.get(1)
        if top and bottom:
            top_vals = [m[excess_key] for m in top]
            bottom_vals = [m[excess_key] for m in bottom]
            mean_diff = statistics.mean(top_vals) - statistics.mean(bottom_vals)
            ci_low, ci_high = _bootstrap_spread_ci(top_vals, bottom_vals)
            spread[str(h)] = {
                "mean_excess_pct": round(mean_diff, 3),
                "ci_low": round(ci_low, 3) if ci_low is not None else None,
                "ci_high": round(ci_high, 3) if ci_high is not None else None,
            }
        else:
            spread[str(h)] = {"mean_excess_pct": None, "ci_low": None, "ci_high": None}

        xs = [e["decile"] for e in entries]
        ys = [e["mean_excess_pct"] for e in entries]
        monotonicity[str(h)] = round(_spearman_rho(xs, ys), 3) if len(xs) >= 2 else 0.0

    return deciles, spread, monotonicity


def _build_by_setup(all_observations: "list[dict]") -> dict:
    """Per-setup-label (SWING_SPEC.md Sec.3.8) mean excess / hit-rate table."""
    buckets = defaultdict(lambda: {"n": 0, **{f"excess_{h}": [] for h in FORWARD_HORIZONS}})
    for o in all_observations:
        setup = o.get("setup") or "—"
        bucket = buckets[setup]
        bucket["n"] += 1
        for h in FORWARD_HORIZONS:
            e = o.get(f"excess_{h}_pct")
            if e is not None:
                bucket[f"excess_{h}"].append(e)

    result = {}
    for setup, bucket in buckets.items():
        entry = {"n": bucket["n"]}
        for h in FORWARD_HORIZONS:
            vals = bucket[f"excess_{h}"]
            entry[f"mean_excess_{h}_pct"] = round(statistics.mean(vals), 3) if vals else None
            entry[f"hit_rate_{h}_pct"] = (
                round(100.0 * sum(1 for v in vals if v > 0) / len(vals), 1) if vals else None
            )
        result[setup] = entry
    return result


def run_swing_study(
    index: str = "SP500",
    start: "str | None" = None,
    end: "str | None" = None,
    tickers: "list[str] | None" = None,
    max_workers: "int | None" = None,
    progress_cb=None,
) -> dict:
    """Run the swing-score decile forward-return study (SWING_STUDY_SPEC.md).

    Args:
        index: Universe code, resolved via
            :func:`sec_analyzer.screener.universe.normalize_index`.
        start: ISO date lower bound for rebalance dates; ``None`` uses the
            earliest month-end in the SPY benchmark's history.
        end: ISO date upper bound; ``None`` uses the latest available bar.
        tickers: Restrict the study to these tickers instead of the full
            bundled universe.
        max_workers: Process/thread-pool size. ``None`` defaults to
            ``min(_DEFAULT_MAX_WORKERS, os.cpu_count())``.
        progress_cb: Optional ``progress_cb(done: int, total: int, ticker:
            str)``, called by the parent as each ticker's worker completes.
            Wrapped in its own try/except -- a broken callback cannot break
            the study. Never called from a worker process (not picklable).

    Returns:
        A dict matching SWING_STUDY_SPEC.md Sec.4: ``generated_at`` (the
        only wall-clock value), ``index``/``index_label``, ``start``/``end``,
        ``rebalance_dates``, ``observations``, ``tickers_scanned``,
        ``skipped`` (``{"ticker", "reason"}`` list), ``dropped_no_score``,
        ``dropped_no_forward``, ``deciles`` (keyed ``"10"``/``"21"``, each a
        list of ``{"decile", "n", "mean_excess_pct", "median_excess_pct",
        "hit_rate_pct", "mean_score", "low_sample"}``), ``spread`` (keyed by
        horizon, ``{"mean_excess_pct", "ci_low", "ci_high"}``),
        ``monotonicity`` (keyed by horizon, Spearman rho float),
        ``by_setup`` (keyed by SWING_SPEC.md Sec.3.8 setup label),
        ``limitations`` (Sec.1), and ``disclaimer`` (``BACKTEST_DISCLAIMER``).

        Never raises: a bad ticker lands in ``skipped``; a benchmark-fetch
        failure degrades relative-strength/excess-return computation rather
        than aborting the run. All values are JSON-native scalars.
    """
    resolved_index = normalize_index(index)

    try:
        universe = load_universe(index=resolved_index)
    except OSError:
        logger.warning("swing_study: could not load universe for %s", resolved_index, exc_info=True)
        universe = []

    if tickers is None:
        ticker_list = [row["ticker"] for row in universe]
    else:
        seen = set()
        ticker_list = []
        for raw in tickers:
            if not raw:
                continue
            t = str(raw).strip().upper()
            if t and t not in seen:
                seen.add(t)
                ticker_list.append(t)

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    bench_df = _get_benchmark_frame()
    rebalance_dates = (
        [] if bench_df is None or bench_df.empty
        else _month_end_trading_dates(bench_df.index, start_date, end_date)
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved_start = start_date or (rebalance_dates[0] if rebalance_dates else None)
    resolved_end = end_date or (rebalance_dates[-1] if rebalance_dates else None)

    all_observations: "list[dict]" = []
    skipped: "list[dict]" = []
    dropped_no_score = 0
    dropped_no_forward = 0
    total = len(ticker_list)
    done = 0

    if total and rebalance_dates:
        dates_iso = [d.isoformat() for d in rebalance_dates]
        tasks = [{"ticker": t, "dates": dates_iso} for t in ticker_list]
        workers = max_workers if max_workers is not None else min(_DEFAULT_MAX_WORKERS, (os.cpu_count() or 4))

        executor, _kind = _make_executor(workers)
        try:
            future_to_ticker = {executor.submit(_score_ticker_task, task): task["ticker"] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - one worker failure must never abort the study
                    logger.warning("swing_study: worker failed for %s", ticker, exc_info=True)
                    result = {
                        "ticker": ticker, "status": "skip", "reason": f"İşlem hatası: {exc}",
                        "observations": [], "dropped_no_score": 0, "dropped_no_forward": 0,
                    }

                if result.get("status") == "ok":
                    all_observations.extend(result.get("observations") or [])
                    dropped_no_score += result.get("dropped_no_score", 0)
                    dropped_no_forward += result.get("dropped_no_forward", 0)
                else:
                    skipped.append({"ticker": result.get("ticker"), "reason": result.get("reason")})

                done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(done, total, ticker)
                    except Exception:  # noqa: BLE001 - a broken progress callback must not break the study
                        logger.warning("swing_study progress_cb failed", exc_info=True)
        finally:
            executor.shutdown(wait=True)

    # Deterministic ordering regardless of completion order (SWING_STUDY_SPEC.md
    # Sec.6: "results are sorted deterministically after the fan-out, never in
    # completion order").
    all_observations.sort(key=lambda o: (o["date"], o["ticker"]))
    skipped.sort(key=lambda s: s.get("ticker") or "")

    obs_with_decile = _assign_deciles(all_observations)
    deciles, spread, monotonicity = _build_decile_tables(obs_with_decile)
    by_setup = _build_by_setup(all_observations)

    return {
        "generated_at": generated_at,
        "index": resolved_index,
        "index_label": universe_label(resolved_index),
        "start": resolved_start.isoformat() if resolved_start else None,
        "end": resolved_end.isoformat() if resolved_end else None,
        "rebalance_dates": len(rebalance_dates),
        "observations": len(all_observations),
        "tickers_scanned": total,
        "skipped": skipped,
        "dropped_no_score": dropped_no_score,
        "dropped_no_forward": dropped_no_forward,
        "deciles": deciles,
        "spread": spread,
        "monotonicity": monotonicity,
        "by_setup": by_setup,
        "limitations": list(_LIMITATIONS),
        "disclaimer": BACKTEST_DISCLAIMER,
    }
