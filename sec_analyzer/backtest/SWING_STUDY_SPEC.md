# SWING_STUDY_SPEC.md — Swing score decile study (binding contract)

Status: **binding**. Code that contradicts this file is wrong.

Purpose: answer one question, with the fewest assumptions possible —
**does a higher swing score actually predict a higher forward return?**

This is Stage 1 of the swing backtest. It is a *cross-sectional ranking study*,
not a trade simulator: no entries, no stops, no targets, no equity curve. If the
score shows no ranking gradient here, no trade simulation would rescue it, so
this comes first.

Governing principle, inherited from `sec_analyzer/ROADMAP.md`
("Backtest — tasarım ilkesi") and `sec_analyzer/backtest/__init__.py`: a
backtest here is an **EVALUATION** tool, never an **OPTIMIZATION** tool.

- The swing weights in `technical/swing.py` **must not** be tuned against this
  study's output. Note the pre-existing comment in `swing.py`/`momentum.py`
  claiming "the backtest layer calibrates these" — that claim contradicts
  ROADMAP.md and is **not** licence to sweep parameters here.
- No parameter search, no threshold optimization, no Sharpe maximization.
- Horizons and the rebalance frequency are fixed by this spec up front (§2), and
  **every** configured horizon is reported. Reporting only the best-looking
  horizon is forbidden.

---

## 1. Known limitations that MUST be surfaced in every output

Alongside `backtest.BACKTEST_DISCLAIMER` (auto-appended, as for every other
backtest output), each study result carries an explicit
`limitations: list[str]` naming at minimum:

1. **Survivorship bias (dominant).** The universe is *today's* index
   membership (`data/sp500.csv` / `data/nasdaq100.csv`), not point-in-time
   membership. Names that were dropped from the index or delisted are absent, so
   the sample is biased toward survivors and every return figure is optimistic.
   Reconstructing point-in-time membership is future work, deliberately not in
   this stage.
2. **Free-source price data.** Delisted tickers have no Stooq/yfinance history
   at all, compounding (1).
3. **No costs.** Returns are gross — no commission, no slippage, no borrow.
4. **Overlapping observations.** Consecutive rebalance dates share price bars,
   so observations are not independent; treat any confidence interval as
   indicative, not a formal significance test.

These are not footnotes to bury: a caller reading only the summary must see
them.

---

## 2. Fixed configuration

| knob | value | note |
|---|---|---|
| rebalance frequency | **monthly** — the last trading bar of each calendar month | keeps the compute budget feasible (§6); date sampling is statistically legitimate for a ranking study |
| forward horizons | **10 and 21 trading days** | both always reported |
| benchmark | `SPY` | same bars as the measured window |
| return type | simple close-to-close, in percent | |
| decile count | 10, assigned **within each rebalance date** | cross-sectional; ranking across dates would blend regimes |

A caller may narrow the date range or universe, but must not add or drop
horizons — they are part of the contract.

---

## 3. Point-in-time correctness (the part that silently invalidates everything)

Non-negotiable:

1. **Slice before computing.** For rebalance date `d`, the indicator set is
   computed on `slice_asof(price_df, d)` — never on the full frame. Computing
   `compute_indicators` on the whole history leaks the future into the 52-week
   high, SMA200, support/resistance levels and the Bollinger percentile, which
   turns the study into fiction.
2. **The benchmark is sliced the same way.** Relative strength at date `d` uses
   SPY sliced to `d`.
3. **No same-bar measurement.** The score is derived from bar `d`'s close, so
   the forward window starts at the **next** bar: return is measured from the
   close of bar `d+1` to the close of bar `d+1+h`. A window starting at `d`'s
   own close would credit the signal with information it did not have.
4. **Insufficient forward data ⇒ drop the observation**, never pad or
   extrapolate. Observations dropped for this reason are counted and reported
   (`dropped_no_forward`).
5. A ticker whose score is `None` at date `d` (coverage floor, §3.0 of
   SWING_SPEC.md) simply yields no observation for that date — counted as
   `dropped_no_score`.

---

## 4. Module — `sec_analyzer/backtest/swing_study.py`

```python
FORWARD_HORIZONS = (10, 21)      # trading days; fixed by §2
DECILES = 10
BENCHMARK = "SPY"

def run_swing_study(
    index: str = "SP500",
    start: str | None = None,      # ISO date; None = earliest usable
    end: str | None = None,        # ISO date; None = latest available bar
    tickers: list[str] | None = None,
    max_workers: int | None = None,
    progress_cb=None,
) -> dict
```

Parallelization unit is **one ticker across all rebalance dates** — each worker
loads that ticker's price frame once from the on-disk cache and slices it per
date. This maximizes frame reuse and keeps inter-process traffic to small
dicts. Uses the same process-pool helper the scan path uses (see §6); never
raises for one bad ticker (recorded in `skipped`, exactly like `scan_swing`).

Per-observation record (internal):

```python
{"ticker", "date", "score", "setup", "fwd_10_pct", "fwd_21_pct",
 "bench_10_pct", "bench_21_pct", "excess_10_pct", "excess_21_pct"}
```

Return value:

```python
{
  "generated_at": str,          # ISO-8601 UTC seconds — the ONLY wall-clock value
  "index": str, "index_label": str,
  "start": str, "end": str,
  "rebalance_dates": int,
  "observations": int,
  "tickers_scanned": int,
  "skipped": list[dict],                 # {"ticker", "reason"}
  "dropped_no_score": int,
  "dropped_no_forward": int,
  "deciles": {                           # keyed by horizon, "10" / "21"
     "10": [ {"decile": 1..10, "n": int, "mean_excess_pct": float,
              "median_excess_pct": float, "hit_rate_pct": float,
              "mean_score": float} , ... ],
     "21": [ ... ],
  },
  "spread": {                            # top decile minus bottom decile
     "10": {"mean_excess_pct": float, "ci_low": float, "ci_high": float},
     "21": {...},
  },
  "monotonicity": {"10": float, "21": float},   # Spearman rho of decile vs mean excess
  "by_setup": {                          # per SWING_SPEC.md Sec.3.8 setup label
     "BREAKOUT": {"n": int, "mean_excess_10_pct": float, "mean_excess_21_pct": float,
                  "hit_rate_10_pct": float, "hit_rate_21_pct": float},
     ...
  },
  "limitations": list[str],              # §1
  "disclaimer": BACKTEST_DISCLAIMER,
}
```

`spread` confidence bounds come from a **fixed-seed** bootstrap
(`random.Random(0)`, 1000 resamples) so the result stays deterministic — the
project's determinism rule applies to the study too: same price inputs ⇒ byte
-identical output apart from `generated_at`.

All values JSON-native (no numpy scalars — they break `json.dumps`).

Cells with `n < 30` must be flagged, not silently reported as meaningful: each
decile entry gains `"low_sample": bool` (`n < 30`), mirroring the existing
`backtest report`'s "yetersiz örneklem" convention.

---

## 5. Persistence + CLI

Reuse the existing pattern. New table, created in `init_db()`:

```sql
CREATE TABLE IF NOT EXISTS swing_studies (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    universe     TEXT NOT NULL,
    start_date   TEXT,
    end_date     TEXT,
    observations INTEGER NOT NULL,
    payload      TEXT NOT NULL          -- the full §4 dict as JSON
)
```

with `save_swing_study(result, db_path=None) -> int` and
`load_latest_swing_study(db_path=None, universe=None) -> dict | None`,
following `save_swing_scan`/`load_latest_swing_scan`'s non-fatal posture
exactly.

CLI:

```
python -m sec_analyzer.cli backtest swing [--index sp500|ndx] [--start YYYY-MM-DD]
                                          [--end YYYY-MM-DD] [--workers N] [--no-save]
```

Prints, in Turkish: a header naming the index and date range; one decile table
per horizon (decile, n, ortalama fazla getiri %, medyan, isabet %, ortalama
skor, plus a `yetersiz örneklem` marker where `low_sample`); the top-minus-
bottom spread with its interval; the monotonicity figure; the per-setup table;
then the limitations list and `BACKTEST_DISCLAIMER`. Never raises out to the
user: a failure prints an error line and exits 1.

---

## 6. Compute budget + the process-pool prerequisite

Measured reality: `compute_indicators` costs ~1s per ticker and is pure-Python
CPU work, so `ThreadPoolExecutor` cannot parallelize it (GIL). Monthly
rebalances over ~8 years × 503 names ≈ 48k indicator computations ≈ 13 hours
single-core — not viable.

Therefore `screener/swing_scan.py`'s fan-out moves to
`concurrent.futures.ProcessPoolExecutor`, and this study uses the same
mechanism:

- Worker functions must be module-level and picklable; only small dicts cross
  the process boundary (workers read price frames from the on-disk cache
  themselves).
- The benchmark frame is loaded **once per worker process** via a lazy
  module-level cache, not passed per task.
- `progress_cb` is NOT picklable — progress is emitted by the **parent** as
  futures complete, exactly as now.
- A process pool that cannot start (restricted environment, frozen app) falls
  back to the existing thread path with a logged warning. Behaviour and results
  must be identical either way.
- Determinism is unaffected: results are sorted deterministically after the
  fan-out, never in completion order.

Expected: ~4-6x on a multi-core box, bringing the study to a ~2-hour background
run and a full S&P 500 scan to ~1-2 minutes.

---

## 7. Invariants (test targets)

1. Point-in-time: a study run whose price frames are stubbed so that all bars
   after date `d` are extreme outliers produces the **same** score at `d` as a
   direct `compute_indicators(slice_asof(df, d))` call — proving no future leak.
2. The forward window starts at `d+1`, not `d`: with a synthetic frame where
   bar `d+1` jumps, the observation's `fwd_10_pct` reflects that jump measured
   from `d+1`'s close, not from `d`'s.
3. Deciles are assigned within a date: two dates with disjoint score ranges
   each get a full 1..10 spread, not one date filling the top deciles.
4. An observation lacking `h` forward bars is dropped and counted in
   `dropped_no_forward`; a `None` score is counted in `dropped_no_score`.
5. Determinism: two runs over identical stubbed inputs return equal dicts apart
   from `generated_at`, bootstrap bounds included.
6. `json.dumps` round-trips the whole result.
7. Every result carries a non-empty `limitations` naming survivorship bias, and
   `disclaimer == BACKTEST_DISCLAIMER`.
8. `low_sample` is `True` exactly when a decile's `n < 30`.
9. `save_swing_study`/`load_latest_swing_study` round-trip through a temp DB;
   per-universe scoping works like the scan table's; empty DB returns `None`.
10. Process-pool path and thread-fallback path produce identical results for the
    same stubbed inputs.
