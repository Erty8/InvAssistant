"""Validation study for the technical action (``technical/action.py``).

Binding, pre-registered contract: ``sec_analyzer/backtest/ACTION_STUDY_SPEC.md``.
For every monthly rebalance date and every sampled ticker, the technical
action is computed point-in-time (price and SPY both sliced to the date), and
the forward return over 10/21/63 trading days is recorded starting at the
next bar. Five pre-registered contrasts are then judged with a date-cluster
bootstrap.

EVALUATION, not optimization: nothing here searches thresholds. The
descriptive tables exist to show the shape of the data, not to pick values.
"""

import concurrent.futures
import json
import logging
import os
import random
import statistics
from collections import defaultdict
from datetime import date, datetime, timezone

from sec_analyzer.backtest import BACKTEST_DISCLAIMER
from sec_analyzer.backtest.swing_study import (
    _LIMITATIONS as _SWING_LIMITATIONS,
    _get_benchmark_frame,
    _make_executor,
    _month_end_trading_dates,
    BENCHMARK,
)
from sec_analyzer.fetch.prices import get_price_history, slice_asof
from sec_analyzer.screener.universe import load_universe, normalize_index, price_symbol, universe_label
from sec_analyzer.technical.action import compute_technical_action
from sec_analyzer.technical.indicators import compute_indicators, relative_strength
from sec_analyzer.technical.momentum import compute_price_momentum

logger = logging.getLogger(__name__)

#: Forward horizons in trading days (ACTION_STUDY_SPEC.md §2) -- all reported.
FORWARD_HORIZONS = (10, 21, 63)
#: Horizon the hypothesis verdicts are judged at.
VERDICT_HORIZON = 21
#: Action horizon passed to compute_technical_action (the app default).
ACTION_HORIZON = "1y"
#: Default rebalance start (ACTION_STUDY_SPEC.md §2).
DEFAULT_START = "2015-01-01"
#: Default deterministic subsample: every Nth ticker of the sorted universe.
DEFAULT_SAMPLE_EVERY = 4

_MIN_N = 30
_BOOTSTRAP_SEED = 0
_BOOTSTRAP_RESAMPLES = 1000

_EXT_BINS = ((None, 0.0, "<0"), (0.0, 1.0, "0–1"), (1.0, 2.0, "1–2"), (2.0, 3.0, "2–3"),
             (3.0, 4.0, "3–4"), (4.0, 5.0, "4–5"), (5.0, None, "≥5"))

_LIMITATIONS = list(_SWING_LIMITATIONS) + [
    "No sector-relative strength: momentum uses SPY-only relative strength "
    "(the live pipeline also folds in the sector ETF), so momentum scores may "
    "differ slightly from a live run.",
]

#: Pre-registered hypotheses (ACTION_STUDY_SPEC.md §3): id -> (description,
#: predicted sign, predicate A, predicate B). Predicates take one observation.
_UP = lambda o: o["regime"] == "up"  # noqa: E731


def _turn_ready(o):
    return o["regime"] != "up" and o["rev_confirmed"] is not None and o["rev_confirmed"] >= o["rev_needed"]


HYPOTHESES = {
    "H1": ("BUY vs REDUCE/AVOID", +1,
           lambda o: o["action"] == "BUY",
           lambda o: o["action"] in ("REDUCE", "AVOID")),
    "H2": ("Uptrend: fuel low (≥2 exhaustion signs) vs strong/moderate", -1,
           lambda o: _UP(o) and o["fuel_level"] == "low",
           lambda o: _UP(o) and o["fuel_level"] in ("strong", "moderate")),
    "H3": ("Uptrend: ≥3× ATR above SMA50 vs not", -1,
           lambda o: _UP(o) and o["checks"].get("extended") is True,
           lambda o: _UP(o) and o["checks"].get("extended") is False),
    "H4": ("Downtrend: BUY (reversal confirmed) vs AVOID", +1,
           lambda o: o["regime"] == "down" and o["action"] == "BUY",
           lambda o: o["regime"] == "down" and o["action"] == "AVOID"),
    "H5": ("Enough confirmations: with price turn vs volume-only", +1,
           lambda o: _turn_ready(o) and o["price_turn"] is True,
           lambda o: _turn_ready(o) and o["price_turn"] is False),
}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _observation(ticker, d_iso, ind, action) -> dict:
    fuel = action.get("fuel") or {}
    rev = action.get("reversal") or {}
    price, s50, atr = ind.get("price"), ind.get("sma50"), ind.get("atr14")
    ext_atr = None
    if _is_num(price) and _is_num(s50) and _is_num(atr) and atr > 0:
        ext_atr = round((price - s50) / atr, 3)
    return {
        "ticker": ticker, "date": d_iso,
        "action": action["action"], "regime": action["regime"],
        "fuel_level": fuel.get("level"), "fuel_fired": fuel.get("fired"),
        "checks": {c["key"]: c["fired"] for c in (fuel.get("checks") or [])},
        "rev_confirmed": rev.get("confirmed"), "rev_needed": rev.get("needed"),
        "price_turn": rev.get("price_turn"),
        "ext_atr": ext_atr,
    }


def _action_ticker_task(task: dict) -> dict:
    """One ticker across all rebalance dates. Module-level (picklable);
    never raises."""
    ticker = task["ticker"]
    try:
        df, _source = get_price_history(price_symbol(ticker))
    except Exception as exc:  # noqa: BLE001 - one bad ticker never aborts the study
        return {"ticker": ticker, "status": "skip", "reason": f"Could not fetch price data: {exc}",
                "observations": [], "dropped_no_action": 0, "dropped_no_forward": 0}

    spy_df = _get_benchmark_frame()
    n = len(df)
    obs, no_action, no_forward = [], 0, 0
    closes = df["Close"]
    for d_iso in task["dates"]:
        try:
            d = date.fromisoformat(d_iso)
            sliced = slice_asof(df, d)
            if sliced.empty:
                no_action += 1
                continue
            ind = compute_indicators(sliced)
            if spy_df is not None:
                spy_sliced = slice_asof(spy_df, d)
                ind["relative_strength"] = relative_strength(
                    sliced["Close"], spy_sliced["Close"] if not spy_sliced.empty else None, benchmark=BENCHMARK)
            ind["momentum"] = compute_price_momentum(ind)
            action = compute_technical_action(ind, ACTION_HORIZON)
            if action is None:
                no_action += 1
                continue
            pos = df.index.get_loc(sliced.index[-1])
            if isinstance(pos, slice):
                pos = pos.stop - 1
            rec = _observation(ticker, d_iso, ind, action)
            any_ok = False
            for h in FORWARD_HORIZONS:
                s, e = pos + 1, pos + 1 + h
                fwd = bench = None
                if e < n and float(closes.iloc[s]) > 0:
                    fwd = (float(closes.iloc[e]) / float(closes.iloc[s]) - 1.0) * 100.0
                    if spy_df is not None:
                        b0 = spy_df.loc[spy_df.index <= df.index[s], "Close"]
                        b1 = spy_df.loc[spy_df.index <= df.index[e], "Close"]
                        if len(b0) and len(b1) and float(b0.iloc[-1]) > 0:
                            bench = (float(b1.iloc[-1]) / float(b0.iloc[-1]) - 1.0) * 100.0
                    any_ok = True
                else:
                    no_forward += 1
                rec[f"fwd_{h}"] = round(fwd, 3) if fwd is not None else None
                rec[f"excess_{h}"] = round(fwd - bench, 3) if fwd is not None and bench is not None else None
            if any_ok:
                obs.append(rec)
        except Exception:  # noqa: BLE001 - one bad date never aborts the ticker
            logger.warning("action_study: failed %s at %s", ticker, d_iso, exc_info=True)
            no_action += 1
    return {"ticker": ticker, "status": "ok", "reason": None, "observations": obs,
            "dropped_no_action": no_action, "dropped_no_forward": no_forward}


# --- Aggregation -------------------------------------------------------------

def _add_demeaned(obs: list) -> None:
    """Attach ``rel_h`` = fwd_h minus the same-date mean fwd_h (primary metric)."""
    for h in FORWARD_HORIZONS:
        by_date = defaultdict(list)
        for o in obs:
            if o.get(f"fwd_{h}") is not None:
                by_date[o["date"]].append(o[f"fwd_{h}"])
        means = {d: statistics.mean(v) for d, v in by_date.items()}
        for o in obs:
            f = o.get(f"fwd_{h}")
            o[f"rel_{h}"] = round(f - means[o["date"]], 3) if f is not None else None


def _summary(values: list) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "hit_rate_pct": None, "low_sample": True}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "hit_rate_pct": round(100.0 * sum(1 for v in values if v > 0) / len(values), 1),
        "low_sample": len(values) < _MIN_N,
    }


def _contrast(obs: list, pred_a, pred_b, key: str) -> dict:
    """mean(A) - mean(B) of ``key`` with a date-cluster bootstrap interval."""
    per_date = defaultdict(lambda: [0.0, 0, 0.0, 0])  # sumA, nA, sumB, nB
    for o in obs:
        v = o.get(key)
        if v is None:
            continue
        if pred_a(o):
            c = per_date[o["date"]]; c[0] += v; c[1] += 1  # noqa: E702
        elif pred_b(o):
            c = per_date[o["date"]]; c[2] += v; c[3] += 1  # noqa: E702
    dates = sorted(per_date)
    tot = [sum(per_date[d][i] for d in dates) for i in range(4)]
    n_a, n_b = tot[1], tot[3]
    if n_a == 0 or n_b == 0:
        return {"n_a": n_a, "n_b": n_b, "diff": None, "ci_low": None, "ci_high": None}
    diff = tot[0] / n_a - tot[2] / n_b
    rng = random.Random(_BOOTSTRAP_SEED)
    diffs = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        sa = na = sb = nb = 0
        for _i in range(len(dates)):
            c = per_date[dates[rng.randrange(len(dates))]]
            sa += c[0]; na += c[1]; sb += c[2]; nb += c[3]  # noqa: E702
        if na and nb:
            diffs.append(sa / na - sb / nb)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))] if diffs else None
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))] if diffs else None
    return {"n_a": n_a, "n_b": n_b, "diff": round(diff, 3),
            "ci_low": round(lo, 3) if lo is not None else None,
            "ci_high": round(hi, 3) if hi is not None else None}


def _verdict(by_h: dict, sign: int) -> str:
    main = by_h[str(VERDICT_HORIZON)]
    if main["diff"] is None or main["n_a"] < _MIN_N or main["n_b"] < _MIN_N or main["ci_low"] is None:
        return "inconclusive"
    lo, hi = main["ci_low"], main["ci_high"]
    if (sign > 0 and hi < 0) or (sign < 0 and lo > 0):
        return "contradicted"
    excludes = lo > 0 if sign > 0 else hi < 0
    others_agree = all(
        by_h[str(h)]["diff"] is not None and by_h[str(h)]["diff"] * sign > 0
        for h in FORWARD_HORIZONS if h != VERDICT_HORIZON
    )
    return "supported" if excludes and others_agree else "inconclusive"


def _hypotheses(obs: list) -> dict:
    dates = sorted({o["date"] for o in obs})
    half = len(dates) // 2
    first, second = set(dates[:half]), set(dates[half:])
    out = {}
    for hid, (desc, sign, pa, pb) in HYPOTHESES.items():
        by_h = {str(h): _contrast(obs, pa, pb, f"rel_{h}") for h in FORWARD_HORIZONS}
        k = f"rel_{VERDICT_HORIZON}"
        stab = {
            "first_half": _contrast([o for o in obs if o["date"] in first], pa, pb, k)["diff"],
            "second_half": _contrast([o for o in obs if o["date"] in second], pa, pb, k)["diff"],
        }
        out[hid] = {"description": desc, "predicted_sign": sign, "by_horizon": by_h,
                    "stability_21d": stab, "verdict": _verdict(by_h, sign)}
    return out


def _table(obs: list, groups: list) -> list:
    """``groups`` = [(label, predicate)] -> per-group summaries for every
    horizon and metric."""
    rows = []
    for label, pred in groups:
        members = [o for o in obs if pred(o)]
        row = {"group": label}
        for h in FORWARD_HORIZONS:
            row[str(h)] = {
                "rel": _summary([o[f"rel_{h}"] for o in members if o.get(f"rel_{h}") is not None]),
                "excess": _summary([o[f"excess_{h}"] for o in members if o.get(f"excess_{h}") is not None]),
                "raw": _summary([o[f"fwd_{h}"] for o in members if o.get(f"fwd_{h}") is not None]),
            }
        rows.append(row)
    return rows


def _descriptive(obs: list) -> dict:
    actions = ["BUY", "HOLD", "WAIT FOR PULLBACK", "WAIT", "REDUCE", "AVOID"]
    by_action = _table(obs, [(a, (lambda a_: lambda o: o["action"] == a_)(a)) for a in actions])
    fuel = _table(obs, [
        ("0 signs", lambda o: _UP(o) and o["fuel_fired"] == 0),
        ("1 sign", lambda o: _UP(o) and o["fuel_fired"] == 1),
        ("2 signs", lambda o: _UP(o) and o["fuel_fired"] == 2),
        ("3+ signs", lambda o: _UP(o) and (o["fuel_fired"] or 0) >= 3),
    ])
    check_keys = sorted({k for o in obs if _UP(o) for k in o["checks"]})
    checks = []
    for k in check_keys:
        checks.extend(_table(obs, [
            (f"{k}: fired", (lambda k_: lambda o: _UP(o) and o["checks"].get(k_) is True)(k)),
            (f"{k}: not fired", (lambda k_: lambda o: _UP(o) and o["checks"].get(k_) is False)(k)),
        ]))

    def _in_bin(lo, hi):
        return lambda o: _UP(o) and o["ext_atr"] is not None and (lo is None or o["ext_atr"] >= lo) \
            and (hi is None or o["ext_atr"] < hi)
    ext = _table(obs, [(label, _in_bin(lo, hi)) for lo, hi, label in _EXT_BINS])
    return {"by_action": by_action, "fuel_gradient": fuel, "exhaustion_checks": checks, "extension_atr": ext}


def _load_checkpoint(path) -> dict:
    """Per-ticker worker results already written to ``path`` (JSON lines),
    keyed by ticker. Unreadable lines are ignored."""
    done = {}
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done[r["ticker"]] = r
            except (ValueError, KeyError, TypeError):
                continue
    return done


def run_action_study(index="SP500", start=DEFAULT_START, end=None, tickers=None,
                     sample_every=DEFAULT_SAMPLE_EVERY, max_workers=None, progress_cb=None,
                     checkpoint_path=None) -> dict:
    """Run the pre-registered action validation study (ACTION_STUDY_SPEC.md).
    Never raises for a bad ticker; returns a JSON-native dict.

    ``checkpoint_path``: optional JSON-lines file. Each finished ticker's
    worker result is appended as it completes, and tickers already present
    are not recomputed -- so an interrupted run resumes where it stopped.
    The checkpoint must come from a run with the same dates/configuration."""
    resolved_index = normalize_index(index)
    if tickers is None:
        try:
            universe = sorted(row["ticker"] for row in load_universe(index=resolved_index))
        except OSError:
            universe = []
        step = max(1, int(sample_every or 1))
        ticker_list = universe[::step]
    else:
        ticker_list = list(dict.fromkeys(str(t).strip().upper() for t in tickers if t))

    bench_df = _get_benchmark_frame()
    start_d = date.fromisoformat(start) if start else None
    end_d = date.fromisoformat(end) if end else None
    rebalance = [] if bench_df is None or bench_df.empty else _month_end_trading_dates(bench_df.index, start_d, end_d)
    dates_iso = [d.isoformat() for d in rebalance]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    obs, skipped, no_action, no_forward = [], [], 0, 0

    def _collect(r):
        nonlocal no_action, no_forward
        if r.get("status") == "ok":
            obs.extend(r["observations"])
            no_action += r["dropped_no_action"]
            no_forward += r["dropped_no_forward"]
        else:
            skipped.append({"ticker": r.get("ticker"), "reason": r.get("reason")})

    cached = _load_checkpoint(checkpoint_path)
    for t in ticker_list:
        if t in cached:
            _collect(cached[t])
    pending = [t for t in ticker_list if t not in cached]

    if pending and dates_iso:
        workers = max_workers or min(8, os.cpu_count() or 4)
        executor, _kind = _make_executor(workers)
        ckpt = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None
        try:
            futs = {executor.submit(_action_ticker_task, {"ticker": t, "dates": dates_iso}): t for t in pending}
            for done, fut in enumerate(concurrent.futures.as_completed(futs), len(ticker_list) - len(pending) + 1):
                t = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {"ticker": t, "status": "skip", "reason": f"Processing error: {exc}"}
                _collect(r)
                if ckpt is not None:
                    ckpt.write(json.dumps(r, ensure_ascii=False) + "\n")
                    ckpt.flush()
                if progress_cb is not None:
                    try:
                        progress_cb(done, len(ticker_list), t)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            executor.shutdown(wait=True)
            if ckpt is not None:
                ckpt.close()

    obs.sort(key=lambda o: (o["date"], o["ticker"]))
    skipped.sort(key=lambda s: s["ticker"] or "")
    _add_demeaned(obs)

    return {
        "generated_at": generated_at,
        "index": resolved_index, "index_label": universe_label(resolved_index),
        "sample_every": sample_every if tickers is None else None,
        "tickers_scanned": len(ticker_list),
        "start": dates_iso[0] if dates_iso else None, "end": dates_iso[-1] if dates_iso else None,
        "rebalance_dates": len(dates_iso),
        "observations": len(obs),
        "skipped": skipped,
        "dropped_no_action": no_action, "dropped_no_forward": no_forward,
        "action_horizon": ACTION_HORIZON,
        "verdict_horizon": VERDICT_HORIZON,
        "hypotheses": _hypotheses(obs),
        "descriptive": _descriptive(obs),
        "limitations": list(_LIMITATIONS),
        "disclaimer": BACKTEST_DISCLAIMER,
    }
