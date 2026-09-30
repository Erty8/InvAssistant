"""Price-factor study v1: do published price factors rank S&P 500 stocks?

Binding, pre-registered contract: ``sec_analyzer/backtest/FACTOR_STUDY_SPEC.md``.
Four price factors (12-1 momentum, 1-month reversal, 52-week-high proximity,
low volatility) and one fixed composite are computed per ticker with
vectorized pandas on closes up to each month-end, stocks are sorted into
quintiles within each date, and the best-minus-worst quintile spread of
forward returns is judged with a moving-block bootstrap on the
**confirmation** sample (tickers the earlier action study never touched).

EVALUATION, not optimization: the lookbacks, directions, horizons, and cost
below are fixed by the spec and never searched.
"""

import logging
import math
import random
import statistics
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from sec_analyzer.backtest import BACKTEST_DISCLAIMER
from sec_analyzer.backtest.swing_study import _get_benchmark_frame, _month_end_trading_dates
from sec_analyzer.fetch.prices import get_price_history, is_total_return_basis, load_cached_history
from sec_analyzer.fetch.sp500_history import load_membership_history
from sec_analyzer.screener.universe import load_universe, price_symbol

logger = logging.getLogger(__name__)

# --- Fixed configuration (FACTOR_STUDY_SPEC.md §2-3) -------------------------
DEFAULT_START = "2005-01-01"
SUBPERIOD_SPLIT = "2015-01-01"
FORWARD_HORIZONS = (21, 63, 126)
QUINTILES = 5
MIN_BARS = 253
MIN_CROSS_SECTION = 100
DISCOVERY_EVERY = 4
COST_BP_PER_SIDE = 10.0
BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 2000
#: A ticker's last bar must be at most this stale relative to the rebalance
#: date, or the ticker is not eligible that month (halted / delisted).
_MAX_STALENESS = pd.Timedelta(days=5)

#: id -> (label, direction, primary horizon). direction +1: highest value is
#: predicted best; -1: lowest value is predicted best.
FEATURES = {
    "F1": ("12-1 momentum", +1, 63),
    "F2": ("1-month reversal", -1, 21),
    "F3": ("52-week-high proximity", +1, 63),
    "F4": ("Low volatility", -1, 63),
    "F5": ("Composite (F1, F3, F2 ranks)", +1, 63),
}
_COLUMNS = {"F1": "mom_12_1", "F2": "ret_1m", "F3": "high52", "F4": "vol_63"}

LIMITATIONS = [
    "Survivorship bias (dominant, can flip signs): the universe is today's S&P 500 membership, "
    "so every name is one that later survived and grew into the index. Past losers, volatile "
    "names and stocks far below their highs are over-represented by their eventual winners, which "
    "biases results AGAINST momentum, 52-week-high and low-volatility factors "
    "(see FACTOR_STUDY_SPEC.md §9).",
    "Free price data: yfinance/Yahoo adjusted closes; tickers with split-only (non total-return) "
    "series are excluded.",
    "Costs are a flat 10 bp per side: no market impact and no borrow cost for the short leg. "
    "The short leg is a measurement device, not a recommendation.",
    "Overlapping 63/126-day windows are handled by a moving-block bootstrap, but intervals "
    "remain approximate.",
    "One market, one era: large US caps, 2005 onward.",
]


PIT_LIMITATIONS = [
    "Residual survivorship bias: Yahoo serves current listings only, so members that were later "
    "delisted or renamed have no price history and are missing on the dates they were in the "
    "index. See the coverage figures for how large that gap is per year.",
    "Forward windows that run past a stock's last bar (merger, bankruptcy) are dropped; "
    "acquisitions (usually positive) and failures (negative) are both lost.",
    "Ticker recycling: Yahoo returns the CURRENT holder of a symbol. If a former member's symbol "
    "now belongs to a company whose Yahoo history reaches back to the old membership dates, its "
    "prices are wrongly attributed. Requiring a bar at each date removes most such cases; the "
    "number of covered symbols that are not current members shows the exposure.",
    "Membership comes from a community-maintained dataset (fja05680/sp500, MIT) built from "
    "Trading Evolved's list plus Wikipedia changes; its accuracy is not independently verified.",
] + [
    "Costs are a flat 10 bp per side: no market impact and no borrow cost for the short leg. "
    "The short leg is a measurement device, not a recommendation.",
    "Overlapping 63/126-day windows are handled by a moving-block bootstrap, but intervals "
    "remain approximate.",
    "One market, one era: large US caps, 2005 onward.",
]


def _ticker_features(close: pd.Series) -> pd.DataFrame:
    """All per-bar features and forward returns for one ticker's close series.

    Every feature at bar i uses closes at positions <= i only; forward returns
    start at bar i+1 (FACTOR_STUDY_SPEC.md §2)."""
    c = close.astype(float)
    out = pd.DataFrame(index=c.index)
    out["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    out["ret_1m"] = c / c.shift(21) - 1.0
    out["high52"] = c / c.rolling(252).max()
    out["vol_63"] = np.log(c).diff().rolling(63).std()
    out["bars"] = np.arange(1, len(c) + 1)
    for h in FORWARD_HORIZONS:
        out[f"fwd_{h}"] = (c.shift(-(1 + h)) / c.shift(-1) - 1.0) * 100.0
    out["bar_date"] = c.index
    return out


def _sample_at_dates(feat: pd.DataFrame, dates: "list[pd.Timestamp]") -> pd.DataFrame:
    """Rows of ``feat`` as of each rebalance date (last bar on/before it),
    dropping dates where that bar is too stale or history is too short."""
    rows = feat.reindex(dates, method="ffill")
    rows.index.name = "date"
    fresh = (rows.index - pd.to_datetime(rows["bar_date"])) <= _MAX_STALENESS
    ok = fresh & (rows["bars"] >= MIN_BARS) & rows["mom_12_1"].notna()
    return rows[ok]


def _network_loader(ticker: str):
    return get_price_history(price_symbol(ticker))


def _cache_loader(ticker: str):
    got = load_cached_history(price_symbol(ticker))
    if got is None:
        raise LookupError("no cached price history")
    return got


def _load_panel(tickers: "list[str]", dates, loader=_network_loader) -> "tuple[pd.DataFrame, list, list]":
    """Stack every eligible ticker's month-end rows into one long frame."""
    frames, failed, excluded = [], [], []
    for t in tickers:
        try:
            df, source = loader(t)
        except Exception as exc:  # noqa: BLE001 - one bad ticker never aborts the study
            failed.append({"ticker": t, "reason": str(exc)[:200]})
            continue
        if not is_total_return_basis(source):
            excluded.append({"ticker": t, "source": source})
            continue
        close = df["Close"]
        close = close[close > 0]
        if len(close) < MIN_BARS:
            failed.append({"ticker": t, "reason": "insufficient history"})
            continue
        rows = _sample_at_dates(_ticker_features(close), dates)
        if rows.empty:
            continue
        rows = rows.drop(columns=["bar_date", "bars"]).reset_index()
        rows["ticker"] = t
        frames.append(rows)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel, failed, excluded


def _add_composite(panel: pd.DataFrame) -> pd.DataFrame:
    """F5 = mean within-date percentile rank of F1 (high), F3 (high), F2 (low)."""
    g = panel.groupby("date")
    r1 = g["mom_12_1"].rank(pct=True)
    r3 = g["high52"].rank(pct=True)
    r2 = g["ret_1m"].rank(pct=True, ascending=False)
    panel = panel.copy()
    panel["composite"] = (r1 + r3 + r2) / 3.0
    return panel


def _quintiles(panel: pd.DataFrame, col: str, direction: int) -> pd.Series:
    """Quintile 1..5 per row within its date; 5 = predicted best. Ties broken
    by ticker for determinism."""
    score = panel[col] * direction
    ordered = panel.assign(_s=score).sort_values(["date", "_s", "ticker"])
    ranks = ordered.groupby("date").cumcount()
    sizes = ordered.groupby("date")["ticker"].transform("size")
    q = (ranks * QUINTILES // sizes) + 1
    return q.reindex(panel.index)


def _block_bootstrap_ci(series: "list[float]", block: int) -> "tuple[float | None, float | None]":
    n = len(series)
    if n < 2:
        return None, None
    block = max(1, min(block, n))
    rng = random.Random(BOOTSTRAP_SEED)
    k = math.ceil(n / block)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = []
        for _b in range(k):
            s = rng.randrange(0, n - block + 1)
            sample.extend(series[s:s + block])
        means.append(sum(sample[:n]) / n)
    means.sort()
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def _r(v, nd=3):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def _spread_series(panel: pd.DataFrame, q: pd.Series, h: int) -> pd.Series:
    """Per-date best-minus-worst quintile mean forward return (pp)."""
    key = f"fwd_{h}"
    sub = panel.loc[panel[key].notna(), ["date", key]].assign(q=q[panel[key].notna()])
    means = sub.groupby(["date", "q"])[key].mean().unstack()
    if QUINTILES not in means or 1 not in means:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = (means[QUINTILES] - means[1]).dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _turnover_cost(panel: pd.DataFrame, q: pd.Series) -> "float | None":
    """Mean monthly round-trip cost (pp) of the best and worst quintiles."""
    costs = []
    prev = None
    for d, grp in panel.assign(q=q).groupby("date", sort=True):
        cur = {5: set(grp.loc[grp["q"] == QUINTILES, "ticker"]), 1: set(grp.loc[grp["q"] == 1, "ticker"])}
        if prev is not None and cur[5] and cur[1]:
            to = sum(1.0 - len(cur[k] & prev[k]) / len(cur[k]) for k in (5, 1))
            costs.append(to * 2.0 * COST_BP_PER_SIDE / 100.0)
        prev = cur
    return statistics.mean(costs) if costs else None


def _quintile_table(panel: pd.DataFrame, q: pd.Series, h: int) -> "list[dict]":
    key = f"fwd_{h}"
    sub = panel.loc[panel[key].notna(), ["date", key]].assign(q=q[panel[key].notna()])
    sub["rel"] = sub[key] - sub.groupby("date")[key].transform("mean")
    rows = []
    for qi in range(1, QUINTILES + 1):
        m = sub[sub["q"] == qi]
        if m.empty:
            rows.append({"quintile": qi, "n": 0, "mean_rel_pct": None, "mean_raw_pct": None,
                         "hit_rate_pct": None, "stdev_raw_pct": None, "return_per_stdev": None})
            continue
        per_date_rel = m.groupby("date")["rel"].mean()
        per_date_raw = m.groupby("date")[key].mean()
        sd = float(m[key].std()) if len(m) > 1 else None
        rows.append({
            "quintile": qi, "n": int(len(m)),
            "mean_rel_pct": _r(per_date_rel.mean()),
            "mean_raw_pct": _r(per_date_raw.mean()),
            "hit_rate_pct": _r(100.0 * (m["rel"] > 0).mean(), 1),
            "stdev_raw_pct": _r(sd),
            "return_per_stdev": _r(float(m[key].mean()) / sd) if sd else None,
        })
    return rows


def _evaluate(panel: pd.DataFrame, with_verdict: bool) -> dict:
    split = pd.Timestamp(SUBPERIOD_SPLIT)
    out = {}
    for fid, (label, direction, primary) in FEATURES.items():
        col = _COLUMNS.get(fid, "composite")
        fp = panel[panel[col].notna()]
        q = _quintiles(fp, col, direction)
        by_h, tables = {}, {}
        for h in FORWARD_HORIZONS:
            s = _spread_series(fp, q, h)
            vals = [float(v) for v in s.values]
            lo, hi = _block_bootstrap_ci(vals, math.ceil(h / 21))
            early = s[s.index < split]
            late = s[s.index >= split]
            by_h[str(h)] = {
                "mean_spread_pct": _r(statistics.mean(vals)) if vals else None,
                "ci_low": _r(lo), "ci_high": _r(hi), "n_dates": len(vals),
                "sub_2005_2014": _r(early.mean()) if len(early) else None,
                "sub_2015_end": _r(late.mean()) if len(late) else None,
            }
            tables[str(h)] = _quintile_table(fp, q, h)
        cost = _turnover_cost(fp, q)
        gross21 = by_h["21"]["mean_spread_pct"]
        net21 = gross21 - cost if gross21 is not None and cost is not None else None
        entry = {
            "label": label, "direction": "highest" if direction > 0 else "lowest",
            "primary_horizon": primary, "by_horizon": by_h, "quintiles": tables,
            "costs": {"mean_monthly_cost_pct": _r(cost), "gross_21d_spread_pct": gross21,
                      "net_21d_spread_pct": _r(net21), "survives_costs": bool(net21 is not None and net21 > 0)},
        }
        if with_verdict:
            entry["verdict"] = _verdict(by_h[str(primary)])
        out[fid] = entry
    return out


def _verdict(m: dict) -> str:
    """FACTOR_STUDY_SPEC.md §4. Spreads are already best-minus-worst, so the
    predicted side is always > 0."""
    lo, hi = m["ci_low"], m["ci_high"]
    if lo is None or hi is None:
        return "inconclusive"
    if hi < 0:
        return "contradicted"
    subs_ok = all(v is not None and v > 0 for v in (m["sub_2005_2014"], m["sub_2015_end"]))
    return "supported" if lo > 0 and subs_ok else "inconclusive"


def _thin_dates(panel: pd.DataFrame) -> "tuple[pd.DataFrame, int]":
    counts = panel.groupby("date")["ticker"].transform("size")
    dropped = int(panel.loc[counts < MIN_CROSS_SECTION, "date"].nunique())
    return panel[counts >= MIN_CROSS_SECTION], dropped


#: v2 secondary analysis keeps only dates with at least this coverage.
PIT_MIN_COVERAGE = 0.90


def _matured(dates: "list", calendar: pd.DatetimeIndex, h: int) -> "set":
    """Rebalance dates with at least h+1 market bars after them (SPY calendar):
    a missing forward return on such a date means the stock stopped trading."""
    out = set()
    for d in dates:
        pos = calendar.searchsorted(pd.Timestamp(d), side="right")
        if len(calendar) - pos >= h + 1:
            out.add(pd.Timestamp(d))
    return out


def _run_pit(dates_ts, bench, start_d, end_d, membership) -> dict:
    """FACTOR_STUDY_SPEC.md §10: point-in-time universe, offline cache only."""
    tickers = sorted(membership.all_tickers(start_d, end_d))
    panel, failed, excluded = _load_panel(tickers, dates_ts, loader=_cache_loader)
    members_n = {d: len(membership.members_on(d.date())) for d in dates_ts}
    if not panel.empty:
        is_member = [t in membership.members_on(d.date()) for d, t in zip(panel["date"], panel["ticker"])]
        panel = panel[is_member]
    covered = panel.groupby("date")["ticker"].size() if not panel.empty else pd.Series(dtype=int)
    coverage = {d: (int(covered.get(d, 0)) / members_n[d]) if members_n[d] else 0.0 for d in dates_ts}

    by_year: dict = {}
    for d, c in coverage.items():
        by_year.setdefault(d.year, []).append(c)
    cov_summary = {
        "mean": _r(statistics.mean(coverage.values())) if coverage else None,
        "by_year": {str(y): _r(statistics.mean(v)) for y, v in sorted(by_year.items())},
        "dates_at_or_above_90pct": sum(1 for c in coverage.values() if c >= PIT_MIN_COVERAGE),
    }

    dropped = {}
    cal = bench.index if bench is not None else pd.DatetimeIndex([])
    for h in FORWARD_HORIZONS:
        mat = _matured(dates_ts, cal, h)
        rows = panel[panel["date"].isin(mat)] if not panel.empty else panel
        n = len(rows)
        dropped[str(h)] = _r(100.0 * rows[f"fwd_{h}"].isna().mean(), 2) if n else None

    current = membership.members_on(membership.last_date) if membership.last_date else frozenset()
    covered_symbols = set(panel["ticker"]) if not panel.empty else set()
    out = {
        "covered_symbols": len(covered_symbols),
        "covered_symbols_not_current_members": len(covered_symbols - current),
        "membership_sha256": membership.sha256,
        "membership_last_date": membership.last_date.isoformat() if membership.last_date else None,
        "tickers_ever_members": len(tickers),
        "tickers_without_cached_prices": len(failed),
        "excluded_non_total_return": excluded,
        "coverage": cov_summary,
        "forward_dropped_pct_of_matured_rows": dropped,
    }
    for name, cov_min in (("primary", 0.0), ("secondary", PIT_MIN_COVERAGE)):
        keep = {d for d, c in coverage.items() if c >= cov_min}
        sub = panel[panel["date"].isin(keep)] if not panel.empty else panel
        if sub.empty:
            out[name] = None
            continue
        sub, thin = _thin_dates(sub)
        if sub.empty:
            out[name] = None
            continue
        sub = _add_composite(sub)
        out[name] = {
            "min_coverage": cov_min,
            "tickers": int(sub["ticker"].nunique()),
            "observations": int(len(sub)),
            "dates_used": int(sub["date"].nunique()),
            "dates_dropped_thin": thin,
            "factors": _evaluate(sub, with_verdict=True),
        }
    if out.get("primary") and out.get("secondary"):
        for fid, f in out["primary"]["factors"].items():
            f["robust"] = f["verdict"] == out["secondary"]["factors"][fid]["verdict"]
    return out


def run_factor_study(start: "str | None" = DEFAULT_START, end: "str | None" = None,
                     tickers: "list[str] | None" = None, pit: bool = False,
                     membership=None) -> dict:
    """Run the pre-registered factor study (FACTOR_STUDY_SPEC.md). Never
    raises for a bad ticker; returns a JSON-native dict.

    ``pit=True`` runs v2 (§10): point-in-time membership, offline price cache
    only, plus the v1 confirmation spreads for comparison. ``membership`` is
    injectable for tests; otherwise it is loaded from the local copy."""
    if pit:
        return _run_factor_study_pit(start, end, membership)
    return _run_factor_study_v1(start, end, tickers)


def _run_factor_study_pit(start, end, membership) -> dict:
    bench = _get_benchmark_frame()
    start_d = date.fromisoformat(start) if start else None
    end_d = date.fromisoformat(end) if end else None
    dates = [] if bench is None or bench.empty else _month_end_trading_dates(bench.index, start_d, end_d)
    dates_ts = [pd.Timestamp(d) for d in dates]
    result = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": "v2-pit",
        "universe": "S&P 500, point-in-time membership",
        "start": dates[0].isoformat() if dates else None,
        "end": dates[-1].isoformat() if dates else None,
        "rebalance_dates": len(dates),
        "cost_bp_per_side": COST_BP_PER_SIDE,
        "limitations": list(PIT_LIMITATIONS),
        "disclaimer": BACKTEST_DISCLAIMER,
    }
    try:
        membership = membership or load_membership_history()
    except Exception as exc:  # noqa: BLE001 - a missing data file must not crash the CLI
        logger.warning("factor_study: membership history unavailable", exc_info=True)
        result["error"] = f"Membership history unavailable: {exc}"
        return result
    result["pit"] = _run_pit(dates_ts, bench, start_d, end_d, membership) if dates_ts else None

    # Diagnostic (§10): v1's confirmation spreads on the same dates, from the
    # same offline cache, for a like-for-like comparison.
    v1 = _run_factor_study_v1(start, end, None, loader=_cache_loader)
    conf = v1.get("confirmation") or {}
    result["v1_confirmation_primary_spreads"] = {
        fid: f["by_horizon"][str(f["primary_horizon"])]["mean_spread_pct"]
        for fid, f in (conf.get("factors") or {}).items()
    }
    return result


def _run_factor_study_v1(start, end, tickers, loader=_network_loader) -> dict:
    if tickers is None:
        try:
            tickers = sorted(r["ticker"] for r in load_universe(index="SP500"))
        except OSError:
            tickers = []
    else:
        tickers = sorted(dict.fromkeys(str(t).strip().upper() for t in tickers if t))
    discovery = set(tickers[::DISCOVERY_EVERY])

    bench = _get_benchmark_frame()
    start_d = date.fromisoformat(start) if start else None
    end_d = date.fromisoformat(end) if end else None
    dates = [] if bench is None or bench.empty else _month_end_trading_dates(bench.index, start_d, end_d)
    dates_ts = [pd.Timestamp(d) for d in dates]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    panel, failed, excluded = _load_panel(tickers, dates_ts, loader) if dates_ts else (pd.DataFrame(), [], [])
    result = {
        "generated_at": generated_at,
        "universe": "S&P 500 (today's membership)",
        "start": dates[0].isoformat() if dates else None,
        "end": dates[-1].isoformat() if dates else None,
        "rebalance_dates": len(dates),
        "tickers": len(tickers),
        "failed": failed, "excluded_non_total_return": excluded,
        "cost_bp_per_side": COST_BP_PER_SIDE,
        "limitations": list(LIMITATIONS),
        "disclaimer": BACKTEST_DISCLAIMER,
    }
    if panel.empty:
        result.update({"confirmation": None, "discovery": None})
        return result

    for name, is_disc in (("confirmation", False), ("discovery", True)):
        sub = panel[panel["ticker"].isin(discovery) == is_disc]
        sub, thin = _thin_dates(sub) if not is_disc else (sub, 0)
        sub = _add_composite(sub)
        result[name] = {
            "tickers": int(sub["ticker"].nunique()),
            "observations": int(len(sub)),
            "dates_used": int(sub["date"].nunique()),
            "dates_dropped_thin": thin,
            "factors": _evaluate(sub, with_verdict=not is_disc),
        }
    return result
