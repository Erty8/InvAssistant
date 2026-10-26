"""Technical snapshot of every thesis ticker, from the cached daily prices.

Runs the engine's own technical layer (indicators, verdict, momentum, swing,
action) on sec_analyzer/raw/prices_<TICKER>_full.csv and writes
technical.json next to this script. Run from the repo root:

    PYTHONPATH=. python thesis/_build/technical_stats.py [--live]

``--live`` asks the engine's price fetcher for today's in-progress bar
(``prefer_live=True``) while a US session is open; without it the snapshot
uses the settled daily closes in the cache.
"""
import json
import os
import sys

import pandas as pd

from sec_analyzer.fetch.prices import get_price_history
from sec_analyzer.technical.action import compute_technical_action
from sec_analyzer.technical.indicators import compute_indicators, relative_strength
from sec_analyzer.technical.momentum import compute_price_momentum
from sec_analyzer.technical.swing import compute_swing_score
from sec_analyzer.technical.verdict import technical_verdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "sec_analyzer/raw"
TICKERS = [d["t"] for d in json.load(open(os.path.join(HERE, "new_theses.json"), encoding="utf-8"))]
TICKERS = ["MU", "NVDA", "NFLX", "APP", "UBER"] + TICKERS


LIVE = "--live" in sys.argv


def prices(t):
    if LIVE:
        return get_price_history(t, prefer_live=True)[0].sort_index()
    df = pd.read_csv(os.path.join(RAW, f"prices_{t}_full.csv"), parse_dates=["Date"]).set_index("Date")
    return df.sort_index()


spy = prices("SPY")["Close"]
out = {}
for t in TICKERS:
    df = prices(t)
    ind = compute_indicators(df)
    ind["relative_strength"] = relative_strength(df["Close"], spy)
    verdict = technical_verdict(ind, horizon="1y") or {}
    mom = compute_price_momentum(ind) or {}
    swing = compute_swing_score(ind) or {}
    action = compute_technical_action(ind, horizon="1y") or {}
    rs = ind.get("relative_strength") or {}
    out[t] = {
        "as_of": ind.get("as_of"), "price": ind.get("price"),
        "verdict": verdict.get("verdict"),
        "action": action.get("action"), "action_label": action.get("label"),
        "trigger": action.get("trigger"), "trigger_label": action.get("trigger_label"), "stop": action.get("stop"),
        "momentum": mom.get("score"), "momentum_label": mom.get("label"), "momentum_accel": mom.get("accel"),
        "swing": swing.get("score"), "swing_label": swing.get("label"), "swing_setup": swing.get("setup"),
        "rsi14": ind.get("rsi14"),
        "dist_sma50_pct": ind.get("dist_sma50_pct"), "dist_sma200_pct": ind.get("dist_sma200_pct"),
        "sma50_above_sma200": ind.get("sma50_above_sma200"),
        "range_position_pct": ind.get("range_position_pct"), "dist_52w_high_pct": ind.get("dist_52w_high_pct"),
        "return_1m_pct": ind.get("return_1m_pct"), "return_3m_pct": ind.get("return_3m_pct"),
        "return_12m_pct": ind.get("return_12m_pct"),
        "rs_3m_pct": rs.get("rs_3m_pct"),
        "macd_hist": ind.get("macd_hist"), "rsi_divergence": ind.get("rsi_divergence"),
        "nearest_support": ind.get("nearest_support"), "nearest_resistance": ind.get("nearest_resistance"),
        "volatility_20d": ind.get("volatility_20d"),
    }
    o = out[t]
    print(f"{t:5s} {o['as_of']} ${o['price']:>8} {o['verdict']:<9} {str(o['action']):<6} {str(o['action_label'])[:34]:34s} "
          f"mom {o['momentum']} swing {o['swing']} RSI {o['rsi14']} 50d {o['dist_sma50_pct']}% 200d {o['dist_sma200_pct']}%")

json.dump(out, open(os.path.join(HERE, "technical.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
