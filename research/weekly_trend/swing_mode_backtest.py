"""Swing-mode backtest for the Weekly Trend Rule indicator (HANDOFF sec.11).

Rules were fixed before the first run:
  Filter F (indicator buy state): close > SMA200, or (EMA8 > EMA21 and close > both).
  S1 RSI(2) pullback: buy at close when RSI(2) < 10. Exit at close when close > SMA5,
     or after 10 trading days; intraday stop at entry - 2 x ATR14.
  S2 EMA21 pullback reversal: buy at close when low <= EMA21 < close and close > previous
     high. Stop = signal-day low - 0.5 x ATR14, target = entry + 2 x (entry - stop),
     time exit at close after 10 trading days. Stop wins if stop and target hit same day.
Each name has its own sub-account (CAPITAL / N), whole sub-account per trade, 0.1% cost
per side, gaps fill at the open. Usage:
  python research/weekly_trend/swing_mode_backtest.py out.json [basket,...] [period,...]
"""
import json
import logging
import sys

import pandas as pd

logging.disable(logging.CRITICAL)
from sec_analyzer.fetch.prices import get_price_history

CAPITAL, COST = 100_000.0, 0.001
LAST = pd.Timestamp("2026-11-03")
PERIODS = {"train": (pd.Timestamp("2020-11-02"), pd.Timestamp("2025-10-31")),
           "test": (pd.Timestamp("2025-11-03"), pd.Timestamp("2026-11-03")),
           "crisis": (pd.Timestamp("2007-01-03"), pd.Timestamp("2012-12-31")),
           "val": (pd.Timestamp("2015-01-02"), pd.Timestamp("2020-10-30"))}
BASKETS = {"semi_AMD": ["NVDA", "AVGO", "MRVL", "AMD", "ORCL"], "first_noAPP": ["UBER", "NFLX", "RDDT"],
           "b3_etf": ["GLD", "SLV", "USO", "BTC-USD"],
           "sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI"],
           "b5": ["HIMS", "EOSE", "RKLB", "SOFI", "HOOD"]}
DEFAULT_PERIODS = {"sectors": ("train", "test", "crisis", "val"), "b5": ("train", "test")}
MODES = ["S1", "S1F", "S2", "S2F"]


def frame(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    f = pd.DataFrame({"close": c, "open": df["Open"], "high": h, "low": l})
    f["sma5"] = c.rolling(5).mean()
    f["sma200"] = c.rolling(200).mean()
    f["ema8"] = c.ewm(span=8, adjust=False).mean()
    f["ema21"] = c.ewm(span=21, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    f["rsi2"] = 100 - 100 / (1 + up / dn)
    f["prev_high"] = h.shift(1)
    f["filt"] = (c > f["sma200"]) | ((f["ema8"] > f["ema21"]) & (c > f["ema8"]) & (c > f["ema21"]))
    f.loc[f["sma200"].isna(), "filt"] = False
    return f


F = {}
for t in sorted({x for v in BASKETS.values() for x in v}):
    df, _ = get_price_history(t)
    F[t] = frame(df[df.index <= LAST])


def signal(mode, r):
    if pd.isna(r["atr"]) or pd.isna(r["sma5"]):
        return None
    if mode.endswith("F") and not bool(r["filt"]):
        return None
    c = float(r["close"])
    if mode.startswith("S1"):
        if not r["rsi2"] < 10:
            return None
        return {"stop": c - 2 * float(r["atr"]), "target": None}
    if not (r["low"] <= r["ema21"] < c and c > r["prev_high"]):
        return None
    stop = float(r["low"]) - 0.5 * float(r["atr"])
    if not stop < c:
        return None
    return {"stop": stop, "target": c + 2 * (c - stop)}


def simulate(mode, tickers, start, end):
    days = pd.DatetimeIndex(sorted({d for t in tickers for d in F[t].index if start <= d <= end}))
    slot = CAPITAL / len(tickers)
    cash = {t: slot for t in tickers}
    pos, trades, eq, inmkt = {}, [], [], 0
    last_close = {t: None for t in tickers}

    def close_out(t, day, fill, why):
        p = pos.pop(t)
        cash[t] += p["sh"] * fill * (1 - COST)
        trades.append({"t": t, "in": str(p["entry"].date()), "px": round(p["px"], 2), "out": str(day.date()),
                       "fill": round(fill, 2), "why": why, "ret": round((cash[t] / p["cost"] - 1) * 100, 2),
                       "pnl": round(cash[t] - p["cost"], 2)})

    for day in days:
        for t in tickers:
            if day not in F[t].index:
                continue
            b = F[t].loc[day]
            if t in pos:
                p = pos[t]
                p["bars"] += 1
                o = float(b["open"])
                if b["low"] <= p["stop"]:
                    close_out(t, day, min(o, p["stop"]), "stop")
                elif p["target"] is not None and b["high"] >= p["target"]:
                    close_out(t, day, max(o, p["target"]), "target")
                elif mode.startswith("S1") and b["close"] > b["sma5"]:
                    close_out(t, day, float(b["close"]), "sma5")
                elif p["bars"] >= 10:
                    close_out(t, day, float(b["close"]), "time")
            elif day < end:
                s = signal(mode, b)
                if s is not None:
                    c = float(b["close"])
                    pos[t] = {"sh": cash[t] * (1 - COST) / c, "cost": cash[t], "px": c, "entry": day,
                              "stop": s["stop"], "target": s["target"], "bars": 0}
                    cash[t] = 0.0
            last_close[t] = float(b["close"])
        inmkt += len(pos)
        eq.append(sum(cash.values()) + sum(p["sh"] * last_close[t] for t, p in pos.items()))
    for t in list(pos):
        p = pos.pop(t)
        val = p["sh"] * last_close[t]
        trades.append({"t": t, "in": str(p["entry"].date()), "px": round(p["px"], 2), "out": "open",
                       "fill": round(last_close[t], 2), "why": "open", "ret": round((val / p["cost"] - 1) * 100, 2),
                       "pnl": round(val - p["cost"], 2)})
    eq = pd.Series(eq, index=days)
    ret = round((eq.iloc[-1] / CAPITAL - 1) * 100, 2)
    return {"ret": ret, "dd": round((eq / eq.cummax() - 1).min() * 100, 2), "n": len(trades),
            "wins": sum(1 for x in trades if x["pnl"] > 0),
            "exposure": round(100 * inmkt / (len(days) * len(tickers)), 1), "trades": trades,
            "chk": round(sum(x["pnl"] for x in trades) - ret * 1000)}


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "swing_mode_result.json"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
    pers = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    out = {}
    for bk, tk in BASKETS.items():
        if only and bk not in only:
            continue
        for per in pers or DEFAULT_PERIODS.get(bk, ("train", "test", "val")):
            for m in MODES:
                r = simulate(m, tk, *PERIODS[per])
                out[f"{m}_{bk}_{per}"] = r
                print("ROW", bk, per, m, "ret", r["ret"], "dd", r["dd"], "n", r["n"], "wins", r["wins"],
                      "exp", r["exposure"], "| chk", r["chk"])
    json.dump(out, open(out_path, "w"), indent=1, default=str)
