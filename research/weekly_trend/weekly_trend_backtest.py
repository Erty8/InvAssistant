import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else 'weekly_trend_result.json'
QUAL = 0
def STRONG(t, d):
    if not QUAL or d not in F[t].index:
        return False
    r = F[t].loc[d]
    if QUAL == 1:
        return bool(r['pct_above252'] >= 0.80)
    return bool(r['cross252'] <= 4)
BLOCKED = {}
REBLOCK = 0
RISKCAP = 0
RISKSIZE = 0
def CAPSTOP(o, s):
    return max(s, o * (1 - RISKCAP)) if RISKCAP else s
def FRAC(o, s):
    return min(1.0, RISKSIZE / ((o - s) / o)) if RISKSIZE else 1.0
ZONE_EXIT = 0
ABOVE_EMA = False
EXITDAY = {}
CLOSE_ENTRY = False
MONDAY_ONLY = False
FRIDAY_N = 0
PREVFRI = {}
HYB2 = False
LASTBUY = {}
ONE_PER_WEEK = False
BUF = 0.05      # stop buffer below the reference MA (0.05 = x0.95)
VOLK = 0        # >0: buffer = VOLK * 252d mean ATR%, clipped 2..12% (HANDOFF sec.7c)
ENTRYF = ""     # rule-1 (Monday) entry filter: A<k> vol band, B SMA200 slope, C momentum vote (HANDOFF sec.9)
ENTRYK = 1.0
DTB = 0         # downtrend-break entry below SMA200: 1 trendline, 2 structure (HANDOFF sec.10)
DTB_REPLACE = False  # True: DTB replaces the EMA8/21 rule-2 entry
PIV = {}
RECLAIM = False # stop-reclaim re-entry (HANDOFF sec.7)
RC = {}
"""Walk-forward strategy search (method stated to the user before running).

Train 2020-11-02 .. 2025-10-31, test 2025-11-03 .. 2026-11-03, both baskets.
Each name has its own sub-account = CAPITAL / N. Weekly entry (Friday signal ->
Monday open, whole sub-account), exit ONLY via a stop (intraday fill, open on
a gap), stop recomputed every Friday and only raised. 0.1% cost per side.
Buy & hold: each sub-account buys at the first open on/after the period start
where the name trades, holds to the end.
Selection (fixed in advance): maximise min over baskets of
(strategy return - buy&hold return) on TRAIN, subject to >= 10 trades total.
"""
import itertools
import json
import logging
import sys

import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)
from sec_analyzer.fetch.prices import get_price_history

CAPITAL, COST = 100_000.0, 0.001
PERIODS = {"train": (pd.Timestamp("2020-11-02"), pd.Timestamp("2025-10-31")),
           "test": (pd.Timestamp("2025-11-03"), pd.Timestamp("2026-11-03")),
           "crisis": (pd.Timestamp("2007-01-03"), pd.Timestamp("2012-12-31")),
           "val": (pd.Timestamp("2015-01-02"), pd.Timestamp("2020-10-30"))}
BASKETS = {"semi_AMD": ["NVDA", "AVGO", "MRVL", "AMD", "ORCL"], "first_noAPP": ["UBER", "NFLX", "RDDT"], "b3_etf": ["GLD", "SLV", "USO", "BTC-USD"], "sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI"], "b5": ["HIMS", "EOSE", "RKLB", "SOFI", "HOOD"]}
LAST = pd.Timestamp("2026-11-03")

ENTRIES = (["DON20", "DON50", "DON100", "TT"]
           + [f"RS{l}_{k}" for l in (63, 126, 252) for k in (1, 2, 3)]
           + ["MA50", "MA100", "MA200", "RSI30", "RSI25", "BBLOW"])
STOPS = ["ATR1.5", "ATR2", "ATR3", "ATR4", "ATR5", "ATR6", "LOW20", "LOW50"]


def kalman_trend(close):
    y = np.log(close.values.astype(float))
    n = len(y)
    lev = np.full(n, np.nan)
    slo = np.full(n, np.nan)
    if n == 0:
        return lev, slo
    x = np.array([y[0], 0.0])
    P = np.eye(2)
    Fm = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([1e-5, 1e-6])
    R = 0.02 ** 2
    for i in range(n):
        if i > 0:
            x = Fm @ x
            P = Fm @ P @ Fm.T + Q
        s = P[0, 0] + R
        k = P[:, 0] / s
        x = x + k * (y[i] - x[0])
        P = P - np.outer(k, P[0, :])
        lev[i], slo[i] = x[0], x[1]
    return np.exp(lev), slo


def frame(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    f = pd.DataFrame({"close": c, "open": df["Open"], "high": h, "low": l})
    for n in (50, 100, 200):
        f[f"sma{n}"] = c.rolling(n).mean()
    f["sma200_prev"] = f["sma200"].shift(21)
    _ab = (c > f["sma200"]).astype(float)
    f["pct_above252"] = _ab.rolling(252).mean()
    f["cross252"] = (_ab != _ab.shift()).astype(float).rolling(252).sum()
    f["ema8"] = c.ewm(span=8, adjust=False).mean()
    f["ema21"] = c.ewm(span=21, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atrp252"] = (f["atr"] / c).rolling(252, min_periods=200).mean()
    for n in (20, 50, 100):
        f[f"hi{n}_prev"] = c.shift(1).rolling(n).max()
    for n in (20, 50):
        f[f"low{n}"] = l.rolling(n).min()
    f["hi252"] = c.rolling(252).max()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn)
    f["bb_lo"] = c.rolling(20).mean() - 2 * c.rolling(20).std()
    f["mom12_1"] = c.shift(21) / c.shift(252) - 1
    for n in (21, 63, 126, 252):
        f[f"ret{n}"] = c / c.shift(n) - 1
    return f


F = {}
for t in sorted({x for v in BASKETS.values() for x in v} | {"SPY"}):
    df, _ = get_price_history(t)
    F[t] = frame(df[df.index <= LAST])
SPY = F["SPY"]
import bisect
for _t, _f in F.items():
    _h, _l = _f["high"], _f["low"]
    _ph = _h == _h.rolling(11, center=True).max()
    _pl = _l == _l.rolling(11, center=True).min()
    PIV[_t] = (list(np.flatnonzero(_ph.values)), list(np.flatnonzero(_pl.values)))


def dtb_signal(t, day, prevfri):
    """Fresh downtrend break on Friday `day` (close above the line/level, previous Friday not)."""
    f = F[t]
    j = f.index.get_loc(day)
    r = f.iloc[j]
    if pd.isna(r["sma200"]) or r["close"] >= r["sma200"] or prevfri not in f.index:
        return False
    jp = f.index.get_loc(prevfri)
    ph, pl = PIV[t]
    hs = [i for i in ph[max(0, bisect.bisect_right(ph, j - 253)):bisect.bisect_right(ph, j - 5)]]
    if len(hs) < 2:
        return False
    p1, p2 = hs[-2], hs[-1]
    h1, h2 = float(f["high"].iloc[p1]), float(f["high"].iloc[p2])
    if not h2 < h1:
        return False
    c, cp = float(r["close"]), float(f["close"].iloc[jp])
    if DTB == 1:
        slope = (h2 - h1) / (p2 - p1)
        line = lambda x: h2 + slope * (x - p2)
        return c > line(j) and cp <= line(jp)
    ls = [i for i in pl[:bisect.bisect_right(pl, j - 5)] if i > p2]
    prior = [i for i in pl[:bisect.bisect_right(pl, p2)]]
    if not ls or not prior:
        return False
    if not float(f["low"].iloc[ls[-1]]) > float(f["low"].iloc[prior[-1]]):
        return False
    return c > h2 and cp <= h2
ALL_DAYS = SPY.index
WK = ALL_DAYS.to_series().dt.isocalendar()
FRIDAYS = sorted(set(ALL_DAYS.to_series().groupby([WK.year, WK.week]).max()))


def entry_stop(entry, stop, t, d, tickers):
    """Initial stop if the entry fires for t on Friday d, else None."""
    f = F[t]
    if d not in f.index:
        return None
    r = f.loc[d]
    c = r["close"]
    if pd.isna(r["sma200"]) or pd.isna(r["atr"]):
        return None
    ok = False
    if entry.startswith("DON"):
        n = int(entry[3:])
        ok = c >= r[f"hi{n}_prev"] and c > r["sma200"]
    elif entry == "TT":
        ok = (c > r["sma50"] > r["sma200"] and r["sma200"] > r["sma200_prev"] and r["mom12_1"] > 0
              and c >= 0.75 * r["hi252"])
    elif entry.startswith("RS") and not entry.startswith("RSI"):
        lb, k = (int(x) for x in entry[2:].split("_"))
        col = f"ret{lb}"
        avail = [x for x in tickers if d in F[x].index and pd.notna(F[x].loc[d, col])]
        top = sorted(avail, key=lambda x: -F[x].loc[d, col])[:k]
        ok = t in top and r[col] > SPY.loc[d, col] and c > r["sma200"]
    elif entry.startswith("RSI"):
        ok = r["rsi"] < float(entry[3:])
    elif entry == "BBLOW":
        ok = c < r["bb_lo"]
    elif entry.startswith("MA"):
        ok = c > r[f"sma{int(entry[2:])}"]
    if not ok:
        return None
    if stop.startswith("ATR"):
        s = c - float(stop[3:]) * r["atr"]
    else:
        s = r[f"low{int(stop[3:])}"]
    return float(s) if pd.notna(s) and 0 < s < c else None


def trail(stop, t, d, p):
    r = F[t].loc[d]
    if stop.startswith("ATR"):
        return p["hi"] - float(stop[3:]) * r["atr"]
    return r[f"low{int(stop[3:])}"]


def simulate(entry, stop, tickers, start, end):
    days = ALL_DAYS[(ALL_DAYS >= start) & (ALL_DAYS <= end)]
    sig = {}
    for d in FRIDAYS:
        i = ALL_DAYS.get_loc(d)
        if i + 1 < len(ALL_DAYS) and start <= ALL_DAYS[i + 1] <= end:
            sig[ALL_DAYS[i + 1]] = d
    fridays_in = {d for d in TRAIL_DAYS if start <= d <= end}
    slot = CAPITAL / len(tickers)
    cash = {t: slot for t in tickers}
    pos, trades, eq = {}, [], []
    LASTBUY.clear()
    BLOCKED.clear()
    EXITDAY.clear()
    RC.clear()
    last_close = {t: None for t in tickers}
    for day in days:
        if (not CLOSE_ENTRY or HYB2) and day in sig:
            fri = sig[day]
            for t in tickers:
                if t in pos or day not in F[t].index:
                    continue
                if ONE_PER_WEEK and LASTBUY.get(t) == tuple(day.isocalendar())[:2]:
                    continue
                s, sk = entry_kind(entry, t, fri, tickers)
                if s is None:
                    continue
                if HYB2 and sk != "TIER":
                    continue
                if not entry_ok(t, fri):
                    continue
                if REBLOCK and BLOCKED.get(t) and not STRONG(t, fri):
                    continue
                o = float(F[t].loc[day, "open"])
                if o <= s:
                    continue
                _s2 = CAPSTOP(o, s); _f = FRAC(o, _s2); pos[t] = {"left": cash[t] * (1 - _f), "sh": cash[t] * _f * (1 - COST) / o, "stop": _s2, "hi": o, "entry": day, "cost": cash[t],
                          "px": o, "fri": fri, "sk": sk}
                cash[t] = pos[t]["left"]
                LASTBUY[t] = tuple(day.isocalendar())[:2]
        for t in list(pos):
            if day not in F[t].index:
                continue
            b, p = F[t].loc[day], pos[t]
            if b["low"] <= p["stop"]:
                fill = p["stop"] if day == p["entry"] else min(float(b["open"]), p["stop"])
                cash[t] = cash[t] + p["sh"] * fill * (1 - COST)
                trades.append({"kind": p["sk"], "t": t, "in": str(p["entry"].date()), "px": round(p["px"], 2), "out": str(day.date()),
                               "fill": round(fill, 2), "ret": round((cash[t] / p["cost"] - 1) * 100, 2),
                               "pnl": round(cash[t] - p["cost"], 2)})
                del pos[t]
                EXITDAY[t] = day; BLOCKED[t] = True
                if RECLAIM:
                    RC[t] = {"lvl": p["stop"], "exit": day, "cnt": 0, "armed": None}
            else:
                p["hi"] = max(p["hi"], float(b["close"]))
        for t in tickers:
            if day in F[t].index:
                last_close[t] = float(F[t].loc[day, "close"])
        if ZONE_EXIT:
            for t in list(pos):
                f = F[t]
                if day not in f.index:
                    continue
                i = f.index.get_loc(day)
                if i < 1:
                    continue
                p = pos[t]
                if ZONE_EXIT >= 5:
                    if day not in PREVFRI or PREVFRI[day] not in f.index:
                        continue
                    d1 = PREVFRI[day]
                    r1 = f.loc[d1]
                else:
                    d1 = f.index[i - 1]
                    r1 = f.iloc[i - 1]
                if d1 < p["entry"]:
                    continue
                r2 = f.iloc[i]
                d0 = PREVFRI.get(d1, d1 - pd.Timedelta(days=7))
                wk_hi = f.loc[(f.index > d0) & (f.index <= d1), "high"].max()
                z1 = r1["close"] < r1["sma50"] and r1["close"] > r1["sma200"] * 1.10
                atrp = r1["atr"] / r1["close"]
                if ZONE_EXIT == 2:
                    ap = r1["high"] >= r1["sma50"] * 0.98
                elif ZONE_EXIT == 3:
                    ap = True if atrp > 0.04 else (r1["high"] >= r1["sma50"] * 0.98)
                elif ZONE_EXIT == 4:
                    ap = atrp > 0.04
                elif ZONE_EXIT == 5:
                    ap = wk_hi >= r1["sma50"] * 0.98
                else:
                    ap = True
                if z1 and ap and r2["close"] < r2["sma50"]:
                    fill = float(r2["close"])
                    cash[t] = cash[t] + p["sh"] * fill * (1 - COST)
                    trades.append({"kind": p["sk"] + "_ZONE", "t": t, "in": str(p["entry"].date()), "px": round(p["px"], 2), "out": str(day.date()),
                                   "fill": round(fill, 2), "ret": round((cash[t] / p["cost"] - 1) * 100, 2),
                                   "pnl": round(cash[t] - p["cost"], 2)})
                    del pos[t]
                    EXITDAY[t] = day; BLOCKED[t] = True
        if REBLOCK and day in PREVFRI:
            for t in list(BLOCKED):
                if day in F[t].index:
                    r = F[t].loc[day]
                    if (REBLOCK == 1 and r['close'] > r['sma50']) or (REBLOCK == 2 and r['ema8'] > r['ema21']):
                        del BLOCKED[t]
        if CLOSE_ENTRY and (not MONDAY_ONLY or day in sig) and (not FRIDAY_N or day in PREVFRI):
            for t in tickers:
                if t in pos or day not in F[t].index or EXITDAY.get(t) == day:
                    continue
                i = F[t].index.get_loc(day)
                if i < 1:
                    continue
                s0, _k0 = (entry_kind(entry, t, PREVFRI[day], tickers) if FRIDAY_N == 2 else ((1.0, None) if FRIDAY_N == 1 else (((1.0, None) if entry_kind(entry, t, day, tickers)[1] == "TIER" else entry_kind(entry, t, PREVFRI[day], tickers)) if FRIDAY_N == 3 else entry_kind(entry, t, F[t].index[i - 1], tickers))))
                if s0 is None:
                    continue
                s, sk = entry_kind(entry, t, day, tickers)
                if HYB2 and sk == "TIER":
                    continue
                if DTB_REPLACE and sk == "E821":
                    continue
                if s is None:
                    continue
                if REBLOCK and BLOCKED.get(t) and not STRONG(t, day):
                    continue
                o = float(F[t].loc[day, "close"])
                if o <= s:
                    continue
                _s2 = CAPSTOP(o, s); _f = FRAC(o, _s2); pos[t] = {"left": cash[t] * (1 - _f), "sh": cash[t] * _f * (1 - COST) / o, "stop": _s2, "hi": o, "entry": day, "cost": cash[t], "px": o, "fri": day, "sk": sk}
                cash[t] = pos[t]["left"]
        if DTB and day in fridays_in and day in PREVFRI:
            for t in tickers:
                if t in pos or day not in F[t].index or EXITDAY.get(t) == day:
                    continue
                if not dtb_signal(t, day, PREVFRI[day]):
                    continue
                r = F[t].loc[day]
                c = float(r["close"])
                s = float(r["ema21"]) * (1 - buf(r))
                if not s < c:
                    continue
                pos[t] = {"left": 0.0, "sh": cash[t] * (1 - COST) / c, "stop": s, "hi": c, "entry": day,
                          "cost": cash[t], "px": c, "fri": day, "sk": "DTB"}
                cash[t] = 0.0
        if RECLAIM:
            for t in list(RC):
                if t in pos:
                    del RC[t]  # a normal H2 entry fired -> rule resets
                    continue
                if day not in F[t].index or day <= RC[t]["exit"]:
                    continue
                st, c = RC[t], float(F[t].loc[day, "close"])
                if st["armed"] is None:
                    st["cnt"] = st["cnt"] + 1 if c > st["lvl"] else 0
                    if st["cnt"] >= 3:
                        st["armed"] = day
                    continue
                if day in fridays_in and day > st["armed"]:
                    del RC[t]
                    if c > st["lvl"]:
                        s = st["lvl"] * 0.95
                        pos[t] = {"left": 0.0, "sh": cash[t] * (1 - COST) / c, "stop": s, "hi": c, "entry": day,
                                  "cost": cash[t], "px": c, "fri": day, "sk": "RECLAIM"}
                        cash[t] = 0.0
        if day in fridays_in:
            for t, p in pos.items():
                if day in F[t].index:
                    n = stop_level(p["sk"], F[t].loc[day])
                    if pd.notna(n) and n > p["stop"]:
                        p["stop"] = float(n)
        eq.append(sum(cash.values()) + sum(p["sh"] * last_close[t] for t, p in pos.items()))
    for t, p in pos.items():
        val = p["sh"] * last_close[t] + p.get("left", 0.0)
        trades.append({"kind": p["sk"], "t": t, "in": str(p["entry"].date()), "px": round(p["px"], 2), "out": "open",
                       "fill": round(last_close[t], 2), "ret": round((val / p["cost"] - 1) * 100, 2),
                       "pnl": round(val - p["cost"], 2)})
    eq = pd.Series(eq, index=days)
    return {"ret": round((eq.iloc[-1] / CAPITAL - 1) * 100, 2), "dd": round((eq / eq.cummax() - 1).min() * 100, 2),
            "n": len(trades), "wins": sum(1 for x in trades if x["pnl"] > 0), "trades": trades}


def buy_hold(tickers, start, end):
    days = ALL_DAYS[(ALL_DAYS >= start) & (ALL_DAYS <= end)]
    slot = CAPITAL / len(tickers)
    vals = []
    first = {}
    for t in tickers:
        live = F[t].index[(F[t].index >= start) & (F[t].index <= end)]
        first[t] = live[0] if len(live) else None
    for day in days:
        v = 0.0
        for t in tickers:
            ft = first[t]
            if ft is None or day < ft:
                v += slot
            else:
                o = float(F[t].loc[ft, "open"])
                c = float(F[t]["close"][F[t].index <= day].iloc[-1])
                v += slot * (1 - COST) * c / o
        vals.append(v)
    s = pd.Series(vals, index=days)
    return {"ret": round((s.iloc[-1] / CAPITAL - 1) * 100, 2), "dd": round((s / s.cummax() - 1).min() * 100, 2),
            "first": {t: str(v.date()) if v is not None else None for t, v in first.items()}}





def buf(r):
    if VOLK:
        v = r["atrp252"]
        return 0.05 if pd.isna(v) else min(0.12, max(0.02, VOLK * float(v)))
    return BUF


def entry_ok(t, d):
    if not ENTRYF:
        return True
    r = F[t].loc[d]
    if ENTRYF == "A":
        v = r["atrp252"]
        band = 0.05 if pd.isna(v) else min(0.12, max(0.02, ENTRYK * float(v)))
        return bool(r["close"] > r["sma200"] * (1 + band))
    if ENTRYF == "B":
        return bool(pd.notna(r["sma200_prev"]) and r["sma200"] > r["sma200_prev"])
    if ENTRYF == "C":
        return sum(bool(pd.notna(r[c]) and r[c] > 0) for c in ("ret21", "ret63", "ret252")) >= 2
    raise ValueError(ENTRYF)


def tier(r):
    return r["sma50"] * (1 - buf(r)) if r["close"] > r["sma50"] > r["sma200"] else r["sma200"] * (1 - buf(r))


def stop_level(kind, r):
    if kind == "TIER" or r["close"] > r["sma200"]:
        return tier(r)
    return r["ema21"] * (1 - buf(r))


def entry_kind(entry, t, d, tickers):
    f = F[t]
    if d not in f.index:
        return None, None
    r = f.loc[d]
    if r[["sma50", "sma200", "ema8", "ema21"]].isna().any():
        return None, None
    c = r["close"]
    if c > r["sma200"]:
        if ABOVE_EMA and not (r["ema8"] > r["ema21"]):
            return None, None
        s = tier(r)
        return (float(s), "TIER") if 0 < s < c else (None, None)
    ok = entry == "E821" and r["ema8"] > r["ema21"] and c > r["ema8"] and c > r["ema21"]
    if not ok:
        return None, None
    s = r["ema21"] * (1 - buf(r))
    return (float(s), "E821") if 0 < s < c else (None, None)


WEEKLY0 = list(FRIDAYS)
out = {}
import re
MODES = [m for m in sys.argv[2].split(",")] if len(sys.argv) > 2 else ["H2", "H2_X2_approach"]
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
for bk, tk in BASKETS.items():
    if ONLY and bk not in ONLY:
        continue
    for per in (sys.argv[4].split(",") if len(sys.argv) > 4 else (("test", "train", "crisis") if bk == "sectors" else ("test", "train"))):
        bh = buy_hold(tk, *PERIODS[per])
        WEEKLY = WEEKLY0
        for name, MODE in (("E821", m) for m in MODES):
            _b = re.search(r"_B(\d+)", MODE)
            globals()["BUF"] = int(_b.group(1)) / 100 if _b else 0.05
            globals()["RECLAIM"] = "_RC" in MODE
            _dm = re.search(r"_D([12])(R?)", MODE)
            globals()["DTB"] = int(_dm.group(1)) if _dm else 0
            globals()["DTB_REPLACE"] = bool(_dm and _dm.group(2))
            _e = re.search(r"_E([ABC])(\d*)", MODE)
            globals()["ENTRYF"] = _e.group(1) if _e else ""
            globals()["ENTRYK"] = int(_e.group(2)) / 10 if _e and _e.group(2) else 1.0
            _v = re.search(r"_V(\d+)", MODE)
            globals()["VOLK"] = int(_v.group(1)) / 10 if _v else 0
            globals()["REBLOCK"] = 1 if "RB50" in MODE else 0
            globals()["QUAL"] = 1 if "_Q1" in MODE else (2 if "_Q2" in MODE else 0)
            globals()["RISKCAP"] = 0.15 if "RCAP" in MODE else 0
            globals()["RISKSIZE"] = 0.10 if "RSIZE" in MODE else 0
            globals()["ABOVE_EMA"] = "_E1" in MODE
            globals()["FRIDAYS"] = WEEKLY
            globals()["TRAIL_DAYS"] = WEEKLY
            globals()["ZONE_EXIT"] = 5 if "_WX2" in MODE else (6 if "_WX1" in MODE else (2 if "_X2" in MODE else 0))
            globals()["ONE_PER_WEEK"] = False
            globals()["CLOSE_ENTRY"] = MODE.startswith(("F", "H"))
            globals()["MONDAY_ONLY"] = False
            globals()["FRIDAY_N"] = 2 if MODE.startswith("F2") else (3 if MODE.startswith("H") else 0)
            globals()["HYB2"] = MODE.startswith("H2")
            globals()["PREVFRI"] = {WEEKLY0[k]: WEEKLY0[k - 1] for k in range(1, len(WEEKLY0))}
            r = simulate(name, "unused", tk, *PERIODS[per])
            chk = round(sum(x["pnl"] for x in r["trades"]) - r["ret"] * 1000)
            kinds = {}
            for x in r["trades"]:
                k = x.get("kind")
                kinds.setdefault(k, [0, 0, 0.0]); kinds[k][0] += 1; kinds[k][1] += x["pnl"] > 0; kinds[k][2] += x["pnl"]
            out[f"{name}_{MODE}_{bk}_{per}"] = {"r": {k: r[k] for k in ("ret", "dd", "n", "wins")}, "bh": bh["ret"], "trades": r["trades"]}
            print("ROW", bk, per, name, MODE, "ret", r["ret"], "dd", r["dd"], "n", r["n"], "wins", r["wins"], "| hold", bh["ret"], bh["dd"],
                  "| chk", chk, "| by kind", {k: (v[0], v[1], round(v[2])) for k, v in kinds.items()})
json.dump(out, open(OUT, "w"), indent=1, default=str)

for _p in PERIODS:
    _b = buy_hold(['SPY'], *PERIODS[_p])
    print('SPY', _p, _b['ret'], _b['dd'])
