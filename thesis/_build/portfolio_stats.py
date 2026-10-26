"""Portfolio risk stats from cached daily prices (last 252 trading days).

Writes weights and risk figures only -- never the position amounts.
"""
import json
import math
import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))  # outputs land next to this script
RAW = "sec_analyzer/raw"

# Position amounts are private: they live in the gitignored
# thesis/_private/portfolio_positions.json and only weights leave this script.
PRIVATE = os.path.join(SP, "..", "_private", "portfolio_positions.json")
if os.path.exists(PRIVATE):
    POS = json.load(open(PRIVATE, encoding="utf-8"))["positions_usd"]
else:  # fall back to the committed weights (same tickers, unit total)
    POS = json.load(open(os.path.join(SP, "portfolio_stats.json"), encoding="utf-8"))["weights"]
SECTOR = {
    "NVDA": "Yarı iletken / AI donanım", "MU": "Yarı iletken / AI donanım", "AVGO": "Yarı iletken / AI donanım",
    "ORCL": "Yazılım, bulut ve AI altyapı", "CRWV": "Yazılım, bulut ve AI altyapı", "APP": "Yazılım, bulut ve AI altyapı",
    "CEG": "Enerji ve elektrik (nükleer, depolama)", "OKLO": "Enerji ve elektrik (nükleer, depolama)",
    "EOSE": "Enerji ve elektrik (nükleer, depolama)",
    "UBER": "Tüketici internet platformları", "NFLX": "Tüketici internet platformları", "SE": "Tüketici internet platformları",
    "HIMS": "Sağlık", "UNH": "Sağlık",
    "SOFI": "Fintech / finans", "HOOD": "Fintech / finans",
    "RKLB": "Uzay, savunma ve kamu güvenliği", "AXON": "Uzay, savunma ve kamu güvenliği",
}

total = sum(POS.values())
w = {t: v / total for t, v in POS.items()}

closes = {}
for t in list(POS) + ["SPY"]:
    df = pd.read_csv(os.path.join(RAW, f"prices_{t}_full.csv"), parse_dates=["Date"]).set_index("Date")
    closes[t] = df["Close"]
px = pd.DataFrame(closes).dropna().iloc[-253:]
rets = px.pct_change().dropna()

ann = math.sqrt(252)
stock = {}
for t in POS:
    r = rets[t]
    beta = r.cov(rets["SPY"]) / rets["SPY"].var()
    dd = (px[t] / px[t].cummax() - 1).min()
    stock[t] = {"vol": r.std() * ann, "beta": beta, "ret_1y": px[t].iloc[-1] / px[t].iloc[0] - 1, "max_dd": dd}

wv = pd.Series(w)
port_r = rets[list(POS)].mul(wv, axis=1).sum(axis=1)
port_curve = (1 + port_r).cumprod()
spy_curve = (1 + rets["SPY"]).cumprod()
corr = rets[list(POS)].corr()
n = len(POS)
avg_corr = (corr.values.sum() - n) / (n * n - n)
res = {
    "window": [str(px.index[0].date()), str(px.index[-1].date())],
    "port_vol": port_r.std() * ann, "spy_vol": rets["SPY"].std() * ann,
    "port_beta": port_r.cov(rets["SPY"]) / rets["SPY"].var(),
    "port_ret_1y": port_curve.iloc[-1] - 1, "spy_ret_1y": spy_curve.iloc[-1] - 1,
    "port_max_dd": (port_curve / port_curve.cummax() - 1).min(),
    "spy_max_dd": (spy_curve / spy_curve.cummax() - 1).min(),
    "avg_pair_corr": avg_corr,
    "worst_day": port_r.min(), "worst_day_date": str(port_r.idxmin().date()),
    "stock": stock,
    "weights": {t: round(v, 4) for t, v in w.items()},
    "sector_by_ticker": SECTOR,
    "sector": {},
}
for t, s in SECTOR.items():
    res["sector"].setdefault(s, 0.0)
    res["sector"][s] = round(res["sector"][s] + w[t], 4)

# AI-capex correlation cluster
ai = ["NVDA", "MU", "AVGO", "ORCL", "CRWV", "APP", "CEG", "OKLO", "EOSE"]
res["ai_corr"] = (corr.loc[ai, ai].values.sum() - len(ai)) / (len(ai) ** 2 - len(ai))
json.dump(res, open(os.path.join(SP, "portfolio_stats.json"), "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)

print(f"window {res['window']}")
print(f"port vol {res['port_vol']:.1%} vs SPY {res['spy_vol']:.1%}; beta {res['port_beta']:.2f}")
print(f"1y return port {res['port_ret_1y']:.1%} vs SPY {res['spy_ret_1y']:.1%}; max DD {res['port_max_dd']:.1%} vs SPY {res['spy_max_dd']:.1%}")
print(f"avg pair corr {avg_corr:.2f}; AI cluster corr {res['ai_corr']:.2f}; worst day {res['worst_day']:.1%} on {res['worst_day_date']}")
for s, v in sorted(res["sector"].items(), key=lambda x: -x[1]):
    print(f"  {s:42s} {v:6.1%}")
for t, d in sorted(stock.items(), key=lambda x: -w[x[0]]):
    print(f"  {t:5s} w {w[t]:5.1%} vol {d['vol']:5.0%} beta {d['beta']:4.2f} 1y {d['ret_1y']:+6.0%} maxDD {d['max_dd']:5.0%}")
