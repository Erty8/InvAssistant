"""Weekly SMA200 / EMA8-21 strategy ("H2") with the optional X2 exit.

Implements ``WEEKLY_STRATEGY_SPEC.md`` (the binding contract). Given a daily
OHLC frame it answers two things:

1. **Signal** -- what the rules say if the current week closes at the last
   price: buy at next week's open (close above SMA200), buy at this Friday's
   close (below SMA200, second Friday of EMA8 > EMA21), watch (first such
   Friday), wait, or hold an existing position.
2. **Model position** -- the rule's own hypothetical position and stop,
   obtained by replaying the rules over the whole frame, plus the last few
   closed model trades. This is the rule's trade, not the user's account.

Design, like the rest of the technical layer: pure, deterministic, None-safe,
never raises (returns ``None`` and logs on any failure or too little data).
Display only -- it never feeds valuation, verdicts or the swing score.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

STOP_BUFFER = 0.05
X2_ZONE_ABOVE_SMA200 = 1.10
X2_APPROACH = 0.98
MIN_BARS = 201
RECENT_TRADES = 5

SIGNAL_BUY_NEXT_OPEN = "BUY_NEXT_OPEN"
SIGNAL_BUY_FRIDAY_CLOSE = "BUY_FRIDAY_CLOSE"
SIGNAL_FIRST_FRIDAY = "FIRST_FRIDAY"
SIGNAL_WAIT = "WAIT"
SIGNAL_IN_POSITION = "IN_POSITION"

KIND_ABOVE = "above_sma200"
KIND_BELOW = "below_sma200"

EXIT_STOP = "stop"
EXIT_X2 = "x2"

SIGNAL_LABELS = {
    SIGNAL_BUY_NEXT_OPEN: "BUY at next week's open (close above SMA200)",
    SIGNAL_BUY_FRIDAY_CLOSE: "BUY at this Friday's close (EMA8 > EMA21, 2nd Friday below SMA200)",
    SIGNAL_FIRST_FRIDAY: "WATCH: 1st Friday of EMA8 > EMA21 below SMA200; buy if next Friday confirms",
    SIGNAL_WAIT: "WAIT: below SMA200 without an EMA8/EMA21 uptrend",
    SIGNAL_IN_POSITION: "IN POSITION: hold; exit only on the stop (or X2)",
}


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    f = pd.DataFrame(
        {
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": close,
            "sma50": close.rolling(50).mean(),
            "sma200": close.rolling(200).mean(),
            "ema8": close.ewm(span=8, adjust=False).mean(),
            "ema21": close.ewm(span=21, adjust=False).mean(),
        },
        index=df.index,
    )
    iso = pd.DatetimeIndex(f.index).isocalendar()
    f["week"] = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    return f.dropna(subset=["open", "high", "low", "close"])


def _ready(r) -> bool:
    return pd.notna(r["sma50"]) and pd.notna(r["sma200"])


def _tier_stop(r) -> float:
    if r["close"] > r["sma50"] > r["sma200"]:
        return float(r["sma50"]) * (1 - STOP_BUFFER)
    return float(r["sma200"]) * (1 - STOP_BUFFER)


def _ema_condition(r) -> bool:
    return bool(
        _ready(r)
        and r["close"] < r["sma200"]
        and r["ema8"] > r["ema21"]
        and r["close"] > r["ema8"]
        and r["close"] > r["ema21"]
    )


def _entry_condition(r) -> bool:
    return bool(_ready(r) and (r["close"] > r["sma200"] or _ema_condition(r)))


def _x2_day1(r) -> bool:
    return bool(
        _ready(r)
        and r["close"] < r["sma50"]
        and r["close"] > r["sma200"] * X2_ZONE_ABOVE_SMA200
        and r["high"] >= r["sma50"] * X2_APPROACH
    )


def _week_ends(f: pd.DataFrame) -> list:
    weeks = f["week"].to_numpy()
    n = len(f)
    ends = [weeks[i + 1] != weeks[i] for i in range(n - 1)]
    ends.append(pd.Timestamp(f.index[-1]).weekday() == 4)
    return ends


def _pct(level: Optional[float], ref: Optional[float]) -> Optional[float]:
    if level is None or ref is None or ref == 0:
        return None
    return round((level / ref - 1) * 100, 1)


def _replay(f: pd.DataFrame, use_x2: bool) -> "tuple[Optional[dict], list]":
    """Replay the rules bar by bar (spec §2-4). Returns (open position, closed trades)."""
    weeks = f["week"].to_numpy()
    ends = _week_ends(f)
    recs = f.to_dict("records")
    dates = [pd.Timestamp(d).date().isoformat() for d in f.index]
    pos = None
    trades = []
    last_week_cond = False

    def _close_trade(i: int, price: float, reason: str) -> None:
        nonlocal pos
        trades.append(
            {
                "kind": pos["kind"],
                "entry_date": pos["entry_date"],
                "entry_price": round(pos["entry_price"], 2),
                "exit_date": dates[i],
                "exit_price": round(price, 2),
                "exit_reason": reason,
                "return_pct": _pct(price, pos["entry_price"]),
            }
        )
        pos = None

    for i in range(1, len(recs)):
        r, p = recs[i], recs[i - 1]
        exited_today = False
        # 1) First bar of a new week: raise the stop, then rule-1 entry at the open.
        if weeks[i] != weeks[i - 1] and _ready(p):
            if pos is not None:
                if pos["kind"] == KIND_ABOVE or p["close"] > p["sma200"]:
                    cand = _tier_stop(p)
                else:
                    cand = float(p["ema21"]) * (1 - STOP_BUFFER)
                if cand > pos["stop"]:
                    pos["stop"] = cand
            if pos is None and p["close"] > p["sma200"]:
                s = _tier_stop(p)
                if s < p["close"] and r["open"] > s:
                    pos = {"kind": KIND_ABOVE, "entry_i": i, "entry_date": dates[i],
                           "entry_price": float(r["open"]), "stop": s}
        # 2) Intraday stop.
        if pos is not None and r["low"] <= pos["stop"]:
            fill = pos["stop"] if i == pos["entry_i"] else min(float(r["open"]), pos["stop"])
            _close_trade(i, fill, EXIT_STOP)
            exited_today = True
        # 3) X2 exit at the close (day 1 = previous bar, while in position).
        if use_x2 and pos is not None and i - 1 >= pos["entry_i"] and _x2_day1(p) and r["close"] < r["sma50"]:
            _close_trade(i, float(r["close"]), EXIT_X2)
            exited_today = True
        # 4) Week end: rule-2 entry at the close, then remember this week's condition.
        if ends[i]:
            if pos is None and not exited_today and last_week_cond and _ema_condition(r):
                s = float(r["ema21"]) * (1 - STOP_BUFFER)
                if s < r["close"]:
                    pos = {"kind": KIND_BELOW, "entry_i": i, "entry_date": dates[i],
                           "entry_price": float(r["close"]), "stop": s}
            last_week_cond = _entry_condition(r)
    return pos, trades


def _prev_week_condition(f: pd.DataFrame) -> bool:
    """Entry condition at the last completed week end before the current week."""
    ends = _week_ends(f)
    current_week = f["week"].iloc[-1]
    for i in range(len(f) - 1, -1, -1):
        if ends[i] and f["week"].iloc[i] != current_week:
            return _entry_condition(f.iloc[i])
    return False


def compute_weekly_strategy(df: Optional[pd.DataFrame], use_x2: bool = True) -> Optional[dict]:
    """Weekly strategy read for the latest bar (spec §5). ``None`` on too little data or any error."""
    try:
        if df is None or len(df) < MIN_BARS:
            return None
        f = _frame(df)
        if len(f) < MIN_BARS:
            return None
        last = f.iloc[-1]
        if not _ready(last):
            return None
        close = float(last["close"])
        pos, trades = _replay(f, use_x2)
        prev_cond = _prev_week_condition(f)

        entry_stop = None
        position = None
        if pos is not None:
            signal = SIGNAL_IN_POSITION
            if pos["kind"] == KIND_ABOVE or close > last["sma200"]:
                cand = _tier_stop(last)
            else:
                cand = float(last["ema21"]) * (1 - STOP_BUFFER)
            next_stop = max(pos["stop"], cand)
            position = {
                "kind": pos["kind"],
                "entry_date": pos["entry_date"],
                "entry_price": round(pos["entry_price"], 2),
                "stop": round(pos["stop"], 2),
                "stop_pct": _pct(pos["stop"], close),
                "next_stop": round(next_stop, 2),
                "next_stop_pct": _pct(next_stop, close),
            }
        elif close > last["sma200"]:
            signal = SIGNAL_BUY_NEXT_OPEN
            entry_stop = _tier_stop(last)
        elif _ema_condition(last):
            signal = SIGNAL_BUY_FRIDAY_CLOSE if prev_cond else SIGNAL_FIRST_FRIDAY
            entry_stop = float(last["ema21"]) * (1 - STOP_BUFFER)
        else:
            signal = SIGNAL_WAIT

        return {
            "as_of": pd.Timestamp(f.index[-1]).date().isoformat(),
            "signal": signal,
            "label": SIGNAL_LABELS[signal],
            "entry_stop": round(entry_stop, 2) if entry_stop is not None else None,
            "entry_stop_pct": _pct(entry_stop, close),
            "above_sma200": bool(close > last["sma200"]),
            "prev_week_condition": bool(prev_cond),
            "levels": {k: round(float(last[k]), 2) for k in ("close", "sma50", "sma200", "ema8", "ema21")},
            "position": position,
            "x2": {"enabled": bool(use_x2), "day1": bool(use_x2 and pos is not None and _x2_day1(last))},
            "recent_trades": trades[-RECENT_TRADES:],
            "stop_buffer_pct": STOP_BUFFER * 100,
        }
    except Exception:  # noqa: BLE001 - display layer must never break analyze
        logger.warning("weekly strategy: computation failed", exc_info=True)
        return None
