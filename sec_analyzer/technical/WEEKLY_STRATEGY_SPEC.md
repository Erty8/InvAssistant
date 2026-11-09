# Weekly SMA200 / EMA8-21 strategy ("H2") — spec

Binding contract for `sec_analyzer/technical/weekly_strategy.py`. The rules were
agreed with the user on 2026-11-05/06 after a series of exploratory backtests
(scratchpad scripts, not part of this repo). This module is a **display layer**:
it reports what the rule says on the latest bars; it never feeds valuation,
verdicts or the swing score.

## 1. Inputs

A daily OHLC frame (`Open`, `High`, `Low`, `Close`, ascending date index), as
returned by `fetch.prices.get_price_history`. At least 201 bars are required;
otherwise the result is `None`.

Indicators (all on `Close`): SMA50, SMA200, EMA8, EMA21 (`ewm(span, adjust=False)`).

A **week** is an ISO calendar week. A bar is the **week end** when the next bar
starts a new week; the last bar of the frame is a week end only if it falls on a
Friday.

## 2. Entry rules (long only)

1. **Above SMA200** — if a week-end close is above SMA200, buy at the **open of
   the first bar of the next week**. Skipped if that open is at or below the
   initial stop.
2. **Below SMA200 (two-Friday confirmation)** — at a week-end close where
   `close < SMA200`, `EMA8 > EMA21` and the close is above both EMAs, buy **at
   that close** if the *previous* week end also met an entry condition (rule 1's
   or rule 2's). Not on a bar where the model was just stopped out.

## 3. Stop (exit only via the stop, plus optional X2)

- Tiered stop `T = SMA50 × (1 − b)` if `close > SMA50 > SMA200`, else
  `SMA200 × (1 − b)`, with buffer `b = 5%`.
- Initial stop: `T` for rule-1 entries; `EMA21 × (1 − b)` for rule-2 entries.
- The stop is recomputed **once a week** from the week-end bar (applied from the
  next week's first bar) and only ever raised. For a rule-2 position the
  candidate is `EMA21 × (1 − b)` while the close is below SMA200 and `T` once it
  is above.
- Intraday fill: if a bar's low touches the stop, the position is sold at the
  stop, or at the open when the bar gaps below it (entry bar: at the stop).

## 4. Optional X2 exit (daily, default on)

Day 1 (in position): close below SMA50, close more than 10% above SMA200, and the
day's high within 2% of SMA50 (`high ≥ 0.98 × SMA50`). Day 2: if the close is
again below SMA50, sell **at day 2's close**. Requires daily monitoring.

## 5. Output

`compute_weekly_strategy(df, use_x2=True) -> dict | None` with:

| key | meaning |
|---|---|
| `as_of` | date of the last bar (ISO) |
| `signal` | one of `BUY_NEXT_OPEN`, `BUY_FRIDAY_CLOSE`, `FIRST_FRIDAY`, `WAIT`, `IN_POSITION` — what the rules say **if the current week closes at the last price** |
| `label` | English one-line description of `signal` |
| `entry_stop`, `entry_stop_pct` | initial stop if a buy happens now (`None` otherwise) |
| `levels` | `close`, `sma50`, `sma200`, `ema8`, `ema21` |
| `above_sma200` | bool |
| `prev_week_condition` | whether the last completed week end met an entry condition |
| `position` | the model's own hypothetical position from replaying the rules over the frame (`None` when flat): `kind`, `entry_date`, `entry_price`, `stop`, `stop_pct`, `next_stop` |
| `x2` | `{"enabled", "day1"}` — `day1` true when today is an X2 day 1 for the model position |
| `recent_trades` | the last 5 closed model trades |

`position` and `recent_trades` describe the rule's own replayed trades, **not**
the user's account; the replay starts at the first bar of the frame, so they can
depend on how much history was loaded.

## 6. Evidence and limits (as tested, 2026-11-06)

Five baskets (chips without MU; UBER/NFLX/RDDT; GLD/SLV/USO/BTC-USD; eight sector
ETFs; HIMS/EOSE/RKLB/SOFI/HOOD), 2025-11 → 2026-11 and 2020-11 → 2025-10:

- H2 beat buy-and-hold in 3 of 5 baskets in 2025-26 (chips without MU,
  UBER/NFLX/RDDT, commodities + BTC; not sector ETFs or the volatile basket) and
  trailed it in all 5 in 2020-25. In 2007-12 (sector ETFs) it made +28% with a
  −20% max drawdown versus +23% / −51% for buy-and-hold.
- X2 beat H2 in 5 of 10 basket/period cells (no average return edge once MU and
  APP were excluded) but gave a shallower drawdown in 7 of 10.
- Profile: crash protection that costs return in long bull markets; very
  volatile names get little protection (stops far below price, repeated
  re-entries).

The sample is small (five baskets, two main periods, user-chosen names). This is
a tested rule, not investment advice.
