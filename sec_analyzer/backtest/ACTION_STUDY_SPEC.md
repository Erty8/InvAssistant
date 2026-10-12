# ACTION_STUDY_SPEC.md — Technical action validation study (binding contract)

Status: **binding, pre-registered.** Written before any result was seen. Code
that contradicts this file is wrong; results must not be used to edit the
hypotheses or thresholds below after the fact.

Purpose: check whether `sec_analyzer/technical/action.py`'s fixed thresholds
(`_EXTENDED_ATR_MULT = 3.0`, `_FUEL_LOW_MIN = 2`, the horizon reversal
tuning, the price-turn requirement) separate better forward returns from
worse ones — or not.

Same governing principle as `SWING_STUDY_SPEC.md`: **evaluation, not
optimization.** No threshold sweep, no picking the best-looking value. If a
hypothesis fails, the report says so; changing a threshold afterwards requires
a *new* pre-registration and out-of-sample data, not a re-run of this study.

## 1. Inherited limitations

All four limitations of `SWING_STUDY_SPEC.md` §1 apply verbatim (survivorship
bias dominant, free price source, no costs, overlapping observations) and are
carried in every result, plus:

5. **No sector-relative strength.** The live pipeline folds sector-ETF
   relative strength into momentum; the study uses SPY-only relative
   strength (no SIC lookup per ticker/date). Momentum scores may differ
   slightly from a live run.

## 2. Fixed configuration

| knob | value |
|---|---|
| universe | S&P 500 (today's list), every 4th ticker in alphabetical order — deterministic, sector-spread subsample (compute budget) |
| rebalance | last trading bar of each month (SPY calendar) |
| date range | 2015-01-01 .. latest date with 63 forward bars |
| action horizon | `"1y"` (the app default) |
| forward horizons | **10, 21, 63** trading days — all always reported |
| primary metric | **date-demeaned forward return**: ticker's forward return minus the mean forward return of all observations on the same date (isolates selection from market timing) |
| secondary metrics | excess vs SPY; raw forward return |
| point-in-time | exactly `SWING_STUDY_SPEC.md` §3: slice before computing, SPY sliced the same way, forward window starts at bar d+1 |
| inference | **date-cluster bootstrap** (resample rebalance dates with replacement), `random.Random(0)`, 1000 resamples, 95% interval |

## 3. Pre-registered hypotheses

Each contrast is `mean(A) − mean(B)` of the primary metric.

| id | A | B | predicted sign | what it tests |
|---|---|---|---|---|
| H1 | action = BUY | action ∈ {REDUCE, AVOID} | > 0 | the action labels rank forward returns at all |
| H2 | uptrend, fuel = low (≥2 exhaustion signs) | uptrend, fuel ∈ {strong, moderate} | < 0 | "stretched → wait for pullback" |
| H3 | uptrend, `extended` check fired (≥3× ATR above SMA50) | uptrend, not fired | < 0 | the 3× ATR extension threshold alone |
| H4 | downtrend, action = BUY (reversal confirmed) | downtrend, action = AVOID | > 0 | reversal confirmation in a downtrend |
| H5 | non-uptrend, confirmations ≥ needed **and** a price turn | non-uptrend, confirmations ≥ needed **without** a price turn | > 0 | the price-turn requirement added to action.py |

Verdict per hypothesis, judged at the **21-day** horizon:

- **supported** — the 95% interval excludes 0 on the predicted side, and the
  point estimates at 10 and 63 days have the predicted sign;
- **contradicted** — the 95% interval excludes 0 on the opposite side;
- **inconclusive** — anything else, including either side having n < 30.

Stability check (reported, not part of the verdict): the 21-day point
estimate for each hypothesis on the first and second half of the rebalance
dates separately.

## 4. Descriptive tables (no verdict attached)

- Per action: n, mean/median of each metric per horizon, hit rate (primary > 0).
- Uptrend fuel gradient: primary metric by exhaustion-sign count 0, 1, 2, 3+.
- Each uptrend exhaustion check individually: fired vs not fired.
- Extension profile: primary metric by `(price − SMA50) / ATR` bins
  `<0, 0–1, 1–2, 2–3, 3–4, 4–5, ≥5`. Shown so the reader can see the shape;
  it is **not** licence to move the threshold (see header).

## 5. Module + CLI

`sec_analyzer/backtest/action_study.py::run_action_study(index, start, end,
tickers, max_workers, progress_cb) -> dict` — reuses `swing_study`'s
benchmark cache, month-end calendar, and process-pool fallback. Never raises
for one bad ticker. JSON-native output; deterministic apart from
`generated_at`.

```
python -m sec_analyzer.cli backtest action [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                           [--sample-every N] [--workers N] [--json PATH]
```

Prints the hypothesis verdicts first, then the descriptive tables, the
limitations, and `BACKTEST_DISCLAIMER`.

## 6. Results — first run (2026-09-24)

Configuration exactly as §2: 126 tickers (1 skipped: AVB, no price data),
141 month-ends 2015-01-30 .. 2026-09-23, 16,750 observations.

**All five hypotheses: inconclusive.** Every 21-day contrast is within
±0.35pp, every 95% interval straddles 0, and the two halves of the sample
disagree in sign for H1, H2, H4 and H5.

| id | 21d A−B | 95% CI | halves |
|---|---|---|---|
| H1 BUY vs REDUCE/AVOID | +0.04pp | −0.62 .. +0.73 | +0.43 / −0.29 |
| H2 fuel low vs not | +0.02pp | −0.38 .. +0.41 | +0.35 / −0.29 |
| H3 ≥3× ATR vs not | −0.24pp | −0.63 .. +0.12 | −0.26 / −0.22 |
| H4 downtrend BUY vs AVOID | +0.34pp | −0.42 .. +1.12 | +1.35 / −0.73 |
| H5 price turn vs volume-only | +0.06pp | −0.87 .. +0.89 | +1.22 / −1.01 |

Descriptive (no verdict): hit rates per action are 47–50% against the
same-date mean; on raw and SPY-excess returns AVOID/REDUCE did not
underperform BUY (21d raw: BUY +1.29%, AVOID +2.03%). Uptrend names with 3+
exhaustion signs and with "momentum slowing" did slightly *better*, the
opposite of the stretched → wait premise, but these were not tested
hypotheses.

Conclusion: in this sample the technical action has **no demonstrated
predictive value** for 10/21/63-day relative returns. Per the header, the
thresholds were not changed in response. The action remains a description of
trend state and risk levels, not a validated signal.

Caveat added 2026-09-25 after `FACTOR_STUDY_SPEC.md` §9: this study uses
the same today's-membership universe. That bias favours past losers and
downtrends, so "AVOID did not underperform BUY" may be partly or wholly a
selection artefact. The null result is therefore weaker evidence than it
first looked, in both directions. It does not validate the action either.

Follow-up (same day): the display was changed accordingly. `action` stays as
an internal code (so this study stays reproducible), the UI shows a
trend-state `label` instead ("UPTREND — STRETCHED", "DOWNTREND — NO TURN",
...), the guidance text no longer tells the reader to buy or sell, and every
read carries `evidence_note` summarising this result.
