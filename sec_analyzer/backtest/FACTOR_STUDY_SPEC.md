# FACTOR_STUDY_SPEC.md — Price-factor study v1 (binding, pre-registered)

Status: **binding, pre-registered.** Written before any factor was computed
on any data. Results must not be used to edit the features, directions,
horizons, costs, or verdict rules below; a change needs a new version of this
file and data not used here.

Purpose: test whether four price factors with published evidence — and one
fixed composite — rank S&P 500 stocks by forward return, on data that played
no part in forming the hypotheses. It is the first step of the plan that
followed the null result in `ACTION_STUDY_SPEC.md` §6.

Governing principle (ROADMAP.md, `SWING_STUDY_SPEC.md`): **evaluation, not
optimization.** No parameter search. Every feature and horizon is reported.

## 1. Samples

- **Universe:** today's S&P 500 list (`data/sp500.csv`), tickers sorted
  alphabetically.
- **Discovery sample:** every 4th ticker starting at the first (`[::4]`) —
  exactly the names the action study used. The momentum/reversal hints that
  motivated this study came from that sample, so it is reported but carries
  **no verdict**.
- **Confirmation sample:** all other tickers. **Verdicts use this sample only.**
- **Eligibility at date d:** at least 253 bars ending at d, positive closes,
  and a total-return price source (`fetch.prices.is_total_return_basis`;
  split-only Nasdaq series are excluded and counted).
- **Minimum cross-section:** a date with fewer than 100 eligible confirmation
  names is skipped (quintiles of < 20 names are too noisy).

## 2. Fixed configuration

| knob | value |
|---|---|
| rebalance | last trading bar of each month (SPY calendar) |
| period | 2005-01-31 .. latest date with a full forward window, per horizon |
| subperiods (stability) | 2005-01 .. 2014-12 and 2015-01 .. end |
| forward horizons | **21, 63, 126** trading days — all reported |
| forward return | close of bar d+1 → close of bar d+1+h (no same-bar measurement) |
| portfolios | quintiles by feature **within each date**, equal-weighted |
| spread | per date: mean forward return of the predicted-best quintile minus the predicted-worst quintile; the statistic is the mean over dates |
| inference | moving-block bootstrap over the date series of spreads, block = ⌈h/21⌉ months (overlapping windows), `random.Random(0)`, 2000 resamples, 95% interval |
| costs | 10 bp per side. Monthly cost = (turnover of best quintile + turnover of worst quintile) × 2 sides × 10 bp, where turnover = share of names replaced vs the previous date |

## 3. Features (all computed from closes on or before d)

| id | feature | definition | predicted best | primary horizon |
|---|---|---|---|---|
| F1 | 12-1 momentum | C[d−21] / C[d−252] − 1 | highest | 63 |
| F2 | 1-month reversal | C[d] / C[d−21] − 1 | **lowest** | 21 |
| F3 | 52-week-high proximity | C[d] / max(C[d−251..d]) | highest | 63 |
| F4 | low volatility | stdev of daily log returns, last 63 bars | **lowest** | 63 |
| F5 | composite | mean of the within-date percentile ranks of F1 (high), F3 (high) and F2 (low) | highest | 63 |

Bars are counted in the ticker's own trading history.

Note on F4: the published low-volatility effect is mainly about *risk-adjusted*
return; in raw returns it can fail in strong bull markets. The hypothesis is
still judged on raw spread as registered. Per-quintile return / stdev is also
reported, but without a verdict.

## 4. Verdicts (confirmation sample, primary horizon)

- **supported** — the 95% interval of the mean spread excludes 0 on the
  predicted side **and** both subperiod means have the predicted sign;
- **contradicted** — the 95% interval excludes 0 on the opposite side;
- **inconclusive** — anything else.

Separately, for every factor, **survives costs** = mean monthly (21-day) spread
minus the mean monthly cost > 0. A factor that is supported but does not
survive costs is reported as such, not rounded up.

## 5. Descriptive output (no verdicts)

- The quintile table (Q1..Q5 mean spread vs same-date mean, hit rate, stdev)
  for every feature and horizon, confirmation and discovery separately.
- Raw (not demeaned) mean forward return per quintile.
- Discovery-sample spreads next to the confirmation ones.

## 6. Limitations (carried in every result)

1. **Survivorship bias:** the universe is today's index membership. Delisted
   and removed names are missing. This inflates long-only returns and
   probably flatters momentum (past losers that died are absent).
   *[Wording as pre-registered. Shown wrong about direction by §9: with
   today's membership the bias favours past losers and hurts momentum.]*
2. **Free price data:** yfinance/Yahoo adjusted closes; some tickers are
   excluded for split-only sources.
3. **Costs are a flat assumption:** no market impact, no borrow for the
   short leg. The short leg is a measurement device, not a recommendation.
4. **Overlapping windows** at 63/126 days are handled by the block bootstrap,
   but the interval is still approximate.
5. **One market, one era:** large US caps, 2005–2026.

## 7. What a positive result would and would not license

It would license **step 2**, a new pre-registration: combine the surviving
factor(s) with the valuation verdict and show a cross-sectional **ranking
score** in the app. It would **not** license per-stock BUY/SELL labels or
tuning the lookbacks above.

## 9. Results — v1 run (2026-09-25)

503 tickers (6 failed to load, 0 excluded for price basis). Confirmation
sample: 374 tickers, 261 month-ends 2005-01-31 .. 2026-09-24.

| id | primary | spread | 95% CI | 2005–14 / 2015– | verdict | survives costs |
|---|---|---|---|---|---|---|
| F1 12-1 momentum | 63d | −0.53pp | −2.40 .. +1.07 | −1.56 / +0.38 | inconclusive | no |
| F2 1-month reversal | 21d | −0.00pp | −0.51 .. +0.51 | +0.12 / −0.11 | inconclusive | no |
| F3 52w-high proximity | 63d | −2.78pp | −4.82 .. −1.17 | −3.68 / −2.00 | **contradicted** | no |
| F4 low volatility | 63d | −4.04pp | −6.23 .. −2.23 | −4.13 / −3.95 | **contradicted** | no |
| F5 composite | 63d | −0.99pp | −2.43 .. +0.28 | −1.79 / −0.29 | inconclusive | no |

The discovery sample points the same way for every factor.

The verdicts stand as registered. **Interpretation (post hoc, not a
verdict):** the pattern matches the survivorship bias of §6.1, and the bias
runs in the opposite direction to the one §6.1 guessed. The universe is the
set of companies that are in the S&P 500 in 2026. A stock that was volatile
and deeply below its high in 2008 is in this sample *because* it later
recovered and grew into the index. The ones that did not recover are absent.
That is look-ahead through the universe choice, and it favours exactly the
names F3 and F4 predict to be worst, and past losers against F1. Three
observations fit:

- the damage is concentrated in the worst-looking quintiles (F3/F4 Q1: +1.8
  and +2.7pp vs the date mean, the highest raw stdev);
- it is larger in 2005–14, when today's membership lies further in the
  future (F1 −1.56 vs +0.38; F3 −3.68 vs −2.00);
- on return per unit of risk, low volatility does rank as the literature
  says (Q5 0.296 vs Q1 0.265), even though it loses on raw return.

**Consequence:** with today's membership as the universe, this design cannot
tell a real factor effect from the selection effect. v1 is therefore
evidence about the *method*, not about the factors. A v2 needs point-in-time
index membership and must be pre-registered anew. The
§6.1 wording in code output was corrected to state the bias direction
honestly; the hypotheses and rules above are unchanged.

## 10. v2 — point-in-time membership (pre-registered 2026-09-25, before any v2 run)

v2 changes **only the universe**. §3's features, directions and primary
horizons, §2's horizons, costs, bootstrap and subperiods, and §4's verdict
rules are all carried over unchanged. No lookback, threshold or direction may
differ from v1.

**Universe at date d:** the S&P 500 members in effect on d according to
`fetch/sp500_history.py` (dataset `fja05680/sp500`, MIT; the SHA-256 of the
file used is recorded in the result). A member is *covered* at d if its
ticker has a total-return price series with a bar within 5 days of d and at
least 253 bars of history, which is the same eligibility as v1.

**Samples:** the whole point-in-time universe. v2 has no discovery split.
Its hypotheses predate v1 and were not changed by it. v1 used the same
price era, which is stated here rather than hidden.

**Coverage:** for each date, covered members ÷ members. Uncovered members
are mostly delisted or renamed symbols that Yahoo no longer serves (Yahoo
keeps current listings only).

- **Primary analysis** (verdicts): every date with ≥ 100 covered members.
- **Secondary analysis** (reported; the verdict is flagged *robust* only if
  it matches): only dates with coverage ≥ 90%.

**Forward windows** that run past a stock's last bar (delisting, merger) are
dropped. The dropped share per horizon is reported. This residual bias cuts
both ways: acquisitions tend to be positive and bankruptcies negative.

**Diagnostic, no verdict:** each factor's v2 primary-horizon spread next to
v1's. If v1's reversals came from universe look-ahead, v2 should move toward
the literature signs. No conclusion is registered in advance.

**CLI:** `backtest factors --pit` runs v2; without the flag it runs v1.

## 11. Results — v2 run (2026-09-25)

Membership file sha256 `36326709d46d…` (last change 2026-08-18). 977
symbols were members at some point since 2005. 621 had price data; 313 did
not (298 unavailable from Yahoo, the rest no total-return data or too short).
121 of the covered symbols are former members, which is the ticker-recycling
exposure. Mean coverage was 0.76, rising from 0.53 (2005) to 0.99 (2026);
57 dates reach ≥ 90% coverage, all from 2022 on. Dropped forward windows:
0.0%, because every covered symbol still trades today. That is the residual
survivorship bias in its plainest form.

| id | v2 primary spread | 95% CI | 2005–14 / 2015– | verdict | robust | v1 |
|---|---|---|---|---|---|---|
| F1 momentum | −0.72pp | −2.90 .. +1.08 | −1.81 / +0.24 | inconclusive | yes | −0.53 |
| F2 reversal | +0.19pp | −0.34 .. +0.75 | +0.36 / +0.04 | inconclusive | yes | −0.00 |
| F3 52w high | −1.59pp | −3.93 .. +0.27 | −2.61 / −0.70 | inconclusive | yes | −2.78 |
| F4 low vol | −1.81pp | −4.22 .. +0.09 | −2.39 / −1.31 | inconclusive | yes | −4.04 |
| F5 composite | −0.52pp | −2.25 .. +0.94 | −1.33 / +0.19 | inconclusive | yes | −0.99 |

No factor survives costs in the primary analysis. In the secondary analysis
(57 dates, 2022–2026 only) every factor is also inconclusive. Momentum's
+2.02pp (CI −0.27 .. +4.39) there is one short, momentum-friendly stretch and
must not be read as a finding.

**Diagnostic (§10, no verdict):** moving to point-in-time membership roughly
halved F3's and F4's negative spreads, and both are no longer contradicted.
That is consistent with v1's contradictions being mostly an artefact of the
universe. Momentum barely moved, and its 2005–14 weakness is still there
(the 2009 rebound of beaten-down stocks is a documented momentum crash).

**Conclusion:** in large-cap S&P 500 stocks since 2005, none of these price
factors shows a reliable edge after costs, as far as free data can show. The
remaining coverage gap (≈ 30–47% of members missing before 2015) can only be
closed with paid delisted-inclusive data. A price-only ranking score is not
licensed (§7).

## 8. Module + CLI

`sec_analyzer/backtest/factor_study.py::run_factor_study(start, end) -> dict`.
Vectorized pandas per ticker (no
`compute_indicators`), deterministic apart from `generated_at`,
JSON-native output, never raises for one bad ticker.

```
python -m sec_analyzer.cli backtest factors [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--json PATH]
```
