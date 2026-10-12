# Self-funded growth screen (`funding-screen`) — spec

Status: **APPROVED 2026-09-30** (English UI strings; §9 defaults accepted). This
file is the binding contract for `normalize/ttm.py`, `screener/funding.py` and
the CLI wiring.

## 1. Purpose and framing

A **research filter**. It narrows an index to companies whose trailing-twelve-month
(TTM) filings show growth funded by their own cash flow: high FCF margin, little
or no net debt, low dilution. It is **not a predictive signal**:

- It has never been backtested. A backtest on today's index membership would
  carry survivorship bias (see the repo notes on the factor v1 work).
- Passing is a statement about the filings, not about the stock's future return.
- The optional "price reading" (§6.4) is a **heuristic**. It is not engine output
  and must never be shown as a valuation verdict.

The CLI help, the console footer and the JSON `meta` block all say this.

## 2. Scope and constraints

- New files: `sec_analyzer/normalize/ttm.py`, `sec_analyzer/screener/funding.py`,
  `sec_analyzer/tests/test_ttm.py`, `sec_analyzer/tests/test_funding_screen.py`,
  plus a subcommand in `sec_analyzer/cli.py`.
- **Unchanged:** the valuation engine (`valuation/**`) and `normalize/concepts.py`.
  The screener keeps its own extra tag lists in `ttm.py` and `funding.py`. Adding
  `RevenueFromContractWithCustomerIncludingAssessedTax` to `CONCEPTS` changes
  engine inputs, so it needs a separate, reviewed change.
- The screener reads `CONCEPTS[...]` lists and appends its own fallbacks. It never
  mutates `CONCEPTS`.
- No new dependencies. The overlay uses only public or already-shared helpers:
  `reverse_dcf.implied_growth_with_status` and the same `cli._fetch_*` helpers
  that `calibrate.run_calibration` uses.
- **Language:** every user-facing string is English (CLAUDE.md sec_analyzer
  rule). The Turkish educational PDF stays out of the repo.
- **Deterministic.** Results never depend on the wall clock or randomness. The
  latest period is chosen from the data. Ties break on the latest `filed` date,
  then on tag list order.
- **Never crashes.** Each ticker runs inside `try/except`. A failure becomes an
  error row `{ticker, status: "error", error}` and is logged. Missing values
  print as `—`.

## 3. Data source

The screen reads SEC companyfacts through the existing cached
`fetch.companyfacts.get_company_facts(cik, client, no_cache)` and
`fetch.tickers.resolve_cik`. Only these forms are used: `10-K`, `10-K/A`,
`10-KT`, `10-Q`, `10-Q/A`. When two facts share a key, the most recently filed
fact wins.

## 4. TTM helpers (`normalize/ttm.py`)

All functions are pure functions of the companyfacts dict.

### 4.1 Period matching

- A fact counts as **annual** only if its duration is 340–390 days **and** its
  form starts with `10-K`. This is pitfall 1: AMZN-style 10-Qs also tag
  trailing-12-month durations, and those must not count as a fiscal year.
- A **YTD** fact starts within ±4 days of (FY end + 1 day), ends after the FY
  end, and lasts fewer than 340 days.
- The **prior-year match** for a period ending on E with duration d is the fact
  whose end is within ±10 days of E − 364 days and whose duration is within
  ±12 days of d. This covers pitfall 4: January fiscal years (ADSK, WDAY, CRWD)
  and 52/53-week years.
- The **prior FY** must end within ±10 days of FY end − 365 days.

### 4.2 `ttm_flow(facts, tags, unit="USD", want_end=None) -> dict | None`

For each tag, in list order:

- `F` = the latest annual end date. `A_F` = the annual value at `F`.
- `Y` = the latest YTD value, ending on `E`, with duration `d`.
- `P` = the prior-year YTD value, ending near `E − 364` days, with duration `d`.
- If `Y` and `P` both exist: `TTM = A_F + Y − P`, ending on `E`.
  Otherwise: `TTM = A_F`, ending on `F`.

Prior-year TTM:

- With YTD data: `A_{F−1} + P − P_{−1}`, where `P_{−1}` is the YTD value two
  years back.
- Without YTD data: the value `A_{F−1}`.
- If neither is available, `ytd_growth = Y / P − 1` is recorded instead.

Tag choice:

- With `want_end`, the first tag whose TTM ends on `want_end` wins (pitfall 3).
- Otherwise the tag with the latest `ttm_end` wins, with ties going to list order.

Return value: `{tag, fy_end, annual: {end: value}, ttm, ttm_end, ttm_prior,
ytd, ytd_prior, ytd_growth}`. Keys that do not apply are `None`.

### 4.3 `instant(facts, tags, end)`

Returns the first tag, in list order, that has a non-duration fact ending exactly
on `end`. The most recently filed fact wins.

### 4.4 `balance(facts, end) -> {debt, liquidity, net_debt}`

- `long_term` = the first match of `LongTermDebtNoncurrent`,
  `LongTermDebtAndCapitalLeaseObligations`, `LongTermNotesPayable`,
  `LongTermNotesAndLoans`, `SeniorLongTermNotes`, `ConvertibleNotesPayable`,
  `ConvertibleDebtNoncurrent`.
- `current` = the first match of `LongTermDebtCurrent`,
  `LongTermDebtAndCapitalLeaseObligationsCurrent`, `DebtCurrent`,
  `NotesPayableCurrent`, or 0 if none match. (Pitfall 2, ORCL:
  `LongTermNotesAndLoans` + `NotesPayableCurrent`.)
- If `long_term` is missing: `long_term` = the first match of `LongTermDebt`,
  `DebtInstrumentCarryingAmount`, `DebtLongtermAndShorttermCombinedAmount`, and
  `current` = 0. These tags already include the current portion, so adding it
  again would double-count.
- `short_term` = `CommercialPaper` or `ShortTermBorrowings`, or 0 if neither is
  present.
- `debt = long_term + current + short_term`.
- `liquidity` = `CashAndCashEquivalentsAtCarryingValue` (falling back to
  `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`) plus the first
  of `ShortTermInvestments`, `MarketableSecuritiesCurrent`,
  `AvailableForSaleSecuritiesDebtSecuritiesCurrent`.
- `net_debt = debt − liquidity`.
- Also returns `debt_tag_found: bool` — whether ANY debt tag (long-term,
  current, fallback or short-term list) matched at `end`. The screen uses it in
  §5.1 so that "no debt tags" is not silently read as "no debt".

All balance-sheet values are taken at the cash-flow `ttm_end`.

### 4.5 `diluted_share_change(facts, end)`

Uses `WeightedAverageNumberOfDilutedSharesOutstanding` (unit: shares).

- Primary: a quarter (80–100 days) ending within ±5 days of `end`, divided by
  the quarter ending within ±10 days of `end − 364` days, minus 1.
- Fallback: the latest annual value divided by the previous annual value, minus 1.
- Otherwise `None`.

### 4.6 `cover_shares(facts)`

Takes the latest `dei:EntityCommonStockSharesOutstanding` filing (by `filed`,
then `end`) and **sums every row with the same `accn`**. This is pitfall 5:
GOOGL reports one row per share class. Returns `(shares, as_of)`.

## 5. Screen (`screener/funding.py`)

### 5.1 Per-ticker metrics

`E` = the cash-flow TTM end, taken from `ttm_flow(CONCEPTS["OperatingCashFlow"])`.
Every other flow is requested with `want_end=E`, and a flow counts only if its
`ttm_end == E`.

| Field | Definition |
|---|---|
| `revenue` | `CONCEPTS["Revenue"]` + `RevenueFromContractWithCustomerIncludingAssessedTax`. If no tag ends on `E`, the row is an **error**. **Net-revenue guard** (mirrors valuation SPEC §19, screener-local): if `RevenuesNetOfInterestExpense` also has a TTM ending on `E` and differs from the chosen revenue by more than 5%, use it instead (note added). |
| `rev_growth` | `ttm / ttm_prior − 1`, falling back to `ytd_growth`; the fallback sets `rev_growth_basis = "ytd"` (else `"ttm"`) and adds a note. |
| `fcf` | `CFO − abs(CapEx)`. If **no CapEx tag exists at all** in the filing, CapEx counts as 0 and a `capex_missing` note is added. If a CapEx tag exists but no TTM ends on `E` (misaligned), `fcf` is `None` (→ `fcf_margin` fails) with a note — never a silent OCF-only FCF. |
| `fcf_margin` | `fcf / revenue` |
| `fcf_margin_trend` | `[FY−1, FY, TTM]` margins. TTM is included only if `E ≠ FY end`. FY revenue is matched within ±5 days. |
| `sbc`, `sbc_rev` | TTM SBC and `sbc / revenue` |
| `fcf_sbc_margin` | `(fcf − sbc) / revenue`. `None` if SBC is missing. |
| `ebitda` | `OperatingIncome + D&A`. Missing D&A counts as 0 and adds a note. `None` if OperatingIncome is missing. |
| `net_debt` | from §4.4, except: if `debt_tag_found` is false **and** interest expense (as resolved below) is > 0, `net_debt` is `None` (debt unknown → `leverage` fails) with a note. |
| `net_debt_ebitda` | `net_debt / ebitda` if `ebitda > 0`, else `None` |
| `interest_cov` | `EBIT / interest`. Interest is the TTM value; if that is not on `E`, the latest FY value is used only if that FY ends within 400 days of `E` (note names the FY end), else `None`. `None` if interest ≤ 0 or is missing. |
| `rule_of_40` | `100·rev_growth + 100·fcf_margin` |
| `dilution` | from §4.5 |

Interest tags, in order: `InterestExpense`, `InterestExpenseNonoperating`,
`InterestExpenseDebt`, `InterestAndDebtExpense`, `InterestExpenseNet`.

Row metadata: `ticker, name, cik, status, ttm_end, fy_end, is_ttm (E ≠ FY end),
revenue_tag, notes[]`. Money values are stored raw (USD). Rounding happens only
at render time.

### 5.2 Tests

A test either passes, fails, or is **not applicable (n/a)**.

| Id | Pass when | Fail when | n/a when |
|---|---|---|---|
| `fcf_margin` | ≥ 15% | < 15%, or missing | — |
| `leverage` | net debt ≤ 0 (net cash), or net debt / EBITDA < 1.0 | net borrower with ratio ≥ 1.0, or EBITDA ≤ 0, or missing | — |
| `interest_cov` | coverage ≥ 8.0 | coverage < 8.0 | net debt ≤ 0 (the test does not apply), **or** coverage is missing |
| `rule_of_40` | ≥ 40 | < 40, or missing | — |
| `dilution` | ≤ 3.0% | > 3.0% | missing |

Threshold comparisons use the value rounded to 6 decimals (fractions) so that
float noise cannot flip a boundary case (e.g. 103/100 − 1 is exactly 3%).

Missing data counts as a **fail** for the three core tests: missing evidence does
not pass a filter. For coverage and dilution it counts as **n/a**, and the
reason goes to `notes`.

Group: **A** = 0 fails, **B** = 1 fail, **C** = 2 or more. Error rows have no
group.

Sort order: group A→C, then `rule_of_40` descending (missing last), then ticker.
Error rows come last, sorted by ticker.

### 5.3 Foreign filers (pitfall 6)

If a company has no `10-K` annual data (20-F/40-F filers such as ASML, ARM, PDD,
CCEP, FER, TRI, NBIS), it gets the error row `"no 10-K annual data (foreign
filer 20-F/40-F?)"`.

## 6. Valuation overlay (`--valuation`)

This runs only for rows that are not errors. It is fully LLM-free.

1. Run the same headless pipeline as `calibrate.run_calibration`
   (`provider="script"`, a single FRED fetch per run). Rather than copying that
   block, it is factored into a small helper that `calibrate` and the screen
   share. **Engine behaviour is unchanged**, and `test_calibrate` must still pass.
2. From the result, take `price` and `price_as_of`, and `discount_rate` and
   `terminal_growth` from `valuation["assumptions"]["base"]`.
3. `shares` = `cover_shares` (§4.6), because it is point-in-time with the TTM.
   If that is missing, use `metrics["shares"]`. Record the source.
4. **Guards** (evaluated before the solve; mirror what the engine/calibrate
   already enforce). Any hit → `implied_growth = None`, `bracket_status =
   "no_data"`, and a human-readable `reason`; P/FCF is also suppressed for the
   price guard:
   - `metrics["price_reliable"] is False` → reason "unreliable price" (valuation
     SPEC §17); `p_fcf` is `None` too.
   - `valuation["sector_type"]` in {financial, reit} or `valuation["dcf"]
     ["enabled"] is False` → "FCF-DCF not applicable (financial/REIT)" (valuation
     SPEC §8).
   - `fcf − sbc ≤ 0` or SBC missing → "FCF−SBC ≤ 0" / "SBC missing". A
     non-positive base has no meaningful implied growth (engine `_select_fcf0`
     returns `None` in the same case); passing it to the solver would falsely
     report "above bracket".
   - The row carries the `capex_missing` note → "CapEx not tagged".
   - `discount_rate < 0.07` or `terminal_growth > 0.04` (outside the engine's
     clamp bounds, e.g. the engine-error path) → "assumptions out of bounds".
5. Calculations:
   - `p_fcf = price·shares / fcf`, only if `fcf > 0` (FCF before SBC, the
     engine's display convention; the column is labelled "P/FCF (pre-SBC)").
   - `(implied_growth, bracket_status) = reverse_dcf.implied_growth_with_status(
     price, fcf − sbc, terminal_growth, discount_rate, shares, 0.0)`.
     `dilution_rate = 0.0` matches the engine (SBC already expensed in fcf0).
   - **Price reading (heuristic, not engine output):** `gap = implied_growth −
     rev_growth`. Implied growth is the constant 5-year growth of FCF−SBC (fading
     to terminal growth by year 10); `rev_growth` is one trailing year of sales
     growth. The gap therefore also absorbs margin change, SBC trend and the
     discount-rate choice (≈3pp of implied growth per 1pp of discount rate).
     - `gap > +10pp` → `"price implies FCF growth well above TTM sales growth"`.
     - `gap < −10pp` → `"price implies FCF growth well below TTM sales growth"`.
     - Otherwise `"price-implied FCF growth near TTM sales growth"`.
     - Non-`ok` status: `above_bracket` → `"implied growth above +60% (off
       scale)"`, `below_bracket` → `"implied growth below −20% (off scale)"`,
       `no_data` → `"n/a"` plus `reason`.
     - If `rev_growth_basis == "ytd"`, the reading carries a "(YTD growth basis)"
       suffix.
6. For auditability the block also carries the engine's own
   `valuation["reverse_dcf"]["implied_growth"]` (and `fcf0` / `fcf0_source` if
   present) as `engine_implied_growth` etc., plus `discount_rate`,
   `terminal_growth`, `shares_source`, `price_as_of`. When the engine's fcf0 is
   known and `|fcf0_ttm / fcf0_engine − 1| > 0.5`, add a `base_divergence` note.
7. If the overlay fails for one ticker, only that row's `valuation` block becomes
   `{error}`. The screen row stays.

Known limitations (disclosed in the disclaimer, not fixed in v1): cover-page
shares are basic, not diluted (engine parity); a split after the latest filing,
or share classes with unequal economics (BRK), distorts market cap; under
FCFE-direct, cash counts only through its interest income in CFO, so net-cash
names (which this screen selects for) show somewhat higher implied growth.

Known divergence, documented rather than fixed: this TTM base is **not** the
engine's `_select_fcf0`. With FY inputs it reproduced the engine's
`implied_growth` for 42 of 48 names. Among the comparable names, the exceptions
were APP, ALNY and VRTX, where the engine picks a different base.

## 7. CLI

```
python -m sec_analyzer.cli funding-screen
    [--index nasdaq100|sp500]   (default nasdaq100; also accepts ndx)
    [--tickers A,B,C]           (overrides --index)
    [--valuation]
    [--json OUT.json]
    [--no-cache] [--verbose | --quiet]
```

- The console shows one table per group (A, B, C) with these columns: ticker,
  TTM end, revenue growth, FCF margin, FCF-SBC margin, trend, ND/EBITDA,
  coverage, Rule of 40, dilution, failed tests. With `--valuation` it adds P/FCF,
  implied growth and the price reading. Error rows follow the tables, then the
  §1 disclaimer.
- `--json` writes `{meta: {universe, generated_from: "SEC companyfacts",
  disclaimer, thresholds}, rows: [...]}`. Missing values are written as `null`
  in the JSON and shown as `—` on the console.
- The screen itself persists nothing in v1. Note: `--valuation` reuses the
  calibrate pipeline, which (like `analyze`/`calibrate`) caches normalized
  financials via `save_normalized`; that is the existing cache, not screen output.
- Console headers: "P/FCF (pre-SBC)", "Impl. FCF-SBC g (5y)".

## 8. Tests (hand-verified, offline, synthetic companyfacts fixtures)

1. TTM formula: FY 100, YTD 9M 80 against a prior 9M of 70 → TTM 110; the prior
   TTM and the growth are checked by hand.
2. A January fiscal year (e.g. an FY ending 2026-01-31, with a 52/53-week shift
   of a few days) still matches the prior-year periods.
3. AMZN trap: a 10-Q that tags a 365-day duration is not taken as the FY.
4. ORCL debt: `LongTermNotesAndLoans` + `NotesPayableCurrent` are summed. Also
   the `LongTermDebt` fallback does not double-count the current portion.
5. Revenue tag alignment: the `...IncludingAssessedTax` tag is chosen when it is
   the only one ending on the CFO end date.
6. Multi-class cover shares are summed per `accn`.
7. Foreign filer / no 10-K → error row, and the CLI exits normally.
8. Test rules: interest coverage is n/a when net cash, pass/fail boundaries sit
   exactly at the thresholds (15 / 1.0 / 8 / 40 / 3), and the group counts and
   sort order are as specified.
9. Missing values render as `—`, and there are no `None`/`nan` strings in the
   console output.
10. Overlay: `fcf0 = fcf − sbc` is passed through, missing SBC gives `no_data`,
    and the ±10pp reading is checked at its boundaries. The engine is
    monkeypatched, so the tests make no network calls.
11. Determinism: two runs on the same fixtures give identical output.

## 9. Open points (defaults above unless you say otherwise)

- a) The prototype also skipped the coverage test when gross debt < $0.5B. This
  spec drops that floor and uses only the net-borrower rule you described.
- b) The spec uses cover shares for the overlay (§6.3). The engine's diluted
  shares are the alternative.
- c) The default index is `nasdaq100`, because that is where the prototype was run.

## 10. Changelog

- **2026-09-30, amendment 1** (after valuation-reviewer pass). Adds engine-parity
  guards to the overlay (§6.4: non-positive FCF−SBC, financial/REIT, unreliable
  price, untagged CapEx, out-of-bounds assumptions); closes two silent-pass
  paths (§5.1: misaligned CapEx → FCF `None`; no debt tags + interest expense →
  net debt unknown); adds the screener-local net-revenue guard (§5.1); bounds
  the interest FY fallback to 400 days; rounds before threshold comparisons
  (§5.2); rewords the price reading so it no longer implies "acceleration"
  (§6.5); adds engine reverse-DCF values for audit (§6.6); documents known
  share/cash limitations and the existing normalized-financials cache write.
