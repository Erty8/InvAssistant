# Valuation Engine + Two-Phase Interpret — Implementation Spec

This is the binding contract for the `valuation/` package, the two-phase
`interpret` flow, the CLI verdict card, and the HTML report. All implementing
agents code against the shapes defined here. Architecture principle: **fair
value NUMBERS are computed by deterministic Python; the LLM only proposes
assumption ranges (phase 1) and comments on computed results (phase 2). Same
inputs must always produce the same numbers.**

Existing inputs (do not change their shape):
- `normalized` — from `normalize.normalizer.normalize_facts`; use
  `to_annual_series(normalized, concept)` → `{fy: value}`. Concepts include
  `Revenue`, `NetIncome`, `OperatingCashFlow`, `CapEx`, `Cash`, `LongTermDebt`,
  `LongTermDebtCurrent`, `SharesOutstanding`, `EPS`, `SBC`, `StockholdersEquity`.
  Annual entries also carry `end` (fiscal period end date, ISO string).
- `ratios` — list of per-FY dicts (`fy`, `net_margin`, `roe`, `fcf`, ...).
- `metrics` — from `normalize.metrics.compute_metrics` (keys: `price`, `shares`,
  `eps`, `net_debt`, `pe`, `ps`, `pfcf`, `revenue_cagr_3y`, `revenue_cagr_5y`,
  `sbc_revenue`, `shares_yoy`, `fcf`, `latest_fy`, `latest_fundamental_fy`, ...).
  `latest_fy` is the newest fiscal year across ALL series, including the
  `SharesOutstanding` cover-page (dei) series — used only for the current
  share count and market cap. `latest_fundamental_fy` excludes
  `SharesOutstanding` and is the fiscal year every OTHER fundamental read
  (EPS, revenue, FCF, CAGRs, NetIncome, StockholdersEquity, ...) is anchored
  to, because a filer's cover-page share count can carry a fiscal year newer
  than its financial statements actually report (e.g. AMZN) — anchoring
  fundamental reads to the wrong, data-less "ghost" fiscal year would collapse
  every downstream valuation. Wherever this spec says "latest FY" for a
  fundamental-data read below, the intended anchor is
  `normalize.metrics.resolve_fundamental_fy(metrics)` (falls back to
  `latest_fy` when `latest_fundamental_fy` is absent, e.g. in older test
  fixtures that construct `metrics` by hand), not `metrics["latest_fy"]`
  directly.
- `price_df` — pandas OHLCV DataFrame from `fetch.prices.get_price_history`
  (columns `Date, Open, High, Low, Close, Volume`), or `None`.
- `submissions` — raw dict from `fetch.companyfacts.get_submissions`; contains
  `sic` and `sicDescription` at top level.

## 1. Package layout

```
sec_analyzer/valuation/
  __init__.py      # re-export run_valuation, validate_assumptions
  dcf.py           # dcf_per_share(), project_fcf()
  reverse_dcf.py   # implied_growth()
  multiples.py     # multiples_history(), percentile_position()
  damodaran.py     # load_sector_data(), sector_medians()
  sector.py        # classify_sector(sic, normalized, metrics) -> sector_type
  sanity.py        # validate_assumptions() -> list[str] violations
  sensitivity.py   # sensitivity_matrix()
  triangulate.py   # triangulate() -> signals + confidence
  engine.py        # run_valuation() orchestrator
```

No new dependencies (pandas already available; bisection is pure Python;
Damodaran CSVs via `csv` stdlib or pandas).

## 2. Assumptions shape (phase-1 output, engine input)

```python
assumptions = {
  "bear": {"growth_5y": 0.08, "terminal_growth": 0.025, "discount_rate": 0.12,
            "story": "one sentence, Turkish"},
  "base": {...}, "bull": {...},
}
sector_type = "cyclical" | "financial" | "growth_unprofitable" | "mature" | "reit"
```
Rates are decimal fractions (0.08 = 8%), never percent numbers.

## 3. Sanity check — `sanity.validate_assumptions(assumptions, is_unprofitable: bool) -> list[str]`

Throughout this section, `discount_rate` is a levered COST OF EQUITY
(özkaynak maliyeti), never a WACC — the engine's DCF/revenue-DCF are
FCFE-direct (Sec.4), so the discount rate must be the rate equity holders
require, not a debt/equity blend.

Return a list of human-readable violation strings (empty = OK). Rules, per
scenario:
- `terminal_growth > 0.04` → violation
- `discount_rate < 0.07` (or `< 0.10` when `is_unprofitable`) → violation
- `discount_rate <= terminal_growth` → violation (Gordon undefined — never
  silently "fix" it)
- Else (Gordon defined) `discount_rate - terminal_growth < 0.045` →
  violation: a discount rate only a point or two above terminal growth
  implies an implausibly thin equity risk premium and over-values the
  perpetuity, even though the Gordon formula itself is defined. This rule and
  the previous one are mutually exclusive per scenario (`elif`) — the
  undefined-Gordon case is never double-reported.
- `growth_5y > 0.20` is allowed only because the model structure always fades
  after year 5 (total high-growth span ≤ 7y is satisfied by design); still add
  violation if `growth_5y > 0.60` (`sanity._GROWTH_5Y_HARD_MAX`, implausible).
  Raised from an earlier 0.40 ceiling (normalization Work Package 5): the
  TAM-share/implied-revenue-multiple "arrival point" flags (§4/§4a) are the
  real honesty mechanism for a hyper-growth assumption, and the old flat 40%
  ceiling clipped genuine hyper-growth (e.g. a filer like NVDA growing
  &gt;100% at points) before those flags could even evaluate it; 60% keeps a
  sane outer bound while letting the fade + arrival flags do the actual work.
- Missing/non-numeric field → violation naming the field.

### Clamping — `sanity.clamp_assumptions(assumptions, is_unprofitable: bool = False) -> tuple[dict, list[str]]` (F5)

Unlike `validate_assumptions` above (report-only), this actually rewrites
out-of-range values so every downstream calculation uses the same numbers
shown to the user. Per scenario: `terminal_growth` capped at 0.04; `growth_5y`
capped at 0.60 (`sanity._GROWTH_5Y_HARD_MAX`, raised from 0.40 by
normalization Work Package 5 — see `validate_assumptions` above for the
rationale); `discount_rate` floored at 0.07 (0.10 if `is_unprofitable`) —
each clamp appends a Turkish note. Then, on the already-clamped
`terminal_growth`/`discount_rate`, a minimum implied equity-risk-premium (ERP)
spread guard fires whenever `terminal_growth < discount_rate < terminal_growth
+ 0.045` (Gordon defined, but the spread is thinner than 4.5%): `discount_rate`
is raised to `terminal_growth + 0.045`, with a Turkish note — raising the rate
is the conservative direction (higher rate → lower value), exactly like the
discount-rate floor clamp. The `discount_rate <= terminal_growth` case is
deliberately NOT clamped by either of the above (the existing per-scenario
`ValueError` path stays the way it's surfaced) — there is no single "correct"
fix for an undefined Gordon term, but raising an already-defined-but-thin rate
is unambiguous. A missing/non-numeric field is left untouched. Also
checks `bear.growth_5y <= base.growth_5y <= bull.growth_5y` across scenarios
— a violation only adds a note, never a clamp (no single "correct"
reordering). Engine calls this right after `validate_assumptions` and uses
the clamped set for everything downstream; the output's `"assumptions"` key
(Sec.11) is this clamped set, not the raw phase-1 input.

**Float-boundary consistency fix (`sanity._ERP_SPREAD_EPS = 1e-9`,
normalization Work Package 2b):** `clamp_assumptions` raises a too-thin
`discount_rate` to exactly `terminal_growth + _MIN_ERP_SPREAD`, but that same
sum re-subtracted can round to `0.0449999... < 0.045` in IEEE-754 —
so a naive strict `< _MIN_ERP_SPREAD` check in `validate_assumptions` could
flag a value `clamp_assumptions` had *just* declared valid. Both the
validator's comparison (`discount_rate - terminal_growth < _MIN_ERP_SPREAD -
_ERP_SPREAD_EPS`) and the clamp's trigger condition (`tg < dr < tg +
_MIN_ERP_SPREAD - _ERP_SPREAD_EPS`) subtract this epsilon, so the two agree
at the boundary. This is not a cosmetic fix: callers that run clamp then
validate and discard the whole assumption set on any violation (e.g.
`rule_based._default_assumptions`) were silently discarding an already-valid,
just-clamped CAPM-based discount rate for a whole class of low-beta filers
before this fix — measured to be the single largest driver of the
calibration-basket undervaluation this normalization effort measured (see
VALUATION.md's calibration-methodology section).

## 4. DCF — `dcf.dcf_per_share(fcf0, growth_5y, terminal_growth, discount_rate, shares, dilution_rate=0.0) -> dict`

Deterministic, raises `ValueError` if `discount_rate <= terminal_growth` or
`shares` is falsy/<=0 or `fcf0` is None. **No `net_debt` parameter** (FCFE-
direct, see below) — `net_debt` stays in `metrics` for display only and never
enters the valuation math.

- Projection horizon 10 years. Growth in years 1–5 = `growth_5y` (constant).
  Years 6–10 fade linearly to terminal: `g_t_y = growth_5y + (terminal_growth
  - growth_5y) * (y - 5) / 5` for y in 6..10 (year 10 growth == terminal_growth).
- `fcf_y = fcf_{y-1} * (1 + g_y)`, fcf_0 = fcf0.
- `pv_y = fcf_y / (1 + r)^y`.
- Terminal value `TV = fcf_10 * (1 + g_t) / (r - g_t)`, discounted by `(1+r)^10`.
- `ev = sum(pv_1..10) + pv(TV)`; `equity = ev` (FCFE-direct — see below);
  `per_share = equity / effective_shares`.
- Dilution: `effective_shares = shares * (1 + dilution_rate) ** 5` (mid-horizon
  share count; document this choice in the docstring).
- Returns `{"per_share": float, "ev": float, "equity": float,
  "fcf_path": [10 floats], "tv": float, "effective_shares": float}` (`ev` and
  `equity` are equal — both keys kept for backward-compatible callers).

### FCFE-direct (no net-debt subtraction)
FCF = OCF − CapEx (US GAAP) is already a *levered* (equity) cash flow: interest
paid to debtholders is deducted inside operating cash flow before it ever
reaches this projection. Its discounted sum is therefore already an equity
value — subtracting net debt again would double-penalize leverage (once via
the interest expense embedded in every projected year's FCF, once again as a
lump-sum balance-sheet deduction). Same rationale applies to
`revenue_dcf.revenue_first_dcf`'s FCF-margin-derived cash flows. Consequently
`discount_rate` throughout this engine (DCF, revenue-DCF, reverse-DCF,
sensitivity, hyper-grower) is a levered COST OF EQUITY, never a WACC —
discounting an already-levered equity cash flow at a WACC would double-count
the leverage adjustment a WACC already bakes in.

### fcf0 selection (engine responsibility, SBC-adjusted)
The "latest FY"/"3-year window" anchor used throughout this selection (and by
the SBC-adjusted per-FY series it builds, `sbc_adjusted_fcf_by_fy`) is
`resolve_fundamental_fy(metrics)`, never the raw `metrics["latest_fy"]` — see
the "Existing inputs" note above. This keeps a cover-page/fiscal-year mismatch
(AMZN-style) from landing the FCF window on a fiscal year with no
financial-statement data at all, which would otherwise leave `fcf0` `None`
and collapse the DCF.

`fcf0` = latest-FY FCF net of SBC (`metrics["fcf"] - sbc_fy`, SBC treated as
`0.0` when missing — stock-based comp is a non-cash OCF add-back that this
engine treats as a genuine cash expense, Damodaran-style). If it is `None`,
non-positive, or deviates more than ±50% from the average of the **prior**
fiscal years in the 3-year window (`latest_fy-1`/`latest_fy-2`, whichever are
present; also SBC-adjusted), use the 3-year average (which DOES include the
latest year — smoothing, not exclusion) instead and set `fcf0_source =
"3y_avg"` plus a Turkish note; else `fcf0_source = "ttm"`. If no positive fcf0
can be derived at all, DCF returns `None` per-share values with a note (do not
raise).

**Why the deviation reference excludes the candidate year (2026-07 fix):**
the rule originally measured deviation against the 3-year average *including*
`latest_fy`. For a window `(a, b, L)` that makes the nominal 50% trigger
algebraically equivalent to `L > a + b` — roughly +100% versus the prior
years' own average — so the documented threshold was silently enforced at
double its stated value (UBER's FY2025 sat 143% above its prior-2 average but
measured as 65%; a borderline one-off spike at, say, +70% escaped entirely).
A reference must not contain the candidate it judges. When NO prior-year
value exists in the window, the deviation is not assessable and does not
fire (unchanged). The monotonic-ramp exception below and the fallback VALUE
(the inclusive 3-year average) are both unchanged. This same SBC-adjusted per-FY series is also the source for the
realized FCF CAGR used by reverse-DCF triangulation (Sec.5/Sec.10 F6) — it
does NOT change the *display* metrics (`ratios[...]["fcf"]`, the P/FCF
multiple), which stay conventional (non-SBC-adjusted).

Exception: when the >50% deviation trips but the trailing 3 fiscal years
(`latest_fy, latest_fy-1, latest_fy-2`, all present) form a monotonic ramp
(non-decreasing throughout, or non-increasing throughout), the deviation is
treated as structural growth/decline rather than a one-off spike -- the
latest-FY figure is kept (`fcf0_source` stays `"ttm"`) with a Turkish note
explaining why the average was not used instead. A spiky/oscillating series
(not monotonic) still falls back to the 3-year average as before.

### Dilution rule (engine responsibility)
Standard DCF always passes `dilution_rate = 0.0`. SBC is now expensed directly
in `fcf0` above, so separately diluting for `shares_yoy`/SBC-driven issuance
would double-count the same drag. (`dcf_per_share`'s `dilution_rate` parameter
stays in the API for callers, e.g. hyper-grower mode, that still need it.)
The hyper-grower and mid-growth revenue-first paths (Sec.3/Sec.8d), which DO
project a non-zero per-share dilution from `shares_yoy`, apply the same
SBC-double-count logic to THEIR dilution input instead — see "SBC-driven
dilution net-out" under Sec.3 (normalization Work Package 1).

### Scenario band (sensitivity-grid-derived, with a fallback)
Each scenario's `lo`/`hi` comes from a local 3×3 sensitivity grid around that
scenario's own point estimate — `growth_5y ± 2pp` × `discount_rate ± 1pp`
(reusing `sensitivity.py`'s own step constants), `terminal_growth` held fixed
— and is the min/max of the grid's usable (non-`None`) cells. If fewer than 2
cells are usable, falls back to the flat `per_share * 0.90` .. `per_share *
1.10` band (point estimate ±10%) with an additional Turkish note. Same
grid-based approach for hyper-grower scenarios (`start_growth ± 2pp` ×
`discount_rate ± 1pp` over `revenue_first_dcf`) and for P/B×ROE (`discount_rate
± 1pp`, re-clamping `fair_pb` at each point). All bands rounded to 2 decimals.
`fair_value_range` shape used everywhere downstream (CLI card, HTML, store):
```python
"fair_value_range": {
  "bear": {"lo": .., "hi": .., "growth": "%8 büyüme", "discount_rate": "%12",
            "note": <story>},
  "base": {...}, "bull": {...}
}
```
(`growth`/`discount_rate` are pre-formatted Turkish strings derived from the
numeric assumptions — keep numbers visible: "cam kutu".)

### Standard-DCF high-growth reporting flag (LEVER 4, engine wiring)

`engine._build_dcf_scenarios` (the function that runs this section's 3
scenarios for the standard/cyclical FCF-DCF) returns an additive third
element, `high_growth_flag: bool` -- `True` iff at least one scenario has a
valid, numeric `growth_5y` strictly greater than
`engine._STANDARD_DCF_HIGH_GROWTH_FLAG` (`= 0.40`). This standard two-stage
DCF has no arrival-point/implied-revenue-multiple safety net the way the
hyper-grower (§3/§4a) and mid-growth (§8d) revenue-first paths do, so a
`growth_5y` this high flowing through it is not cross-checked against any
TAM-share/revenue-multiple sanity gate. The flag is reporting-only -- it
appends one Turkish note naming the triggering scenario(s) and never
changes any computed per-share value or which scenario band is used. Surfaced
at `valuation["dcf"]["high_growth_flag"]` (Sec.11). **Known latent-risk
caveat** (surfaced by finance review, not yet acted on): for the `script`
provider this flag is currently inert in practice, because
`rule_based._default_growth_anchor` already clamps its own proposed
`growth_5y` to `_DEFAULT_GROWTH_CLAMP_MAX = 0.25` before it ever reaches
this check; the flag only has teeth for an LLM-proposed (`ollama`/
`anthropic`) assumption set that isn't similarly pre-clamped, so a young,
high-growth filer routed to the standard DCF (rather than hyper-grower/
mid-growth) by an LLM's own sector-type call could still be overvalued
without a safety net there. See ROADMAP.md's normalization-effort entry.

### Senaryo getirileri (`scenario_returns`) — companion structure

METODOLOJI.md §4 ("Senaryo tablosu") requires each scenario row to also show
the % return from the current price to that scenario's band edge, not just
the price target itself. This is **not** computed here in the valuation
engine — `fair_value_range` above is the complete, final output of `run_
valuation()` and is never mutated after the fact. Instead, `scenario_returns`
is a separate, sibling structure computed downstream, in the interpret
phase-2 post-processing step (`interpret/planning.py`'s
`compute_scenario_returns`, injected by `interpret/analyzer.py`'s
`_postprocess_phase2_result` — see Sec.12):

```python
"scenario_returns": {
  "bear": {"ret_lo_pct": .., "ret_hi_pct": ..},  # float|None, 1dp
  "base": {...}, "bull": {...},
}
```

`ret_lo_pct`/`ret_hi_pct` = `(band_edge / price - 1) * 100`, rounded to 1
decimal — the percentage (not fraction) return from the current price to
that band's `lo`/`hi` edge. `None` when the price is missing/non-positive or
the corresponding band edge is `None`. Always all three scenario keys, even
when every value degrades to `None`.

## 5. Reverse DCF — `reverse_dcf.implied_growth(price, fcf0, terminal_growth, discount_rate, shares, dilution_rate=0.0) -> Optional[float]`

Bisection on `growth_5y` over `[-0.20, 0.60]` (`reverse_dcf._BRACKET_LO`/
`_BRACKET_HI`; the upper bound was raised from 0.40 in lockstep with
`sanity._GROWTH_5Y_HARD_MAX`, normalization Work Package 5) so that
`dcf_per_share(...)["per_share"] == price`, tolerance `1e-4` on growth or 80
iterations. Uses base-scenario `r` and `g_t` (fixed). If no sign change over
the bracket or inputs unusable → `None`. **No `net_debt` parameter** (see
Sec.4's FCFE-direct note).

`reverse_dcf.implied_growth_with_status(price, fcf0, terminal_growth,
discount_rate, shares, dilution_rate=0.0) -> tuple[Optional[float], str]` is
the same bisection, plus a `status` that classifies *why* a `None` happened:
`"ok"` (root found, or price sits exactly on a bracket endpoint),
`"above_bracket"` (no sign change; model per-share stays below the market
price at both bracket ends — price implies growth above +40%),
`"below_bracket"` (no sign change; model per-share stays above the market
price at both ends — price implies growth below -20%), or `"no_data"` (a
required input is unusable). `implied_growth` is a thin wrapper that drops the
status and returns the same growth value it always has.

Engine (standard mode, F6): the reference growth rate to compare `implied_growth`
against is the **realized FCF CAGR** (5y, falling back to 3y — both/either
endpoint must be positive; from the same SBC-adjusted per-FY series as `fcf0`,
Sec.4), not a revenue CAGR — apples-to-apples, since the implied growth rate
itself is FCF growth. `reverse_dcf.realized_cagr_5y` carries this FCF CAGR;
`realized_label` becomes `"FCF 5y"`/`"FCF 3y"`/`None` (free text consumed by
`cli.py`/`rule_based.py` — key names unchanged). In hyper-grower mode, the
reverse-DCF pair shown is instead revenue-based: `implied_growth` =
`hyper_growth_detail["implied"]["growth"]` (from
`revenue_dcf.implied_start_growth`), reference = realized revenue CAGR
(`metrics["revenue_cagr_5y"]`/`_3y`), `realized_label` = `"gelir 5y"`/`"gelir
3y"`. The output dict gains an additive `bracket_status` key (`"ok"` /
`"above_bracket"` / `"below_bracket"` / `"no_data"`) from
`implied_growth_with_status` in standard mode; hyper-grower mode doesn't have
an equivalent status-returning revenue bisection, so it defaults to `"ok"`
there. An above/below-bracket status also appends a Turkish note ("Fiyat,
ters-DCF aralığının (%-20..%60) üzerinde/altında bir büyüme ima ediyor." —
the bounds are formatted straight from `reverse_dcf._BRACKET_LO`/
`_BRACKET_HI`, raised from `%-20..%40` to `%-20..%60` in lockstep with
`sanity._GROWTH_5Y_HARD_MAX`, normalization Work Package 5 — no hard-coded
text to fall out of sync) and is threaded into
`triangulate.triangulate(..., reverse_dcf_status=...)` so the reverse-DCF
signal can be "pahalı"/"ucuz" even when `implied_growth` is `None`.

## 6. Multiples — `multiples.multiples_history(normalized, price_df) -> list[dict]`

For every fiscal year that has an annual `end` date and a usable price:
`fy_price` = last `Close` on or before `end` (skip FY if price history doesn't
cover it). Then:
- `pe = fy_price / eps_fy` (eps > 0 else None)
- `ps = fy_price * shares_fy / revenue_fy` (revenue > 0 and shares else None)
- `pfcf = fy_price * shares_fy / fcf_fy` (fcf > 0 and shares else None)
- `ev_sales = (fy_price * shares_fy + net_debt_fy) / revenue_fy` (revenue > 0
  and shares else None), where `net_debt_fy = (LongTermDebt_fy or 0) +
  (LongTermDebtCurrent_fy or 0) - (Cash_fy or 0)`, treated as `0.0` (EV = market
  cap) when none of those three concepts is present for that fy. This is the
  sales multiple the hyper-grower growth-adjusted EV/Sales layer ranks against.
- `ev_ebit = ev_fy / ebit_fy` (`ebit_fy > 0` and shares else None), where
  `ev_fy = fy_price * shares_fy + net_debt_fy` (the SAME `net_debt_fy` as
  `ev_sales`, same `0.0` degradation) and `ebit_fy = OperatingIncome_fy`.
- `ev_ebitda = ev_fy / ebitda_fy` (`ebitda_fy > 0` and shares else None), where
  `ebitda_fy = OperatingIncome_fy + Depreciation_fy` (BOTH concepts must be
  present for the fy, else None — no zero-fill, unlike net debt). `Depreciation`
  is the same D&A concept the REIT FFO proxy uses (`DepreciationDepletionAnd
  Amortization` family). These two are EV/EBIT(DA) — enterprise-value earnings
  multiples that, unlike P/E, are capital-structure-neutral (numerator adds net
  debt, denominator is pre-interest), so they compare a leveraged filer against
  its own history and against peers without the leverage distortion P/E carries
  (VALUATION.md Sec.2/Sec.7). `ev_ebit` is informational (reported everywhere,
  never a triangulation signal). `ev_ebitda` is informational for a
  non-leveraged filer, but becomes the PRIMARY own-history multiples signal —
  ahead of P/E — when the filer is leveraged (`net_debt / EBITDA >=
  triangulate._LEVERAGE_EBITDA_RATIO`, `= 1.0`), for every sector except
  `financial`/`reit`/`growth_unprofitable` (which keep their own primary). See
  Sec.10 (`triangulate`) and VALUATION.md Sec.2/Sec.7 for the routing.

Returns `[{"fy", "end", "price", "pe", "ps", "pfcf", "ev_sales", "ev_ebit",
"ev_ebitda", "pffo"}, ...]` sorted by fy (`pffo` per Sec.8c).

`multiples.percentile_position(history_values: list[float], current: float) ->
Optional[float]` — percentage (0–100) of historical values strictly less than
`current`, plus half of ties (midrank). Requires ≥5 non-None historical values,
else `None`.

Current multiples come from `metrics["pe"|"ps"|"pfcf"|"ev_ebit"|"ev_ebitda"]`
(the current EV multiples use current market cap + `metrics["net_debt"]` over
latest-fundamental-FY EBIT/EBITDA — see `metrics.compute_metrics`).

### Growth-adjusted multiples (PEG layer, VALUATION.md Sec.7)

- `multiples.forward_revenue_cagr(revenue_series, fy, years=3) -> Optional[float]`
  — realized revenue CAGR over the `years` fiscal years *following* `fy`
  (`(rev_{fy+years}/rev_fy)**(1/years) - 1`; both endpoints present and > 0,
  else `None`).
- `multiples.growth_adjusted_value(multiple, growth_fraction, min_growth=0.05)
  -> Optional[float]` — `multiple / (growth_fraction * 100)` (growth in
  percentage points, so a 15% denominator is `15`). Returns `None` (never a
  negative/exploded figure) unless `multiple > 0` AND `growth_fraction >=
  min_growth` (5% floor guards the PEG linearity flaw).
- `multiples.growth_adjusted_history(history, revenue_series, multiple_key,
  min_growth=0.05) -> list[float]` — each history year's `multiple_key` value
  (`"pe"` for PEG, `"ev_sales"` for the hyper sales multiple) growth-adjusted by
  *its own* forward-3y revenue CAGR; only complete years contribute (the most
  recent ~3 fys drop out), the list is already `None`-free for
  `percentile_position`.

The engine assembles these into the `multiples.growth_adjusted` output block
(Sec.11): standard mode ranks PEG (current P/E ÷ base growth) against the raw
P/E percentile; hyper-grower mode ranks growth-adjusted EV/Sales (current
EV/Sales ÷ base growth) against the raw EV/Sales percentile. The denominator is
ALWAYS the assumptions base `growth_5y` (surfaced as `base_growth_pct`).

## 7. Damodaran — `damodaran.load_sector_data(dir_path) -> Optional[dict]`

Reads `data/damodaran/` (path from `Config.DAMODARAN_DIR`, default
`<cwd>/data/damodaran`). Expected files (documented in that folder's README):
- `multiples.csv` — columns: `industry, pe, ps, pfcf` (medians per industry),
  plus OPTIONAL `growth` (expected multi-year growth, decimal fraction e.g.
  `0.15`) and/or `peg` columns used only for the sector-median PEG comparison
  (VALUATION.md Sec.7); both default to `None` when absent, so older
  four-column CSVs keep working
- `erp.csv` — columns: `region, erp` (only the row `region == "US"` is used)

Loader is tolerant: missing dir/file/columns → return what's available and log
which pieces are missing; never raise. `sector_medians(sector_data,
sic_description)` matches the company's `sicDescription` to an `industry` row
by case-insensitive substring/keyword overlap; no match → `None`.

The matched `pe`/`ps`/`pfcf` medians are surfaced in the output's
`multiples.sector` block AND feed the triangulate multiples signal's
sector-relative axis-b (`sector_ratio`, Sec.11) — the current primary multiple
over its matching sector median. When the medians are absent (no match, or no
Damodaran data), `sector_ratio` is `None` and axis-b is silently skipped.

## 8. Sector classification — `sector.classify_sector(sic, normalized, metrics) -> str`

Deterministic from SIC (int or str), with financial-statement overrides:
- 6798 → `"reit"`
- 6500, 6510–6519 (real-estate operators/lessors) → `"reit"`: these carry the
  same GAAP real-estate-depreciation distortion as REITs, so they get the same
  FFO treatment (Sec.8c). Excludes 6531 (real-estate agents/managers) and 6552
  (land subdividers/developers), which stay `"financial"` -- asset-light/
  inventory businesses, not depreciable-property owners. Purely a SIC rule (no
  fundamentals condition): a non-REIT filer routed here self-corrects, since
  the FFO valuation falls back to P/B×ROE when no usable depreciation series
  exists.
- 6000–6999 (except the reit codes above) → `"financial"`
- SIC in cyclical set → `"cyclical"`: 1000–1499 (mining/energy), 2911,
  2800–2899 (chemicals), 3310–3399 (metals), 3559, 3711–3716 (autos),
  4400–4599 (shipping/air)
- 3674 (semiconductors) → no longer unconditionally cyclical: `"cyclical"`
  only when realized revenue CAGR (5y, falling back to 3y) is unknown or
  `<= 15%` (through-cycle/commodity/memory-type semi); otherwise falls
  through to the profitability check below like any other SIC, so a
  secular-growth semi classifies as `"mature"`/`"growth_unprofitable"` and
  can independently enter hyper-grower mode (see the gray-zone tier
  cross-referenced below)
- else if latest-FY `NetIncome < 0` → `"growth_unprofitable"`, UNLESS the firm
  is normally profitable (>= 2 prior fiscal years of `NetIncome` data, a
  profitable majority among them, AND the immediately prior year profitable),
  in which case the loss is treated as a one-off (writedown/litigation/tax
  charge) and the firm still classifies `"mature"` -- a single bad year
  shouldn't raise the discount floor or exclude the firm from the EPV path
- else → `"mature"`
If SIC missing → fall back to the LLM's phase-1 `sector_type` (engine wiring),
else `"mature"`.

("latest-FY" above, and the fiscal year `detect_hyper_grower` reads
`latest_revenue`/FCF-margin from below, both mean
`resolve_fundamental_fy(metrics)` — the "Existing inputs" note at the top of
this file.)

### Sector → method adjustments (engine)
- `financial`/`reit`: FCF-DCF disabled for both (`dcf.enabled = False`,
  Turkish `disabled_reason`; the specific wording differs per sector, see
  below). Hyper-grower detection is never attempted for either sector (see
  the cross-reference below).
  - `financial`: compute a P/B×ROE anchor using the justified (growth-aware)
    price-to-book multiple:
    `fair_pb = (roe - g) / (discount_rate_base - g)`, where
    `g` is the base scenario's `terminal_growth` (degrading to the no-growth
    `roe / discount_rate_base` form when `g` is missing, negative, or would
    make the denominator non-positive), `per_share = fair_pb * (equity_latest
    / shares)`; band from a `discount_rate_base ± 1pp` sensitivity re-clamp
    with `g` held fixed across the band (±10% fallback, Sec.4); bear/base/bull
    scale `fair_pb` by (0.8 / 1.0 / 1.2). Output under key `"pb_roe"`
    mirroring the dcf scenario shape.

    **No longer clamped to `[0.5, 4.0]` (normalization Work Package 5).**
    `fair_pb` used to be hard-clamped to that reference band
    (`_PB_CLAMP_LO`/`_PB_CLAMP_HI`); a high-ROE compounder can legitimately
    warrant a justified P/B above 4 (or a structurally low-ROE financial
    below 0.5), and clamping silently discarded that signal. `_build_pb_roe`
    now returns the RAW `fair_pb` and, when it falls outside `[0.5, 4.0]`,
    sets `justified_pb_flag` to `"above_reference"`/`"below_reference"` (else
    `None`) plus a Turkish note naming the value, the ROE, and the discount
    rate -- flagging the contradiction/information for the report layer
    instead of hiding it. `_PB_CLAMP_HI` (4.0) is still used, unchanged, as a
    SEPARATE advisory threshold elsewhere (`_build_earnings_power`'s
    over-capitalization advisory, Sec.8a) -- that use is untouched by this
    change. `pb_roe`'s output dict gains two additive keys: `"fair_pb"` (the
    raw, unclamped base justified P/B) and `"justified_pb_flag"`.

    **Non-positive `fair_pb` (or book value) makes the anchor unavailable.**
    When the raw base `fair_pb` comes out `<= 0` — which happens exactly when
    `roe <= g` (a loss-making, or sub-terminal-growth, filer; the denominator
    `r - g` is always positive here) — or when book value per share is `<= 0`
    (negative equity), `_build_pb_roe` returns `None` (anchor unavailable) with
    a Turkish note, rather than emitting a negative per-share fair value. A
    price-to-book multiple applied to book equity can never make a share worth
    zero or negative dollars, so there is no meaningful P/B×ROE fair value for
    such a filer (e.g. MSTR, classified `financial`, with a negative-ROE year:
    `fair_pb = (-0.09 - 0.04)/(0.10 - 0.04) = -2.12` → anchor unavailable
    instead of a −$337/share "fair value"). This is a guard against a
    degenerate/meaningless multiple, distinct from — and not in tension with —
    the Work-Package-5 rule above (which is about NOT clamping a legitimately
    high/low **positive** `fair_pb`). When this anchor is unavailable and no
    other anchor exists for the sector (e.g. `financial`, whose DCF is
    disabled), `fair_value_range` is all-`None` (the honest "no opinion"
    outcome) per Sec.11.
  - `reit`: compute an FFO-based Gordon-growth anchor instead (Sec.8c) --
    P/B×ROE systematically understates a REIT, since GAAP real-estate
    depreciation is a large non-cash charge that depresses both net income
    and book equity. Output under a NEW key, `"ffo"`, with the SAME
    `{"scenarios": {...}}` shape as `"pb_roe"`. When FFO can't be built at
    all (no fiscal year has both `NetIncome` and the new `Depreciation`
    concept, or the resulting FFO is `<= 0`), the engine falls back to the
    same P/B×ROE anchor `financial` uses (output under `"pb_roe"` instead,
    `"ffo"` stays `None`), with a Turkish note explaining the fallback.
    Wherever the engine/triangulation would otherwise consume the `pb_roe`
    block as the headline/triangulation-DCF-equivalent signal for this
    sector (Sec.11's `fair_value_range`, the triangulate `dcf_base_band`),
    it consumes `ffo` instead (or `pb_roe`, when FFO fell back) -- see
    Sec.8c for the full mechanics.
- `growth_unprofitable`: DCF still attempted (fcf may be negative → note),
  multiples use P/S only (pe/pfcf percentiles likely None), triangulation
  weights reverse-DCF + P/S. Additionally, when the filer grows the top line
  at a real but sub-hyper rate (realized CAGR ≥ 12%) and is not picked up by
  `detect_hyper_grower`, a mid-growth revenue-first DCF becomes the headline
  instead of a multiples-only one (Sec.8d).
- `cyclical`: additionally compute a **normalized-earnings DCF variant**:
  `normalized_fcf0 = mean(top ceil(N/2) fcf_margin values over available FYs)
  * latest revenue`, where each year's margin is `(ocf - capex - sbc) /
  revenue` (SBC treated as `0.0` when missing — same SBC-as-expense
  treatment as the standard fcf0, Sec.4) — the mean of the upper-half
  (mid-to-upper-cycle) FCF margins rather than the median, since the median
  degenerated to the trough year for deep cyclicals; a non-positive
  normalized margin yields `None` plus a Turkish note instead of a variant.
  Run the same 3 scenarios; report under `dcf.normalized_variant` with the
  same scenario shape. Both variants are reported side by side. The reported
  `sensitivity` matrix (Sec.9) is taken from `normalized_variant`'s base
  whenever it was successfully computed, UNCONDITIONALLY of which headline
  below actually wins (the engine's `headline_fcf0` selector only checks
  `sector_type == "cyclical" and normalized_variant is not None`, not the
  gate described next) — otherwise it falls back to the raw `fcf0`.

  The headline `fair_value_range` and the triangulation DCF band, however,
  are **not unconditionally `normalized_variant`'s**: when the SAME FCF-DCF
  reliability gate the mature path uses (`_fcf_dcf_unreliable`, Sec.8a) ALSO
  fires for this cyclical — i.e. the raw FCF-DCF isn't merely near-trough but
  structurally capex-suppressed every year, cash-backed, and
  investment-driven (the canonical capital-intensive-cyclical case, e.g.
  Micron/MU) — the headline instead comes from **Sec.8e**'s sustainable-growth
  FCFE anchor, or, when that anchor can't clear the zero-growth EPV floor,
  the EPV floor itself (Sec.8a); `normalized_variant` is then demoted to a
  secondary cross-check reported alongside the raw FCF-DCF, not the headline.
  When the gate does NOT fire (the ordinary near-trough cyclical, not
  growth-CapEx suppressed), this section's original behavior is unchanged:
  the headline and triangulation DCF band both come from `normalized_variant`
  whenever it was computed, else from the raw FCF-DCF band/fcf0. See Sec.8e
  for the full gate/guardrail mechanics.

Cross-reference (Sec.11/Sec.3): independently of `sector_type`, when
`sector.detect_hyper_grower` triggers (and the engine can build the
scenarios), the revenue-first DCF's own base band takes over as the
headline `fair_value_range`/triangulation-DCF source, ahead of both the
cyclical `normalized_variant` and the raw `dcf.scenarios` band — see
`hyper_growth`/`hyper_growth_detail` in Sec.11. Hyper-grower detection
itself is gated off entirely for `sector_type in ("financial", "reit")` — a
revenue-margin hyper-DCF doesn't make sense for those sectors, which use
P/B×ROE (`financial`) or the FFO Gordon-growth anchor (`reit`, Sec.8c)
instead.

`sector.detect_hyper_grower(metrics, ratios, normalized)`'s trigger
condition, keyed off the realized revenue CAGR (5y, falling back to 3y),
has two tiers:
- **Strong tier**: CAGR strictly above 25% AND at least one of (a) FCF ≤ 0,
  (b) FCF margin < 5%, (c) (R&D + SBC)/revenue > 40%.
- **Gray zone**: CAGR in `(0.20, 0.25]` (strictly above 20%, up to and
  including 25%) AND at least one of clauses (a)/(b)/(c) above AND current
  P/S strictly above 8.0 — a fired clause alone isn't enough in the gray
  zone; the market also has to already be pricing in high growth. This is
  what lets a filer like a fast-growing semiconductor (22–24% realized
  CAGR, negative or thin FCF from R&D/SBC intensity, but a rich P/S) enter
  hyper-grower mode instead of being valued by a trailing-FCF DCF that
  systematically undervalues it — see the semiconductor bullet above.
- CAGR at or below 20% never triggers, regardless of clauses or P/S.

Both tiers apply uniformly (independently of `sector_type`, subject to the
financial/reit gating above) — the gray zone is not semiconductor-specific,
it's just the tier most likely to matter for SIC 3674 given the 15%
secular-growth threshold used by `classify_sector` above.

## 8a. Earnings-power-value (EPV) anchor + FCF-DCF reliability gate — `engine._build_earnings_power` / `engine._fcf_dcf_unreliable`

Mirrors `_build_pb_roe` (Sec.8) in structure and return shape. Addresses
mature, genuinely profitable filers whose FCF-DCF headline collapses to a
near-worthless band because free cash flow is suppressed by heavy growth
CapEx and/or stock-based compensation (SBC) even though the business is
cash-flow-backed profitable — canonical case: Amazon. Everywhere else
(`cyclical`/`financial`/`reit`/`growth_unprofitable`, or any filer already in
hyper-grower mode) this section does not apply and behavior is unchanged.

### `_build_earnings_power(assumptions, normalized, metrics, ratios) -> tuple[Optional[dict], list[str]]`

- Only called by the engine when `sector_type == "mature"` AND hyper-grower
  mode is NOT active (`not hyper_growth_active`, built after
  `hyper_growth_active` is resolved — Sec.11's hyper-grower block runs first).
- `shares = metrics.get("shares")` (current share count — same `latest_fy`
  convention as `_build_pb_roe`, Sec.8; EPV never anchors share count to
  `latest_fundamental_fy`). Missing/`<= 0` → `(None, ["Kazanç-gücü çapası
  hesaplanamadı: geçerli hisse sayısı yok."])`.
- `dr_base = assumptions["base"]["discount_rate"]`, used directly as the cost
  of equity. Missing/non-numeric/`<= 0` → `(None, ["Kazanç-gücü çapası
  hesaplanamadı: geçerli iskonto oranı (cost of equity) yok."])`.
- `fy = resolve_fundamental_fy(metrics)` — never the raw `latest_fy` (Sec.4's
  ghost-year problem). Reads `NetIncome`/`Revenue` at `fy` via
  `to_annual_series`. `latest_ni is None or <= 0` → `(None, ["Kazanç-gücü
  çapası hesaplanamadı: son yılın net kârı negatif veya eksik."])` — EPV never
  applies to a filer that isn't profitable in its latest fundamental year.
- **Mandatory margin-median sanity guard**, `_EPV_SANITY_DEVIATION = 0.5`:
  protects against a one-off non-operating swing in net income (e.g. a
  mark-to-market gain/loss, a tax one-off) distorting the anchor, the same
  way `_select_fcf0`'s own ±50% deviation check protects `fcf0` (Sec.4).
  `margins` = `{NetIncome_y / Revenue_y}` over every fiscal year where both
  are strictly positive; `ref_ni = median(margins) * latest_rev`. If
  `margins` is empty or `latest_rev` is unusable, sanity can't be evaluated:
  `normalized_ni = latest_ni`, `sanity_applied = False`. Otherwise, if
  `ref_ni > 0` and `abs(latest_ni/ref_ni - 1.0) > 0.5`: `normalized_ni =
  ref_ni`, `sanity_applied = True`, plus a Turkish note naming both figures
  ("Kazanç-gücü tabanı için son yılın net kârı (...) geçmiş marj medyanından
  belirgin saptı; ... marj-medyanı bazlı normalize kazanç (...) kullanıldı.");
  else `normalized_ni = latest_ni`, `sanity_applied = False`.
- **Value**: `base_value_per_share = normalized_ni / dr_base / shares` — a
  zero-growth, no-net-debt-bridge equity anchor (Bruce Greenwald earnings
  power). **No growth term** — deliberate, consistent with Sec.3's Gordon-
  growth invariant: EPV is a conservative floor, not a growth valuation.
  **No net-debt bridge** — deliberate, consistent with Sec.4's FCFE-direct
  convention: `NetIncome` is already a levered/equity figure (interest to
  debtholders already deducted), so subtracting net debt again would
  double-penalize leverage.
- **Scenarios**: reuses the existing `_PB_SCENARIO_SCALE` constant from
  Sec.8 as-is (bear 0.8 / base 1.0 / bull 1.2 — no separate EPV scale
  constant) — `per_share = round(base_value_per_share * scale, 2)`. Each
  scenario's `lo`/`hi` band comes from a new helper, `_epv_scenario_band
  (normalized_ni, dr_base, scale, shares, per_share)`, mirroring
  `_pb_roe_scenario_band` (Sec.8): recompute `normalized_ni / dr / shares *
  scale` at `dr_base` and `dr_base ± sensitivity._DISCOUNT_RATE_STEP`
  (excluding any `dr <= 0`), take the min/max; falls back to the flat
  `_band(per_share)` (±10%) when fewer than `_MIN_GRID_CELLS_FOR_BAND` points
  are usable, with the same fallback Turkish note pattern as
  `_pb_roe_scenario_band`/`_dcf_scenario_band`.
- **Over-capitalization advisory** (advisory only — never alters the computed
  value): if `StockholdersEquity` at `fy` is known and positive, and the
  implied `roe = normalized_ni / equity` divided by `dr_base` exceeds
  `_PB_CLAMP_HI` (the same ceiling `_build_pb_roe`'s `fair_pb` clamp uses,
  Sec.8), append a Turkish advisory note that reading the value as a floor
  may be misleading if that return isn't sustainable. Unlike `_build_pb_roe`'s
  `fair_pb`, the EPV value itself is NEVER clamped — EPV doesn't touch book
  equity to begin with, so there's no multiple to clamp.
- Returns `({"scenarios": {"bear"/"base"/"bull": {"per_share","lo","hi"}},
  "per_share": <base per_share>, "normalized_net_income": normalized_ni,
  "cost_of_equity": dr_base, "sanity_applied": bool}, notes)`, or `(None,
  notes)` if any precondition above failed. Never raises.

### `_fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics) -> tuple[bool, Optional[str]]`

The gate deciding whether the FCF-DCF headline should be REPLACED by the EPV
headline. A suppressed-looking FCF band is deliberately NOT sufficient on its
own to flip the switch: FCF can also be low because net income itself is
low-quality (not actually converting into cash), in which case an
NetIncome-based EPV anchor would be a WORSE headline than the (correctly)
suppressed FCF-DCF, not a better one. This is the reliability gate's central
purpose — it must guard against masking a genuine earnings-quality problem
behind a reassuringly "healthy-looking" EPV number. ALL three conditions must
hold to fire:
- `fcf_suppressed`: `dcf_scenarios` is `None`, or `base.hi` is `None`, or
  `base.hi < _EPV_GATE_FCF_RATIO * epv_base` (constant `= 0.5`), where
  `epv_base = earnings_power["scenarios"]["base"]["per_share"]`.
- `cash_backed`: at `fy = resolve_fundamental_fy(metrics)`, `OperatingCashFlow`
  and `NetIncome` are both known, `NetIncome > 0`, and `OperatingCashFlow >=
  _EPV_GATE_CASH_BACKED_RATIO * NetIncome` (constant `= 0.8`) — net income
  must actually be converting into cash for EPV to be a trustworthy
  numerator.
- `investment_driven`: `OperatingCashFlow > 0`, `CapEx` known, and
  `CapEx / OperatingCashFlow >= _EPV_GATE_CAPEX_OCF_RATIO` (constant `= 0.5`)
  — the suppression must plausibly be attributable to heavy growth
  investment (the Amazon story), not some other drag on cash flow.

If `earnings_power` is `None` (couldn't be built, or has no `base` per-share
value), returns `(False, None)` immediately — nothing to gate. If all three
conditions hold, returns `(True, None)`: switch to EPV. If `fcf_suppressed`
but NOT `cash_backed`, the gate refuses to fire (`(False, ...)`) but still
returns a Turkish earnings-quality warning note — this is the cash-conversion
guard's payoff: rather than silently doing nothing, it explicitly surfaces
that low FCF here is a quality red flag, not (yet) evidence for an EPV switch:

```
"Serbest nakit akışı düşük ve işletme nakit akışı net kârı yeterince
desteklemiyor (OCF < 0.8×net kâr); bu bir kazanç-kalitesi/nakde-çevirme
uyarısıdır — manşet değerleme FCF-DCF'te bırakıldı, kazanç-gücü çapasına
geçilmedi."
```

Otherwise (FCF isn't suppressed at all) returns `(False, None)`.

New constants (`engine.py`): `_EPV_SANITY_DEVIATION = 0.5`,
`_EPV_GATE_FCF_RATIO = 0.5`, `_EPV_GATE_CASH_BACKED_RATIO = 0.8`,
`_EPV_GATE_CAPEX_OCF_RATIO = 0.5`.

### Engine integration (`run_valuation`)

Built right after `hyper_growth_active` is resolved, before the primary-DCF
priority chain. **Updated by Sec.8e:** `earnings_power` is now built for
`cyclical` filers too, not just `mature` — for `cyclical` it doubles as both
the zero-growth floor and the earnings base Sec.8e's sustainable-growth FCFE
anchor grows:

```python
earnings_power = None
if sector_type in ("mature", "cyclical") and not hyper_growth_active:
    earnings_power, ep_notes = _build_earnings_power(assumptions, normalized, metrics, ratios)
```

The priority chain (Sec.3/Sec.8's existing hyper-grower branch, then the
cyclical `normalized_variant` branch) gains a new trailing `elif` branch, so
both hyper-grower mode and the cyclical normalized-earnings variant still
take precedence over EPV **for `mature` filers** (the `cyclical` branch has
its own, separate priority chain — see Sec.8e):

```python
epv_headline = False
# ... existing hyper_growth_active branch ...
# ... existing "sector_type == 'cyclical' and normalized_variant is not None" branch ...
elif sector_type == "mature" and earnings_power is not None:
    unreliable, quality_note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics)
    if quality_note:
        notes.append(quality_note)
    if unreliable:
        primary_dcf_scenarios = earnings_power["scenarios"]
        epv_headline = True
        notes.extend(ep_notes)  # margin-median/over-cap/band-fallback notes -- see below
        notes.append(<Turkish EPV-headline explanation note, quoted below>)
```

`_build_earnings_power`'s own notes (`ep_notes` — margin-median sanity,
over-capitalization advisory, band-fallback) are held back and only appended
to `notes` when `epv_headline` is actually `True`; for a mature filer where
EPV was built but the gate never fired (FCF-DCF stayed the headline), they
would be confusing noise about a value the reader isn't being shown.

When the switch fires, the following Turkish note explains it:

```
"Bu şirkette serbest nakit akışı büyük büyüme yatırımı (yüksek CapEx) nedeniyle
kazanç gücünü yansıtmıyor; manşet makul değer aralığı sıfır-büyüme kazanç-gücü
(EPV) çapasına dayandırıldı. Ham FCF-DCF senaryoları ikincil olarak
'dcf.scenarios' altında raporlanıyor. NOT: EPV, büyüme primini KASITLI
dışlayan muhafazakâr bir tabandır; fiyatın ima ettiği büyümeyi ters-DCF
ölçer."
```

`fair_value_range` (Sec.4/Sec.11): when `epv_headline` is `True`, the
per-scenario `growth`/`discount_rate`/`note` metadata comes from a new
`_epv_scenario_meta(earnings_power)` helper (mirrors `_hyper_scenario_meta`,
Sec.11) instead of the standard clamped-assumptions strings — each scenario's
`growth` reads as zero-growth ("sıfır büyüme (kazanç gücü çapası)"),
`discount_rate` is the formatted cost of equity, and `note` carries the
scenario's story.

### Exception to Sec.9's "same cash-flow base" invariant (documented, intentional)

Sec.9 states the reported `sensitivity` grid never silently describes a
different cash-flow base than the headline `fair_value_range`. **EPV is a
deliberate, explicitly-noted exception to that rule**: EPV has no growth
axis at all, so re-deriving a sensitivity grid or reverse-DCF around it would
either be meaningless or would have to invent a growth dimension EPV
purposely excludes. Instead, whenever `epv_headline` is `True`:

- The `sensitivity` grid (Sec.9) and `reverse_dcf.implied_growth` (Sec.5)
  BOTH continue to describe the secondary, suppressed FCF-DCF base
  (`dcf_scenarios`/`fcf0`) — kept on purpose, as evidence of *why* free cash
  flow looks suppressed relative to earnings power, not as a description of
  the EPV headline itself.
- A Turkish note is appended making the exception explicit to the reader, so
  the divergence from Sec.9's normal invariant is never silent:

```
"Duyarlılık tablosu ve ters-DCF, manşet EPV çapasını değil, ikincil
(baskılanmış) FCF-DCF tabanını yansıtır; serbest nakit akışının neden düşük
olduğunu gösteren kanıt olarak korunmuştur."
```

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

`run_valuation` passes `earnings_power_headline=epv_headline` into
`triangulate.triangulate(...)`. See Sec.10 for the resulting `CONFIDENCE_HIGH`
→ `CONFIDENCE_MEDIUM` cap and its rationale.

### Output shape additions (Sec.11)

`run_valuation`'s returned dict gains two additive keys — see Sec.11 for the
full return shape:
```python
"earnings_power": {"scenarios": {...}, "per_share": float,
                    "normalized_net_income": float, "cost_of_equity": float,
                    "sanity_applied": bool} | None,
"earnings_power_headline": bool,
```
`earnings_power` is populated whenever `sector_type == "mature"` and
hyper-grower mode is off, REGARDLESS of whether it ends up as the headline
(so a caller can always inspect the EPV anchor even when FCF-DCF stayed
primary); `earnings_power_headline` is `True` only when `_fcf_dcf_unreliable`
actually gated the switch.

### Scope

Purely additive: does not change `dcf.scenarios`, `pb_roe`, `sensitivity`, or
any other existing output key's meaning, and does not apply to
`cyclical`/`financial`/`reit`/`growth_unprofitable` filers or to any filer
already in hyper-grower mode. A mature, healthy-FCF filer (e.g. AAPL) still
gets `earnings_power` populated but `earnings_power_headline == False`, and
`fair_value_range`/`triangulation`/confidence are unchanged from before this
section existed.

## 8b. Mature revenue-first DCF (growth-inclusive alternative to EPV) — `engine._build_mature_revenue_dcf`

A second, growth-inclusive alternative to the zero-growth EPV anchor (Sec.8a)
for mature filers whose FCF-DCF is unreliable (same
`_fcf_dcf_unreliable` gate) but that, unlike a truly mature no-longer-growing
filer, still have genuine, realized top-line growth left to fade — the
canonical case is Amazon: FCF is suppressed by growth CapEx/SBC, but revenue
is still compounding at a real double-digit rate. Reuses the hyper-grower
mode's own machinery (`revenue_dcf.revenue_first_dcf`, `_hyper_scenario_band`)
with a shorter fade and a much lower margin ceiling — this is a mature,
already-large filer's steady state, not a still-searching hyper-grower's.

### Why this doesn't double-count growth investment (unlike a rejected owner-earnings/CapEx-add-back variant)

`FCF_t = revenue_t × margin_t` for every projected year — nothing is ever
added back to FCF. Revenue fades from the realized growth rate toward
terminal growth (same fade discipline as Sec.4/§4a's "growth isn't free"
principle); the FCF margin is projected independently, starting from today's
(suppressed) margin and converging to a data-derived mature target by year 7.
Early years combine high growth with low (today's) margin; later years
combine faded growth with the mature margin — the reinvestment drag a
growing filer keeps paying is implicit in that margin fade, not modeled as a
separate CapEx line to subtract or add back. This is deliberately different
from an owner-earnings-style variant that grows FCF directly and then adds
back a CapEx estimate — that approach was considered and rejected because it
risks double-counting (or arbitrarily mismatching) the reinvestment the
margin-fade approach already prices in structurally.

### New constants (`engine.py`)

```python
_MATURE_REV_DCF_MIN_GROWTH   = 0.10   # realized revenue CAGR floor to even attempt this method
_MATURE_TAX_ASSUMPTION       = 0.25   # flat tax-rate proxy for the NOPAT margin anchor
_MATURE_REINVEST_HAIRCUT     = 0.85   # reinvestment-drag haircut applied to the NOPAT anchor
_MATURE_HIST_UPLIFT          = 1.5    # multiplier on the single best historical raw FCF margin
_MATURE_TARGET_CAP           = 0.15   # absolute ceiling on the mature target FCF margin
_MATURE_STEADY_STATE_YEAR    = 7      # full convergence year (<= revenue_dcf.HORIZON_YEARS = 10)
_MATURE_TARGET_MARGIN_SCALE  = {"bear": 0.7, "base": 1.0, "bull": 1.2}
```

### Helper 1 — `_mature_current_margin(normalized, metrics) -> float`

The fade's *starting point*: the median of the last 3 fiscal years'
SBC-adjusted FCF margin (`(OCF - CapEx - SBC) / Revenue`, SBC `0.0` when
missing), anchored at `resolve_fundamental_fy(metrics)` — never a single
year, so one working-capital swing doesn't set where the whole projection
starts from. Returns `0.0` (never `None`) when no fiscal year has usable
data.

### Helper 2 — `_mature_target_fcf_margin(normalized, metrics, ratios) -> Optional[float]`

The fade's *mature target*, the smaller of two independent, data-derived
anchors, further floored at today's margin:
- **op-anchor (NOPAT proxy):** median of every fiscal year's positive
  `OperatingIncome / Revenue`, converted with
  `op_margin * (1 - _MATURE_TAX_ASSUMPTION) * _MATURE_REINVEST_HAIRCUT`.
  `None` if no fiscal year has a positive operating margin.
- **hist-anchor:** `_MATURE_HIST_UPLIFT ×` the single best historical raw
  FCF margin (`(OCF - CapEx) / Revenue`, positive years only). `None` if no
  fiscal year has a positive raw FCF margin.
- `target = min(nopat, hist_anchor)` over whichever of `nopat`/`hist_anchor`
  are available; `None` only when **both** are unavailable (the method can't
  be built without at least one anchor). **No longer additionally clamped to
  `_MATURE_TARGET_CAP` here (normalization Work Package 4)** -- that
  constant (0.15) is now a reporting-only flag threshold the caller
  (`_build_mature_revenue_dcf`) compares this function's return value
  against, rather than the value itself being silently truncated to it (see
  "Main function" below).
- Finally floored at `_mature_current_margin(...)` whenever that figure is
  positive — a filer already earning more than the computed mature ceiling
  today must never be modeled as if its margin falls.

### Helper 3 — `_mature_start_growth(metrics, normalized) -> Optional[float]`

Mirrors the hyper-grower F4 blend pattern: realized CAGR (`revenue_cagr_5y`,
falling back to `revenue_cagr_3y`) blended 50/50 with the latest single
fiscal year's revenue YoY when both are computable
(`0.5 * realized + 0.5 * latest_yoy`), else the realized CAGR alone. `None`
if no realized CAGR is available at all (the method can't be built without
some realized-growth reference). Unlike hyper-grower mode, this single
`start_growth` figure is **not** scaled per scenario — it's the same
realized number in bear/base/bull; only the discount rate and the target
margin (via `_MATURE_TARGET_MARGIN_SCALE`) vary by scenario.

### Main function — `_build_mature_revenue_dcf(assumptions, normalized, metrics, ratios, price, shares) -> (Optional[dict], list[str])`

Never raises (try/except wraps the whole body, mirroring
`_build_hyper_growth`). Steps:
1. Resolve `revenue0` at `resolve_fundamental_fy(metrics)` and `shares`;
   missing/non-positive either → `(None, note)`.
2. `start_growth = _mature_start_growth(...)`; `None` → `(None, note)`.
3. **Growth gate:** `start_growth < _MATURE_REV_DCF_MIN_GROWTH` OR
   `start_growth <= assumptions["base"]["terminal_growth"]` (nothing left to
   fade) → `(None, note)` — this is what limits the method to filers with a
   real, still-fading growth story; a slow/stagnant "mature" filer falls
   through to EPV instead.
4. `target_base = _mature_target_fcf_margin(...)`; `None` → `(None, note)`.
   **WP4:** if `target_base > _MATURE_TARGET_CAP` (0.15), append a Turkish
   note naming the value and set `target_margin_flag = "above_reference"`
   (else `None`) -- reporting only, `target_base` itself is used unclamped.
5. `current_margin = _mature_current_margin(...)`; `steady_state_year =
   _MATURE_STEADY_STATE_YEAR` (7, shorter than hyper-grower's 10 — a mature
   filer's growth story is closer to already playing out).
6. Per scenario (bear/base/bull): `dr`/`terminal_growth` come from the
   **clamped assumptions pipeline** (not hard-coded hyper-style rates);
   `target_margin = target_base * _MATURE_TARGET_MARGIN_SCALE[scenario]`;
   `start_growth` itself is identical across all three. Skips (with a note)
   any scenario with a missing/non-numeric `dr`/`terminal_growth` or
   `dr <= terminal_growth`. Calls
   `revenue_dcf.revenue_first_dcf(revenue0, start_growth, terminal_growth,
   dr, current_margin, target_margin, steady_state_year, shares,
   annual_dilution=0.0)`, then `_hyper_scenario_band(...)` for that
   scenario's `lo`/`hi` (same fallback-to-±10% behavior as the hyper path
   when fewer than 2 sensitivity-grid cells are usable).
7. No scenario built → `(None, note)`. Otherwise returns
   `({"scenarios": {...}, "start_growth", "target_margin_base",
   "target_margin_flag", "current_margin", "steady_state_year"}, notes)`.
   The caller (`_run_valuation`) additionally mutates this dict with a
   `growth_vs_floor` key (`"adds"`/`"destroys"`/`None`) after this function
   returns -- see "EPV-floor guardrail" below and normalization Work
   Package 7.

### `run_valuation` integration (priority chain)

Attempted only where EPV is also attempted — `sector_type == "mature"` and
`_fcf_dcf_unreliable(...)` fired (Sec.8a). The priority chain becomes:

```
hyper-grower > cyclical normalized_variant > (mature-gate fired):
    mature revenue-first DCF builds AND clears its growth gate
    AND its base per-share >= EPV's base per-share (guardrail)
        -> mature_revenue_headline = True, revenue-first band leads
    else
        -> epv_headline = True, EPV floor leads (existing Sec.8a behavior)
> raw FCF-DCF (unchanged fallback)
```

**EPV-floor guardrail (`mr_beats_floor`):** a growth-inclusive revenue-first
value that lands *below* the zero-growth EPV floor is not a credible growth
case — it means the defensible mature FCF margin is thinner than the
earnings the EPV floor already capitalizes, so publishing it as the headline
would present a "growth" number weaker than the conservative no-growth
floor. When the revenue-first base per-share is below the EPV base
per-share, EPV stays the headline and the revenue-first band is demoted to a
secondary cross-check under `mature_revenue_detail` (still returned, just
not headlined) — a Turkish note names both figures and explains why EPV was
kept. **Normalization Work Package 7** made this dual-figure disclosure a
report, not merely a gate: `mature_revenue_detail["growth_vs_floor"]` is set
(via `engine._growth_vs_floor(epv_base_ps, mr_base_ps)`, a caller-side
classification applied after `_build_mature_revenue_dcf` returns) to
`"destroys"` when the revenue-first base is below the EPV floor (the growth
case is destroying value relative to the no-growth anchor) or `"adds"` when
it meets/clears the floor, whenever both figures are numeric (`None`
otherwise) — regardless of which one ends up headlined, so a caller/report
layer can always show both numbers and the relationship between them, not
just whichever one won. Why the guardrail compares against EPV specifically: EPV is a
net-income-based, zero-growth floor; the revenue-first model uses a
strictly thinner FCF margin (net of the tax/reinvestment haircut) — if
growth alone can't lift the growth-inclusive value above the no-growth
floor, the growth case isn't adding real value yet. **Empirical note:** in
every example tested against the current calibration (including AMZN and
ORCL), the revenue-first value stayed *below* the EPV floor, so in practice
this guardrail currently keeps the method in its secondary
cross-check role rather than ever heading the report — this may change as
more filers are tested or the calibration is refined.

When `mature_revenue_headline` fires, this Turkish note explains the switch:

```
"Serbest nakit akışı büyüme yatırımıyla bastırıldığı için manşet, geliri
fade eden ve FCF marjını olgun bir hedefe (%<X>) yakınsayan büyüme-dahil bir
revenue-first DCF'e dayandırıldı. Sıfır-büyüme EPV tabanı ($<Y>) ve ham
FCF-DCF ikincil olarak raporlanır."
```

`_build_mature_revenue_dcf`'s own notes (`mr_notes`) are appended to `notes`
only when `mature_revenue_headline` is actually `True` — the same
"don't surface notes about a value the reader isn't being shown" discipline
`_build_earnings_power`'s `ep_notes` already follow (Sec.8a). When the
guardrail instead keeps EPV as the headline but a revenue-first value was
successfully computed, a distinct note names both per-share figures and
explains the revenue-first band is reported as a secondary cross-check
under `mature_revenue_detail`.

`fair_value_range`'s `scenario_meta` (Sec.11 `_build_fair_value_range`):
when `mature_revenue_headline` is true, a new `_mature_scenario_meta
(mature_revenue_detail)` helper (mirroring `_hyper_scenario_meta`) supplies
per-scenario `growth` (`"gerçekleşen büyüme %<X>, olgun hedef marj %<Y>"`),
`discount_rate` (formatted cost of capital), and `note` (naming the
scenario, the realized growth, the fade horizon, the target margin, and the
discount rate) — any scenario missing its cell (a failed
`revenue_first_dcf` call) falls back to the standard assumptions-derived
string for that scenario/field.

### §9's "same cash-flow base" invariant — same documented exception as EPV

When `mature_revenue_headline` is `True`, the reported `sensitivity` grid
(Sec.9) and `reverse_dcf.implied_growth` (Sec.5) **both keep reflecting the
secondary, suppressed FCF-DCF base**, not the mature revenue-first headline
— for the same reason as Sec.8a's EPV exception (there's no standard
`growth_5y ± 2pp` grid to build around a revenue-first fade path), with an
analogous Turkish note appended. See Sec.9's own "Exception (Sec.8b)"
entry.

### Reverse-DCF override (§5) — same-base invariant

When `mature_revenue_headline` is `True`, the reverse-DCF pair shown in the
output switches to a revenue-based one, mirroring the hyper-grower override
exactly: `output_implied = revenue_dcf.implied_start_growth(price, revenue0,
base_terminal_growth, base_discount_rate, mature_revenue_detail["current_
margin"], mature_revenue_detail["target_margin_base"], mature_revenue_
detail["steady_state_year"], shares, 0.0)` (revenue reference, not FCF);
`output_realized_cagr` = `revenue_cagr_5y`/`_3y`; `output_realized_label` =
`"gelir 5y"`/`"gelir 3y"`; `output_bracket_status` defaults to `"ok"`
(`implied_start_growth` doesn't expose a bracket-boundary status the way
`implied_growth_with_status` does). This keeps the reverse-DCF's reference
growth apples-to-apples with what the headline model itself solves over —
the same rationale as Sec.5's hyper-grower override and Sec.8a's EPV
exception, just pointed at this method's own revenue/margin path instead.

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

`run_valuation` passes `mature_revenue_headline=mature_revenue_headline`
into `triangulate.triangulate(...)`. See Sec.10's "Mature revenue-first DCF
confidence ceiling" entry for the resulting `CONFIDENCE_HIGH` →
`CONFIDENCE_MEDIUM` cap and its rationale.

### Output shape additions (Sec.11)

`run_valuation`'s returned dict gains two additive keys — see Sec.11 for the
full return shape: `"mature_revenue_detail"` (the dict above, or `None`) and
`"mature_revenue_headline"` (bool). `mature_revenue_detail` is *attempted*
whenever the same gate that attempts EPV fires (`sector_type == "mature"`
and `_fcf_dcf_unreliable` is `True`), regardless of whether it ends up
non-`None` or headlined; `mature_revenue_headline` is `True` only when it
was built, cleared its own growth gate, AND beat the EPV guardrail.

### Scope

Purely additive: does not change `dcf.scenarios`, `pb_roe`, `earnings_
power`, `sensitivity`, or any other existing output key's meaning, and does
not apply outside the same `sector_type == "mature"` + `_fcf_dcf_unreliable`
gate that EPV (Sec.8a) already uses. A mature filer for which the gate never
fires, or fires but the growth gate rejects the revenue-first attempt, is
unaffected — `earnings_power`/EPV-headline behavior from Sec.8a is
unchanged from before this section existed.

## 8c. FFO-based REIT valuation (Gordon growth) — `engine._build_ffo` / `engine._select_latest_ffo`

Replaces the P/B×ROE anchor for `sector_type == "reit"` (Package 2): a P/B×ROE
(or P/E) anchor systematically understates a REIT's fair value, because GAAP
real-estate depreciation is a large non-cash charge that depresses both net
income and book equity. `financial` is UNCHANGED (still P/B×ROE, Sec.8).

**FFO (funds from operations) selection — `_select_latest_ffo`:** mirrors
`_build_pb_roe`'s FY-selection logic (Sec.8) exactly: walks the `NetIncome`
annual series newest → oldest and picks the first fiscal year that ALSO has a
`Depreciation` figure for that same year (does not require alignment with
`metrics`'s own notion of the latest fiscal year), then (Package 2/P2a):
```
gain    = GainOnSaleRealEstate_fy or 0.0   # signed: +gain increases NI
impair  = RealEstateImpairment_fy or 0.0   # positive expense that reduced NI
FFO_fy  = NetIncome_fy + Depreciation_fy - gain + impair
ffo_per_share = FFO_fy / shares   # shares = SharesOutstanding_fy, falling back to
                                  # metrics["shares"] (current count) if that FY's
                                  # share count is missing
```
(Deliberately consistent with the `pffo` column's per-FY share basis described
below in "P/FFO multiples signal" — both divide a fiscal year's FFO by that
SAME fiscal year's own share count, not today's.)
`gain`/`impair` are read for the SAME selected fiscal year only and never
affect FY selection (still only `NetIncome` + `Depreciation`); both default
to `0.0` when untagged for that year, so a filer/fixture that never reports
them computes byte-for-byte the same `FFO_fy` as before this change
(backward compatible). Sign handling: a us-gaap `GainLoss` element is
positive for a realized gain (which already inflated GAAP net income) and
negative for a loss, so `- gain` removes a gain and, for a negative value (a
loss), adds it back — both match Nareit. Impairments are positive expense
amounts that already reduced net income, so `+ impair` adds them back.

**This is a PROXY, still not true Nareit FFO — P2a narrows one of three
gaps:** Nareit's standardized FFO adds back only real-estate depreciation and
removes gains/losses on property sales and impairments. This engine now
handles the gains/impairments piece via two new best-effort,
real-estate-specific `FLOW_CONCEPTS` entries in `normalize/concepts.py`:
```
GainOnSaleRealEstate: GainLossOnSaleOfProperties /
    GainsLossesOnSalesOfInvestmentRealEstate /
    GainLossOnDispositionOfRealEstateInvestments
RealEstateImpairment: ImpairmentOfRealEstate / RealEstateImpairment
```
Deliberately NOT broad tags like `AssetImpairmentCharges` or generic
`GainLossOnDispositionOfAssets`, which would over-adjust for non-real-estate
items Nareit does not touch. Coverage is necessarily partial (a filer using a
tag not in either list silently contributes `0.0` for that adjustment); this
is acceptable since the adjustment defaults to 0 rather than raising or
fabricating a value. Two known gaps remain:
* Total D&A (the cash-flow-statement depreciation/depletion/amortization
  add-back, via the `Depreciation` concept in `normalize/concepts.py`, a
  `FLOW_CONCEPTS` entry falling back across
  `DepreciationDepletionAndAmortization` /
  `DepreciationAmortizationAndAccretionNet` / `DepreciationAndAmortization` /
  `Depreciation`) is added back wholesale instead of real-estate-only
  depreciation, since the latter isn't separable from this engine's
  normalized data. This slightly OVERSTATES FFO for a filer with meaningful
  non-real-estate amortization (e.g. intangibles from an acquisition); for a
  pure-play REIT (whose D&A is overwhelmingly building/property
  depreciation) it is a close approximation.
* Partial tag coverage for the gain/impairment concepts above, as noted.

If no fiscal year has both `NetIncome`/`Depreciation`, or the resulting FFO
is `<= 0`, FFO is considered unusable (the walk does NOT continue past that
first fiscal year looking for an older, possibly-positive one).

**`_build_ffo` — Gordon growth model on FFO per share, per scenario:**
```
gordon_multiple = (1 + g) / (r - g)     # g = terminal_growth, r = discount_rate (cost of equity)
per_share = round(ffo_per_share * gordon_multiple, 2)
```
`gordon_multiple` IS the scenario's implied fair P/FFO multiple — no arbitrary
target-multiple constant (unlike P/B×ROE's clamped `fair_pb`) is needed. Each
scenario reads its OWN `discount_rate`/`terminal_growth`; a scenario whose
`r`/`g` are missing/non-numeric or `r <= g` is skipped (Turkish note, not
fabricated) — Package 1's ERP-spread guard (`sanity._MIN_ERP_SPREAD`) makes
`r > g` the normal case, but this still guards defensively. Band: recompute
`per_share` at `discount_rate ± 1pp` (`sensitivity._DISCOUNT_RATE_STEP`, `g`
held fixed), take min/max, round 2dp — exactly like `_pb_roe_scenario_band`
(Sec.8), including the same ±10% (`_band`) fallback when fewer than
`_MIN_GRID_CELLS_FOR_BAND` grid points are usable. Returns `(None, notes)`
when FFO itself can't be built, or `({"scenarios": {...}, "ffo_per_share":
float, "implied_pffo": {"bear"/"base"/"bull": float}}, notes)` otherwise —
`scenarios` has the SAME `{"per_share", "lo", "hi"}` shape as `_build_pb_roe`
(Sec.8), so downstream consumption is unchanged; `implied_pffo` is a sibling
key (per-scenario `gordon_multiple`, rounded 1dp), informational only.

**Engine routing (Sec.11):** `reit` calls `_build_ffo` first. If it returns
`None` (no fiscal year with both `NetIncome`/`Depreciation`, or FFO `<= 0`),
the engine falls back to `_build_pb_roe` (same call `financial` makes) with a
Turkish note explaining the fallback. Output: `"ffo"` (new key, `None` unless
the FFO build succeeded) alongside the existing `"pb_roe"` key (populated
only when `financial`, or when `reit` fell back — `None` otherwise for
`reit`). Every site that would otherwise read `pb_roe` as the headline/
triangulation anchor for `reit` (the `fair_value_range` build, the
triangulate `dcf_base_band`) reads `ffo` instead when it's non-`None`, else
`pb_roe` (the fallback) — same "pick the right block" pattern, not new band
logic, since both blocks share one shape.

**P/FFO multiples signal (Sec.6/VALUATION.md Sec.7):** `multiples.
multiples_history` gains a `pffo` column (`fy_price * shares_fy / ffo_fy`,
`None` unless `ffo_fy > 0` and `shares_fy` present — `ffo_fy = net_income_fy +
depreciation_fy - gain_on_sale_re_fy + re_impairment_fy` (gain/impairment
default to 0.0 when untagged), same proxy as above). The engine computes the current
P/FFO (`price / ffo_per_share`, using the same latest-usable FFO
`_select_latest_ffo` returns) and its historical percentile (`pffo_pct`),
threaded into `triangulate.triangulate(..., pffo_pct=...)`. For
`sector_type == "reit"`, `triangulate._raw_multiples_signal`'s primary
candidates become `(pffo_pct, ps_pct)` — P/FFO first, P/S fallback — NOT P/E
(P/E is meaningless for REITs for the same depreciation reason FFO exists).
Every other sector's multiples-signal candidates are unchanged.

## 9. Sensitivity — `sensitivity.sensitivity_matrix(base_assumptions, fcf0, shares, dilution_rate) -> dict`

3×3 over base scenario: growth `g-0.02, g, g+0.02` (rows) × discount rate
`r-0.01, r, r+0.01` (cols). Each cell = `dcf_per_share(...)` per-share (None if
that cell has `r <= g_t`). **No `net_debt` parameter** (Sec.4). Returns:
```python
{"growth_values": [...3], "dr_values": [...3], "matrix": [[3x3 floats|None]],
 "lo": min, "hi": max, "high_uncertainty": bool}   # (hi-lo)/base_cell > 0.60
```

Engine passes whichever `fcf0` the headline `fair_value_range` actually
reflects: for `cyclical` filers with a successfully-computed
`normalized_variant`, that means the normalized fcf0, not the raw one (Sec.8)
— so the reported grid is never silently describing a different cash-flow
base than the headline band. Hyper-grower mode's own sensitivity behavior
(each hyper scenario's band, Sec.4) is unrelated to and unchanged by this —
this `sensitivity` key always reflects the standard/cyclical FCF-DCF grid.

**Exception (Sec.8a):** when the headline is the earnings-power-value (EPV)
anchor (`earnings_power_headline == True`), this invariant is deliberately
broken by design — EPV has no growth axis to grid over, so this
`sensitivity` matrix (and `reverse_dcf.implied_growth`, Sec.5) keep reflecting
the secondary, suppressed FCF-DCF base instead, as documented evidence of the
suppression. See Sec.8a's "Exception to Sec.9's 'same cash-flow base'
invariant" for the full rationale and the note appended in that case.

**Exception (Sec.8b):** the same break applies, for the same underlying
reason, when the headline is instead the mature revenue-first DCF
(`mature_revenue_headline == True`, Sec.8b) — this method's own growth axis
is the realized revenue CAGR fading to terminal growth over a 7-year
horizon, not the standard `growth_5y ± 2pp` grid this matrix builds, so
re-gridding around it would describe a different model, not the headline's
own sensitivity. This `sensitivity` matrix (and `reverse_dcf.implied_growth`)
keep reflecting the secondary, suppressed FCF-DCF base here too, with the
Turkish note documented in Sec.8b.

## 10. Triangulation — `triangulate.triangulate(price, dcf_base_band, implied_growth, realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type, hyper_growth=False, bull_band=None, reverse_dcf_status=None, raw_growth_pair_pct=None, growth_adj_pct=None, earnings_power_headline=False, mature_revenue_headline=False, midgrowth_revenue_headline=False, pffo_pct=None, cyclical_fcfe_headline=False) -> dict`

Direction signal per method (`"ucuz" | "makul" | "pahali" | "veri_yok"`):
- **DCF**: price < band.lo → ucuz; price > band.hi → pahali; else makul.
  (For `financial` use the pb_roe base band; for `reit` use the FFO
  Gordon-growth anchor's base band -- Sec.8c -- or its pb_roe fallback.)
- **Reverse DCF**: compare `implied_growth` to reference growth
  (`realized_cagr` if not None else `base_growth`): implied > ref + 0.03 →
  pahali; implied < ref - 0.03 → ucuz; else makul. When `reverse_dcf_status`
  is `"above_bracket"`/`"below_bracket"` (Sec.5), this signal is forced to
  pahali/ucuz directly, even when `implied_growth` is `None` — there's no
  numeric implied growth to compare, but the direction is already known (a
  price the model can't reach even at its most optimistic/pessimistic growth
  is definitionally expensive/cheap). Default `None`/`"ok"` preserves the
  original implied-vs-reference comparison.
- **Multiples**: primary percentile = pe (fallback ps, then pfcf; for
  growth_unprofitable use ps first; for reit use pffo first, fallback ps --
  Sec.8c -- never pe; **for a leveraged filer use ev_ebitda first**, see next
  paragraph). pct > 70 → pahali; pct < 30 → ucuz; else makul (position against
  the company's OWN multiple history — axis-a). **Two divergence checks
  (VALUATION.md Sec.7) can flip the raw signal to `"karisik"` (mixed), in this
  precedence order:**
  1. **Growth-adjusted (axis-a refinement, highest precedence):** when both
     `raw_growth_pair_pct` (the raw multiple's percentile — P/E in standard
     mode, EV/Sales in hyper-grower mode) and `growth_adj_pct` (the
     growth-adjusted multiple's percentile) are present AND fall in different
     directional buckets. **Skipped entirely when ev_ebitda is primary** (the
     leveraged case): this axis is P/E-based, and a leveraged filer's P/E is
     exactly the distorted read the ev_ebitda-primary routing exists to avoid,
     so a P/E-vs-PEG disagreement must not override the ev_ebitda own-history
     signal.
  2. **Sector-relative (axis-b):** `sector_ratio` = current primary multiple ÷
     its Damodaran sector median (SAME primary the axis-a percentile uses;
     picked with the identical candidate order + first-non-None-percentile
     rule). Bucketed `> 1.25` → pahali-vs-sector, `< 0.80` → ucuz-vs-sector,
     else in-line — a **sector-RELATIVE** band, never an absolute multiple
     value. When its bucket disagrees with the own-history bucket, the signal
     becomes `"karisik"`. Only evaluated when a usable sector median exists;
     `sector_ratio is None` (missing median — e.g. reit whose primary is
     P/FFO, for which no Damodaran median exists) disables the axis and
     preserves the pure own-history signal.

  **Leverage gate (ev_ebitda-primary routing, VALUATION.md Sec.2/Sec.7):**
  `triangulate.triangulate` takes `ev_ebitda_pct` and `net_debt_to_ebitda`;
  it computes `leveraged = net_debt_to_ebitda is not None and
  net_debt_to_ebitda >= _LEVERAGE_EBITDA_RATIO` (`= 1.0`). For a leveraged
  filer (any sector except the three with their own primary above —
  `growth_unprofitable`/`reit`, plus `financial` whose DCF slot isn't a
  multiples signal), the raw own-history primary becomes `ev_ebitda_pct`,
  falling back to the usual pe→ps→pfcf order only when `ev_ebitda_pct` is
  `None` (no usable EV/EBITDA history). Rationale: EV/EBIT(DA) is
  capital-structure-neutral, so it ranks a leveraged filer cleanly where P/E
  is distorted by leverage. The engine derives `net_debt_to_ebitda` from
  current `metrics["net_debt"]/metrics["ebitda"]` (both > 0, else `None` =
  net-cash/unusable = not leveraged) and mirrors the same candidate reorder
  when picking the sector-axis primary — so a leveraged filer's axis-b is
  disabled (no Damodaran EV/EBITDA median exists, exactly like reit's P/FFO)
  rather than silently falling back to a P/E sector comparison. Surfaced at
  `valuation["multiples"]["net_debt_to_ebitda"]` / `["leveraged"]` (Sec.11).

  When neither divergence fires (both agree, or the relevant inputs are
  `None`), the raw own-history signal stands unchanged. `karisik` is a
  substantive signal (not `veri_yok`), so it naturally can't join a
  pahali/ucuz/makul majority — it lowers confidence exactly as a genuine
  disagreement should.

Confidence: all three agree (ignoring veri_yok) → `"YÜKSEK"`; exactly two agree
→ `"ORTA"`; else (scattered, or ≥2 veri_yok) → `"DÜŞÜK"`. Returns
`{"signals": {"dcf": .., "reverse_dcf": .., "multiples": ..},
  "confidence": .., "direction": <majority signal or "belirsiz">,
  "rationale": {...}}`. Signal codes are ASCII: `ucuz`/`makul`/`pahali`/
`yuksek_beklenti`/`karisik`/`veri_yok`.

**EPV confidence ceiling (Sec.8a):** `earnings_power_headline` (default
`False`) is set by the engine when the headline `fair_value_range` came from
the earnings-power-value (EPV) anchor rather than the FCF-DCF (Sec.8a —
mature, FCF-suppressed-but-profitable filers like Amazon). When `True` and
the confidence computed above would otherwise be `"YÜKSEK"`, it is capped to
`"ORTA"`: with EPV as the headline, the DCF leg (now EPV, NetIncome-derived)
and the multiples leg both ultimately derive from the same underlying
earnings signal, so three-way agreement is weaker evidence than when the DCF
leg is an independent, FCF-based estimate. The `rationale["confidence"]`
string gets an appended Turkish clause explaining the cap. `False` (the
default) preserves existing behavior for every other caller/sector.

**Mature revenue-first DCF confidence ceiling (Sec.8b):** `mature_revenue_
headline` (default `False`) is the equivalent flag for the mature,
FCF-suppressed-but-growing revenue-first DCF (Sec.8b — e.g. Amazon-shaped
filers whose realized growth clears the growth gate). Same `"YÜKSEK"` →
`"ORTA"` cap, same appended rationale clause, for the same reason as the EPV
cap immediately above: the DCF leg (now the revenue-first model) and its own
reverse-DCF leg (`revenue_dcf.implied_start_growth` over the identical
revenue/margin path) are derived from one model, not two independent ones,
so three-way agreement is weaker evidence here too. Mutually exclusive with
`earnings_power_headline` in practice — `engine.py` never sets both `True` —
but if both were ever `True`, the `earnings_power_headline` cap message
takes precedence (same cap either way, just one rationale string).

**Mid-growth revenue-first DCF confidence ceiling (Sec.8d):**
`midgrowth_revenue_headline` (default `False`) is the equivalent flag for the
mid-growth, loss-making revenue-first DCF (Sec.8d — `growth_unprofitable`
filers growing 12–20%). Same `YÜKSEK → ORTA` cap, same appended rationale
clause, for the same reason as the two caps above: the DCF leg (the
revenue-first model) and its reverse-DCF leg derive from one model, not two
independent ones. Mutually exclusive in practice with the two headline flags
above.

**Cyclical sustainable-growth FCFE confidence ceiling (Sec.8e):**
`cyclical_fcfe_headline` (default `False`) is the equivalent flag for the
capital-intensive-cyclical sustainable-growth FCFE anchor (Sec.8e —
`cyclical` filers, e.g. Micron/MU, whose FCF-DCF is gated as unreliable by
the SAME `_fcf_dcf_unreliable` test the mature/EPV path uses). Same `YÜKSEK
→ ORTA` cap, same appended rationale clause, for the same reason as the
other caps: the DCF leg IS the earnings-based FCFE anchor here, so three-way
"agreement" with multiples is not independent confirmation. Mutually
exclusive in practice with the other headline flags above.

## 11. Engine — `engine.run_valuation(normalized, ratios, metrics, price, price_df, assumptions, sector_type, damodaran_dir=None, sic_description=None, hyper_growth_extras=None) -> dict`

Orchestrates everything above. Never raises for missing data (only for
programmer errors); every unavailable piece is None + a Turkish note in
`notes`. `hyper_growth_extras` is the optional LLM/user-refined
hyper-grower input (Sec.3/Sec.5 below); `None` (the default) keeps the
hyper-grower path fully deterministic. Return shape (the **`valuation` dict**
consumed by interpret phase 2, CLI card, HTML report, and store):

```python
{
  "sector_type": str,
  "fcf0": float|None, "fcf0_source": "ttm"|"3y_avg"|None,
  "dcf": {
     "enabled": bool, "disabled_reason": str|None,
     "scenarios": {"bear": {"per_share", "lo", "hi"}, "base": {...}, "bull": {...}}|None,
     "normalized_variant": same shape|None,
     "high_growth_flag": bool,  # LEVER 4; True iff at least one scenario's
                                # growth_5y > 0.40 (_STANDARD_DCF_HIGH_GROWTH_FLAG)
                                # -- reporting-only signal (a Turkish note names the
                                # triggering scenario(s)), since this standard
                                # two-stage path has no arrival-point safety net
                                # (unlike hyper-grower/mid-growth revenue-first).
                                # Never changes any computed value.
  },
  "pb_roe": {"scenarios": {...}, "fair_pb": float,
             "justified_pb_flag": "above_reference"|"below_reference"|None,
            }|None,  # financial's FALLBACK anchor (Sec.8f) when rim (below)
                                         # couldn't be built; also reit's
                                         # FALLBACK anchor when ffo (below)
                                         # couldn't be built -- Sec.8c. fair_pb/
                                         # justified_pb_flag: WP5, raw (unclamped)
                                         # justified P/B + reference-band flag.
  "rim": {"scenarios": {"bear"/"base"/"bull": {"per_share","lo","hi"}},
          "per_share": float, "bve0": float, "book_value_per_share": float,
          "normalized_net_income": float, "roe": float}|None,
         # Sec.8f; financial's PRIMARY anchor (multi-period residual income
         # model). None for every sector other than financial, and for
         # financial itself when RIM couldn't be built (pb_roe above is
         # populated instead in that fallback case).
  "ffo": {"scenarios": {"bear"/"base"/"bull": {"per_share","lo","hi"}},
           "ffo_per_share": float,
           "implied_pffo": {"bear"/"base"/"bull": float}}|None,
          # Sec.8c; reit's FFO Gordon-growth anchor. None for every sector
          # other than reit, and for reit itself when FFO couldn't be built
          # (pb_roe above is populated instead in that fallback case).
  "earnings_power": {"scenarios": {"bear"/"base"/"bull": {"per_share","lo","hi"}},
                      "per_share": float, "normalized_net_income": float,
                      "cost_of_equity": float, "sanity_applied": bool}|None,
                     # Sec.8a; populated whenever sector_type == "mature" and
                     # hyper-grower mode is off, regardless of whether it
                     # became the headline.
  "earnings_power_headline": bool,  # Sec.8a; True only when the FCF-DCF
                                     # reliability gate (_fcf_dcf_unreliable)
                                     # switched the headline to EPV.
  "fair_value_range": <shape from §4, built from dcf.scenarios or pb_roe (or,
                        for reit, ffo -- §8c);
                        for cyclical sector_type, from dcf.normalized_variant
                        instead when available -- see §8 -- UNLESS the
                        cyclical FCF-DCF reliability gate additionally fires
                        (§8e; same _fcf_dcf_unreliable test §8a uses), in
                        which case the headline instead comes from the
                        cyclical sustainable-growth FCFE anchor
                        (cyclical_fcfe_headline true) or, when that anchor
                        can't clear the EPV floor, the EPV floor itself
                        (epv_headline true) -- normalized_variant is then a
                        secondary cross-check, not the headline; overridden by
                        the hyper-grower revenue-first DCF base band, ahead of
                        all of the above, whenever hyper_growth is true --
                        see §3 below;
                        for mature sector_type, overridden by the
                        earnings-power-value (EPV) anchor instead, ahead of
                        the raw dcf.scenarios band, whenever
                        earnings_power_headline is true -- see §8a; UNLESS the
                        mature revenue-first DCF (§8b) both cleared its own
                        growth gate AND its base per-share beats the EPV base
                        floor, in which case mature_revenue_headline is true
                        instead and the revenue-first band leads>,
  "reverse_dcf": {"implied_growth": float|None, "realized_cagr_5y": float|None,
                   "realized_label": "FCF 5y"|"FCF 3y"|"gelir 5y"|"gelir 3y"|None,
                   "bracket_status": "ok"|"above_bracket"|"below_bracket"|"no_data"},
                  # standard mode: FCF-CAGR reference (Sec.5/F6); hyper-grower
                  # mode: revenue-CAGR reference + revenue_first_dcf's own
                  # implied start-growth (Sec.5/F6); bracket_status from
                  # reverse_dcf.implied_growth_with_status (standard mode) or
                  # a fixed "ok" (hyper-grower mode, see Sec.5).
  "multiples": {"history": [...],
                 "current": {"pe","ps","pfcf","pffo","ev_ebit","ev_ebitda"},
                 "pe_percentile", "ps_percentile", "pfcf_percentile",
                 "pffo_percentile",  # Sec.8c; reit's primary multiples signal input
                 "ev_ebit_percentile", "ev_ebitda_percentile",  # Sec.6; informational EV multiples
                 "net_debt_to_ebitda": float|None,  # current leverage ratio
                 "leveraged": bool,  # net_debt/EBITDA >= 1.0 -> EV/EBITDA is the primary multiples signal
                 "history_years": int,
                 "sector": {"available": bool, "industry": str|None,
                             "pe_median","ps_median","pfcf_median"},
                 "growth_adjusted": {   # PEG layer, §6 / VALUATION.md §7
                    "metric": "peg"|"growth_adj_ps",  # peg standard, ev/sales-based in hyper mode
                    "label": str, "raw_label": "P/E"|"EV/S",
                    "value": float|None,          # the growth-adjusted ratio (PEG etc.)
                    "percentile": float|None,     # its position in the historical growth-adjusted series
                    "raw_percentile": float|None, # the raw multiple's own percentile (the divergence pair)
                    "applicable": bool,           # False when P/E<=0 or base growth < 5%
                    "reason": str|None,           # Turkish "uygulanamaz" reason when not applicable
                    "base_growth_pct": float|None,# denominator (base growth_5y in % points), always shown
                    "sector_peg": float|None}},   # Damodaran sector-median PEG, only if growth/peg column present
  "sensitivity": <shape from §9>|None,
  "triangulation": <shape from §10>,
  "hyper_growth": bool,
  "hyper_growth_detail": None | {
     "reasons": [str],   # from sector.detect_hyper_grower, echoed
     "scenarios": {"bear": {"per_share","lo","hi","start_growth",
                              "target_fcf_margin","final_year_revenue",
                              "revenue_multiple"}, "base": {...}, "bull": {...}},
     "probabilities": {"bear": 0.25, "base": 0.50, "bull": 0.25},  # or extras-overridden
     "expected_value": float|None,   # prob-weighted per_share
     "arrival_flag": "makul"|"agresif"|"asiri_agresif"|"gecersiz",
     "tam_usd": float|None,          # from hyper_growth_extras, else None
     "implied": {"growth": float|None, "revenue_10y": float|None,
                  "revenue_multiple": float|None, "steady_state_margin": float|None,
                  "tam_share": float|None},
     "target_margin_source": str,    # e.g. "brüt marj × 0.5 (tavan %30)"
     "target_margin_flag": "above_reference"|None,  # WP4; set when the
                                      # (uncapped) target_base exceeds
                                      # _HYPER_TARGET_BASE_CAP (0.30) -- a
                                      # reporting flag, not an applied clamp.
     "target_margin_pct": float,     # WP4; the (uncapped) target_base itself
     "annual_dilution": float,       # WP1; net-of-SBC dilution rate actually
                                      # used, clamp(shares_yoy - sbc_dilution, 0, 0.05)
     "sbc_dilution_excluded": float, # WP1; the raw SBC-implied issuance rate
                                      # subtracted (0.0 when not applicable)
     "mature_discount_rate": float|None,  # WP3; the shared fade target (see
                                      # "Hyper-grower discount-rate fade" above),
                                      # None when the base discount rate was
                                      # unusable (fade skipped, flat rate used)
     "capex_normalization": None | { # Sec.3.6; None unless the maintenance/growth
        "applied": True,             # CapEx split was applied (capex-heavy filer)
        "capex_intensity": float,    # CapEx / revenue
        "maintenance_capex": float,  # max(D&A, sector Cap Ex/Sales or 5% of
                                      # revenue, WP6) -- floored proxy
        "growth_capex": float,       # capex - maintenance_capex
        "raw_current_margin": float, # actual starting margin (drives the HEADLINE)
        "ops_current_margin": float, # relieved margin (drives the UPSIDE only)
        "upside_per_share": float|None,  # AGGRESSIVE upside base value, NOT headlined
        "upside_lo": float|None,     # upside base band
        "upside_hi": float|None,
        "maintenance_capex_floor_note": str,  # WP6; present only when the
                                      # sector Cap Ex/Sales floor was used AND
                                      # exceeded D&A
     },                              # relief NEVER changes the headline scenarios or
                                      # the suppression decision (reviewer Findings 1-2)
     "suppressed": bool,             # True when the base scenario's per_share <= 0
                                      # (non-credible negative equity value) -- computed
                                      # from the ACTUAL (unrelieved) margin, so capex-heavy
                                      # names still suppress; see "Non-credible negative
                                      # valuation guard" below.
     "suppressed_reason": str|None,  # Turkish explanation, set only when suppressed
     "notes": [str],
  },
  "mature_revenue_headline": bool,  # Sec.8b; True only when sector_type ==
                                     # "mature", the FCF-DCF reliability gate
                                     # fired, the growth gate inside
                                     # _build_mature_revenue_dcf cleared, AND
                                     # its base per-share >= the EPV base
                                     # per-share (the guardrail).
  "mature_revenue_detail": None | {
     "scenarios": {"bear": {"per_share","lo","hi","start_growth",
                              "target_fcf_margin","terminal_growth",
                              "discount_rate"}, "base": {...}, "bull": {...}},
     "start_growth": float,          # realized CAGR/YoY blend, same across all 3 scenarios
     "target_margin_base": float,    # mature target FCF margin before per-scenario scaling
     "target_margin_flag": "above_reference"|None,  # WP4; set when target_margin_base
                                      # exceeds _MATURE_TARGET_CAP (0.15) -- a reporting
                                      # flag, not an applied clamp.
     "current_margin": float,        # 3y-median SBC-adjusted FCF margin (fade start point)
     "steady_state_year": int,       # 7 (_MATURE_STEADY_STATE_YEAR)
     "growth_vs_floor": "adds"|"destroys"|None,  # WP7; caller-side classification
                                      # of this base per-share vs the EPV base per-share
                                      # (set after this dict is returned -- see
                                      # "EPV-floor guardrail" above); None when either
                                      # figure is missing/non-numeric.
  },                                 # Sec.8b; built (attempted) whenever sector_type ==
                                     # "mature" and the FCF-DCF reliability gate fired,
                                     # REGARDLESS of whether it became the headline (may
                                     # be None if the growth gate rejected it or a
                                     # precondition was missing; may be non-None but NOT
                                     # the headline if the guardrail kept EPV instead).
  "midgrowth_revenue_headline": bool,  # Sec.8d; True only when sector_type ==
                                     # "growth_unprofitable", NOT hyper, the method
                                     # built, cleared its 12% growth gate, and its
                                     # base per-share was not suppressed (<= 0).
  "midgrowth_revenue_detail": None | {
     "scenarios": {"bear": {"per_share","lo","hi","start_growth",
                              "target_fcf_margin","terminal_growth",
                              "discount_rate"}, "base": {...}, "bull": {...}},
     "start_growth": float,          # realized CAGR/YoY blend, same across all 3 scenarios
     "target_margin_base": float,    # mature target FCF margin (reference threshold
                                      # _MIDGROWTH_TARGET_CAP 20%, WP4: flag not clamp)
     "target_margin_flag": "above_reference"|None,  # WP4; set when target_margin_base
                                      # exceeds _MIDGROWTH_TARGET_CAP (0.20)
     "current_margin": float,        # 3y-median SBC-adjusted FCF margin (fade start point)
     "steady_state_year": int,       # 8 (_MIDGROWTH_STEADY_STATE_YEAR)
     "annual_dilution": float,       # WP1; clamp(shares_yoy - sbc_dilution, 0, 0.05)
                                      # (net of SBC-driven issuance -- see
                                      # "SBC-driven dilution net-out" under §3)
     "sbc_dilution_excluded": float, # WP1; the raw SBC-implied issuance rate subtracted
                                      # (0.0 when not applicable)
     "financing_shares": float,      # cumulative-burn / price (hyper-style)
     "suppressed": bool,             # True when base per_share <= 0 (falls back to multiples)
  },                                 # Sec.8d; built (attempted) for growth_unprofitable
                                     # non-hyper filers; None when the growth gate rejected
                                     # it or a precondition was missing.
  "cyclical_fcfe_headline": bool,   # Sec.8e; True only when sector_type ==
                                     # "cyclical", the FCF-DCF reliability gate
                                     # (_fcf_dcf_unreliable, shared with §8a)
                                     # fired, AND the FCFE anchor's base
                                     # per-share beat the EPV base floor.
  "cyclical_fcfe_detail": None | {
     "scenarios": {"bear": {"per_share","lo","hi"}, "base": {...}, "bull": {...}},
     "per_share": float,             # base scenario's point estimate
     "normalized_net_income": float, # == earnings_power["normalized_net_income"]
     "roe": float,                   # normalized_net_income / spot latest-FY equity
     "equity": float,                # the spot latest-FY StockholdersEquity used
     "cost_of_equity": float,        # == earnings_power["cost_of_equity"]
     "reinvestment_base": float,     # base scenario's implied g/ROE, display only
     "growth_vs_floor": "adds"|"destroys"|None,  # WP7; caller-side classification
                                      # of this base per-share vs the EPV base per-share
                                      # (set after this dict is returned); None when
                                      # either figure is missing/non-numeric.
  },                                 # Sec.8e; built (attempted) for cyclical
                                     # non-hyper filers whenever the FCF-DCF
                                     # reliability gate fired; None when the
                                     # gate never fired, or a precondition
                                     # (earnings_power/shares/equity/ROE) was
                                     # missing; may be non-None but NOT the
                                     # headline if the guardrail kept EPV
                                     # instead (epv_headline true).
  "assumptions": <the validated AND CLAMPED assumptions dict (Sec.3's
                   clamp_assumptions, F5) -- what's shown here is exactly
                   what every DCF/reverse-DCF/sensitivity/hyper calculation
                   above used>,
  "notes": [str, ...],   # Turkish, e.g. fcf0 fallback, missing Damodaran files,
                          # assumption-clamp notes, reverse-DCF bracket notes,
                          # EPV headline switch/margin-normalization/quality
                          # notes (Sec.8a), mature revenue-first DCF headline
                          # switch/growth-gate/guardrail notes (Sec.8b),
                          # cyclical sustainable-growth FCFE headline switch/
                          # trough-excluded-assumption notes (Sec.8e)
}
```

### Hyper-grower revenue-first DCF (Sec.1/Sec.3, engine wiring)

Independently of `sector_type` (a filer can be `growth_unprofitable` or
`mature` and still trip this), the engine calls
`sector.detect_hyper_grower(metrics, ratios, normalized)` at the top of
`_run_valuation` -- EXCEPT for `sector_type in ("financial", "reit")`, where
hyper-grower detection is skipped entirely (F4: forced `is_hyper_grower =
False`; a revenue-margin hyper-DCF doesn't make sense for those sectors,
which use P/B×ROE (`financial`) or the FFO Gordon-growth anchor (`reit`,
Sec.8c) instead). When it triggers, `engine._build_hyper_growth`
runs the deterministic bear/base/bull `revenue_dcf.revenue_first_dcf`
scenarios (Sec.3.1's per-scenario start-growth/target-margin/discount-rate
table, Sec.3.2's dilution/financing rule -- F2/normalization Work Package 1:
dilution is share-count growth net of SBC-driven issuance, via
`engine._non_sbc_dilution` -- see "SBC-driven dilution net-out" below --
`clamp(shares_yoy - sbc_dilution, 0, 0.05)` where `sbc_dilution =
sbc_latest / market_cap`; SBC is ALSO expensed directly into
`current_margin`/target margins, so netting it back out of the dilution
projection avoids charging the same SBC cost twice), computes a prob-weighted
`expected_value`, an "arrival point" flag from the base scenario's 10-year
revenue multiple (or from `tam_usd`'s share of that revenue when known --
this overrides the multiple-based flag), and the price's implied
start-growth/target-margin (`revenue_dcf.implied_start_growth` /
`implied_target_margin`). `hyper_growth_extras` (Sec.5) can override each
scenario's `target_fcf_margin`/`steady_state_year`/`probability` and supply
`tam_usd`; anything not overridden stays deterministic.

Start-growth anchor (F4): the base scenario's start-growth is
`min(growth_anchor, 0.60)` (`engine._HYPER_START_GROWTH_CAP`, raised from
0.40 by normalization Work Package 5 in lockstep with
`sanity._GROWTH_5Y_HARD_MAX` -- same rationale: the arrival-point flags
below, not this cap, are the real honesty mechanism) (bear/bull scale this
by 0.6x/1.2x before the same cap), where `growth_anchor` blends the realized
multi-year CAGR with the
latest single fiscal year's YoY growth -- `0.5 * realized_cagr + 0.5 *
latest_yoy` -- whenever `latest_yoy` is computable (both the fundamental
fiscal year, `resolve_fundamental_fy(metrics)`, and the year before it have
positive revenue); otherwise `growth_anchor = realized_cagr` alone (a
smoothed 5y/3y CAGR can otherwise lag a hyper-grower's own most recent, and
often materially different, growth rate). A Turkish note is added whenever
the blend is actually used.

Sec.3.1's `target_base` (the mature-state FCF-margin ceiling that the
base-scenario `target_fcf_margin` equals, with bear/bull scaling it by
0.7/1.2), via `engine._hyper_target_base`, is `gross_margin * 0.5` when the
latest-FY gross margin is a known positive number, or a flat `0.20` ceiling
when gross margin is unavailable (replacing the previous
15%-gross-margin-fallback rule, which produced an unrealistically low 7.5%
ceiling for filers with no gross-margin data at all). Either way,
`target_base` is then floored at today's FCF margin (`fcf / latest_revenue`)
whenever that margin is positive -- a filer that is already FCF-profitable
today must never be modeled as if its mature margin collapses below what it
already earns -- and, when gross margin is known, capped back down at that
gross margin (so the floor can raise `target_base` but never push it past
the gross-margin ceiling).

**No longer clamped to an absolute 30% ceiling (normalization Work Package
4).** `_hyper_target_base`'s result used to also be hard-capped at
`_HYPER_TARGET_BASE_CAP = 0.30`; that constant is now a reporting-only flag
threshold, not an applied ceiling -- a genuinely high-gross-margin business
(e.g. gross margin above 60%) can legitimately warrant a mature FCF margin
above 30%, and silently truncating it there was a second, blunt penalty on
top of the discount rate/probabilities that already price the underlying
risk. When the (uncapped) `target_base` exceeds `_HYPER_TARGET_BASE_CAP`,
`_build_hyper_growth` appends a Turkish note naming the value and the
reference threshold, and sets `hyper_growth_detail["target_margin_flag"] =
"above_reference"` (else `None`) -- the report layer surfaces the flag
instead of the value being silently clipped.

`target_margin_source` reports which construction path fired, e.g. `"brüt
marj × 0.5 (tavan %30)"` when the gross-margin path applied unchanged,
`"brüt marj %60 × 0.5 (tavan %30), bugünkü FCF marjına tabanlanmış"` when a
known gross margin was overridden by the current-margin floor, or `"brüt
marj yok: %20 varsayılan tavan, bugünkü FCF marjına (%30) tabanlanmış"` when
gross margin was missing and the 20% default ceiling was overridden by the
current-margin floor -- the `"(tavan %30)"` wording in these strings is
historical/descriptive (it still names the reference threshold used to
derive the source label) and does not imply an applied clamp.

If any sub-step can't be computed (missing revenue/shares/realized growth,
or every scenario's `revenue_first_dcf` call fails), the whole block
degrades to `hyper_growth = False` / `hyper_growth_detail = None` plus a
Turkish note -- even though `detect_hyper_grower` itself returned `True` --
so a broken hyper build never costs the standard valuation below it.

When `hyper_growth` is `True`, the headline `fair_value_range` (and the
triangulation DCF band) are built from the hyper base band instead of
`dcf.normalized_variant`/`dcf.scenarios`, and a Turkish note is appended
explaining the switch; the standard FCF-DCF (`dcf.scenarios`) is still
computed and returned as a secondary figure, exactly as the cyclical
`normalized_variant` is. `triangulate()`'s signature/behavior is unchanged
in this milestone -- the hyper band simply flows into the existing DCF
signal via `primary_dcf_scenarios`.

### SBC-driven dilution net-out (`engine._non_sbc_dilution`, normalization Work Package 1)

Shared by the hyper-grower path (above) and the mid-growth revenue-first DCF
(Sec.8d) -- both project per-share dilution from `metrics["shares_yoy"]`
(raw share-count growth), which already includes shares issued via
stock-based compensation (SBC). Both paths ALSO expense SBC directly in
`current_margin`/target margins (SBC is subtracted from cash flow the same
way `_select_fcf0` treats it for the standard DCF, Sec.4). Projecting the
raw `shares_yoy` as the dilution rate on top of that would charge the same
SBC cost twice -- once as a margin drag, once again as per-share dilution.

`_non_sbc_dilution(metrics, normalized, fy) -> (rate, note, sbc_dilution_
excluded)`: when `metrics["market_cap"]` is a usable positive number,
`sbc_dilution = sbc_latest / market_cap` (`sbc_latest` from the `SBC`
concept at `fy`, `0.0` when missing), `non_sbc = max(0.0, shares_yoy -
sbc_dilution)`, and `rate = clamp(non_sbc, 0.0, _HYPER_DILUTION_CAP)` (the
existing 0.05 cap, unchanged). When `market_cap` is unusable, the function
falls back to the pre-WP1 behavior byte-for-byte: `rate =
clamp(shares_yoy, 0.0, _HYPER_DILUTION_CAP)`, no netting. Whenever the net
actually changed something (`sbc_dilution > 0` and `shares_yoy > 0`), a
Turkish note is appended ("SBC ihraçları marjda gider olarak zaten
fiyatlandığı için dilüsyon projeksiyonundan çıkarıldı (çift sayım
önlendi)...") and `sbc_dilution_excluded` carries the raw SBC-implied
issuance rate that was subtracted (`0.0` otherwise). `hyper_growth_detail`
and `midgrowth_revenue_detail` (Sec.8d) both gain two additive fields from
this: `annual_dilution` (the net, already-clamped rate actually used) and
`sbc_dilution_excluded` (the subtracted amount, for transparency). Never
raises. Financing shares (extra shares issued to fund cash-burn years) are
unaffected -- that mechanism is SBC-independent and unchanged.

### Terminal-growth anchor: `min(risk_free, 4%)` (normalization Work Package 2 / LEVER 1)

Every path in this engine that needs a terminal/perpetuity growth rate --
the hyper-grower revenue-first DCF, the standard/mature/mid-growth paths via
`assumptions["*"]["terminal_growth"]` -- now derives it from the SAME rule,
Damodaran's practical guideline that a stable perpetuity growth rate should
not exceed the (nominal) risk-free rate:

```
terminal_growth = min(risk_free_rate, sanity._TERMINAL_GROWTH_MAX)   # 4% cap
```

There is deliberately no cohort differentiation any more (an earlier
version of the hyper-grower path used a fixed 2.5% terminal growth
regardless of the risk-free rate) -- a hyper-grower that has reached
`steady_state_year` is, by definition, a now-mature company; the extra risk
it carried while still growing is already priced into the elevated
bear/base/bull discount rates and scenario probabilities, so discounting
its terminal growth a second time on top of that would be a third, layered
penalty for the same risk.

**Two independent call sites, two resolution orders (LEVER 1 fix -- see
"Known gaps/roadmap" below for the bug this replaced):**

- **Hyper-grower path (`engine._run_valuation`):** `sector_data =
  damodaran.load_sector_data(...)` is loaded once, early (moved ahead of the
  hyper-grower build specifically so this anchor is available to it --
  purely a load-order change, same deterministic local CSV read either
  way). `risk_free_pct = sector_data.get("risk_free")` (the GLOBAL US
  risk-free rate off `erp.csv`, independent of any SIC/industry matching).
  When numeric: `terminal_growth_anchor = min(risk_free_pct / 100.0,
  sanity._TERMINAL_GROWTH_MAX)`. Else: falls back to the historical flat
  constant `engine._HYPER_TERMINAL_GROWTH = 0.025`. This same
  `terminal_growth_anchor` is threaded into `_build_hyper_growth` as its
  `terminal_growth` parameter (superseding the old always-0.025 default).
- **Standard/mature/mid-growth paths (`rule_based._terminal_growth_anchor`,
  used by `rule_based._default_assumptions` for EVERY scenario, script
  provider and LLM-fallback alike):** resolution order is (1)
  `capm["risk_free"]` (the SIC-matched sector's CAPM `risk_free`, when
  `capm` is present and numeric), (2) `risk_free_pct` -- the SAME global
  `erp.csv` risk-free rate the hyper path reads, passed down from
  `interpret/analyzer.py`'s `run_valuation` orchestration specifically as a
  fallback for filers whose SIC doesn't match any Damodaran industry (so
  `capm.compute_cost_of_equity` returns `None`) -- without this fallback
  those filers' terminal growth would ALSO flatten to the old constant on
  top of their already-flat (non-CAPM) discount rate, a double penalty; (3)
  `rule_based._DEFAULT_TERMINAL_GROWTH = 0.025`, only when neither of the
  above is numeric. Returns `(terminal_growth, from_risk_free)` -- the
  second element feeds the per-scenario `story` sentence's terminal-growth
  clause (naming whether the number is risk-free-derived or the flat
  fallback).

Both call sites therefore converge on the identical `min(risk_free, 4%)`
rule; they differ only in WHERE their risk-free number can come from (a
SIC-matched CAPM figure first, for the assumptions-driven paths) -- not in
the rule itself. `sanity._TERMINAL_GROWTH_MAX` (4%) remains the ultimate
ceiling in both places, and `sanity.validate_assumptions`/`clamp_assumptions`
(Sec.3) still independently enforce it on whatever `terminal_growth` ends up
in `assumptions`.

### Hyper-grower discount-rate fade to a mature rate (normalization Work Package 3)

The hyper-grower revenue-first DCF fades revenue growth and FCF margin
toward a mature steady state (Sec.3.1/above) but, before this Work Package,
discounted every projected year at a FIXED cohort rate (14%/12%/10% for
bear/base/bull) all the way through the terminal value -- internally
inconsistent with a model whose whole point is that the business becomes
mature by `steady_state_year`: since most of a hyper-grower's value sits in
the far years and the terminal value, a permanently-elevated flat rate
systematically crushes exactly the cash flows the fade is supposed to have
already de-risked. Damodaran's standard fix, now implemented, is to fade the
discount rate alongside the cash flows.

**`mature_discount_rate` (`engine._run_valuation`):** computed once, shared
across all three scenarios' fades:

```
base_discount_rate_for_fade = assumptions["base"]["discount_rate"]
mature_discount_rate = max(base_discount_rate_for_fade,
                            terminal_growth_anchor + sanity._MIN_ERP_SPREAD)
```

`assumptions["base"]["discount_rate"]` is already CAPM-aware (Damodaran
sector beta relevered with the firm's own D/E, plus ERP and the risk-free
rate, Sec.4's VALUATION.md cross-reference) and already run through
`sanity.clamp_assumptions` (Sec.3) by this point -- no separate CAPM
computation happens in this module. The `max(...)` floor guards against the
fade ever landing inside the ERP-spread guard's forbidden zone (a rate only
barely above `terminal_growth`). When `base_discount_rate_for_fade` is
missing/non-numeric, `mature_discount_rate = None` and every downstream call
degrades to the pre-WP3 flat-rate behavior exactly (see below) -- this
parameter is purely additive.

**Threaded through to `_build_hyper_growth` and `revenue_dcf.
revenue_first_dcf`:** each of the three scenarios still STARTS its fade from
its own fixed cohort rate (14%/12%/10%, unchanged) -- only the fade's
MATURE TARGET is now this one shared `mature_discount_rate`, so bear/base/
bull each fade from a different starting point to the same ending point.
`revenue_first_dcf` gained an optional `mature_discount_rate=None` parameter
(default `None` preserves byte-for-byte the old flat-rate behavior for every
existing direct caller/test):

- **Rate path (`revenue_dcf._discount_path`):** mirrors the growth/margin
  fade's exact shape -- linear from `discount_rate` (year 1, the cohort
  rate) to `mature_discount_rate`, reaching it exactly at
  `steady_state_year` and holding there for any remaining years: `r_t =
  discount_rate + (mature_discount_rate - discount_rate) * min(t-1,
  steady_state_year-1) / (steady_state_year-1)` (collapses to a flat
  `mature_discount_rate` for every year when `steady_state_year <= 1`,
  avoiding a division by zero).
- **Discounting:** each year's present value uses the CUMULATIVE product of
  `(1 + r_t)` across years 1..t (`Π(1+r_i)`), not `(1+r)**t` -- a proper
  path-dependent discount factor for a rate that changes year to year.
- **Terminal value:** `tv = fcf_terminal * (1 + terminal_growth) /
  (mature_discount_rate - terminal_growth)` (Gordon growth AT THE MATURE
  RATE -- a steady-state perpetuity should be discounted at the steady-state
  cost of equity, not the elevated cohort rate), discounted by the
  cumulative product over the full horizon (`pv_tv = tv / df_horizon`).
  Must have `mature_discount_rate > terminal_growth` (raises `ValueError`
  otherwise, mirroring the existing `discount_rate > terminal_growth` guard).
- The returned dict gains an additive `discount_path` key (`horizon` floats)
  whenever `mature_discount_rate` was provided.
- `revenue_dcf.implied_start_growth`/`implied_target_margin` both accept and
  forward the same optional `mature_discount_rate` parameter, so the
  reverse-DCF/implied-metric bisections used by the hyper-grower path solve
  over the SAME faded-rate model the headline scenarios use.

`hyper_growth_detail`'s per-scenario dict gains an additive
`mature_discount_rate` field (Sec.11) mirroring this. Mid-growth (Sec.8d)
and mature (Sec.8b) revenue-first DCF do NOT receive this fade -- their own
`discount_rate` already comes from the CAPM-aware, clamped `assumptions`
pipeline per scenario (not a fixed hyper-style cohort rate), so there is no
elevated-then-mature gap to fade across.

### Non-credible negative valuation guard (`suppressed`)

For capex-heavy hyper-growers (e.g. data-center builders) growth CapEx can run
many multiples of revenue, so today's FCF margin is deeply negative; since the
margin only fades LINEARLY to a positive mature target over the horizon, the
discounted early-year cash burn can exceed the positive terminal value and
yield a negative equity value for the base scenario. **Trigger:** base
scenario `per_share <= 0`. When this fires, `_build_hyper_growth` sets
`hyper_growth_detail["suppressed"] = True` plus a Turkish
`suppressed_reason`, and `run_valuation` responds with two effects: the
headline `fair_value_range` is emptied (`_build_fair_value_range` falls back
to `_empty_fair_value_range()` because `primary_dcf_scenarios` is set to
`None`), and the DCF leg of `triangulation` becomes `"veri_yok"` (`_dcf_signal`
returns `SIGNAL_NO_DATA` for a missing band). `hyper_growth`/`is_hyper_grower`
stays `True` (the mode is still detected) and the (negative) bear/base/bull
scenarios remain visible under `hyper_growth_detail["scenarios"]` for
transparency -- they are simply never published as a fair value. The
`scenario_meta`/headline-note switch described below is skipped while
suppressed (the engine checks `hyper_growth_active and not
hyper_growth_detail.get("suppressed")` before applying it).

Display consistency: in hyper-grower mode, `fair_value_range`'s per-scenario
`growth`/`discount_rate`/`note` fields are also switched over to reflect the
revenue-first DCF's own scenario inputs (`engine._hyper_scenario_meta`) --
NOT the standard clamped `assumptions[scenario]` the headline band no
longer actually uses. Concretely, for each scenario whose hyper cell has a
`start_growth`/`target_fcf_margin`: `growth` = `"%<start_growth> başlangıç
→ %2.5 terminale fade"`, `discount_rate` = the fixed hyper per-scenario rate
(`"%14"`/`"%12"`/`"%10"` for bear/base/bull — raised from an earlier
12/10/9% to reflect the risk premium a hyper-grower carries over a mature
filer, and to never dip below the 10% unprofitable-company discount-rate
floor, Sec.3), and `note` names the scenario
(`"kötümser"`/`"temel"`/`"iyimser"`), the start growth, the fade, the mature
target FCF margin, and the discount rate, e.g. `"Hiper-büyüme temel:
başlangıç büyüme %40 (10 yılda %2.5 terminale fade), olgun FCF marjı %30,
iskonto %12."` A scenario missing its hyper cell (a failed
`revenue_first_dcf` call for that scenario) falls back to the
assumptions-derived string for that scenario, exactly as the non-hyper path
always has.

**Known display inconsistency (documented, not yet fixed):** `"%2.5"` in the
`growth`/`note` strings above is a LITERAL, hard-coded substring in
`_hyper_scenario_meta` -- it does NOT read the actual `terminal_growth_anchor`
(above) the engine passes into `_build_hyper_growth`/`revenue_first_dcf` for
this run. Since the terminal-growth-anchor rule (normalization Work Package
2/LEVER 1) can now resolve to any value up to `sanity._TERMINAL_GROWTH_MAX`
(4%) depending on the risk-free rate, the displayed fade target can be wrong
whenever the resolved anchor isn't exactly 2.5% -- unlike the sibling
`_mature_scenario_meta`/`_midgrowth_scenario_meta` helpers (Sec.8b/Sec.8d),
which already interpolate the actual resolved terminal growth into their
own equivalent strings via a `terminal_str` variable. This affects only the
DISPLAYED text (`fair_value_range`'s `growth`/`note` fields for hyper-grower
mode); the underlying `revenue_first_dcf` computation itself correctly uses
the resolved `terminal_growth_anchor`, so the computed per-share values are
unaffected. Flagged here as a known gap rather than silently documented as
if it were dynamic.
Round all per-share values to 2 decimals, percentiles to 1, growth rates to 4.

### Sec.3.6 — Maintenance/growth CapEx split (`engine._maintenance_adjusted_margin`)

The `suppressed` guard above is a correct-but-conservative backstop: it
declines to publish a negative band, but it also leaves a genuinely
financeable capex-heavy grower (a data-center builder like APLD) with no
DCF headline at all. Roadmap Madde 1 addresses the root cause instead of
only guarding the symptom. The problem: `_build_hyper_growth`'s starting FCF
margin was `(OCF − total CapEx − SBC) / revenue`. For a filer whose CapEx is
many multiples of revenue, that margin is deeply negative — but most of that
CapEx is **growth** CapEx that builds the very future revenue the
revenue-first projection already captures via its growth path. Subtracting it
from the *starting* margin double-penalizes the same expansion (once as
today's cash outflow, again as forgone terminal cash flow).

`_maintenance_adjusted_margin(normalized, metrics, raw_current_margin,
sector_capex_sales=None) -> (ops_margin, capex_normalization | None)`
computes the growth-CapEx-relieved margin, using depreciation & amortization
(the `Depreciation` concept), **floored at a maintenance-CapEx-as-%-of-
revenue rate**, as the maintenance-CapEx proxy. That floor rate is
`sector_capex_sales` (normalization Work Package 6) when it is a usable
positive number, else the flat `_MAINTENANCE_CAPEX_MIN_PCT_REVENUE` (`=
0.05`) default. `sector_capex_sales` comes from the matched Damodaran
sector's Cap Ex/Sales ratio -- `damodaran.sector_medians(...)["capex_sales"]`
(new optional `capex_sales` column in `multiples.csv`, see
`data/damodaran/README.md`), computed once by `_run_valuation` (alongside
the existing sector-median match used for the multiples comparison, no
second lookup) and threaded through `_build_hyper_growth` into this
function. The revenue-floor mechanism exists (independent of which rate
feeds it) because current-year D&A understates the maintenance burden of a
still-ramping asset base (reviewer Finding 2: a data-center builder's future
depreciation reflects its grown-out fleet, not today's small one); a
sector's own Cap Ex/Sales ratio sizes that floor more accurately than one
flat 5% for sectors (e.g. data-center/telecom/utility) with a genuinely
higher maintenance-capex intensity than the generic default. All figures are
read at `resolve_fundamental_fy(metrics)`. Gate — BOTH must hold, else the
raw margin is returned unchanged and `capex_normalization` is `None`:

- `capex / revenue > _CAPEX_HEAVY_INTENSITY_THRESHOLD` (new constant `= 0.30`)
  — genuinely capex-heavy, not an asset-light software grower.
- `capex > max(d&a, maintenance_floor_pct · revenue)` — there is growth
  CapEx above the floored maintenance level to relieve.

When applied: `maintenance_capex = max(d&a, maintenance_floor_pct ·
revenue)`, `growth_capex = capex − maintenance_capex`, `ops_margin =
raw_current_margin + growth_capex / revenue` (an additive correction on the
caller's raw margin). When the sector floor was actually used AND it
exceeds D&A (i.e. it actually moved the maintenance-CapEx figure),
`capex_normalization` gains an additive
`maintenance_capex_floor_note` Turkish string naming the sector percentage
used in place of the flat 5% default. `sector_capex_sales` absent/non-
positive keeps this whole computation byte-for-byte identical to the
pre-WP6 flat-5%-only behavior.

**The relief is deliberately NOT the headline (reviewer Findings 1–2).** A
finance review showed that relieving growth CapEx from the *starting* margin
while revenue still compounds up the growth path books the revenue ramp but
charges the CapEx funding it *nowhere* — a one-directional over-valuation,
the same owner-earnings add-back double-count SPEC Sec.8b explicitly rejects
(and current-year D&A understates steady-state maintenance for a ramping
fleet, compounding it). And the "correct" fix — a growth-tied reinvestment
charge — is itself unreliable for these names (single-year sales-to-capital
is wildly unstable given lumpy forward CapEx). So:

- `_build_hyper_growth`'s **headline scenarios keep using the ACTUAL
  (unrelieved) `current_margin`**. Capex-heavy names therefore still hit the
  `suppressed` (base `per_share <= 0`) guard above and are dropped from the
  headline — the honest, conservative behavior.
- The relieved `ops_margin` is used ONLY to compute a separate base-scenario
  value reported as an **explicitly-labeled AGGRESSIVE UPSIDE, never the
  headline**: `_build_hyper_growth` adds `capex_normalization` to
  `hyper_growth_detail` = `{"applied": True, "capex_intensity",
  "maintenance_capex", "growth_capex", "raw_current_margin",
  "ops_current_margin", "upside_per_share", "upside_lo", "upside_hi"}` (or
  `None` when not applied), and appends a Turkish note stating the headline
  DCF is suppressed and this upside is what a "growth CapEx normalizes" view
  implies (flagged upward-biased). The `upside_*` band uses the same base
  start-growth/target/discount-rate/dilution/financing-shares as the headline
  base scenario — only the starting margin differs.
- **Finding 3 fix:** the mature-target floor (`_hyper_target_base`) is passed
  the ACTUAL current margin, never the relieved one, so a relieved (possibly
  positive) margin can never leak into the terminal margin.

This helper is used only by the hyper-grower path. The mid-growth path
(Sec.8d) deliberately does NOT apply it — its whole point is a defensible,
not aggressive, value, so a capex-heavy mid-grower whose base suppresses
simply falls back to multiples.

## 8d. Mid-growth loss-making revenue-first DCF — `engine._build_midgrowth_revenue_dcf`

A revenue-first alternative to a **multiples-only** headline for
`sector_type == "growth_unprofitable"` filers that grow the top line at a
real but sub-hyper rate (realized CAGR roughly 12–20%) and are therefore NOT
picked up by `sector.detect_hyper_grower` (which needs CAGR > 20%). Roadmap
Madde 2 — previously deferred deliberately (a multiples fallback was
preferred over a speculative DCF value); now built. Sits between the mature
(Sec.8b) and hyper-grower (Sec.3) revenue-first paths.

Attempted in `run_valuation` only when `sector_type == "growth_unprofitable"`
AND hyper-grower mode is NOT active — a new trailing branch after the mature
`elif`. Reuses `revenue_dcf.revenue_first_dcf` + `_hyper_scenario_band`.

New constants (`engine.py`): `_MIDGROWTH_MIN_GROWTH = 0.12`,
`_MIDGROWTH_TARGET_CAP = 0.20`, `_MIDGROWTH_STEADY_STATE_YEAR = 8`.

- `revenue0` at `resolve_fundamental_fy`; `shares`; missing/non-positive →
  `(None, note)` (falls back to multiples).
- `start_growth = _mature_start_growth(...)` (reused: blended realized CAGR).
- **Growth gate:** `start_growth < _MIDGROWTH_MIN_GROWTH` (12%) OR
  `start_growth <= base.terminal_growth` → `(None, note)`.
- **Target mature FCF margin:** `_hyper_target_base(gm, current_margin)` where
  `gm` = latest-FY positive gross margin. The gross-margin construction
  (hyper path) is used rather than the mature path's operating-margin/
  historical-FCF anchors, which degenerate for a loss-maker with no
  positive-margin history. **No longer clamped to `_MIDGROWTH_TARGET_CAP`
  (normalization Work Package 4)** -- that 20% is now a reporting-only flag
  threshold: when the (uncapped) result exceeds it, a Turkish note is
  appended and `target_margin_flag = "above_reference"` is set (else
  `None`). Reference threshold 20% (between mature's
  15% and hyper's 30%).
- **Current (starting) margin:** `_mature_current_margin(...)` (3-year median,
  negative for loss-makers). The Sec.3.6 CapEx relief is deliberately NOT
  applied here — this path aims for a defensible value, so a capex-heavy
  mid-grower whose base value suppresses falls back to multiples instead.
- **Fade horizon:** `_MIDGROWTH_STEADY_STATE_YEAR` (8).
- **Dilution & financing shares:** hyper-style — `annual_dilution` from
  `engine._non_sbc_dilution` (normalization Work Package 1: net of
  SBC-driven issuance, since SBC is already expensed in the margin fade --
  see "SBC-driven dilution net-out" under §3) and `financing_shares` derived
  from the base scenario's cumulative burn / price (a mid-growth loss-maker
  still funds burn by issuing equity), unlike the mature path's 0.
- **Per scenario:** `discount_rate`/`terminal_growth` from the **clamped
  assumptions** (`growth_unprofitable` is clamped `is_unprofitable=True`, so
  the discount rate is already floored at 10%), NOT hard-coded hyper rates;
  `target_margin = target_base * _MATURE_TARGET_MARGIN_SCALE[scenario]`;
  `start_growth` identical across scenarios (as in the mature path).
- **Suppression guardrail** (hyper-style): base `per_share <= 0` →
  `suppressed = True`; the caller leaves `primary_dcf_scenarios` untouched
  (multiples fallback) rather than publishing a negative band.
- Returns `{"scenarios": {...bear/base/bull {"per_share","lo","hi",
  "start_growth","target_fcf_margin","terminal_growth","discount_rate"}},
  "start_growth", "target_margin_base", "target_margin_flag", "current_margin",
  "steady_state_year", "annual_dilution", "sbc_dilution_excluded",
  "financing_shares", "suppressed"}`, or `(None, notes)`.

### `run_valuation` integration

Priority chain: `hyper-grower > cyclical normalized_variant > (mature-gate
fired: mature-revenue / EPV) > (growth_unprofitable, not hyper: mid-growth
revenue-first) > raw FCF-DCF`. When the mid-growth band is built, not
suppressed, and its base `per_share` is a number, `primary_dcf_scenarios`
becomes its scenarios and `midgrowth_revenue_headline = True`; otherwise the
filer keeps its existing raw-FCF-DCF/multiples fallback (the method's notes
are still surfaced so the reader knows why).

- `scenario_meta`: `_midgrowth_scenario_meta` (mirrors `_mature_scenario_meta`
  with the 8-year fade and "orta-büyüme" wording).
- **Reverse-DCF override (§5 same-base invariant):** revenue-based, mirroring
  the mature override — `revenue_dcf.implied_start_growth(price, revenue0,
  base_terminal_growth, base_discount_rate, current_margin, target_margin_base,
  steady_state_year, shares, annual_dilution, financing_shares)` (uses the
  detail's own `annual_dilution`/`financing_shares` so the implied growth is
  apples-to-apples with the published band); realized reference = revenue
  CAGR; `realized_label` = `"gelir 5y"`/`"gelir 3y"`; `bracket_status = "ok"`.
- **§9 sensitivity exception:** same documented break as EPV/mature — the
  `sensitivity` grid keeps reflecting the secondary FCF-DCF base, with a
  Turkish note.
- **Confidence ceiling (Sec.10):** `midgrowth_revenue_headline` → same
  `YÜKSEK → ORTA` cap as `mature_revenue_headline` (DCF and reverse-DCF legs
  derive from one model).

### Scope

Purely additive: new output keys `midgrowth_revenue_headline` (bool) and
`midgrowth_revenue_detail` (dict|None); does not change any existing key's
meaning, and applies only to `growth_unprofitable` non-hyper filers. A
`growth_unprofitable` filer whose growth gate rejects the attempt, or whose
base value is suppressed, is unaffected (multiples-only headline as before).

## 8e. Cyclical sustainable-growth FCFE anchor — `dcf.fcfe_sustainable_growth_per_share` / `engine._build_cyclical_fcfe`

Addresses capital-intensive `cyclical` filers in a capacity-expansion phase
(canonical case: Micron/MU) whose FCF-DCF headline isn't merely near-trough
(Sec.8's `normalized_variant` already handles that ordinary case) but
structurally suppressed EVERY year by heavy growth CapEx (fab expansion):
even the cycle-mid normalized FCF margin badly understates fair value,
because the FCF-DCF charges the entire growth CapEx as a permanent cash
drain while only booking the modest revenue growth that CapEx is funding.
The theoretically-correct growth-inclusive value instead grows NORMALIZED
EARNINGS with reinvestment-funded ("sustainable") growth: to grow earnings
at rate `g` while holding ROE constant, a firm must retain `b = g / roe` of
net income; the rest is distributable FCFE. Growth genuinely adds value
over the zero-growth EPV floor (Sec.8a) only when `roe > cost of equity` —
this anchor is literally "EPV's normalized earnings, grown along that
identity", so it collapses toward EPV as `roe` approaches the discount rate
and undershoots it when `roe < discount rate` (the guardrail below then
falls back to EPV).

### `dcf.fcfe_sustainable_growth_per_share(ni0, roe, growth_5y, terminal_growth, discount_rate, shares, dilution_rate=0.0, terminal_roe=None) -> dict`

- Raises `ValueError` — never silently "fixes" an invalid input — when
  `ni0 is None`, `shares` is falsy/`<= 0`, `roe <= 0`, or `discount_rate <=
  terminal_growth` (the Gordon-growth terminal value is undefined), mirroring
  `dcf_per_share`'s raise-don't-fix discipline (Sec.4).
- Projects normalized earnings along the SAME 10-year, two-stage path
  `project_fcf` uses for FCF (years 1-5 at `growth_5y`, years 6-10 fading
  linearly to `terminal_growth` via `_year_growth_rate`, Sec.4). For each
  year: `g_eff = min(g_year, roe)`; `ni_year = previous_ni * (1 + g_eff)`;
  reinvestment rate `b = g_eff / roe`; `fcfe_year = ni_year * (1 - b)`,
  discounted at `(1 + discount_rate) ** year`.
  - **Growth capped at ROE, not a flat reinvestment ceiling.** When
    `g >= roe`, the firm cannot fund that growth purely out of its own
    earnings without external equity. Rather than clamp `b` at an arbitrary
    ceiling (booking growth the earnings base can't actually sustain) or let
    `b` exceed 1.0 (inventing cash via a negative payout), the model caps the
    BOOKED growth itself at `roe`: `b` rides up to (but never past) 1.0 and
    distributable FCFE rides down to (but never below) 0 as `g_eff`
    approaches `roe`.
- **Terminal ROE fades to the cost of equity.** The terminal (perpetuity)
  year's OWN reinvestment rate uses `terminal_roe` (default `None`, which
  falls back to the current-period `roe` — backward compatible); years 1-10
  always use the current-period `roe` regardless of `terminal_roe`.
  `terminal_roe_resolved = terminal_roe if terminal_roe is not None else roe`;
  `g_t_eff = min(terminal_growth, terminal_roe_resolved)`;
  `ni_terminal = ni_10 * (1 + g_t_eff)`;
  `b_t = g_t_eff / terminal_roe_resolved`;
  `fcfe_terminal = ni_terminal * (1 - b_t)`. The Gordon-growth denominator
  (`tv = fcfe_terminal / (discount_rate - terminal_growth)`) deliberately
  keeps the ORIGINAL, uncapped `terminal_growth` — only the terminal
  earnings/reinvestment computation is capped, so the already-validated
  `discount_rate > terminal_growth` guard above stays the only gate on
  Gordon-growth validity. Callers pass the scenario's own cost of equity as
  `terminal_roe` (the Damodaran stable-growth-phase convention: a firm's
  excess return over its cost of equity cannot persist indefinitely once
  competitive advantages erode, even when its near-term ROE is higher).
- **FCFE-direct, cost-of-equity discounting, no net-debt bridge** — like
  `dcf_per_share` (Sec.4): normalized net income is already a post-interest,
  post-tax (levered/equity) figure, so `discount_rate` must be a levered cost
  of equity, and `ev == equity` (both keys kept for caller convenience).
  `effective_shares = shares * (1 + dilution_rate) ** 5` — the same
  year-5 dilution-horizon convention `dcf_per_share` uses.
- `equity = sum(discounted fcfe_1..10) + pv(tv)`; `per_share = equity /
  effective_shares`.
- Returns `{"per_share", "ev", "equity", "ni_path" (10 floats), "fcfe_path"
  (10 floats), "tv", "effective_shares"}`. Nothing rounded here — rounding
  is the caller's (`engine.py`'s) responsibility.

### `engine._build_cyclical_fcfe(assumptions, earnings_power, normalized, metrics, shares, dilution_rate) -> tuple[Optional[dict], list[str]]`

- Requires `earnings_power` (Sec.8a's detail dict, now also built for
  `cyclical` filers — see the Sec.8a engine-integration update) with both
  `normalized_net_income` and `cost_of_equity` present, and `shares > 0`;
  missing either → `(None, [])`.
- `ni_norm = earnings_power["normalized_net_income"]` — the SAME
  margin-median-sanitized normalized net income EPV itself uses (Sec.8a);
  this anchor does not introduce a second earnings-normalization rule.
- **ROE = `ni_norm` / SPOT latest-FY `StockholdersEquity`** —
  `to_annual_series(normalized, "StockholdersEquity").get(resolve_fundamental_fy(metrics))`,
  a balance-sheet snapshot at the fundamental fiscal year, **not** an average
  across years (an average-equity denominator was proposed during finance
  review to match the normalized/cycle-adjusted numerator, but that change
  was NOT shipped — see "Design notes" below; this section documents the
  code as it actually behaves). `equity` missing/`<= 0` → `(None,
  ["Döngüsel FCFE çapası hesaplanamadı: özkaynak verisi eksik/negatif."])`;
  `roe <= 0` → `(None, ["Döngüsel FCFE çapası hesaplanamadı: normalize
  edilmiş ROE pozitif değil."])`.
- Per scenario (bear/base/bull): reads that scenario's own
  `growth_5y`/`terminal_growth`/`discount_rate` from `assumptions`; any
  non-numeric value or `discount_rate <= terminal_growth` → that scenario's
  cell is `{"per_share": None, "lo": None, "hi": None}` plus a Turkish note
  (mirrors `_build_dcf_scenarios`, Sec.4). Otherwise calls
  `fcfe_sustainable_growth_per_share(ni_norm, roe, growth_5y, terminal_growth,
  discount_rate, shares, dilution_rate, terminal_roe=discount_rate)` — the
  scenario's OWN discount rate doubles as its `terminal_roe`, so each
  scenario's terminal phase fades to ITS OWN cost of equity, not one shared
  rate across scenarios.
- **Band**, per scenario, from `_cyclical_fcfe_scenario_band`: recompute at
  `discount_rate +/- sensitivity._DISCOUNT_RATE_STEP` (re-passing that SAME
  nearby rate as `terminal_roe` each time, so the fade convention travels
  with the sensitivity band too), `growth_5y`/`terminal_growth` held fixed;
  a rate that doesn't clear `> terminal_growth`, or a failed call, is
  excluded (not clamped); falls back to the flat `_band(per_share)` (+/-10%)
  when fewer than `_MIN_GRID_CELLS_FOR_BAND` points are usable, with the
  standard fallback Turkish note.
- `reinvestment_base = round(min(base_growth_5y, roe) / roe, 4)` — the base
  scenario's own implied reinvestment rate, for display only.
- Returns `(None, notes)` if no scenario computed a `per_share`; else
  `({"scenarios": {...}, "per_share": scenarios["base"]["per_share"],
  "normalized_net_income": ni_norm, "roe": round(roe, 4), "equity": equity,
  "cost_of_equity": earnings_power["cost_of_equity"], "reinvestment_base":
  ...}, notes)`. Never raises.

### Engine integration (`run_valuation`) — headline hierarchy for `cyclical`

`earnings_power` (Sec.8a) is built whenever `sector_type in ("mature",
"cyclical")` and hyper-grower mode is off — for `cyclical` it doubles as
both the zero-growth floor AND the earnings base this anchor grows.
Hyper-grower mode (Sec.3/Sec.11) still takes precedence over everything
below, on the same `if hyper_growth_active: ... elif sector_type ==
"cyclical": ...` chain Sec.8 already uses:

```python
elif sector_type == "cyclical":
    unreliable, quality_note = (False, None)
    if earnings_power is not None:
        unreliable, quality_note = _fcf_dcf_unreliable(
            dcf_scenarios, earnings_power, normalized, metrics
        )
        if quality_note:
            notes.append(quality_note)
    if unreliable and earnings_power is not None:
        cyclical_fcfe_detail, cf_notes = _build_cyclical_fcfe(
            assumptions, earnings_power, normalized, metrics, shares, dilution_rate
        )
        cf_beats_floor = (  # cf_base_ps / epv_base_ps read from each detail's "base"
            cyclical_fcfe_detail is not None and _is_number(cf_base_ps)
            and (not _is_number(epv_base_ps) or cf_base_ps >= epv_base_ps)
        )
        if cf_beats_floor:
            primary_dcf_scenarios = cyclical_fcfe_detail["scenarios"]
            cyclical_fcfe_headline = True
        else:
            primary_dcf_scenarios = earnings_power["scenarios"]
            epv_headline = True
    elif normalized_variant is not None:
        primary_dcf_scenarios = normalized_variant   # unchanged existing behavior
    # else: falls back to the raw dcf.scenarios band, exactly as before
```

The gate, `_fcf_dcf_unreliable` (Sec.8a), is REUSED byte-for-byte from the
mature path — the same three conditions (FCF suppressed vs. the EPV base,
cash-backed, investment-driven) decide whether a cyclical's raw FCF-DCF is
unreliable enough to switch off. This is deliberate: a cyclical's low FCF
can ALSO be a genuine earnings-quality problem rather than growth-CapEx
suppression, and the gate's cash-conversion guard protects against masking
that behind a reassuring earnings-based number here, exactly as it does for
Amazon-shaped mature filers.

**Priority within `cyclical`, most to least specific:**

1. `unreliable` fires (growth-CapEx-suppressed FCF, cash-backed,
   investment-driven) AND the FCFE anchor's base beats the EPV floor (the
   **`cf_base_ps >= epv_base_ps` guardrail**, mirroring Sec.8b/8d's own
   EPV-floor guardrails) → **`cyclical_fcfe_headline = True`**, headline =
   the sustainable-growth FCFE band. Turkish note: `"Döngüsel + sermaye-
   yoğun: serbest nakit akışı büyüme yatırımıyla (yüksek CapEx) bastırıldığı
   için manşet, döngü-ortası normalize kazanca sürdürülebilir-büyüme
   (reinvestment=g/ROE) uygulayan bir FCFE çapasına dayandırıldı. Sıfır-
   büyüme EPV tabanı, döngü-ortası FCF-DCF ve ham FCF-DCF ikincil olarak
   raporlanır."`
2. `unreliable` fires but the FCFE anchor couldn't be built, or its base
   falls below the EPV floor → **`epv_headline = True`**, headline = the
   zero-growth EPV floor (Sec.8a) — still strictly better than publishing
   the capex-suppressed raw FCF-DCF. Turkish note: `"Döngüsel + sermaye-
   yoğun: serbest nakit akışı büyüme yatırımı nedeniyle kazanç gücünü
   yansıtmıyor; manşet sıfır-büyüme kazanç-gücü (EPV) çapasına
   dayandırıldı. Döngü-ortası ve ham FCF-DCF ikincil olarak raporlanır."`
3. `unreliable` never fires (the ordinary near-trough cyclical, not
   growth-CapEx suppressed) AND `normalized_variant` was computed → the
   EXISTING cycle-mid normalized FCF-DCF headline, byte-for-byte unchanged
   from before this section existed.
4. Neither → falls back to the raw FCF-DCF band, exactly as before.

`scenario_meta` (Sec.11's `fair_value_range`): when `cyclical_fcfe_headline`,
per-scenario `growth`/`discount_rate`/`note` come from
`_cyclical_fcfe_scenario_meta(cyclical_fcfe_detail, assumptions)` (mirrors
`_epv_scenario_meta`, Sec.8a) — `growth` reads e.g. `"%6.7 büyüme (kazanç +
sürdürülebilir büyüme)"`, and `note` names the scenario's implied
reinvestment rate and ROE, e.g. `"Sürdürülebilir-büyüme FCFE çapası (base):
normalize net kâr büyütülür, büyümeyi fonlamak için kârın ~%43'ü (g/ROE, ROE
%16) reinvest edilir, kalanı iskonto edilir."`

### Exception to Sec.9's "same cash-flow base" invariant + corrected disclosure

Same documented, intentional exception as Sec.8a/8b: whenever
`cyclical_fcfe_headline` is `True`, the `sensitivity` grid and
`reverse_dcf.implied_growth` do NOT describe the FCFE headline. Specifically
for `cyclical`:

- The `sensitivity` grid reflects `normalized_fcf0` (Sec.8's cycle-mid
  normalized FCF-DCF base) whenever that variant was computable — the
  engine's `headline_fcf0` selector is unconditional on `sector_type ==
  "cyclical" and normalized_variant is not None`, regardless of which
  headline above actually won.
- `reverse_dcf.implied_growth` always solves over the raw, suppressed
  `fcf0` — it never switches to `normalized_fcf0`.
- Both differ from the FCFE headline itself, and from EACH OTHER. The
  Turkish note makes this precise (correcting an earlier draft that wrongly
  described both legs as reflecting the "same suppressed base"):

```
"Duyarlılık tablosu döngü-ortası normalize FCF-DCF tabanını, ters-DCF ise ham
(baskılanmış) FCF tabanını yansıtır; ikisi de manşet FCFE çapasından
farklıdır ve serbest nakit akışının neden düşük olduğunu gösteren kanıt
olarak korunur."
```

The same corrected wording is used for the cyclical `epv_headline` case
(priority case 2 above); the mature-sector EPV note (Sec.8a, which has no
`normalized_variant` concept) is unaffected and stays as written there.

### Structural re-rating / trough-excluded assumption (explicit, transparency note)

This anchor's earnings base is EPV's `normalized_net_income` — recent,
representative (profitable) years, with the margin-median sanity guard
(Sec.8a) as its only outlier control. It does **not** average in a severe
cyclical trough (e.g. a memory-glut loss year) as a recurring feature of the
cycle; it treats such a trough as a non-recurring exception to a
structurally re-rated, now-more-profitable business. This is a deliberate
modeling choice, not an oversight, and whenever `cyclical_fcfe_headline` (or
the cyclical `epv_headline`) fires, the engine appends a note stating the
assumption plus the conservative alternative it did NOT compute a full
second valuation for:

```
"NOT: Bu çapa, kazanç tabanını son temsili (kârlı) yıllardan alır ve şiddetli
döngü diplerini (ör. bir bellek-glut zarar yılı) tekrar etmeyecek istisna
olarak DIŞLAR (yapısal re-rating varsayımı). Dipleri döngünün kalıcı parçası
sayan tam-döngü ortalaması, değeri belirgin biçimde düşürür."
```

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

`run_valuation` passes `cyclical_fcfe_headline=cyclical_fcfe_headline` into
`triangulate.triangulate(...)`. Same `CONFIDENCE_HIGH → CONFIDENCE_MEDIUM`
cap as the other headline-override flags (Sec.8a/8b/8d), same underlying
reason (the DCF leg IS the earnings-based anchor here, so three-way
"agreement" with multiples isn't independent confirmation). Mutually
exclusive with the other headline flags in practice.

### Output shape additions (Sec.11)

```python
"cyclical_fcfe_headline": bool,   # True only when sector_type == "cyclical",
                                   # the FCF-DCF reliability gate fired, AND
                                   # the FCFE anchor's base beat the EPV
                                   # floor (priority case 1 above).
"cyclical_fcfe_detail": None | {
   "scenarios": {"bear"/"base"/"bull": {"per_share", "lo", "hi"}},
   "per_share": float,             # base scenario's point estimate
   "normalized_net_income": float, # == earnings_power["normalized_net_income"]
   "roe": float,                   # normalized_net_income / spot latest-FY equity
   "equity": float,                # the spot latest-FY StockholdersEquity used
   "cost_of_equity": float,        # == earnings_power["cost_of_equity"]
   "reinvestment_base": float,     # base scenario's implied g/ROE, display only
   "growth_vs_floor": "adds"|"destroys"|None,  # WP7; vs the EPV base per-share
},                                  # built (attempted) whenever sector_type ==
                                   # "cyclical", not hyper, and the FCF-DCF
                                   # reliability gate fired -- REGARDLESS of
                                   # whether it beat the EPV floor (may be
                                   # non-None but NOT the headline, priority
                                   # case 2 above).
```

`cli._fair_value_method_label` (Sec.13) checks `cyclical_fcfe_headline`
FIRST (ahead of `earnings_power_headline`) and returns `"FCFE
(kazanç+büyüme)"`.

### Design notes

- The reinvestment rate is `b = min(g, roe) / roe` for BOTH the 1-10 year
  path (against the current `roe`) and the terminal year (against
  `terminal_roe`, defaulting to `roe`) — there is no separate flat
  reinvestment ceiling (e.g. a fixed 90%); an earlier draft of this anchor
  used one and it was removed in favor of the growth-capped-at-ROE
  construction above, which is both simpler and structurally never produces
  a negative payout.
- The ROE denominator is the SPOT latest-FY `StockholdersEquity`, not a
  through-cycle average. An average-equity alternative was proposed during
  finance review specifically to match the normalized (cycle-adjusted)
  numerator, but was **not adopted** in the shipped code — do not assume
  that proposal landed; this section documents the code's actual (spot-
  equity) behavior.

### Scope

Purely additive: does not change `dcf.scenarios`, `dcf.normalized_variant`,
`earnings_power`, `pb_roe`, `sensitivity`, or any other existing output
key's meaning; applies only to `cyclical` filers whose `_fcf_dcf_unreliable`
gate fires, and never to `financial`/`reit`/`growth_unprofitable`/`mature`
or to any filer already in hyper-grower mode. A cyclical filer whose FCF
suppression is the ordinary near-trough case (gate does not fire) keeps its
EXISTING Sec.8 `normalized_variant` headline behavior unchanged.

## 8f. Residual income model (RIM) — financial-sector anchor — `dcf.rim_per_share` / `engine._build_rim`

Replaces Sec.8's single-period justified-P/B heuristic
(`_justified_pb`, `(ROE - g) / (r - g)` applied as ONE static multiple to
today's book value forever) as the PRIMARY anchor for `sector_type ==
"financial"` with a genuine multi-period residual-income model: a 10-year,
two-stage fade of growth/discount-rate assumptions (the same fade curve
`dcf_per_share`/`fcfe_sustainable_growth_per_share` already use), rather
than one perpetuity-style multiple applied to a single point-in-time ROE/g/r
triple. `sector_type == "reit"` is UNAFFECTED — it keeps its own FFO/Gordon-
growth anchor (Sec.8c), with `pb_roe` as ITS fallback exactly as before,
since GAAP real-estate depreciation makes book value/net income unreliable
inputs for a bank-shaped residual-income compounding too.

### `dcf.rim_per_share(bve0, ni0, roe, growth_5y, terminal_growth, discount_rate, shares, dilution_rate=0.0, terminal_roe=None) -> dict`

- Raises `ValueError` — never silently "fixes" an invalid input — when
  `ni0 is None`, `bve0 is None` or `<= 0`, `shares` is falsy/`<= 0`, `roe <=
  0`, or `discount_rate <= terminal_growth` (Gordon-growth terminal value
  undefined), mirroring `dcf_per_share`/`fcfe_sustainable_growth_per_share`'s
  raise-don't-fix discipline (Sec.4/8e).
- Ohlson-style residual income: intrinsic equity value = book value today
  PLUS the present value of all future "excess" earnings the firm generates
  above what its cost of equity requires on the book value it starts each
  period with — `RI_t = NI_t - discount_rate * BVE_{t-1}`.
- Projects earnings along the SAME 10-year, two-stage path `project_fcf`
  uses (years 1-5 at `growth_5y`, years 6-10 fading linearly to
  `terminal_growth` via `_year_growth_rate`, Sec.4), reusing
  `fcfe_sustainable_growth_per_share`'s exact `g_eff = min(g_year, roe)`
  reinvestment-cap idiom (Sec.8e) rather than re-deriving it:
  `ni_year = previous_ni * (1 + g_eff)`; reinvestment rate `b = g_eff / roe`;
  `retained = ni_year * b`; `bve_year = previous_bve + retained` (clean-
  surplus roll-forward — book value grows by the RETAINED portion of
  earnings only, never the full `ni_year`, which would silently assume 100%
  retention regardless of `g_eff`); `ri_year = ni_year - discount_rate *
  previous_bve`, discounted at `(1 + discount_rate) ** year`.
  - **Front-loading note (F4 — known modeling choice, minor):** the path
    grows net income immediately (`ni_1 = ni0 * (1 + g_eff)`) while the
    year-1 equity charge is on the ungrown opening book (`r * bve0`), so the
    IMPLIED year-1 ROE is `roe * (1 + g)` rather than the input `roe`, and
    the whole path runs slightly above the stated ROE — front-loading
    residual income a few percent. This is a deliberate consequence of
    mirroring the FCFE sibling's `ni0`-compounding path; the Ohlson identity
    still holds for the forecast chosen, so it is documented, not "fixed". A
    strictly ROE-consistent variant would set `ni_t = roe * bve_{t-1}` (no
    year-1 jump) — not adopted, to keep the two anchors' earnings paths
    identical in construction.
- **Terminal ROE fades to the cost of equity (F1 — genuine fade).** The
  terminal (perpetuity) net income is what the ending book value earns at the
  faded terminal ROE, so terminal residual income is the excess of that over
  the cost-of-equity charge on the same book:
  `terminal_roe_resolved = terminal_roe if terminal_roe is not None else roe`;
  `ri_terminal = (terminal_roe_resolved - discount_rate) * bve_10`;
  `tv = ri_terminal / (discount_rate - terminal_growth)` (Gordon denominator
  keeps the original `terminal_growth`). The engine passes `terminal_roe =
  discount_rate`, so `ri_terminal == 0` and `tv == 0`: intrinsic equity is
  `bve0 + PV(RI_1..10)` with NO terminal excess return — the standard RIM
  "excess returns compete away in steady state" terminal (Damodaran/Penman).
  **This is NOT the same mechanism as Sec.8e's FCFE fade** (there, terminal
  reinvestment earns exactly the cost of equity and the terminal DIVIDEND
  perpetuity stays positive; here the book value already captures the normal
  return, so a faded terminal adds nothing beyond it — the RIM analog of the
  same economics). A `terminal_roe` ABOVE the cost of equity (durable
  franchise) yields a positive, `terminal_growth`-growing terminal residual.
  **Regression note:** the earlier `g_t_eff = min(terminal_growth,
  terminal_roe_resolved)` / `ni_terminal = ni_10 * (1 + g_t_eff)`
  construction left `terminal_roe` INERT (the `min` always collapsed to
  `terminal_growth` under the `r > terminal_growth` guard), silently booking
  a permanent excess-return perpetuity — the current form is the fix.
- No net-debt bridge: book value/net income are already equity-level
  figures, so `discount_rate` must be a levered cost of equity, exactly like
  Sec.4/8e. `effective_shares = shares * (1 + dilution_rate) ** 5` (same
  year-5 dilution-horizon convention).
- `equity = bve0 + sum(discounted ri_1..10) + pv(tv)`; `per_share = equity /
  effective_shares`.
- Returns `{"per_share", "equity", "bve0", "ri_path" (10 floats), "bve_path"
  (10 floats), "tv", "effective_shares"}`. Nothing rounded here — rounding is
  the caller's (`engine.py`'s) responsibility.

### `engine._build_rim(assumptions, normalized, metrics, ratios) -> tuple[Optional[dict], list[str]]`

- FY selection mirrors `_build_pb_roe` (Sec.8) byte-for-byte in spirit:
  walks the `StockholdersEquity` series from the newest fiscal year down and
  picks the first one that ALSO has both a `NetIncome` figure and a `roe`
  figure (from `ratios`), independent of `metrics["latest_fy"]` (guards the
  same JPM-shaped edge case Sec.8 documents). Missing shares, or no fiscal
  year with all three → `(None, [Turkish note])`. `roe <= 0` → `(None,
  [Turkish note])` (a non-positive ROE makes the reinvestment identity
  meaningless, mirroring `_build_pb_roe`'s `fair_pb_base <= 0` guard).
- Per scenario (bear/base/bull): reads that scenario's own
  `growth_5y`/`terminal_growth`/`discount_rate`; any non-numeric value or
  `discount_rate <= terminal_growth` → that scenario's cell is
  `{"per_share": None, "lo": None, "hi": None}` plus a Turkish note (mirrors
  `_build_dcf_scenarios`/`_build_cyclical_fcfe`). Otherwise calls
  `rim_per_share(bve0, ni0, roe, growth_5y, terminal_growth, discount_rate,
  shares, terminal_roe=discount_rate)` — the scenario's OWN discount rate
  doubles as its `terminal_roe`, same convention as Sec.8e.
- **Band**, per scenario, from `_rim_scenario_band`: recompute at
  `discount_rate +/- sensitivity._DISCOUNT_RATE_STEP` (re-passing that same
  nearby rate as `terminal_roe`), `growth_5y`/`terminal_growth` held fixed; a
  rate that doesn't clear `> terminal_growth`, or a failed call, is excluded
  (not clamped); falls back to the flat `_band(per_share)` (+/-10%) when
  fewer than `_MIN_GRID_CELLS_FOR_BAND` points are usable, with the standard
  fallback Turkish note.
- Returns `(None, notes)` if no scenario computed a `per_share`; else
  `({"scenarios": {...}, "per_share": scenarios["base"]["per_share"],
  "bve0": bve0, "book_value_per_share": bve0 / shares,
  "normalized_net_income": ni0, "roe": round(roe, 4)}, notes)`. Never raises.

### Engine integration (`run_valuation`) — replaces `pb_roe` as the primary `financial` anchor

```python
pb_roe = None
rim = None
ffo = None
if sector_type == "financial":
    rim, rim_notes = _build_rim(assumptions, normalized, metrics, ratios)
    notes.extend(rim_notes)
    if rim is None:
        pb_roe, pb_notes = _build_pb_roe(assumptions, normalized, metrics, ratios)
        notes.extend(pb_notes)
        notes.append(
            "Finansal sektörde RIM (kazanç-gücü/özkaynak bileşik modeli) hesaplanamadı; "
            "manşet/üçgenleme çapası olarak P/B x ROE'ye geri dönüldü."
        )
elif sector_type == "reit":
    ...  # unchanged (Sec.8c)

if sector_type == "reit" and ffo is not None:
    reit_or_financial_anchor = ffo
elif sector_type == "financial" and rim is not None:
    reit_or_financial_anchor = rim
else:
    reit_or_financial_anchor = pb_roe
```

`reit_or_financial_anchor` is the SAME slot `_build_fair_value_range`,
`triangulate.triangulate`'s `dcf_base_band` parameter, and the CLI/report
method label already read for `financial`/`reit` (Sec.8/8c/10/11/13) — RIM
is a drop-in replacement for `pb_roe` in that slot, not a new triangulation
leg. **This required zero changes to `triangulate.py`**: `_dcf_signal`
already treats whatever dict is passed as `dcf_base_band` generically (its
own docstring says "DCF (or P/B x ROE, if that's what's passed)"), so RIM's
identical `{"scenarios": {...}}` shape flows through unchanged.

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

**N/A — no new parameter, no cap.** Unlike Sec.8a/8b/8d/8e's headline-
override flags (which compete with an ALREADY-COMPUTED alternative anchor
and therefore need a confidence cap to flag that the DCF leg isn't
independent), RIM occupies the exact same `reit_or_financial_anchor` slot
`pb_roe` already occupied for `financial` — there is no competing anchor to
flag non-independence against, so `triangulate.triangulate`'s existing
signature and confidence logic are entirely unchanged by this section.

### Output shape additions (Sec.11)

```python
"rim": None | {
   "scenarios": {"bear"/"base"/"bull": {"per_share", "lo", "hi"}},
   "per_share": float,              # base scenario's point estimate
   "bve0": float,                   # the selected fiscal year's StockholdersEquity
   "book_value_per_share": float,   # bve0 / metrics["shares"]
   "normalized_net_income": float,  # the selected fiscal year's NetIncome (ni0)
   "roe": float,                    # the selected fiscal year's ROE (from ratios)
},                                   # built (attempted) only when
                                    # sector_type == "financial"; None for
                                    # every other sector, and None when the
                                    # build itself fails (falls back to pb_roe).
```

`_empty_valuation` (the crash-safety shape, Sec.11) also gains `"rim":
None`. `cli._valuation_method_label` (Sec.13) checks `ffo` first (unchanged
REIT precedence), then a populated `rim` (`"scenarios" in rim`) → returns
`"RIM"`, THEN falls through to `"P/B×ROE"` — so a `financial` filer whose
RIM build failed still gets an accurate label. `report/template.html`'s
`triangulationRowHtml` mirrors the same FFO → RIM → P/B×ROE chain for its
`dcfLabel`.

### Scope

Purely additive/substitutive within `financial`: does not change `dcf`,
`earnings_power`, `ffo`, `sensitivity`, `multiples`, or any other existing
output key's meaning; does not touch `reit` at all (FFO/Gordon-growth stays
`reit`'s primary anchor, `pb_roe` stays ITS fallback, unchanged); does not
touch `cyclical`/`growth_unprofitable`/`mature`/hyper-grower filers.
`_build_pb_roe`/`_justified_pb` themselves are UNCHANGED code — they remain
exactly as documented in Sec.8, now used only as `financial`'s fallback
(when RIM can't be built) rather than its primary anchor.

## 8g. Altman Z-score (distress screen) — `distress.altman_z_score` / `engine._build_altman_z`

An ADVISORY-ONLY bankruptcy-risk overlay (new `sec_analyzer/valuation/
distress.py` module — also the future home of Beneish M-score, Sec.8j, and
Merton distance-to-default, Sec.8k). Unlike every anchor in Sec.8/8a-8f, this
section adds NO new fair-value candidate: it never headlines
`fair_value_range`, never feeds `primary_dcf_scenarios`, and never
participates in `triangulate.triangulate`'s confidence vote. It exists
purely to surface a classic bankruptcy-risk signal alongside whatever
valuation anchor is already in use.

### `distress.altman_z_score(working_capital, total_assets, retained_earnings, ebit, market_cap, total_liabilities, revenue) -> Optional[dict]`

- Classic Altman (1968) formula: `Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 +
  1.0*X5`, where `X1 = working_capital / total_assets`, `X2 =
  retained_earnings / total_assets`, `X3 = ebit / total_assets`, `X4 =
  market_cap / total_liabilities`, `X5 = revenue / total_assets`.
- Zone: `Z > 2.99` → `"safe"`; `1.81 <= Z <= 2.99` → `"grey"`; `Z < 1.81` →
  `"distress"`.
- Returns `None` (never raises, never fabricates) when any input is `None`,
  or when `total_assets`/`total_liabilities` is non-positive (both the model
  and its ratios are undefined in that case).
- Returns `{"z_score": float (2dp), "zone": "safe"|"grey"|"distress",
  "components": {"x1", "x2", "x3", "x4", "x5"} (each 4dp)}` on success.

### `engine._build_altman_z(normalized, metrics) -> tuple[Optional[dict], list[str]]`

- Resolves the fiscal year via `resolve_fundamental_fy(metrics)`; `None` →
  `(None, [])`.
- `working_capital = CurrentAssets_fy - CurrentLiabilities_fy`; either
  missing → `(None, [Turkish note])`.
- Reads `TotalAssets`/`TotalLiabilities`/`RetainedEarningsAccumulatedDeficit`
  (new WP8 concept)/`OperatingIncome` (EBIT proxy)/`Revenue` for the same
  fiscal year, and `metrics["market_cap"]`, then delegates to
  `distress.altman_z_score`. A `None` result there (missing input, or
  non-positive total assets/liabilities) → `(None, [Turkish note])`.
- On success, appends ONE Turkish note naming the zone and Z-score (e.g.
  `"Altman Z-skoru gri bölgede (belirsiz iflas riski -- izlenmeli). (Z=1.95)"`)
  and returns `(result, notes)`. Never raises.

### Engine integration (`run_valuation`) — sector gate, no chain participation

```python
altman_z = None
if sector_type not in _SECTORS_WITHOUT_FCF_DCF:  # excludes financial/reit
    altman_z, altman_notes = _build_altman_z(normalized, metrics)
    notes.extend(altman_notes)
```

Called ONCE, near the end of `_run_valuation`, entirely independent of the
DCF/EPV/revenue-first/RIM/FFO priority chain above it — nothing about this
call can change `primary_dcf_scenarios`, `fair_value_range`, or any
headline flag. Not computed at all for `financial`/`reit` (same
`_SECTORS_WITHOUT_FCF_DCF` gate the FCF-DCF disablement uses): the classic
Altman model is calibrated on industrial/manufacturing balance sheets, and
neither a bank's structural leverage nor a REIT's GAAP real-estate
depreciation make its ratios meaningful for those sectors.

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

**N/A — no parameter, no interaction.** `altman_z` is computed entirely
outside `triangulate.triangulate`'s call and is never passed into it; the
Z-score cannot raise, lower, or cap `triangulate.confidence` in any way.

### Output shape additions (Sec.11)

```python
"altman_z": None | {
   "z_score": float, "zone": "safe"|"grey"|"distress",
   "components": {"x1", "x2", "x3", "x4", "x5"},
},  # None for financial/reit (never attempted), and None for every other
    # sector when the underlying balance-sheet data is incomplete.
```

`_empty_valuation` (Sec.11) also gains `"altman_z": None`.

### CLI / HTML / script-provider wiring — the shared advisory-card pattern

Because Beneish M-score (Sec.8j) and Merton distance-to-default (Sec.8k)
are the same shape of ADVISORY-ONLY screen, this section establishes ONE
shared presentation pattern all three reuse (avoiding three near-duplicate
wiring paths):

- `cli._distress_flags_line(valuation)`: a single "Risk skoru:" card line
  reading `valuation["altman_z"]`/`["beneish_m"]`/`["merton_dtd"]`, rendering
  whichever are present (`None` when none are), printed after
  `_sensitivity_line` in `_print_verdict_card`.
- `report/template.html`'s `distressFlagsCardHtml(valuation)`: one card
  ("Risk Taramaları"), same three-key read, rendered in `renderBody` right
  before the red-flags card; returns `""` (renders nothing) when no screen
  produced a result.
- `interpret/rule_based.py`'s `_distress_risk_from_valuation(valuation)`:
  folds a risk sentence into the script provider's `key_risks` list ONLY
  when a screen is actually flagging risk (Altman zone `"grey"`/`"distress"`,
  a Beneish manipulation flag) — a `"safe"` Z-score consumes none of
  `key_risks`'s 5 slots.
- LLM provider: sees `valuation.altman_z` automatically (full-dict
  passthrough, Sec.12); `_PHASE2_OUTPUT_CONTRACT` (Sec.12) instructs it to
  fold a `"grey"`/`"distress"` zone into `key_risks` the same way, and to
  never treat it as a valuation input.
- `VALUATION.md` §11 documents the "advisory only, never a fair-value input"
  rule for both the LLM system prompt and human readers.

### Scope

Purely additive: does not change `dcf`, `earnings_power`, `rim`, `ffo`,
`fair_value_range`, `sensitivity`, `multiples`, `triangulation`, or any
other existing output key's meaning or value. Computed for every sector
EXCEPT `financial`/`reit`. A filer missing the underlying balance-sheet
concepts degrades to `altman_z: None` plus an explanatory note, exactly
like every other engine anchor's missing-data behavior.

## 8h. LBO-implied floor value — `lbo.lbo_implied_floor_per_share` / `engine._build_lbo_floor`

An ADVISORY-ONLY, private-equity-return-based value floor (new
`sec_analyzer/valuation/lbo.py` module). Standard LBO "deleveraging return"
logic: a financial (private-equity) buyer doesn't need organic growth OR
multiple expansion to earn a return — paying down acquisition debt out of
the company's own free cash flow over a hold period mechanically grows the
buyer's equity stake even with EBITDA and the exit multiple both held FLAT
(deliberately conservative). Discounting the resulting exit equity value
back to today at the sponsor's hurdle IRR gives the highest price a
disciplined financial buyer could justify paying today — a value floor,
distinct from (and never a substitute for) the public-market DCF anchor.
Like Sec.8g, this adds NO new fair-value candidate: never headlines
`fair_value_range`, never feeds `primary_dcf_scenarios`, never
participates in `triangulate.triangulate`'s confidence vote.

### `lbo.lbo_implied_floor_per_share(ebitda, entry_multiple, existing_debt, exit_multiple, fcf0, fcf_growth, shares, target_irr=0.20, hold_years=5) -> Optional[dict]`

- `entry_ev = entry_multiple * ebitda`; `entry_equity = entry_ev -
  existing_debt` (today's implied sponsor equity check at `entry_multiple`
  — surfaced for display, not itself the floor).
- FCF is projected `hold_years` forward via `dcf.project_fcf(fcf0,
  fcf_growth, fcf_growth, years=hold_years)` — passing the SAME rate as both
  `growth_5y` and `terminal_growth` keeps the path flat with no fade, since
  `hold_years <= 5` (the default) never reaches `project_fcf`'s fade phase
  (Sec.4).
- Debt paydown is a 100% cash sweep: each year's entire projected FCF
  reduces the outstanding balance, floored at `0.0`.
- `exit_ev = exit_multiple * ebitda` — EBITDA held FLAT over the hold period
  (no organic-growth credit; this is what makes the result a floor, not a
  base-case LBO return). `exit_equity = exit_ev - remaining_debt`.
- `floor_equity_today = exit_equity / (1 + target_irr) ** hold_years`;
  `per_share = floor_equity_today / shares`.
- Returns `None` (never raises, never fabricates) when `ebitda` is
  missing/`<= 0`, `existing_debt` is missing/`< 0`, `entry_multiple`/
  `exit_multiple` is missing/`<= 0`, `fcf0` is missing, `shares` is
  falsy/`<= 0`, `target_irr <= -1`, or `hold_years <= 0`.
- Returns `{"per_share", "entry_ev", "entry_equity", "exit_ev",
  "exit_equity", "remaining_debt", "fcf_path" (hold_years floats),
  "debt_path" (hold_years floats), "floor_equity_today"}` on success.
  Nothing rounded — rounding is the caller's responsibility.

### `engine._build_lbo_floor(metrics, fcf0) -> tuple[Optional[dict], list[str]]`

- `entry_multiple`/`exit_multiple` are BOTH `metrics["ev_ebitda"]` (the
  filer's OWN current EV/EBITDA — "could a sponsor justify TODAY's price",
  no multiple-expansion view of its own). `existing_debt =
  metrics["total_debt"]`. `fcf0` is the SAME base-year FCF the standard
  DCF uses (Sec.4's already-selected `fcf0`, passed through from
  `_run_valuation` — not a separately re-derived figure).
- **F3: FCF is held FLAT (`fcf_growth=0.0`).** A genuine conservative floor
  credits NO organic growth ANYWHERE — EBITDA flat, exit multiple flat, and
  the debt-paydown FCF stream flat too. The earlier version passed the base
  scenario's `growth_5y` as `fcf_growth` while pinning EBITDA flat, which is
  internally incoherent: for a high-`growth_5y` filer the projected FCF
  could exceed EBITDA (economically impossible), over-sweeping the debt and
  inflating the "floor" past any conservative reading. Consequently
  `_build_lbo_floor` **no longer takes/reads `assumptions` at all** (it
  dropped the `assumptions` parameter and the earlier "base growth missing"
  guard).
- Any missing/degenerate input degrades via `lbo_implied_floor_per_share`'s
  own guards → `(None, [Turkish note])`.
- On success, appends ONE Turkish note explicitly labeled "bilgi amaçlı,
  manşete GİRMEZ" (informational, does NOT enter the headline) naming the
  target IRR and the resulting per-share floor. Never raises.
- **Cyclical spot-EBITDA caveat (known limitation):** entry/exit EBITDA is
  the filer's SPOT current EBITDA. For a cyclical filer at a cycle peak,
  that spot EBITDA (× the spot EV/EBITDA multiple) can be well above
  mid-cycle earning power, so the "floor" for such a name is really "floor
  conditional on today's cycle position," not a through-cycle floor. This is
  advisory-only and clearly labeled, so it is documented rather than
  normalized here (normalizing EBITDA for the LBO path would duplicate the
  cyclical-normalization machinery in Sec.3/8e for a non-headline number).

### Engine integration (`run_valuation`) — sector gate, no chain participation

```python
lbo_floor_detail = None
if sector_type not in _SECTORS_WITHOUT_FCF_DCF:  # excludes financial/reit
    lbo_floor_detail, lbo_notes = _build_lbo_floor(metrics, fcf0)
    notes.extend(lbo_notes)
```

Called once, near the end of `_run_valuation`, alongside (and independent
of) `altman_z` (Sec.8g) — entirely outside the DCF/EPV/revenue-first/RIM/
FFO priority chain. Not computed for `financial`/`reit` (same
`_SECTORS_WITHOUT_FCF_DCF` gate as Sec.8g): EV/EBITDA-based leverage
doesn't describe a bank's regulated capital structure or a REIT's
FFO-centric one.

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

**N/A — no parameter, no interaction**, identical to Sec.8g's Altman Z.

### Output shape additions (Sec.11)

```python
"lbo_floor_detail": None | {
   "per_share": float, "entry_ev": float, "entry_equity": float,
   "exit_ev": float, "exit_equity": float, "remaining_debt": float,
   "fcf_path": list[float], "debt_path": list[float],
   "floor_equity_today": float,
},  # None for financial/reit (never attempted), and None for every other
    # sector when EBITDA/EV-EBITDA/debt/fcf0/shares is missing.
```

`_empty_valuation` (Sec.11) also gains `"lbo_floor_detail": None`.

### CLI / HTML / script-provider wiring

- `cli._lbo_floor_line(valuation)`: a single "LBO çapası:" card line,
  printed after `_distress_flags_line` in `_print_verdict_card`; `None`
  (renders nothing) when `lbo_floor_detail` is absent.
- `report/template.html`'s `lboFloorCardHtml(valuation)`: one card ("LBO
  Çapası"), rendered in `renderBody` right after the distress-screens card;
  returns `""` when `lbo_floor_detail` is absent.
- Not folded into the script provider's `key_risks`/`cyclical_risk` (unlike
  Sec.8g's distress screens) — this is a floor/opportunity signal, not a
  risk, so it doesn't belong in a risk list; it is surfaced only via the
  CLI/HTML lines above and via `valuation["notes"]`.

### Scope

Purely additive: does not change `dcf`, `earnings_power`, `rim`, `ffo`,
`altman_z`, `fair_value_range`, `sensitivity`, `multiples`,
`triangulation`, or any other existing output key's meaning or value.
Computed for every sector EXCEPT `financial`/`reit`. A filer missing the
underlying EBITDA/leverage/FCF data degrades to `lbo_floor_detail: None`
plus an explanatory note, exactly like every other engine anchor's
missing-data behavior.

## 8i. Precedent-transaction (M&A comps) multiples — `precedent_transactions.load_precedent_transactions` / `.find_industry_medians`

Purely additive reference-data input to the EXISTING multiples leg
(Sec.6/Sec.7) — new `sec_analyzer/valuation/precedent_transactions.py`
module, mirroring `damodaran.py`'s architecture exactly (optional, local,
no network access, tolerant of a missing directory/file). Deal comps aren't
available from any of this project's data sources (SEC EDGAR, Damodaran,
yfinance, FRED), so — exactly like `data/damodaran/*.csv` — this is data an
analyst curates by hand, NOT a new software dependency. **Not a new
triangulation vote**: `triangulate.triangulate`'s hardcoded 3-way signal
count (`dcf`/`reverse_dcf`/`multiples`) is untouched; this section only
fills in a value that the EXISTING `sector_ratio` parameter already
threads through.

### `precedent_transactions.load_precedent_transactions(dir_path) -> Optional[list[dict]]`

- Reads `dir_path/deals.csv` (header row: `industry, ev_ebitda, ev_revenue,
  control_premium[, deal_count]`), reusing `damodaran._read_csv_rows`/
  `damodaran._to_float` (not re-implemented). Rows without a usable
  `industry` value are skipped; other columns individually degrade to
  `None` on missing/malformed data rather than dropping the row.
- Returns `None` when `dir_path` doesn't exist, the file is missing/
  unreadable, or no row has a usable industry name. Never raises.

### `precedent_transactions.find_industry_medians(deals, industry) -> Optional[dict]`

- **Deliberately NOT a second fuzzy SIC matcher.** `industry` is expected
  to be whatever name `damodaran.sector_medians` ALREADY resolved for the
  filer (`sector_medians_result["industry"]`) — this function just needs
  to find that SAME name in the precedent-transaction rows via
  `damodaran._normalize_text`-normalized exact match, not re-derive a
  second SIC-to-industry mapping. One taxonomy, one matcher, reused.
- Returns `None` when `deals`/`industry` is empty/`None`, or no row
  matches. Never raises.

### Engine integration (`run_valuation`) — fills the EV/EBITDA sector-median gap

```python
precedent_deals = precedent_transactions.load_precedent_transactions(
    precedent_transactions_dir if precedent_transactions_dir is not None else Config.PRECEDENT_TRANSACTIONS_DIR
)
precedent_medians = precedent_transactions.find_industry_medians(
    precedent_deals, (sector_medians_result or {}).get("industry")
)
```

Loaded once, right after `sector_medians_result`/`sector_capex_sales` are
resolved (Sec.8/WP6), reusing that SAME already-resolved industry name.
`run_valuation` gains a new optional keyword argument,
`precedent_transactions_dir` (default `None` → `Config.PRECEDENT_TRANSACTIONS_DIR`),
mirroring `damodaran_dir`'s existing pattern — backward compatible with
every existing caller.

**The actual integration point** is the multiples-comparison block's
`sector_info` dict (Sec.6/Sec.7's axis-b): Damodaran's own `multiples.csv`
carries NO EV/EBITDA sector median at all (a pre-existing gap — a leveraged
filer's axis-b comparison for its own PRIMARY multiple, FD/FAVÖK, was
previously always disabled, hardcoded `None`). This section fills that gap:

```python
sector_info["ev_ebitda_median"] = (precedent_medians or {}).get("ev_ebitda")
sector_info["precedent_transactions"] = precedent_medians  # full row, for display

# In the `leveraged` branch of `_ratio_candidates` (previously hardcoded None):
_ratio_candidates = (
    (ev_ebitda_pct, "FD/FAVÖK", current.get("ev_ebitda"), sector_info["ev_ebitda_median"]),
    ...  # P/E, P/S, P/FCF fallbacks unchanged
)
```

Since `sector_ratio` (fed into `triangulate.triangulate`, Sec.10) is
derived from whichever `_ratio_candidates` entry has a usable percentile
(the existing `_primary` selection logic, unchanged), populating
`ev_ebitda_median` from precedent-transaction data automatically makes
`sector_ratio` computable for leveraged filers where it previously stayed
`None` — **no `triangulate.py` code changes are needed**; this is purely a
data-availability fix flowing through the existing wiring.

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

**N/A — no new parameter.** This section adds no new argument to
`triangulate.triangulate`; it only supplies a value for the pre-existing
`sector_ratio` parameter in a case (`leveraged`) that previously always
passed `None` for it.

### Output shape additions (Sec.11)

```python
"multiples": {
  ...,
  "sector": {
    ...,  # available/industry/pe_median/ps_median/pfcf_median unchanged
    "ev_ebitda_median": float | None,  # NEW: from precedent-transaction data
    "precedent_transactions": None | {
       "industry": str, "ev_ebitda": float|None, "ev_revenue": float|None,
       "control_premium": float|None, "deal_count": float|None,
    },  # NEW: the full matched row, for display (control premium etc.)
    "comparison": {...},  # unchanged shape; now CAN populate for a
                           # leveraged filer's FD/FAVÖK when precedent data matches
  },
},
```

`_empty_valuation` (Sec.11) also gains `"ev_ebitda_median": None,
"precedent_transactions": None` inside its `sector` sub-dict.

### Config

New `Config.PRECEDENT_TRANSACTIONS_DIR = os.getenv("PRECEDENT_TRANSACTIONS_DIR",
os.path.join(os.getcwd(), "data", "precedent_transactions"))` — identical
idiom to `Config.DAMODARAN_DIR`.

### Scope

Purely additive: does not change `dcf`, `earnings_power`, `rim`, `ffo`,
`altman_z`, `lbo_floor_detail`, `fair_value_range`, `sensitivity`,
`triangulation`, or any other existing output key's meaning; does not add a
new triangulation vote or a new `triangulate.triangulate` parameter. A
filer/sector with no curated precedent-transaction data, or no directory at
all, degrades silently to today's exact behavior (`ev_ebitda_median: None`,
axis-b comparison unavailable for the leveraged branch, exactly as before
this section existed).

## 8j. Beneish M-score (earnings-manipulation screen) — `distress.beneish_m_score` / `engine._build_beneish_m`

An ADVISORY-ONLY earnings-manipulation screen (extends `distress.py`,
Sec.8g's module). Same non-chain-touching contract as Altman Z (Sec.8g):
never headlines `fair_value_range`, never feeds `primary_dcf_scenarios`,
never participates in `triangulate.triangulate`'s confidence vote. Unlike
Sec.8g/8h, this IS computed for every sector including `financial`/`reit` --
revenue/receivables/gross-margin manipulation signals are just as
meaningful there as anywhere else (only the leverage-ratio-based screens,
Sec.8g/8h, are sector-gated).

### `distress.beneish_m_score(current, prior) -> Optional[dict]`

- **Full 8-variable model** (Beneish 1999) when both fiscal years carry
  SG&A AND the leverage/accruals inputs (long-term debt, current
  liabilities, net income, operating cash flow): `M = -4.84 + 0.920*DSRI +
  0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA -
  0.327*LVGI`.
- **Degrades to the 5-variable model** (a real, SEPARATELY published
  Beneish variant with its OWN calibrated coefficients — NOT an ad hoc
  truncation of the 8-variable regression, which would be statistically
  invalid) when SG&A or the leverage/accruals inputs are missing for either
  year: `M = -6.065 + 0.823*DSRI + 0.906*GMI + 0.593*AQI + 0.717*SGI +
  0.107*DEPI`.
- Both models flag `M > -1.78` (`_BENEISH_MANIPULATION_THRESHOLD`) as a
  manipulation-likelihood signal (`"flag": bool`).
- Index formulas (`t`/`t-1` = current/prior fiscal year): `DSRI =
  (receivables_t/revenue_t) / (receivables_t-1/revenue_t-1)`; `GMI =
  gm_t-1/gm_t` where `gm = gross_profit/revenue`; `SGI = revenue_t /
  revenue_t-1`; `AQI = [1 - (current_assets_t+ppe_gross_t)/total_assets_t] /
  [1 - (current_assets_t-1+ppe_gross_t-1)/total_assets_t-1]`; `DEPI =
  [depreciation_t-1/(ppe_gross_t-1+depreciation_t-1)] /
  [depreciation_t/(ppe_gross_t+depreciation_t)]`; `SGAI =
  (sga_t/revenue_t) / (sga_t-1/revenue_t-1)` (8-variable only); `LVGI =
  [(long_term_debt_t+current_liabilities_t)/total_assets_t] /
  [(long_term_debt_t-1+current_liabilities_t-1)/total_assets_t-1]`
  (8-variable only); `TATA = (net_income_t - operating_cash_flow_t) /
  total_assets_t` (8-variable only).
- **Gross-PP&E proxy note (AQI/DEPI)**: the classic formula uses NET PP&E;
  this engine has no net-PP&E concept, so `PropertyPlantAndEquipmentGross`
  (WP8) is used instead — a documented approximation, the same kind of
  disclosed proxy choice as the FFO anchor's total-D&A substitute for
  real-estate-only depreciation (Sec.8c).
- Returns `None` (never raises, never fabricates) when any of DSRI/GMI/
  AQI/SGI/DEPI (the 5-variable model's own required inputs) can't be
  computed — a missing operand, or a zero denominator anywhere in the
  ratio-of-ratios chain (`_beneish_ratio` is the shared safe-division
  helper: `None` on a `None` operand or a zero denominator).
- Returns `{"m_score": float (2dp), "partial": bool (True when the
  5-variable model was used), "flag": bool, "components": {...}
  (whichever of dsri/gmi/aqi/sgi/depi/sgai/lvgi/tata were computed, 4dp)}`
  on success.

### `engine._build_beneish_m(normalized, metrics) -> tuple[Optional[dict], list[str]]`

- Resolves the current fiscal year via `resolve_fundamental_fy(metrics)`;
  the prior year is simply `fy - 1`. `fy is None` → `(None, [])`.
- Pulls both fiscal years' worth of the 12 concepts in
  `_BENEISH_CONCEPT_MAP` (`receivables, revenue, gross_profit,
  current_assets, ppe_gross, total_assets, depreciation, sga,
  long_term_debt, current_liabilities, net_income, operating_cash_flow`)
  and delegates entirely to `distress.beneish_m_score`.
- On success, appends ONE Turkish note naming the M-score, whether it's the
  partial (5-variable) model, and whether the manipulation flag fired.
  Never raises.

### Engine integration (`run_valuation`) — computed unconditionally, no chain participation

```python
beneish_m, beneish_notes = _build_beneish_m(normalized, metrics)
notes.extend(beneish_notes)
```

Called once, near the end of `_run_valuation`, alongside `altman_z`
(Sec.8g)/`lbo_floor_detail` (Sec.8h) — entirely outside the DCF/EPV/
revenue-first/RIM/FFO priority chain. Unlike those two, this is computed
for EVERY sector (no `_SECTORS_WITHOUT_FCF_DCF` gate): revenue/receivables/
gross-margin trends are meaningful manipulation signals regardless of
sector.

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

**N/A — no parameter, no interaction**, identical to Sec.8g/8h.

### Output shape additions (Sec.11)

```python
"beneish_m": None | {
   "m_score": float, "partial": bool, "flag": bool,
   "components": {"dsri", "gmi", "aqi", "sgi", "depi"[, "sgai", "lvgi", "tata"]},
},  # None only when even the 5-variable model's inputs are missing for
    # either fiscal year.
```

`_empty_valuation` (Sec.11) also gains `"beneish_m": None`.

### CLI / HTML / script-provider wiring — reuses Sec.8g's shared advisory-card mechanism, unchanged

`cli._distress_flags_line`, `report/template.html`'s
`distressFlagsCardHtml`, and `interpret/rule_based.py`'s
`_distress_risk_from_valuation` were ALL written in Sec.8g to already read
`valuation["beneish_m"]` defensively (`.get("beneish_m")`, degrading to
nothing when the key is absent) — landing this section required **zero
changes** to any of those three, exactly as Sec.8g's own design intended.

### Scope

Purely additive: does not change `dcf`, `earnings_power`, `rim`, `ffo`,
`altman_z`, `lbo_floor_detail`, `fair_value_range`, `sensitivity`,
`multiples`, `triangulation`, or any other existing output key's meaning or
value. Computed for every sector. A filer missing even the 5-variable
model's required two-year data degrades to `beneish_m: None` plus an
explanatory note, exactly like every other engine anchor's missing-data
behavior.

## 8k. Merton distance-to-default — `distress.merton_distance_to_default` / `distress._annualized_volatility` / `engine._build_merton_dtd`

An ADVISORY-ONLY overlay (extends `distress.py`, Sec.8g/8j's module). Same
non-chain-touching contract: never headlines `fair_value_range`, never
feeds `primary_dcf_scenarios`, never participates in
`triangulate.triangulate`'s confidence vote. Computed for every sector
(like Beneish M, Sec.8j) — market cap/total debt/price history/risk-free
rate are equally meaningful across sectors, unlike the leverage-RATIO
models in Sec.8g/8h that are structurally wrong for `financial`/`reit`.

Equity is modeled as a call option on the firm's assets (Merton 1974),
struck at the face value of debt. The two simultaneous Black-Scholes-shaped
equations relating OBSERVABLE equity value/volatility to UNOBSERVABLE asset
value/volatility are solved via 2D Newton-Raphson with a numerically
(finite-difference) approximated Jacobian — pure Python, `math.erf` for the
normal CDF, no scipy.

### `distress._annualized_volatility(price_df, window_days=252) -> Optional[float]`

- Computed INDEPENDENTLY from `price_df` (the same DataFrame
  `run_valuation` already receives) — deliberately NOT
  `technical/indicators.py`'s `volatility_20d` (a 20-day window, the wrong
  horizon for this model's 1-year default horizon).
- Annualized stdev of daily returns (`price_df["Close"].pct_change()`)
  over the trailing `window_days` (default 252, ~1 trading year),
  annualized by `sqrt(252)`.
- Returns `None` (never raises) when `price_df` is missing/malformed, or
  fewer than `_MERTON_MIN_RETURN_OBSERVATIONS` (60) daily returns are
  available in the trailing window (guards against a noisy estimate from a
  handful of days, e.g. a recent IPO).

### `distress.merton_distance_to_default(equity_value, equity_vol, debt_face_value, risk_free_rate, horizon_years=1.0) -> Optional[dict]`

- Solves `F1(V, σ_V) = V·N(d1) - debt_face_value·e^(-r·T)·N(d2) -
  equity_value = 0` and `F2(V, σ_V) = N(d1)·σ_V·V - equity_vol·equity_value
  = 0` simultaneously for the implied asset value `V` and asset volatility
  `σ_V`, where `d1 = [ln(V/debt_face_value) + (r + 0.5σ_V²)T] / (σ_V√T)`
  and `d2 = d1 - σ_V√T`.
- **Snapshot-only formulation, not the classic KMV iterative time-series
  procedure**: this engine has no historical asset-return series to
  iterate against, so both equations are solved AT ONCE via 2D
  Newton-Raphson (finite-difference Jacobian, central differences,
  `_MERTON_FD_EPS` relative step), starting from `V₀ = equity_value +
  debt_face_value`, `σ_V₀ = equity_vol · equity_value / (equity_value +
  debt_face_value)`. Capped at `_MERTON_MAX_ITER` (100) iterations;
  convergence requires both residuals below `_MERTON_TOLERANCE` (1e-6,
  scaled by `max(1, equity_value)`/`max(1, equity_vol·equity_value)`
  respectively, since equity values span orders of magnitude across
  filers).
- **Non-convergence, or a degenerate iterate (non-positive `V`/`σ_V`
  reached mid-solve, a domain error, a singular Jacobian) degrades to
  `None`** — never a garbage distance-to-default, never a partial/best-
  effort number.
- **Distance-to-default and probability of default read directly off the
  solved `d2`** (not a separate formula): `distance_to_default = d2`;
  `probability_of_default = 1 - N(d2)`. This is the RISK-NEUTRAL convention
  (asset drift assumed = risk-free rate) — a documented simplification of
  the classic KMV model, which estimates the asset's ACTUAL drift from a
  historical return series this engine doesn't reconstruct.
- Zone (practitioner heuristic, NOT a precisely calibrated academic
  cutoff — documented as such): `distance_to_default >= 3.0` → `"safe"`;
  `1.0 <= distance_to_default < 3.0` → `"elevated"`; `< 1.0` →
  `"distress"`.
- Returns `None` (never raises, never fabricates) when `equity_value`/
  `equity_vol`/`debt_face_value` is missing/non-positive, `risk_free_rate`
  is `None` (may legitimately be zero/negative otherwise), `horizon_years
  <= 0`, or the solve doesn't converge.
- Returns `{"distance_to_default": float (4dp), "probability_of_default":
  float (6dp), "asset_value": float (the solved implied total firm asset
  value), "asset_vol": float (the solved implied asset volatility),
  "zone": "safe"|"elevated"|"distress"}` on success.

### `engine._build_merton_dtd(metrics, price_df, risk_free_pct) -> tuple[Optional[dict], list[str]]`

- `equity_value = metrics["market_cap"]`; `debt_face_value =
  metrics["total_debt"]`; `equity_vol =
  distress._annualized_volatility(price_df)`.
- `risk_free_pct` is the Damodaran sector risk-free rate ALREADY resolved
  earlier in `_run_valuation` (a PERCENTAGE number, e.g. `4.5`) — divided
  by 100 before being passed through as `risk_free_rate`. `risk_free_pct is
  None` → `(None, [Turkish note])` — this engine never guesses a risk-free
  rate for this model (unlike the terminal-growth anchor, which has a flat-
  constant fallback; Merton's sensitivity to the risk-free rate doesn't
  warrant the same fallback).
- `equity_vol is None` (insufficient price history) → `(None, [Turkish
  note])`. Any other missing/degenerate input, or solver non-convergence,
  degrades via `merton_distance_to_default`'s own guards → `(None,
  [Turkish note])`.
- On success, appends ONE Turkish note naming the zone, distance-to-
  default, and probability of default. Never raises.

### Engine integration (`run_valuation`) — computed unconditionally, no chain participation

```python
merton_dtd, merton_notes = _build_merton_dtd(metrics, price_df, risk_free_pct)
notes.extend(merton_notes)
```

Called once, near the end of `_run_valuation`, alongside `altman_z`
(Sec.8g)/`beneish_m` (Sec.8j)/`lbo_floor_detail` (Sec.8h) — entirely
outside the DCF/EPV/revenue-first/RIM/FFO priority chain. `risk_free_pct`
is the SAME variable the terminal-growth anchor (Sec.4/WP2) already
resolved from Damodaran sector data earlier in this function — reused, not
recomputed.

### Confidence ceiling (`triangulate.triangulate`, Sec.10)

**N/A — no parameter, no interaction**, identical to Sec.8g/8h/8j.

### Output shape additions (Sec.11)

```python
"merton_dtd": None | {
   "distance_to_default": float, "probability_of_default": float,
   "asset_value": float, "asset_vol": float,
   "zone": "safe"|"elevated"|"distress",
},  # None when risk_free_pct is missing, price history is insufficient
    # for a volatility estimate, or the solver doesn't converge.
```

`_empty_valuation` (Sec.11) also gains `"merton_dtd": None`.

### CLI / HTML / script-provider wiring — reuses Sec.8g's shared advisory-card mechanism, unchanged

Same zero-new-wiring story as Sec.8j: `cli._distress_flags_line`,
`report/template.html`'s `distressFlagsCardHtml`, and
`interpret/rule_based.py`'s `_distress_risk_from_valuation` were all
written in Sec.8g to already read `valuation["merton_dtd"]` defensively
(`.get("merton_dtd")`) — landing this section required **zero changes** to
any of those three.

### Scope

Purely additive: does not change `dcf`, `earnings_power`, `rim`, `ffo`,
`altman_z`, `beneish_m`, `lbo_floor_detail`, `fair_value_range`,
`sensitivity`, `multiples`, `triangulation`, or any other existing output
key's meaning or value. Computed for every sector. A filer missing
market cap/total debt/sufficient price history/a Damodaran risk-free rate,
or whose solve doesn't converge, degrades to `merton_dtd: None` plus an
explanatory note, exactly like every other engine anchor's missing-data
behavior.

## 12. Two-phase interpret (`interpret/analyzer.py` refactor)

New public functions (keep module import-safe without `anthropic` installed;
keep ollama/anthropic/script providers; system prompt order METODOLOJI.md →
VALUATION.md (new: `Config.VALUATION_PATH`, default `<pkg>/VALUATION.md`) →
PROFIL.md → horizon instruction → output contract):

1. `propose_assumptions(normalized, ratios, metrics, sector_hint, provider,
   horizon, ...) -> dict` — returns `{"assumptions": {...§2...},
   "sector_type": str}`. Validation loop: run
   `sanity.validate_assumptions`; on violations, re-call the LLM once with the
   violation list appended ("şu sınırları ihlal ettin, revize et"); if still
   invalid (or provider is script / LLM unavailable), fall back to
   deterministic default assumptions from `rule_based.default_assumptions
   (metrics, sector_type)`:
   - base growth = clamp(revenue_cagr_5y or revenue_cagr_3y or 0.04, -0.05, 0.25)
   - bear = base - 0.05, bull = base + 0.05
   - terminal_growth = 0.025 all scenarios
   - discount_rate: base 0.10 (0.12 if unprofitable), bear +0.02, bull -0.01
   - story: template Turkish sentence naming the inputs used
2. `interpret_results(normalized, ratios, metrics, technical, red_flags,
   catalyst, valuation, provider, horizon, ...) -> dict` — phase-2 commentary.
   The LLM receives the full `valuation` dict and returns ONLY commentary
   fields:
   ```json
   {"fundamental_verdict": "UCUZ|MAKUL|PAHALI",
    "profile_fit": {"verdict": "UYUMLU|KISMEN|UYUMSUZ", "reason": "..."},
    "reverse_dcf_comment": "...", "cyclical_risk": "...",
    "horizon_note": "...", "key_risks": [...], "red_flags_comment": "...",
    "catalyst": "...", "summary": "..."}
   ```
   The phase-2 LLM must NOT emit `fair_value_range`, `technical_verdict`,
   `confidence`, `valuation`, `scenario_returns`, `entry_plan`, `stop_adding`,
   or `thesis_metric` — those keys are always supplied/overwritten by
   application code regardless of what the provider returns (matches the
   exclusion list in `analyzer.py`'s `_PHASE2_OUTPUT_CONTRACT`).

   Code-enforced post-processing (LLM cannot override):
   - `technical_verdict` from technical module (existing rule)
   - `confidence` from `valuation["triangulation"]["confidence"]`
   - `fair_value_range` injected from `valuation["fair_value_range"]`
   - `fundamental_verdict` cross-checked against the DCF signal: if the LLM's
     verdict contradicts `triangulation.signals.dcf` (ucuz↔PAHALI), override
     with the code signal and log.
   - full `valuation` dict attached under result key `"valuation"`.
   - `_provider`, `_model`, `_horizon`, `_weights` stamped as today.
   - `scenario_returns`, `entry_plan`, `stop_adding`, `thesis_metric` — the
     four METODOLOJI.md §4-§7 mechanical structures below, computed by
     `interpret/planning.py` and injected uniformly for **every** provider
     (`ollama`, `anthropic`, and `script` alike) by `_postprocess_phase2_
     result`, exactly like `fair_value_range`/`confidence` above — no
     provider, including the LLMs, computes any of these four fields itself:

     - **`scenario_returns`** (`planning.compute_scenario_returns`): see
       Sec.4's "Senaryo getirileri" subsection for the exact shape —
       `{"bear": {"ret_lo_pct": float|None, "ret_hi_pct": float|None},
       "base": {...}, "bull": {...}}`.
     - **`entry_plan`** (`planning.compute_entry_plan`, METODOLOJI.md §5,
       "Kademeli giriş planı"): a list of 0-5 tranche dicts, ordered by
       descending trigger price (breakout tranches, when present, on top):
       ```python
       [{"n": 1, "trigger": "Günlük kapanış 180.00 USD seviyesinin altına "
                              "inerse (bölge 177.30-182.70 USD — baz senaryo "
                              "alt bandı); gün içi dokunuş tetik saymaz.",
         "price_zone": {"lo": 177.30, "hi": 182.70}, "size_pct": 10.0,
         "invalidation": 142.50, "target": 250.0, "rr": 2.3, "note": None,
         "kind": "dip"},
        ...]
       ```
       Candidate trigger levels are pulled ONLY from already-computed figures,
       collected in two directional kinds (METODOLOJI.md §1 item 5):
       - **dip** (`kind="dip"`, level ≤ price): `fair_value_range`'s
         `bear.lo`/`base.lo`/`base.hi`/`bull.hi` plus the technical read's
         `low_52w`/`sma50`/`sma200`. The technical `support_levels` zones are
         deliberately NOT dip candidates — dip levels stay value-anchored.
         Instead, when a kept dip level coincides with a `support_levels`
         zone (falls inside the zone's `low`/`high` band widened by the 2%
         dedupe tolerance on each side), the tranche carries an informational
         Turkish confluence `note` ("Teknik destek bölgesiyle örtüşüyor
         (lo-hi USD).") — selection, sizing, invalidation, and R:R are
         untouched by this pass, and it never applies to breakout tranches.
       - **breakout** (`kind="breakout"`, level > price): `sma50`/`sma200`
         reclaims, each technical `resistance_levels` zone's midpoint
         (`zone["price"]`), and a `high_52w` breakout — unless an above-price
         resistance zone is itself flagged `is_52w_high`, in which case the
         separate `high_52w` candidate is skipped (same "new highs" event;
         avoid double-counting).
       Each side is deduplicated independently when two of its own levels sit
       within 2% of each other (the dip pass keeps the higher level, the
       breakout pass the lower/nearest-to-price one). Selection: if only one
       side has candidates, up to 5 are taken from it; if both fit within 5,
       all are kept; if they exceed 5, one slot per side is guaranteed
       (nearest to price on each side) and the rest are filled alternating
       sides by proximity. A single shared `target` (`bull.hi`, else
       `base.hi`) applies to every tranche. Invalidation is per kind: all dip
       tranches share one structural level (a fixed buffer below the lowest
       of `bear.lo`/`low_52w`/the lowest kept dip level); each breakout
       tranche carries its own failed-breakout invalidation (a fixed buffer
       below its own trigger level). `rr` is computed per tranche against its
       own invalidation and folds in a round-trip transaction cost
       (METODOLOJI.md §2); because dip tranches share one invalidation and
       target, dip-side R:R is mathematically non-decreasing as price falls —
       a guarantee (and mechanical check) scoped to consecutive dip tranches
       only. A breakout tranche whose entry sits at/above the shared target
       keeps its slot but reports `rr = None` plus a "Model üstü" `note`
       (trend-following add; no value-anchored reward). `trigger` text is
       Turkish, names the source level it came from (e.g. "baz senaryo alt
       bandı", "SMA200 desteği", "direnç/önceki zirve kırılımı"), and is
       explicitly daily-close-only — an intraday touch never counts. `[]`
       when price is missing/non-positive or neither side yields a usable
       candidate; fewer than 3 tranches is possible (never fabricated) when
       fewer distinct levels survive filtering/dedup/selection.
     - **`stop_adding`** (`planning.compute_stop_adding`, METODOLOJI.md §6,
       "Stop-adding sinyalleri"): `[{"code": str, "message": str}, ...]`,
       Turkish messages, `[]` if none fire. Checked in this fixed order:
       `BELOW_BEAR_FLOOR` (price below the bear-scenario floor),
       `NEAR_INVALIDATION` (price within 3% of the entry plan's shared
       invalidation level), `HIGH_UNCERTAINTY`
       (`valuation.sensitivity.high_uncertainty`), `ACTIVE_RED_FLAG` (one
       summarized entry for all active red flags), `BINARY_CATALYST_NEAR`
       (an upcoming named catalyst). **Concentration-limit signals are
       explicitly out of scope** — no `POZISYONLAR.md` position/portfolio
       schema exists yet (see ROADMAP.md's "Faz 2" item); once that schema
       lands, a concentration-limit signal can be added to this same list.
     - **`thesis_metric`** (`planning.select_thesis_metric`, METODOLOJI.md
       §7, "Tez doğrulama metriği"): `{"name": str, "latest_value":
       str|None, "trend": str|None, "rationale": str, "cycle": dict|None}`.
       `trend` is one of `"iyileşiyor"` / `"bozuluyor"` / `"yatay"`, or
       `None` if no prior fiscal year is available to compare against. The
       anchor metric is chosen from `valuation["sector_type"]` via a fixed
       sector→metric map (`mature`→net margin, falling back to ROE;
       `growth_unprofitable`→YoY revenue growth; `financial`→ROE as a NIM
       proxy; `reit`→FCF margin as an FFO proxy; `cyclical`→gross margin,
       falling back to net margin; unrecognized/`None`→net margin), read
       from `ratios`/`metrics` and never fabricated — `latest_value` is
       `None` (with `rationale` saying so) when the chosen metric isn't
       computable from the given inputs. `cycle` locates the latest value
       inside the anchor metric's own multi-year trough→peak range so the
       report can visualize where the business sits in its cycle (the same
       idea as the `CYCLICAL_TRAP` red flag's "latest margin vs historical
       peak"): `{"low": float, "high": float, "current": float, "position":
       float (0..1, `(current-low)/(high-low)` clamped), "low_fy": int,
       "high_fy": int, "current_fy": int, "n_years": int, "is_cyclical":
       bool, "series": [{"fy": int, "value": float}, ...]}`. `series` is
       the full annual series sorted ascending by fiscal year, used to draw
       a level sparkline of the metric's trajectory next to the positional
       bar. It is `None` when the metric came from the single-point
       `metrics` fallback (no series), when fewer than two fiscal years
       exist, or when the series is perfectly flat (trough == peak). All
       sectors get a `cycle` when a multi-year series exists; `is_cyclical`
       only drives display terminology, never the numbers.
       `rationale` always ends with the METODOLOJI.md §7 rule that two
       consecutive quarters against the thesis invalidate it.

   `rule_based.commentary()`'s own returned key set is **unchanged** by this
   addition — it still returns exactly the phase-2 LLM contract's commentary
   fields (`fundamental_verdict`, `profile_fit`, `reverse_dcf_comment`, ...);
   the four fields above are injected downstream by `_postprocess_phase2_
   result` for the `script` provider exactly as they are for `ollama`/
   `anthropic`, not computed inside `rule_based.py` itself.
3. Keep a thin `interpret(...)` wrapper (same signature as today, plus
   optional `valuation=None`, `submissions=None`) that runs phase 1 → engine →
   phase 2 internally, so `web/app.py` and old callers keep working. The
   `script` provider goes through the same engine with
   `rule_based.default_assumptions` and template-based commentary
   (`rule_based.commentary(valuation, ...)`) — fully offline, no LLM.

## 13. CLI verdict card additions (cli.py)

After the existing lines, following the plan's sample output, add (None-safe,
`—` for missing):
```
Fair Value (base, DCF): $95–$115   Güven: ORTA      # method label: DCF or P/B×ROE
Reverse DCF: fiyat 10y %19 CAGR ima ediyor (gerçekleşen 5y: %14)
Multiples:   P/E kendi Ny medyanının 88. yüzdeliğinde   # primary multiple used
Üçgenleme:   DCF pahalı · rDCF pahalı · multiples pahalı → yön net/karışık
Duyarlılık:  base $87–$131 (g±2pp, r±1pp) [+ " — yüksek belirsizlik" if flagged]
```
`analyze` flow becomes: fetch/normalize → prices/technical → metrics/red flags
→ submissions (SIC) → phase-1 assumptions → `run_valuation` → phase-2 →
card/HTML/store. Reuse the already-fetched submissions for both catalyst and
SIC (single fetch).

## 14. Store (store/database.py)

Extend `verdicts` via the `_ensure_columns` pattern with: `confidence TEXT`,
`sector_type TEXT`, `implied_growth REAL`, `fair_value_json TEXT`,
`valuation_json TEXT` (full valuation dict as JSON). `save_verdict` gains an
optional `valuation=None` kwarg; existing positional signature unchanged.

## 15. Config additions (config.py)

- `VALUATION_PATH` (env `VALUATION_PATH`, default `<pkg>/VALUATION.md`)
- `DAMODARAN_DIR` (env `DAMODARAN_DIR`, default `<cwd>/data/damodaran`)

## 16. HTML report (report/generator.py) — design spec

Single self-contained file `reports/{TICKER}_{date}_{horizon}.html`. Theme:
page #0d1420, card #111b2b, borders #223349; monospace for figures, system
sans for text; verdict colors red #ff6b5e / amber #ffb648 / green #4ade80;
band colors bear #ff6b5e, base #5aa7ff, bull #4ade80. Card max-width 560px,
single column on mobile. Layout top→bottom:
1. Header: ticker + price + date + data-source note; horizon badge right.
2. Signal-weight bar (fundamental/technical % from `_weights`).
3. Fan chart: horizontal price scale spanning min(bear.lo, price)·0.95 to
   max(bull.hi, price)·1.05; three semi-transparent scenario strips; current
   price ▼ marker with vertical line. Clicking a strip reveals that
   scenario's assumption row below (name + growth + dr + story); base
   selected by default. Pure inline JS.
4. Three verdict boxes (Fundamental / Teknik / Profil): label, colored verdict
   badge, position marker on a green→amber→red gradient gauge, one-line note.
5. Triangulation row: three method direction signals side by side (✓/✗/– +
   Turkish label) + confidence badge.
6. Sensitivity mini-table 3×3, base cell highlighted; "yüksek belirsizlik"
   tag when flagged.
7. Red-flags warning box (only if flags exist).
8. Catalyst + summary panel; reverse-DCF comment line.
9. Senaryo satırları (per-scenario returns): each bear/base/bull row in the
   fan-chart's assumption panel (item 3 above) additionally shows
   `result["scenario_returns"][key]`'s `ret_lo_pct`/`ret_hi_pct` next to that
   scenario's `lo`/`hi` price target (e.g. "$150–190 (%-8.1 / %+16.5)") —
   price target and % return always shown together, never one without the
   other (METODOLOJI.md §2).
10. Kademeli giriş planı (tiered entry plan): a table driven by
    `result["entry_plan"]`, one row per tranche in list order (already
    descending by trigger price) — columns tranche # (`n`), trigger price
    zone (`price_zone.lo`–`price_zone.hi`), size (`size_pct`), invalidation,
    target, and R:R (`rr`, `—` if `None`); the tranche's `trigger` text
    renders as a hover/footnote and `note` (if any) as an inline warning
    (e.g. the R:R-monotonicity flag). Section renders nothing (or a "giriş
    planı hesaplanamadı" note) when `entry_plan` is `[]`.
11. Stop-adding sinyalleri: a warning list from `result["stop_adding"]`, one
    line per `{"code", "message"}` entry in the fixed check order documented
    in Sec.12 (BELOW_BEAR_FLOOR → ... → BINARY_CATALYST_NEAR); hidden
    entirely when the list is empty.
12. Tez doğrulama metriği: a small panel from `result["thesis_metric"]`
    showing `name`, `latest_value` (`—` if `None`), a colored `trend` chip
    (`iyileşiyor`/`bozuluyor`/`yatay`, neutral styling if `None`), and
    `rationale` as supporting text.

Items 9-12 are the tiered entry plan / stop-adding / thesis-metric / per-
scenario-return additions (METODOLOJI.md §4-§7); every one of them carries
the same **"eğitim amaçlı, mekanik referans; yatırım tavsiyesi değildir"**
framing that governs the rest of the report (METODOLOJI.md §6's "Hiçbir
çıktı yatırım tavsiyesi değildir; mekanik referans çerçevesidir" rule /
README's "Not investment advice" section) — trigger levels and R:R are
mechanical outputs of already-computed numbers, not a recommendation to act
on them.

Data comes from `result` (incl. `result["valuation"]`, `result["scenario_
returns"]`, `result["entry_plan"]`, `result["stop_adding"]`, `result["thesis_
metric"]`), `metrics`, `technical`, `flags`. Missing pieces (e.g. no
valuation, or an empty `entry_plan`/`stop_adding`) degrade gracefully to the
old simpler card, never a crash.

## 17. Calibration harness (`sec_analyzer/calibrate.py`, `cli.py calibrate` subcommand)

Normalization Work Package 0 -- a measurement tool for the layered-penalty
undervaluation effort documented throughout §3/§8/§8a-§8e above, not a new
valuation code path. Adds no assumption/DCF logic of its own; it only
orchestrates the existing `cli.py` fetch/normalize/price/submissions/
catalyst helpers plus `interpret.analyzer.interpret(provider="script")` over
a fixed ticker basket, so the same production code path (including SIC-based
sector classification and the CAPM cost of equity) is what gets measured.

- `DEFAULT_TICKERS`: a ~28-ticker basket (`AAPL MSFT GOOGL AMZN META NVDA JPM
  BAC O PLD XOM CVX JNJ PFE PG KO CAT DE MU CRM ADBE RDDT PLTR UBER SHOP WMT
  COST VZ`) spanning mega-cap tech, financials, REITs, energy, healthcare,
  consumer staples, industrials/cyclicals, and high-growth/unprofitable
  names, broad enough that a shift in the basket's median fair-value/price
  ratio is signal rather than single-sector noise.
- `run_calibration(tickers, years=5, no_cache=False) -> List[dict]`: per
  ticker, resolves/fetches/normalizes/stores, fetches price + technical,
  computes `metrics`/`red_flags`/SEC submissions/catalyst, then calls
  `interpret(..., provider="script")`. Never raises -- any per-ticker
  failure is caught and recorded as a `"skipped"`/`"error"` row so one bad
  ticker never aborts the basket. An `"ok"` row carries `"price"`,
  `"fv_base_mid"` (`(fair_value_range.base.lo + .hi) / 2`), `"ratio"`
  (`fv_base_mid / price`), and `"method"` (`_method_slug`, one of `"hyper"`,
  `"cyclical-fcfe"`, `"epv"`, `"mature-rev"`, `"midgrowth-rev"`, `"dcf"`,
  `"ffo"`, `"pb-roe"` -- mirrors `cli._valuation_method_label`'s exact
  headline-precedence order, hyper-growth checked first). A row is `"skipped"`
  (never `"ok"`) when the fair-value base range or price is missing, OR when
  `metrics["price_reliable"]` is `False` (reason `"unreliable price
  (implausible P/E and P/S)"`).

  **Price-reliability guard (`metrics.compute_metrics`).** A corrupt
  market-data feed (e.g. a yfinance fallback that returns a uniformly
  downscaled series -- observed for BKNG, whose history came back ~31x too
  small, giving latest ≈ $182 against a true ≈ $5,700) leaves the engine's
  fair value correct but poisons any `fair_value_range / price` ratio (BKNG
  read as +1911% "upside"). `compute_metrics` cross-checks the fetched price
  against SEC per-share fundamentals -- the only price-independent reference --
  and sets `price_reliable=False` (+ a Turkish `price_reliability_note`) when
  the implied trailing **P/E and P/S are BOTH positive and below their floors
  at once** (`_PE_IMPLAUSIBLE_FLOOR=2.0`, `_PS_IMPLAUSIBLE_FLOOR=0.5`). This
  is the signature of a uniformly-downscaled feed; a legitimately cheap value
  stock or a peak-cyclical has a low P/E OR a low P/S but essentially never
  both, so the composite is deliberately conservative (prefers a false
  negative over wrongly discarding a genuinely cheap stock). The flag is
  purely additive -- it suppresses no existing metric; only consumers that
  divide by price (the calibration ratio here) act on it.
- `summarize_ratios(rows) -> dict`: a pure function over `"ok"` rows'
  `"ratio"` values -- `count`, `median`/`mean`/`p25`/`p75` (deterministic
  sorted-list linear-interpolation percentiles, `None` if `count == 0`), and
  three bucket counts (`bucket_under_0.8`/`bucket_0.8_1.2`/`bucket_over_1.2`)
  partitioning cheap/fair/expensive relative to price. **Healthy calibration
  target (re-baselined 2026-07-31): a median near 0.8-1.0** with wide
  dispersion (a tight cluster around 1.0 would itself be suspicious -- it
  would mean the engine is anchoring toward price rather than computing it
  independently). The original 0.9-1.1 band was drawn on 2026-07-17 under
  conditions three later corrections invalidated: a 5-year window in which
  `revenue_cagr_5y` could never compute (every consumer silently used the 3y
  figure), a two-endpoint CAGR (Sec.27), and an fcf0 deviation reference that
  included its own candidate year (Sec.4). With those fixed, the honest
  median measured ~0.855-0.864 across two independent weeks (`post-fixes`
  2026-07-24, `post-fcf0` 2026-07-31), so the band was moved to fit the
  corrected engine rather than the engine tuned to fit the stale band.
  **Reference baseline snapshot:** `reports/calibration_post-fcf0_
  20260731-1152.json` (median 0.8638, buckets 11/7/8, n=26; years=12,
  log-linear trend, prior-avg fcf0 reference). Future runs compare against
  the LATEST documented baseline snapshot per-ticker -- not an older
  snapshot (the 07-17 `final` predates the WP8-14 financial-anchor fixes and
  gives wrong per-name deltas), and not the median alone (the tail names
  have documented, individually-attributed reasons; see VALUATION.md's
  trajectory table).
- `save_calibration_snapshot(label, rows, summary) -> Optional[str]`: writes
  `Config.REPORTS_DIR/calibration_<label>_<YYYYMMDD-HHMM>.json`; never
  raises (logs a warning and returns `None` on failure).
- CLI: `python -m sec_analyzer.cli calibrate [--tickers AAPL,MSFT,...]
  [--label run] [--years 5] [--no-cache]` -- `cmd_calibrate` runs the basket,
  prints the per-ticker table (`print_calibration_table`) and the summary,
  and saves the JSON snapshot. `--tickers` defaults to `DEFAULT_TICKERS`;
  `--label` defaults to `"run"` and only affects the snapshot filename.

See VALUATION.md's calibration-methodology section for the measured
before/after trajectory of this normalization effort and what remains open.

## 18. Point-in-time (as-of) mode

`python -m sec_analyzer analyze MU --as-of 2022-06-30` runs the entire engine
using only data knowable on that date -- a backtest/autopsy mode, not a new
valuation method. Every existing output shape (Sec.11's `valuation` dict,
`fair_value_range`, `triangulation`, etc.) is unchanged; `as_of` only changes
which raw inputs (facts, prices, macro, filings) feed the SAME deterministic
pipeline. **Determinism guarantee:** `as_of=None` (the default, every existing
caller) is bit-for-bit identical to pre-as-of behavior -- every function below
takes `as_of` as an additive, defaulted keyword and short-circuits to its old
code path when it is `None`.

### Contract

- **Fundamentals:** `normalize.normalizer.normalize_facts(facts_json, years=5,
  as_of=None)` gains the `as_of` parameter. When set (`datetime.date` or ISO
  `"YYYY-MM-DD"` string), only facts with `filed <= as_of` (plain ISO string
  compare) survive before the existing dedup step -- so the value that was
  *actually public* on that date wins ("latest filed as of D"), and any
  restatement filed later is invisible, exactly as a contemporaneous analyst
  would have seen it. `filed == as_of` counts as knowable.
- **Prices/technical:** `fetch.prices.slice_asof(df, as_of) -> pd.DataFrame`
  returns only the rows of a price-history frame dated `<= as_of` (`as_of=None`
  returns `df` unchanged). The `analyze` CLI flow slices `price_df` through
  this before anything downstream touches it, so price, indicators, market
  cap, the P/S price-reliability gate (Sec.17), and `multiples.
  multiples_history` all derive from the truncated frame -- there is no
  separate as-of branch in any of those functions; they simply never see rows
  after the cutoff.
- **Macro (ERP / risk-free / terminal-growth anchor):**
  `damodaran.load_sector_data(dir_path, as_of=None, fred_rate=None)` (Sec.7)
  gains both parameters. When `as_of` is set:
  - **ERP:** the row for `as_of.year` in `data/damodaran/erp_history.csv`
    (columns `year, erp[, risk_free]`, same percentage format as `erp.csv`),
    falling back to the current `erp.csv` value when that year has no row or
    the row's `erp` cell is empty.
  - **Risk-free:** precedence `fred_rate` (the caller-supplied dict from
    `fetch.fred.get_risk_free_asof`, i.e. the actual FRED DGS10 observation on
    that date) -> `erp_history.csv`'s row's `risk_free` cell -> current
    `erp.csv`'s risk-free value.
  - Provenance is recorded on the returned dict as
    `sector_data["macro_asof"] = {"as_of": <iso>, "erp_source": <Turkish
    source string>, "risk_free_source": <Turkish source string>}`, and
    `engine.run_valuation` copies this block into the output's
    `valuation["macro_asof"]` (present only when `as_of` was given), plus
    appends two Turkish notes to `valuation["notes"]`: the macro-source
    provenance itself, and the static-sector-data caveat (below).
  - Sector multiples/betas in `multiples.csv` are **NOT** date-keyed -- they
    remain the current static snapshot regardless of `as_of` (see
    "Limitations" below).
  - `fetch.fred.get_risk_free_asof(as_of, no_cache=False) -> Optional[dict]`
    returns `{"value_pct": float, "date": "YYYY-MM-DD", "series": "DGS10",
    "source": "FRED DGS10"}` -- the last DGS10 observation on/before `as_of`
    (walks backward through weekends/holidays) -- or `None` on any failure
    (network, unparseable CSV, no observation before the cutoff). Never
    raises. Cached on disk (`Config.RAW_DIR/fred_DGS10.csv`, 24h freshness
    window); the series is append-only history so a stale cache is harmless
    for a historical `as_of`.
- **Filing signals:** `signals.events.detect_events(..., today=as_of)` and
  `fetch.filings.estimate_next_earnings(submissions, today=None)` (renamed
  reference-date parameter, defaults to `date.today()`) both take the as-of
  date as their reference "today" -- filings/events dated after it are
  excluded, so the 8-K event feed and the next-earnings catalyst estimate
  both reflect only what had actually been filed by `as_of`.
- **Suppressed:** analyst consensus price targets (yfinance) are undated at
  the source and cannot be made point-in-time, so `cli.cmd_analyze` skips
  fetching them entirely whenever `as_of is not None`, with a Turkish note
  rather than a silently-live (and thus anachronistic) figure.
- **Persistence:** in as-of mode, `_fetch_normalize_store` does **not** call
  `save_normalized` -- the `financials`/`ratios` tables are a current-view
  upsert, and writing a truncated, pre-restatement slice into them would
  corrupt the live view for ordinary (non-as-of) callers. `store.database.
  save_verdict(..., as_of: Optional[str] = None)` gains an `as_of` column
  (`_VERDICTS_EXTRA_COLUMNS`, `TEXT`, `NULL` for ordinary live runs) on the
  append-only `verdicts` table, so a backtest verdict is distinguishable from
  a live one. A new read helper, `store.database.load_verdicts(ticker,
  db_path=None, limit=100) -> List[dict]`, returns the stored scalar-column
  verdict history for a ticker (newest first, case-insensitive ticker match)
  powering a web verdict-history screen (`GET /history?ticker=X`); a sibling
  helper, `load_latest_stored_price(ticker, db_path=None) -> Optional[dict]`,
  returns the most recent stored `prices` row (`{"date", "close"}`) for that
  screen's current-price delta column, network-free.
- **`calibrate --as-of`** runs the whole calibration basket (Sec.17) as of a
  past date, e.g. `--as-of 2021-11-19 --label peak2021` vs. `--as-of
  2022-10-14 --label trough2022`, to separate engine conservatism from the
  market regime the basket happened to be priced in. `run_calibration(tickers,
  years=5, no_cache=False, as_of=None)` and `save_calibration_snapshot(label,
  rows, summary, as_of: Optional[str] = None)` both gain the parameter; the
  saved snapshot JSON records `as_of` (ISO string or `None`) alongside the
  existing summary fields.

### Signature amendments (additive, all default to the pre-as-of behavior)

```python
normalize.normalizer.normalize_facts(facts_json, years=5, as_of=None) -> dict
fetch.prices.slice_asof(df, as_of) -> pd.DataFrame
fetch.fred.get_risk_free_asof(as_of, no_cache=False) -> Optional[dict]
valuation.damodaran.load_sector_data(dir_path, as_of=None, fred_rate=None) -> Optional[dict]
valuation.engine.run_valuation(..., as_of=None, fred_rate=None) -> dict
interpret.analyzer.interpret(..., as_of=None, fred_rate=None) -> dict
signals.events.detect_events(..., today=as_of) -> List[dict]   # today: Optional[date]
fetch.filings.estimate_next_earnings(submissions, today=None) -> Optional[dict]
store.database.save_verdict(..., as_of: Optional[str] = None) -> int
store.database.load_verdicts(ticker, db_path=None, limit=100) -> List[dict]
store.database.load_latest_stored_price(ticker, db_path=None) -> Optional[dict]
calibrate.run_calibration(tickers, years=5, no_cache=False, as_of=None) -> List[dict]
calibrate.save_calibration_snapshot(label, rows, summary, as_of=None) -> Optional[str]
```

### CLI wiring (`cli.py`)

`analyze` and `calibrate` both gain `--as-of YYYY-MM-DD`, parsed by
`_parse_as_of(value) -> date` (argparse `type`): rejects unparseable strings
and any date after `date.today()` (a future cutoff is meaningless and almost
always a typo) via `argparse.ArgumentTypeError`. `cmd_analyze`'s flow, with
`as_of = getattr(args, "as_of", None)`:

1. `_fetch_normalize_store` normalizes with `as_of` and skips the DB write
   when it is set (above).
2. **No-data guard:** if `as_of is not None` and every `normalized["annual"]`
   series is empty (the cutoff predates the filer's first filing), the
   command prints a minimal Turkish `{"error": "as_of_no_data", "summary":
   ...}` card and returns before touching price/technical/valuation --
   avoids dividing by absent fundamentals rather than crashing.
3. Prices/technical are fetched then sliced via `slice_asof`; analyst targets
   are skipped; `fred_rate = _fetch_risk_free_asof(as_of, args.no_cache)` is
   resolved once and threaded into both `damodaran.load_sector_data` (via
   `interpret.analyzer.interpret`) and the report.
4. `catalyst`/`events` are computed with `as_of` as their reference date.
5. Price-row persistence (`_save_price_rows`) is skipped when `as_of is not
   None` (same current-view-upsert rationale as financials).
6. The result dict gains `result["as_of"] = as_of.isoformat()` when set, is
   passed to `save_verdict(..., as_of=...)`, and to `report.generator.
   generate_report(..., analysis_as_of=as_of.isoformat())` (report-card
   `as_of` continues to mean "date `price` is as of"; `analysis_as_of` is the
   new, separate point-in-time-cutoff field -- the two coincide in as-of mode
   but are conceptually distinct, e.g. `as_of` could lag `analysis_as_of` by a
   weekend).

### Limitations (must be surfaced, not hidden)

1. **Split-adjusted prices vs. historical share counts (highest-risk
   caveat).** yfinance closes are adjusted to today, so a stock split
   that happened *after* `as_of` (e.g. NVDA's 10:1 split in 2024) skews
   market cap and every price-derived multiple (P/E, P/S, EV/Sales) by the
   split factor when analyzing a date before that split. `metrics
   ["price_reliable"]` (Sec.17's P/E+P/S implausibility gate) catches some,
   but not all, of the resulting distortion -- it is not a complete guard
   against this specific failure mode.
2. **Survivorship bias.** A delisted ticker has no price history on
   yfinance, so an as-of calibration basket (Sec.17/`calibrate --as-of`) can
   only include names that are still trading today -- its measured
   ratio/median distribution is systematically skewed toward survivors.
3. **Static sector multiples and betas.** Only `erp`/`risk_free` are sourced
   historically (`erp_history.csv`); `multiples.csv`'s `pe`/`ps`/`pfcf`/
   `unlevered_beta`/`capex_sales` columns remain the current snapshot,
   NOT date-keyed, for every `as_of` value. Sector-relative multiples
   comparisons (Sec.10's axis-b) and CAPM beta in as-of mode therefore use
   today's sector data applied to a historical filer -- a documented
   approximation, not a historical sector reconstruction.

## 19. Financial-filer net-revenue basis (WP1) — `normalize.normalizer._apply_net_revenue_basis`

**Problem this fixes (real, currently-shipping defect).** `concepts.CONCEPTS
["Revenue"]` tries `RevenueFromContractWithCustomerExcludingAssessedTax`
first. For a lender/bank-shaped filer that tag carries ONLY the ASC-606
contract-fee slice of revenue, not the income statement's top line. SoFi
(CIK 1818874, SIC 6199) reports both tags: FY2025 contract-revenue
`$0.619B` vs. `RevenuesNetOfInterestExpense` `$3.613B`. Every revenue-derived
figure downstream — `ratios.net_margin`, `yoy_revenue_growth`,
`metrics.ps`/`revenue_cagr_5y`, the multiples history, the hyper-grower
detector — is therefore computed off a number ~5.8x too small. Note the
direction: the rejected value is an UNDER-statement here, not the
gross-revenue over-statement an aggregator "sales" field would produce, so
the rule below is deliberately symmetric (absolute divergence), catching
both.

**Ground truth is the income-statement top line net of interest expense**
("total net revenue"), which the financial-filer XBRL tag
`RevenuesNetOfInterestExpense` states directly. This section makes the
normalizer prefer it when the two disagree materially, and makes the
rejection auditable rather than silent.

### New canonical concepts (`normalize/concepts.py`)

Additive entries in `CONCEPTS`; no existing concept's tag list changes.

```python
"NetRevenue": ["RevenuesNetOfInterestExpense"],
"InterestIncome": ["InterestIncomeOperating", "InterestAndDividendIncomeOperating"],
"NoninterestIncome": ["NoninterestIncome"],
"Deposits": ["Deposits", "InterestBearingDepositLiabilities"],   # Sec.20
```

`NetRevenue`, `InterestIncome`, `NoninterestIncome` are added to
`FLOW_CONCEPTS`. `Deposits` is NOT (it is a balance-sheet stock, and
`STOCK_CONCEPTS` is defined as the complement, so it lands there
automatically). No `CONCEPT_UNITS` or `TAG_TAXONOMY` entry is needed — all
four are plain `USD` us-gaap tags.

`Deposits`' second tag is a strictly narrower measure (interest-bearing
deposits only). Because `_extract_concept` merges across tags by period, a
filer reporting the preferred tag in some years and only the fallback in
others gets a mixed series. This is accepted: `Deposits`' only consumer is
Sec.20's coarse `> 20% of liabilities` materiality trigger, where a narrower
measure can only UNDER-state the ratio — the failure mode is "trigger doesn't
fire", never a false positive.

### `_apply_net_revenue_basis(annual, quarterly, matched_tags, missing) -> dict`

A private helper called from `normalize_facts` **at the very end**, after the
global fiscal-year windowing and immediately before the return dict is
assembled, so it operates on the same windowed series every consumer sees.

Constant: `_NET_REVENUE_DIVERGENCE_THRESHOLD = 0.05` (5%, strictly above
triggers).

1. Build `net = {fy: value}` from `annual["NetRevenue"]` and `rep = {fy:
   value}` from `annual["Revenue"]`.
2. **No `NetRevenue` data at all** → no change; return the `basis:
   "as_reported"` metadata block below with every other field `None`/empty.
   This is the path EVERY non-financial filer takes (the tag is
   financial-specific), so their behavior is bit-for-bit unchanged.
3. `overlap = sorted(fy for fy in net if fy in rep and net[fy])` (a zero
   `net[fy]` is excluded — it cannot be a divergence denominator).
   - `divergence(fy) = abs(rep[fy] - net[fy]) / abs(net[fy])`
   - `max_divergence = max(divergence(fy) for fy in overlap)`, `None` if
     `overlap` is empty.
   - `divergent_fys = [fy for fy in overlap if divergence(fy) >
     _NET_REVENUE_DIVERGENCE_THRESHOLD]`
4. **Swap decision.** Swap iff `NetRevenue` has at least one annual value AND
   (`overlap` is empty — nothing to contradict, and `Revenue` may be missing
   entirely — OR `max_divergence > _NET_REVENUE_DIVERGENCE_THRESHOLD`).
   Otherwise no swap: the two tags agree within tolerance and the existing
   `Revenue` series stands.
5. **On swap:** `annual["Revenue"]` becomes a copy of `annual["NetRevenue"]`
   and `quarterly["Revenue"]` a copy of `quarterly["NetRevenue"]` (copy, not
   alias — the `NetRevenue` buckets stay independently readable). Copies are
   deep enough that mutating one record list cannot affect the other.
   `matched_tags["Revenue"] = matched_tags["NetRevenue"]`, and `"Revenue"` is
   removed from `missing` if present.
   - **Mixed-basis years are dropped, not backfilled.** Fiscal years present
     in the old `Revenue` series but absent from `NetRevenue` are simply gone
     from the swapped series. A short single-basis series is strictly better
     than a longer series that silently splices two revenue definitions
     (`concepts.py`'s module docstring already states this principle). The
     dropped years are recorded in `dropped_fys`.
   - When `quarterly["NetRevenue"]` is empty but the annual swap happened,
     `quarterly["Revenue"]` is set to `None` rather than left on the rejected
     basis (same no-mixed-basis rule).
6. **Never raises.** The whole body is wrapped so that any unexpected shape
   logs a warning and degrades to the no-swap `basis: "as_reported"` result —
   consistent with the module's "never raise on an individual concept" rule.

### `normalized["revenue_basis"]` — new top-level key (always present)

```python
{
  "basis": "net_revenue" | "as_reported",
  "swapped": bool,
  "max_divergence": float | None,        # fraction, rounded to 4dp
  "divergent_fys": [int, ...],
  "dropped_fys": [int, ...],
  "rejected_annual": {fy: float} | None, # the as-reported series NOT used
  "rejected_tags": [str, ...] | None,    # matched_tags of that series
  "gross_annual": {fy: float} | None,    # InterestIncome + NoninterestIncome
  "note": str | None,                    # Turkish, only when swapped
}
```

`gross_annual` is **informational only** — it is never used as a revenue
basis, never feeds a ratio, and exists so the report can show the
gross/net wedge a financial filer's headline "revenue" figure hides (SoFi
FY2025: gross `$4.77B` = interest income `$3.375B` + noninterest income
`$1.394B`, vs. net revenue `$3.613B`). Computed only for fiscal years where
BOTH inputs exist; `None` when neither concept resolved.

Turkish `note` when swapped, naming the worst year concretely, e.g.:

> "Gelir bazi duzeltildi: raporlanan gelir etiketi (FY2025: 0,62 Mr$) ile
> faiz gideri dusulmus net gelir (FY2025: 3,61 Mr$) arasinda %82,9 sapma
> var. Finansal kuruluslarda dogru baz net gelirdir; tum gelir turevli
> oranlar net gelir uzerinden hesaplandi."

Note the denominator: divergence is measured against the NET figure
(`abs(rep - net) / abs(net)`), so SoFi's FY2025 pair is `abs(0.619 - 3.613) /
3.613 = 0.8287` → 82.9%, NOT the 483% you get from expressing the same gap as
a percentage of the (much smaller) rejected figure. Both describe the same
5.8x error; the spec's ratio is the one the code computes and reports.

(The implementation writes proper Turkish with correct diacritics; the quote
above is ASCII-folded only to keep this spec file encoding-safe.)

The note is appended to the engine's `notes` list by the CLI/engine wiring
(the same channel every other Turkish data-quality note uses) so it reaches
the verdict card and the HTML report.

### Scope

Purely a normalize-layer input correction. No valuation function, output key,
or sector routing changes. Non-financial filers are unaffected by
construction (step 2). `to_annual_series(normalized, "Revenue")` keeps
identical semantics — only the underlying data is corrected.

## 20. Deposit-funded detection + financial-sector metric hygiene (WP2)

Three independent, additive changes, all keyed off `sector_type ==
"financial"`. `reit` is NOT included in any of them: a REIT legitimately uses
EV/EBITDA and has no deposit funding, and its FFO path (Sec.8c) is already
correct.

### 20a. Deposit-funded override — `sector._is_deposit_funded`

A filer whose balance sheet is funded by deposits is a bank in economic
substance regardless of the SIC code it files under. Constant:
`_DEPOSIT_FUNDED_LIABILITY_SHARE = 0.20` (strictly above triggers).

```python
def _is_deposit_funded(normalized: dict, metrics: dict) -> bool:
```
- Reads the `Deposits` and `TotalLiabilities` annual series and walks fiscal
  years **descending**, taking the first FY where BOTH have a value AND
  `liabilities > 0` — the two figures must come from the SAME fiscal year (a
  deposits figure divided by a different year's liabilities is meaningless),
  and a non-positive-liabilities year is unusable data that is skipped rather
  than treated as a decision. Independent of `metrics["latest_fy"]`, mirroring
  `_build_pb_roe`/`_build_rim`'s FY-selection discipline (Sec.8/8f).
- Returns `True` iff that year's `deposits / liabilities >
  _DEPOSIT_FUNDED_LIABILITY_SHARE`. No usable year, or any missing input →
  `False`.
- Never raises (broad except → `False`, matching `detect_hyper_grower`).

**Placement in `classify_sector`** — after the REIT and financial-SIC checks,
before the semiconductor/cyclical branch:

```python
if sic_int == _REIT_SIC or _is_reit_like_sic(sic_int):
    return SECTOR_REIT
if _FINANCIAL_SIC_RANGE[0] <= sic_int <= _FINANCIAL_SIC_RANGE[1]:
    return SECTOR_FINANCIAL
if _is_deposit_funded(normalized, metrics):        # NEW
    return SECTOR_FINANCIAL
```

Ordering rationale: it must never override `reit` (a mortgage REIT keeps its
FFO anchor), and it must run before `cyclical`/profitability so a
deposit-funded filer with a non-6xxx SIC is routed to the RIM anchor rather
than an FCF-DCF. SoFi (SIC 6199) already classifies `financial` on SIC alone —
this override exists for the filers SIC misses, and SoFi is the proof the
trigger's arithmetic is right (deposits `$40.24B` / liabilities `$42.89B` =
94%).

`classify_sector`'s docstring and the Sec.8 classification list above gain
this rule. **The `sic is None` engine-wiring fallback is unchanged** — the
override lives inside `classify_sector`, which the CLI only calls when `sic`
is not `None` (Sec.8).

### 20b. EV metrics are not defined for a financial filer

Enterprise value adds net debt to market cap to value the whole capital
structure. For a deposit-funded lender, "debt" IS the raw material of the
business, so EV and every EV multiple are meaningless. New engine constant:

```python
_SECTORS_WITHOUT_EV = ("financial",)
```

1. `_derive_current_multiples(normalized, ratios, metrics, price,
   suppress_ev=False)` gains a defaulted keyword. When `True` it neither
   seeds `current["ev_ebit"]/["ev_ebitda"]` from `metrics` nor runs the
   `metrics["ev"]` back-fill block, and emits no `FD/FVÖK`/`FD/FAVÖK`
   "derived from FY…" note. Both keys come back `None`. `suppress_ev=False`
   (every existing caller) is bit-for-bit unchanged.
2. In `_run_valuation`, the call passes `suppress_ev=sector_type in
   _SECTORS_WITHOUT_EV`. Immediately after it, when suppressed:
   - `current["ev_sales"] = None`;
   - every row of `history` gets `ev_sales`/`ev_ebit`/`ev_ebitda` set to
     `None` (so no percentile can be computed from history either);
   - `net_debt_to_ebitda` is forced to `None` and `leveraged` to `False`.
   `ev_ebit_pct`/`ev_ebitda_pct` then fall out as `None` on their own
   (`percentile_position` returns `None` for a `None` current value) — no
   special-casing at the percentile call sites.
3. `multiples_out` gains one additive key: `"ev_applicable": bool` (`False`
   only for `_SECTORS_WITHOUT_EV`, `True` everywhere else, including in
   `_empty_valuation`'s crash-safety shape where it is `True`).
4. One Turkish note when suppressed, stating that enterprise value is
   undefined for a financial institution because deposits/borrowings are the
   raw material of the business rather than a capital-structure adjustment,
   and that EV/EBITDA, EV/EBIT and EV/Sales were therefore not computed.

**Downstream degradation is already correct and requires no changes.**
`triangulate` treats `ev_ebitda_pct=None` + `leveraged=False` as "not EV
primary" and falls back to its existing P/E → P/S → P/FCF order (it never
emits an `FD/FAVÖK` label in that path). `cli._ev_multiples_line` and the
report's `evMultiplesLineHtml` already return nothing when the values are
`None`. The LBO floor and Altman Z are already gated off for `financial` via
`_SECTORS_WITHOUT_FCF_DCF`. `metrics` itself is **NOT mutated** — the engine
suppresses only within its own output, so the stored metrics payload and any
non-valuation consumer keep their existing shape.

### 20c. Liquidity/leverage checks are re-based for a financial filer

`ratios.compute_ratios` already computes `debt_to_equity` as
`TotalLiabilities / StockholdersEquity` — deposit-inclusive leverage, exactly
the measure the source analysis asks for. Only the LABEL and the THRESHOLD
are wrong for this sector.

1. **`interpret/rule_based.py`** — `_build_checks(latest_fy, series,
   ratio_by_fy, sector_type=None)` gains a defaulted keyword, threaded from
   `_analyze`. When `sector_type == "financial"`:
   - the **"Liquidity"** check (current ratio `< 1.0`) is reported as
     not-applicable (`None`, the existing "not computable" state) with a
     Turkish detail explaining that the current ratio is undefined for a
     financial institution (assets/liabilities are not classified by
     maturity) — a bank's current ratio is an artifact of how it happens to
     tag `AssetsCurrent`, not a liquidity signal;
   - the **"Leverage"** check keeps the non-positive-equity auto-fail
     unchanged, but its `debt_to_equity` threshold moves from `2.0` to a
     separate constant `_FINANCIAL_LEVERAGE_MAX = 10.0`, and its detail
     string names the measure as liabilities-to-equity including deposits.
     SoFi's `3.97` is unremarkable for a deposit funder and must not surface
     in `key_risks`; a genuinely over-levered bank (above ~10x) still does.
   - `sector_type=None` (any existing caller that doesn't pass it) preserves
     today's behavior exactly.
2. **`report/template.html`** — `ratioTrendCardHtml(ratios)` gains a
   `sectorType` argument, passed by `financialsTabHtml(payload)` from
   `payload.result.valuation.sector_type` (already in scope one frame up).
   For `"financial"`: the "Cari Oran" row is omitted entirely, the
   "Borç / Özkaynak" row is relabeled to a deposit-inclusive
   liabilities-to-equity label with its red threshold moved from `>2` to
   `>10`, and the card footnote is replaced with one describing the
   deposit-inclusive measure.
3. **`cli._print_ratios(ratios, sector_type=None)`** gains the same defaulted
   keyword; for `"financial"` the `Current Ratio` column is dropped from the
   table. Its one caller, `_fetch_normalize_store`, has no SIC in scope
   (submissions are fetched later, and only by `analyze`), so it passes
   `SECTOR_FINANCIAL` when `sector._is_deposit_funded(normalized, {})` holds
   and `None` otherwise — the deposit test needs no extra network call and is
   sufficient to know the column is meaningless. A `financial`-by-SIC filer
   that is NOT deposit-funded (an insurer or broker) therefore still shows the
   column in this one raw-data table; that is a deliberate, documented limit
   of the cheap check, not an oversight. Any other caller passing nothing is
   unchanged.
4. `web/templates/index.html` is a developer-facing raw-data table and is
   left alone — it is explicitly not a user-facing verdict surface.

### Scope

No valuation anchor, fair-value number, triangulation weight, or output key
MEANING changes. The only new output keys are `multiples.ev_applicable`
(20b) and `normalized["revenue_basis"]` (Sec.19). Non-financial filers are
unaffected by every rule in both sections.

## 21. Earnings-catalyst correctness (WP5) — `fetch.filings.estimate_next_earnings` / cache TTL

**Problem this fixes (real, observed on SOFI 2026-07-30).** The verdict card
showed "~9 gün içinde earnings (tahmini 2026-08-07)" on a day when SoFi had
ALREADY released Q2'26 earnings — the day before, on 2026-07-29. Two
independent defects combined:

1. **Wrong event.** `estimate_next_earnings` projected from `10-Q`/`10-K`
   FILING dates. The earnings *release* reaches EDGAR earlier, as an `8-K`
   carrying item **2.02** ("Results of Operations and Financial Condition").
   For SoFi the 8-K→10-Q lag is a stable **9 days** (median over the last 8
   quarters: `[9, 28, 7, 9, 9, 18, 8]`), which is exactly the "9 gün" the card
   displayed. The estimate was therefore pointing at the paperwork that
   follows the catalyst, not the catalyst.
   - Real 2.02 releases: `2025-04-29, 2025-07-29, 2025-10-28, 2026-01-30,
     2026-04-29, 2026-07-29` → correct next estimate **2026-10-28** (Q3),
     90 days out, not 8 days out.
2. **No cache freshness.** `get_submissions` returned any cache file that
   merely EXISTED. The on-disk SoFi submissions document had stopped at
   `2026-06-29`, so the 2026-07-29 earnings 8-K was invisible regardless of
   fix (1). A catalyst estimate is the most time-sensitive field the report
   carries and was being computed from month-old data.

### 21a. Earnings releases come from `8-K` item 2.02

`estimate_next_earnings(submissions, today=None)` keeps its signature and its
never-raises contract. Its internals gain a **primary** source with the
existing periodic-filing logic demoted to a **fallback**.

- **Primary — release dates.** Filings where `form == "8-K"` AND the
  filing's `items` string contains `"2.02"` (SEC packs a filing's 8-K item
  numbers into a comma-separated string, e.g. `"2.02,9.01"`). Same
  point-in-time guard as today: a filing dated after `today` is ignored.
- **Fallback — periodic filings.** When fewer than `_MIN_USABLE_FILINGS` (3)
  release dates are available — which includes every submissions dict with
  no `items` key at all (older fixtures, the backtest's synthetic
  documents) — the function behaves EXACTLY as before, projecting from
  `10-Q`/`10-K` filing dates. This keeps every existing caller and test
  bit-for-bit unchanged and degrades gracefully for filers whose 8-K items
  SEC has not populated.
- Median-gap projection, `_MAX_FILINGS_CONSIDERED` windowing, and the
  roll-forward-to-`today` loop are unchanged in both paths; only the input
  series differs.

**Quarter label from releases.** `_next_quarter_label` (which counts 10-Qs
since the last 10-K) cannot read 8-K forms, so the primary path uses a
sibling, `_next_quarter_label_from_releases(release_dates, last_10k_date)`:
count the releases strictly AFTER the most recent `10-K` FILING date, then map
through the existing `_QUARTER_LABELS` (`0→Q1, 1→Q2, 2→Q3`, else `"FY"`).
Verified against SoFi's real calendar at four points in time:

| As of | Last 10-K filed | Releases since | Next label | Correct? |
|---|---|---|---|---|
| after 10-K (2026-02-17) | 2026-02-17 | 0 | Q1 | yes (Q1 rel. 2026-04-29) |
| after Q1 rel. | 2026-02-17 | 1 | Q2 | yes (Q2 rel. 2026-07-29) |
| after Q2 rel. | 2026-02-17 | 2 | Q3 | yes (Q3 rel. ~2026-10-28) |
| after Q3 rel. (2025-10-28) | 2025-02-24 | 3 | FY | yes (FY rel. 2026-01-30) |

The FY/Q4 release lands BEFORE the 10-K that reports the same year, which is
why anchoring on the 10-K FILING date (not fiscal-year end) gives the right
count. When the history carries no `10-K` at all, the quarter label is `None`
and the label reads "Sonraki bilanço ~<date>" instead of "Q<n> earnings ~".

### 21b. An already-released quarter is not an upcoming catalyst

New constant `_RECENTLY_REPORTED_DAYS = 3`. The returned dict gains four
additive keys (existing three unchanged):

```python
{
  "estimate_date": "YYYY-MM-DD",   # unchanged meaning: the NEXT release
  "label": str,
  "based_on": str,
  "source": "8-K 2.02" | "10-Q/10-K",   # NEW - which series produced it
  "last_report_date": "YYYY-MM-DD" | None,  # NEW - most recent release/filing
  "days_until": int,                # NEW - estimate_date - today, >= 0
  "recently_reported": bool,        # NEW - last report within 3 days of today
}
```

- `estimate_date` is ALWAYS the next release, never the one just published.
  For SoFi on 2026-07-30 that is `2026-10-28`, so the report's existing
  21-day proximity badge (`daysUntil(...) <= 21`, template.html) stops firing
  on its own — **no template change is needed for the badge**; it was
  reporting a wrong date, not applying a wrong rule.
- When `recently_reported`, the Turkish `label` leads with the fact rather
  than the projection, e.g. `"Q2 açıklandı (29 Tem) · sonraki: Q3 earnings
  ~28 Eki"`. Otherwise it keeps today's shape, `"Q3 earnings ~28 Eki"`.
- `days_until` is computed against the same `today` the projection used, so
  it stays deterministic in as-of mode (no wall-clock read downstream).

**`planning._compute_stop_adding` proximity gate.** Today it appends a
`BINARY_CATALYST_NEAR` signal ("Yaklaşan binary katalizör…") whenever the
catalyst dict merely HAS a label — unconditionally, even for an event a
quarter away, and even for one that already happened. It now fires only when
`catalyst["recently_reported"]` is false AND `0 <= catalyst["days_until"] <=
_CATALYST_NEAR_DAYS` (21, matching the report badge's window). A catalyst
dict lacking `days_until` (an older/hand-built dict) keeps the previous
unconditional behavior, so no existing caller silently loses the signal.

### 21c. Cache freshness — `fetch.companyfacts`

`get_company_facts` and `get_submissions` currently return any cache file
that exists. Both gain a TTL, with new `Config` entries:

```python
SUBMISSIONS_CACHE_TTL_HOURS = 24      # filing history moves daily
COMPANYFACTS_CACHE_TTL_HOURS = 168    # XBRL facts move quarterly (7 days)
```

- A helper `_cache_is_fresh(path, ttl_hours) -> bool` compares
  `os.path.getmtime(path)` against `time.time()`. A non-positive TTL means
  "never expires" (the pre-existing behavior, available as an escape hatch).
- Cache hit AND fresh → return it, as today.
- Cache exists but STALE → re-fetch. **If the re-fetch fails** (network
  error, SEC 5xx), fall back to the stale cache with a warning rather than
  propagating the exception: a stale document is strictly better than no
  analysis, and this preserves offline/degraded operation that the
  exists-only check gave for free.
- No cache → fetch, exactly as today.
- `no_cache=True` still bypasses everything and overwrites.

These are fetch-layer, wall-clock-dependent by nature; CLAUDE.md's determinism
rule constrains the analysis/valuation path, which is unaffected — the same
inputs still produce the same outputs.

### Scope

No valuation anchor, fair-value number, or triangulation weight changes. The
report template is untouched (21b explains why the badge self-corrects). The
only behavior changes are: which dates the catalyst estimate is built from,
two extra Turkish label shapes, one newly-gated planning signal, and cache
re-fetch timing.

## 22. Growth-cap transparency + externally-funded growth (WP6) — `dcf.rim_external_growth_per_share` / engine wiring

**Problem this fixes (real, observed on SOFI 2026-07-30).** `rim_per_share`
and `fcfe_sustainable_growth_per_share` both cap each projection year at
`g_eff = min(g_year, roe)`, encoding the internal-funding identity
`g = b x ROE` with `b <= 1`. For a filer whose ROE sits below its assumed
growth, that cap silently discards the entire growth assumption:

| assumed `growth_5y` | `g_eff` actually used | RIM per share |
|---|---|---|
| 0% | 0% | $5.48 |
| 5% | 4.59% | $5.05 |
| 10% | 4.59% | $5.05 |
| 25% | 4.59% | $5.05 |
| 40% | 4.59% | $5.05 |

(SoFi FY2025: `ni0 = 481.3M`, `bve0 = 10,489.5M`, `roe = 4.59%`, `r = 10%`.)

Two consequences, both defects:

1. **The report states a growth rate the model did not use.** SoFi's base
   `fair_value_range` note reads "bu senaryoda uygulanan büyüme %25.0" while
   the model compounded earnings at 4.59%. `financial` has no
   `scenario_meta` override (Sec.11's chain covers hyper/mature/midgrowth/
   cyclical/EPV but not RIM), so the label falls through to the raw
   assumption. `_cyclical_fcfe_scenario_meta` does disclose its reinvestment
   rate but still prints the UNCAPPED `growth_5y` as the growth.
2. **The bear/base/bull band is a discount-rate band wearing a growth
   label.** Holding `r` fixed and moving `g` from 5% to 30% leaves every
   scenario byte-identical; only `r` moves the number
   (bear $4.22 / base $5.05 / bull $5.53 come from r = 12/10/9%).

### 22a. Both anchors report the growth path they actually used

`rim_per_share` and `fcfe_sustainable_growth_per_share` gain three additive
return keys (no signature change, no math change):

```python
"growth_path": [float, ...],       # the 10 g_eff values actually applied
"effective_growth_5y": float,      # min(growth_5y, roe) -- years 1-5 are flat
"growth_capped": bool,             # True iff any year's g_eff < its g_year
```

Existing keys and every computed number stay bit-for-bit identical.

### 22b. Externally-funded growth — `dcf.rim_external_growth_per_share`

```python
rim_external_growth_per_share(bve0, ni0, roe, growth_5y, terminal_growth,
                              discount_rate, shares, terminal_roe=None) -> dict
```

Answers the question the cap suppresses: *what if the firm funds its assumed
growth by issuing equity instead of only from retained earnings?* SoFi did
exactly that in FY2025 -- equity went `6,525M -> 10,489M` (+3,964M) on 481M of
net income, so roughly 3.5B came from external issuance.

**Issuance is modeled at fair value, which is value-neutral by construction.**
This is the textbook treatment (new shareholders pay in exactly what their
claim is worth), and it is what lets the model avoid inventing an
issue-price convention -- an issue price below fair value would transfer
value from existing holders, above it would transfer value to them, and
either choice would make an *intrinsic* estimate depend on the market price
it is supposed to be tested against. Consequently:

- **the share count is NOT inflated** -- `per_share` divides by today's
  shares, since the injected capital and the claim it buys cancel;
- the entire value effect comes from applying the `ROE - r` spread to a
  **larger book value**.

Projection, years 1..10 (`HORIZON_YEARS`), using the same
`_year_growth_rate` fade as every sibling:

- `ni_t = roe * bve_{t-1}` (ROE-consistent, so the earnings path cannot drift
  away from the stated ROE);
- `ri_t = ni_t - discount_rate * bve_{t-1} = (roe - discount_rate) * bve_{t-1}`;
- `bve_t = bve_{t-1} * (1 + g_t)` at the FULL, uncapped `g_t`;
- `external_funding_t = max(0.0, bve_{t-1} * (g_t - roe))` -- the equity that
  retention alone cannot supply (informational);
- terminal exactly as `rim_per_share`: `ri_terminal = (terminal_roe_resolved -
  discount_rate) * bve_10`, `tv = ri_terminal / (discount_rate -
  terminal_growth)`, so the engine's `terminal_roe = discount_rate` gives
  `tv = 0`.

**The sign is the whole point, and it is unambiguous:**

| | effect of faster growth |
|---|---|
| `roe > discount_rate` | `ri_t` more positive -> value ADDED |
| `roe == discount_rate` | `ri_t == 0` every year -> value UNCHANGED at `bve0` |
| `roe < discount_rate` | `ri_t` more negative -> value DESTROYED |

So for SoFi (ROE 4.59% vs COE 10%) growing faster on external capital makes
the per-share value **lower**, not higher. The diagnostic exists to quantify
that, not to rescue the number.

Returns `{"per_share", "equity", "bve0", "ri_path", "bve_path", "tv",
"effective_shares", "per_share_internal", "value_gap_per_share",
"external_funding_total", "external_funding_path"}`.

`per_share_internal` re-runs the SAME ROE-consistent path with `g_t` capped
at `roe`, so `value_gap_per_share = per_share - per_share_internal` isolates
the funding assumption alone -- it is not contaminated by the difference
between this function's ROE-consistent earnings path and `rim_per_share`'s
`ni0`-compounding one (the documented F4 front-loading, Sec.8f). Comparing
this function's `per_share` against `rim_per_share`'s output would mix two
changes; comparing it against `per_share_internal` isolates one.

Raises `ValueError` on the same invalid inputs as `rim_per_share`
(`ni0`/`bve0`/`shares`/`roe`/`discount_rate <= terminal_growth`). Nothing
rounded here.

### 22c. Engine wiring

**`_build_rim`** (Sec.8f) gains, in its returned dict:

```python
"growth_capped": bool,
"effective_growth_5y": float | None,   # base scenario's
"assumed_growth_5y": float | None,     # base scenario's, for the disclosure
"external_growth": None | {            # base scenario only; diagnostic
    "per_share": float,
    "per_share_internal": float,
    "value_gap_per_share": float,
    "external_funding_total": float,
},
```

`external_growth` is built ONLY when the base scenario's growth was actually
capped (there is nothing to diagnose otherwise) and is **advisory**: it never
headlines `fair_value_range`, never feeds `primary_dcf_scenarios`, and never
enters triangulation -- same advisory-only discipline as Sec.8g/8j/8k.

**Turkish note when the cap binds**, naming both rates, the reason, and the
quantified consequence, e.g.:

> "RIM büyüme varsayımı içsel finansman kısıtına takıldı: varsayılan %25,0
> büyüme yerine %4,6 uygulandı (g = b x ROE, ROE %4,6 -- şirket kârından daha
> hızlı büyümeyi kendi kaynağıyla fonlayamaz). Büyümenin dışarıdan özkaynak
> ihracıyla fonlandığı varsayılsa, ROE (%4,6) özkaynak maliyetinin (%10,0)
> altında olduğu için hisse başına değer 5,05 $ değil 3,12 $ olurdu."

**New `_rim_scenario_meta(rim_detail, assumptions)`**, mirroring
`_epv_scenario_meta`/`_cyclical_fcfe_scenario_meta`, added to the
`scenario_meta` chain for `sector_type == "financial"`. Its per-scenario
`growth` string reports the **effective** rate, with the assumed rate named
as discarded when they differ:

- capped: `"%4,6 büyüme (varsayılan %25,0 içsel finansman kısıtıyla sınırlandı)"`
- not capped: `"%X büyüme (kazanç + sürdürülebilir büyüme)"`

**`_cyclical_fcfe_scenario_meta`** gets the same treatment: its `growth`
string now leads with the effective rate instead of the uncapped
`growth_5y`. Its existing reinvestment-rate sentence is unchanged.

Because the growth label is derived per scenario, a band that varies only by
discount rate now says so on every row -- defect (2) above is fixed by the
same change that fixes (1).

### Scope

No fair-value NUMBER changes anywhere: `growth_path`/`growth_capped` are
observational, `external_growth` is advisory, and `scenario_meta` only
rewrites display strings. `mature`/`growth_unprofitable`/`reit`/hyper-grower
filers are untouched. The only behavior change for a filer whose growth was
never capped is that `growth_capped` reads `False`.

## 23. Tangible-equity diagnostics: ROTCE + P/TBV (WP7)

Two reported measures, no new valuation anchor. Motivation, and the boundary
this section deliberately does NOT cross, both come out of Sec.22's finding:

- **ROTCE is the right measure of the return on INCREMENTAL capital**, because
  new equity a bank deploys does not arrive with goodwill attached. That makes
  it a forecasting input and a quality diagnostic. It is NOT the right
  denominator for the RIM value base: residual income is accounting-invariant
  by construction (write book down by `G` and every future equity charge
  `r * B` falls by `r * G`, whose perpetuity PV is exactly `G`), so rebasing
  the anchor on tangible equity would move the number ONLY through the
  engine's 10-year truncation -- measured at `-$0.46/share` for SoFi against a
  predicted `-G/(1+r)^10 = -$0.57` artifact. Sec.8f's RIM therefore keeps
  using `StockholdersEquity`/`roe`.
- **P/TBV is the market convention for comparing banks**, precisely because
  P/B is not comparable across filers carrying different goodwill loads. That
  belongs in the multiples layer.

### 23a. New concepts (`normalize/concepts.py`)

```python
"Goodwill": ["Goodwill"],
"IntangibleAssets": [
    "IntangibleAssetsNetExcludingGoodwill",
    "FiniteLivedIntangibleAssetsNet",
],
```

Both are balance-sheet stocks (absent from `FLOW_CONCEPTS`, so
`STOCK_CONCEPTS` picks them up).

### 23b. Tangible equity + ROTCE (`normalize/ratios.py`)

Two additive keys on every per-fiscal-year ratio row:

```python
"tangible_equity": float | None,   # StockholdersEquity - Goodwill - IntangibleAssets
"rotce": float | None,             # NetIncome / tangible_equity
```

- A missing `Goodwill` or `IntangibleAssets` for that fiscal year is treated
  as `0.0`, the standard convention (an SEC filer with no goodwill simply does
  not tag it). Consequence to keep in mind: if a filer HAS goodwill but left
  it untagged for a year, `tangible_equity` reads too high, which
  **understates** `rotce` and **overstates** cheapness in `ptbv` (23c). The
  raw `tangible_equity` is exposed alongside so the figure is auditable
  rather than having to be trusted.
- `tangible_equity` is `None` when `StockholdersEquity` is missing for that
  year; `rotce` is `None` when either operand is missing or
  `tangible_equity <= 0` (goodwill exceeding book equity makes the ratio
  meaningless, not merely negative).
- **Naming caveat, deliberate:** the conventional term is *return on tangible
  COMMON equity*, which also deducts preferred equity. The normalized concept
  set carries no reliable preferred-stock line, so none is deducted. For a
  filer with material preferred this reads slightly high versus a strict
  ROTCE. The key keeps the recognizable name `rotce`; the docstring states the
  exact denominator.
- For a filer with no goodwill or intangibles at all, `tangible_equity ==
  StockholdersEquity` and `rotce == roe`. That is the correct answer, not a
  degenerate one -- which is why 23e shows the row only where it adds
  information.

### 23c. P/TBV (`normalize/metrics.py`, `valuation/multiples.py`)

`compute_metrics` gains three additive keys:

```python
"tangible_equity": float | None,    # latest fundamental FY's, from ratios
"tbv_per_share": float | None,      # tangible_equity / shares
"ptbv": float | None,               # price / tbv_per_share
```

All `None`-safe and all `None` when `tangible_equity <= 0` or shares/price
are unavailable, mirroring how `pb`-style figures already degrade.

`multiples.multiples_history` gains `"ptbv"` on each history row:
`fy_price * shares / tangible_equity_fy`, defined only for
`tangible_equity_fy > 0` and a usable share count -- same guard shape as the
existing `pffo`/`ps` entries. Tangible equity is derived inside
`multiples_history` from the `StockholdersEquity`/`Goodwill`/
`IntangibleAssets` annual series with the same zero-fill rule as 23b, so the
function keeps its existing `(normalized, price_df)` signature.

`engine._run_valuation` computes `ptbv_pct = multiples.percentile_position(
[h["ptbv"] for h in history], current["ptbv"])` and adds two additive keys to
`multiples_out`: `"ptbv_percentile"` and `current["ptbv"]`.
`_derive_current_multiples` seeds `current["ptbv"]` from `metrics["ptbv"]`
and, when that is `None`, back-fills it from the latest fiscal year with a
positive tangible equity -- the same fy-mismatch recovery it already does for
`pe`/`ps`/`pfcf`, with the same Turkish "derived from FYxxxx" note.
`_empty_valuation` gains both keys (`None` / `None`).

**Explicitly NOT changed:** P/TBV does not enter the sector-axis candidate
order, does not become any sector's PRIMARY own-history multiple, and does not
reach `triangulate.triangulate`. Promoting it would change the multiples
signal -- and therefore the verdict -- for every `financial` filer, which is a
separate decision from reporting the number. `financial` keeps its existing
P/E-primary order (Sec.10).

### 23d. RIM block carries the tangible figures (`engine._build_rim`)

`_build_rim`'s returned dict gains two additive, advisory keys so the reader
can see the tangible base next to the book base the anchor actually used:

```python
"tangible_equity": float | None,       # the SELECTED fiscal year's
"tbv_per_share": float | None,
"rotce": float | None,
```

They are read from the same `ratios` row and the same fiscal year the anchor
already selected (so they cannot describe a different year than `roe`/`bve0`),
and they never feed the projection -- see the invariance argument at the top
of this section.

### 23e. Display

- **`report/template.html`**: the "Önemli Oranlar" card gains a **ROTCE** row,
  rendered only when `sectorType === "financial"` (elsewhere it either
  duplicates ROE or describes a measure nobody reads for that sector). It sits
  directly under the existing ROE row so the goodwill drag is visible as a
  pair. A new muted `tangibleMultiplesLineHtml(valuation)` renders
  `P/TBV <value> (persentil <n>) · ROTCE <n>%` under the MULTIPLES
  triangulation method for `financial` -- the same slot, and the same visual
  treatment, that `evMultiplesLineHtml` occupies for other sectors and that
  Sec.20b emptied for this one.
- **`cli.py`**: `_tangible_multiples_line(valuation)` mirrors
  `_ev_multiples_line`'s shape ("Maddi çarpan: ..."), printed only for
  `financial` and only when a P/TBV figure exists, right where the suppressed
  "FD çarpanı:" line used to print.

### Scope

No fair-value number, anchor, triangulation weight, or verdict changes. Every
new key is additive; every new display element is gated on
`sector_type == "financial"` except the raw `rotce`/`tangible_equity`/`ptbv`
figures, which are computed for all filers and simply not rendered elsewhere.

## 24. Quarterly fiscal-year anchoring + TTM base (WP8)

**Problem this fixes (real, observed on MU 2026-07-30).** Two compounding
defects left a deep cyclical valued off a base three quarters out of date.

### 24a. Quarterly fiscal-year assignment — `normalizer._fiscal_year_end_month`

`_fiscal_year(period_end)` labels a period by the CALENDAR year of its end
date. For an annual record that is right (the filer's FY-end month is the
period end). For a QUARTER it is wrong whenever the fiscal year does not end
in December: Micron's fiscal year ends in late August, so its Q1 (ending
2025-11-27) lands in calendar 2025 while the rest of the same fiscal year
(Q2 2026-02, Q3 2026-05, Q4/annual 2026-08) lands in 2026.

`to_quarterly_series` groups by that label and then derives
`Q4 = annual - sum(first three)`. For MU's FY2025 the calendar-2025 group held
FY2025's Q2 (`8.05`) and Q3 (`9.30`) plus FY2026's Q1 (`13.64`), so it
computed `37.38 - 30.99 = 6.39` and emitted **$6.38B** as Q4 FY2025. The
correct figure is `37.38 - (8.71 + 8.05 + 9.30) = ` **$11.32B**.

Fix, contained entirely inside `to_quarterly_series`:

- `_fiscal_year_end_month(normalized) -> Optional[int]`: the most common
  month among ANNUAL record `period_end`s across all concepts (ties resolved
  toward the most recent). `None` when there are no annual records.
- `_quarter_fiscal_year(period_end, fy_end_month) -> Optional[int]`: with
  `M = fy_end_month`, a quarter ending in month `m` of calendar year `Y`
  belongs to fiscal year `Y` when `m <= M`, else `Y + 1`.
- `to_quarterly_series` uses this instead of the record's stored `fy`
  whenever a fiscal-year-end month is resolvable; otherwise it falls back to
  today's behavior.

**Backward compatible for the common case:** a December fiscal-year-end
(`M = 12`) makes `m <= 12` always true, so every quarter keeps its calendar
year and the output is byte-for-byte unchanged. Only non-December filers
change, and for them the current labels are wrong.

**Known limitation, documented not fixed:** a 52/53-week filer whose year-end
drifts across a month boundary (e.g. ending Jan 28 one year and Feb 2 the
next) can still be misassigned by one bucket. The month rule is a large
improvement over the calendar rule, not a calendar-exact reconstruction.

The stored `fy` on individual quarterly RECORDS is left alone; only the
grouping/accessor is corrected, keeping the change to one function.

### 24b. TTM figures — `metrics.compute_metrics`

The valuation path reads annual series only, so a filer mid-fiscal-year is
valued on data that can be three quarters stale. For MU on 2026-07-30 that is
not a rounding issue:

| | latest FY (FY2025) | true TTM |
|---|---|---|
| revenue | `$37.38B` | `$90.28B` |
| net income | `$8.54B` | `$50.47B` |
| net margin | 22.8% | 55.9% |
| EPS | `$7.59` | `$44.86` |
| P/E at `$739` | **97.4** (reported) | **16.5** |

`compute_metrics` gains additive keys, all `None`-safe:

```python
"ttm_revenue": float | None,
"ttm_net_income": float | None,
"ttm_eps": float | None,          # ttm_net_income / shares
"ttm_net_margin": float | None,
"ttm_period_end": str | None,     # the newest quarter in the window
"ttm_quarters": int,              # how many quarters the window found (<= 4)
"ttm_complete": bool,             # exactly 4 quarters
"pe_ttm": float | None,           # price / ttm_eps, only when complete
"ttm_vs_fy_net_income": float | None,   # ttm_net_income / latest-FY net income
```

Built from `to_quarterly_series(normalized, "Revenue"/"NetIncome")` -- so it
inherits 24a's corrected grouping -- taking the newest 4 quarters. An
incomplete window (`ttm_quarters < 4`) leaves `ttm_*` populated but
`ttm_complete` `False` and `pe_ttm` `None`, since a partial sum is not a
trailing-twelve-month figure.

**`pe`/`ps`/`pfcf` are NOT redefined.** They keep their existing
latest-fiscal-year basis so every percentile, history comparison and stored
verdict stays comparable. `pe_ttm` is reported ALONGSIDE.

### 24c. Stale-base disclosure (`engine._run_valuation`)

New constant `_TTM_STALENESS_THRESHOLD = 0.25`. When `ttm_complete` and the
latest-FY net income is positive and
`abs(ttm_vs_fy_net_income - 1) > _TTM_STALENESS_THRESHOLD`, the engine
appends a Turkish note naming both bases, both P/E readings, and the fact
that the anchors below are built on the fiscal-year figure. Applies to every
sector -- a stale base is not a cyclical-only problem, it is just worst there.

`cli` prints the TTM P/E next to the FY P/E on the multiples line when they
diverge; `report/template.html` does the same in its multiples sub-line.

## 25. Through-cycle statistics + historical P/B band (WP9) — `valuation/cyclical.py`

New module. Pure, deterministic, never raises; no new dependency.

### 25a. `multiples_history` gains `pb`

`fy_price * shares / StockholdersEquity`, defined only for a strictly
positive book value and a usable share count -- same guard shape as the
existing `ps`/`pffo`/`ptbv` entries. This is what makes a historical P/B band
possible at all; before this the engine carried no P/B history of any kind
(`_build_pb_roe`'s justified P/B is a forward-looking construct, not an
observed multiple).

### 25b. `cyclical.through_cycle_stats(normalized, ratios, price_df, metrics) -> Optional[dict]`

```python
{
  "years": int,                       # fiscal years with a usable net margin
  "margin_mean": float,               # through-cycle average net margin
  "margin_median": float,
  "margin_trough": float,             # worst fiscal year
  "margin_trough_fy": int,
  "margin_peak": float,
  "margin_peak_fy": int,
  "margin_current": float | None,     # TTM when complete, else latest FY
  "margin_current_basis": "ttm" | "fy",
  "margin_percentile": float | None,  # current within the historical set
  "pb_trough": float | None, "pb_median": float | None, "pb_peak": float | None,
  "pb_current": float | None,
  "pe_median": float | None,          # through-cycle median annual P/E
  "book_value_per_share": float | None,
  "normalized_revenue": float | None, # see below
  "revenue_basis_years": [int, ...],
}
```

- Margins come from `ratios`' per-fiscal-year `net_margin`; at least
  `_MIN_CYCLE_YEARS = 6` usable years are required (a "through-cycle"
  statistic from fewer years is not one), else the function returns `None`.
- `margin_percentile` reuses `multiples.percentile_position`, so the
  "where in the cycle are we" reading uses the same midrank convention as
  every other percentile in the engine.
- P/B band from `multiples_history`'s new `pb` column: min / median / max.
- `pe_median` is the median of the history's `pe` column (positive entries
  only -- a loss year has no meaningful P/E).
- `normalized_revenue`: mean of the last three fiscal years' revenue, with
  **TTM revenue substituted for the most recent point when the TTM window is
  complete and newer** -- otherwise a filer three quarters into a violent
  upswing gets a "normalized" base that predates the upswing entirely. The
  fiscal years actually used are reported in `revenue_basis_years`.
  Deliberately NO bit-growth/unit-growth uplift: that is a judgment input
  this layer cannot derive, and omitting it is the conservative direction.

## 26. Two-regime cyclical valuation + implied break probability (WP10)

`cyclical.two_regime_valuation(stats, shares, price) -> Optional[dict]`, wired
into the engine as an **advisory block** for `sector_type == "cyclical"`.

**Advisory, exactly like Sec.8g/8j/8k:** it never headlines
`fair_value_range`, never feeds `primary_dcf_scenarios`, never enters
triangulation, and never changes a verdict. Promoting it to the cyclical
headline is a deliberate follow-up decision, not a side effect of adding it.

### 26a. Regime A -- mean reversion

- `normalized_eps_a = margin_mean * normalized_revenue / shares`
- `center = min(normalized_eps_a * pe_median, pb_median * book_value_per_share)`
  -- the prompt's "compute both, take the lower" rule; whichever leg is
  missing, the other stands alone.
- `low = pb_trough * book_value_per_share`
- `high = pb_peak * book_value_per_share`
- `trough_eps` (`margin_trough * normalized_revenue / shares`) is reported as
  a diagnostic input.

**Interpretation note, flagged rather than silently resolved:** the source
prompt defines the low end as "trough-margin normalized EPS x trough P/B
band", which mixes an EPS with a book multiple and is not dimensionally
sound (and a trough-earnings P/E is meaningless -- it explodes as earnings
approach zero). This section implements the low end as the **trough P/B
applied to current book value per share** -- the multiple the market has
actually paid at previous cycle bottoms -- and reports `trough_eps`
separately. The choice is recorded here so it can be revisited.

### 26b. Regime B -- structural break

Judgment inputs, module constants, defaults taken from the source prompt and
**documented as assumptions, not derivations**:

```python
_REGIME_B_MARGINS = (0.30, 0.38, 0.45)   # low / mid / high "new normal"
_REGIME_B_MULTIPLES = (12.0, 13.5, 15.0) # below the semis median: still capital-intensive
```

- Revenue base is **TTM revenue** (`stats["normalized_revenue"]` is the
  mean-reverting base; Regime B is the "this level persists" thesis, so it
  uses the current run-rate). `None` when the TTM window is incomplete.
- `eps_b[m] = margin_m * ttm_revenue / shares` for each of the three margins.
- `sensitivity`: the full 3x3 `margin x multiple` grid.
- `low` / `center` / `high` = `(margin_low, mult_low)`, `(margin_mid,
  mult_mid)`, `(margin_high, mult_high)` corners of that grid.

### 26c. Blend and implied probability

- `blend[p] = p * fv_b_center + (1 - p) * fv_a_center` for
  `p in (0.25, 0.50, 0.75)`.
- `p_implied = (price - fv_a_center) / (fv_b_center - fv_a_center)`, `None`
  when the denominator is zero or either center is missing.
- `p_implied_status`: `"ok"` when in `[0, 1]`; `"above_range"` when `> 1`
  (the price exceeds even the full structural-break value -- no probability
  mix explains it); `"below_range"` when `< 0`. The raw value is reported
  either way, never clamped.
- Turkish verdict sentence in the prompt's required shape: "Mevcut fiyat,
  yapisal kirilima >= %X olasilik vermeyi zorunlu kiliyor. Kendi p tahminin
  bunun ustunde ise fiyat ucuz, altinda ise pahali." With an
  `above_range`/`below_range` status the sentence instead states that no
  probability in [0,1] reconciles the price with the two regimes.

### 26d. Flags

- `peak_cycle_pe_trap`: current net margin at or above the
  `_PEAK_MARGIN_PERCENTILE = 80`th percentile of its own history AND
  `pe_ttm` below `_PEAK_PE_MAX = 15`. This is the trap the existing
  `red_flags._check_cyclical_trap` aims at, but that rule reads
  `metrics["pe"]` -- the stale fiscal-year P/E -- so for MU it saw 97.4 and
  stayed silent while the true TTM P/E was 16.5. This flag reads `pe_ttm`.
- `peak_annualization`: the newest quarter's net margin exceeds the
  through-cycle mean by more than `_PEAK_ANNUALIZATION_SPREAD = 20pp`,
  i.e. any forward figure built by annualizing it is a peak extrapolation.
- `regime_change_premium`: `pb_current > pb_peak`, the market already paying
  above every historical cycle top.

### 26e. Output shape and display

`valuation["cycle"]` (`None` for every non-cyclical sector and whenever
`through_cycle_stats` returns `None`):

```python
{"stats": {...}, "regime_a": {...}, "regime_b": {...},
 "blend": {...}, "p_implied": float | None, "p_implied_status": str,
 "verdict_sentence": str, "flags": {...}}
```

`_empty_valuation` gains `"cycle": None`. The CLI prints a compact
cycle-position + p_implied block; `report/template.html` renders a "Döngü
Konumu" card with the regime bands, the 3x3 sensitivity grid, the p-table and
the flags.

### Scope

WP8 changes numbers only where they were wrong (non-December filers'
quarterly grouping). WP9 and WP10 add reported figures and one advisory
block; no anchor, `fair_value_range`, triangulation weight or verdict changes
for any filer.

## 27. Log-linear trend growth replaces two-endpoint CAGR (WP11) — `metrics._trend_growth`

**Problem this fixes (measured, 2026-07-31 calibration).** `metrics._cagr`
computed `(latest / latest_fy-N) ** (1/N) - 1` -- two points, so whatever
happened in the single starting year drives the whole estimate. With the
window widened to 12 years (Sec.24) the 5-year figure became computable for
the first time and immediately landed its start point on FY2020, the COVID
trough. Filers whose 2020 was depressed then read as growers while actually
shrinking:

| | FY2020 | FY2025 | 3y CAGR | 5y endpoint CAGR |
|---|---|---|---|---|
| PFE | `$41.7B` | `$62.6B` | −12.0% | **+8.5%** |
| CVX | `$94.5B` | `$184.4B` | −7.9% | **+14.3%** |
| DE | `$35.5B` | `$45.7B` | −4.6% | +5.2% |

Pfizer's revenue has fallen from `$91.8B` (FY2022) to `$62.6B`, and the
estimator called it an 8.5% grower. The calibration basket's upper tail blew
out accordingly (PFE fair-value/price 1.12 → 3.64, CVX 0.94 → 2.52; the
`>1.2` bucket doubled from 4 to 8 names).

### `_trend_growth(series, latest_fy, years) -> Optional[float]`

Replaces `_cagr` as the estimator behind `revenue_cagr_3y`/`revenue_cagr_5y`.
Ordinary least squares on the natural log of the series over the window:

- Window = fiscal years in `[latest_fy - years, latest_fy]` inclusive (up to
  `years + 1` points).
- Only strictly positive values participate (the log is undefined otherwise);
  non-positive years are dropped, not zero-filled.
- `latest_fy` itself must be present. This is deliberately asymmetric: the
  START point is the one the old estimator over-weighted, while the END point
  is what anchors the figure to the present, and every other metric in this
  module is already read at `latest_fundamental_fy`.
- At least `_MIN_TREND_POINTS = 3` usable points, spanning at least 2 fiscal
  years. With only 2 points a least-squares line IS the two-endpoint line, so
  it would add nothing while pretending to be robust.
- Fit `ln(value) = a + b * fy`; return `exp(b) - 1`, the annualized trend
  growth rate.
- Returns `None` when any condition above fails. Never raises.

**Key names and meaning are unchanged.** `revenue_cagr_3y`/`revenue_cagr_5y`
still mean "annualized realized revenue growth over N years" -- only the
estimator changed -- so every consumer (`rule_based`'s growth anchor,
`sector.detect_hyper_grower`'s trigger, `classify_sector`'s SIC-3674 branch,
`engine`'s realized-CAGR reference, `planning`'s thesis metric) is unaffected
in shape and reads a better number.

### Measured effect

Robust where the series is well-behaved, corrective where it is distorted --
which is the whole property being bought:

| | endpoint 5y | log-linear | delta |
|---|---|---|---|
| PFE | +8.5% | **+2.9%** | −5.6pp |
| MU | +11.8% | **+5.3%** | −6.5pp |
| CVX | +14.3% | **+11.5%** | −2.8pp |
| AAPL | +8.7% | **+6.6%** | −2.0pp |
| CAT | +10.1% | +9.7% | −0.4pp |
| DE | +5.2% | +5.6% | +0.4pp |

**Residual limitation, documented not fixed:** CVX stays at 11.5% because its
FY2020 is an outlier deep enough to tilt even a six-point trend. A trend line
is robust to endpoint NOISE, not to a genuine structural outlier inside the
window. A median-of-year-over-year (Theil-Sen style) estimator would suppress
it further, at the cost of discarding the compounding information a trend
keeps; that trade-off was not taken here.

### Scope

One estimator swap inside `normalize/metrics.py`. No output key added,
removed or renamed; no valuation formula, anchor or threshold changed. Filers
whose revenue series is monotone see essentially no movement.
