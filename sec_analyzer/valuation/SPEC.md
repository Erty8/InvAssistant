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
  violation if `growth_5y > 0.40` (implausible).
- Missing/non-numeric field → violation naming the field.

### Clamping — `sanity.clamp_assumptions(assumptions, is_unprofitable: bool = False) -> tuple[dict, list[str]]` (F5)

Unlike `validate_assumptions` above (report-only), this actually rewrites
out-of-range values so every downstream calculation uses the same numbers
shown to the user. Per scenario: `terminal_growth` capped at 0.04; `growth_5y`
capped at 0.40; `discount_rate` floored at 0.07 (0.10 if `is_unprofitable`) —
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
non-positive, or deviates more than ±50% from the 3-year average (also
SBC-adjusted) FCF, use the 3-year average instead and set `fcf0_source =
"3y_avg"` plus a Turkish note; else `fcf0_source = "ttm"`. If no positive fcf0
can be derived at all, DCF returns `None` per-share values with a note (do not
raise). This same SBC-adjusted per-FY series is also the source for the
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

Bisection on `growth_5y` over [-0.20, 0.40] so that
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
ters-DCF aralığının (%-20..%40) üzerinde/altında bir büyüme ima ediyor.") and
is threaded into `triangulate.triangulate(..., reverse_dcf_status=...)` so the
reverse-DCF signal can be "pahalı"/"ucuz" even when `implied_growth` is `None`.

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

Returns `[{"fy", "end", "price", "pe", "ps", "pfcf", "ev_sales"}, ...]` sorted
by fy.

`multiples.percentile_position(history_values: list[float], current: float) ->
Optional[float]` — percentage (0–100) of historical values strictly less than
`current`, plus half of ties (midrank). Requires ≥5 non-None historical values,
else `None`.

Current multiples come from `metrics["pe"|"ps"|"pfcf"]`.

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
    `fair_pb = clamp((roe - g) / (discount_rate_base - g), 0.5, 4.0)`, where
    `g` is the base scenario's `terminal_growth` (degrading to the no-growth
    `roe / discount_rate_base` form when `g` is missing, negative, or would
    make the denominator non-positive), `per_share = fair_pb * (equity_latest
    / shares)`; band from a `discount_rate_base ± 1pp` sensitivity re-clamp
    with `g` held fixed across the band (±10% fallback, Sec.4); bear/base/bull
    scale `fair_pb` by (0.8 / 1.0 / 1.2). Output under key `"pb_roe"`
    mirroring the dcf scenario shape.
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
  same scenario shape. Both variants are reported side by side. When
  `normalized_variant` was successfully computed, the headline
  `fair_value_range`, the triangulation DCF band, AND the reported
  `sensitivity` matrix (Sec.9) are all taken from it instead of the raw
  FCF-DCF band/fcf0; otherwise all three fall back to the raw FCF-DCF
  band/fcf0 as usual.

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
priority chain:

```python
earnings_power = None
if sector_type == "mature" and not hyper_growth_active:
    earnings_power, ep_notes = _build_earnings_power(assumptions, normalized, metrics, ratios)
```

The priority chain (Sec.3/Sec.8's existing hyper-grower branch, then the
cyclical `normalized_variant` branch) gains a new trailing `elif` branch, so
both hyper-grower mode and the cyclical normalized-earnings variant still
take precedence over EPV:

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
- `target = min(nopat, hist_anchor, _MATURE_TARGET_CAP)` over whichever of
  `nopat`/`hist_anchor` are available; `None` only when **both** are
  unavailable (the method can't be built without at least one anchor).
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
   "current_margin", "steady_state_year"}, notes)`.

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
kept. Why the guardrail compares against EPV specifically: EPV is a
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

## 10. Triangulation — `triangulate.triangulate(price, dcf_base_band, implied_growth, realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type, hyper_growth=False, bull_band=None, reverse_dcf_status=None, raw_growth_pair_pct=None, growth_adj_pct=None, earnings_power_headline=False, mature_revenue_headline=False, midgrowth_revenue_headline=False, pffo_pct=None) -> dict`

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
  Sec.8c -- never pe). pct > 70 → pahali; pct < 30 → ucuz; else
  makul. **Two-component (VALUATION.md Sec.7):** when both `raw_growth_pair_pct`
  (the raw multiple's percentile — P/E in standard mode, EV/Sales in
  hyper-grower mode) and `growth_adj_pct` (the growth-adjusted multiple's
  percentile) are present AND fall in different directional buckets, the signal
  becomes `"karisik"` (mixed); when they agree, or either is `None`, the raw
  signal above stands unchanged. `karisik` is a substantive signal (not
  `veri_yok`), so it naturally can't join a pahali/ucuz/makul majority — it
  lowers confidence exactly as a genuine disagreement should.

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
  },
  "pb_roe": {"scenarios": {...}}|None,  # financial's anchor; also reit's
                                         # FALLBACK anchor when ffo (below)
                                         # couldn't be built -- Sec.8c.
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
                        instead when available -- see §8; overridden by the
                        hyper-grower revenue-first DCF base band, ahead of
                        both, whenever hyper_growth is true -- see §3 below;
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
  "multiples": {"history": [...], "current": {"pe","ps","pfcf","pffo"},
                 "pe_percentile", "ps_percentile", "pfcf_percentile",
                 "pffo_percentile",  # Sec.8c; reit's primary multiples signal input
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
     "capex_normalization": None | { # Sec.3.6; None unless the maintenance/growth
        "applied": True,             # CapEx split was applied (capex-heavy filer)
        "capex_intensity": float,    # CapEx / revenue
        "maintenance_capex": float,  # max(D&A, 5% of revenue) -- floored proxy
        "growth_capex": float,       # capex - maintenance_capex
        "raw_current_margin": float, # actual starting margin (drives the HEADLINE)
        "ops_current_margin": float, # relieved margin (drives the UPSIDE only)
        "upside_per_share": float|None,  # AGGRESSIVE upside base value, NOT headlined
        "upside_lo": float|None,     # upside base band
        "upside_hi": float|None,
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
     "current_margin": float,        # 3y-median SBC-adjusted FCF margin (fade start point)
     "steady_state_year": int,       # 7 (_MATURE_STEADY_STATE_YEAR)
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
     "target_margin_base": float,    # mature target FCF margin (<= _MIDGROWTH_TARGET_CAP 20%)
     "current_margin": float,        # 3y-median SBC-adjusted FCF margin (fade start point)
     "steady_state_year": int,       # 8 (_MIDGROWTH_STEADY_STATE_YEAR)
     "annual_dilution": float,       # clamp(shares_yoy if > 0 else 0, 0, 0.05)
     "financing_shares": float,      # cumulative-burn / price (hyper-style)
     "suppressed": bool,             # True when base per_share <= 0 (falls back to multiples)
  },                                 # Sec.8d; built (attempted) for growth_unprofitable
                                     # non-hyper filers; None when the growth gate rejected
                                     # it or a precondition was missing.
  "assumptions": <the validated AND CLAMPED assumptions dict (Sec.3's
                   clamp_assumptions, F5) -- what's shown here is exactly
                   what every DCF/reverse-DCF/sensitivity/hyper calculation
                   above used>,
  "notes": [str, ...],   # Turkish, e.g. fcf0 fallback, missing Damodaran files,
                          # assumption-clamp notes, reverse-DCF bracket notes,
                          # EPV headline switch/margin-normalization/quality
                          # notes (Sec.8a), mature revenue-first DCF headline
                          # switch/growth-gate/guardrail notes (Sec.8b)
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
table, Sec.3.2's dilution/financing rule -- F2: dilution is share-count
growth only, `clamp(shares_yoy if > 0 else 0, 0, 0.05)`; SBC is expensed
directly into `current_margin`/target margins instead of also inflating
dilution), computes a prob-weighted
`expected_value`, an "arrival point" flag from the base scenario's 10-year
revenue multiple (or from `tam_usd`'s share of that revenue when known --
this overrides the multiple-based flag), and the price's implied
start-growth/target-margin (`revenue_dcf.implied_start_growth` /
`implied_target_margin`). `hyper_growth_extras` (Sec.5) can override each
scenario's `target_fcf_margin`/`steady_state_year`/`probability` and supply
`tam_usd`; anything not overridden stays deterministic.

Start-growth anchor (F4): the base scenario's start-growth is
`min(growth_anchor, 0.40)` (bear/bull scale this by 0.6x/1.2x before the same
cap), where `growth_anchor` blends the realized multi-year CAGR with the
latest single fiscal year's YoY growth -- `0.5 * realized_cagr + 0.5 *
latest_yoy` -- whenever `latest_yoy` is computable (both the fundamental
fiscal year, `resolve_fundamental_fy(metrics)`, and the year before it have
positive revenue); otherwise `growth_anchor = realized_cagr` alone (a
smoothed 5y/3y CAGR can otherwise lag a hyper-grower's own most recent, and
often materially different, growth rate). A Turkish note is added whenever
the blend is actually used.

Sec.3.1's `target_base` (the mature-state FCF-margin ceiling that the
base-scenario `target_fcf_margin` equals, with bear/bull scaling it by
0.7/1.2) is `min(gross_margin * 0.5, 0.30)` when the latest-FY gross
margin is a known positive number, or a flat `0.20` ceiling when gross
margin is unavailable (replacing the previous 15%-gross-margin-fallback
rule, which produced an unrealistically low 7.5% ceiling for filers with
no gross-margin data at all). Either way, `target_base` is then floored
at today's FCF margin (`fcf / latest_revenue`) whenever that margin is
positive -- a filer that is already FCF-profitable today must never be
modeled as if its mature margin collapses below what it already earns --
and, when gross margin is known, capped back down at that gross margin
(so the floor can raise `target_base` but never push it past the
gross-margin ceiling). `target_margin_source` reports which of these
paths fired, e.g. `"brüt marj × 0.5 (tavan %30)"` when the gross-margin
ceiling applied unchanged, `"brüt marj %60 × 0.5 (tavan %30), bugünkü FCF
marjına tabanlanmış"` when a known gross margin was overridden by the
current-margin floor, or `"brüt marj yok: %20 varsayılan tavan, bugünkü
FCF marjına (%30) tabanlanmış"` when gross margin was missing and the
20% default ceiling was overridden by the current-margin floor.

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

`_maintenance_adjusted_margin(normalized, metrics, raw_current_margin) ->
(ops_margin, capex_normalization | None)` computes the growth-CapEx-relieved
margin, using depreciation & amortization (the `Depreciation` concept),
**floored at `_MAINTENANCE_CAPEX_MIN_PCT_REVENUE` (= 0.05) of revenue**, as
the maintenance-CapEx proxy. The revenue floor exists because current-year
D&A understates the maintenance burden of a still-ramping asset base
(reviewer Finding 2: a data-center builder's future depreciation reflects
its grown-out fleet, not today's small one). All figures are read at
`resolve_fundamental_fy(metrics)`. Gate — BOTH must hold, else the raw margin
is returned unchanged and `capex_normalization` is `None`:

- `capex / revenue > _CAPEX_HEAVY_INTENSITY_THRESHOLD` (new constant `= 0.30`)
  — genuinely capex-heavy, not an asset-light software grower.
- `capex > max(d&a, 0.05·revenue)` — there is growth CapEx above the floored
  maintenance level to relieve.

When applied: `maintenance_capex = max(d&a, 0.05·revenue)`, `growth_capex =
capex − maintenance_capex`, `ops_margin = raw_current_margin + growth_capex /
revenue` (an additive correction on the caller's raw margin).

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
- **Target mature FCF margin:** `min(_hyper_target_base(gm, current_margin),
  _MIDGROWTH_TARGET_CAP)` where `gm` = latest-FY positive gross margin. The
  gross-margin construction (hyper path) is used rather than the mature
  path's operating-margin/historical-FCF anchors, which degenerate for a
  loss-maker with no positive-margin history. Capped at 20% (between mature's
  15% and hyper's 30%).
- **Current (starting) margin:** `_mature_current_margin(...)` (3-year median,
  negative for loss-makers). The Sec.3.6 CapEx relief is deliberately NOT
  applied here — this path aims for a defensible value, so a capex-heavy
  mid-grower whose base value suppresses falls back to multiples instead.
- **Fade horizon:** `_MIDGROWTH_STEADY_STATE_YEAR` (8).
- **Dilution & financing shares:** hyper-style — `annual_dilution =
  clamp(shares_yoy if > 0 else 0, 0, 0.05)` and `financing_shares` derived
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
  "start_growth", "target_margin_base", "current_margin",
  "steady_state_year", "annual_dilution", "financing_shares",
  "suppressed"}`, or `(None, notes)`.

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
       descending trigger price:
       ```python
       [{"n": 1, "trigger": "Günlük kapanış 180.00 USD seviyesinin altına "
                              "inerse (bölge 177.30-182.70 USD); gün içi "
                              "dokunuş tetik saymaz.",
         "price_zone": {"lo": 177.30, "hi": 182.70}, "size_pct": 10.0,
         "invalidation": 142.50, "target": 250.0, "rr": 2.3, "note": None},
        ...]
       ```
       Candidate trigger levels are pulled ONLY from already-computed figures
       — `fair_value_range`'s `bear.lo`/`base.lo`/`base.hi`/`bull.hi` plus the
       technical read's `low_52w`/`sma50`/`sma200` — filtered to levels at or
       below the current price, deduplicated when two levels sit within 2% of
       each other, sorted descending, capped at 5. A single shared
       `invalidation` (a fixed buffer below the lower of `bear.lo`/`low_52w`)
       and a single shared `target` (`bull.hi`, else `base.hi`) apply to every
       tranche, so R:R is mathematically non-decreasing as price falls
       (lower entry → larger reward, smaller risk); `rr` folds in a
       round-trip transaction cost (METODOLOJI.md §2). `trigger` text is
       Turkish and explicitly daily-close-only — an intraday touch never
       counts. `[]` when price is missing/non-positive, or no candidate level
       sits at or below the current price; fewer than 3 tranches is possible
       (never fabricated) when fewer than 3 distinct levels survive
       filtering/dedup.
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
       str|None, "trend": str|None, "rationale": str}`. `trend` is one of
       `"iyileşiyor"` / `"bozuluyor"` / `"yatay"`, or `None` if no prior
       fiscal year is available to compare against. The anchor metric is
       chosen from `valuation["sector_type"]` via a fixed sector→metric map
       (`mature`→net margin, falling back to ROE; `growth_unprofitable`→YoY
       revenue growth; `financial`→ROE as a NIM proxy; `reit`→FCF margin as
       an FFO proxy; `cyclical`→gross margin, falling back to net margin;
       unrecognized/`None`→net margin), read from `ratios`/`metrics` and
       never fabricated — `latest_value` is `None` (with `rationale` saying
       so) when the chosen metric isn't computable from the given inputs.
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
