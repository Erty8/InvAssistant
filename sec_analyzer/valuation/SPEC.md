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
  `sbc_revenue`, `shares_yoy`, `fcf`, `latest_fy`, ...).
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

Return a list of human-readable violation strings (empty = OK). Rules, per
scenario:
- `terminal_growth > 0.04` → violation
- `discount_rate < 0.07` (or `< 0.10` when `is_unprofitable`) → violation
- `discount_rate <= terminal_growth` → violation (Gordon undefined — never
  silently "fix" it)
- `growth_5y > 0.20` is allowed only because the model structure always fades
  after year 5 (total high-growth span ≤ 7y is satisfied by design); still add
  violation if `growth_5y > 0.40` (implausible).
- Missing/non-numeric field → violation naming the field.

### Clamping — `sanity.clamp_assumptions(assumptions, is_unprofitable: bool = False) -> tuple[dict, list[str]]` (F5)

Unlike `validate_assumptions` above (report-only), this actually rewrites
out-of-range values so every downstream calculation uses the same numbers
shown to the user. Per scenario: `terminal_growth` capped at 0.04; `growth_5y`
capped at 0.40; `discount_rate` floored at 0.07 (0.10 if `is_unprofitable`) —
each clamp appends a Turkish note. The `discount_rate <= terminal_growth` case
is deliberately NOT clamped (the existing per-scenario `ValueError` path stays
the way it's surfaced). A missing/non-numeric field is left untouched. Also
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
`revenue_dcf.revenue_first_dcf`'s FCF-margin-derived cash flows.

### fcf0 selection (engine responsibility, SBC-adjusted)
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
- 6000–6999 (except 6798) → `"financial"`
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
- else if latest-FY `NetIncome < 0` → `"growth_unprofitable"`
- else → `"mature"`
If SIC missing → fall back to the LLM's phase-1 `sector_type` (engine wiring),
else `"mature"`.

### Sector → method adjustments (engine)
- `financial`/`reit`: FCF-DCF disabled (`dcf.enabled = False`, Turkish
  `disabled_reason`). Compute a P/B×ROE anchor instead:
  `fair_pb = clamp(roe / discount_rate_base, 0.5, 4.0)`,
  `per_share = fair_pb * (equity_latest / shares)`; band from a
  `discount_rate_base ± 1pp` sensitivity re-clamp (±10% fallback, Sec.4);
  bear/base/bull scale `fair_pb` by (0.8 / 1.0 / 1.2). Output under key
  `"pb_roe"` mirroring the dcf scenario shape. Hyper-grower detection is
  never attempted for these two sectors (see the cross-reference below).
- `growth_unprofitable`: DCF still attempted (fcf may be negative → note),
  multiples use P/S only (pe/pfcf percentiles likely None), triangulation
  weights reverse-DCF + P/S.
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
P/B×ROE instead.

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

## 10. Triangulation — `triangulate.triangulate(price, dcf_base_band, implied_growth, realized_cagr, base_growth, pe_pct, ps_pct, pfcf_pct, sector_type, hyper_growth=False, bull_band=None, reverse_dcf_status=None) -> dict`

Direction signal per method (`"ucuz" | "makul" | "pahali" | "veri_yok"`):
- **DCF**: price < band.lo → ucuz; price > band.hi → pahali; else makul.
  (For financial/reit use the pb_roe base band.)
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
  growth_unprofitable use ps first). pct > 70 → pahali; pct < 30 → ucuz; else
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
  "pb_roe": {"scenarios": {...}}|None,
  "fair_value_range": <shape from §4, built from dcf.scenarios or pb_roe;
                        for cyclical sector_type, from dcf.normalized_variant
                        instead when available -- see §8; overridden by the
                        hyper-grower revenue-first DCF base band, ahead of
                        both, whenever hyper_growth is true -- see §3 below>,
  "reverse_dcf": {"implied_growth": float|None, "realized_cagr_5y": float|None,
                   "realized_label": "FCF 5y"|"FCF 3y"|"gelir 5y"|"gelir 3y"|None,
                   "bracket_status": "ok"|"above_bracket"|"below_bracket"|"no_data"},
                  # standard mode: FCF-CAGR reference (Sec.5/F6); hyper-grower
                  # mode: revenue-CAGR reference + revenue_first_dcf's own
                  # implied start-growth (Sec.5/F6); bracket_status from
                  # reverse_dcf.implied_growth_with_status (standard mode) or
                  # a fixed "ok" (hyper-grower mode, see Sec.5).
  "multiples": {"history": [...], "current": {"pe","ps","pfcf"},
                 "pe_percentile", "ps_percentile", "pfcf_percentile",
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
     "notes": [str],
  },
  "assumptions": <the validated AND CLAMPED assumptions dict (Sec.3's
                   clamp_assumptions, F5) -- what's shown here is exactly
                   what every DCF/reverse-DCF/sensitivity/hyper calculation
                   above used>,
  "notes": [str, ...],   # Turkish, e.g. fcf0 fallback, missing Damodaran files,
                          # assumption-clamp notes, reverse-DCF bracket notes
}
```

### Hyper-grower revenue-first DCF (Sec.1/Sec.3, engine wiring)

Independently of `sector_type` (a filer can be `growth_unprofitable` or
`mature` and still trip this), the engine calls
`sector.detect_hyper_grower(metrics, ratios, normalized)` at the top of
`_run_valuation` -- EXCEPT for `sector_type in ("financial", "reit")`, where
hyper-grower detection is skipped entirely (F4: forced `is_hyper_grower =
False`; a revenue-margin hyper-DCF doesn't make sense for those sectors,
which use P/B×ROE instead). When it triggers, `engine._build_hyper_growth`
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
latest_yoy` -- whenever `latest_yoy` is computable (both `latest_fy` and
`latest_fy - 1` revenue positive); otherwise `growth_anchor = realized_cagr`
alone (a smoothed 5y/3y CAGR can otherwise lag a hyper-grower's own most
recent, and often materially different, growth rate). A Turkish note is added
whenever the blend is actually used.

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

Display consistency: in hyper-grower mode, `fair_value_range`'s per-scenario
`growth`/`discount_rate`/`note` fields are also switched over to reflect the
revenue-first DCF's own scenario inputs (`engine._hyper_scenario_meta`) --
NOT the standard clamped `assumptions[scenario]` the headline band no
longer actually uses. Concretely, for each scenario whose hyper cell has a
`start_growth`/`target_fcf_margin`: `growth` = `"%<start_growth> başlangıç
→ %2.5 terminale fade"`, `discount_rate` = the fixed hyper per-scenario rate
(`"%12"`/`"%10"`/`"%9"` for bear/base/bull), and `note` names the scenario
(`"kötümser"`/`"temel"`/`"iyimser"`), the start growth, the fade, the mature
target FCF margin, and the discount rate, e.g. `"Hiper-büyüme temel:
başlangıç büyüme %40 (10 yılda %2.5 terminale fade), olgun FCF marjı %30,
iskonto %10."` A scenario missing its hyper cell (a failed
`revenue_first_dcf` call for that scenario) falls back to the
assumptions-derived string for that scenario, exactly as the non-hyper path
always has.
Round all per-share values to 2 decimals, percentiles to 1, growth rates to 4.

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
