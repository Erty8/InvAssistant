---
name: valuation-sanity
description: Validate a produced sec_analyzer valuation output against the valuation-theory invariants and SPEC bounds before trusting it. Use after running an analysis (or when reviewing a stored verdict) to confirm the fair-value numbers, assumptions, and triangulation are internally coherent and not silently broken. Triggers — "is this valuation sane?", "check the DCF output", "validate the fair value", "sanity-check TICKER's analysis".
---

# Valuation sanity check

Goal: confirm a `sec_analyzer` valuation output obeys the non-negotiable rules in
`sec_analyzer/valuation/SPEC.md` and `sec_analyzer/METODOLOJI.md`. This is a
mechanical correctness gate, **not** investment judgment — it says "the numbers
are self-consistent", never "this is a good buy".

## 1. Get the structured valuation dict

Prefer real output over eyeballing the HTML. Two ways:

**A — from a fresh offline run (deterministic, no API/LLM):**
```
python -m sec_analyzer.cli analyze TICKER --horizon 1y --provider script --html --years 12
```
Then read the persisted structured dict from the store (canonical source, SPEC §14):
```python
import json, sqlite3
con = sqlite3.connect("sec_analyzer/sec_data.sqlite3")
row = con.execute(
    "SELECT ticker, fair_value_json, valuation_json, confidence, sector_type "
    "FROM verdicts WHERE ticker=? ORDER BY id DESC LIMIT 1", ("TICKER",)
).fetchone()
val = json.loads(row[2])   # the full valuation dict (valuation_json)
```

**B — reviewing an existing stored verdict:** skip the run, read the latest
`valuation_json` for the ticker as above.

Write any check helper to the scratchpad, not the repo.

## 2. Assert the invariants

Walk the checklist below against `val`. Report every violation with the offending
number and the SPEC/METODOLOJI section; a clean run explicitly states "no
violations". Do NOT fix anything from this skill — surface findings; if code is
at fault, hand it to the `valuation-reviewer` agent.

Per scenario in `val["assumptions"]` (bear/base/bull) — note these are the
**clamped** values actually used downstream (SPEC §3):
- [ ] `discount_rate > terminal_growth` strictly. Equality/inversion = blocker (Gordon undefined).
- [ ] `terminal_growth ≤ 0.04`.
- [ ] `discount_rate ≥ 0.07` (≥ `0.10` if the filer is unprofitable / `sector_type == "growth_unprofitable"`).
- [ ] `growth_5y ≤ 0.40`.
- [ ] Scenario ordering: `bear.growth_5y ≤ base.growth_5y ≤ bull.growth_5y`.

Fair value & bands (`val["fair_value_range"]`, `val["dcf"]`, `val["pb_roe"]`):
- [ ] Each scenario band has `lo ≤ per_share ≤ hi` (per-share within its own band).
- [ ] Bands non-negative; per-share values rounded to 2dp (percentiles 1dp, growth 4dp — SPEC §11).
- [ ] `net_debt` never subtracted into per-share (FCFE-direct, SPEC §4) — if `equity != ev` in `dcf`, that's a red flag.
- [ ] Headline `fair_value_range` matches the sector's active method: financial/reit → from `pb_roe` (DCF disabled with a reason); cyclical with a computed `normalized_variant` → from that variant; `hyper_growth == True` → from the hyper base band. The `sensitivity` grid must describe the **same** fcf0/method as the headline band (SPEC §8/§9).

Reverse DCF (`val["reverse_dcf"]`):
- [ ] `implied_growth`, if present, is within the bisection bracket [-0.20, 0.40]; outside → `bracket_status` must be `above_bracket`/`below_bracket`, not a bogus number.
- [ ] Implied vs `realized_cagr_5y` gap plausible; a wildly-above-bracket implied growth should surface as "pahalı" in triangulation.

Triangulation (`val["triangulation"]`):
- [ ] `confidence` consistent with signal agreement: all three agree → `YÜKSEK`; exactly two → `ORTA`; scattered or ≥2 `veri_yok` → `DÜŞÜK` (SPEC §10).
- [ ] `direction` is the majority signal (or `belirsiz`); DCF signal agrees with where `price` sits vs the headline band.

Determinism & safety:
- [ ] Re-run step 1A a second time → identical `valuation_json` (same inputs must give same numbers, SPEC intro). A diff is a determinism bug.
- [ ] No `null`/`NaN`/`None` leaking as display text; every missing piece is `None` + a Turkish `note` in `val["notes"]`.

## 3. Report

Summarize: ticker, sector_type, active valuation method, headline base band,
confidence — then the violation list (or "clean"). For any code-level defect,
recommend escalating to the `valuation-reviewer` agent with the specific
file/section. Close with the standard caveat: mechanical reference only, not
investment advice.
