# Frozen Assumption Sets — Implementation Spec (DRAFT)

Status: **implemented.** This file is the binding contract for the
assumption-cache layer (`store/assumptions.py`), the `assumptions` CLI
subcommand, the `analyze --assumptions` resolution policy, and
`interpret(phase1_override=...)`. It is additive to `valuation/SPEC.md` (which
stays the contract for the engine itself) and changes NO existing output
shape. Tests: `sec_analyzer/tests/test_assumptions_cache.py`.

## 0. Motivation / architecture principle

Today, phase 1 (`interpret.analyzer.propose_assumptions`) may call an LLM at
**runtime, on every `analyze` run**. The engine is deterministic, but its
inputs are not: the same filer analyzed twice can get different
growth/discount assumptions and therefore a different fair-value band. This
also sits in tension with the repo rule that LLM usage belongs in offline
ETL steps writing to a structured cache, while the runtime valuation path
stays deterministic and LLM-free (CLAUDE.md).

Fix: assumptions become a **persisted, reviewable, per-filing artifact**:

- A company's fair value should only change when its **fundamentals change**
  (a new filing / restatement) or when the analyst deliberately revises the
  assumption set — never because of LLM sampling noise between two runs on
  the same data.
- The LLM is still used — but **once per filing, in an explicit offline
  step** (`assumptions propose`), whose output the analyst can inspect,
  edit, and freeze. Runtime `analyze` reads the frozen set from the store
  and calls no LLM for phase 1.
- Every frozen set is versioned, carries provenance (provider, model,
  timestamps, clamp notes, analyst note), and old sets are never deleted —
  the audit trail explains *why* the model said what it said at the time.

Lifecycle: `propose` → (review / `edit`) → `freeze` → `analyze` reads cache.

## 1. Store — new table `assumption_sets`

New module `sec_analyzer/store/assumptions.py` (stdlib `sqlite3` only, same
conventions as `store/database.py`; table created via `init_db`-style
idempotent DDL from this module's own `init_assumptions_table`, called by
`store.database.init_db` so one entry point still ensures the full schema).

```sql
CREATE TABLE IF NOT EXISTS assumption_sets (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    cik                  TEXT NOT NULL,
    ticker               TEXT,
    status               TEXT NOT NULL,      -- 'draft' | 'frozen' | 'superseded'
    fundamental_fy       INTEGER,            -- resolve_fundamental_fy(metrics) at propose time
    facts_fingerprint    TEXT,               -- see Sec.2
    source_provider      TEXT,               -- 'ollama' | 'anthropic' | 'script' | '<x>+manual'
    source_model         TEXT,               -- e.g. 'gemma4:latest'; NULL for script/manual
    proposed_at          TEXT,               -- ISO-8601
    frozen_at            TEXT,               -- ISO-8601, NULL until frozen
    superseded_at        TEXT,               -- ISO-8601, stamped when a newer freeze supersedes this row
    sector_type          TEXT,
    assumptions_json     TEXT NOT NULL,      -- POST-clamp bear/base/bull dict (SPEC.md Sec.2 shape)
    hyper_extras_json    TEXT,               -- phase-1 hyper_growth_extras, or NULL
    script_baseline_json TEXT,               -- rule_based.default_assumptions(...) at propose time
    sanity_notes_json    TEXT,               -- JSON list[str]: clamp/validation notes at propose time
    review_note          TEXT                -- analyst free text, set by freeze/edit
);
```

Invariants:

- **At most one `draft` and at most one `frozen` row per `cik`.**
  `propose` overwrites (replaces) any existing draft for that cik. `freeze`
  promotes the draft to `frozen` and flips any previously-frozen row for
  that cik to `superseded` in the same transaction. Superseded rows are
  never deleted (append-only history, like `verdicts`).
- `assumptions_json` always stores the **clamped** set
  (`sanity.clamp_assumptions` output) — the same numbers every downstream
  computation would use. Rates are decimal fractions per SPEC.md Sec.2.
- `script_baseline_json` stores the deterministic CAPM/CAGR-anchored
  default (`rule_based.default_assumptions` with the same `capm` /
  `risk_free_pct` inputs the propose run resolved) — the "second opinion"
  the review step diffs against. NULL only if it could not be built.

Public API (all take `db_path: Optional[str] = None`, all defensive /
never-raise-into-CLI per package rules):

```python
save_draft(cik, ticker, payload: dict) -> int         # replaces existing draft
load_active(cik, status: str = "frozen") -> Optional[dict]   # or "draft"
freeze_draft(cik, review_note: Optional[str]) -> Optional[dict]
update_draft(cik, assumptions, sector_type, review_note, source_suffix="+manual") -> Optional[dict]
load_history(cik, limit: int = 20) -> list[dict]
```

`verdicts` gains one additive column via `_VERDICTS_EXTRA_COLUMNS`:
`("assumption_set_id", "INTEGER")` — NULL for legacy rows and for runs that
did not use a cached set. Written by `save_verdict` when the result dict
carries `result["_assumption_set_id"]` (stamped by the CLI, see Sec.4).

## 2. Freshness — `facts_fingerprint`

A cached set is only trustworthy while the fundamentals it was proposed
from are unchanged. Freshness is decided by TWO checks, both required:

1. `fundamental_fy` equality: stored `fundamental_fy ==
   resolve_fundamental_fy(metrics)` of the current run.
2. Fingerprint equality: `facts_fingerprint == fingerprint_annual(normalized)`.

```python
def fingerprint_annual(normalized: dict) -> str:
    """sha256 hex digest of the canonical annual-series payload."""
```

Canonical form: `analyzer._annual_by_concept(normalized)` (already exists;
concept → {fy: value}, sorted keys), serialized with
`json.dumps(..., sort_keys=True, separators=(",", ":"))`, UTF-8, sha256.

Properties: daily price moves and quarterly buckets do NOT change it (only
annual series feed the fingerprint), so a set stays fresh between filings;
a new 10-K (new fy in any annual series) or a restatement (changed value)
invalidates it. A stale set is never silently used — see the resolution
policy in Sec.4.

## 3. CLI — new `assumptions` subcommand

Follows the existing `backtest` sub-subparser pattern in `cli.py`. All
user-facing strings Turkish.

### `assumptions propose TICKER [--provider P] [--model M] [--no-cache]`

The ONLY step that may call an LLM. Flow:

1. `_fetch_normalize_store(args)` (same as `analyze`), `compute_metrics`,
   `_fetch_submissions` → `classify_sector` sector hint, Damodaran load →
   `compute_cost_of_equity` + `risk_free` (mirrors `interpret()`'s own
   resolution at analyzer.py — the propose step must feed phase 1 exactly
   what `interpret()` feeds it today, so cached and legacy-live proposals
   are comparable).
2. `propose_assumptions(...)` with the chosen provider (default
   `Config.ANALYZER_PROVIDER`; `--provider script` is allowed and simply
   caches the deterministic baseline as the set).
3. `sanity.validate_assumptions` + `sanity.clamp_assumptions` (propose
   stores the CLAMPED set; clamp notes go to `sanity_notes_json`).
4. Build `script_baseline_json` via `rule_based.default_assumptions` with
   the same `capm`/`risk_free_pct`.
5. Print a review card: per scenario, proposed vs. script baseline side by
   side (growth_5y / terminal_growth / discount_rate, percentage-formatted,
   plus `story`), any clamp notes, and the divergence per field
   (`Δ discount_rate = +1.2pp` style). Large divergence is information, not
   an error — the card just surfaces it.
6. `save_draft(...)` (replaces any prior draft). Print Turkish next-step
   hint: gözden geçir → `assumptions freeze TICKER`.

As-of mode is NOT supported here (`propose` is a live-judgment step); no
`--as-of` flag.

### `assumptions show TICKER`

Prints the current draft (if any), the active frozen set (if any) with
provenance (provider/model/proposed_at/frozen_at/review_note, freshness
check against the currently-stored fundamentals: "GÜNCEL" / "BAYAT — yeni
dosyalama var, yeniden 'propose' önerilir"), and a short superseded-history
list (id, frozen_at → superseded date, provider).

### `assumptions edit TICKER --set PATH=VALUE [...] [--note TEXT]`

Edits the DRAFT in place (error card if no draft exists — Turkish message
suggesting `propose` first). `--set` is repeatable;
`PATH` ∈ `{bear,base,bull}.{growth_5y,terminal_growth,discount_rate,story}`
(decimal fractions for rates). After applying: re-validate + re-clamp,
print resulting violations/clamp notes, update `sanity_notes_json`, append
`+manual` to `source_provider` (once), store `--note` into `review_note`.

### `assumptions freeze TICKER [--note TEXT]`

Requires a draft. Re-runs `validate_assumptions` on the stored set as a
final guard (defense in depth — the draft is already clamped), then in one
transaction: prior frozen row (if any) → `superseded`, draft → `frozen`
with `frozen_at` stamped and `review_note` updated if `--note` given.
Prints confirmation with the set id.

## 4. `analyze` integration — resolution policy

New flag on the `analyze` subcommand:

```
--assumptions {auto, frozen, script, llm}      (default: auto)
```

Resolution, executed in `cmd_analyze` before `interpret(...)`:

- **`auto`** (default): load the frozen set for the cik; if present AND
  fresh (Sec.2) → use it. Else (absent or stale) → fall back to the
  deterministic script phase 1, with a Turkish notice naming which case it
  was ("Dondurulmuş varsayım seti yok/bayat; deterministik (script)
  varsayımlar kullanıldı. Öneri: `assumptions propose TICKER`."). **A stale
  frozen set is never silently used.**
- **`frozen`**: require a fresh frozen set; if absent/stale, print a
  Turkish error card and stop (no analysis) — the strict mode for when the
  user wants to be certain nothing but the reviewed set is in play.
- **`script`**: ignore the cache, force deterministic phase 1.
- **`llm`**: legacy behavior — live phase-1 LLM call each run (today's
  flow, kept for comparison/experimentation).

As-of mode (`--as-of`): the cache is **never** consulted, regardless of the
flag (`auto`/`frozen` degrade to `script` with a notice). A cached set was
proposed with knowledge as of `proposed_at`, which post-dates the as-of
cutoff — using it would be a hindsight leak into backtests. This mirrors
the existing as-of default-to-script rule in `cmd_analyze`.

### `interpret()` plumbing

`interpret()` gains one optional keyword argument:

```python
phase1_override: Optional[dict] = None
# shape: {"assumptions": {...}, "sector_type": str,
#         "hyper_growth_extras": dict|None,
#         "_provider": str, "_assumption_set_id": int}
```

When given, `interpret()` skips `propose_assumptions` entirely and uses the
override as the phase-1 result (everything downstream — SIC-wins sector
resolution, `run_valuation`, phase 2 — unchanged), and also skips the
CAPM/risk-free lookup that only feeds the phase-1 fallback. The override's
`_provider` for a cached set is `"cached:<source_provider>"` (e.g.
`"cached:ollama"`, `"cached:ollama+manual"`). `interpret()` stamps the
override's `_assumption_set_id` and `_provider` onto the returned result (as
`result["_assumption_set_id"]` and `result["_phase1_provider"]`) so
`save_verdict` persists the provenance (Sec.1) without the CLI having to
re-copy it. Phase 2's own `_provider` (the commentary backend) is left intact
— `_phase1_provider` is what records the source of the NUMBERS.

Phase 2 (commentary) is untouched by this spec: it remains
provider-selectable via `--provider` and never affects numbers
(`_postprocess_phase2_result` already enforces that). With a cached set,
the fair-value NUMBERS are fully deterministic regardless of the phase-2
provider.

`web/app.py` integration is explicitly **out of scope for v1** (the web
path keeps its current behavior); a follow-up can reuse the same
resolution helper.

## 5. What the LLM is asked (unchanged in v1, noted for follow-up)

v1 changes WHERE/WHEN phase 1 runs, not WHAT it sees: `propose` sends the
same numeric payload as today. A known follow-up (out of scope here) is to
enrich the propose-time payload with qualitative, filing-derived context
(e.g. the 8-K event layer in `signals/events.py`, MD&A extracts via an
offline ETL cache) — the offline propose step is exactly where such
enrichment becomes safe and affordable.

## 6. Tests (pytest, `sec_analyzer/tests/test_assumptions_store.py` + CLI tests)

- Store: draft save/replace roundtrip; freeze promotes + supersedes prior
  frozen atomically; history preserved; `update_draft` re-clamps and
  appends `+manual` exactly once.
- Fingerprint: stable across dict ordering; unchanged by quarterly/price
  data; changed by a new annual fy and by a restated value.
- Resolution: `auto` uses fresh frozen; `auto` falls back on stale (with
  note); `frozen` errors on stale; `llm` bypasses cache; as-of never reads
  cache even with `--assumptions frozen`.
- `interpret(phase1_override=...)` skips phase 1 (assert no provider
  dispatch), produces identical valuation for identical override (run
  twice, compare `fair_value_range`).
- `save_verdict` persists `assumption_set_id`; NULL for non-cached runs.
- Determinism end-to-end: two `analyze --assumptions auto` runs against an
  unchanged store produce byte-identical `fair_value_range` and
  `_provider == "cached:..."`.

## 7. Non-goals / explicitly out of scope

- No change to `valuation/` engine code or `valuation/SPEC.md` shapes.
- No change to phase-2 contracts or the rule-based provider.
- No web UI surface in v1.
- No automatic re-propose on staleness (the analyst decides when to re-run
  the LLM; `auto` mode's fallback keeps `analyze` usable meanwhile).
- No new third-party dependencies (stdlib `hashlib`/`sqlite3` only).
