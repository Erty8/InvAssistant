# InvAssistant

Two independent codebases live in this repo — do not mix changes across the boundary in one task:

- **Root portfolio pipeline** (`main.py`, `agents.py`, `config.py`, `scheduler.py`, `market_calendar.py`, `tools/`): daily portfolio email digest. No pytest suite.
- **`sec_analyzer/` package**: SEC-filing fundamental analysis, valuation engine, HTML verdict-card report, and Flask web UI (`sec_analyzer/web/`). Has a pytest suite.

## Rules for all code

- No new third-party dependencies unless the task explicitly says so.
- LLM usage belongs in offline ETL steps that write to a structured cache; the runtime analysis/valuation path stays deterministic and LLM-free.
- Never actually send email or trigger a real scheduled run during development; stub or stop before the side-effect boundary.

## sec_analyzer rules

- Everything is English: user-facing strings (verdict labels such as CHEAP/FAIR/EXPENSIVE, signal values such as `cheap`/`expensive`/`mixed`/`no_data`, notes, CLI card, report/UI text) as well as code, comments, and docstrings. Use English number formats (`19.0%`, `4.21`), not Turkish (`%19`, `4,21`). Compare against shared constants (e.g. `triangulate.SIGNAL_MIXED`) instead of hard-coding label literals.
- Computation must be deterministic: same inputs → same outputs. No randomness, no wall-clock dependence in results.
- Never let analysis-layer code crash the CLI: catch and convert to error dicts / `None` + log. Missing data renders as "—", never a printed "None"/"NaN".
- Ground truth, in this order: `sec_analyzer/valuation/SPEC.md` (binding contract — code that contradicts it is wrong), then `sec_analyzer/METODOLOJI.md` and `sec_analyzer/VALUATION.md`.

## Tests

```
python -m pytest sec_analyzer/tests -q
```

On Windows set `PYTHONIOENCODING=utf-8` if test output hits encoding errors. The root pipeline has no pytest suite — verify it with import checks and dry paths that stop before SMTP send.
