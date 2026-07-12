# sec_analyzer

`sec_analyzer` fetches a public company's financial statements straight from
the **official SEC EDGAR API**, normalizes the raw XBRL facts into a clean,
per-fiscal-year time series, computes a handful of standard fundamental
ratios, stores everything locally in SQLite, and (optionally) produces a
conservative, data-grounded fundamental read -- either from Claude, from a
local Ollama/Gemma model, or from a fully deterministic, no-AI script
analyzer (see "Script-based analysis" below).

**The only data source is SEC EDGAR.** There is no `yfinance`, no
`openai`, and no other third-party market-data or LLM library anywhere in
this package -- just `requests` against `data.sec.gov` / `www.sec.gov`, and
the official `anthropic` SDK for the optional Claude-backed analysis step
(not required for the `ollama` or `script` providers).

## What it does

1. **Resolve** a ticker symbol to its SEC CIK via `company_tickers.json`.
2. **Fetch** the filer's full XBRL "company facts" document
   (`data.sec.gov/api/xbrl/companyfacts/...`).
3. **Normalize** the raw, messy XBRL facts (tag fallbacks, restatements,
   annual vs. quarterly period selection) into a tidy per-concept,
   per-fiscal-year series.
4. **Compute ratios**: net margin, ROE, current ratio, and YoY revenue /
   net-income growth.
5. **Store** the ticker, CIK, normalized series, and ratios in a local
   SQLite database.
6. **Interpret** (optional, `analyze` command): run the normalized figures
   and ratios through a selectable backend -- Claude, a local Ollama/Gemma
   model, or the deterministic script analyzer -- and get back a structured
   JSON verdict: a conservative fair-value range, a fundamental-quality
   verdict, cyclicality commentary, and a plain-language summary.

## Install

```powershell
pip install -r sec_analyzer/requirements.txt
```

## ⚠️ Set your SEC User-Agent first

SEC EDGAR **blocks or rate-limits requests with a generic or missing
User-Agent header.** Every request this tool makes must identify a real
requester. Before running anything, create a `.env` file (in the directory
you run the CLI from, or anywhere `python-dotenv` will discover it) with:

```env
SEC_USER_AGENT="Your Name your.email@example.com"
ANTHROPIC_API_KEY="sk-ant-..."
```

`SEC_USER_AGENT` is required for every command (`fetch` and `analyze`).
`ANTHROPIC_API_KEY` is only required for `analyze`. There is no default for
`SEC_USER_AGENT` on purpose -- the tool will raise a clear configuration
error rather than silently sending an anonymous-looking request that SEC
might block.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SEC_USER_AGENT` | Yes (all commands) | *(none -- must be set)* | Identifies the requester to SEC EDGAR, e.g. `"Jane Doe jane.doe@example.com"`. |
| `ANTHROPIC_API_KEY` | Yes (`analyze` only) | *(none -- must be set)* | Anthropic API key used by the `analyze` command. |
| `SEC_ANTHROPIC_MODEL` | No | `claude-opus-4-8` | Anthropic model ID used for the Claude analysis. Deliberately *not* the generic `ANTHROPIC_MODEL`, so a shared `.env` can't silently downgrade the model. |
| `SEC_DB_PATH` | No | `sec_analyzer/sec_data.sqlite3` | Path to the local SQLite database. |
| `METODOLOJI_PATH` | No | `sec_analyzer/METODOLOJI.md` | Optional path to a custom analysis methodology (see below). |
| `SEC_MAX_RPS` | No | `8` | Maximum requests per second sent to SEC EDGAR (SEC's fair-access limit is 10/s). |

## Usage

```powershell
# Fetch, normalize, and store 5 years of financials for Apple (no Claude call)
python -m sec_analyzer.cli fetch AAPL --years 5

# Same, plus a Claude fundamental analysis
python -m sec_analyzer.cli analyze AAPL

# Same, but with the deterministic, no-AI script analyzer instead
python -m sec_analyzer.cli analyze AAPL --provider script

# Bypass the on-disk cache and re-fetch fresh data from SEC
python -m sec_analyzer.cli fetch AAPL --no-cache

# Verbose / quiet logging
python -m sec_analyzer.cli analyze AAPL --verbose
python -m sec_analyzer.cli fetch AAPL --quiet
```

Both subcommands accept `--years N` (default 5) and `--no-cache`. If the
`analyze` command's Claude call fails for any reason (missing API key,
network error, invalid API response, or a non-JSON model reply), it prints a
clear warning to stderr but still shows the financials that were fetched,
normalized, and stored -- it never crashes on an analysis failure.

## SEC rate limits

SEC EDGAR's fair-access policy caps requests at **10 per second per IP**.
`SecHttpClient` throttles itself to `SEC_MAX_RPS` (default 8, safely under
the limit) and automatically retries with exponential backoff (capped at
30 seconds per attempt, honoring any `Retry-After` header) on `429` (rate
limited), `403` (transient block), and `5xx` responses. A `404` is raised
immediately without retrying, since it usually means the resource genuinely
doesn't exist.

## Caching

Raw JSON responses from SEC (the ticker index and each filer's
`companyfacts` document) are cached on disk under `sec_analyzer/raw/`, keyed
by CIK. Subsequent runs against the same ticker reuse the cache instead of
re-fetching. Pass `--no-cache` to bypass the cache and force a fresh fetch
(the cache file is overwritten with the new response).

## Custom analysis methodology (`METODOLOJI.md`)

The `analyze` command's system prompt comes from an optional methodology
document. If a file exists at `METODOLOJI_PATH` (default:
`sec_analyzer/METODOLOJI.md`) and is non-empty, its contents are used
verbatim as the system prompt's analysis framework. If the file is missing,
empty, or unreadable, a built-in default fundamental-analysis framework is
used instead (see `sec_analyzer/interpret/analyzer.py`,
`DEFAULT_METHODOLOGY`). This file is intentionally **not** checked into the
repository -- it's an optional, user-supplied override, and the default
methodology must always remain reachable when it's absent.

## Script-based analysis (no AI)

Alongside the two LLM providers, `analyze` supports a third backend:
`--provider script` (also selectable in the web UI as **Script (no AI ·
deterministic)**).

* **What it is.** A deterministic Graham-style fundamental screen computed
  entirely with plain arithmetic over the same normalized SEC figures and
  ratios the LLM providers see -- no network call, no model, no randomness.
  It runs a fixed ten-point checklist (profitability, margin level and
  trend, revenue/earnings growth, liquidity, leverage, cash-flow quality,
  and shareholder returns), derives a cyclicality read from the volatility
  of year-over-year revenue growth, and estimates a conservative per-share
  fair-value range using the classic Benjamin Graham growth formula
  (`V = EPS x (8.5 + 2g)`), cross-checked against the Graham number when
  book value and share count are available. The result includes a `score`
  field listing every check by name with a pass/fail/n/a and the exact
  figures behind it, so the verdict is fully auditable -- see
  `sec_analyzer/interpret/rule_based.py`.
* **Who it's for.** Anyone who wants a quick, reproducible fundamentals
  read without an API key, without running Ollama, and fully offline
  (beyond the SEC EDGAR fetch itself).
* **How to use it.** CLI: `python -m sec_analyzer.cli analyze AAPL --provider script`.
  Web UI: select **Script (no AI · deterministic)** from the provider dropdown.
* **What it isn't.** The fair-value band is a heuristic screen based only on
  the figures provided, not a price target or a personalized investment
  recommendation.

## Known edge cases

* **ADR / 20-F filers (IFRS reporters):** Foreign private issuers filing
  Form 20-F typically report under the `ifrs-full` XBRL taxonomy instead of
  `us-gaap`. Since this tool only extracts `us-gaap` concepts, such filers
  will show every concept as `missing` rather than raising an error --
  `analyze` will still run and will call out the data gap in its summary.
* **Restatements:** When the same fiscal period appears in more than one
  filing (a correction/restatement), the row with the most recently *filed*
  date wins.
* **Fiscal year labeling:** The fiscal year for each record is derived from
  its `period_end` date (calendar year of the period end), not from the raw
  `fy` field SEC stamps on the fact -- SEC stamps a filing's *comparative*
  prior-year columns with the filing's own `fy`, which would otherwise
  mislabel and collapse distinct years onto one label.
* **Insufficient data:** If a filer has too little usable annual data (e.g.
  a recent IPO with only one or two 10-Ks on file), `compute_ratios` may
  return an empty or partial list, and `analyze` is instructed to report
  that uncertainty and return `null` fair-value bounds rather than
  guessing.

## Not investment advice

Output from the `analyze` command is educational commentary on public SEC
filings, generated from a limited set of historical figures. It is **not**
personalized investment advice, and should not be treated as a
recommendation to buy, hold, or sell any security.

## Web UI

A small Flask front end wraps the same fetch/normalize/store/interpret
pipeline as the CLI: type a ticker, click **Get Earnings** for financials
straight from SEC EDGAR, or **Analyze with LLM** to also get a fundamental
read.

```powershell
python -m sec_analyzer.web.app
```

This serves the UI at **http://127.0.0.1:5000**. As with the CLI,
`SEC_USER_AGENT` must be set (e.g. in `.env`) before fetching anything.

The page has a provider selector for the analysis step:

* **Script (no AI · deterministic)** -- the deterministic rule-based
  screen described above; no API key or local model required.
* **Gemma (local · Ollama)** -- the default. Requires
  [Ollama](https://ollama.com) running locally with the model pulled:
  ```powershell
  ollama pull gemma4:latest
  ```
* **Claude (Anthropic)** -- requires `ANTHROPIC_API_KEY` to be set.

## Tests

```powershell
pytest sec_analyzer/tests
```
