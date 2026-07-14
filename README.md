# InvAssistant

A personal investing toolkit with two parts:

- **`sec_analyzer/`** — fetches a company's financials straight from SEC EDGAR and turns them into a fundamental + technical verdict (fair-value range, support/resistance, momentum, MACD, price chart), via a CLI or a local web UI. No paid API required (has a fully deterministic, offline analysis mode).
- **Root-level portfolio pipeline** (`main.py`, `agents.py`, `scheduler.py`, `tools/`) — a scheduled job that pulls your portfolio's metrics/news, runs it through LLM agents, and emails a daily HTML report.

## Quick start: sec_analyzer

The easiest way in is the three `.bat` launchers at the repo root (Windows, double-click):

| File | What it does |
|---|---|
| `run.bat` | Starts the web UI at **http://127.0.0.1:5050** (auto-installs missing packages, asks for your SEC User-Agent on first run and saves it to `.env`). |
| `Analiz.bat [TICKER] [3m\|1y\|5y]` | Runs one deterministic analysis from the terminal and opens the HTML report. Prompts for a ticker/horizon if you don't pass them. |
| `Raporu-Ac.bat` | Re-opens the most recently generated HTML report. |

First time only: SEC EDGAR requires a real identity on every request. `run.bat` will ask for this itself (`Name Surname email@example.com`) and save it to `.env` — you don't need to do anything by hand.

### Command line

```powershell
pip install -r requirements.txt
python -m sec_analyzer.cli analyze AAPL --horizon 1y --provider script --html
```

`--provider script` is the default: fully deterministic, no AI/API key needed. `ollama` (local Gemma) and `anthropic` (Claude API) are also available. Full CLI reference, environment variables, and methodology: [`sec_analyzer/README.md`](sec_analyzer/README.md).

## Portfolio email pipeline

```powershell
python main.py
```

Configure via `.env` (LLM provider/keys, SMTP, portfolio holdings) — see `config.py` for the full list of settings.
