"""Small Flask web UI for the sec_analyzer package.

Serves a single-page, vanilla-JS front end that lets a user type a stock
ticker, fetch its earnings/financials straight from SEC EDGAR, and
optionally run a fundamental analysis using a selectable backend: local
Ollama/Gemma (default), the hosted Anthropic Claude API, or a deterministic
script-based (no-AI) analyzer.

This module is a thin HTTP wrapper around the existing fetch/normalize/
store/interpret pipeline (see ``sec_analyzer.cli`` for the equivalent CLI
flow) -- it does not reimplement any of that logic.

Run it with::

    python -m sec_analyzer.web.app

Then open http://127.0.0.1:5000 in a browser.

Before starting the server, ``SEC_USER_AGENT`` must be set (typically via a
``.env`` file in the working directory) -- SEC EDGAR requires every request
to identify a real requester. See ``sec_analyzer.config.Config.get_user_agent``
for details. If it's missing, the API routes return a clear 400 error rather
than crashing.
"""

import logging
from typing import Optional, Tuple

from flask import Flask, jsonify, render_template, request

from sec_analyzer.config import Config, ConfigError
from sec_analyzer.fetch.companyfacts import get_company_facts
from sec_analyzer.fetch.tickers import resolve_cik
from sec_analyzer.http_client import SecHttpClient
from sec_analyzer.interpret.analyzer import interpret
from sec_analyzer.normalize.normalizer import normalize_facts
from sec_analyzer.normalize.ratios import compute_ratios
from sec_analyzer.store.database import save_normalized

logger = logging.getLogger(__name__)

app = Flask(__name__)

#: Canonical annual concept keys, in the order the front end should display
#: them. Kept here (rather than only in the template) so the API and UI stay
#: in sync with what ``normalize_facts`` actually produces.
_ANNUAL_CONCEPTS = (
    "Revenue",
    "GrossProfit",
    "OperatingIncome",
    "NetIncome",
    "TotalAssets",
    "TotalLiabilities",
    "StockholdersEquity",
    "OperatingCashFlow",
    "CapEx",
    "Cash",
    "CurrentAssets",
    "CurrentLiabilities",
    "LongTermDebt",
    "DividendsPaid",
    "EPS",
    "SharesOutstanding",
)

#: Quarterly concepts surfaced to the front end (a narrower set than annual --
#: quarterly balance-sheet figures are less commonly the point of interest
#: here, and keeping the payload small matters for a page rendered client-side).
_QUARTERLY_CONCEPTS = ("Revenue", "NetIncome")

#: Number of most-recent quarterly periods to include per concept.
_QUARTERLY_LIMIT = 8

#: Selectable analysis providers shown in the UI, as (value, label) pairs.
_PROVIDERS = [
    ("script", "Script (no AI · deterministic)"),
    ("ollama", "Gemma (local · Ollama)"),
    ("anthropic", "Claude (Anthropic)"),
]


def _serialize_financials(normalized: dict, ratios: list) -> dict:
    """Convert a normalized facts dict + ratios list into a JSON-friendly payload.

    This trims each record down to just the fields the front end renders
    (dropping ``tag``, ``form``, ``filed``, ``reported_fy``, ``start``, etc.),
    so the API response stays small and stable regardless of internal
    normalization details.

    Args:
        normalized: The dict returned by
            :func:`sec_analyzer.normalize.normalizer.normalize_facts`.
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.

    Returns:
        A dict of the form::

            {
              "cik": ..., "entity_name": ..., "currency": "USD",
              "annual": {"<concept>": [{"fy", "period_end", "value"}, ...]},
              "quarterly": {"Revenue": [...], "NetIncome": [...]},
              "ratios": [...],
              "missing": [...],
            }

        Every concept key in ``annual``/``quarterly`` is always present, with
        an empty list when there is no data, so the front end never has to
        guard against a missing key.
    """
    annual_bucket = normalized.get("annual") or {}
    quarterly_bucket = normalized.get("quarterly") or {}

    annual_out = {}
    for concept in _ANNUAL_CONCEPTS:
        records = annual_bucket.get(concept) or []
        # Records are already sorted by period_end descending by
        # normalize_facts; re-sort defensively so the API contract doesn't
        # silently depend on that upstream ordering.
        sorted_records = sorted(
            records, key=lambda r: r.get("period_end") or "", reverse=True
        )
        annual_out[concept] = [
            {
                "fy": record.get("fy"),
                "period_end": record.get("period_end"),
                "value": record.get("value"),
            }
            for record in sorted_records
        ]

    quarterly_out = {}
    for concept in _QUARTERLY_CONCEPTS:
        records = quarterly_bucket.get(concept) or []
        sorted_records = sorted(
            records, key=lambda r: r.get("period_end") or "", reverse=True
        )
        quarterly_out[concept] = [
            {
                "fy": record.get("fy"),
                "fp": record.get("fp"),
                "period_end": record.get("period_end"),
                "value": record.get("value"),
            }
            for record in sorted_records[:_QUARTERLY_LIMIT]
        ]

    return {
        "cik": normalized.get("cik"),
        "entity_name": normalized.get("entity_name"),
        "currency": normalized.get("currency", "USD"),
        "annual": annual_out,
        "quarterly": quarterly_out,
        "ratios": ratios or [],
        "missing": normalized.get("missing") or [],
    }


def _run_pipeline(ticker: str, years: int, no_cache: bool) -> Tuple[str, str, dict, list]:
    """Resolve, fetch, normalize, compute ratios for, and persist a ticker.

    Shared by both API routes so the fetch/normalize/store logic (and its
    error behavior) stays identical whether or not the caller goes on to
    request an LLM analysis.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        years: Number of most-recent fiscal years to retain.
        no_cache: When True, bypass the on-disk raw JSON cache.

    Returns:
        ``(cik, name, normalized, ratios)``.

    Raises:
        ConfigError: If required configuration (e.g. SEC_USER_AGENT) is missing.
        ValueError: If the ticker cannot be resolved to a CIK.
        Exception: Any other failure (network errors, etc.) propagates as-is
            for the caller to log and translate into a 500 response.
    """
    client = SecHttpClient()

    cik, name = resolve_cik(ticker, client, no_cache=no_cache)
    facts = get_company_facts(cik, client, no_cache=no_cache)
    normalized = normalize_facts(facts, years=years)
    ratios = compute_ratios(normalized)

    save_normalized(ticker, cik, name, normalized, ratios, db_path=Config.DB_PATH)

    return cik, name, normalized, ratios


def _bool_param(value: Optional[str]) -> bool:
    """Parse a query-string boolean parameter (``"true"``/``"false"``, etc.)."""
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


@app.route("/")
def index():
    """Render the single-page ticker explorer UI."""
    return render_template(
        "index.html",
        providers=_PROVIDERS,
        default_provider=Config.ANALYZER_PROVIDER,
        default_model=Config.OLLAMA_MODEL,
    )


@app.route("/api/financials", methods=["GET"])
def api_financials():
    """Fetch, normalize, store, and return a ticker's SEC financials.

    Query params:
        ticker: Stock ticker symbol (required).
        years: Number of most-recent fiscal years to retain (default 5).
        no_cache: "true"/"false" -- bypass the on-disk raw JSON cache.
    """
    ticker = (request.args.get("ticker") or "").strip()
    if not ticker:
        return jsonify({"ok": False, "error": "Query parameter 'ticker' is required."}), 400

    try:
        years = int(request.args.get("years", 5))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "'years' must be an integer."}), 400

    no_cache = _bool_param(request.args.get("no_cache"))

    try:
        cik, name, normalized, ratios = _run_pipeline(ticker, years, no_cache)
        payload = _serialize_financials(normalized, ratios)
        return jsonify({"ok": True, **payload, "cik": cik, "name": name})

    except ConfigError as exc:
        logger.error("Configuration error while fetching %s: %s", ticker, exc)
        return jsonify({"ok": False, "error": str(exc)}), 400

    except ValueError as exc:
        logger.info("Ticker resolution failed for %s: %s", ticker, exc)
        return jsonify({"ok": False, "error": str(exc)}), 404

    except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
        logger.exception("Unexpected error fetching financials for %s", ticker)
        return jsonify(
            {"ok": False, "error": "An unexpected server error occurred while fetching financials."}
        ), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Fetch/normalize/store a ticker's financials, then run an LLM analysis.

    JSON body:
        ticker: Stock ticker symbol (required).
        years: Number of most-recent fiscal years to retain (default 5).
        provider: Analysis provider to use ("ollama", "anthropic", or
            "script"); defaults to ``Config.ANALYZER_PROVIDER`` when
            omitted/None.
        no_cache: Bypass the on-disk raw JSON cache (default False).

    The financials pipeline uses the same error handling as
    ``/api/financials``. Once financials succeed, ``interpret(...)`` is
    called and its result -- success or error dict -- is passed through
    under the ``analysis`` key with a 200 status, since a failed/degraded
    analysis is not itself a request failure.
    """
    body = request.get_json(silent=True) or {}

    ticker = str(body.get("ticker") or "").strip()
    if not ticker:
        return jsonify({"ok": False, "error": "JSON field 'ticker' is required."}), 400

    try:
        years = int(body.get("years", 5))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "'years' must be an integer."}), 400

    provider = body.get("provider") or None
    no_cache = bool(body.get("no_cache", False))

    try:
        cik, name, normalized, ratios = _run_pipeline(ticker, years, no_cache)

    except ConfigError as exc:
        logger.error("Configuration error while analyzing %s: %s", ticker, exc)
        return jsonify({"ok": False, "error": str(exc)}), 400

    except ValueError as exc:
        logger.info("Ticker resolution failed for %s: %s", ticker, exc)
        return jsonify({"ok": False, "error": str(exc)}), 404

    except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
        logger.exception("Unexpected error fetching financials for %s", ticker)
        return jsonify(
            {"ok": False, "error": "An unexpected server error occurred while fetching financials."}
        ), 500

    logger.info("Running LLM analysis for %s (provider=%s)", ticker, provider or Config.ANALYZER_PROVIDER)
    analysis = interpret(normalized, ratios, provider=provider)

    payload = _serialize_financials(normalized, ratios)
    return jsonify({"ok": True, **payload, "cik": cik, "name": name, "analysis": analysis})


def main() -> None:
    """Configure logging and start the development server on 127.0.0.1:5000."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
