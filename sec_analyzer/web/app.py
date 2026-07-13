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
from html import escape
from typing import Optional, Tuple

from flask import Flask, jsonify, request

from sec_analyzer.config import Config, ConfigError
from sec_analyzer.fetch.companyfacts import get_company_facts, get_submissions
from sec_analyzer.fetch.filings import estimate_next_earnings
from sec_analyzer.fetch.prices import PriceDataError, get_price_history, latest_price
from sec_analyzer.fetch.tickers import resolve_cik
from sec_analyzer.http_client import SecHttpClient
from sec_analyzer.interpret.analyzer import interpret
from sec_analyzer.normalize.metrics import compute_metrics
from sec_analyzer.normalize.normalizer import normalize_facts
from sec_analyzer.normalize.ratios import compute_ratios
from sec_analyzer.normalize.red_flags import detect_red_flags
from sec_analyzer.report.generator import render_report_html, render_search_page
from sec_analyzer.store.database import save_normalized, save_prices, save_verdict
from sec_analyzer.technical.indicators import compute_indicators
from sec_analyzer.technical.verdict import technical_verdict

logger = logging.getLogger(__name__)

#: Selectable investment horizons shown in the UI, as (value, label) pairs.
_HORIZONS = [
    ("3m", "3 ay"),
    ("1y", "1 yıl"),
    ("5y", "5 yıl"),
]

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


def _fetch_price_and_technical(ticker: str, horizon: str, no_cache: bool):
    """Fetch price history and derive the merged technical indicators/verdict.

    Mirrors ``sec_analyzer.cli._fetch_price_and_technical`` so the web UI's
    ``/api/analyze`` pipeline behaves identically to the CLI's ``analyze``
    command. Fully graceful: if price data can't be obtained, this logs a
    warning and returns ``(None, None, None, None)`` rather than raising --
    the fundamental side of the pipeline must keep working even with no
    usable price data at all.

    Returns:
        ``(price, as_of, technical, price_df)``, where ``technical`` is the
        merged ``{**indicators, **technical_verdict_result}`` dict expected
        by :func:`sec_analyzer.interpret.analyzer.interpret`, and
        ``price_df`` is the raw OHLCV DataFrame (kept so the caller can
        persist it without re-fetching). All four are ``None`` if price data
        is unavailable.
    """
    try:
        price_df, source = get_price_history(ticker, no_cache=no_cache)
        price, as_of = latest_price(price_df)
        indicators = compute_indicators(price_df)
        verdict_result = technical_verdict(indicators, horizon)
        technical = {**indicators, **verdict_result}
        logger.info("Price data for %s from %s: %.2f as of %s", ticker, source, price, as_of)
        return price, as_of, technical, price_df
    except PriceDataError as exc:
        logger.warning("Price data unavailable for %s: %s", ticker, exc)
        return None, None, None, None


def _fetch_submissions(cik: str, ticker: str, no_cache: bool) -> Optional[dict]:
    """Best-effort fetch of a filer's raw SEC submissions document; never raises.

    Mirrors ``sec_analyzer.cli._fetch_submissions``: fetched exactly once per
    request (SPEC.md Sec.13) and reused for both the next-earnings catalyst
    estimate (:func:`_fetch_catalyst`) and SIC-based sector classification
    (passed straight through to
    :func:`sec_analyzer.interpret.analyzer.interpret` as ``submissions=``).

    Returns:
        The dict returned by
        :func:`sec_analyzer.fetch.companyfacts.get_submissions`, or ``None``
        if the fetch fails for any reason.
    """
    try:
        client = SecHttpClient()
        return get_submissions(cik, client, no_cache=no_cache)
    except Exception:  # noqa: BLE001 - submissions are best-effort, never fatal
        logger.warning("Could not fetch SEC submissions for %s", ticker, exc_info=True)
        return None


def _fetch_catalyst(submissions: Optional[dict], ticker: str) -> Optional[dict]:
    """Best-effort next-earnings estimate from already-fetched submissions;
    never raises (see the CLI's equivalent helper for the rationale).

    Args:
        submissions: The dict returned by :func:`_fetch_submissions`, or
            ``None``.
        ticker: Stock ticker symbol, used only for the warning log message.
    """
    if not submissions:
        return None
    try:
        return estimate_next_earnings(submissions)
    except Exception:  # noqa: BLE001 - a catalyst estimate is a nice-to-have, never fatal
        logger.warning("Could not estimate next earnings date for %s", ticker, exc_info=True)
        return None


def _save_price_rows(cik: str, price_df) -> None:
    """Convert a price-history DataFrame to row dicts and persist them.

    Never raises: a failure to persist price history must not prevent the
    rest of the request (interpretation, JSON response) from completing.
    """
    try:
        rows = [
            {
                "date": row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }
            for _, row in price_df.reset_index().iterrows()
        ]
        save_prices(cik, rows, db_path=Config.DB_PATH)
    except Exception:  # noqa: BLE001 - persistence failure must not be fatal
        logger.warning("Failed to save price history for CIK %s", cik, exc_info=True)


def _run_full_pipeline(
    ticker: str, years: int, no_cache: bool, horizon: str
) -> Tuple[
    str, str, dict, list, dict, Optional[dict], list, Optional[dict], Optional[float],
    Optional[dict], object,
]:
    """Extend ``_run_pipeline`` with the price/technical/metrics/red-flags/
    submissions/catalyst steps used by both the CLI's ``analyze`` command and
    this module's ``/api/analyze``/``/report`` routes, so all three stay in
    sync. Mirrors ``sec_analyzer.cli.cmd_analyze``: SEC submissions are
    fetched exactly once and reused for both the catalyst estimate and (by
    the caller, via the returned value) SIC-based sector classification.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        years: Number of most-recent fiscal years to retain.
        no_cache: When True, bypass the on-disk raw JSON cache.
        horizon: Investment horizon ("3m", "1y", or "5y") used to weight and
            frame the technical verdict and red-flag commentary.

    Returns:
        ``(cik, name, normalized, ratios, metrics, technical, flags,
        catalyst, price, submissions, price_df)``. ``submissions`` and
        ``price_df`` are meant to be threaded straight into
        :func:`sec_analyzer.interpret.analyzer.interpret` as
        ``submissions=``/``price_df=`` so the web UI gets the same full
        deterministic valuation the CLI does.

    Raises:
        Same as ``_run_pipeline`` -- financials fetch/normalize/store
        failures propagate for the caller to translate into an HTTP error.
        Everything past that point (price, technical, submissions, catalyst)
        is best-effort and never raises.
    """
    cik, name, normalized, ratios = _run_pipeline(ticker, years, no_cache)

    price, as_of, technical, price_df = _fetch_price_and_technical(ticker, horizon, no_cache)

    metrics = compute_metrics(normalized, ratios, price)
    flags = detect_red_flags(normalized, ratios, metrics, horizon)
    submissions = _fetch_submissions(cik, ticker, no_cache)
    catalyst = _fetch_catalyst(submissions, ticker)

    if price_df is not None:
        _save_price_rows(cik, price_df)

    return cik, name, normalized, ratios, metrics, technical, flags, catalyst, price, submissions, price_df


def _bool_param(value: Optional[str]) -> bool:
    """Parse a query-string boolean parameter (``"true"``/``"false"``, etc.)."""
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


@app.route("/")
def index():
    """Render the interactive "Verdict Terminal" search page.

    Serves the same self-contained ``template.html`` shell used by
    ``GET /report`` (see ``sec_analyzer.report.generator``), but with a
    ``mode: "search"`` payload: the template's client-side script renders a
    live ticker/horizon/provider search box instead of a baked result, and
    POSTs to ``/api/analyze`` on submit.
    """
    return render_search_page(
        _HORIZONS, _PROVIDERS, "1y", Config.ANALYZER_PROVIDER, Config.OLLAMA_MODEL
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


#: Valid ``horizon`` values accepted by ``/api/analyze``; anything else
#: (missing, malformed, or unrecognized) falls back to "1y".
_VALID_HORIZONS = ("3m", "1y", "5y")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Fetch/normalize/store a ticker's financials, then run a full
    fundamental + technical analysis.

    JSON body:
        ticker: Stock ticker symbol (required).
        years: Number of most-recent fiscal years to retain (default 5).
        horizon: Investment horizon -- "3m", "1y", or "5y" (default "1y").
            Controls the fundamental/technical weighting and the framing of
            the verdict; see ``Config.HORIZON_WEIGHTS``.
        provider: Analysis provider to use ("ollama", "anthropic", or
            "script"); defaults to ``Config.ANALYZER_PROVIDER`` when
            omitted/None.
        no_cache: Bypass the on-disk raw JSON cache (default False).

    The financials pipeline uses the same error handling as
    ``/api/financials``. Price/technical data is fetched best-effort -- if
    it's unavailable, the fundamental analysis still runs (with
    ``technical: null`` in the response) rather than failing the whole
    request. SEC submissions are also fetched once (best-effort) and passed
    into ``interpret(...)`` as ``submissions=``/``price_df=``, exactly like
    the CLI's ``analyze`` command, so ``analysis`` carries the full
    deterministic valuation engine output (``analysis["valuation"]``,
    ``analysis["confidence"]``, ``analysis["reverse_dcf_comment"]``) rather
    than a degraded fallback. Once financials succeed, ``interpret(...)`` is
    called and its result -- success or error dict -- is passed through
    under the ``analysis`` key with a 200 status, since a failed/degraded
    analysis is not itself a request failure.

    Response gains, alongside the usual financials payload and ``analysis``:
        technical: merged indicators + technical-verdict dict, or ``null``.
        metrics: valuation/quality metrics dict (see
            ``sec_analyzer.normalize.metrics.compute_metrics``).
        red_flags: list of ``{"code", "message", "detail"}`` dicts.
        catalyst: ``{"estimate_date", "label", "based_on"}`` dict, or ``null``.
    """
    body = request.get_json(silent=True) or {}

    ticker = str(body.get("ticker") or "").strip()
    if not ticker:
        return jsonify({"ok": False, "error": "JSON field 'ticker' is required."}), 400

    try:
        years = int(body.get("years", 5))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "'years' must be an integer."}), 400

    horizon = str(body.get("horizon") or "1y").strip().lower()
    if horizon not in _VALID_HORIZONS:
        horizon = "1y"

    provider = body.get("provider") or None
    no_cache = bool(body.get("no_cache", False))

    try:
        cik, name, normalized, ratios, metrics, technical, flags, catalyst, price, submissions, price_df = (
            _run_full_pipeline(ticker, years, no_cache, horizon)
        )

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

    logger.info(
        "Running %s analysis for %s (horizon=%s)", provider or Config.ANALYZER_PROVIDER, ticker, horizon
    )
    analysis = interpret(
        normalized,
        ratios,
        provider=provider,
        horizon=horizon,
        metrics=metrics,
        technical=technical,
        red_flags=flags,
        catalyst=catalyst,
        submissions=submissions,
        price_df=price_df,
    )

    if "error" not in analysis:
        try:
            save_verdict(
                ticker, cik, horizon, provider or Config.ANALYZER_PROVIDER, price, analysis,
                db_path=Config.DB_PATH, valuation=analysis.get("valuation"),
            )
        except Exception:  # noqa: BLE001 - persistence failure must not fail the request
            logger.warning("Failed to save verdict for %s", ticker, exc_info=True)

    payload = _serialize_financials(normalized, ratios)
    return jsonify({
        "ok": True,
        **payload,
        "cik": cik,
        "name": name,
        "analysis": analysis,
        "technical": technical,
        "metrics": metrics,
        "red_flags": flags,
        "catalyst": catalyst,
    })


#: Shell used by ``_error_page`` for ``/report`` failures -- a small,
#: self-contained (no external resources) HTML page in the same dark
#: palette as ``sec_analyzer.report.template``, so an error looks like a
#: degraded report rather than a bare Flask error response.
_ERROR_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Rapor Hatası</title>
<style>
  html, body {{
    margin: 0; padding: 0;
    background: #0d1420; color: #e7ecf5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.45;
  }}
  .page {{ max-width: 560px; margin: 0 auto; padding: 48px 16px; }}
  .card {{
    background: #111b2b; border: 1px solid #223349; border-radius: 14px;
    padding: 22px 20px;
  }}
  h1 {{ font-size: 1rem; color: #ff6b5e; margin: 0 0 10px; }}
  p {{ margin: 0; font-size: 0.9rem; color: #e7ecf5; }}
</style>
</head>
<body>
<div class="page"><div class="card">
  <h1>Rapor oluşturulamadı</h1>
  <p>{message}</p>
</div></div>
</body>
</html>"""


def _error_page(message: str) -> str:
    """Render a minimal, styled, self-contained HTML error page.

    Used by ``GET /report`` so a pipeline failure (bad ticker, missing
    config, unexpected error) renders as a small HTML page rather than a
    JSON error body or a bare stack trace -- the route is meant to be
    opened directly in a browser or embedded in an ``<iframe>``, where a
    JSON response would just show as raw text.
    """
    return _ERROR_PAGE_TEMPLATE.format(message=escape(str(message)))


@app.route("/report", methods=["GET"])
def report():
    """Run the full fundamental + technical analysis pipeline for a ticker
    and return the standalone HTML verdict-card report -- the same report
    ``sec_analyzer.cli``'s ``analyze --html`` flag writes to disk (see
    ``sec_analyzer.report.generator.render_report_html``) -- rendered live
    as the response body.

    This is the "type a ticker, see the exact report card" counterpart to
    ``/api/analyze``: where that route returns JSON, this one returns a
    complete, self-contained HTML page (the same unified "Verdict Terminal"
    template the ``/`` search page renders client-side, only with the
    analysis baked in server-side) suitable for opening directly in a
    browser tab.

    Query params:
        ticker: Stock ticker symbol (required).
        horizon: Investment horizon -- "3m", "1y", or "5y" (default "1y").
        provider: Analysis provider -- "ollama", "anthropic", or "script";
            defaults to ``Config.ANALYZER_PROVIDER`` when omitted/blank.
        years: Number of most-recent fiscal years to retain (default 12,
            wider than ``/api/financials``'/``/api/analyze``'s default 5 so
            the valuation engine's multiples-percentile history has more to
            work with).
        no_cache: Bypass the on-disk raw JSON cache (default False).

    Returns:
        On success, a ``200`` response with ``Content-Type: text/html``
        whose body is the rendered verdict-card report. On any failure
        (missing ticker, bad ticker, missing SEC_USER_AGENT config, or an
        unexpected error), an HTML error page (via :func:`_error_page`)
        with a ``400``/``404``/``500`` status -- never a JSON body, and
        never a bare Python traceback.
    """
    ticker = (request.args.get("ticker") or "").strip()
    if not ticker:
        return _error_page("Query parameter 'ticker' is required."), 400

    try:
        years = int(request.args.get("years", 12))
    except (TypeError, ValueError):
        return _error_page("'years' must be an integer."), 400

    horizon = (request.args.get("horizon") or "1y").strip().lower()
    if horizon not in _VALID_HORIZONS:
        horizon = "1y"

    provider = request.args.get("provider") or None
    no_cache = _bool_param(request.args.get("no_cache"))

    try:
        cik, name, normalized, ratios, metrics, technical, flags, catalyst, price, submissions, price_df = (
            _run_full_pipeline(ticker, years, no_cache, horizon)
        )

    except ConfigError as exc:
        logger.error("Configuration error while generating report for %s: %s", ticker, exc)
        return _error_page(str(exc)), 400

    except ValueError as exc:
        logger.info("Ticker resolution failed for %s: %s", ticker, exc)
        return _error_page(str(exc)), 404

    except Exception:  # noqa: BLE001 - last-resort guard, never leak a stack trace to the client
        logger.exception("Unexpected error generating report for %s", ticker)
        return _error_page(
            "An unexpected server error occurred while generating the report."
        ), 500

    resolved_provider = provider or Config.ANALYZER_PROVIDER
    logger.info("Running %s analysis for %s report (horizon=%s)", resolved_provider, ticker, horizon)
    analysis = interpret(
        normalized,
        ratios,
        provider=provider,
        horizon=horizon,
        metrics=metrics,
        technical=technical,
        red_flags=flags,
        catalyst=catalyst,
        submissions=submissions,
        price_df=price_df,
    )

    if "error" not in analysis:
        try:
            save_verdict(
                ticker, cik, horizon, resolved_provider, price, analysis,
                db_path=Config.DB_PATH, valuation=analysis.get("valuation"),
            )
        except Exception:  # noqa: BLE001 - persistence failure must not fail the request
            logger.warning("Failed to save verdict for %s", ticker, exc_info=True)

    as_of = technical.get("as_of") if technical else None
    html = render_report_html(
        ticker, horizon, analysis,
        metrics=metrics, technical=technical, flags=flags, price=price, as_of=as_of,
        entity_name=name,
    )
    return html


def main() -> None:
    """Configure logging and start the development server on 127.0.0.1:5000."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
