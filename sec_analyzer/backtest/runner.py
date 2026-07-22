"""Backtest grid runner: as-of analysis over a (ticker x date) grid.

For each (ticker, as-of date) pair this runs the deterministic point-in-time
analysis (``--no-ai``, provider ``"script"``), persists the verdict (with its
``as_of`` filled), and -- after the whole grid -- evaluates forward outcomes
(:func:`sec_analyzer.backtest.outcomes.evaluate_outcomes`).

No AI/LLM calls are ever made here: the grid uses the deterministic engine so a
backtest is reproducible and free of hindsight leakage. EDGAR access goes
through the existing throttled ``SecHttpClient``; price series are disk-cached
per ticker by ``fetch.prices``.
"""

import logging
import os
from datetime import date
from typing import List, Optional

logger = logging.getLogger(__name__)


def read_tickers_file(path: str) -> List[str]:
    """Read a watchlist file: one ticker per line, ``#`` comments/blank lines ignored."""
    tickers: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            # Allow "AAPL  # Apple" trailing comments and comma-separated lines.
            token = token.split("#", 1)[0]
            for part in token.replace(",", " ").split():
                part = part.strip().upper()
                if part and part not in tickers:
                    tickers.append(part)
    return tickers


def parse_dates(spec: str) -> List[date]:
    """Parse a comma-separated ISO date list into ``date`` objects (order kept)."""
    dates: List[date] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        parsed = date.fromisoformat(token)
        if parsed > date.today():
            raise ValueError(f"backtest date {token} is in the future.")
        if parsed not in dates:
            dates.append(parsed)
    return dates


def _run_one(ticker: str, as_of: date, years: int, no_cache: bool, db_path: Optional[str]) -> str:
    """Run one as-of analysis (script provider) and persist the verdict.

    Returns a short status string: ``"ok"``, ``"no_data"``, or ``"error"``.
    Never raises -- a single failing cell must not abort the grid.
    """
    # Imported lazily to avoid an import cycle (cli imports the backtest CLI wiring).
    import argparse

    from sec_analyzer.config import Config
    from sec_analyzer.cli import (
        _detect_filing_events,
        _fetch_catalyst,
        _fetch_normalize_store,
        _fetch_price_and_technical,
        _fetch_risk_free_asof,
        _fetch_submissions,
    )
    from sec_analyzer.interpret.analyzer import interpret
    from sec_analyzer.normalize.metrics import compute_metrics
    from sec_analyzer.normalize.red_flags import detect_red_flags
    from sec_analyzer.store.database import save_verdict

    horizon = "1y"
    try:
        args = argparse.Namespace(ticker=ticker, years=years, no_cache=no_cache, as_of=as_of)
        cik, _name, normalized, ratios = _fetch_normalize_store(args)

        if not any((normalized.get("annual") or {}).values()):
            logger.info("backtest: %s @ %s -> no filed data as of that date.", ticker, as_of)
            return "no_data"

        price, _price_as_of, technical, price_df = _fetch_price_and_technical(
            ticker, horizon, no_cache, as_of
        )
        fred_rate = _fetch_risk_free_asof(as_of, no_cache)
        metrics = compute_metrics(normalized, ratios, price)
        flags = detect_red_flags(normalized, ratios, metrics, horizon)
        submissions = _fetch_submissions(cik, ticker, no_cache)
        catalyst = _fetch_catalyst(submissions, ticker, as_of)
        events = _detect_filing_events(submissions, as_of)

        result = interpret(
            normalized, ratios, provider="script", horizon=horizon,
            metrics=metrics, technical=technical, red_flags=flags,
            catalyst=catalyst, submissions=submissions, price_df=price_df,
            as_of=as_of, fred_rate=fred_rate,
        )
        if isinstance(result, dict):
            result["events"] = events
            result["as_of"] = as_of.isoformat()

        if "error" in result:
            logger.info("backtest: %s @ %s -> interpret error: %s", ticker, as_of, result.get("error"))
            return "error"

        save_verdict(
            ticker, cik, horizon, "script", price, result,
            db_path=db_path or Config.DB_PATH, valuation=result.get("valuation"),
            as_of=as_of.isoformat(),
        )
        return "ok"
    except Exception:  # noqa: BLE001 - one grid cell must not abort the batch
        logger.warning("backtest: %s @ %s failed", ticker, as_of, exc_info=True)
        return "error"


def run_backtest(
    tickers: List[str],
    dates: List[date],
    years: int = 5,
    no_cache: bool = False,
    db_path: Optional[str] = None,
    evaluate: bool = True,
) -> dict:
    """Run the (ticker x date) as-of grid, persist verdicts, then evaluate outcomes.

    Args:
        tickers: Ticker symbols to run.
        dates: As-of dates (each a past ``date``).
        years: Fiscal-year window passed to normalization.
        no_cache: Bypass raw JSON / price caches.
        db_path: SQLite path. Defaults to ``Config.DB_PATH``.
        evaluate: When True (default), run
            :func:`sec_analyzer.backtest.outcomes.evaluate_outcomes` after the grid.

    Returns:
        ``{"cells": int, "ok": int, "no_data": int, "error": int,
        "outcomes": <evaluate summary or None>}``.
    """
    from sec_analyzer.backtest.outcomes import evaluate_outcomes

    tally = {"cells": 0, "ok": 0, "no_data": 0, "error": 0}
    for as_of in dates:
        for ticker in tickers:
            status = _run_one(ticker, as_of, years, no_cache, db_path)
            tally["cells"] += 1
            tally[status] = tally.get(status, 0) + 1
            print(f"  {ticker} @ {as_of.isoformat()}: {status}")

    outcomes_summary = None
    if evaluate:
        print("\nEvaluating forward outcomes...")
        outcomes_summary = evaluate_outcomes(db_path=db_path, no_cache=no_cache)
    tally["outcomes"] = outcomes_summary
    return tally
