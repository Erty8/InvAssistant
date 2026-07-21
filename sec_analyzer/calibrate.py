"""Headless script-provider calibration harness (normalization Work Package 0).

Runs the existing ``fetch``/``analyze`` pipeline over a fixed basket of
tickers with ``provider="script"`` (deterministic, rule-based -- no LLM, no
network call beyond the usual SEC/price fetches) and reports, per ticker, the
ratio of the base-scenario fair-value-range midpoint to the current price.
This is a measurement tool for the larger valuation-normalization effort:
running it before and after a change to :mod:`sec_analyzer.valuation.engine`
lets that change be judged by how far it shifts the basket's median
fair-value/price ratio, rather than by eyeballing individual tickers.

This module only orchestrates already-existing pipeline pieces
(:mod:`sec_analyzer.cli`'s fetch/normalize/price/submissions/catalyst
helpers, :mod:`sec_analyzer.normalize.metrics`,
:mod:`sec_analyzer.normalize.red_flags`,
:func:`sec_analyzer.interpret.analyzer.interpret`) -- it adds no new
valuation logic. It mirrors :func:`sec_analyzer.cli.cmd_analyze`'s input
assembly end-to-end (including the SEC submissions fetch that feeds
SIC-based sector classification and the CAPM cost of equity) so the
measured valuation path matches the one production actually exercises. It
never raises: any per-ticker failure is caught and recorded as an
``"error"``/``"skipped"`` row so one bad ticker never aborts the rest of
the basket run.
"""

import argparse
import json
import logging
import math
import os
import statistics
from datetime import datetime
from typing import List, Optional

from sec_analyzer.config import Config
from sec_analyzer.interpret.analyzer import interpret
from sec_analyzer.normalize.metrics import compute_metrics
from sec_analyzer.normalize.red_flags import detect_red_flags

logger = logging.getLogger(__name__)

#: Default ~28-ticker basket spanning mega-cap tech, financials, REITs,
#: energy, healthcare, consumer staples, industrials/cyclicals, and
#: high-growth/unprofitable names -- broad enough that a shift in the
#: basket's median fair-value/price ratio is meaningful signal rather than
#: noise from a single sector's quirks.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "O", "PLD",
    "XOM", "CVX", "JNJ", "PFE", "PG", "KO", "CAT", "DE", "MU", "CRM", "ADBE",
    "RDDT", "PLTR", "UBER", "SHOP", "WMT", "COST", "VZ",
]

#: Column width used when rendering the calibration table in
#: print_calibration_table, mirroring cli.py's _print_ratios column style.
_COL_WIDTH = 14

#: Fixed horizon used for every calibration run. Calibration is about
#: measuring the valuation engine's fair-value output, not the technical/
#: horizon-weighting layer, so a single representative horizon keeps every
#: ticker's ratio comparable.
_CALIBRATION_HORIZON = "1y"


def _method_slug(valuation: dict) -> str:
    """Return a short machine-readable slug for the headline valuation method.

    Mirrors the exact precedence order of
    :func:`sec_analyzer.cli._valuation_method_label` (SPEC.md Sec.8/8e/11),
    but returns a compact slug instead of that function's Turkish/mixed
    display label, and splits its combined "Revenue-DCF" branch (which
    mirrors ``mature_revenue_headline`` and ``midgrowth_revenue_headline``
    with an ``or``) into two slugs for finer-grained diagnostics -- this
    does not change which branch fires, only how it's labeled.

    Hyper-growth takes precedence over every other headline (checked
    first, before ``cyclical_fcfe_headline``/``earnings_power_headline``/
    ``mature_revenue_headline``/``midgrowth_revenue_headline``): when a
    filer is hyper-headlined, ``run_valuation`` still leaves the standard
    two-stage DCF enabled and populated as a secondary/comparison scenario
    (``valuation["dcf"]["enabled"]`` stays ``True``), so without this check
    a hyper-grower (e.g. RDDT) would fall through to the ``"dcf"`` branch
    below and be mislabeled.

    Args:
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation` (or ``None``).

    Returns:
        One of ``"hyper"``, ``"cyclical-fcfe"``, ``"epv"``, ``"mature-rev"``,
        ``"midgrowth-rev"``, ``"dcf"``, ``"ffo"``, or ``"pb-roe"``.
    """
    valuation = valuation or {}
    detail = valuation.get("hyper_growth_detail") or {}
    if valuation.get("hyper_growth") and detail and not detail.get("suppressed"):
        return "hyper"
    if valuation.get("cyclical_fcfe_headline"):
        return "cyclical-fcfe"
    if valuation.get("earnings_power_headline"):
        return "epv"
    if valuation.get("mature_revenue_headline"):
        return "mature-rev"
    if valuation.get("midgrowth_revenue_headline"):
        return "midgrowth-rev"
    dcf = valuation.get("dcf") or {}
    if dcf.get("enabled", True):
        return "dcf"
    ffo = valuation.get("ffo")
    if isinstance(ffo, dict) and "scenarios" in ffo:
        return "ffo"
    return "pb-roe"


def run_calibration(
    tickers: List[str], years: int = 5, no_cache: bool = False, as_of=None
) -> List[dict]:
    """Run the headless script-provider pipeline for each ticker in ``tickers``.

    For each ticker: resolve/fetch/normalize/store (reusing
    :func:`sec_analyzer.cli._fetch_normalize_store`), fetch price + technical
    data (reusing :func:`sec_analyzer.cli._fetch_price_and_technical`), then
    run ``compute_metrics`` -> ``detect_red_flags`` -> fetch SEC submissions
    (reusing :func:`sec_analyzer.cli._fetch_submissions`) and derive the
    next-earnings catalyst from them (reusing
    :func:`sec_analyzer.cli._fetch_catalyst`), then
    :func:`sec_analyzer.interpret.analyzer.interpret` with
    ``provider="script"`` (no LLM/API calls). ``submissions`` is passed
    through to ``interpret`` exactly as :func:`sec_analyzer.cli.cmd_analyze`
    does, so SIC-based sector classification and the CAPM cost-of-equity
    (and therefore the financial/REIT sector anchors) are measured the same
    way production computes them -- calibration would otherwise silently
    exercise a non-representative sector-agnostic-discount-rate code path.

    The ``cli`` imports happen inside this function (not at module level) to
    avoid an import cycle: ``cli.py`` imports this module to wire up the
    ``calibrate`` subcommand.

    This function never raises: any exception anywhere in a given ticker's
    pipeline is caught, logged, and recorded as an ``"error"`` row so the
    rest of the basket still runs.

    Args:
        tickers: Stock ticker symbols to run, e.g. ``["AAPL", "MSFT"]``.
        years: Number of most-recent fiscal years to retain (passed straight
            through to ``_fetch_normalize_store``; default: 5).
        no_cache: Bypass the on-disk raw JSON/price caches and re-fetch
            (default: ``False``).
        as_of: Optional point-in-time date (``datetime.date`` or ``None``).
            When set, the whole basket runs as of that past date (fundamentals
            filed on/before it, prices up to it, archived ERP + FRED risk-free
            macro), so the fair-value/price ratio distribution can be compared
            across market regimes (e.g. the 2021 peak vs the 2022 trough) to
            separate engine conservatism from the period's valuation level.

    Returns:
        One row dict per ticker, each with a ``"ticker"`` and ``"status"``
        (``"ok"``, ``"skipped"``, or ``"error"``) key:

        * ``"ok"``: also has ``"price"``, ``"fv_base_mid"``, ``"ratio"``
          (``fv_base_mid / price``), and ``"method"`` (see :func:`_method_slug`).
        * ``"skipped"``: also has ``"reason"`` (e.g. missing price or
          fair-value range, or an ``interpret`` error result).
        * ``"error"``: also has ``"error"`` (the exception message).
    """
    # Imported here, not at module level, to avoid an import cycle: cli.py
    # imports run_calibration/etc. from this module.
    from sec_analyzer.cli import (
        _fetch_catalyst,
        _fetch_normalize_store,
        _fetch_price_and_technical,
        _fetch_risk_free_asof,
        _fetch_submissions,
    )

    # One (cached) FRED fetch for the whole basket in as-of mode.
    fred_rate = _fetch_risk_free_asof(as_of, no_cache) if as_of is not None else None

    rows: List[dict] = []
    for ticker in tickers:
        try:
            args = argparse.Namespace(ticker=ticker, years=years, no_cache=no_cache, as_of=as_of)
            cik, _name, normalized, ratios = _fetch_normalize_store(args)
            price, _price_as_of, technical, price_df = _fetch_price_and_technical(
                ticker, _CALIBRATION_HORIZON, no_cache, as_of
            )
            metrics = compute_metrics(normalized, ratios, price)
            flags = detect_red_flags(normalized, ratios, metrics, _CALIBRATION_HORIZON)
            submissions = _fetch_submissions(cik, ticker, no_cache)
            catalyst = _fetch_catalyst(submissions, ticker, as_of)
            result = interpret(
                normalized,
                ratios,
                provider="script",
                horizon=_CALIBRATION_HORIZON,
                metrics=metrics,
                technical=technical,
                red_flags=flags,
                catalyst=catalyst,
                submissions=submissions,
                price_df=price_df,
                as_of=as_of,
                fred_rate=fred_rate,
            )
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not abort the basket
            logger.warning("Calibration failed for %s", ticker, exc_info=True)
            rows.append({"ticker": ticker, "status": "error", "error": str(exc)})
            continue

        valuation = (result or {}).get("valuation") or {}
        fv = ((valuation.get("fair_value_range") or {}).get("base") or {})
        lo, hi = fv.get("lo"), fv.get("hi")

        price_unreliable = isinstance(metrics, dict) and metrics.get("price_reliable") is False
        if lo is None or hi is None or not price or price_unreliable:
            if isinstance(result, dict) and "error" in result:
                reason = f"interpret error: {result.get('error')}"
            elif lo is None or hi is None:
                reason = "missing fair-value base range"
            elif not price:
                reason = "missing price"
            else:
                reason = "unreliable price (implausible P/E and P/S)"
            print(f"{ticker}: skipped ({reason})")
            rows.append({"ticker": ticker, "status": "skipped", "reason": reason})
            continue

        mid = (lo + hi) / 2
        rows.append({
            "ticker": ticker,
            "status": "ok",
            "price": price,
            "fv_base_mid": mid,
            "ratio": mid / price,
            "method": _method_slug(valuation),
        })

    return rows


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Deterministic linear-interpolation percentile over an already-sorted list.

    Args:
        sorted_values: Values sorted ascending; must be non-empty.
        pct: Percentile to compute, 0-100.

    Returns:
        The interpolated percentile value.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (pct / 100) * (len(sorted_values) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    lo, hi = sorted_values[f], sorted_values[c]
    return lo + (hi - lo) * (k - f)


def summarize_ratios(rows: List[dict]) -> dict:
    """Compute summary statistics over the fair-value/price ratios in ``rows``.

    Pure function: only reads the ``"ratio"`` field of rows whose
    ``"status"`` is ``"ok"`` (skipped/error rows have no ratio and are
    excluded). Percentiles use deterministic sorted-list linear
    interpolation (:func:`_percentile`), not any random sampling.

    Args:
        rows: The row list returned by :func:`run_calibration`.

    Returns:
        A dict with ``"count"`` (int), ``"median"``/``"mean"``/``"p25"``/
        ``"p75"`` (float or ``None`` if ``count == 0``), and three bucket
        counts -- ``"bucket_under_0.8"``, ``"bucket_0.8_1.2"``,
        ``"bucket_over_1.2"`` -- partitioning the "ok" ratios by whether the
        fair-value midpoint implies the stock is cheap (<0.8x), roughly
        fair (0.8x-1.2x), or expensive (>1.2x) relative to price.
    """
    ratios = [row["ratio"] for row in rows if row.get("status") == "ok" and row.get("ratio") is not None]
    count = len(ratios)

    summary = {
        "count": count,
        "median": None,
        "mean": None,
        "p25": None,
        "p75": None,
        "bucket_under_0.8": 0,
        "bucket_0.8_1.2": 0,
        "bucket_over_1.2": 0,
    }
    if count == 0:
        return summary

    summary["median"] = statistics.median(ratios)
    summary["mean"] = statistics.mean(ratios)

    sorted_ratios = sorted(ratios)
    summary["p25"] = _percentile(sorted_ratios, 25)
    summary["p75"] = _percentile(sorted_ratios, 75)

    for ratio in ratios:
        if ratio < 0.8:
            summary["bucket_under_0.8"] += 1
        elif ratio > 1.2:
            summary["bucket_over_1.2"] += 1
        else:
            summary["bucket_0.8_1.2"] += 1

    return summary


def print_calibration_table(rows: List[dict]) -> None:
    """Print the per-ticker calibration results as a compact aligned table.

    Mirrors :func:`sec_analyzer.cli._print_ratios`'s column-width/rjust
    style. This is internal diagnostic output (like ``_print_ratios``), so
    it's in English rather than the Turkish used for the analyze command's
    user-facing verdict card.

    Args:
        rows: The row list returned by :func:`run_calibration`.
    """
    if not rows:
        print("No calibration rows to display.")
        return

    def as_money(value) -> str:
        return f"{value:.2f}" if value is not None else "-"

    def as_ratio(value) -> str:
        return f"{value:.3f}" if value is not None else "-"

    headers = ["Ticker", "Status", "Price", "FV Mid", "Ratio", "Method"]
    print("".join(h.rjust(_COL_WIDTH) for h in headers))
    print("-" * (_COL_WIDTH * len(headers)))

    for row in rows:
        cells = [
            str(row.get("ticker", "-")),
            str(row.get("status", "-")),
            as_money(row.get("price")),
            as_money(row.get("fv_base_mid")),
            as_ratio(row.get("ratio")),
            str(row.get("method") or "-"),
        ]
        print("".join(c.rjust(_COL_WIDTH) for c in cells))


def save_calibration_snapshot(
    label: str, rows: List[dict], summary: dict, as_of: Optional[str] = None
) -> Optional[str]:
    """Persist a calibration run's rows and summary as a JSON snapshot.

    Written to ``Config.REPORTS_DIR/calibration_<label>_<YYYYMMDD-HHMM>.json``
    (``REPORTS_DIR`` is created with ``exist_ok=True`` first). Never raises:
    on any failure (e.g. an unwritable directory) this logs a warning and
    returns ``None``, matching the codebase's defensive-persistence pattern
    (see :func:`sec_analyzer.cli._save_price_rows`).

    Args:
        label: Short run label used in the filename, e.g. ``"baseline"`` or
            ``"run"``.
        rows: The row list returned by :func:`run_calibration`.
        summary: The dict returned by :func:`summarize_ratios`.
        as_of: Point-in-time cutoff (ISO ``"YYYY-MM-DD"``) the basket was run
            against, or ``None`` for a live run. Recorded in the payload and,
            when set, added to the filename as an ``_asof-<date>`` segment.

    Returns:
        The path written, or ``None`` if persistence failed.
    """
    try:
        os.makedirs(Config.REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now()
        asof_segment = f"_asof-{as_of}" if as_of else ""
        filename = f"calibration_{label}{asof_segment}_{timestamp.strftime('%Y%m%d-%H%M')}.json"
        path = os.path.join(Config.REPORTS_DIR, filename)
        payload = {
            "label": label,
            "as_of": as_of,
            "timestamp": timestamp.isoformat(),
            "rows": rows,
            "summary": summary,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path
    except Exception:  # noqa: BLE001 - persistence failure must not be fatal
        logger.warning("Failed to save calibration snapshot for label %s", label, exc_info=True)
        return None
