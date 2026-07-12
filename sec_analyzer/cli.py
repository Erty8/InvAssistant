"""Command-line entry point for sec_analyzer.

Two subcommands:

* ``fetch TICKER`` -- resolve the ticker to a CIK, pull SEC XBRL company
  facts, normalize them, compute ratios, print both, and persist everything
  to the local SQLite database.
* ``analyze TICKER`` -- everything ``fetch`` does, plus a fundamental
  interpretation (fair-value range, verdict, cyclicality, and a summary)
  from a selectable backend: local Ollama/Gemma (default), the hosted
  Anthropic Claude API, or a deterministic script-based (no-AI) analyzer.

Usage::

    python -m sec_analyzer.cli fetch AAPL --years 5
    python -m sec_analyzer.cli analyze AAPL
    python -m sec_analyzer.cli analyze AAPL --provider script

Only the official SEC EDGAR API and (for ``analyze`` with the ``ollama`` or
``anthropic`` providers) an LLM API are used -- no third-party finance data
libraries. The ``script`` provider makes no network calls at all beyond the
SEC EDGAR fetch.
"""

import argparse
import json
import logging
import sys
from typing import List, Tuple

import requests

from sec_analyzer.config import Config, ConfigError
from sec_analyzer.fetch.companyfacts import get_company_facts
from sec_analyzer.fetch.tickers import resolve_cik
from sec_analyzer.http_client import SecHttpClient
from sec_analyzer.interpret.analyzer import interpret
from sec_analyzer.normalize.normalizer import format_table, normalize_facts
from sec_analyzer.normalize.ratios import compute_ratios
from sec_analyzer.store.database import save_normalized

logger = logging.getLogger(__name__)

#: Column width used when rendering the ratios table in _print_ratios.
_RATIO_COL_WIDTH = 14


def _print_ratios(ratios: List[dict]) -> None:
    """Render the per-fiscal-year ratio list as a compact aligned table.

    Net margin and the two YoY growth ratios are shown as percentages; ROE
    and the current ratio are shown as plain decimals. Missing (``None``)
    values are rendered as ``-``.

    Args:
        ratios: The list returned by
            :func:`sec_analyzer.normalize.ratios.compute_ratios`.
    """
    if not ratios:
        print("No ratios available (insufficient annual data).")
        return

    def as_pct(value) -> str:
        return f"{value * 100:.1f}%" if value is not None else "-"

    def as_dec(value) -> str:
        return f"{value:.2f}" if value is not None else "-"

    headers = ["FY", "Net Margin", "ROE", "Current Ratio", "Rev YoY", "NI YoY"]
    print("".join(h.rjust(_RATIO_COL_WIDTH) for h in headers))
    print("-" * (_RATIO_COL_WIDTH * len(headers)))

    for row in ratios:
        cells = [
            str(row.get("fy", "-")),
            as_pct(row.get("net_margin")),
            as_dec(row.get("roe")),
            as_dec(row.get("current_ratio")),
            as_pct(row.get("yoy_revenue_growth")),
            as_pct(row.get("yoy_net_income_growth")),
        ]
        print("".join(c.rjust(_RATIO_COL_WIDTH) for c in cells))


def _fetch_normalize_store(args: argparse.Namespace) -> Tuple[dict, List[dict]]:
    """Resolve, fetch, normalize, print, and persist financials for a ticker.

    Shared by both ``fetch`` and ``analyze`` so the two subcommands stay in
    sync and ``analyze`` doesn't duplicate any of this logic.

    Args:
        args: Parsed CLI arguments; must have ``ticker``, ``years``, and
            ``no_cache`` attributes.

    Returns:
        The ``(normalized, ratios)`` pair produced for this ticker.
    """
    client = SecHttpClient()

    cik, name = resolve_cik(args.ticker, client, no_cache=args.no_cache)
    logger.info("Resolved ticker %s -> CIK %s (%s)", args.ticker, cik, name)

    facts = get_company_facts(cik, client, no_cache=args.no_cache)
    normalized = normalize_facts(facts, years=args.years)
    ratios = compute_ratios(normalized)

    print(format_table(normalized))
    print()
    _print_ratios(ratios)

    save_normalized(args.ticker, cik, name, normalized, ratios, db_path=Config.DB_PATH)
    print(f"\nSaved to database: {Config.DB_PATH}")

    return normalized, ratios


def cmd_fetch(args: argparse.Namespace) -> None:
    """Handle the ``fetch`` subcommand: fetch, normalize, and store only."""
    _fetch_normalize_store(args)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the ``analyze`` subcommand: ``fetch`` plus an LLM analysis."""
    normalized, ratios = _fetch_normalize_store(args)

    provider = getattr(args, "provider", None) or Config.ANALYZER_PROVIDER
    print(f"\nRunning {provider} analysis...")
    result = interpret(normalized, ratios, provider=provider)

    if "error" in result:
        print(
            f"\nWARNING: analysis unavailable ({result['error']}): "
            f"{result.get('summary', 'no further details')}",
            file=sys.stderr,
        )
        if "raw" in result:
            logger.debug("Raw model output that failed to parse: %s", result["raw"])
        return

    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))

    summary = result.get("summary")
    if summary:
        print(f"\nSummary: {summary}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``argparse`` parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="sec_analyzer",
        description=(
            "Fetch, normalize, store, and (optionally) interpret SEC EDGAR "
            "financial data for a public company."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared options for every subcommand: the ticker, plus fetch/store knobs
    # and logging verbosity.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("ticker", metavar="TICKER", help="Stock ticker symbol, e.g. AAPL")
    common.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of most-recent fiscal years to retain (default: 5).",
    )
    common.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk raw JSON cache and re-fetch from SEC EDGAR.",
    )
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )
    verbosity.add_argument(
        "--quiet", action="store_true", help="Only log WARNING and above."
    )

    fetch_parser = subparsers.add_parser(
        "fetch",
        parents=[common],
        help="Fetch, normalize, and store SEC financials for TICKER.",
    )
    fetch_parser.set_defaults(func=cmd_fetch)

    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[common],
        help="Fetch/normalize/store, then run a fundamental analysis (LLM or script) for TICKER.",
    )
    analyze_parser.add_argument(
        "--provider",
        choices=["ollama", "gemma", "anthropic", "script"],
        default=None,
        help=(
            "Analysis backend (default: ANALYZER_PROVIDER env, i.e. 'ollama'). "
            "script = deterministic rule-based analysis, no AI/LLM required."
        ),
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    return parser


def main() -> None:
    """CLI entry point: parse arguments, configure logging, dispatch."""
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"SEC EDGAR request failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
