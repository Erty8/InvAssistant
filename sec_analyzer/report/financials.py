"""Serialize normalized SEC financials into the compact payload the report
template's "Bilanço" (balance-sheet) tab consumes.

This is shared by both entry points that render the unified verdict-card
template so they stay in sync by construction:

* the Flask web UI (``sec_analyzer.web.app``'s ``/api/analyze`` JSON payload
  and server-baked ``/report`` page), and
* the CLI's ``analyze --html`` report (``sec_analyzer.cli.cmd_analyze`` via
  :func:`sec_analyzer.report.generator.generate_report`).

It trims each normalized record down to just the fields the front end renders
(dropping ``tag``, ``form``, ``filed``, ``reported_fy``, ``start``, etc.), so
the payload stays small and stable regardless of internal normalization
details. All figures are passed through untouched -- no financial computation
happens here; the derived ratios come from the already-computed ``ratios``
list (see :func:`sec_analyzer.normalize.ratios.compute_ratios`).
"""

from typing import List

#: Canonical annual concept keys, in the order the front end should display
#: them. Kept here (rather than only in the template) so the API/report and UI
#: stay in sync with what ``normalize_facts`` actually produces.
ANNUAL_CONCEPTS = (
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
QUARTERLY_CONCEPTS = ("Revenue", "NetIncome")

#: Number of most-recent quarterly periods to include per concept.
QUARTERLY_LIMIT = 8


def serialize_financials(normalized: dict, ratios: list) -> dict:
    """Convert a normalized facts dict + ratios list into a JSON-friendly payload.

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
    for concept in ANNUAL_CONCEPTS:
        records = annual_bucket.get(concept) or []
        # Records are already sorted by period_end descending by
        # normalize_facts; re-sort defensively so the payload contract doesn't
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
    for concept in QUARTERLY_CONCEPTS:
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
            for record in sorted_records[:QUARTERLY_LIMIT]
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
