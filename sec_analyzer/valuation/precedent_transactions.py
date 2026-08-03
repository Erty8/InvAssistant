"""Load operator-curated precedent-transaction (M&A comps) reference data
(SPEC.md Sec.8i).

Optional, purely local reference data (no network access): a single CSV an
operator drops into ``Config.PRECEDENT_TRANSACTIONS_DIR`` -- mirrors
``damodaran.py``'s architecture and tolerance philosophy exactly (tolerant of
a missing directory, missing file, or malformed columns; logs what's
unavailable and returns whatever subset *is* usable rather than raising).
Deal comps aren't available from any of this project's existing data
sources (SEC EDGAR, Damodaran, yfinance, FRED all lack them), so -- exactly
like ``data/damodaran/*.csv`` -- this is data an analyst curates by hand, not
a new software dependency.

This module deliberately does NOT re-implement ``damodaran.sector_medians``'s
fuzzy SIC-to-industry matching: precedent-transaction rows are keyed by the
SAME Damodaran industry-name taxonomy that function already resolves each
filer to, so this module's own lookup is a plain normalized-name match
against whatever industry name ``damodaran.sector_medians`` already picked
for the filer -- one taxonomy, one matcher, reused rather than duplicated.

Expected file, ``deals.csv``, with a header row:
``industry, ev_ebitda, ev_revenue, control_premium[, deal_count]``.
"""

import logging
import os
from typing import List, Optional

from sec_analyzer.valuation import damodaran

logger = logging.getLogger(__name__)

_DEALS_FILENAME = "deals.csv"


def load_precedent_transactions(dir_path: Optional[str]) -> Optional[List[dict]]:
    """Load precedent-transaction (M&A comps) reference data from ``dir_path``.

    Args:
        dir_path: Directory expected to contain ``deals.csv`` (typically
            ``Config.PRECEDENT_TRANSACTIONS_DIR``).

    Returns:
        A list of ``{"industry": str, "ev_ebitda": float|None,
        "ev_revenue": float|None, "control_premium": float|None,
        "deal_count": float|None}`` dicts (one per row with a usable
        ``industry`` value; other columns individually degrade to ``None``
        on missing/malformed data rather than dropping the row), or
        ``None`` if ``dir_path`` doesn't exist, ``deals.csv`` is missing/
        unreadable, or no row has a usable industry name. Never raises.
    """
    if not dir_path or not os.path.isdir(dir_path):
        logger.info("precedent_transactions: directory %r not found; comps unavailable.", dir_path)
        return None

    rows = damodaran._read_csv_rows(os.path.join(dir_path, _DEALS_FILENAME))
    if rows is None:
        logger.info(
            "precedent_transactions: %s not found or unreadable in %s.", _DEALS_FILENAME, dir_path
        )
        return None

    parsed = []
    for row in rows:
        industry = (row.get("industry") or "").strip()
        if not industry:
            continue
        parsed.append({
            "industry": industry,
            "ev_ebitda": damodaran._to_float(row, "ev_ebitda"),
            "ev_revenue": damodaran._to_float(row, "ev_revenue"),
            "control_premium": damodaran._to_float(row, "control_premium"),
            "deal_count": damodaran._to_float(row, "deal_count"),
        })
    return parsed or None


def find_industry_medians(deals: Optional[List[dict]], industry: Optional[str]) -> Optional[dict]:
    """Look up ``industry``'s precedent-transaction medians.

    Exact match after ``damodaran._normalize_text`` normalization (lowercase,
    punctuation collapsed to spaces) -- deliberately NOT a second fuzzy SIC
    matcher; ``industry`` is expected to be whatever name
    ``damodaran.sector_medians`` already resolved for the filer (e.g.
    ``sector_medians_result["industry"]``), so this only needs to find that
    SAME name in the precedent-transaction data, not re-derive it from SIC.

    Args:
        deals: The list returned by :func:`load_precedent_transactions`, or
            ``None``.
        industry: The already-resolved Damodaran industry name, or ``None``.

    Returns:
        The matching row dict (see :func:`load_precedent_transactions`), or
        ``None`` if ``deals``/``industry`` is empty/``None``, or no row's
        industry name matches. Never raises.
    """
    if not deals or not industry:
        return None
    target = damodaran._normalize_text(industry)
    if not target:
        return None
    for row in deals:
        if damodaran._normalize_text(row.get("industry")) == target:
            return row
    return None
