"""Historical and current-period valuation multiples (P/E, P/S, P/FCF).

Pairs each fiscal year's fundamentals with the market price prevailing at
that fiscal year-end to build a company-specific multiples *history*, and
computes where the *current* multiple (from
``normalize.metrics.compute_metrics``) sits within that history via a
midrank percentile. Both functions are pure and never raise: missing price
coverage, missing fundamentals, or too little history simply produce
``None``/an empty list rather than an exception.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from sec_analyzer.normalize.normalizer import to_annual_series

logger = logging.getLogger(__name__)

#: Minimum number of non-None historical values required for
#: ``percentile_position`` to return a result rather than ``None``.
_MIN_PERCENTILE_SAMPLE = 5


def _period_end_by_fy(normalized: dict) -> Dict[int, str]:
    """Build a ``{fiscal_year: period_end}`` lookup across all annual
    concepts, scanning every concept in case any single one has gaps
    (mirrors ``normalize.ratios._period_end_by_fy``)."""
    mapping: Dict[int, str] = {}
    for records in (normalized.get("annual") or {}).values():
        if not records:
            continue
        for record in records:
            fy = record.get("fy")
            period_end = record.get("period_end")
            if fy is not None and fy not in mapping and period_end:
                mapping[fy] = period_end
    return mapping


def _price_on_or_before(price_df: Optional[pd.DataFrame], period_end: str) -> Optional[float]:
    """Return the last available ``Close`` on or before ``period_end``.

    Returns ``None`` if ``price_df`` is missing/empty, ``period_end`` can't
    be parsed as a date, or the price history's earliest row is already
    after ``period_end`` (doesn't cover that far back).
    """
    if price_df is None or getattr(price_df, "empty", True):
        return None
    try:
        cutoff = pd.Timestamp(period_end)
    except (ValueError, TypeError):
        return None

    eligible = price_df[price_df.index <= cutoff]
    if eligible.empty or "Close" not in eligible.columns:
        return None
    return float(eligible.iloc[-1]["Close"])


def multiples_history(normalized: dict, price_df: Optional[pd.DataFrame]) -> List[dict]:
    """Build a per-fiscal-year P/E, P/S, P/FCF history for one filer.

    For every fiscal year that has a usable annual period-end date AND a
    price on or before that date, computes:

    * ``pe = fy_price / eps_fy`` (``None`` unless ``eps_fy > 0``)
    * ``ps = fy_price * shares_fy / revenue_fy`` (``None`` unless
      ``revenue_fy > 0`` and ``shares_fy`` is present)
    * ``pfcf = fy_price * shares_fy / fcf_fy`` (``None`` unless
      ``fcf_fy > 0`` -- ``fcf_fy = OperatingCashFlow_fy - CapEx_fy`` -- and
      ``shares_fy`` is present)

    Args:
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.
        price_df: The DataFrame returned by
            ``sec_analyzer.fetch.prices.get_price_history`` (Date index,
            ``Close`` column), or ``None``.

    Returns:
        A list of ``{"fy", "end", "price", "pe", "ps", "pfcf"}`` dicts
        sorted by ``fy`` ascending. Empty list if no fiscal year has both a
        period-end date and price coverage. Never raises.
    """
    try:
        return _multiples_history(normalized or {}, price_df)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("multiples_history() failed unexpectedly; returning an empty history.")
        return []


def _multiples_history(normalized: dict, price_df: Optional[pd.DataFrame]) -> List[dict]:
    eps_series = to_annual_series(normalized, "EPS")
    revenue_series = to_annual_series(normalized, "Revenue")
    shares_series = to_annual_series(normalized, "SharesOutstanding")
    ocf_series = to_annual_series(normalized, "OperatingCashFlow")
    capex_series = to_annual_series(normalized, "CapEx")
    end_by_fy = _period_end_by_fy(normalized)

    history: List[dict] = []
    for fy in sorted(end_by_fy):
        period_end = end_by_fy[fy]
        fy_price = _price_on_or_before(price_df, period_end)
        if fy_price is None:
            continue

        eps = eps_series.get(fy)
        revenue = revenue_series.get(fy)
        shares = shares_series.get(fy)
        ocf = ocf_series.get(fy)
        capex = capex_series.get(fy)
        fcf = None if ocf is None or capex is None else ocf - capex

        pe = fy_price / eps if eps is not None and eps > 0 else None
        ps = (
            fy_price * shares / revenue
            if revenue is not None and revenue > 0 and shares
            else None
        )
        pfcf = (
            fy_price * shares / fcf
            if fcf is not None and fcf > 0 and shares
            else None
        )

        history.append(
            {"fy": fy, "end": period_end, "price": fy_price, "pe": pe, "ps": ps, "pfcf": pfcf}
        )

    return history


def percentile_position(history_values: List[Optional[float]], current: Optional[float]) -> Optional[float]:
    """Return the midrank percentile of ``current`` within ``history_values``.

    Percentile = (percentage of historical values strictly less than
    ``current``) + half the percentage of values tied with ``current``
    (standard midrank treatment of ties).

    Args:
        history_values: Historical multiple values (``None`` entries are
            dropped before counting).
        current: The current multiple value to rank.

    Returns:
        A float in ``[0, 100]`` rounded to 1 decimal, or ``None`` if
        ``current`` is ``None`` or fewer than 5 non-``None`` historical
        values are available.
    """
    if current is None:
        return None

    valid = [v for v in (history_values or []) if v is not None]
    if len(valid) < _MIN_PERCENTILE_SAMPLE:
        return None

    less_count = sum(1 for v in valid if v < current)
    equal_count = sum(1 for v in valid if v == current)
    pct = (less_count + 0.5 * equal_count) / len(valid) * 100.0
    return round(pct, 1)
