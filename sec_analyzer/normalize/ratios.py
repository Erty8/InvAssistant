"""Compute basic per-fiscal-year financial ratios from normalized facts.

This module consumes the ``annual`` bucket produced by
``sec_analyzer.normalize.normalizer.normalize_facts`` and derives a small
set of standard ratios (profitability, leverage/liquidity, and YoY growth)
for every fiscal year that has at least one usable input.

Every ratio is computed defensively: if either operand for a given fiscal
year is missing, or a denominator is zero (or, for growth rates, the prior
period's base is zero or negative), the ratio is reported as ``None``
rather than raising or producing a misleading number.
"""

import logging
from typing import Dict, List, Optional

from sec_analyzer.normalize.normalizer import to_annual_series

logger = logging.getLogger(__name__)


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide two optional numbers, guarding against ``None`` and zero.

    Returns ``None`` if either operand is ``None`` or the denominator is 0;
    otherwise returns the quotient rounded to 4 decimal places.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _safe_growth(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Compute ``(current - prior) / prior`` as a year-over-year growth rate.

    Returns ``None`` if either value is missing, or if ``prior`` is zero or
    negative -- a non-positive base makes a percentage-growth figure
    meaningless (e.g. going from a $-10M loss to a $5M profit isn't a
    sensible "growth rate").
    """
    if current is None or prior is None or prior <= 0:
        return None
    return round((current - prior) / prior, 4)


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Subtract two optional numbers, guarding against ``None``.

    Returns ``None`` if either operand is ``None``; otherwise returns the
    difference rounded to 4 decimal places. Used for ``fcf`` (a raw USD
    figure, not a ratio) rather than ``_safe_div``.
    """
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _period_end_by_fy(normalized: dict) -> Dict[int, str]:
    """Build a ``{fiscal_year: period_end}`` lookup from any annual concept.

    Different concepts for the same filer/fiscal year should share the same
    period end date, so we simply take the first one encountered per fy,
    scanning across all concepts in case any single concept has gaps.
    """
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


def compute_ratios(normalized: dict) -> List[dict]:
    """Compute per-fiscal-year ratios from a normalized facts dict.

    Args:
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.

    Returns:
        A list of per-fiscal-year dicts, sorted by ``fy`` descending::

            {
              "fy": int,
              "period_end": str or None,
              "net_margin": float or None,             # NetIncome / Revenue
              "roe": float or None,                    # NetIncome / StockholdersEquity
              "current_ratio": float or None,          # CurrentAssets / CurrentLiabilities
              "yoy_revenue_growth": float or None,      # (Rev_t - Rev_t-1) / Rev_t-1
              "yoy_net_income_growth": float or None,   # (NI_t - NI_t-1) / NI_t-1
              "gross_margin": float or None,            # GrossProfit / Revenue
              "operating_margin": float or None,        # OperatingIncome / Revenue
              "roa": float or None,                     # NetIncome / TotalAssets
              "debt_to_equity": float or None,          # TotalLiabilities / StockholdersEquity
              "fcf": float or None,                     # OperatingCashFlow - CapEx (raw USD)
              "fcf_margin": float or None,               # fcf / Revenue
            }

        Returns an empty list if no fiscal year has any usable input data.
    """
    revenue = to_annual_series(normalized, "Revenue")
    net_income = to_annual_series(normalized, "NetIncome")
    equity = to_annual_series(normalized, "StockholdersEquity")
    current_assets = to_annual_series(normalized, "CurrentAssets")
    current_liabilities = to_annual_series(normalized, "CurrentLiabilities")
    gross_profit = to_annual_series(normalized, "GrossProfit")
    operating_income = to_annual_series(normalized, "OperatingIncome")
    total_assets = to_annual_series(normalized, "TotalAssets")
    total_liabilities = to_annual_series(normalized, "TotalLiabilities")
    operating_cash_flow = to_annual_series(normalized, "OperatingCashFlow")
    capex = to_annual_series(normalized, "CapEx")

    fiscal_years = (
        set(revenue)
        | set(net_income)
        | set(equity)
        | set(current_assets)
        | set(current_liabilities)
        | set(gross_profit)
        | set(operating_income)
        | set(total_assets)
        | set(total_liabilities)
        | set(operating_cash_flow)
        | set(capex)
    )

    if not fiscal_years:
        logger.warning(
            "compute_ratios: no annual data available for %s (CIK %s); "
            "returning an empty ratio list.",
            normalized.get("entity_name"), normalized.get("cik"),
        )
        return []

    period_end_by_fy = _period_end_by_fy(normalized)

    results: List[dict] = []
    for fy in sorted(fiscal_years, reverse=True):
        rev = revenue.get(fy)
        ni = net_income.get(fy)
        eq = equity.get(fy)
        curr_assets = current_assets.get(fy)
        curr_liabs = current_liabilities.get(fy)

        prev_fy = fy - 1
        prev_rev = revenue.get(prev_fy)
        prev_ni = net_income.get(prev_fy)

        gp = gross_profit.get(fy)
        oi = operating_income.get(fy)
        ta = total_assets.get(fy)
        tl = total_liabilities.get(fy)
        ocf = operating_cash_flow.get(fy)
        cpx = capex.get(fy)

        # fcf is a raw USD figure (not a ratio): OperatingCashFlow - CapEx.
        # CapEx (PaymentsToAcquirePropertyPlantAndEquipment) is reported as
        # a positive outflow, so it's subtracted rather than added. If CapEx
        # is missing, fcf is None rather than silently falling back to OCF
        # alone, which would overstate free cash flow.
        fcf = _safe_sub(ocf, cpx)

        results.append(
            {
                "fy": fy,
                "period_end": period_end_by_fy.get(fy),
                "net_margin": _safe_div(ni, rev),
                "roe": _safe_div(ni, eq),
                "current_ratio": _safe_div(curr_assets, curr_liabs),
                "yoy_revenue_growth": _safe_growth(rev, prev_rev),
                "yoy_net_income_growth": _safe_growth(ni, prev_ni),
                "gross_margin": _safe_div(gp, rev),
                "operating_margin": _safe_div(oi, rev),
                "roa": _safe_div(ni, ta),
                "debt_to_equity": _safe_div(tl, eq),
                "fcf": fcf,
                "fcf_margin": _safe_div(fcf, rev),
            }
        )

    return results
