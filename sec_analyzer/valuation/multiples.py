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

#: Minimum growth rate (as a decimal fraction, so ``0.05`` == 5%) below
#: which a growth-adjusted multiple (PEG / growth-adjusted EV/Sales) is NOT
#: computed. Guards against the PEG linearity flaw blowing up (or flipping
#: sign) as the denominator approaches zero -- see ``growth_adjusted_value``
#: and ``growth_adjusted_history``. Matches VALUATION.md Sec.7.
_PEG_MIN_GROWTH = 0.05

#: Forward window (in fiscal years) over which each historical year's own
#: realized revenue CAGR is measured to build the historical growth-adjusted
#: multiple series: a fiscal year's period-end multiple is divided by the
#: revenue CAGR *it went on to realize* over the following 3 years.
_GROWTH_ADJ_FORWARD_YEARS = 3


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
    * ``ev_sales = (fy_price * shares_fy + net_debt_fy) / revenue_fy``
      (``None`` unless ``revenue_fy > 0`` and ``shares_fy`` is present),
      where ``net_debt_fy = (LongTermDebt_fy or 0) + (LongTermDebtCurrent_fy
      or 0) - (Cash_fy or 0)`` -- treated as ``0.0`` (i.e. an unlevered EV
      equal to market cap) when none of those three concepts is present for
      that fiscal year. This is the sales multiple the hyper-grower
      growth-adjusted EV/Sales layer ranks against (VALUATION.md Sec.7).

    Args:
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.
        price_df: The DataFrame returned by
            ``sec_analyzer.fetch.prices.get_price_history`` (Date index,
            ``Close`` column), or ``None``.

    Returns:
        A list of ``{"fy", "end", "price", "pe", "ps", "pfcf", "ev_sales"}``
        dicts sorted by ``fy`` ascending. Empty list if no fiscal year has
        both a period-end date and price coverage. Never raises.
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
    ltd_series = to_annual_series(normalized, "LongTermDebt")
    ltdc_series = to_annual_series(normalized, "LongTermDebtCurrent")
    cash_series = to_annual_series(normalized, "Cash")
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

        # EV/Sales: net debt = long-term debt (noncurrent + current) minus
        # cash; when NONE of those three concepts is present for this fiscal
        # year, EV degrades to market cap (unlevered), so ev_sales == ps.
        ev_sales = None
        if revenue is not None and revenue > 0 and shares:
            ltd = ltd_series.get(fy)
            ltdc = ltdc_series.get(fy)
            cash = cash_series.get(fy)
            if ltd is None and ltdc is None and cash is None:
                net_debt = 0.0
            else:
                net_debt = (ltd or 0.0) + (ltdc or 0.0) - (cash or 0.0)
            ev_sales = (fy_price * shares + net_debt) / revenue

        history.append(
            {"fy": fy, "end": period_end, "price": fy_price, "pe": pe, "ps": ps, "pfcf": pfcf, "ev_sales": ev_sales}
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


def forward_revenue_cagr(
    revenue_series: Dict[int, Optional[float]],
    fy: int,
    years: int = _GROWTH_ADJ_FORWARD_YEARS,
) -> Optional[float]:
    """Realized revenue CAGR over the ``years`` fiscal years *following* ``fy``.

    ``(revenue_{fy+years} / revenue_fy) ** (1/years) - 1``, computed only
    when both endpoints are present in ``revenue_series`` AND strictly
    positive (a CAGR across a zero/negative endpoint isn't meaningful).
    Returns a decimal fraction (e.g. ``0.15`` for 15%), or ``None``.

    This is the denominator each historical year's multiple is
    growth-adjusted by (see :func:`growth_adjusted_history`): the multiple
    the market assigned at ``fy``'s period-end, divided by the growth that
    year actually went on to deliver.
    """
    start = revenue_series.get(fy)
    end = revenue_series.get(fy + years)
    if start is not None and end is not None and start > 0 and end > 0:
        return (end / start) ** (1.0 / years) - 1.0
    return None


def growth_adjusted_value(
    multiple: Optional[float],
    growth_fraction: Optional[float],
    min_growth: float = _PEG_MIN_GROWTH,
) -> Optional[float]:
    """Growth-adjusted multiple: ``multiple / (growth_fraction * 100)``.

    Divides a raw multiple (P/E for PEG, EV/Sales for the hyper-grower
    growth-adjusted sales multiple) by growth expressed in *percentage
    points* (so a 15% growth denominator is ``15``, yielding the familiar
    "PEG ~1" scale), matching the PEG convention.

    Returns ``None`` -- i.e. "not applicable", never a negative or exploded
    figure -- unless the multiple is present and strictly positive AND
    ``growth_fraction`` is present and at least ``min_growth`` (5% by
    default). The floor is what keeps the PEG linearity flaw from producing
    nonsense as the denominator approaches zero (VALUATION.md Sec.7).

    Args:
        multiple: The raw multiple (e.g. current P/E), or ``None``.
        growth_fraction: The growth rate as a decimal fraction (e.g. ``0.15``
            for 15%), typically the assumptions pipeline's base ``growth_5y``.
        min_growth: Minimum growth fraction below which ``None`` is returned.

    Returns:
        The growth-adjusted multiple rounded to 2 decimals, or ``None``.
    """
    if multiple is None or multiple <= 0:
        return None
    if growth_fraction is None or growth_fraction < min_growth:
        return None
    return round(multiple / (growth_fraction * 100.0), 2)


def growth_adjusted_history(
    history: List[dict],
    revenue_series: Dict[int, Optional[float]],
    multiple_key: str,
    min_growth: float = _PEG_MIN_GROWTH,
) -> List[float]:
    """Historical growth-adjusted multiple series for percentile ranking.

    For every year in ``history``, divides that year's ``multiple_key``
    multiple (``"pe"`` for PEG, ``"ev_sales"`` for the hyper-grower sales
    multiple) by the revenue CAGR that year went on to realize over the
    following :data:`_GROWTH_ADJ_FORWARD_YEARS` fiscal years (see
    :func:`forward_revenue_cagr`), expressed in percentage points.

    Only fiscal years with a *complete* pairing -- a positive multiple AND a
    forward CAGR at or above ``min_growth`` -- contribute a value; every
    other year is simply omitted (not emitted as ``None``), so the returned
    list is ready to hand straight to :func:`percentile_position`. The most
    recent ~3 fiscal years naturally drop out (their forward window isn't
    complete yet), which is expected.

    Args:
        history: The list returned by :func:`multiples_history`.
        revenue_series: The ``{fy: revenue}`` annual series (from
            ``normalize.normalizer.to_annual_series(normalized, "Revenue")``).
        multiple_key: Which per-year multiple to growth-adjust (``"pe"`` or
            ``"ev_sales"``).
        min_growth: Minimum forward-CAGR fraction for a year to qualify.

    Returns:
        A list of growth-adjusted multiple floats (already ``None``-free).
        Never raises.
    """
    values: List[float] = []
    for row in (history or []):
        fy = row.get("fy")
        if fy is None:
            continue
        multiple = row.get(multiple_key)
        fwd_cagr = forward_revenue_cagr(revenue_series, fy)
        adjusted = growth_adjusted_value(multiple, fwd_cagr, min_growth)
        if adjusted is not None:
            values.append(adjusted)
    return values
