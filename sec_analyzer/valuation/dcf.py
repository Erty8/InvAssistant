"""Deterministic discounted cash flow (DCF) valuation.

A single, plain 10-year, two-stage FCF projection: years 1-5 grow at a
constant rate, years 6-10 fade linearly to a terminal growth rate, and the
year-10 cash flow anchors a Gordon-growth terminal value. Nothing here talks
to the network, a database, or an LLM -- given the same numeric inputs this
always returns the same numbers (see ``sec_analyzer/valuation/SPEC.md``
Sec.4 for the binding formulas).

This module never silently "fixes" an invalid input (e.g. a discount rate at
or below the terminal growth rate, which makes the Gordon-growth terminal
value mathematically undefined): it raises :class:`ValueError` instead, and
it is the caller's (``engine.py``'s) job to catch that and turn it into a
Turkish note rather than letting it propagate to the CLI.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

#: Total DCF projection horizon, in years.
HORIZON_YEARS = 10

#: Number of years the growth rate stays constant at ``growth_5y`` before
#: fading toward ``terminal_growth``.
_HIGH_GROWTH_YEARS = 5

#: Number of years used to derive the "effective" (dilution-adjusted) share
#: count -- see the ``dcf_per_share`` docstring for why the mid-horizon year
#: (5) is used rather than year 0 or year 10.
_DILUTION_HORIZON_YEARS = 5


def _year_growth_rate(year: int, growth_5y: float, terminal_growth: float) -> float:
    """Return the growth rate applied to project year ``year``'s FCF.

    Years 1-5 use the constant ``growth_5y``. Years 6-10 fade linearly from
    ``growth_5y`` down (or up) to ``terminal_growth``, reaching exactly
    ``terminal_growth`` at year 10:
    ``g = growth_5y + (terminal_growth - growth_5y) * (year - 5) / 5``.
    """
    if year <= _HIGH_GROWTH_YEARS:
        return growth_5y
    return growth_5y + (terminal_growth - growth_5y) * (year - _HIGH_GROWTH_YEARS) / _HIGH_GROWTH_YEARS


def project_fcf(
    fcf0: float, growth_5y: float, terminal_growth: float, years: int = HORIZON_YEARS
) -> List[float]:
    """Project ``years`` of free cash flow forward from a base ``fcf0``.

    Args:
        fcf0: Base-year (year 0) free cash flow. Not included in the
            returned path -- the path starts at year 1.
        growth_5y: Constant annual growth rate applied to years 1-5 (decimal
            fraction, e.g. ``0.08`` for 8%).
        terminal_growth: Growth rate the projection fades to by year 10 (and
            uses for the terminal value beyond the horizon).
        years: Projection horizon. Defaults to :data:`HORIZON_YEARS` (10);
            only the caller-facing default matters for the documented
            10-year DCF contract, but the loop itself is generic.

    Returns:
        A list of ``years`` floats: ``[fcf_1, fcf_2, ..., fcf_years]``. Each
        year compounds off the *previous projected year* (not off ``fcf0``
        directly), i.e. ``fcf_y = fcf_{y-1} * (1 + g_y)``.
    """
    fcf_path: List[float] = []
    previous = fcf0
    for year in range(1, years + 1):
        growth_rate = _year_growth_rate(year, growth_5y, terminal_growth)
        fcf_year = previous * (1 + growth_rate)
        fcf_path.append(fcf_year)
        previous = fcf_year
    return fcf_path


def dcf_per_share(
    fcf0: Optional[float],
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares: Optional[float],
    dilution_rate: float = 0.0,
) -> dict:
    """Compute a per-share DCF fair value from a base FCF and growth path.

    Ten-year, two-stage projection (see :func:`project_fcf`): years 1-5 grow
    at ``growth_5y``, years 6-10 fade linearly to ``terminal_growth``. Each
    projected year is discounted at ``discount_rate``; a Gordon-growth
    terminal value is anchored on year 10's cash flow and discounted back
    the same 10 years. The sum of the discounted cash flows and discounted
    terminal value is divided by an "effective" share count to get equity
    value per share directly.

    FCFE-direct, no net-debt subtraction: ``fcf0`` (FCF = OCF - CapEx, per
    US GAAP) is already a *levered* (equity) cash flow, since interest paid
    to debtholders is deducted inside operating cash flow before it ever
    reaches this projection. Its present value is therefore already an
    equity value -- subtracting net debt from it a second time would
    double-penalize leverage (once through the interest expense embedded in
    every projected year's FCF, once again as a lump-sum balance-sheet
    deduction from the PV). So ``ev`` and ``equity`` below are the same
    number; both keys are kept for backward-compatible callers that read
    either one. Consequently ``discount_rate`` must be a levered cost of
    equity, not a WACC: discounting an already-levered equity cash flow at a
    WACC would double-count the leverage adjustment a WACC already bakes in.

    Dilution: ``effective_shares = shares * (1 + dilution_rate) ** 5``. Five
    years (the midpoint of the 10-year horizon, not year 0 or year 10) is
    used as a deliberate simplification: a decade-out dilution/buyback path
    is inherently uncertain, so compounding to the horizon's midpoint
    approximates the average per-share dilutive drag over the projection
    without pretending to know the exact annual share count.

    Args:
        fcf0: Base-year free cash flow the projection starts from.
        growth_5y: Constant growth rate for projection years 1-5 (decimal
            fraction).
        terminal_growth: Growth rate years 6-10 fade to, and the Gordon-
            growth terminal-value growth rate.
        discount_rate: Annual discount rate (decimal fraction) -- a levered
            COST OF EQUITY, not a WACC (see the FCFE-direct note above: the
            discounted cash flow is already an equity cash flow, so the
            rate that discounts it must be the rate equity holders require,
            not a blend with the cost of debt).
        shares: Diluted shares outstanding used as the pre-dilution base.
        dilution_rate: Annual share-count growth rate applied over 5 years
            to derive the effective share count. Defaults to ``0.0`` (no
            dilution adjustment).

    Returns:
        A dict with keys ``per_share``, ``ev``, ``equity`` (equal, see
        above), ``fcf_path`` (the 10 projected FCF floats from
        :func:`project_fcf`), ``tv`` (undiscounted terminal value), and
        ``effective_shares``. Nothing is rounded here -- rounding is the
        caller's (``engine.py``'s) responsibility, since intermediate
        callers (reverse-DCF bisection, the sensitivity grid) need full
        precision.

    Raises:
        ValueError: If ``fcf0`` is ``None``, if ``shares`` is falsy or
            ``<= 0``, or if ``discount_rate <= terminal_growth`` (the
            Gordon-growth terminal value is undefined in that case --this is
            never silently "fixed").
    """
    if fcf0 is None:
        raise ValueError("dcf_per_share: fcf0 is None; cannot project a cash-flow path without a base FCF.")
    if not shares or shares <= 0:
        raise ValueError(f"dcf_per_share: shares must be a positive number, got {shares!r}.")
    if discount_rate <= terminal_growth:
        raise ValueError(
            f"dcf_per_share: discount_rate ({discount_rate}) must be strictly greater than "
            f"terminal_growth ({terminal_growth}) for the Gordon-growth terminal value to be defined."
        )

    fcf_path = project_fcf(fcf0, growth_5y, terminal_growth, HORIZON_YEARS)

    pv_sum = 0.0
    for year, fcf_year in enumerate(fcf_path, start=1):
        pv_sum += fcf_year / (1 + discount_rate) ** year

    fcf_terminal = fcf_path[-1]
    tv = fcf_terminal * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_tv = tv / (1 + discount_rate) ** HORIZON_YEARS

    ev = pv_sum + pv_tv
    equity = ev  # FCFE-direct: no net-debt subtraction, see the docstring above.

    effective_shares = shares * (1 + dilution_rate) ** _DILUTION_HORIZON_YEARS
    per_share = equity / effective_shares

    return {
        "per_share": per_share,
        "ev": ev,
        "equity": equity,
        "fcf_path": fcf_path,
        "tv": tv,
        "effective_shares": effective_shares,
    }
