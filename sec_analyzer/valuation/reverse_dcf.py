"""Reverse DCF: solve for the growth rate the current price implies.

Instead of asking "what is fair value given an assumed growth rate", reverse
DCF asks "what constant 5-year growth rate would the DCF model need in order
to produce today's market price". This is a plain bisection over
``dcf.dcf_per_share``'s ``growth_5y`` input, holding the base scenario's
discount rate and terminal growth fixed -- deterministic and dependency-free
(no numeric libraries beyond pure Python).
"""

import logging
from typing import Optional

from sec_analyzer.valuation.dcf import dcf_per_share

logger = logging.getLogger(__name__)

#: Bisection bracket for growth_5y (decimal fractions): -20% .. +60%.
_BRACKET_LO = -0.20
_BRACKET_HI = 0.60

#: Stop once the bracket half-width is below this, or after _MAX_ITERATIONS.
_TOLERANCE = 1e-4
_MAX_ITERATIONS = 80


def _per_share_diff(
    growth_5y: float,
    price: float,
    fcf0: float,
    terminal_growth: float,
    discount_rate: float,
    shares: float,
    dilution_rate: float,
) -> Optional[float]:
    """Return ``dcf_per_share(...) - price`` at a candidate ``growth_5y``.

    Returns ``None`` if ``dcf_per_share`` can't be evaluated at this point
    (it only raises for fixed inputs -- ``fcf0``/``shares``/``discount_rate``
    vs. ``terminal_growth`` -- none of which vary during the bisection, but
    the guard is kept here so a single unusable input degrades to "no
    result" rather than propagating an exception out of the solver).
    """
    try:
        result = dcf_per_share(fcf0, growth_5y, terminal_growth, discount_rate, shares, dilution_rate)
    except ValueError:
        return None
    return result["per_share"] - price


#: Status values returned by :func:`implied_growth_with_status`.
STATUS_OK = "ok"
STATUS_ABOVE_BRACKET = "above_bracket"
STATUS_BELOW_BRACKET = "below_bracket"
STATUS_NO_DATA = "no_data"


def implied_growth_with_status(
    price: Optional[float],
    fcf0: Optional[float],
    terminal_growth: float,
    discount_rate: float,
    shares: Optional[float],
    dilution_rate: float = 0.0,
) -> "tuple[Optional[float], str]":
    """Bisect for ``growth_5y`` like :func:`implied_growth`, plus *why* a
    ``None`` happened.

    Same bisection as :func:`implied_growth` (base scenario's fixed
    ``discount_rate``/``terminal_growth``, bracket ``[-0.20, 0.60]``,
    tolerance ``1e-4``/80 iterations), but additionally classifies the
    "no root in the bracket" case: since the DCF per-share value is
    monotonically increasing in ``growth_5y``, a lack of sign change across
    the bracket's ends means the market price sits entirely on one side of
    every value the model can produce there.

    Args:
        price: Current market price per share.
        fcf0: Base-year free cash flow (see ``dcf.dcf_per_share``).
        terminal_growth: Base-scenario terminal growth rate.
        discount_rate: Base-scenario discount rate.
        shares: Diluted shares outstanding.
        dilution_rate: Annual dilution rate (see ``dcf.dcf_per_share``).

    Returns:
        A ``(growth, status)`` tuple. ``status`` is one of:

        * :data:`STATUS_OK` -- a root was found (or the price sits exactly
          on a bracket endpoint); ``growth`` is the implied ``growth_5y``
          (decimal fraction, rounded to 4 decimals).
        * :data:`STATUS_ABOVE_BRACKET` -- no sign change, and the model
          per-share value stays *below* the market price at both bracket
          ends (``diff_hi < 0``): the price implies growth above the
          bracket's +60% ceiling. ``growth`` is ``None``.
        * :data:`STATUS_BELOW_BRACKET` -- no sign change, and the model
          per-share value stays *above* the market price at both ends
          (``diff_lo > 0``): the price implies growth below the bracket's
          -20% floor. ``growth`` is ``None``.
        * :data:`STATUS_NO_DATA` -- a required input is unusable (missing
          price/fcf0, non-positive/missing shares,
          ``discount_rate <= terminal_growth``), or ``dcf_per_share``
          couldn't be evaluated at all. ``growth`` is ``None``.
    """
    if price is None or price <= 0 or fcf0 is None:
        return None, STATUS_NO_DATA
    if not shares or shares <= 0:
        return None, STATUS_NO_DATA
    if discount_rate <= terminal_growth:
        return None, STATUS_NO_DATA

    diff_lo = _per_share_diff(_BRACKET_LO, price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)
    diff_hi = _per_share_diff(_BRACKET_HI, price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)
    if diff_lo is None or diff_hi is None:
        return None, STATUS_NO_DATA
    if diff_lo == 0:
        return round(_BRACKET_LO, 4), STATUS_OK
    if diff_hi == 0:
        return round(_BRACKET_HI, 4), STATUS_OK
    if (diff_lo > 0) == (diff_hi > 0):
        # No sign change across the bracket -- the target price isn't
        # reachable by any growth rate in [-20%, 60%] at this r/g_t.
        return (None, STATUS_ABOVE_BRACKET) if diff_hi < 0 else (None, STATUS_BELOW_BRACKET)

    lo, hi = _BRACKET_LO, _BRACKET_HI
    for _ in range(_MAX_ITERATIONS):
        mid = (lo + hi) / 2.0
        diff_mid = _per_share_diff(mid, price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)
        if diff_mid is None:
            return None, STATUS_NO_DATA
        if diff_mid == 0 or (hi - lo) / 2.0 < _TOLERANCE:
            return round(mid, 4), STATUS_OK
        if (diff_mid > 0) == (diff_lo > 0):
            lo, diff_lo = mid, diff_mid
        else:
            hi, diff_hi = mid, diff_mid

    return round((lo + hi) / 2.0, 4), STATUS_OK


def implied_growth(
    price: Optional[float],
    fcf0: Optional[float],
    terminal_growth: float,
    discount_rate: float,
    shares: Optional[float],
    dilution_rate: float = 0.0,
) -> Optional[float]:
    """Bisect for the ``growth_5y`` that makes the DCF per-share price match ``price``.

    Thin wrapper around :func:`implied_growth_with_status` that drops the
    status and keeps returning exactly the same growth value it always has
    -- kept as the stable, backward-compatible entry point for callers that
    only need the number, not the bracket-boundary diagnosis.

    Args:
        price: Current market price per share.
        fcf0: Base-year free cash flow (see ``dcf.dcf_per_share``).
        terminal_growth: Base-scenario terminal growth rate.
        discount_rate: Base-scenario discount rate.
        shares: Diluted shares outstanding.
        dilution_rate: Annual dilution rate (see ``dcf.dcf_per_share``).

    Returns:
        The implied ``growth_5y`` (decimal fraction, rounded to 4 decimals),
        or ``None`` if any required input is unusable (missing price/fcf0,
        non-positive/missing shares, ``discount_rate <= terminal_growth``)
        or if the DCF per-share value doesn't change sign across the
        bracket (no root to find -- e.g. the price is unreachable at either
        extreme of the growth range).
    """
    growth, _status = implied_growth_with_status(price, fcf0, terminal_growth, discount_rate, shares, dilution_rate)
    return growth
