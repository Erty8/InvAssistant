"""LBO-implied floor value (SPEC.md Sec.8h): a private-equity-return-based
value floor, independent of the public-market DCF/multiples/reverse-DCF
triangulation.

The idea (standard LBO "deleveraging return" logic): a financial (private
equity) buyer doesn't need organic growth OR multiple expansion to earn a
return -- paying down acquisition debt out of the company's own free cash
flow over a hold period mechanically grows the buyer's equity stake even if
EBITDA and the exit multiple are both held FLAT (a deliberately conservative
assumption). Discounting the resulting exit equity value back to today at
the sponsor's minimum acceptable return (hurdle IRR) gives the highest price
a disciplined financial buyer could justify paying today -- a defensible
value FLOOR, distinct from (and not a substitute for) the public-market DCF
anchor.

This is advisory only (SPEC.md Sec.8h): it never headlines
``fair_value_range`` and never participates in
``triangulate.triangulate``'s confidence vote.
"""

import logging
from typing import List, Optional

from sec_analyzer.valuation.dcf import project_fcf

logger = logging.getLogger(__name__)

#: Sponsor hurdle rate (minimum acceptable IRR) this floor targets -- a
#: conservative, generic private-equity return threshold, not tailored per
#: sector/deal (SPEC.md Sec.8h).
_LBO_TARGET_IRR = 0.20

#: Hold period, in years. Kept <= the DCF's high-growth-year window (5) so a
#: flat ``fcf_growth`` passed as both ``growth_5y`` and ``terminal_growth``
#: to :func:`project_fcf` never triggers that function's fade logic.
_LBO_HOLD_YEARS = 5


def lbo_implied_floor_per_share(
    ebitda: Optional[float],
    entry_multiple: Optional[float],
    existing_debt: Optional[float],
    exit_multiple: Optional[float],
    fcf0: Optional[float],
    fcf_growth: float,
    shares: Optional[float],
    target_irr: float = _LBO_TARGET_IRR,
    hold_years: int = _LBO_HOLD_YEARS,
) -> Optional[dict]:
    """Compute the LBO-implied per-share value floor (SPEC.md Sec.8h).

    Mechanics:

    1. ``entry_ev = entry_multiple * ebitda``; ``entry_equity = entry_ev -
       existing_debt`` (today's implied sponsor equity check, at
       ``entry_multiple`` -- surfaced for display, not itself the floor).
    2. Free cash flow is projected ``hold_years`` forward via
       :func:`project_fcf` (``fcf0``, growing at the flat ``fcf_growth``
       rate every year -- passing the SAME rate as both ``growth_5y`` and
       ``terminal_growth`` keeps the projection flat with no fade, since
       ``hold_years <= 5`` never reaches :func:`project_fcf`'s fade phase).
    3. Debt is paid down via a 100% cash sweep: each year's entire projected
       FCF reduces the outstanding balance, floored at ``0.0`` (a sponsor
       never owes negative debt).
    4. ``exit_ev = exit_multiple * ebitda`` -- EBITDA is held FLAT over the
       hold period (no organic-growth credit; this is what makes the result
       a conservative FLOOR rather than a base-case LBO return).
       ``exit_equity = exit_ev - remaining_debt``.
    5. ``floor_equity_today = exit_equity / (1 + target_irr) ** hold_years``
       -- discounting the exit equity back to today at the hurdle IRR gives
       the highest entry price a disciplined financial buyer could justify.
    6. ``per_share = floor_equity_today / shares``.

    Args:
        ebitda: Current-year EBITDA (operating income + D&A).
        entry_multiple: EV/EBITDA multiple the entry enterprise value is
            based on (the caller typically passes the filer's OWN current
            EV/EBITDA -- "could a sponsor justify today's price").
        existing_debt: Total debt at entry (the caller typically passes
            ``metrics["total_debt"]``).
        exit_multiple: EV/EBITDA multiple assumed at exit. Passing the SAME
            value as ``entry_multiple`` (no multiple expansion) is the
            standard conservative assumption; the caller may pass a
            different value if it has a specific view.
        fcf0: Base-year free cash flow the debt-paydown projection starts
            from.
        fcf_growth: Flat annual FCF growth rate applied every hold year (see
            the fade note above).
        shares: Diluted shares outstanding.
        target_irr: The sponsor's hurdle rate (decimal fraction). Defaults
            to :data:`_LBO_TARGET_IRR` (20%).
        hold_years: The hold period, in years. Defaults to
            :data:`_LBO_HOLD_YEARS` (5).

    Returns:
        ``None`` (never raises, never fabricates) when ``ebitda`` is
        missing/``<= 0``, ``existing_debt`` is missing/``< 0``,
        ``entry_multiple``/``exit_multiple`` is missing/``<= 0``, ``fcf0``
        is missing, ``shares`` is falsy/``<= 0``, ``target_irr <= -1``
        (a degenerate discount factor), or ``hold_years <= 0``. Otherwise a
        dict with ``per_share``, ``entry_ev``, ``entry_equity``, ``exit_ev``,
        ``exit_equity``, ``remaining_debt``, ``fcf_path`` (``hold_years``
        floats), ``debt_path`` (``hold_years`` floats, end-of-year balance),
        and ``floor_equity_today``. Nothing is rounded here -- rounding is
        the caller's (``engine.py``'s) responsibility.
    """
    if ebitda is None or ebitda <= 0:
        return None
    if existing_debt is None or existing_debt < 0:
        return None
    if entry_multiple is None or entry_multiple <= 0:
        return None
    if exit_multiple is None or exit_multiple <= 0:
        return None
    if fcf0 is None:
        return None
    if not shares or shares <= 0:
        return None
    if target_irr <= -1:
        return None
    if hold_years <= 0:
        return None

    entry_ev = entry_multiple * ebitda
    entry_equity = entry_ev - existing_debt

    fcf_path = project_fcf(fcf0, fcf_growth, fcf_growth, years=hold_years)

    debt_path: List[float] = []
    remaining_debt = existing_debt
    for fcf_year in fcf_path:
        remaining_debt = max(0.0, remaining_debt - fcf_year)
        debt_path.append(remaining_debt)

    exit_ev = exit_multiple * ebitda
    exit_equity = exit_ev - remaining_debt

    floor_equity_today = exit_equity / (1 + target_irr) ** hold_years
    per_share = floor_equity_today / shares

    return {
        "per_share": per_share,
        "entry_ev": entry_ev,
        "entry_equity": entry_equity,
        "exit_ev": exit_ev,
        "exit_equity": exit_equity,
        "remaining_debt": remaining_debt,
        "fcf_path": fcf_path,
        "debt_path": debt_path,
        "floor_equity_today": floor_equity_today,
    }
