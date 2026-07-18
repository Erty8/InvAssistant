"""Revenue-first DCF for hyper-growth, margin-suppressed companies.

A hyper-growth company's near-term FCF is *suppressed* by choice (growth
spend -- R&D, S&M, SBC -- is expensed rather than capitalized), so a
standard FCF-DCF (see ``sec_analyzer/valuation/dcf.py``) that grows today's
depressed cash flow at a clamped rate systematically undervalues it. This
module instead projects *revenue* forward on a growth path that fades
(mean-reverts) linearly from a realized/near-term growth rate down to a
terminal growth rate, and lets the FCF margin converge linearly to a
data-derived mature target over the same window. The fade itself -- not an
arbitrary flat growth cap -- is the constraint that keeps the projection
honest (see ``sec_analyzer/valuation/SPEC.md`` and ``VALUATION.md`` Sec.4a
for the methodology this implements).

Nothing here talks to the network, a database, or an LLM -- given the same
numeric inputs this always returns the same numbers. Like ``dcf.py``, this
module does not silently "fix" a mathematically-undefined or
programmer-error input (non-positive ``revenue0``/``shares0``, a discount
rate at or below the terminal growth rate, a non-positive
``steady_state_year``): it raises :class:`ValueError` instead, and it is the
caller's (``engine.py``'s) job to catch that and turn it into a Turkish note
rather than letting it propagate to the CLI.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

#: Total revenue-first DCF projection horizon, in years.
HORIZON_YEARS = 10

#: Bisection bracket for ``start_growth`` in :func:`implied_start_growth`
#: (decimal fractions): hyper-growers can imply a starting growth rate well
#: above the ``dcf.py``/``reverse_dcf.py`` bracket's +40% ceiling, so this
#: one runs wider, -20% .. +60%.
_START_GROWTH_BRACKET_LO = -0.20
_START_GROWTH_BRACKET_HI = 0.60

#: Bisection bracket for ``target_fcf_margin`` in :func:`implied_target_margin`
#: (decimal fractions): 0% .. 90% mature FCF margin.
_TARGET_MARGIN_BRACKET_LO = 0.0
_TARGET_MARGIN_BRACKET_HI = 0.90

#: Stop once the bracket half-width is below this, or after _MAX_ITERATIONS.
_TOLERANCE = 1e-4
_MAX_ITERATIONS = 80


def _growth_path(start_growth: float, terminal_growth: float, steady_state_year: int, horizon: int) -> List[float]:
    """Return the year-1..``horizon`` growth-rate path (linear fade).

    Fades linearly from ``start_growth`` (year 1) to ``terminal_growth``,
    reaching ``terminal_growth`` exactly at ``steady_state_year`` and
    staying there for any remaining years:
    ``g_t = start_growth + (terminal_growth - start_growth) *
    min(t-1, steady_state_year-1) / (steady_state_year-1)``. When
    ``steady_state_year == 1`` every year uses ``terminal_growth`` (the
    fade collapses to a single point, avoiding a division by zero).
    """
    path: List[float] = []
    for t in range(1, horizon + 1):
        if steady_state_year <= 1:
            path.append(terminal_growth)
            continue
        fraction = min(t - 1, steady_state_year - 1) / (steady_state_year - 1)
        path.append(start_growth + (terminal_growth - start_growth) * fraction)
    return path


def _margin_path(current_margin: float, target_fcf_margin: float, steady_state_year: int, horizon: int) -> List[float]:
    """Return the year-1..``horizon`` FCF-margin path (linear convergence).

    Converges linearly from ``current_margin`` (today's, possibly
    negative, FCF margin) to ``target_fcf_margin``, reaching the target
    exactly at ``steady_state_year`` and staying there afterward:
    ``margin_t = current_margin + (target_fcf_margin - current_margin) *
    min(t, steady_state_year) / steady_state_year``.
    """
    path: List[float] = []
    for t in range(1, horizon + 1):
        fraction = min(t, steady_state_year) / steady_state_year
        path.append(current_margin + (target_fcf_margin - current_margin) * fraction)
    return path


def _discount_path(
    discount_rate: float, mature_discount_rate: float, steady_state_year: int, horizon: int
) -> List[float]:
    """Return the year-1..``horizon`` discount-rate path (linear fade).

    Mirrors :func:`_growth_path`'s exact fade shape: fades linearly from
    ``discount_rate`` (year 1) to ``mature_discount_rate``, reaching
    ``mature_discount_rate`` exactly at ``steady_state_year`` and staying
    there for any remaining years: ``r_t = discount_rate +
    (mature_discount_rate - discount_rate) * min(t-1, steady_state_year-1)
    / (steady_state_year-1)``. When ``steady_state_year == 1`` every year
    uses ``mature_discount_rate`` (the fade collapses to a single point,
    avoiding a division by zero) -- same collapse rule as
    :func:`_growth_path`.

    This models Damodaran's standard fix for a hyper-grower DCF that fades
    its cash flows toward a mature steady state while discounting every
    year at a fixed, permanently-elevated cohort rate: since most of a
    hyper-grower's value sits in the far years and the terminal value, a
    flat high rate systematically crushes it even after the cash flows
    themselves have already matured. Fading the discount rate alongside
    the cash flows keeps the risk price consistent with the risk being
    priced.
    """
    path: List[float] = []
    for t in range(1, horizon + 1):
        if steady_state_year <= 1:
            path.append(mature_discount_rate)
            continue
        fraction = min(t - 1, steady_state_year - 1) / (steady_state_year - 1)
        path.append(discount_rate + (mature_discount_rate - discount_rate) * fraction)
    return path


def revenue_first_dcf(
    revenue0: float,
    start_growth: float,
    terminal_growth: float,
    discount_rate: float,
    current_margin: float,
    target_fcf_margin: float,
    steady_state_year: int,
    shares0: float,
    annual_dilution: float,
    financing_shares: float = 0.0,
    horizon: int = HORIZON_YEARS,
    mature_discount_rate: Optional[float] = None,
) -> dict:
    """Project revenue (not FCF) forward on a fading growth path and value it.

    Two linear paths drive the projection over ``horizon`` years: a
    **growth** path that fades from ``start_growth`` (year 1) to
    ``terminal_growth`` by ``steady_state_year``, and a **margin** path that
    converges from ``current_margin`` (today's, possibly non-positive, FCF
    margin) to ``target_fcf_margin`` over the same window (see
    :func:`_growth_path` / :func:`_margin_path`). Revenue compounds off the
    previous projected year (``revenue_t = revenue_{t-1} * (1 + g_t)``,
    ``revenue_0 = revenue0``); each year's FCF is ``revenue_t * margin_t``.
    Each projected FCF is discounted at ``discount_rate``; a Gordon-growth
    terminal value is anchored on the horizon's final FCF and discounted
    back the same number of years. FCFE-direct, like ``dcf.dcf_per_share``:
    no net-debt subtraction -- the projected FCF is already a levered
    (equity) cash flow (interest is paid out of operating cash flow before
    it reaches this projection), so its discounted sum is divided directly
    by an "effective" (dilution- and financing-adjusted) share count for
    the per-share result. Consequently ``discount_rate`` must be a levered
    cost of equity, not a WACC -- discounting an already-levered equity cash
    flow at a WACC would double-count the leverage adjustment a WACC
    already bakes in.

    Args:
        revenue0: Base-year (year 0) revenue. Must be positive.
        start_growth: Year-1 revenue growth rate (decimal fraction), e.g.
            the realized recent-growth rate the fade starts from.
        terminal_growth: Growth rate the projection fades to by
            ``steady_state_year`` (and the Gordon-growth terminal-value
            growth rate).
        discount_rate: Annual discount rate (decimal fraction) -- a levered
            cost of equity, not a WACC (see the FCFE-direct note above).
            Must be strictly greater than ``terminal_growth``.
        current_margin: Today's FCF margin (decimal fraction; may be zero
            or negative for a cash-burning company).
        target_fcf_margin: Mature-state FCF margin the margin path
            converges to.
        steady_state_year: Year by which both the growth and margin paths
            have fully converged (``<= horizon``). Must be ``>= 1``.
        shares0: Base share count. Must be positive.
        annual_dilution: Geometric annual share-count growth rate
            (decimal fraction, ``>= 0``) compounded over
            ``steady_state_year`` years to derive the effective share
            count.
        financing_shares: Extra shares assumed issued to fund cumulative
            cash burn during the projection (see ``engine.py`` for how
            this is derived from the base scenario's negative-FCF years).
            Defaults to ``0.0``.
        horizon: Projection horizon in years. Defaults to
            :data:`HORIZON_YEARS` (10).
        mature_discount_rate: Optional mature (steady-state) discount rate
            (decimal fraction) to fade toward (Damodaran fade, WP3). When
            ``None`` (the default), behavior is byte-for-byte unchanged from
            before this parameter existed: every year is discounted at the
            flat ``discount_rate``, and the terminal value also uses
            ``discount_rate``. When a number, ``discount_rate`` is treated
            as the YEAR-1 (cohort) rate and :func:`_discount_path` builds a
            year-1..``horizon`` rate path that fades linearly from
            ``discount_rate`` to ``mature_discount_rate`` by
            ``steady_state_year`` (mirroring the growth/margin fades above)
            and stays at ``mature_discount_rate`` afterward; each year's
            discount factor is then the CUMULATIVE product of
            ``(1 + r_t)`` (not ``(1 + r) ** t``), and the Gordon-growth
            terminal value -- itself a mature-firm perpetuity -- is
            discounted at ``mature_discount_rate`` rather than the
            elevated cohort rate. Must be strictly greater than
            ``terminal_growth`` when provided.

    Returns:
        A dict with keys ``per_share``, ``ev``, ``equity``,
        ``revenue_path`` (``horizon`` floats), ``fcf_path`` (``horizon``
        floats), ``margin_path`` (``horizon`` floats), ``growth_path``
        (``horizon`` floats), ``tv`` (undiscounted terminal value),
        ``effective_shares``, ``final_year_revenue`` (``revenue_path``'s
        last entry), and ``revenue_multiple`` (``final_year_revenue /
        revenue0``). When ``mature_discount_rate`` is not ``None``, an
        additional ``discount_path`` key (``horizon`` floats, see above)
        is also present. Nothing is rounded here -- rounding is the
        caller's responsibility.

    Raises:
        ValueError: If ``revenue0 <= 0``, ``shares0 <= 0``,
            ``discount_rate <= terminal_growth``, ``steady_state_year < 1``,
            or (when ``mature_discount_rate`` is not ``None``)
            ``mature_discount_rate <= terminal_growth``.
    """
    if revenue0 <= 0:
        raise ValueError(f"revenue_first_dcf: revenue0 must be positive, got {revenue0!r}.")
    if shares0 <= 0:
        raise ValueError(f"revenue_first_dcf: shares0 must be positive, got {shares0!r}.")
    if discount_rate <= terminal_growth:
        raise ValueError(
            f"revenue_first_dcf: discount_rate ({discount_rate}) must be strictly greater than "
            f"terminal_growth ({terminal_growth}) for the Gordon-growth terminal value to be defined."
        )
    if mature_discount_rate is not None and mature_discount_rate <= terminal_growth:
        raise ValueError(
            f"revenue_first_dcf: mature_discount_rate ({mature_discount_rate}) must be strictly greater "
            f"than terminal_growth ({terminal_growth}) for the Gordon-growth terminal value to be defined."
        )
    if steady_state_year < 1:
        raise ValueError(f"revenue_first_dcf: steady_state_year must be >= 1, got {steady_state_year!r}.")

    growth_path = _growth_path(start_growth, terminal_growth, steady_state_year, horizon)
    margin_path = _margin_path(current_margin, target_fcf_margin, steady_state_year, horizon)

    revenue_path: List[float] = []
    previous_revenue = revenue0
    for growth_rate in growth_path:
        revenue_year = previous_revenue * (1 + growth_rate)
        revenue_path.append(revenue_year)
        previous_revenue = revenue_year

    fcf_path = [revenue_year * margin_year for revenue_year, margin_year in zip(revenue_path, margin_path)]

    fcf_terminal = fcf_path[-1]

    if mature_discount_rate is None:
        pv_sum = 0.0
        for year, fcf_year in enumerate(fcf_path, start=1):
            pv_sum += fcf_year / (1 + discount_rate) ** year
        tv = fcf_terminal * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_tv = tv / (1 + discount_rate) ** horizon
        discount_path = None
    else:
        discount_path = _discount_path(discount_rate, mature_discount_rate, steady_state_year, horizon)
        pv_sum = 0.0
        cumulative_df = 1.0
        for fcf_year, rate_year in zip(fcf_path, discount_path):
            cumulative_df *= 1 + rate_year
            pv_sum += fcf_year / cumulative_df
        # cumulative_df now equals df_horizon (the product over all `horizon`
        # years) -- reused below for the terminal value's discount factor.
        tv = fcf_terminal * (1 + terminal_growth) / (mature_discount_rate - terminal_growth)
        pv_tv = tv / cumulative_df

    ev = pv_sum + pv_tv
    equity = ev  # FCFE-direct: no net-debt subtraction, see the docstring above.

    effective_shares = shares0 * (1 + annual_dilution) ** steady_state_year + financing_shares
    per_share = equity / effective_shares

    result = {
        "per_share": per_share,
        "ev": ev,
        "equity": equity,
        "revenue_path": revenue_path,
        "fcf_path": fcf_path,
        "margin_path": margin_path,
        "growth_path": growth_path,
        "tv": tv,
        "effective_shares": effective_shares,
        "final_year_revenue": revenue_path[-1],
        "revenue_multiple": revenue_path[-1] / revenue0,
    }
    if discount_path is not None:
        result["discount_path"] = discount_path
    return result


def _per_share_diff_for_growth(
    start_growth: float,
    price: float,
    revenue0: float,
    terminal_growth: float,
    discount_rate: float,
    current_margin: float,
    target_fcf_margin: float,
    steady_state_year: int,
    shares0: float,
    annual_dilution: float,
    financing_shares: float,
    mature_discount_rate: Optional[float] = None,
) -> Optional[float]:
    """Return ``revenue_first_dcf(...) - price`` at a candidate ``start_growth``.

    Returns ``None`` if ``revenue_first_dcf`` can't be evaluated at this
    point (it only raises for fixed inputs, none of which vary during this
    bisection, but the guard keeps a single unusable input degrading to "no
    result" rather than propagating an exception out of the solver).
    ``mature_discount_rate`` (WP3 discount-rate fade), when not ``None``,
    is passed straight through to :func:`revenue_first_dcf`.
    """
    try:
        result = revenue_first_dcf(
            revenue0, start_growth, terminal_growth, discount_rate, current_margin,
            target_fcf_margin, steady_state_year, shares0, annual_dilution, financing_shares,
            mature_discount_rate=mature_discount_rate,
        )
    except ValueError:
        return None
    return result["per_share"] - price


def implied_start_growth(
    price: Optional[float],
    revenue0: Optional[float],
    terminal_growth: float,
    discount_rate: float,
    current_margin: float,
    target_fcf_margin: float,
    steady_state_year: int,
    shares0: Optional[float],
    annual_dilution: float,
    financing_shares: float = 0.0,
    mature_discount_rate: Optional[float] = None,
) -> Optional[float]:
    """Bisect for the ``start_growth`` that makes the revenue-first DCF price match ``price``.

    Holds every other input fixed and searches ``start_growth`` over
    ``[-0.20, 0.60]`` (wider than ``reverse_dcf.implied_growth``'s bracket:
    hyper-growers can imply a starting growth rate above 40%) with a
    tolerance of ``1e-4`` on the growth rate or up to 80 bisection
    iterations, whichever comes first.

    Args:
        price: Current market price per share.
        revenue0: Base-year revenue.
        terminal_growth: Base-scenario terminal growth rate.
        discount_rate: Base-scenario discount rate.
        current_margin: Today's FCF margin.
        target_fcf_margin: Mature-state FCF margin.
        steady_state_year: Year by which growth and margin fully converge.
        shares0: Base share count.
        annual_dilution: Annual dilution rate (see :func:`revenue_first_dcf`).
        financing_shares: Extra financing-driven shares. Defaults to ``0.0``.
        mature_discount_rate: Optional mature discount rate to fade toward
            (WP3), passed straight through to :func:`revenue_first_dcf`.
            Defaults to ``None`` (flat ``discount_rate``, unchanged
            behavior) so existing callers/tests are unaffected.

    Returns:
        The implied ``start_growth`` (decimal fraction, rounded to 4
        decimals), or ``None`` if any required input is unusable (missing
        price/revenue0, non-positive/missing shares0) or if the per-share
        value doesn't change sign across the bracket (no root to find).
    """
    if price is None or price <= 0 or revenue0 is None or revenue0 <= 0:
        return None
    if not shares0 or shares0 <= 0:
        return None

    args = (
        price, revenue0, terminal_growth, discount_rate, current_margin,
        target_fcf_margin, steady_state_year, shares0, annual_dilution, financing_shares,
        mature_discount_rate,
    )
    diff_lo = _per_share_diff_for_growth(_START_GROWTH_BRACKET_LO, *args)
    diff_hi = _per_share_diff_for_growth(_START_GROWTH_BRACKET_HI, *args)
    if diff_lo is None or diff_hi is None:
        return None
    if diff_lo == 0:
        return round(_START_GROWTH_BRACKET_LO, 4)
    if diff_hi == 0:
        return round(_START_GROWTH_BRACKET_HI, 4)
    if (diff_lo > 0) == (diff_hi > 0):
        # No sign change across the bracket -- the target price isn't
        # reachable by any start_growth in [-20%, 60%] at these other inputs.
        return None

    lo, hi = _START_GROWTH_BRACKET_LO, _START_GROWTH_BRACKET_HI
    for _ in range(_MAX_ITERATIONS):
        mid = (lo + hi) / 2.0
        diff_mid = _per_share_diff_for_growth(mid, *args)
        if diff_mid is None:
            return None
        if diff_mid == 0 or (hi - lo) / 2.0 < _TOLERANCE:
            return round(mid, 4)
        if (diff_mid > 0) == (diff_lo > 0):
            lo, diff_lo = mid, diff_mid
        else:
            hi, diff_hi = mid, diff_mid

    return round((lo + hi) / 2.0, 4)


def _per_share_diff_for_margin(
    target_fcf_margin: float,
    price: float,
    revenue0: float,
    start_growth: float,
    terminal_growth: float,
    discount_rate: float,
    current_margin: float,
    steady_state_year: int,
    shares0: float,
    annual_dilution: float,
    financing_shares: float,
    mature_discount_rate: Optional[float] = None,
) -> Optional[float]:
    """Return ``revenue_first_dcf(...) - price`` at a candidate ``target_fcf_margin``.

    Returns ``None`` if ``revenue_first_dcf`` can't be evaluated at this
    point, mirroring :func:`_per_share_diff_for_growth`. ``mature_discount_
    rate`` (WP3 discount-rate fade), when not ``None``, is passed straight
    through to :func:`revenue_first_dcf`.
    """
    try:
        result = revenue_first_dcf(
            revenue0, start_growth, terminal_growth, discount_rate, current_margin,
            target_fcf_margin, steady_state_year, shares0, annual_dilution, financing_shares,
            mature_discount_rate=mature_discount_rate,
        )
    except ValueError:
        return None
    return result["per_share"] - price


def implied_target_margin(
    price: Optional[float],
    revenue0: Optional[float],
    start_growth: float,
    terminal_growth: float,
    discount_rate: float,
    current_margin: float,
    steady_state_year: int,
    shares0: Optional[float],
    annual_dilution: float,
    financing_shares: float = 0.0,
    mature_discount_rate: Optional[float] = None,
) -> Optional[float]:
    """Bisect for the ``target_fcf_margin`` that makes the revenue-first DCF price match ``price``.

    Holds every other input fixed and searches ``target_fcf_margin`` over
    ``[0.0, 0.90]`` with a tolerance of ``1e-4`` or up to 80 bisection
    iterations, whichever comes first, mirroring
    :func:`implied_start_growth`'s structure.

    Args:
        price: Current market price per share.
        revenue0: Base-year revenue.
        start_growth: Base-scenario year-1 growth rate.
        terminal_growth: Base-scenario terminal growth rate.
        discount_rate: Base-scenario discount rate.
        current_margin: Today's FCF margin.
        steady_state_year: Year by which growth and margin fully converge.
        shares0: Base share count.
        annual_dilution: Annual dilution rate (see :func:`revenue_first_dcf`).
        financing_shares: Extra financing-driven shares. Defaults to ``0.0``.
        mature_discount_rate: Optional mature discount rate to fade toward
            (WP3), passed straight through to :func:`revenue_first_dcf`.
            Defaults to ``None`` (flat ``discount_rate``, unchanged
            behavior) so existing callers/tests are unaffected.

    Returns:
        The implied mature ``target_fcf_margin`` (decimal fraction,
        rounded to 4 decimals), or ``None`` if any required input is
        unusable (missing price/revenue0, non-positive/missing shares0) or
        if the per-share value doesn't change sign across the bracket.
    """
    if price is None or price <= 0 or revenue0 is None or revenue0 <= 0:
        return None
    if not shares0 or shares0 <= 0:
        return None

    args = (
        price, revenue0, start_growth, terminal_growth, discount_rate, current_margin,
        steady_state_year, shares0, annual_dilution, financing_shares, mature_discount_rate,
    )
    diff_lo = _per_share_diff_for_margin(_TARGET_MARGIN_BRACKET_LO, *args)
    diff_hi = _per_share_diff_for_margin(_TARGET_MARGIN_BRACKET_HI, *args)
    if diff_lo is None or diff_hi is None:
        return None
    if diff_lo == 0:
        return round(_TARGET_MARGIN_BRACKET_LO, 4)
    if diff_hi == 0:
        return round(_TARGET_MARGIN_BRACKET_HI, 4)
    if (diff_lo > 0) == (diff_hi > 0):
        # No sign change across the bracket -- the target price isn't
        # reachable by any target_fcf_margin in [0%, 90%] at these other inputs.
        return None

    lo, hi = _TARGET_MARGIN_BRACKET_LO, _TARGET_MARGIN_BRACKET_HI
    for _ in range(_MAX_ITERATIONS):
        mid = (lo + hi) / 2.0
        diff_mid = _per_share_diff_for_margin(mid, *args)
        if diff_mid is None:
            return None
        if diff_mid == 0 or (hi - lo) / 2.0 < _TOLERANCE:
            return round(mid, 4)
        if (diff_mid > 0) == (diff_lo > 0):
            lo, diff_lo = mid, diff_mid
        else:
            hi, diff_hi = mid, diff_mid

    return round((lo + hi) / 2.0, 4)


#: Upper bound of the flat cost-of-equity the discount-rate reverse solve
#: searches over (decimal). The lower bound is derived per-call as
#: ``terminal_growth + _DISCOUNT_RATE_BRACKET_EPS`` so the Gordon terminal
#: value stays defined (``r > terminal_growth``).
_DISCOUNT_RATE_BRACKET_HI = 0.60
_DISCOUNT_RATE_BRACKET_EPS = 1e-3


def implied_discount_rate(
    price: Optional[float],
    revenue0: Optional[float],
    start_growth: float,
    terminal_growth: float,
    current_margin: float,
    target_fcf_margin: float,
    steady_state_year: int,
    shares0: Optional[float],
    annual_dilution: float,
    financing_shares: float = 0.0,
) -> Optional[float]:
    """Bisect for the FLAT cost of equity that makes the revenue-first DCF match ``price``.

    The mirror of :func:`implied_start_growth`/:func:`implied_target_margin`
    for the third reverse lens: holding revenue growth and the mature margin
    at the model's base-scenario assumptions, what single (un-faded) discount
    rate must the market be applying to justify today's price? A large gap
    between this and the model's own cost of equity is one face of a
    model-market divergence (the price implies a far higher risk premium than
    the model charges).

    A flat rate is used deliberately -- both the year-1 cohort rate AND the
    mature fade target are set to the solved ``r`` -- so the answer is a
    single interpretable "the market discounts this at X%", not a fade pair.
    Per-share value is strictly decreasing in ``r``, so the bracket is
    ``[terminal_growth + _DISCOUNT_RATE_BRACKET_EPS, _DISCOUNT_RATE_BRACKET_HI]``.

    Returns:
        The implied flat discount rate (decimal, rounded to 4), or ``None``
        if any required input is unusable or the price isn't reachable by any
        rate in the bracket (no sign change).
    """
    if price is None or price <= 0 or revenue0 is None or revenue0 <= 0:
        return None
    if not shares0 or shares0 <= 0:
        return None

    lo = terminal_growth + _DISCOUNT_RATE_BRACKET_EPS
    hi = _DISCOUNT_RATE_BRACKET_HI
    if lo >= hi:
        return None

    def diff_at(r: float) -> Optional[float]:
        try:
            result = revenue_first_dcf(
                revenue0, start_growth, terminal_growth, r, current_margin, target_fcf_margin,
                steady_state_year, shares0, annual_dilution, financing_shares,
                mature_discount_rate=r,
            )
        except ValueError:
            return None
        return result["per_share"] - price

    diff_lo = diff_at(lo)
    diff_hi = diff_at(hi)
    if diff_lo is None or diff_hi is None:
        return None
    if diff_lo == 0:
        return round(lo, 4)
    if diff_hi == 0:
        return round(hi, 4)
    if (diff_lo > 0) == (diff_hi > 0):
        return None

    for _ in range(_MAX_ITERATIONS):
        mid = (lo + hi) / 2.0
        diff_mid = diff_at(mid)
        if diff_mid is None:
            return None
        if diff_mid == 0 or (hi - lo) / 2.0 < _TOLERANCE:
            return round(mid, 4)
        if (diff_mid > 0) == (diff_lo > 0):
            lo, diff_lo = mid, diff_mid
        else:
            hi, diff_hi = mid, diff_mid

    return round((lo + hi) / 2.0, 4)
