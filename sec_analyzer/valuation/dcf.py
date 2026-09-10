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


def fcfe_sustainable_growth_per_share(
    ni0: Optional[float],
    roe: float,
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares: Optional[float],
    dilution_rate: float = 0.0,
    terminal_roe: Optional[float] = None,
) -> dict:
    """Compute a per-share, sustainable-growth FCFE fair value from
    normalized earnings and ROE (SPEC.md Sec.8e).

    This is the growth-inclusive counterpart to earnings-power-value (EPV,
    see ``engine._build_earnings_power``): instead of capitalizing
    zero-growth normalized earnings, it grows those earnings along the SAME
    10-year, two-stage path :func:`project_fcf` uses for FCF (years 1-5 at
    ``growth_5y``, years 6-10 fading linearly to ``terminal_growth`` via
    :func:`_year_growth_rate`), and funds that growth out of the earnings
    themselves via the sustainable-growth identity: to grow earnings at rate
    ``g`` while holding ROE constant, the firm must retain (reinvest) a
    fraction ``b = g / roe`` of net income; the rest, ``ni * (1 - b)``, is
    the free cash flow to equity (FCFE) actually distributable to
    shareholders that year. This is the textbook reason growth only adds
    value over the no-growth EPV floor when ``roe > discount_rate``: at
    ``roe == discount_rate`` every dollar reinvested earns exactly the
    return investors require, so retaining it instead of distributing it is
    value-neutral (this anchor collapses to EPV in that case); at
    ``roe > discount_rate`` reinvested dollars earn an excess return and
    growth is genuinely value-accretive; at ``roe < discount_rate`` growth
    actually destroys value; this function does not special-case that last
    condition (it isn't the caller's gate to enforce here), so a low-ROE
    filer will simply produce a lower FCFE-anchor value than its EPV floor,
    and it is the caller's job (mirroring the "beats-floor" guardrail
    already used for the mature-sector revenue-first anchor) to fall back to
    EPV when that happens.

    Growth cannot exceed the return that funds it: for every projected year
    (including the terminal year), the growth rate actually booked is
    ``g_eff = min(g, roe)`` -- both the earnings compounding for that year
    AND the reinvestment rate ``b = g_eff / roe`` use this capped rate, not
    the raw assumption. When ``g >= roe`` the firm cannot fund that growth
    purely out of its own earnings (it would require external equity), so
    the model caps growth at what ROE can internally fund rather than
    inventing cash via a reinvestment rate above 1.0 or booking growth the
    earnings base can't actually sustain. This also means the implied
    reinvestment rate is never artificially clamped at a fixed ceiling
    (e.g. a flat 90%): it rides up to (but never past) 1.0 as ``g_eff``
    approaches ``roe``, driving distributable FCFE to (but never below) 0
    for that year.

    Terminal ROE fade to cost of equity: by default (``terminal_roe=None``)
    the terminal year's reinvestment is computed against the same ``roe``
    used for years 1-10 (backward compatible). When ``terminal_roe`` is
    supplied, the terminal-year reinvestment rate ``b_t = min(terminal_growth,
    terminal_roe) / terminal_roe`` is computed against it instead -- callers
    typically pass the scenario's own ``discount_rate`` here, so the
    terminal (perpetuity) phase assumes the firm's excess return has faded
    to zero (terminal ROE == cost of equity), the standard Damodaran
    stable-growth-phase convention: a firm cannot sustain an ROE above its
    cost of equity indefinitely once competitive advantages erode, even if
    its near-term (years 1-10) ROE is higher. Years 1-10 are unaffected by
    ``terminal_roe`` -- they always use the current-period ``roe``.

    FCFE-direct, cost-of-equity discounting: like :func:`dcf_per_share`,
    this projects an already-levered (equity) cash flow -- normalized net
    income is a post-interest, post-tax figure -- so ``discount_rate`` must
    be a levered cost of equity, not a WACC, and no net-debt bridge is
    applied (``ev`` and ``equity`` are the same number, both kept for
    caller convenience). Dilution uses the same ``effective_shares = shares
    * (1 + dilution_rate) ** 5`` convention as :func:`dcf_per_share` (see
    its docstring for why year 5, the horizon's midpoint, is used).

    Args:
        ni0: Base-year (year 0) normalized net income the earnings
            projection starts from (e.g. EPV's own
            ``normalized_net_income``).
        roe: Return on equity (normalized net income / stockholders'
            equity) used both to derive the sustainable reinvestment rate
            ``b = g_eff / roe`` for projection years 1-10 and, implicitly,
            to determine whether growth adds or destroys value relative to
            the ``discount_rate``.
        growth_5y: Constant earnings-growth rate for projection years 1-5
            (decimal fraction).
        terminal_growth: Growth rate years 6-10 fade to, and the terminal
            year's own growth rate (before the ``min(terminal_growth,
            terminal_roe)`` cap described above).
        discount_rate: Annual discount rate (decimal fraction) -- a levered
            COST OF EQUITY, exactly like :func:`dcf_per_share`.
        shares: Diluted shares outstanding used as the pre-dilution base.
        dilution_rate: Annual share-count growth rate applied over 5 years
            to derive the effective share count. Defaults to ``0.0``.
        terminal_roe: ROE assumed for the terminal (perpetuity) year's
            reinvestment rate only. Defaults to ``None``, which falls back
            to ``roe`` (backward compatible). Callers modeling a fade to a
            stable, competition-eroded terminal phase should pass the
            scenario's own cost of equity here (see the terminal-ROE-fade
            note above).

    Returns:
        A dict with keys ``per_share``, ``ev``, ``equity`` (equal, see
        above), ``ni_path`` (the 10 projected normalized-net-income
        floats), ``fcfe_path`` (the 10 projected FCFE floats), ``tv``
        (undiscounted terminal value), and ``effective_shares``. Nothing is
        rounded here -- rounding is the caller's (``engine.py``'s)
        responsibility.

    Raises:
        ValueError: If ``ni0`` is ``None``, if ``shares`` is falsy or
            ``<= 0``, if ``roe <= 0`` (a non-positive ROE makes the
            reinvestment rate meaningless), or if ``discount_rate <=
            terminal_growth`` (the Gordon-growth terminal value is
            undefined in that case -- this is never silently "fixed").
    """
    if ni0 is None:
        raise ValueError(
            "fcfe_sustainable_growth_per_share: ni0 is None; cannot project an earnings path without a base."
        )
    if not shares or shares <= 0:
        raise ValueError(f"fcfe_sustainable_growth_per_share: shares must be a positive number, got {shares!r}.")
    if roe <= 0:
        raise ValueError(f"fcfe_sustainable_growth_per_share: roe must be positive, got {roe!r}.")
    if discount_rate <= terminal_growth:
        raise ValueError(
            f"fcfe_sustainable_growth_per_share: discount_rate ({discount_rate}) must be strictly greater than "
            f"terminal_growth ({terminal_growth}) for the Gordon-growth terminal value to be defined."
        )

    ni_path: List[float] = []
    fcfe_path: List[float] = []
    growth_path: List[float] = []
    growth_capped = False
    previous_ni = ni0
    pv_sum = 0.0
    for year in range(1, HORIZON_YEARS + 1):
        growth_rate = _year_growth_rate(year, growth_5y, terminal_growth)
        g_eff = min(growth_rate, roe)
        if g_eff < growth_rate:
            growth_capped = True
        ni_year = previous_ni * (1 + g_eff)
        reinvestment_rate = g_eff / roe
        fcfe_year = ni_year * (1 - reinvestment_rate)

        ni_path.append(ni_year)
        fcfe_path.append(fcfe_year)
        growth_path.append(g_eff)
        pv_sum += fcfe_year / (1 + discount_rate) ** year
        previous_ni = ni_year

    # Terminal phase: fade to `terminal_roe` (cost of equity, by convention)
    # when supplied, else fall back to the current-period `roe`. NOTE: the
    # Gordon-growth denominator below deliberately keeps the ORIGINAL,
    # uncapped `terminal_growth` -- only the terminal earnings/reinvestment
    # computation is capped, matching the already-validated
    # `discount_rate > terminal_growth` guard above.
    terminal_roe_resolved = terminal_roe if terminal_roe is not None else roe
    g_t_eff = min(terminal_growth, terminal_roe_resolved)
    ni_terminal = ni_path[-1] * (1 + g_t_eff)
    terminal_reinvestment_rate = g_t_eff / terminal_roe_resolved
    fcfe_terminal = ni_terminal * (1 - terminal_reinvestment_rate)
    tv = fcfe_terminal / (discount_rate - terminal_growth)
    pv_tv = tv / (1 + discount_rate) ** HORIZON_YEARS

    equity = pv_sum + pv_tv
    ev = equity  # FCFE-direct: no net-debt subtraction, see the docstring above.

    effective_shares = shares * (1 + dilution_rate) ** _DILUTION_HORIZON_YEARS
    per_share = equity / effective_shares

    return {
        "per_share": per_share,
        "ev": ev,
        "equity": equity,
        "ni_path": ni_path,
        "fcfe_path": fcfe_path,
        "tv": tv,
        "effective_shares": effective_shares,
        # SPEC.md Sec.22a: the growth actually applied, so a caller can tell
        # the reader when the internal-funding cap discarded the assumption.
        "growth_path": growth_path,
        "effective_growth_5y": min(growth_5y, roe),
        "growth_capped": growth_capped,
    }


def rim_per_share(
    bve0: Optional[float],
    ni0: Optional[float],
    roe: float,
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares: Optional[float],
    dilution_rate: float = 0.0,
    terminal_roe: Optional[float] = None,
) -> dict:
    """Compute a per-share residual income model (RIM) fair value from
    book value, normalized net income, and ROE (SPEC.md Sec.8f).

    Ohlson-style residual income: intrinsic equity value = book value today
    plus the present value of all future "excess" earnings (residual income,
    ``RI_t = NI_t - discount_rate * BVE_{t-1}``) the firm is expected to
    generate above what its cost of equity requires on the book value it
    starts each period with. This is the financial-sector counterpart to
    :func:`fcfe_sustainable_growth_per_share` -- same ``g_eff = min(g, roe)``
    reinvestment cap and ``terminal_roe`` fade-to-cost-of-equity convention,
    just applied to a book-value/residual-income compounding path instead of
    a dividend/FCFE payout path. It replaces the single-period justified-P/B
    heuristic (``engine._justified_pb``, ``(ROE - g) / (r - g)`` applied as a
    static multiple to today's book value) with a genuine multi-year fade:
    years 1-5 grow earnings at ``growth_5y``, years 6-10 fade linearly to
    ``terminal_growth`` (via :func:`_year_growth_rate`, the same fade curve
    :func:`dcf_per_share` uses), each year's earnings are capped at what ROE
    can fund (``g_eff = min(g, roe)``), and book value compounds forward by
    the fraction of earnings actually retained to fund that capped growth.

    Earnings/book-value roll-forward: earnings compound at ``g_eff`` exactly
    like :func:`fcfe_sustainable_growth_per_share`'s ``ni_path``
    (``ni_year = previous_ni * (1 + g_eff)``); the reinvestment rate ``b =
    g_eff / roe`` is the fraction of that year's earnings retained (added to
    book value) rather than distributed, so book value grows by
    ``retained = ni_year * b`` each year (clean-surplus accounting with no
    external capital raise) -- NOT by the full ``ni_year`` (that would imply
    100% retention regardless of ``g_eff``, silently assuming no dividend
    ever gets paid).

    Terminal ROE fade to cost of equity: the terminal (perpetuity) net
    income is what the ending book value earns at the terminal ROE --
    ``ni_terminal = terminal_roe_resolved * bve_10`` -- so terminal residual
    income is ``(terminal_roe_resolved - discount_rate) * bve_10``. When the
    caller fades terminal ROE to the cost of equity (typically by passing the
    scenario's own ``discount_rate`` as ``terminal_roe``), the terminal
    spread is exactly zero, so the terminal residual income AND the Gordon
    terminal value are zero: intrinsic equity is then today's book value plus
    the present value of only the finite-horizon excess returns (the standard
    RIM "excess returns compete away in steady state" terminal, Damodaran/
    Penman). This is the RIM analog of
    :func:`fcfe_sustainable_growth_per_share`'s terminal fade -- NOT the same
    mechanism: there, terminal reinvestment earns exactly the cost of equity
    (value-neutral) and the terminal DIVIDEND perpetuity is still positive;
    here, the book value already captures the normal return, so a faded
    terminal adds nothing beyond it. A caller passing ``terminal_roe`` ABOVE
    the cost of equity (a durable-franchise assumption) still gets a
    positive, ``terminal_growth``-growing terminal residual. ``terminal_roe``
    defaults to ``roe`` when omitted (a permanent-excess-return terminal --
    the engine always passes ``discount_rate`` for the faded convention).

    No net-debt bridge: book value of equity and net income are already
    equity-level (post-interest, post-tax) figures for a financial-sector
    filer, so ``discount_rate`` must be a levered cost of equity, exactly
    like :func:`dcf_per_share`/:func:`fcfe_sustainable_growth_per_share`.

    Args:
        bve0: Base-year (year 0) book value of equity (``StockholdersEquity``)
            the roll-forward starts from.
        ni0: Base-year (year 0) net income (used only to validate the anchor
            is being built from a profitable base; the earnings PATH itself
            compounds off ``ni0`` at each year's ``g_eff``, mirroring
            :func:`fcfe_sustainable_growth_per_share`).
        roe: Return on equity (net income / book value of equity) used to
            derive each projection year's sustainable reinvestment rate
            ``b = g_eff / roe``.
        growth_5y: Constant earnings-growth rate for projection years 1-5
            (decimal fraction).
        terminal_growth: Growth rate years 6-10 fade to, and the growth rate
            the terminal residual income grows at in perpetuity (Gordon
            denominator ``discount_rate - terminal_growth``).
        discount_rate: Annual discount rate (decimal fraction) -- a levered
            COST OF EQUITY, exactly like :func:`dcf_per_share`.
        shares: Diluted shares outstanding used as the pre-dilution base.
        dilution_rate: Annual share-count growth rate applied over 5 years
            to derive the effective share count. Defaults to ``0.0``.
        terminal_roe: The ROE the ending book value is assumed to earn in the
            terminal (perpetuity) phase; ``terminal_roe == discount_rate``
            makes the terminal residual income (and the Gordon terminal
            value) zero -- see the "Terminal ROE fade" note above. Defaults
            to ``None`` (falls back to ``roe`` -- a permanent-excess-return
            terminal; the engine always passes ``discount_rate``).

    Returns:
        A dict with keys ``per_share``, ``equity`` (total intrinsic equity
        value, ``bve0`` plus the PV of all residual income), ``bve0``,
        ``ri_path`` (the 10 projected residual-income floats), ``bve_path``
        (the 10 projected book-value-of-equity floats, end of each year),
        ``tv`` (undiscounted terminal value), and ``effective_shares``.
        Nothing is rounded here -- rounding is the caller's (``engine.py``'s)
        responsibility.

    Raises:
        ValueError: If ``ni0`` is ``None``, if ``bve0`` is ``None`` or
            ``<= 0`` (a non-positive book value makes the roll-forward
            meaningless), if ``shares`` is falsy or ``<= 0``, if ``roe <= 0``,
            or if ``discount_rate <= terminal_growth`` (the Gordon-growth
            terminal value is undefined in that case -- never silently
            "fixed").
    """
    if ni0 is None:
        raise ValueError("rim_per_share: ni0 is None; cannot project an earnings path without a base.")
    if bve0 is None or bve0 <= 0:
        raise ValueError(f"rim_per_share: bve0 must be a positive number, got {bve0!r}.")
    if not shares or shares <= 0:
        raise ValueError(f"rim_per_share: shares must be a positive number, got {shares!r}.")
    if roe <= 0:
        raise ValueError(f"rim_per_share: roe must be positive, got {roe!r}.")
    if discount_rate <= terminal_growth:
        raise ValueError(
            f"rim_per_share: discount_rate ({discount_rate}) must be strictly greater than "
            f"terminal_growth ({terminal_growth}) for the Gordon-growth terminal value to be defined."
        )

    ri_path: List[float] = []
    bve_path: List[float] = []
    growth_path: List[float] = []
    growth_capped = False
    previous_ni = ni0
    previous_bve = bve0
    pv_sum = 0.0
    for year in range(1, HORIZON_YEARS + 1):
        growth_rate = _year_growth_rate(year, growth_5y, terminal_growth)
        g_eff = min(growth_rate, roe)
        if g_eff < growth_rate:
            growth_capped = True
        ni_year = previous_ni * (1 + g_eff)
        reinvestment_rate = g_eff / roe
        retained = ni_year * reinvestment_rate
        bve_year = previous_bve + retained
        ri_year = ni_year - discount_rate * previous_bve

        ri_path.append(ri_year)
        bve_path.append(bve_year)
        growth_path.append(g_eff)
        pv_sum += ri_year / (1 + discount_rate) ** year
        previous_ni = ni_year
        previous_bve = bve_year

    # Terminal phase: the terminal (perpetuity) net income is what the ending
    # book value earns at the FADED terminal ROE -- `ni_terminal =
    # terminal_roe_resolved * bve_10` -- NOT the year-10 net income compounded
    # forward. This is what makes `terminal_roe` actually bite (the earlier
    # `min(terminal_growth, terminal_roe)` construction left it inert, since
    # `terminal_growth < terminal_roe` always held under the `r > terminal_growth`
    # guard). Terminal residual income is then the excess of that terminal
    # earning power over the cost-of-equity charge on the same book:
    # `ri_terminal = (terminal_roe_resolved - discount_rate) * bve_10`.
    #
    # When the caller fades terminal ROE to the cost of equity (the engine
    # passes `terminal_roe = discount_rate`), the terminal spread is exactly
    # 0, so terminal residual income and the Gordon terminal value are 0 --
    # the standard RIM "excess returns compete away in steady state" terminal
    # (Damodaran/Penman): intrinsic equity then equals today's book value plus
    # the present value of only the finite-horizon excess returns. This is the
    # RIM analog of the FCFE sibling's terminal fade (there, terminal
    # reinvestment earns exactly the cost of equity and is value-neutral;
    # here, the book value already captures the normal return, so only the
    # vanishing excess return is what the terminal adds). A caller passing a
    # terminal_roe ABOVE the cost of equity (a durable-franchise assumption)
    # still gets a positive, `terminal_growth`-growing terminal residual.
    terminal_roe_resolved = terminal_roe if terminal_roe is not None else roe
    ri_terminal = (terminal_roe_resolved - discount_rate) * bve_path[-1]
    tv = ri_terminal / (discount_rate - terminal_growth)
    pv_tv = tv / (1 + discount_rate) ** HORIZON_YEARS

    equity = bve0 + pv_sum + pv_tv

    effective_shares = shares * (1 + dilution_rate) ** _DILUTION_HORIZON_YEARS
    per_share = equity / effective_shares

    return {
        "per_share": per_share,
        "equity": equity,
        "bve0": bve0,
        "ri_path": ri_path,
        "bve_path": bve_path,
        "tv": tv,
        "effective_shares": effective_shares,
        # SPEC.md Sec.22a: the growth actually applied. When `growth_capped`
        # is True the caller's `growth_5y` was discarded in favor of what
        # retained earnings alone can fund, and the reader must be told.
        "growth_path": growth_path,
        "effective_growth_5y": min(growth_5y, roe),
        "growth_capped": growth_capped,
    }


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


def rim_external_growth_per_share(
    bve0: float,
    ni0: float,
    roe: float,
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares: float,
    terminal_roe: Optional[float] = None,
) -> dict:
    """Residual income when growth is funded by issuing equity, not retention.

    :func:`rim_per_share` caps every year at ``min(g, roe)`` -- the growth
    retained earnings alone can fund (``g = b x ROE``, ``b <= 1``). A firm can
    of course grow faster by issuing shares; SoFi's FY2025 equity went
    ``6,525M -> 10,489M`` on ``481M`` of net income, so roughly ``3.5B`` came
    from external issuance. This function answers what that does to value.

    **Issuance is modeled at fair value, which is value-neutral**: new
    shareholders pay in exactly what their claim is worth, so the injected
    capital and the claim it buys cancel. Two consequences:

    * the share count is NOT inflated -- ``per_share`` divides by today's
      shares;
    * the whole value effect comes from applying the ``roe - discount_rate``
      spread to a LARGER book value.

    Modeling issuance at any other price would transfer value between old and
    new holders, and picking that price would make an *intrinsic* estimate
    depend on the market price it exists to be tested against.

    The sign is the point, and it is unambiguous: with ``roe >
    discount_rate`` faster growth ADDS value, with ``roe == discount_rate``
    it is exactly neutral (residual income is zero every year, so the value
    is ``bve0`` regardless of growth), and with ``roe < discount_rate`` it
    DESTROYS value. This function quantifies that; it does not rescue a
    number.

    Earnings follow an ROE-consistent path (``ni_t = roe * bve_{t-1}``) rather
    than ``rim_per_share``'s ``ni0``-compounding path, so the stated ROE
    cannot drift (see Sec.8f's documented F4 front-loading). To keep the
    comparison honest, ``per_share_internal`` re-runs THIS path with growth
    capped at ``roe`` -- so ``value_gap_per_share`` isolates the funding
    assumption alone instead of mixing in the two functions' different
    earnings paths.

    Args:
        bve0: Opening book value of equity. Must be positive.
        ni0: Latest net income. Accepted for signature symmetry with
            ``rim_per_share`` and validated identically; the projection
            itself derives earnings from ``roe * bve`` (see above).
        roe: Return on equity, as a fraction. Must be positive.
        growth_5y: Assumed growth for years 1-5, applied UNCAPPED here.
        terminal_growth: Perpetuity growth; also the years 6-10 fade target.
        discount_rate: Levered cost of equity (no net-debt bridge -- book
            value and net income are already equity-level figures).
        shares: Today's share count. Must be positive.
        terminal_roe: Terminal-phase ROE. The engine passes the cost of
            equity, making terminal residual income (and so the terminal
            value) exactly zero. Defaults to ``roe``.

    Returns:
        ``{"per_share", "equity", "bve0", "ri_path", "bve_path", "tv",
        "effective_shares", "per_share_internal", "value_gap_per_share",
        "external_funding_total", "external_funding_path"}``. Nothing is
        rounded -- that is the caller's job.

    Raises:
        ValueError: On the same invalid inputs as :func:`rim_per_share`.
    """
    if ni0 is None:
        raise ValueError("rim_external_growth_per_share: ni0 must not be None.")
    if bve0 is None or bve0 <= 0:
        raise ValueError(
            f"rim_external_growth_per_share: bve0 must be positive, got {bve0!r}."
        )
    if not shares or shares <= 0:
        raise ValueError(
            f"rim_external_growth_per_share: shares must be a positive number, got {shares!r}."
        )
    if roe <= 0:
        raise ValueError(f"rim_external_growth_per_share: roe must be positive, got {roe!r}.")
    if discount_rate <= terminal_growth:
        raise ValueError(
            f"rim_external_growth_per_share: discount_rate ({discount_rate}) must be strictly "
            f"greater than terminal_growth ({terminal_growth}) for the Gordon-growth terminal "
            "value to be defined."
        )

    terminal_roe_resolved = terminal_roe if terminal_roe is not None else roe

    def _run(cap_growth: bool):
        ri_path: List[float] = []
        bve_path: List[float] = []
        funding_path: List[float] = []
        previous_bve = bve0
        pv_sum = 0.0
        for year in range(1, HORIZON_YEARS + 1):
            growth_rate = _year_growth_rate(year, growth_5y, terminal_growth)
            g = min(growth_rate, roe) if cap_growth else growth_rate
            # Residual income on the OPENING book, exactly as rim_per_share.
            ri_year = (roe - discount_rate) * previous_bve
            # Retention supplies `roe * previous_bve`; the rest must be issued.
            funding_path.append(max(0.0, previous_bve * (g - roe)))
            bve_year = previous_bve * (1 + g)

            ri_path.append(ri_year)
            bve_path.append(bve_year)
            pv_sum += ri_year / (1 + discount_rate) ** year
            previous_bve = bve_year

        ri_terminal = (terminal_roe_resolved - discount_rate) * bve_path[-1]
        tv = ri_terminal / (discount_rate - terminal_growth)
        pv_sum += tv / (1 + discount_rate) ** HORIZON_YEARS
        return bve0 + pv_sum, ri_path, bve_path, funding_path, tv

    equity, ri_path, bve_path, funding_path, tv = _run(cap_growth=False)
    equity_internal, _, _, _, _ = _run(cap_growth=True)

    per_share = equity / shares
    per_share_internal = equity_internal / shares

    return {
        "per_share": per_share,
        "equity": equity,
        "bve0": bve0,
        "ri_path": ri_path,
        "bve_path": bve_path,
        "tv": tv,
        "effective_shares": shares,
        "per_share_internal": per_share_internal,
        "value_gap_per_share": per_share - per_share_internal,
        "external_funding_total": sum(funding_path),
        "external_funding_path": funding_path,
    }
