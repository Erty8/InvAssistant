"""Distress and earnings-quality screens (SPEC.md Sec.8g/8j/8k): Altman
Z-score, Beneish M-score, and Merton distance-to-default.

Every function here is a pure, deterministic classification/screen -- given
the same numeric inputs it always returns the same result, never talks to
the network/database/LLM, and degrades to ``None`` (never raises, never
fabricates a value) when a required input is missing or the computation is
economically degenerate (e.g. a non-positive denominator, a non-converging
numerical solve).

These are ADVISORY overlays. SPEC.md Sec.8g/8j/8k are explicit that none of
them ever headline ``fair_value_range`` or participate in
``triangulate.triangulate``'s confidence vote -- they exist to flag risk
(bankruptcy, earnings manipulation) alongside the valuation, not to replace
or adjust it.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

#: Altman (1968) Z-score zone thresholds for public, non-financial filers
#: (SPEC.md Sec.8g). Z > 2.99 is conventionally "safe"; Z < 1.81 is
#: conventionally "distress"; the band between is the "grey" zone.
_ALTMAN_SAFE_THRESHOLD = 2.99
_ALTMAN_DISTRESS_THRESHOLD = 1.81


def altman_z_score(
    working_capital: Optional[float],
    total_assets: Optional[float],
    retained_earnings: Optional[float],
    ebit: Optional[float],
    market_cap: Optional[float],
    total_liabilities: Optional[float],
    revenue: Optional[float],
) -> Optional[dict]:
    """Compute the classic Altman (1968) Z-score bankruptcy-risk screen
    (SPEC.md Sec.8g) for a public, non-financial filer.

    ``Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5``, where:

    - ``X1 = working_capital / total_assets`` (liquidity)
    - ``X2 = retained_earnings / total_assets`` (cumulative profitability)
    - ``X3 = ebit / total_assets`` (operating efficiency)
    - ``X4 = market_cap / total_liabilities`` (market-based leverage cushion)
    - ``X5 = revenue / total_assets`` (asset turnover)

    Zone: ``Z > 2.99`` -> ``"safe"``; ``1.81 <= Z <= 2.99`` -> ``"grey"``;
    ``Z < 1.81`` -> ``"distress"``.

    This model was calibrated on manufacturing/industrial firms and is not
    meaningful for financial-sector or REIT filers (their balance-sheet
    leverage is structural, not distress-indicative) -- the caller
    (``engine._build_altman_z``) never calls this for those sectors.

    Args:
        working_capital: Current assets minus current liabilities, for the
            resolved fiscal year.
        total_assets: Total assets, for the same fiscal year.
        retained_earnings: Cumulative retained earnings/accumulated deficit,
            for the same fiscal year.
        ebit: Earnings before interest and taxes (this engine uses operating
            income as the EBIT proxy), for the same fiscal year.
        market_cap: Current market capitalization (price x shares).
        total_liabilities: Total liabilities, for the same fiscal year.
        revenue: Revenue, for the same fiscal year.

    Returns:
        ``None`` if any input is missing, or if ``total_assets`` or
        ``total_liabilities`` is non-positive (both ratios and the
        underlying model are undefined in that case). Otherwise a dict with
        ``z_score`` (rounded to 2dp), ``zone`` (``"safe"``/``"grey"``/
        ``"distress"``), and ``components`` (the five raw X1-X5 ratios,
        rounded to 4dp, for display/debugging).
    """
    values = (working_capital, total_assets, retained_earnings, ebit, market_cap, total_liabilities, revenue)
    if any(v is None for v in values):
        return None
    if total_assets <= 0 or total_liabilities <= 0:
        return None

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = revenue / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > _ALTMAN_SAFE_THRESHOLD:
        zone = "safe"
    elif z >= _ALTMAN_DISTRESS_THRESHOLD:
        zone = "grey"
    else:
        zone = "distress"

    return {
        "z_score": round(z, 2),
        "zone": zone,
        "components": {
            "x1": round(x1, 4), "x2": round(x2, 4), "x3": round(x3, 4),
            "x4": round(x4, 4), "x5": round(x5, 4),
        },
    }


#: Beneish (1999) M-score manipulation-likelihood threshold: M above this
#: value is conventionally read as an earnings-manipulation risk signal.
_BENEISH_MANIPULATION_THRESHOLD = -1.78


def _beneish_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safe division for a Beneish index ratio: ``None`` (never raises,
    never fabricates) when either operand is missing or the denominator is
    exactly zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def beneish_m_score(current: dict, prior: dict) -> Optional[dict]:
    """Compute the Beneish M-score earnings-manipulation screen (SPEC.md
    Sec.8j) from two consecutive fiscal years' worth of raw figures.

    Full 8-variable model (Beneish 1999) when both fiscal years carry SG&A
    AND the leverage/accruals inputs; degrades to the well-established
    **5-variable model** (a real, separately published Beneish variant --
    NOT an ad hoc truncation of the 8-variable regression's calibrated
    coefficients, which would be statistically invalid) when SG&A or the
    leverage/accruals inputs are missing for either year:

    - **8-variable**: ``M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI +
      0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI``.
    - **5-variable**: ``M = -6.065 + 0.823*DSRI + 0.906*GMI + 0.593*AQI +
      0.717*SGI + 0.107*DEPI`` (drops SGAI/TATA/LVGI entirely -- their own
      calibrated coefficients, not a rescaled subset of the 8-variable ones).

    Both models flag ``M > -1.78`` (:data:`_BENEISH_MANIPULATION_THRESHOLD`)
    as a manipulation-likelihood signal.

    Index formulas (``t`` = current year, ``t-1`` = prior year):

    - ``DSRI = (receivables_t/revenue_t) / (receivables_t-1/revenue_t-1)``
      (days-sales-in-receivables index).
    - ``GMI = gm_t-1 / gm_t``, where ``gm = gross_profit/revenue`` (gross
      margin index -- a FALLING margin, i.e. ``GMI > 1``, is the
      manipulation-risk direction).
    - ``AQI = [1 - (current_assets_t+ppe_gross_t)/total_assets_t] / [1 -
      (current_assets_t-1+ppe_gross_t-1)/total_assets_t-1]`` (asset-quality
      index). **Proxy note**: the classic formula uses NET PP&E; this
      engine has no net-PP&E concept, so GROSS PP&E
      (``PropertyPlantAndEquipmentGross``) is used instead -- a documented
      approximation, exactly like the FFO anchor's D&A-proxy note (SPEC.md
      Sec.8c).
    - ``SGI = revenue_t / revenue_t-1`` (sales growth index).
    - ``DEPI = [depreciation_t-1/(ppe_gross_t-1+depreciation_t-1)] /
      [depreciation_t/(ppe_gross_t+depreciation_t)]`` (depreciation index --
      same gross-PP&E proxy note as AQI).
    - ``SGAI = (sga_t/revenue_t) / (sga_t-1/revenue_t-1)`` (SG&A index,
      8-variable model only).
    - ``LVGI = [(long_term_debt_t+current_liabilities_t)/total_assets_t] /
      [(long_term_debt_t-1+current_liabilities_t-1)/total_assets_t-1]``
      (leverage index, 8-variable model only).
    - ``TATA = (net_income_t - operating_cash_flow_t)/total_assets_t``
      (total-accruals-to-total-assets, 8-variable model only).

    Args:
        current: The fiscal year ``t``'s raw figures --
            ``{"receivables","revenue","gross_profit","current_assets",
            "ppe_gross","total_assets","depreciation"}`` required;
            ``{"sga","long_term_debt","current_liabilities","net_income",
            "operating_cash_flow"}`` optional (gate the 8- vs 5-variable
            choice).
        prior: The fiscal year ``t-1``'s raw figures, same shape.

    Returns:
        ``None`` (never raises, never fabricates) when any REQUIRED input
        (the 5-variable model's inputs) is missing, or any ratio's
        denominator is exactly zero or produces a non-positive value inside
        a ratio-of-ratios where that's undefined (e.g. ``gm_t == 0``,
        an AQI/DEPI denominator of ``0``). Otherwise a dict with
        ``m_score`` (2dp), ``partial`` (``True`` when the 5-variable model
        was used), ``flag`` (``bool`` -- ``m_score >
        _BENEISH_MANIPULATION_THRESHOLD``), and ``components`` (whichever
        of DSRI/GMI/AQI/SGI/DEPI/SGAI/LVGI/TATA were computed, 4dp).
    """
    dsri = _beneish_ratio(
        _beneish_ratio(current.get("receivables"), current.get("revenue")),
        _beneish_ratio(prior.get("receivables"), prior.get("revenue")),
    )
    gm_current = _beneish_ratio(current.get("gross_profit"), current.get("revenue"))
    gm_prior = _beneish_ratio(prior.get("gross_profit"), prior.get("revenue"))
    gmi = _beneish_ratio(gm_prior, gm_current)
    sgi = _beneish_ratio(current.get("revenue"), prior.get("revenue"))

    # AQI uses the classic non-current-non-PP&E asset-quality term
    # `1 - (CurrentAssets + NET PP&E)/TotalAssets`, which is essentially
    # always positive. We only have GROSS PP&E (WP8), so for old, asset-heavy,
    # low-intangible filers (utilities, railroads, manufacturers where
    # accumulated depreciation exceeds goodwill/intangibles) `(CA + gross
    # PP&E)/TA` can exceed 1 and drive the inner term NEGATIVE -- an
    # out-of-domain proxy that would feed a sign-flipped, meaningless AQI into
    # the M-score. Guard it: require both years' inner terms strictly positive;
    # otherwise leave AQI None (-> the whole score returns None below rather
    # than emitting a garbage index). Sourcing NET PP&E would remove the guard.
    aqi = None
    ca_c, ppe_c, ta_c = current.get("current_assets"), current.get("ppe_gross"), current.get("total_assets")
    ca_p, ppe_p, ta_p = prior.get("current_assets"), prior.get("ppe_gross"), prior.get("total_assets")
    if None not in (ca_c, ppe_c, ta_c, ca_p, ppe_p, ta_p) and ta_c != 0 and ta_p != 0:
        aq_current = 1 - (ca_c + ppe_c) / ta_c
        aq_prior = 1 - (ca_p + ppe_p) / ta_p
        if aq_current > 0 and aq_prior > 0:
            aqi = _beneish_ratio(aq_prior, aq_current)

    # DEPI's depreciation-rate denominators (`ppe_gross + depreciation`) must
    # be strictly positive for the rate to be meaningful; guard alongside AQI.
    depi = None
    dep_c, dep_p = current.get("depreciation"), prior.get("depreciation")
    if None not in (dep_c, dep_p, ppe_c, ppe_p) and (ppe_c + dep_c) > 0 and (ppe_p + dep_p) > 0:
        rate_current = _beneish_ratio(dep_c, ppe_c + dep_c)
        rate_prior = _beneish_ratio(dep_p, ppe_p + dep_p)
        depi = _beneish_ratio(rate_prior, rate_current)

    if any(v is None for v in (dsri, gmi, aqi, sgi, depi)):
        return None

    components = {"dsri": dsri, "gmi": gmi, "aqi": aqi, "sgi": sgi, "depi": depi}

    sgai = _beneish_ratio(
        _beneish_ratio(current.get("sga"), current.get("revenue")),
        _beneish_ratio(prior.get("sga"), prior.get("revenue")),
    )
    lvgi = None
    ltd_c, cl_c = current.get("long_term_debt"), current.get("current_liabilities")
    ltd_p, cl_p = prior.get("long_term_debt"), prior.get("current_liabilities")
    if None not in (ltd_c, cl_c, ta_c, ltd_p, cl_p, ta_p) and ta_c != 0 and ta_p != 0:
        lev_current = (ltd_c + cl_c) / ta_c
        lev_prior = (ltd_p + cl_p) / ta_p
        lvgi = _beneish_ratio(lev_prior, lev_current)
    tata = None
    ni_c, ocf_c = current.get("net_income"), current.get("operating_cash_flow")
    if None not in (ni_c, ocf_c, ta_c) and ta_c != 0:
        tata = (ni_c - ocf_c) / ta_c

    if None not in (sgai, lvgi, tata):
        components.update({"sgai": sgai, "lvgi": lvgi, "tata": tata})
        m_score = (
            -4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
            + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
        )
        partial = False
    else:
        m_score = -6.065 + 0.823 * dsri + 0.906 * gmi + 0.593 * aqi + 0.717 * sgi + 0.107 * depi
        partial = True

    return {
        "m_score": round(m_score, 2),
        "partial": partial,
        "flag": m_score > _BENEISH_MANIPULATION_THRESHOLD,
        "components": {k: round(v, 4) for k, v in components.items()},
    }


#: Merton distance-to-default (SPEC.md Sec.8k) practitioner-heuristic zone
#: thresholds -- NOT a precisely calibrated cutoff (no published academic
#: consensus fixes these), documented explicitly rather than presented as
#: exact: DD >= 3.0 is conventionally "safe" (investment-grade-like);
#: DD < 1.0 is conventionally "distress" (elevated near-term default risk);
#: the band between is "elevated" (worth monitoring, not alarming).
_MERTON_SAFE_DD = 3.0
_MERTON_ELEVATED_DD = 1.0

#: Newton-Raphson solver bounds (SPEC.md Sec.8k): iteration cap and
#: convergence tolerance (RELATIVE to equity_value/equity_vol*equity_value
#: scale, since both can span orders of magnitude across filers) for the
#: two simultaneous Merton equations' residuals.
_MERTON_MAX_ITER = 100
_MERTON_TOLERANCE = 1e-6

#: Finite-difference step size (relative to the current iterate), used to
#: numerically approximate the 2x2 Jacobian each Newton-Raphson step. Pure
#: Python, no scipy -- deterministic central differences.
_MERTON_FD_EPS = 1e-5


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf`` -- no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _merton_residuals(
    asset_value: float, asset_vol: float, equity_value: float, equity_vol: float,
    debt_face_value: float, risk_free_rate: float, horizon_years: float, sqrt_horizon: float,
) -> Optional[tuple]:
    """One evaluation of the two Merton-model equations' residuals at a
    candidate ``(asset_value, asset_vol)``, plus the ``d1``/``d2`` used to
    compute them (reused by the caller for distance-to-default once solved).

    Equation 1 (equity as a call option on firm assets, Black-Scholes):
    ``asset_value*N(d1) - debt_face_value*exp(-r*T)*N(d2) == equity_value``.
    Equation 2 (Ito's lemma relationship between equity and asset
    volatility): ``N(d1)*asset_vol*asset_value == equity_vol*equity_value``.

    Returns ``None`` (a degenerate candidate point, e.g. a negative
    ``asset_value``/``asset_vol`` reached mid-iteration) rather than raising
    on a domain error (``math.log`` of a non-positive number).
    """
    if asset_value <= 0 or asset_vol <= 0:
        return None
    try:
        d1 = (
            math.log(asset_value / debt_face_value) + (risk_free_rate + 0.5 * asset_vol ** 2) * horizon_years
        ) / (asset_vol * sqrt_horizon)
    except ValueError:
        return None
    d2 = d1 - asset_vol * sqrt_horizon
    n_d1 = _norm_cdf(d1)
    n_d2 = _norm_cdf(d2)

    f1 = asset_value * n_d1 - debt_face_value * math.exp(-risk_free_rate * horizon_years) * n_d2 - equity_value
    f2 = n_d1 * asset_vol * asset_value - equity_vol * equity_value
    return f1, f2, d1, d2


def merton_distance_to_default(
    equity_value: Optional[float],
    equity_vol: Optional[float],
    debt_face_value: Optional[float],
    risk_free_rate: Optional[float],
    horizon_years: float = 1.0,
) -> Optional[dict]:
    """Compute the Merton (1974) distance-to-default (SPEC.md Sec.8k):
    equity is modeled as a call option on the firm's assets, struck at the
    face value of debt. Solving the two simultaneous Black-Scholes-shaped
    equations relating observable equity value/volatility to UNOBSERVABLE
    asset value/volatility gives an implied ``distance_to_default`` (how
    many standard deviations of asset-value movement separate today's
    implied asset value from the default point) and a risk-neutral
    ``probability_of_default``.

    Solved via 2D Newton-Raphson with a numerically (finite-difference)
    approximated Jacobian -- pure Python, ``math.erf`` for the normal CDF
    (no scipy). This snapshot-only formulation (a single equity
    value/volatility pair, not a full historical asset-return time series)
    solves both equations SIMULTANEOUSLY rather than the classic KMV
    iterative time-series procedure, which needs a return history this
    engine doesn't reconstruct.

    Distance-to-default and probability of default are read directly off
    the SOLVED ``d2`` (the same ``d2`` already computed to solve the
    equations) rather than as a separate formula: ``distance_to_default =
    d2`` and ``probability_of_default = 1 - N(d2)`` -- this is the
    risk-neutral (drift = risk-free rate) convention, a documented
    simplification of the classic KMV model (which estimates the asset's
    ACTUAL drift from a historical return series this engine doesn't have).

    Args:
        equity_value: Current market value of equity (market cap).
        equity_vol: Annualized equity return volatility (decimal fraction,
            e.g. ``0.35`` for 35%) -- callers typically pass
            :func:`_annualized_volatility`'s output over a ~1-year window
            (NOT the technical/momentum subsystem's 20-day
            ``volatility_20d``, whose window is wrong for a 1-year default
            horizon).
        debt_face_value: Face value of debt (this engine uses total debt).
        risk_free_rate: Annual risk-free rate (decimal fraction).
        horizon_years: The default horizon, in years. Defaults to ``1.0``
            (the standard 1-year KMV/Merton convention).

    Returns:
        ``None`` (never raises, never fabricates) when any input is
        missing/non-positive (except ``risk_free_rate``, which may
        legitimately be zero or negative), or the Newton-Raphson solve
        doesn't converge within :data:`_MERTON_MAX_ITER` iterations (a
        non-convergent solve degrades to ``None`` + the caller's note,
        never a garbage distance-to-default). Otherwise a dict with
        ``distance_to_default`` (4dp), ``probability_of_default`` (6dp),
        ``asset_value`` (the solved, implied total firm asset value),
        ``asset_vol`` (the solved, implied asset volatility), and ``zone``
        (``"safe"``/``"elevated"``/``"distress"``, see
        :data:`_MERTON_SAFE_DD`/:data:`_MERTON_ELEVATED_DD`).
    """
    if equity_value is None or equity_value <= 0:
        return None
    if equity_vol is None or equity_vol <= 0:
        return None
    if debt_face_value is None or debt_face_value <= 0:
        return None
    if risk_free_rate is None:
        return None
    if horizon_years <= 0:
        return None

    sqrt_horizon = math.sqrt(horizon_years)

    def _residuals_at(v: float, sv: float) -> Optional[tuple]:
        return _merton_residuals(
            v, sv, equity_value, equity_vol, debt_face_value, risk_free_rate, horizon_years, sqrt_horizon
        )

    # Standard starting point: total asset value ~= equity + debt face value;
    # asset vol ~= equity vol scaled down by equity's share of that total
    # (a rough initial de-leveraging of the observed equity volatility).
    asset_value = equity_value + debt_face_value
    asset_vol = equity_vol * equity_value / (equity_value + debt_face_value)

    converged = False
    d2 = None
    for _ in range(_MERTON_MAX_ITER):
        point = _residuals_at(asset_value, asset_vol)
        if point is None:
            return None
        f1, f2, _d1, d2 = point

        scale1 = max(1.0, abs(equity_value))
        scale2 = max(1.0, abs(equity_vol * equity_value))
        if abs(f1) < _MERTON_TOLERANCE * scale1 and abs(f2) < _MERTON_TOLERANCE * scale2:
            converged = True
            break

        h_v = max(abs(asset_value), 1.0) * _MERTON_FD_EPS
        h_s = max(abs(asset_vol), 1e-4) * _MERTON_FD_EPS

        p_v_plus = _residuals_at(asset_value + h_v, asset_vol)
        p_v_minus = _residuals_at(asset_value - h_v, asset_vol)
        p_s_plus = _residuals_at(asset_value, asset_vol + h_s)
        p_s_minus = _residuals_at(asset_value, asset_vol - h_s)
        if None in (p_v_plus, p_v_minus, p_s_plus, p_s_minus):
            return None

        df1_dv = (p_v_plus[0] - p_v_minus[0]) / (2 * h_v)
        df2_dv = (p_v_plus[1] - p_v_minus[1]) / (2 * h_v)
        df1_ds = (p_s_plus[0] - p_s_minus[0]) / (2 * h_s)
        df2_ds = (p_s_plus[1] - p_s_minus[1]) / (2 * h_s)

        det = df1_dv * df2_ds - df1_ds * df2_dv
        if det == 0 or not math.isfinite(det):
            return None

        delta_v = (-f1 * df2_ds + f2 * df1_ds) / det
        delta_s = (f1 * df2_dv - f2 * df1_dv) / det

        new_asset_value = asset_value + delta_v
        new_asset_vol = asset_vol + delta_s
        if (
            new_asset_value <= 0 or new_asset_vol <= 0
            or not math.isfinite(new_asset_value) or not math.isfinite(new_asset_vol)
        ):
            return None
        asset_value, asset_vol = new_asset_value, new_asset_vol

    if not converged or d2 is None:
        return None

    distance_to_default = d2
    probability_of_default = 1.0 - _norm_cdf(d2)

    if distance_to_default >= _MERTON_SAFE_DD:
        zone = "safe"
    elif distance_to_default >= _MERTON_ELEVATED_DD:
        zone = "elevated"
    else:
        zone = "distress"

    return {
        "distance_to_default": round(distance_to_default, 4),
        "probability_of_default": round(probability_of_default, 6),
        "asset_value": asset_value,
        "asset_vol": asset_vol,
        "zone": zone,
    }


#: Trading days per year, for annualizing daily-return volatility (SPEC.md
#: Sec.8k) -- same convention as ``technical/indicators.py``'s
#: ``_TRADING_DAYS_PER_YEAR``, NOT imported from there (that module's own
#: ``volatility_20d`` is a 20-day window scoped to the technical/momentum
#: subsystem, the wrong horizon for a 1-year Merton default horizon).
_MERTON_TRADING_DAYS_PER_YEAR = 252

#: Minimum number of daily-return observations required before an
#: annualized volatility estimate is trusted (SPEC.md Sec.8k) -- guards
#: against a noisy estimate from a handful of days (e.g. a recent IPO).
_MERTON_MIN_RETURN_OBSERVATIONS = 60


def _annualized_volatility(price_df, window_days: int = _MERTON_TRADING_DAYS_PER_YEAR) -> Optional[float]:
    """Annualized volatility of daily returns over the trailing
    ``window_days`` (default ~1 trading year), computed independently from
    ``price_df`` (the same DataFrame ``run_valuation`` already receives) --
    deliberately NOT ``technical/indicators.py``'s ``volatility_20d`` (a
    20-day window, the wrong horizon for this 1-year default-horizon model).

    Args:
        price_df: A DataFrame with a ``"Close"`` column (the same shape
            ``fetch.prices.get_price_history`` returns), or ``None``.
        window_days: Trailing window size, in trading days. Defaults to
            :data:`_MERTON_TRADING_DAYS_PER_YEAR` (252, ~1 year).

    Returns:
        The annualized volatility (decimal fraction), or ``None`` (never
        raises) when ``price_df`` is missing/malformed, or fewer than
        :data:`_MERTON_MIN_RETURN_OBSERVATIONS` daily returns are available
        in the trailing window.
    """
    if price_df is None:
        return None
    try:
        close = price_df["Close"].dropna()
        if len(close) < 2:
            return None
        returns = close.pct_change().dropna()
        window = returns.tail(window_days)
        if len(window) < _MERTON_MIN_RETURN_OBSERVATIONS:
            return None
        stdev = window.std()
        if stdev is None or stdev != stdev:  # NaN check without importing pandas/numpy here
            return None
        return float(stdev) * math.sqrt(_MERTON_TRADING_DAYS_PER_YEAR)
    except (KeyError, TypeError, ValueError):
        logger.warning("distress: failed to compute annualized volatility from price_df.", exc_info=True)
        return None
