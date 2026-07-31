"""Compute valuation and quality-of-earnings metrics from normalized facts.

This module sits alongside ``sec_analyzer.normalize.ratios``: where
``ratios.compute_ratios`` derives per-fiscal-year statement ratios (margins,
returns, leverage, FCF) purely from SEC filings, ``compute_metrics`` layers
on a handful of *valuation* metrics that need an external market ``price``
(P/E, P/S, P/FCF, market cap) plus a few "quality" signals used downstream
by ``sec_analyzer.normalize.red_flags`` (share-count dilution trend, SBC and
R&D intensity, revenue CAGR).

Every metric is computed defensively: if a required input is missing (or a
denominator is missing/non-positive where that would make the ratio
meaningless), the metric is reported as ``None`` rather than raising or
producing a misleading number. Nothing in this module raises for missing or
malformed data.
"""

import logging
import math
from typing import Dict, Optional

from sec_analyzer.normalize.normalizer import to_annual_series, to_quarterly_series

logger = logging.getLogger(__name__)

#: Fields in the returned dict that are ratios (P/E-style multiples, growth
#: rates, intensity ratios) and get rounded to 4 decimal places.
_RATIO_FIELDS = (
    "pe", "ps", "pfcf",
    "revenue_cagr_3y", "revenue_cagr_5y",
    "sbc_revenue", "rnd_revenue", "shares_yoy",
)

#: Fields that are a price or a per-share dollar figure, rounded to 2
#: decimal places.
_PER_SHARE_FIELDS = ("price", "eps", "fcf_per_share")

#: Number of fiscal years back used for the two CAGR windows.
_CAGR_WINDOWS = {"revenue_cagr_3y": 3, "revenue_cagr_5y": 5}

#: Price-plausibility floors. A corrupt price feed (e.g. a yfinance fallback
#: that returns a uniformly downscaled series -- observed for BKNG, whose
#: history came back ~31x too small) collapses EVERY price-based multiple
#: together. A legitimately cheap value stock or a peak-cyclical has a low P/E
#: OR a low P/S, but essentially never BOTH a sub-2 trailing P/E AND a sub-0.5
#: P/S at the same time on solidly positive earnings and revenue. So the price
#: is flagged unreliable only when both multiples are positive and below these
#: floors simultaneously -- a deliberately conservative composite that prefers
#: a false negative (occasionally trusting a bad price) over a false positive
#: (wrongly discarding a genuinely cheap stock). See SPEC.md Sec.17.
_PE_IMPLAUSIBLE_FLOOR = 2.0
_PS_IMPLAUSIBLE_FLOOR = 0.5


def _assess_price_reliability(pe: Optional[float], ps: Optional[float]) -> Optional[str]:
    """Return a Turkish note when the price looks corrupt, else ``None``.

    Fires only when both the trailing P/E and P/S are positive and below their
    implausibility floors at once (see :data:`_PE_IMPLAUSIBLE_FLOOR` /
    :data:`_PS_IMPLAUSIBLE_FLOOR`) -- the signature of a uniformly-downscaled
    price feed. Never raises; a missing (``None``) multiple simply can't
    trigger the flag.
    """
    if pe is None or ps is None:
        return None
    if 0 < pe < _PE_IMPLAUSIBLE_FLOOR and 0 < ps < _PS_IMPLAUSIBLE_FLOOR:
        return (
            f"Fiyat güvenilmez olabilir: ima edilen F/K {pe:.2f} ve F/S {ps:.2f} "
            "aynı anda olağandışı düşük (muhtemelen bozuk fiyat verisi); "
            "fiyat bağımlı oranlar dikkatle yorumlanmalı."
        )
    return None


def _round_or_none(value: Optional[float], ndigits: int) -> Optional[float]:
    """Round ``value`` to ``ndigits``, passing ``None`` through unchanged."""
    return None if value is None else round(value, ndigits)


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide two optional numbers, guarding against ``None`` and a
    zero/negative denominator (a negative denominator makes most of the
    multiples computed here -- P/E, P/S, P/FCF -- meaningless)."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Subtract two optional numbers, returning ``None`` if either is missing."""
    if a is None or b is None:
        return None
    return a - b


def _safe_div_allow_negative(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Like ``_safe_div``, but allows a negative numerator/denominator.

    Used for ``fcf_per_share``: unlike the P/E-style multiples, a per-share
    FCF figure is still meaningful (and informative) when FCF is negative,
    so only a missing operand or an exactly-zero share count should suppress
    it, not the sign of either value.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


#: Fewest usable points a log-linear trend needs. With two points the
#: least-squares line IS the two-endpoint line, so it would add nothing while
#: presenting itself as robust.
_MIN_TREND_POINTS = 3


def _trend_growth(series: Dict[int, float], latest_fy: Optional[int], years: int) -> Optional[float]:
    """Annualized growth from a log-linear least-squares trend (SPEC.md Sec.27).

    Replaces a two-endpoint CAGR, whose entire estimate hangs on whatever
    happened in the single starting year. In 2026 that starting year is
    FY2020 for the five-year window, so every filer with a COVID-depressed
    2020 read as a grower while actually shrinking -- Pfizer's revenue has
    fallen from ``$91.8B`` (FY2022) to ``$62.6B``, and the endpoint estimator
    called it an 8.5% grower. Fitting ``ln(value)`` across every point in the
    window puts that year in its place: PFE lands at +2.9%, while a
    well-behaved series barely moves (CAT 10.1% -> 9.7%).

    The window is ``[latest_fy - years, latest_fy]`` inclusive. Only strictly
    positive values participate (the log is undefined otherwise) and
    ``latest_fy`` itself must be present -- deliberately asymmetric, since the
    START point is what the old estimator over-weighted while the END point is
    what keeps the figure anchored to the present.

    Returns ``None`` when fewer than :data:`_MIN_TREND_POINTS` usable points
    remain or they span under two fiscal years. Never raises.
    """
    if latest_fy is None or not series:
        return None
    points = [
        (fy, value) for fy, value in series.items()
        if latest_fy - years <= fy <= latest_fy and value is not None and value > 0
    ]
    if len(points) < _MIN_TREND_POINTS:
        return None
    if not any(fy == latest_fy for fy, _ in points):
        return None
    fiscal_years = [fy for fy, _ in points]
    if max(fiscal_years) - min(fiscal_years) < 2:
        return None

    logs = [math.log(value) for _, value in points]
    fy_mean = sum(fiscal_years) / len(fiscal_years)
    log_mean = sum(logs) / len(logs)
    covariance = sum((fy - fy_mean) * (log_value - log_mean)
                     for fy, log_value in zip(fiscal_years, logs))
    variance = sum((fy - fy_mean) ** 2 for fy in fiscal_years)
    if variance == 0:
        return None
    return math.exp(covariance / variance) - 1.0


def resolve_fundamental_fy(metrics: dict) -> Optional[int]:
    """En yeni *temel* (finansal-tablo) mali yılı; hisse-sayısı kapak
    sayfasının kirlettiği ``latest_fy``'den ayrı.

    ``metrics["latest_fundamental_fy"]`` yoksa veya ``None`` ise
    ``metrics["latest_fy"]``'ye düşer. Böylece bu anahtarı üretmeden metrics
    dict'i kuran çağıranlar (özellikle testler) eski davranışı korur.
    """
    m = metrics or {}
    fy = m.get("latest_fundamental_fy")
    return fy if fy is not None else m.get("latest_fy")


#: How many single quarters make a trailing-twelve-month window.
_TTM_QUARTERS = 4


def _empty_ttm() -> dict:
    return {
        "ttm_revenue": None, "ttm_net_income": None, "ttm_eps": None,
        "ttm_net_margin": None, "ttm_period_end": None, "ttm_quarters": 0,
        "ttm_complete": False, "pe_ttm": None, "ttm_vs_fy_net_income": None,
    }


def _compute_ttm(
    normalized: dict, shares: Optional[float], price: Optional[float],
    latest_fy_net_income: Optional[float],
) -> dict:
    """Trailing-twelve-month revenue/earnings from the quarterly series.

    The valuation path reads ANNUAL series only, so a filer three quarters
    into its fiscal year is valued on data that predates those quarters --
    which for a cyclical mid-upswing is not a rounding issue (Micron on
    2026-07-30: latest-FY net income ``$8.54B`` against a true TTM of
    ``$50.47B``, a reported P/E of 97.4 against a true 16.5). These figures
    are reported ALONGSIDE the fiscal-year ones; ``pe``/``ps``/``pfcf`` keep
    their existing basis so percentiles and stored verdicts stay comparable
    (SPEC.md Sec.24b).

    Inherits Sec.24a's corrected fiscal-year grouping through
    ``to_quarterly_series``. An incomplete window (fewer than four quarters)
    still reports its partial sums but sets ``ttm_complete`` False and leaves
    ``pe_ttm`` None -- a partial sum is not a trailing-twelve-month figure.
    Never raises.
    """
    try:
        revenue_q = to_quarterly_series(normalized, "Revenue")[-_TTM_QUARTERS:]
        net_income_q = to_quarterly_series(normalized, "NetIncome")[-_TTM_QUARTERS:]
    except Exception:  # noqa: BLE001 - metrics must never crash the pipeline
        logger.warning("compute_metrics: TTM window could not be built.", exc_info=True)
        return _empty_ttm()

    if not net_income_q:
        return _empty_ttm()

    result = _empty_ttm()
    result["ttm_quarters"] = len(net_income_q)
    result["ttm_complete"] = len(net_income_q) == _TTM_QUARTERS
    result["ttm_period_end"] = net_income_q[-1]["period_end"]
    result["ttm_net_income"] = sum(q["value"] for q in net_income_q)
    if revenue_q:
        result["ttm_revenue"] = sum(q["value"] for q in revenue_q)
        result["ttm_net_margin"] = _safe_div(result["ttm_net_income"], result["ttm_revenue"])

    result["ttm_eps"] = _safe_div_allow_negative(result["ttm_net_income"], shares)
    if result["ttm_complete"] and price is not None:
        result["pe_ttm"] = _safe_div(price, result["ttm_eps"])
    if latest_fy_net_income:
        result["ttm_vs_fy_net_income"] = _safe_div(
            result["ttm_net_income"], latest_fy_net_income
        )
    return result


def compute_metrics(normalized: dict, ratios: list, price: Optional[float]) -> dict:
    """Compute valuation and quality metrics for the latest fiscal year.

    Args:
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.
        ratios: The list returned by
            ``sec_analyzer.normalize.ratios.compute_ratios`` (used for its
            per-fiscal-year ``fcf`` figure; if a fiscal year's ``fcf`` isn't
            present there, it's recomputed from OperatingCashFlow - CapEx).
        price: The latest market price per share, or ``None`` if unknown --
            every price-dependent metric (market cap, P/E, P/S, P/FCF) is
            ``None`` when ``price`` is ``None``, but everything else
            (CAGRs, SBC/R&D intensity, dilution trend, raw FCF) is still
            computed.

    Returns:
        A dict with keys ``price``, ``shares``, ``eps``, ``market_cap``,
        ``total_debt``, ``net_debt``, ``pe``, ``ps``, ``pfcf``,
        ``operating_income`` (EBIT), ``ebitda`` (EBIT + D&A), ``ev``
        (market cap + net debt), ``ev_ebit``, ``ev_ebitda`` (enterprise-value
        earnings multiples, ``None`` unless the denominator is strictly
        positive), ``revenue_cagr_3y``, ``revenue_cagr_5y``, ``sbc_revenue``,
        ``shares_yoy``, ``buyback_latest``, ``dividends_latest``,
        ``rnd_revenue``, ``fcf``, ``fcf_per_share``, ``latest_fy``,
        ``latest_fundamental_fy``, ``price_reliable`` (bool), and
        ``price_reliability_note`` (str or ``None`` -- see
        :func:`_assess_price_reliability`). Every value is ``None``-safe; ratios/growth
        rates are rounded to 4 decimal places, price/per-share dollar figures
        to 2, and raw USD amounts (``market_cap``, ``total_debt``,
        ``net_debt``, ``buyback_latest``, ``dividends_latest``, ``fcf``) are
        left unrounded. ``latest_fy`` is the latest fiscal year across ALL
        series including ``SharesOutstanding`` (used only for share count and
        market cap); ``latest_fundamental_fy`` is the latest fiscal year
        across every series EXCEPT ``SharesOutstanding`` and is what every
        other fundamental-data read (EPS, revenue, FCF, CAGRs, ...) is
        anchored to, since the ``SharesOutstanding`` cover-page series can
        carry a fiscal year newer than the financial statements actually
        report (e.g. AMZN).
    """
    shares_series = to_annual_series(normalized, "SharesOutstanding")
    eps_series = to_annual_series(normalized, "EPS")
    ltd_series = to_annual_series(normalized, "LongTermDebt")
    ltdc_series = to_annual_series(normalized, "LongTermDebtCurrent")
    cash_series = to_annual_series(normalized, "Cash")
    revenue_series = to_annual_series(normalized, "Revenue")
    sbc_series = to_annual_series(normalized, "SBC")
    rnd_series = to_annual_series(normalized, "RnD")
    buyback_series = to_annual_series(normalized, "Buyback")
    dividends_series = to_annual_series(normalized, "DividendsPaid")
    ocf_series = to_annual_series(normalized, "OperatingCashFlow")
    capex_series = to_annual_series(normalized, "CapEx")
    # EBIT / EBITDA inputs for the enterprise-value earnings multiples. NOT
    # folded into the fiscal-year union sets below (latest_fy /
    # latest_fundamental_fy anchoring is intentionally left unchanged) -- the
    # EV multiples are additive and simply read at latest_fundamental_fy.
    operating_income_series = to_annual_series(normalized, "OperatingIncome")
    depreciation_series = to_annual_series(normalized, "Depreciation")

    all_series = (
        shares_series, eps_series, ltd_series, ltdc_series, cash_series,
        revenue_series, sbc_series, rnd_series, buyback_series,
        dividends_series, ocf_series, capex_series,
    )
    fiscal_years: set = set()
    for series in all_series:
        fiscal_years |= set(series)

    if not fiscal_years:
        logger.debug(
            "compute_metrics: no annual data available for %s (CIK %s); "
            "returning all-None metrics.",
            normalized.get("entity_name"), normalized.get("cik"),
        )
        return {
            "price": _round_or_none(price, 2),
            "shares": None, "eps": None, "market_cap": None,
            "total_debt": None, "net_debt": None,
            "pe": None, "ps": None, "pfcf": None,
            "operating_income": None, "ebitda": None, "ev": None,
            "ev_ebit": None, "ev_ebitda": None,
            "revenue_cagr_3y": None, "revenue_cagr_5y": None,
            "tangible_equity": None, "tbv_per_share": None, "ptbv": None,
            **_empty_ttm(),
            "sbc_revenue": None, "shares_yoy": None,
            "buyback_latest": None, "dividends_latest": None,
            "rnd_revenue": None, "fcf": None, "fcf_per_share": None,
            "latest_fy": None, "latest_fundamental_fy": None,
            "price_reliable": True, "price_reliability_note": None,
        }

    latest_fy = max(fiscal_years)
    # SharesOutstanding kapak sayfası (dei) nokta-zaman serisidir ve bazı
    # filer'larda en yeni 10-K'nın finansal tablolarından daha yeni bir mali
    # yıl taşır (ör. AMZN). Değerleme çapasını bu seriden ayır: fundamental
    # veriler (gelir tablosu / nakit akışı / bilanço) SharesOutstanding HARİÇ
    # serilerin en yenisinden okunur. Bu dışlama, mevcut kavram setinde tek
    # nokta-zaman serisinin SharesOutstanding olması varsayımına dayanır;
    # gelecekte başka bir kapak-sayfası serisi eklenirse buradaki dışlama
    # listesi güncellenmelidir.
    fundamental_series = (
        eps_series, ltd_series, ltdc_series, cash_series, revenue_series,
        sbc_series, rnd_series, buyback_series, dividends_series,
        ocf_series, capex_series,
    )
    fundamental_years: set = set()
    for series in fundamental_series:
        fundamental_years |= set(series)
    latest_fundamental_fy = max(fundamental_years) if fundamental_years else latest_fy

    prev_fy = latest_fy - 1

    shares = shares_series.get(latest_fy)
    shares_prev = shares_series.get(prev_fy)
    eps = eps_series.get(latest_fundamental_fy)
    ltd = ltd_series.get(latest_fundamental_fy)
    ltdc = ltdc_series.get(latest_fundamental_fy)
    cash = cash_series.get(latest_fundamental_fy)
    revenue = revenue_series.get(latest_fundamental_fy)
    sbc = sbc_series.get(latest_fundamental_fy)
    rnd = rnd_series.get(latest_fundamental_fy)
    buyback = buyback_series.get(latest_fundamental_fy)
    dividends = dividends_series.get(latest_fundamental_fy)

    if ltd is None and ltdc is None:
        total_debt = None
    else:
        total_debt = (ltd or 0.0) + (ltdc or 0.0)
    net_debt = _safe_sub(total_debt, cash)

    market_cap = None if price is None or shares is None else price * shares

    pe = None if price is None else _safe_div(price, eps)
    ps = None if price is None else _safe_div(market_cap, revenue)

    ttm = _compute_ttm(
        normalized, shares, price,
        to_annual_series(normalized, "NetIncome").get(latest_fundamental_fy),
    )

    ratio_by_fy = {r["fy"]: r for r in (ratios or []) if r.get("fy") is not None}

    # Tangible-equity figures (SPEC.md Sec.23c). Read from the SAME fiscal
    # year's ratio row that supplies `fcf` below, so they never describe a
    # different period than the rest of this dict. `ptbv` is only defined for
    # a strictly positive tangible base.
    tangible_equity = ratio_by_fy.get(latest_fundamental_fy, {}).get("tangible_equity")
    if tangible_equity is not None and tangible_equity <= 0:
        tangible_equity = None
    tbv_per_share = _safe_div(tangible_equity, shares)
    ptbv = None if price is None else _safe_div(price, tbv_per_share)

    fcf = ratio_by_fy.get(latest_fundamental_fy, {}).get("fcf")
    if fcf is None:
        fcf = _safe_sub(ocf_series.get(latest_fundamental_fy), capex_series.get(latest_fundamental_fy))
    pfcf = None if price is None else _safe_div(market_cap, fcf)

    # Enterprise-value earnings multiples (EV/EBIT, EV/EBITDA). EV = market cap
    # + net debt (net debt treated as 0.0 -> unlevered EV = market cap when it
    # can't be derived, mirroring multiples_history's ev_sales degradation).
    # EBIT = OperatingIncome; EBITDA = EBIT + D&A (both concepts required, no
    # zero-fill). Multiples only defined for a strictly positive denominator.
    operating_income = operating_income_series.get(latest_fundamental_fy)
    depreciation = depreciation_series.get(latest_fundamental_fy)
    ebitda = (
        None
        if operating_income is None or depreciation is None
        else operating_income + depreciation
    )
    ev = None if market_cap is None else market_cap + (net_debt or 0.0)
    ev_ebit = (
        _safe_div(ev, operating_income)
        if operating_income is not None and operating_income > 0
        else None
    )
    ev_ebitda = _safe_div(ev, ebitda) if ebitda is not None and ebitda > 0 else None

    revenue_cagr_3y = _trend_growth(revenue_series, latest_fundamental_fy, _CAGR_WINDOWS["revenue_cagr_3y"])
    revenue_cagr_5y = _trend_growth(revenue_series, latest_fundamental_fy, _CAGR_WINDOWS["revenue_cagr_5y"])

    sbc_revenue = _safe_div(sbc, revenue)
    rnd_revenue = _safe_div(rnd, revenue)

    if shares is None or shares_prev is None or shares_prev == 0:
        shares_yoy = None
    else:
        shares_yoy = shares / shares_prev - 1.0

    fcf_per_share = None if shares is None or shares == 0 else _safe_div_allow_negative(fcf, shares)

    result = {
        "price": price,
        "shares": shares,
        "eps": eps,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "net_debt": net_debt,
        "pe": pe,
        "ps": ps,
        "pfcf": pfcf,
        "operating_income": operating_income,
        "ebitda": ebitda,
        "ev": ev,
        "ev_ebit": ev_ebit,
        "ev_ebitda": ev_ebitda,
        "tangible_equity": tangible_equity,
        "tbv_per_share": tbv_per_share,
        "ptbv": ptbv,
        **ttm,
        "revenue_cagr_3y": revenue_cagr_3y,
        "revenue_cagr_5y": revenue_cagr_5y,
        "sbc_revenue": sbc_revenue,
        "shares_yoy": shares_yoy,
        "buyback_latest": buyback,
        "dividends_latest": dividends,
        "rnd_revenue": rnd_revenue,
        "fcf": fcf,
        "fcf_per_share": fcf_per_share,
        "latest_fy": latest_fy,
        "latest_fundamental_fy": latest_fundamental_fy,
    }

    for field in _RATIO_FIELDS:
        result[field] = _round_or_none(result[field], 4)
    for field in _PER_SHARE_FIELDS:
        result[field] = _round_or_none(result[field], 2)

    # Price-plausibility cross-check against SEC per-share fundamentals (the
    # only price-independent reference available): a corrupt market-data feed
    # collapses P/E and P/S together (see _assess_price_reliability). Additive,
    # never suppresses any existing metric -- downstream consumers decide what
    # to do with an unreliable price (e.g. the calibration ratio skips it).
    price_reliability_note = _assess_price_reliability(result["pe"], result["ps"])
    result["price_reliable"] = price_reliability_note is None
    result["price_reliability_note"] = price_reliability_note

    logger.debug(
        "compute_metrics: %s (CIK %s) latest_fy=%s pe=%s ps=%s pfcf=%s",
        normalized.get("entity_name"), normalized.get("cik"),
        latest_fy, result["pe"], result["ps"], result["pfcf"],
    )
    return result
