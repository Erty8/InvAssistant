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
from typing import Dict, Optional

from sec_analyzer.normalize.normalizer import to_annual_series

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


def _cagr(series: Dict[int, float], latest_fy: Optional[int], years: int) -> Optional[float]:
    """Compute a fixed-window CAGR: ``(latest / (latest_fy - years))^(1/years) - 1``.

    Unlike ``rule_based._windowed_cagr``, this does NOT fall back to the
    oldest year available within the window -- per the spec, it needs the
    exact earlier fiscal year present, otherwise it's ``None``. Both
    endpoints must be strictly positive for the growth rate to be meaningful.
    """
    if latest_fy is None:
        return None
    latest_val = series.get(latest_fy)
    earlier_val = series.get(latest_fy - years)
    if latest_val is None or earlier_val is None or latest_val <= 0 or earlier_val <= 0:
        return None
    return (latest_val / earlier_val) ** (1.0 / years) - 1.0


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
        ``revenue_cagr_3y``, ``revenue_cagr_5y``, ``sbc_revenue``,
        ``shares_yoy``, ``buyback_latest``, ``dividends_latest``,
        ``rnd_revenue``, ``fcf``, ``fcf_per_share``, ``latest_fy``,
        ``latest_fundamental_fy``. Every value is ``None``-safe; ratios/growth
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
            "revenue_cagr_3y": None, "revenue_cagr_5y": None,
            "sbc_revenue": None, "shares_yoy": None,
            "buyback_latest": None, "dividends_latest": None,
            "rnd_revenue": None, "fcf": None, "fcf_per_share": None,
            "latest_fy": None, "latest_fundamental_fy": None,
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

    ratio_by_fy = {r["fy"]: r for r in (ratios or []) if r.get("fy") is not None}
    fcf = ratio_by_fy.get(latest_fundamental_fy, {}).get("fcf")
    if fcf is None:
        fcf = _safe_sub(ocf_series.get(latest_fundamental_fy), capex_series.get(latest_fundamental_fy))
    pfcf = None if price is None else _safe_div(market_cap, fcf)

    revenue_cagr_3y = _cagr(revenue_series, latest_fundamental_fy, _CAGR_WINDOWS["revenue_cagr_3y"])
    revenue_cagr_5y = _cagr(revenue_series, latest_fundamental_fy, _CAGR_WINDOWS["revenue_cagr_5y"])

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

    logger.debug(
        "compute_metrics: %s (CIK %s) latest_fy=%s pe=%s ps=%s pfcf=%s",
        normalized.get("entity_name"), normalized.get("cik"),
        latest_fy, result["pe"], result["ps"], result["pfcf"],
    )
    return result
