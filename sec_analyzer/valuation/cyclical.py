"""Through-cycle statistics and two-regime valuation for deep cyclicals.

A commodity producer's current earnings are a function of where the cycle
sits, not of its earning power, so capitalizing them -- in either direction --
is the classic way to be wrong about one. This module supplies the two things
a cyclical read needs and the rest of the engine did not have:

* **Where in the cycle are we** (:func:`through_cycle_stats`): the net-margin
  distribution across full cycles, today's percentile within it, and the
  historical price-to-book band the market has actually paid at troughs,
  medians and peaks.
* **What is the price assuming** (:func:`two_regime_valuation`): instead of
  one fair value, two -- a mean-reverting regime and a structural-break
  regime -- plus the probability of the break that today's price implies.

That last number is the point. "Fair value $X" invites an argument about $X;
"the price requires you to believe there is at least a 70% chance the
economics permanently changed" is a claim the reader can actually assess.

Everything here is pure and deterministic, never raises, and adds no
dependency. See ``sec_analyzer/valuation/SPEC.md`` Sec.25-26 for the binding
contract.
"""

import logging
import statistics
from typing import List, Optional

from sec_analyzer.valuation import multiples

logger = logging.getLogger(__name__)

#: Fewer fiscal years than this cannot describe a cycle, so no through-cycle
#: statistic is produced at all rather than one built from half a cycle. Set
#: to the CLI's default normalization window: a memory/commodity cycle runs
#: roughly 3-4 years, so five annual observations can just span one
#: peak-to-trough-to-peak round trip.
_MIN_CYCLE_YEARS = 5

#: Below this many years the window is flagged as short. It is wide enough to
#: hold one cycle but narrow enough that WHICH cycle it caught dominates the
#: average -- Micron over FY2021-25 averages a 7.6% net margin (the 2023
#: collapse, no 2018 peak) against 14.7% over FY2016-25. Same statistic,
#: nearly double, purely from the window.
_SHORT_CYCLE_WINDOW_YEARS = 8

#: How many recent fiscal years feed the mean-reverting revenue base.
_NORMALIZED_REVENUE_YEARS = 3

#: Regime B "new normal" net margins -- low / mid / high. JUDGMENT INPUTS,
#: not derived from any filing: they encode the thesis that the industry's
#: economics changed permanently. Defaults follow the operator's cyclical
#: brief for a memory maker in an HBM-led re-rating.
_REGIME_B_MARGINS = (0.30, 0.38, 0.45)

#: Regime B exit multiples -- deliberately below the semiconductor median,
#: because a structurally re-rated memory maker is still capital-intensive
#: and price-taking at the margin. Also a judgment input.
_REGIME_B_MULTIPLES = (12.0, 13.5, 15.0)

#: Probabilities tabulated for the regime blend.
_BLEND_PROBABILITIES = (0.25, 0.50, 0.75)

#: Current net margin at or above this percentile of its own history counts
#: as "at the top of the cycle" for the peak-cycle P/E trap flag.
_PEAK_MARGIN_PERCENTILE = 80.0

#: TTM P/E below this, WITH a peak-percentile margin, is the trap: a cyclical
#: looks cheapest exactly when its earnings are least sustainable.
_PEAK_PE_MAX = 15.0

#: Newest-quarter net margin exceeding the through-cycle mean by more than
#: this (in percentage points) makes any annualization of it a peak
#: extrapolation.
_PEAK_ANNUALIZATION_SPREAD = 0.20


def _finite(values) -> List[float]:
    return [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def through_cycle_stats(
    normalized: dict, ratios: list, history: Optional[list], metrics: dict
) -> Optional[dict]:
    """Describe where in its cycle a filer currently sits.

    Args:
        normalized: ``normalize_facts`` output (revenue series for the
            mean-reverting base).
        ratios: ``compute_ratios`` output (per-fiscal-year ``net_margin``).
        history: ``multiples.multiples_history`` output, for the observed
            P/B and P/E bands. May be ``None``/empty -- the margin statistics
            still compute, the band fields degrade to ``None``.
        metrics: ``compute_metrics`` output (TTM figures, shares, book value).

    Returns:
        The stats dict documented in SPEC.md Sec.25b, or ``None`` when fewer
        than :data:`_MIN_CYCLE_YEARS` fiscal years carry a usable net margin
        (a "through-cycle" average from half a cycle is not one). Never
        raises.
    """
    try:
        return _through_cycle_stats(normalized or {}, ratios or [], history or [], metrics or {})
    except Exception:  # noqa: BLE001 - analysis layer must never crash the CLI
        logger.warning("through_cycle_stats: unexpected error; returning None.", exc_info=True)
        return None


def _through_cycle_stats(normalized: dict, ratios: list, history: list, metrics: dict) -> Optional[dict]:
    from sec_analyzer.normalize.normalizer import to_annual_series

    margin_by_fy = {
        row["fy"]: row["net_margin"]
        for row in ratios
        if row.get("fy") is not None and isinstance(row.get("net_margin"), (int, float))
    }
    if len(margin_by_fy) < _MIN_CYCLE_YEARS:
        return None

    margins = list(margin_by_fy.values())
    trough_fy = min(margin_by_fy, key=lambda fy: margin_by_fy[fy])
    peak_fy = max(margin_by_fy, key=lambda fy: margin_by_fy[fy])

    # Today's cycle position uses TTM when the window is complete: for a filer
    # three quarters into an unreported fiscal year, the latest ANNUAL margin
    # can describe a different part of the cycle entirely.
    if metrics.get("ttm_complete") and isinstance(metrics.get("ttm_net_margin"), (int, float)):
        margin_current, basis = metrics["ttm_net_margin"], "ttm"
    else:
        latest_fy = max(margin_by_fy)
        margin_current, basis = margin_by_fy[latest_fy], "fy"

    pb_values = _finite(h.get("pb") for h in history)
    pe_values = [v for v in _finite(h.get("pe") for h in history) if v > 0]

    shares = metrics.get("shares")
    equity_series = to_annual_series(normalized, "StockholdersEquity")
    book_value_per_share = None
    if equity_series and shares:
        latest_equity = equity_series[max(equity_series)]
        if latest_equity and latest_equity > 0:
            book_value_per_share = latest_equity / shares

    revenue_series = to_annual_series(normalized, "Revenue")
    normalized_revenue, revenue_basis_years = _normalized_revenue(revenue_series, metrics)

    return {
        "years": len(margin_by_fy),
        "window_short": len(margin_by_fy) < _SHORT_CYCLE_WINDOW_YEARS,
        "margin_mean": statistics.fmean(margins),
        "margin_median": statistics.median(margins),
        "margin_trough": margin_by_fy[trough_fy],
        "margin_trough_fy": trough_fy,
        "margin_peak": margin_by_fy[peak_fy],
        "margin_peak_fy": peak_fy,
        "margin_current": margin_current,
        "margin_current_basis": basis,
        "margin_percentile": multiples.percentile_position(margins, margin_current),
        "pb_trough": min(pb_values) if pb_values else None,
        "pb_median": statistics.median(pb_values) if pb_values else None,
        "pb_peak": max(pb_values) if pb_values else None,
        "pb_current": (
            metrics["price"] / book_value_per_share
            if book_value_per_share and isinstance(metrics.get("price"), (int, float))
            else None
        ),
        "pe_median": statistics.median(pe_values) if pe_values else None,
        "book_value_per_share": book_value_per_share,
        "normalized_revenue": normalized_revenue,
        "revenue_basis_years": revenue_basis_years,
    }


def _normalized_revenue(revenue_series: dict, metrics: dict):
    """Mean-reverting revenue base: the last three fiscal years' average.

    TTM revenue replaces the most recent point when the TTM window is
    complete and covers a period the annual series has not reported yet --
    otherwise a filer deep into an unreported upswing gets a "normalized"
    base that predates the upswing entirely.

    Deliberately applies NO unit/bit-growth uplift: that is a judgment input
    this layer cannot derive from filings, and omitting it is the
    conservative direction.
    """
    if not revenue_series:
        return None, []

    years = sorted(revenue_series, reverse=True)[:_NORMALIZED_REVENUE_YEARS]
    values = [revenue_series[fy] for fy in years]
    basis = list(reversed(years))

    ttm_revenue = metrics.get("ttm_revenue")
    if metrics.get("ttm_complete") and isinstance(ttm_revenue, (int, float)) and values:
        # Substitute for the newest annual point rather than appending, so the
        # window stays three periods wide.
        values[0] = ttm_revenue
        basis = basis[:-1] + ["TTM"]

    return statistics.fmean(values), basis


def two_regime_valuation(stats: Optional[dict], metrics: dict, price: Optional[float]) -> Optional[dict]:
    """Two regime fair values, their blend, and the price-implied break odds.

    Regime A assumes the cycle mean-reverts: normalized earnings at the
    through-cycle margin, capitalized at the multiples the market has
    historically paid. Regime B assumes the economics changed permanently and
    the current run-rate is the new base.

    Neither is a forecast. The output that matters is ``p_implied`` -- the
    probability of the structural break that today's price already requires
    the buyer to believe.

    Returns ``None`` when ``stats`` is missing or neither regime can be built.
    Never raises. Advisory only: SPEC.md Sec.26 forbids this from headlining
    ``fair_value_range``, feeding ``primary_dcf_scenarios``, or entering
    triangulation.
    """
    try:
        return _two_regime_valuation(stats, metrics or {}, price)
    except Exception:  # noqa: BLE001
        logger.warning("two_regime_valuation: unexpected error; returning None.", exc_info=True)
        return None


def _two_regime_valuation(stats: Optional[dict], metrics: dict, price: Optional[float]) -> Optional[dict]:
    if not stats:
        return None
    shares = metrics.get("shares")
    if not shares or shares <= 0:
        return None

    regime_a = _regime_a(stats, shares)
    regime_b = _regime_b(stats, metrics, shares)
    if regime_a is None and regime_b is None:
        return None

    fv_a = (regime_a or {}).get("center")
    fv_b = (regime_b or {}).get("center")

    blend = {}
    if fv_a is not None and fv_b is not None:
        blend = {
            f"{p:.2f}": p * fv_b + (1.0 - p) * fv_a for p in _BLEND_PROBABILITIES
        }

    p_implied, status = _implied_probability(price, fv_a, fv_b)

    return {
        "stats": stats,
        "regime_a": regime_a,
        "regime_b": regime_b,
        "blend": blend,
        "p_implied": p_implied,
        "p_implied_status": status,
        "verdict_sentence": _verdict_sentence(p_implied, status),
        "flags": _flags(stats, metrics),
    }


def _regime_a(stats: dict, shares: float) -> Optional[dict]:
    """Mean-reversion regime: through-cycle margins at historical multiples."""
    normalized_revenue = stats.get("normalized_revenue")
    bvps = stats.get("book_value_per_share")
    if not isinstance(normalized_revenue, (int, float)):
        normalized_revenue = None

    eps = (
        stats["margin_mean"] * normalized_revenue / shares
        if normalized_revenue is not None else None
    )
    trough_eps = (
        stats["margin_trough"] * normalized_revenue / shares
        if normalized_revenue is not None else None
    )

    # "Compute both, take the lower": an earnings-based and a book-based read
    # of the same mean-reverting thesis, with the conservative one leading.
    candidates = []
    if eps is not None and eps > 0 and stats.get("pe_median"):
        candidates.append(eps * stats["pe_median"])
    if bvps and stats.get("pb_median"):
        candidates.append(stats["pb_median"] * bvps)
    center = min(candidates) if candidates else None

    low = stats["pb_trough"] * bvps if bvps and stats.get("pb_trough") else None
    high = stats["pb_peak"] * bvps if bvps and stats.get("pb_peak") else None

    if center is None and low is None and high is None:
        return None

    # The two ends are derived independently (a book multiple for the floor,
    # the lower of an earnings and a book read for the centre), so they can
    # cross: the earnings read coming in BELOW the historical trough multiple
    # means the two methods disagree about the bottom. A band whose floor sits
    # above its own centre is unreadable, so the ends are pulled in -- and the
    # disagreement is recorded rather than hidden.
    band_crossed = False
    if low is not None and center is not None and low > center:
        low, band_crossed = center, True
    if high is not None and center is not None and high < center:
        high, band_crossed = center, True

    return {
        "normalized_eps": eps,
        "trough_eps": trough_eps,
        "low": low,
        "center": center,
        "high": high,
        "band_crossed": band_crossed,
        "basis": "through-cycle ortalama marj + tarihsel çarpan bandı",
    }


def _regime_b(stats: dict, metrics: dict, shares: float) -> Optional[dict]:
    """Structural-break regime: the current run-rate is the new base."""
    revenue = metrics.get("ttm_revenue")
    if not metrics.get("ttm_complete") or not isinstance(revenue, (int, float)) or revenue <= 0:
        return None

    grid = {}
    for margin in _REGIME_B_MARGINS:
        eps = margin * revenue / shares
        grid[f"{margin:.2f}"] = {
            f"{mult:.1f}": eps * mult for mult in _REGIME_B_MULTIPLES
        }

    lo_m, mid_m, hi_m = _REGIME_B_MARGINS
    lo_x, mid_x, hi_x = _REGIME_B_MULTIPLES
    return {
        "revenue_base": revenue,
        "revenue_basis": "TTM",
        "margins": list(_REGIME_B_MARGINS),
        "multiples": list(_REGIME_B_MULTIPLES),
        "eps": {f"{m:.2f}": m * revenue / shares for m in _REGIME_B_MARGINS},
        "sensitivity": grid,
        "low": grid[f"{lo_m:.2f}"][f"{lo_x:.1f}"],
        "center": grid[f"{mid_m:.2f}"][f"{mid_x:.1f}"],
        "high": grid[f"{hi_m:.2f}"][f"{hi_x:.1f}"],
        "basis": "yapısal kırılım: mevcut TTM gelir kalıcı, yeni-normal marj bandı",
    }


def _implied_probability(price, fv_a, fv_b):
    """Solve ``price = p * fv_b + (1 - p) * fv_a`` for ``p``.

    Reported raw, never clamped: a value outside ``[0, 1]`` is the finding --
    no mix of the two regimes reconciles the price with either.
    """
    if price is None or fv_a is None or fv_b is None:
        return None, "no_data"
    if fv_b == fv_a:
        return None, "degenerate"
    p = (price - fv_a) / (fv_b - fv_a)
    if p > 1.0:
        return p, "above_range"
    if p < 0.0:
        return p, "below_range"
    return p, "ok"


def _verdict_sentence(p_implied: Optional[float], status: str) -> str:
    if status == "ok" and p_implied is not None:
        return (
            f"Mevcut fiyat, yapısal kırılıma ≥ %{p_implied * 100:.0f} olasılık vermeyi "
            "zorunlu kılıyor. Kendi p tahminin bunun üstünde ise fiyat ucuz, altında ise pahalı."
        )
    if status == "above_range":
        return (
            "Mevcut fiyat, yapısal kırılıma %100 olasılık verilse bile iki rejimin "
            "üstünde kalıyor: fiyatı bu iki rejimin hiçbir olasılık karışımı açıklamıyor "
            "(kırılım senaryosunun kendisi de fiyattan ucuz)."
        )
    if status == "below_range":
        return (
            "Mevcut fiyat, döngü-ortalaması rejiminin bile altında: yapısal kırılıma sıfır "
            "olasılık verilse dahi fiyat iki rejimin altında kalıyor."
        )
    return "Fiyatın ima ettiği yapısal-kırılım olasılığı hesaplanamadı (rejim verisi eksik)."


def _flags(stats: dict, metrics: dict) -> dict:
    """The three cycle-specific traps the source methodology calls out."""
    percentile = stats.get("margin_percentile")
    pe_ttm = metrics.get("pe_ttm")

    # The existing red_flags._check_cyclical_trap reads metrics["pe"] -- the
    # fiscal-year P/E -- which for a filer mid-upswing is wildly overstated
    # (Micron: 97.4 reported against a true TTM 16.5), so it stays silent
    # exactly when the trap is live. This reads the TTM P/E.
    peak_pe_trap = (
        isinstance(percentile, (int, float)) and percentile >= _PEAK_MARGIN_PERCENTILE
        and isinstance(pe_ttm, (int, float)) and 0 < pe_ttm < _PEAK_PE_MAX
    )

    margin_current = stats.get("margin_current")
    peak_annualization = (
        isinstance(margin_current, (int, float))
        and margin_current - stats["margin_mean"] > _PEAK_ANNUALIZATION_SPREAD
    )

    pb_current, pb_peak = stats.get("pb_current"), stats.get("pb_peak")
    regime_change_premium = (
        isinstance(pb_current, (int, float)) and isinstance(pb_peak, (int, float))
        and pb_current > pb_peak
    )

    return {
        "peak_cycle_pe_trap": bool(peak_pe_trap),
        "peak_annualization": bool(peak_annualization),
        "regime_change_premium": bool(regime_change_premium),
    }
