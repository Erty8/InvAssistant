"""CAPM cost of equity for the DCF discount rate.

The valuation engine's discount rate is a levered COST OF EQUITY (özkaynak
maliyeti, SPEC.md Sec.3), not a WACC. Historically the deterministic
(``script``) path used a flat sector-agnostic default (10%, or 12% for
unprofitable filers -- see ``rule_based._DEFAULT_DISCOUNT_RATE_BASE``). This
module replaces that flat constant, when the reference data is available, with
a firm-specific CAPM estimate:

    cost_of_equity = risk_free + β_levered × ERP

built from Aswath Damodaran's public sector data (already loaded by
:mod:`sec_analyzer.valuation.damodaran`):

* ``risk_free`` and ``ERP`` come from ``erp.csv`` (US row), as PERCENTAGE
  numbers (e.g. ``4.20`` means 4.2%).
* ``β_unlevered`` is the matched sector's unlevered (asset) beta from
  ``multiples.csv``'s optional ``unlevered_beta`` column.
* ``β_levered`` re-levers that sector asset beta with the *filer's own*
  market debt/equity and a marginal tax rate (Hamada):

      β_levered = β_unlevered × (1 + (1 − tax) × D/E)

The result is a decimal fraction (``0.106`` for 10.6%), clamped into a sane
band. It is only ever a *base* rate: the caller derives bear/bull rates from
it, and :func:`sec_analyzer.valuation.sanity.clamp_assumptions` (applied inside
``run_valuation``) still floors every per-scenario rate and enforces the
Gordon ERP-spread guard downstream -- so a low CAPM estimate can't produce an
invalid scenario.

Everything here is best-effort: if any required input is missing (no sector
data, unmatched SIC, no beta/ERP/risk-free), :func:`compute_cost_of_equity`
returns ``None`` and the caller falls back to the flat default. It never
raises.
"""

import logging
from typing import Optional

from sec_analyzer.valuation import sanity
from sec_analyzer.valuation.damodaran import sector_medians

logger = logging.getLogger(__name__)

#: Marginal corporate tax rate used to re-lever the sector asset beta
#: (Damodaran's standard US marginal-rate convention). A modeling assumption,
#: not a per-filer effective rate -- refining it to the filer's own effective
#: tax rate is a possible future improvement.
_DEFAULT_TAX_RATE = 0.25

#: Upper clamp on the CAPM cost of equity. A very high re-levered beta (deep
#: leverage) could otherwise produce an implausibly high rate; 25% is already
#: at the extreme end for an equity discount rate.
_COST_OF_EQUITY_MAX = 0.25


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def relever_beta(unlevered_beta: float, de_ratio: float, tax_rate: float) -> float:
    """Re-lever a sector asset (unlevered) beta to the firm's leverage (Hamada).

    ``β_levered = β_unlevered × (1 + (1 − tax) × D/E)``. With ``de_ratio == 0``
    (no debt, or leverage unknown) this returns ``unlevered_beta`` unchanged.
    """
    return unlevered_beta * (1.0 + (1.0 - tax_rate) * de_ratio)


def capm_rate(risk_free_pct: float, levered_beta: float, erp_pct: float) -> float:
    """CAPM cost of equity as a DECIMAL FRACTION.

    ``risk_free_pct`` and ``erp_pct`` are PERCENTAGE numbers (matching the
    ``erp.csv`` convention, e.g. ``4.2`` for 4.2%); the ``/100`` converts the
    percentage result to the decimal fraction the engine expects.
    """
    return (risk_free_pct + levered_beta * erp_pct) / 100.0


def compute_cost_of_equity(
    sector_data: Optional[dict],
    sic_description: Optional[str],
    metrics: Optional[dict],
    tax_rate: float = _DEFAULT_TAX_RATE,
    is_unprofitable: bool = False,
) -> Optional[dict]:
    """Estimate a firm-specific CAPM cost of equity, or ``None`` if not possible.

    Args:
        sector_data: The dict from
            :func:`sec_analyzer.valuation.damodaran.load_sector_data` (carries
            ``erp``/``risk_free`` and the per-industry ``beta``), or ``None``.
        sic_description: The filer's ``sicDescription`` (SEC submissions), used
            to match a Damodaran industry row for its unlevered beta.
        metrics: The dict from
            :func:`sec_analyzer.normalize.metrics.compute_metrics`; its
            ``total_debt`` and ``market_cap`` give the market D/E used to
            re-lever. Missing leverage degrades to ``D/E = 0`` (asset beta used
            as-is), not to ``None``.
        tax_rate: Marginal tax rate for re-levering. Defaults to
            :data:`_DEFAULT_TAX_RATE`.
        is_unprofitable: Raises the lower clamp from 7% to 10%, matching
            :func:`sec_analyzer.valuation.sanity.clamp_assumptions`.

    Returns:
        ``None`` if sector data, the sector beta, ERP, or the risk-free rate
        is unavailable. Otherwise::

            {
              "rate": 0.106,            # decimal fraction, clamped
              "unlevered_beta": 1.50,
              "levered_beta": 1.51,
              "de_ratio": 0.004,
              "erp": 4.23,              # percent
              "risk_free": 4.20,        # percent
              "tax_rate": 0.25,
              "industry": "Semiconductor",
              "clamped": False,         # True if the floor/cap bound the raw rate
              "detail": "CAPM: rf %4.2 + βL 1.51 × ERP %4.23 = %10.6",
            }

        Never raises.
    """
    try:
        return _compute_cost_of_equity(
            sector_data, sic_description, metrics or {}, tax_rate, is_unprofitable
        )
    except Exception:  # noqa: BLE001 - cost-of-equity estimation must never raise
        logger.exception("compute_cost_of_equity() failed unexpectedly; returning None.")
        return None


def _compute_cost_of_equity(
    sector_data: Optional[dict],
    sic_description: Optional[str],
    metrics: dict,
    tax_rate: float,
    is_unprofitable: bool,
) -> Optional[dict]:
    if not sector_data:
        return None

    risk_free = sector_data.get("risk_free")
    erp = sector_data.get("erp")
    if not _is_number(risk_free) or not _is_number(erp):
        return None

    matched = sector_medians(sector_data, sic_description)
    unlevered_beta = (matched or {}).get("beta")
    if not _is_number(unlevered_beta):
        return None
    industry = (matched or {}).get("industry")

    # Market debt/equity from the filer's own balance sheet + market cap. If
    # either piece is missing or non-positive, fall back to D/E = 0 (use the
    # sector asset beta as-is) rather than giving up on CAPM entirely.
    total_debt = metrics.get("total_debt")
    equity = metrics.get("market_cap")
    if _is_number(total_debt) and total_debt >= 0 and _is_number(equity) and equity > 0:
        de_ratio = total_debt / equity
    else:
        de_ratio = 0.0

    levered_beta = relever_beta(unlevered_beta, de_ratio, tax_rate)
    raw_rate = capm_rate(risk_free, levered_beta, erp)

    floor = (
        sanity._DISCOUNT_RATE_MIN_UNPROFITABLE
        if is_unprofitable
        else sanity._DISCOUNT_RATE_MIN
    )
    rate = min(max(raw_rate, floor), _COST_OF_EQUITY_MAX)
    clamped = rate != raw_rate

    detail = (
        f"CAPM: rf %{risk_free:.1f} + βL {levered_beta:.2f} × ERP %{erp:.2f} "
        f"= %{rate * 100:.1f}"
    )
    if clamped:
        detail += f" (%{raw_rate * 100:.1f} sınıra çekildi)"

    return {
        "rate": rate,
        "unlevered_beta": unlevered_beta,
        "levered_beta": levered_beta,
        "de_ratio": de_ratio,
        "erp": erp,
        "risk_free": risk_free,
        "tax_rate": tax_rate,
        "industry": industry,
        "clamped": clamped,
        "detail": detail,
    }
