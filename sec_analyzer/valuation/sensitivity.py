"""3x3 growth x discount-rate sensitivity grid around the base DCF scenario.

Shows how sensitive the base-scenario per-share DCF value is to small moves
in the two most consequential inputs: the 5-year growth rate and the
discount rate. Pure re-application of ``dcf.dcf_per_share`` at nine nearby
input combinations -- no new valuation logic.
"""

import logging
from typing import Dict, List, Optional

from sec_analyzer.valuation.dcf import dcf_per_share

logger = logging.getLogger(__name__)

#: Step sizes for the 3x3 grid: growth +/- 2pp, discount rate +/- 1pp.
_GROWTH_STEP = 0.02
_DISCOUNT_RATE_STEP = 0.01

#: A grid is flagged "high uncertainty" when (hi - lo) / |base_cell| exceeds
#: this fraction (60%).
_HIGH_UNCERTAINTY_THRESHOLD = 0.60


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sensitivity_matrix(
    base_assumptions: dict,
    fcf0: Optional[float],
    shares: Optional[float],
    dilution_rate: float = 0.0,
) -> Optional[dict]:
    """Build the 3x3 (growth x discount-rate) DCF sensitivity grid.

    Rows vary ``growth_5y`` over ``[g-0.02, g, g+0.02]``; columns vary
    ``discount_rate`` over ``[r-0.01, r, r+0.01]``, where ``g``/``r`` come
    from ``base_assumptions``. ``terminal_growth`` is held fixed at the
    base scenario's value in every cell.

    Args:
        base_assumptions: The base scenario's assumption dict (``growth_5y``,
            ``terminal_growth``, ``discount_rate``).
        fcf0: Base-year free cash flow (see ``dcf.dcf_per_share``).
        shares: Diluted shares outstanding.
        dilution_rate: Annual dilution rate (see ``dcf.dcf_per_share``).

    Returns:
        ``{"growth_values": [g-.02, g, g+.02], "dr_values": [r-.01, r,
        r+.01], "matrix": [[3x3 per-share floats or None]], "lo": min,
        "hi": max, "high_uncertainty": bool}`` where a cell is ``None`` if
        that combination has ``discount_rate <= terminal_growth`` (Gordon
        undefined) or otherwise can't be computed. ``high_uncertainty`` is
        ``True`` when ``(hi - lo) / abs(base_cell) > 0.60``. Returns
        ``None`` if ``fcf0``/``shares``/the base assumption fields aren't
        usable at all. Never raises.
    """
    try:
        return _sensitivity_matrix(base_assumptions or {}, fcf0, shares, dilution_rate)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("sensitivity_matrix() failed unexpectedly; returning None.")
        return None


def _sensitivity_matrix(
    base_assumptions: dict,
    fcf0: Optional[float],
    shares: Optional[float],
    dilution_rate: float,
) -> Optional[dict]:
    growth = base_assumptions.get("growth_5y")
    terminal_growth = base_assumptions.get("terminal_growth")
    discount_rate = base_assumptions.get("discount_rate")

    if fcf0 is None or not shares or shares <= 0:
        return None
    if not all(_is_number(v) for v in (growth, terminal_growth, discount_rate)):
        return None

    growth_values = [round(growth - _GROWTH_STEP, 4), round(growth, 4), round(growth + _GROWTH_STEP, 4)]
    dr_values = [
        round(discount_rate - _DISCOUNT_RATE_STEP, 4),
        round(discount_rate, 4),
        round(discount_rate + _DISCOUNT_RATE_STEP, 4),
    ]

    matrix: List[List[Optional[float]]] = []
    for g in growth_values:
        row: List[Optional[float]] = []
        for r in dr_values:
            if r <= terminal_growth:
                row.append(None)
                continue
            try:
                result = dcf_per_share(fcf0, g, terminal_growth, r, shares, dilution_rate)
                row.append(round(result["per_share"], 2))
            except ValueError:
                row.append(None)
        matrix.append(row)

    flat = [v for row in matrix for v in row if v is not None]
    if not flat:
        return {
            "growth_values": growth_values,
            "dr_values": dr_values,
            "matrix": matrix,
            "lo": None,
            "hi": None,
            "high_uncertainty": False,
        }

    lo = min(flat)
    hi = max(flat)
    base_cell = matrix[1][1]
    high_uncertainty = bool(
        base_cell is not None and base_cell != 0 and (hi - lo) / abs(base_cell) > _HIGH_UNCERTAINTY_THRESHOLD
    )

    return {
        "growth_values": growth_values,
        "dr_values": dr_values,
        "matrix": matrix,
        "lo": lo,
        "hi": hi,
        "high_uncertainty": high_uncertainty,
    }
