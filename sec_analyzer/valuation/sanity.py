"""Deterministic sanity checks over phase-1 assumption ranges.

The LLM (or the rule-based fallback) proposes growth/terminal-growth/
discount-rate ranges per scenario; this module never trusts them blindly.
Every rule either fires (appending one Turkish violation string) or
doesn't -- an invalid assumption is never silently "fixed" (e.g. a discount
rate at or below the terminal growth rate, which would make the DCF's
Gordon-growth terminal value mathematically undefined), it's just reported
so the caller can re-prompt the LLM or fall back to a deterministic default.
"""

import copy
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Upper bound on terminal_growth (4%) -- above this is implausible for a
#: perpetuity growth rate (roughly nominal long-run GDP growth or higher).
_TERMINAL_GROWTH_MAX = 0.04

#: Lower bound on discount_rate for a profitable company (7%) and for an
#: unprofitable one, which should carry a higher risk premium (10%).
_DISCOUNT_RATE_MIN = 0.07
_DISCOUNT_RATE_MIN_UNPROFITABLE = 0.10

#: Hard upper bound on growth_5y (40%) -- above this is implausible even
#: granting that the model always fades growth after year 5. Values above
#: 20% but at/below this are allowed by design (see the module docstring in
#: SPEC.md Sec.3).
_GROWTH_5Y_HARD_MAX = 0.40

_REQUIRED_FIELDS = ("growth_5y", "terminal_growth", "discount_rate")
_SCENARIO_LABELS = {"bear": "Bear", "base": "Base", "bull": "Bull"}


def _is_number(value) -> bool:
    """True for int/float, explicitly excluding bool (a bool is technically
    an int subclass in Python but is never a valid rate here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_assumptions(assumptions: Dict[str, dict], is_unprofitable: bool = False) -> List[str]:
    """Validate a phase-1 bear/base/bull assumption set.

    Args:
        assumptions: ``{"bear": {"growth_5y", "terminal_growth",
            "discount_rate", "story"}, "base": {...}, "bull": {...}}`` (see
            ``sec_analyzer/valuation/SPEC.md`` Sec.2). Rates are decimal
            fractions (0.08 = 8%).
        is_unprofitable: Whether the company is currently unprofitable --
            raises the minimum acceptable ``discount_rate`` from 7% to 10%.

    Returns:
        A list of human-readable Turkish violation strings, one per rule
        that fires per scenario. Empty list means every scenario passed
        every check. Rules (per scenario):

        * ``terminal_growth > 0.04`` -> violation.
        * ``discount_rate < 0.07`` (``< 0.10`` if ``is_unprofitable``) ->
          violation.
        * ``discount_rate <= terminal_growth`` -> violation (Gordon-growth
          terminal value undefined).
        * ``growth_5y > 0.40`` -> violation (``> 0.20`` is allowed by
          design).
        * A missing or non-numeric required field -> violation naming the
          field (and no further numeric comparisons for that scenario,
          since they'd be meaningless).

        Never raises.
    """
    try:
        return _validate_assumptions(assumptions or {}, is_unprofitable)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("validate_assumptions() failed unexpectedly.")
        return ["Varsayımlar doğrulanırken beklenmeyen bir hata oluştu."]


def _validate_assumptions(assumptions: Dict[str, dict], is_unprofitable: bool) -> List[str]:
    violations: List[str] = []
    min_discount_rate = _DISCOUNT_RATE_MIN_UNPROFITABLE if is_unprofitable else _DISCOUNT_RATE_MIN

    for scenario_key, label in _SCENARIO_LABELS.items():
        scenario = assumptions.get(scenario_key)
        if not isinstance(scenario, dict):
            violations.append(f"{label}: senaryo verisi eksik veya geçersiz.")
            continue

        bad_fields = [f for f in _REQUIRED_FIELDS if not _is_number(scenario.get(f))]
        for field in bad_fields:
            violations.append(f"{label}: '{field}' alanı eksik veya sayısal değil.")
        if bad_fields:
            # Can't safely compare fields that are missing/non-numeric.
            continue

        growth_5y = scenario["growth_5y"]
        terminal_growth = scenario["terminal_growth"]
        discount_rate = scenario["discount_rate"]

        if terminal_growth > _TERMINAL_GROWTH_MAX:
            violations.append(
                f"{label}: uçtaki büyüme (terminal_growth) %{terminal_growth * 100:.1f}, "
                f"üst sınır %{_TERMINAL_GROWTH_MAX * 100:.0f}'i aşıyor."
            )

        if discount_rate < min_discount_rate:
            unprofitable_note = " (zarar eden şirket için)" if is_unprofitable else ""
            violations.append(
                f"{label}: iskonto oranı %{discount_rate * 100:.1f}, "
                f"alt sınır %{min_discount_rate * 100:.0f}'in altında{unprofitable_note}."
            )

        if discount_rate <= terminal_growth:
            violations.append(
                f"{label}: iskonto oranı (%{discount_rate * 100:.1f}) uçtaki büyümeye "
                f"(%{terminal_growth * 100:.1f}) eşit veya ondan düşük -- Gordon büyüme formülü tanımsız."
            )

        if growth_5y > _GROWTH_5Y_HARD_MAX:
            violations.append(
                f"{label}: 5 yıllık büyüme %{growth_5y * 100:.1f}, gerçekçi olmayan bir şekilde "
                f"%{_GROWTH_5Y_HARD_MAX * 100:.0f}'i aşıyor."
            )

    return violations


def clamp_assumptions(
    assumptions: Dict[str, dict], is_unprofitable: bool = False
) -> Tuple[Dict[str, dict], List[str]]:
    """Clamp phase-1 assumptions into a sane range and report every clamp.

    Unlike :func:`validate_assumptions` (which only reports violations),
    this function actually rewrites out-of-range values so every downstream
    calculation (DCF, reverse-DCF, sensitivity, hyper-grower) works from the
    same clamped numbers that are shown to the user -- what's shown is what
    gets used. Each scenario is clamped independently, reusing this
    module's own violation thresholds:

    * ``terminal_growth`` is capped at :data:`_TERMINAL_GROWTH_MAX` (4%).
    * ``growth_5y`` is capped at :data:`_GROWTH_5Y_HARD_MAX` (40%).
    * ``discount_rate`` is floored at :data:`_DISCOUNT_RATE_MIN` (7%), or
      :data:`_DISCOUNT_RATE_MIN_UNPROFITABLE` (10%) if ``is_unprofitable``.

    The ``discount_rate <= terminal_growth`` case is deliberately NOT
    clamped here -- the Gordon-growth terminal value is mathematically
    undefined in that case, and the existing per-scenario ``ValueError``
    path in ``dcf.py``/``revenue_dcf.py`` (surfaced as a per-scenario note
    by ``engine.py``) already handles it; silently nudging one of the two
    rates to "fix" it would hide a real modeling conflict instead of
    reporting it.

    A missing/non-numeric field for a scenario is left untouched (there is
    nothing sane to clamp it to); that field is simply skipped for that
    scenario.

    Also checks, across scenarios (not per-scenario), whether
    ``bear.growth_5y <= base.growth_5y <= bull.growth_5y`` holds. A
    violation only produces a note -- it is never itself clamped, since
    there is no single "correct" reordering to apply.

    Args:
        assumptions: The phase-1 bear/base/bull assumption dict (see
            :func:`validate_assumptions`).
        is_unprofitable: Whether the company is currently unprofitable --
            raises the discount-rate floor from 7% to 10%, matching
            :func:`validate_assumptions`. Defaults to ``False``.

    Returns:
        A ``(clamped, notes)`` tuple. ``clamped`` is a deep copy of
        ``assumptions`` with any out-of-range numeric field rewritten in
        place (the input dict itself is never mutated); ``notes`` is a
        list of Turkish strings, one per clamp that actually fired (plus,
        at most, one for the bear/base/bull growth-ordering check). Never
        raises.
    """
    try:
        return _clamp_assumptions(assumptions or {}, is_unprofitable)
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("clamp_assumptions() failed unexpectedly; returning the input unchanged.")
        return copy.deepcopy(assumptions or {}), []


def _clamp_assumptions(
    assumptions: Dict[str, dict], is_unprofitable: bool
) -> Tuple[Dict[str, dict], List[str]]:
    notes: List[str] = []
    min_discount_rate = _DISCOUNT_RATE_MIN_UNPROFITABLE if is_unprofitable else _DISCOUNT_RATE_MIN
    clamped: Dict[str, dict] = copy.deepcopy(assumptions)

    for scenario_key, label in _SCENARIO_LABELS.items():
        scenario = clamped.get(scenario_key)
        if not isinstance(scenario, dict):
            continue

        terminal_growth = scenario.get("terminal_growth")
        if _is_number(terminal_growth) and terminal_growth > _TERMINAL_GROWTH_MAX:
            scenario["terminal_growth"] = _TERMINAL_GROWTH_MAX
            notes.append(
                f"{label}: uçtaki büyüme (terminal_growth) %{terminal_growth * 100:.1f} idi, "
                f"%{_TERMINAL_GROWTH_MAX * 100:.0f} ile sınırlandırıldı."
            )

        growth_5y = scenario.get("growth_5y")
        if _is_number(growth_5y) and growth_5y > _GROWTH_5Y_HARD_MAX:
            scenario["growth_5y"] = _GROWTH_5Y_HARD_MAX
            notes.append(
                f"{label}: 5 yıllık büyüme %{growth_5y * 100:.1f} idi, "
                f"%{_GROWTH_5Y_HARD_MAX * 100:.0f} ile sınırlandırıldı."
            )

        discount_rate = scenario.get("discount_rate")
        if _is_number(discount_rate) and discount_rate < min_discount_rate:
            scenario["discount_rate"] = min_discount_rate
            unprofitable_note = " (zarar eden şirket için)" if is_unprofitable else ""
            notes.append(
                f"{label}: iskonto oranı %{discount_rate * 100:.1f} idi, "
                f"%{min_discount_rate * 100:.0f} tabanına yükseltildi{unprofitable_note}."
            )

    bear_growth = (clamped.get("bear") or {}).get("growth_5y")
    base_growth = (clamped.get("base") or {}).get("growth_5y")
    bull_growth = (clamped.get("bull") or {}).get("growth_5y")
    if all(_is_number(v) for v in (bear_growth, base_growth, bull_growth)):
        if not (bear_growth <= base_growth <= bull_growth):
            notes.append(
                "Senaryo büyüme sıralaması beklenmedik (kötümser <= temel <= iyimser olmalı): "
                f"bear=%{bear_growth * 100:.1f}, base=%{base_growth * 100:.1f}, bull=%{bull_growth * 100:.1f}."
            )

    return clamped, notes
