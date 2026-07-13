"""Deterministic valuation engine for sec_analyzer.

This package computes fair-value NUMBERS with plain, deterministic Python --
no LLM involved. The interpret layer's phase 1 only *proposes* assumption
ranges (growth/terminal-growth/discount-rate per scenario); this package
turns those assumptions into DCF/reverse-DCF/multiples/sector-relative
figures, and phase 2 only *comments* on the resulting numbers. Same inputs
always produce the same outputs.

See ``sec_analyzer/valuation/SPEC.md`` for the full binding contract (shapes,
formulas, rounding rules) that every module here follows exactly.

Public entry points:

* :func:`run_valuation` -- orchestrates DCF, reverse-DCF, multiples,
  sector-relative anchors (P/B x ROE, normalized-earnings DCF), sensitivity,
  and triangulation into the single ``valuation`` dict consumed by the
  interpret layer, CLI verdict card, HTML report, and store.
* :func:`validate_assumptions` -- sanity-checks a phase-1 (LLM or
  rule-based) assumption set before it's fed into :func:`run_valuation`.
"""

from sec_analyzer.valuation.engine import run_valuation
from sec_analyzer.valuation.sanity import validate_assumptions

__all__ = ["run_valuation", "validate_assumptions"]
