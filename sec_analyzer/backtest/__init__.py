"""Backtest / point-in-time evaluation subsystem for sec_analyzer.

Design principle (see ROADMAP.md "Backtest — tasarım ilkesi"): backtesting here
is an EVALUATION tool, not an OPTIMIZATION tool. Engine parameters are never
tuned to improve backtest hit-rates; corrections are made only on
conceptual/accounting grounds, and the backtest is used afterward to check
whether a correction improved coherence. Return optimization (Sharpe, equity
curves, parameter sweeps) is explicitly out of scope -- the sample is small
(~20-30 names), survivorship-biased (delisted firms are absent from free data),
and single-regime, so such results would overfit noise.

Every backtest output carries :data:`BACKTEST_DISCLAIMER` automatically.
"""

#: Auto-appended to every backtest output (terminal, HTML, snapshots).
BACKTEST_DISCLAIMER = (
    "Küçük ve hayatta-kalan yanlılığı olan örneklem; parametre seçim aracı değildir."
)

__all__ = ["BACKTEST_DISCLAIMER"]
