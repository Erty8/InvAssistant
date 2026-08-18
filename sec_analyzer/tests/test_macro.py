"""Unit tests for ``signals.macro`` (FRED panel -> valuation macro context).

Pure unit tests: no network, no disk I/O. Panels are hand-built dicts shaped
exactly like :func:`sec_analyzer.fetch.fred.get_macro_panel`'s return value
(``{series_id: observation_dict_or_None}``), constructed via the ``_obs``
helper below so each test only has to state the numbers it cares about.
"""

import re

import pytest

from sec_analyzer.fetch import fred
from sec_analyzer.signals import macro


def _obs(value, date_str="2026-08-06", percentile=None, provider="FRED"):
    """Build a minimal fake observation dict for a hand-built panel."""
    return {
        "value_pct": value,
        "date": date_str,
        "series": "X",
        "source": "FRED X" if provider == "FRED" else "Treasury X",
        "percentile": percentile,
        "provider": provider,
    }


# ---------------------------------------------------------------------------
# _fmt_pct / _fmt_percentile_int
# ---------------------------------------------------------------------------


def test_fmt_pct_none_returns_dash():
    assert macro._fmt_pct(None) == "—"


def test_fmt_pct_drops_trailing_zeros_and_uses_comma():
    assert macro._fmt_pct(4.21) == "4,21"
    assert macro._fmt_pct(3.90) == "3,9"
    assert macro._fmt_pct(-0.35) == "-0,35"
    assert macro._fmt_pct(0.0) == "0"


def test_fmt_percentile_int_rounds_and_passes_none_through():
    assert macro._fmt_percentile_int(None) is None
    assert macro._fmt_percentile_int(62.5) == "62"
    assert macro._fmt_percentile_int(18.5) == "18"


# ---------------------------------------------------------------------------
# _credit_regime
# ---------------------------------------------------------------------------


def test_credit_regime_boundaries_on_hy_percentile():
    assert macro._credit_regime(75.0, None) == "sıkı"
    assert macro._credit_regime(74.9, None) == "normal"
    assert macro._credit_regime(25.0, None) == "gevşek"
    assert macro._credit_regime(25.1, None) == "normal"
    assert macro._credit_regime(50.0, None) == "normal"


def test_credit_regime_falls_back_to_ig_percentile_when_hy_missing():
    assert macro._credit_regime(None, 80.0) == "sıkı"
    assert macro._credit_regime(None, 10.0) == "gevşek"
    assert macro._credit_regime(None, 50.0) == "normal"


def test_credit_regime_none_when_both_missing():
    assert macro._credit_regime(None, None) is None


def test_credit_regime_prefers_hy_over_ig_when_both_present():
    # HY says tight, IG says loose -- HY wins (more sensitive gauge).
    assert macro._credit_regime(90.0, 5.0) == "sıkı"


# ---------------------------------------------------------------------------
# _build_notes -- each rule fires on its own minimal fixture, not otherwise.
# ---------------------------------------------------------------------------


def test_build_notes_rule1_curve_inverted_fires():
    ctx = {"curve_inverted": True, "curve_slope_pct": -0.35}
    notes = macro._build_notes(ctx)
    assert any("ters" in n and "-0,35 puan" in n for n in notes)


def test_build_notes_rule1_does_not_fire_when_not_inverted():
    ctx = {"curve_inverted": False, "curve_slope_pct": 0.35}
    assert not any("ters" in n for n in macro._build_notes(ctx))


def test_build_notes_rule1_does_not_fire_when_slope_missing():
    ctx = {"curve_inverted": None, "curve_slope_pct": None}
    assert macro._build_notes(ctx) == []


def test_build_notes_rule2_credit_tight_fires():
    ctx = {"credit_regime": "sıkı", "hy_spread_pct": 5.0, "hy_spread_percentile": 90.0}
    notes = macro._build_notes(ctx)
    assert any("sıkı" in n and "5" in n for n in notes)


def test_build_notes_rule3_credit_loose_fires():
    ctx = {"credit_regime": "gevşek", "hy_spread_pct": 2.0, "hy_spread_percentile": 10.0}
    notes = macro._build_notes(ctx)
    assert any("gevşek" in n and "ucuza" in n for n in notes)


def test_build_notes_rule2_and_3_do_not_fire_when_regime_normal():
    ctx = {"credit_regime": "normal", "hy_spread_pct": 3.0, "hy_spread_percentile": 50.0}
    assert not any("Kredi marjı" in n for n in macro._build_notes(ctx))


def test_build_notes_rule2_and_3_do_not_fire_when_regime_missing():
    ctx = {"credit_regime": None}
    assert not any("Kredi marjı" in n for n in macro._build_notes(ctx))


def test_build_notes_rule4_exceeds_inflation_fires():
    ctx = {
        "terminal_growth_pct": 3.9,
        "breakeven_inflation_pct": 2.3,
        "terminal_vs_inflation_pct": 1.6,
    }
    notes = macro._build_notes(ctx)
    assert any("belirgin üzerinde" in n and "%3,9" in n and "%2,3" in n for n in notes)


def test_build_notes_rule4_below_inflation_fires():
    ctx = {
        "terminal_growth_pct": 1.8,
        "breakeven_inflation_pct": 2.3,
        "terminal_vs_inflation_pct": -0.5,
    }
    notes = macro._build_notes(ctx)
    assert any("altında" in n and "%1,8" in n and "%2,3" in n for n in notes)


@pytest.mark.parametrize("diff,tg", [(0.0, 2.3), (0.5, 2.8), (1.0, 3.3)])
def test_build_notes_rule4_deadband_neither_direction_fires(diff, tg):
    ctx = {
        "terminal_growth_pct": tg,
        "breakeven_inflation_pct": 2.3,
        "terminal_vs_inflation_pct": diff,
    }
    assert not any("Terminal büyüme" in n for n in macro._build_notes(ctx))


def test_build_notes_rule4_absent_when_terminal_growth_not_given():
    ctx = {"terminal_growth_pct": None, "breakeven_inflation_pct": 2.3, "terminal_vs_inflation_pct": None}
    assert not any("Terminal büyüme" in n for n in macro._build_notes(ctx))


def test_build_notes_rule5_extreme_high_fires():
    ctx = {"risk_free_percentile": 85.0, "risk_free_pct": 4.5}
    notes = macro._build_notes(ctx)
    assert any("10 yıllık faiz" in n and "üst ucunda" in n for n in notes)


def test_build_notes_rule5_extreme_low_fires():
    ctx = {"risk_free_percentile": 15.0, "risk_free_pct": 3.0}
    notes = macro._build_notes(ctx)
    assert any("10 yıllık faiz" in n and "alt ucunda" in n for n in notes)


def test_build_notes_rule5_does_not_fire_in_the_middle():
    ctx = {"risk_free_percentile": 50.0, "risk_free_pct": 4.0}
    assert not any("10 yıllık faiz" in n for n in macro._build_notes(ctx))


def test_build_notes_rule5_boundary_exact_80_fires_79_does_not():
    ctx80 = {"risk_free_percentile": 80.0, "risk_free_pct": 4.0}
    ctx79 = {"risk_free_percentile": 79.0, "risk_free_pct": 4.0}
    assert any("10 yıllık faiz" in n for n in macro._build_notes(ctx80))
    assert not any("10 yıllık faiz" in n for n in macro._build_notes(ctx79))


def test_build_notes_rule5_boundary_exact_20_fires_21_does_not():
    ctx20 = {"risk_free_percentile": 20.0, "risk_free_pct": 4.0}
    ctx21 = {"risk_free_percentile": 21.0, "risk_free_pct": 4.0}
    assert any("10 yıllık faiz" in n for n in macro._build_notes(ctx20))
    assert not any("10 yıllık faiz" in n for n in macro._build_notes(ctx21))


def test_build_notes_empty_ctx_returns_empty_list():
    assert macro._build_notes({}) == []


# ---------------------------------------------------------------------------
# build_macro_context -- field arithmetic on a hand-built full panel.
# ---------------------------------------------------------------------------


def test_build_macro_context_full_panel_arithmetic():
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5),
        fred.SERIES_DGS2: _obs(3.86, "2026-08-06"),
        fred.SERIES_DGS30: _obs(4.69, "2026-08-05"),
        fred.SERIES_T10YIE: _obs(2.31, "2026-08-06"),
        fred.SERIES_BAA10Y: _obs(1.62, "2026-08-06", 24.0),
        fred.SERIES_HY_OAS: _obs(3.18, "2026-08-06", 18.5),
    }
    ctx = macro.build_macro_context(panel, terminal_growth=0.039)

    assert ctx is not None
    assert ctx["risk_free_pct"] == pytest.approx(4.21)
    assert ctx["risk_free_date"] == "2026-08-06"
    assert ctx["risk_free_percentile"] == pytest.approx(62.5)
    assert ctx["curve_slope_pct"] == pytest.approx(0.35)
    assert ctx["curve_inverted"] is False
    assert ctx["long_slope_pct"] == pytest.approx(0.48)
    assert ctx["breakeven_inflation_pct"] == pytest.approx(2.31)
    assert ctx["real_rate_pct"] == pytest.approx(1.90)
    assert ctx["ig_spread_pct"] == pytest.approx(1.62)
    assert ctx["ig_spread_percentile"] == pytest.approx(24.0)
    assert ctx["hy_spread_pct"] == pytest.approx(3.18)
    assert ctx["hy_spread_percentile"] == pytest.approx(18.5)
    assert ctx["credit_regime"] == "gevşek"
    assert ctx["terminal_growth_pct"] == pytest.approx(3.9)
    assert ctx["terminal_vs_inflation_pct"] == pytest.approx(1.59)
    assert ctx["as_of"] == "2026-08-06"
    assert set(ctx["series_available"]) == set(panel.keys())
    assert ctx["series_missing"] == []


def test_curve_inverted_exactly_zero_is_not_inverted():
    panel = {fred.SERIES_DGS10: _obs(4.0), fred.SERIES_DGS2: _obs(4.0)}
    ctx = macro.build_macro_context(panel)
    assert ctx["curve_slope_pct"] == pytest.approx(0.0)
    assert ctx["curve_inverted"] is False


def test_curve_inverted_just_below_zero_is_inverted():
    panel = {fred.SERIES_DGS10: _obs(3.99), fred.SERIES_DGS2: _obs(4.0)}
    ctx = macro.build_macro_context(panel)
    assert ctx["curve_slope_pct"] == pytest.approx(-0.01)
    assert ctx["curve_inverted"] is True


def test_build_macro_context_partial_panel_only_dgs10():
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5),
        fred.SERIES_DGS2: None,
        fred.SERIES_DGS30: None,
        fred.SERIES_T10YIE: None,
        fred.SERIES_BAA10Y: None,
        fred.SERIES_HY_OAS: None,
    }
    ctx = macro.build_macro_context(panel)

    assert ctx is not None
    assert ctx["risk_free_pct"] == pytest.approx(4.21)
    assert ctx["curve_slope_pct"] is None
    assert ctx["curve_inverted"] is None
    assert ctx["long_slope_pct"] is None
    assert ctx["breakeven_inflation_pct"] is None
    assert ctx["real_rate_pct"] is None
    assert ctx["ig_spread_pct"] is None
    assert ctx["ig_spread_percentile"] is None
    assert ctx["hy_spread_pct"] is None
    assert ctx["hy_spread_percentile"] is None
    assert ctx["credit_regime"] is None
    assert ctx["terminal_growth_pct"] is None
    assert ctx["terminal_vs_inflation_pct"] is None
    assert ctx["series_available"] == [fred.SERIES_DGS10]
    assert set(ctx["series_missing"]) == {
        fred.SERIES_DGS2, fred.SERIES_DGS30, fred.SERIES_T10YIE,
        fred.SERIES_BAA10Y, fred.SERIES_HY_OAS,
    }


def test_build_macro_context_empty_dict_returns_none():
    assert macro.build_macro_context({}) is None


def test_build_macro_context_none_returns_none():
    assert macro.build_macro_context(None) is None


def test_build_macro_context_all_none_values_returns_none():
    panel = {s: None for s in fred.MACRO_PANEL}
    assert macro.build_macro_context(panel) is None


def test_build_macro_context_never_raises_on_malformed_panel():
    # A malformed entry (missing "value_pct") must not raise -- the outer
    # never-raising wrapper catches it and degrades to None.
    panel = {fred.SERIES_DGS10: {"date": "2026-08-06"}}  # missing value_pct
    result = macro.build_macro_context(panel)
    assert result is None


# ---------------------------------------------------------------------------
# summarize_macro
# ---------------------------------------------------------------------------


def test_summarize_macro_none_returns_dash():
    assert macro.summarize_macro(None) == "—"


def test_summarize_macro_empty_dict_returns_dash():
    assert macro.summarize_macro({}) == "—"


def test_summarize_macro_full_example_line():
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5),
        fred.SERIES_DGS2: _obs(3.86, "2026-08-06"),
        fred.SERIES_DGS30: _obs(4.69, "2026-08-05"),
        fred.SERIES_T10YIE: _obs(2.31, "2026-08-06"),
        fred.SERIES_BAA10Y: _obs(1.62, "2026-08-06", 24.0),
        fred.SERIES_HY_OAS: _obs(3.18, "2026-08-06", 18.5),
    }
    ctx = macro.build_macro_context(panel, terminal_growth=0.039)
    line = macro.summarize_macro(ctx)

    assert "10y %4,21" in line
    assert "62. yüzdelik" in line
    assert "eğri +0,35" in line
    assert "HY spread %3,18" in line
    assert "18. yüzdelik" in line
    assert "gevşek" in line


def test_summarize_macro_falls_back_to_ig_when_hy_missing():
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5),
        fred.SERIES_BAA10Y: _obs(1.62, "2026-08-06", 24.0),
    }
    ctx = macro.build_macro_context(panel)
    line = macro.summarize_macro(ctx)
    assert "IG spread %1,62" in line
    assert "HY spread" not in line


# ---------------------------------------------------------------------------
# No "None"/"nan"/period-decimal leakage anywhere in rendered text.
# ---------------------------------------------------------------------------


def test_no_rendered_string_contains_none_nan_or_dot_decimal():
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5),
        fred.SERIES_DGS2: _obs(3.86, "2026-08-06"),
        fred.SERIES_DGS30: None,
        fred.SERIES_T10YIE: _obs(1.8, "2026-08-06"),
        fred.SERIES_BAA10Y: None,
        fred.SERIES_HY_OAS: _obs(5.0, "2026-08-06", 90.0),
    }
    ctx = macro.build_macro_context(panel, terminal_growth=0.005)
    line = macro.summarize_macro(ctx)
    blob = line + " ".join(ctx["notes"])

    # Word-boundary matches -- a plain substring check on "nan" would false-
    # positive on ordinary Turkish words like "refinansman".
    assert not re.search(r"\bNone\b", blob)
    assert not re.search(r"\bnan\b", blob)
    assert not re.search(r"\d+\.\d+", blob), f"found a period-decimal number in: {blob!r}"
    # Sanity: this fixture should actually have produced some notes/output,
    # otherwise the assertions above would be vacuously true.
    assert line != "—"
    assert ctx["notes"]


# ---------------------------------------------------------------------------
# Provider tracking (FRED primary, Treasury fallback -- fetch/fred.py).
# ---------------------------------------------------------------------------


def test_providers_all_fred_is_the_normal_case():
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5, provider="FRED"),
        fred.SERIES_DGS2: _obs(3.86, "2026-08-06", provider="FRED"),
    }
    ctx = macro.build_macro_context(panel)
    assert ctx["providers"] == ["FRED"]


def test_providers_treasury_sourced_panel_with_both_credit_series_missing():
    """When FRED is unreachable, DGS10/DGS2/DGS30/T10YIE fall back to
    Treasury (faithful substitutes) but BAA10Y/BAMLH0A0HYM2 have none and
    stay missing. credit_regime must be None, no credit note must fire, and
    the rate-based notes (curve/risk-free) must still work normally off the
    Treasury-sourced values."""
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 85.0, provider="Treasury"),
        fred.SERIES_DGS2: _obs(4.56, "2026-08-06", provider="Treasury"),  # inverted curve
        fred.SERIES_DGS30: _obs(4.69, "2026-08-06", provider="Treasury"),
        fred.SERIES_T10YIE: _obs(2.31, "2026-08-06", provider="Treasury"),
        fred.SERIES_BAA10Y: None,
        fred.SERIES_HY_OAS: None,
    }
    ctx = macro.build_macro_context(panel, terminal_growth=0.04)

    assert ctx["providers"] == ["Treasury"]
    assert ctx["credit_regime"] is None
    assert ctx["ig_spread_pct"] is None
    assert ctx["hy_spread_pct"] is None
    assert not any("Kredi marjı" in n for n in ctx["notes"])
    # Rate-based notes still fire off the Treasury-sourced values: curve is
    # inverted (4.21 - 4.56 < 0) and risk-free percentile is extreme (85).
    assert ctx["curve_inverted"] is True
    assert any("ters" in n for n in ctx["notes"])
    assert any("10 yıllık faiz" in n for n in ctx["notes"])
    assert ctx["series_missing"] == [fred.SERIES_BAA10Y, fred.SERIES_HY_OAS]


def test_providers_mixed_fred_and_treasury():
    panel = {
        fred.SERIES_DGS10: _obs(4.21, "2026-08-06", provider="Treasury"),
        fred.SERIES_BAA10Y: _obs(1.62, "2026-08-06", 24.0, provider="FRED"),
    }
    ctx = macro.build_macro_context(panel)
    assert ctx["providers"] == ["FRED", "Treasury"]


def test_providers_empty_list_when_panel_lacks_provider_key():
    # Defensive: a caller passing observation dicts without "provider" (e.g.
    # an older test fixture) must not raise -- providers just comes back
    # empty rather than guessing.
    panel = {fred.SERIES_DGS10: {"value_pct": 4.21, "date": "2026-08-06"}}
    ctx = macro.build_macro_context(panel)
    assert ctx["providers"] == []


def test_summarize_macro_appends_treasury_note_only_when_treasury_used():
    fred_panel = {fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5, provider="FRED")}
    treasury_panel = {fred.SERIES_DGS10: _obs(4.21, "2026-08-06", 62.5, provider="Treasury")}

    fred_line = macro.summarize_macro(macro.build_macro_context(fred_panel))
    treasury_line = macro.summarize_macro(macro.build_macro_context(treasury_panel))

    assert "Treasury" not in fred_line
    assert "Treasury üzerinden" in treasury_line
    # Same numbers otherwise -- only the provenance suffix differs.
    assert treasury_line.startswith(fred_line)
