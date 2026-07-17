"""Tests for WP6 -- the sector-derived (Damodaran Cap Ex/Sales) maintenance-
CapEx floor, added as the ``sector_capex_sales`` parameter to
``valuation.engine._maintenance_adjusted_margin``/``_build_hyper_growth`` and
as the ``capex_sales`` column/key threaded through
``valuation.damodaran._parse_multiples_rows``/``_row_to_result``/
``sector_medians``.

Background (see the docstrings of ``_maintenance_adjusted_margin`` in
``sec_analyzer/valuation/engine.py``): the maintenance-CapEx proxy used to
relieve growth CapEx from a capex-heavy hyper-grower's starting margin was
floored at a flat 5% of revenue (``_MAINTENANCE_CAPEX_MIN_PCT_REVENUE``).
WP6 lets that floor's percent-of-revenue term instead use the matched
Damodaran sector's own Cap Ex/Sales ratio (``sector_capex_sales``) when it is
a usable positive number -- a data-center/telecom/utility sector with a
genuinely higher maintenance-capex intensity than 5% no longer has its
growth CapEx (and thus its relieved margin) mis-sized by the generic
default. ``None`` (the parameter's default) keeps the pre-WP6 flat-5%
behavior exactly as before.

Part A (engine-level) mirrors the fixture/style of
``test_valuation_capex_normalization.py``'s ``_annual`` helper. Part B
(damodaran-level) mirrors ``test_damodaran.py``'s CSV-fixture style for
``load_sector_data``/``sector_medians``.
"""

import pytest

from sec_analyzer.valuation.damodaran import load_sector_data, sector_medians
from sec_analyzer.valuation.engine import _maintenance_adjusted_margin

# ---------------------------------------------------------------------------
# Part A -- _maintenance_adjusted_margin(sector_capex_sales=...) (engine)
# ---------------------------------------------------------------------------


def _annual(**concepts) -> dict:
    """Minimal ``normalized``-shaped dict, matching
    ``test_valuation_capex_normalization.py``'s helper of the same name:
    ``_annual(Revenue={2023: 1000.0})`` ->
    ``{"annual": {"Revenue": [{"fy": 2023, "value": 1000.0}]}}``."""
    return {
        "annual": {
            concept: [{"fy": fy, "value": value} for fy, value in by_fy.items()]
            for concept, by_fy in concepts.items()
        }
    }


def test_sector_capex_floor_higher_than_dep_and_default_applies_and_notes():
    # revenue=1000, capex=500, dep=100 -> capex_intensity=500/1000=0.5>0.30
    # (gate 1 OK). sector_capex_sales=0.20 (20% of revenue) is passed as the
    # 4th arg -> maintenance_floor_pct=0.20 (replacing the flat 5% default)
    # -> maintenance_capex = max(dep=100, 0.20*1000=200) = 200. This sector
    # floor (200) is HIGHER than both D&A (100) and the flat-5%-of-revenue
    # default (0.05*1000=50) would have been.
    # gate 2: capex(500) > maintenance_capex(200) -> OK, relief applies.
    # growth_capex = capex - maintenance_capex = 500 - 200 = 300.
    # ops_margin = raw_current_margin + growth_capex/revenue
    #            = -0.60 + 300/1000 = -0.60 + 0.30 = -0.30.
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 500.0}, Depreciation={2023: 100.0})
    metrics = {"latest_fy": 2023}

    ops_margin, capex_normalization = _maintenance_adjusted_margin(
        normalized, metrics, -0.60, sector_capex_sales=0.20
    )

    assert ops_margin == pytest.approx(-0.30)
    assert capex_normalization is not None
    assert capex_normalization["applied"] is True
    assert capex_normalization["capex_intensity"] == pytest.approx(0.5)
    # The sector floor (200) beat both D&A (100) and the flat 5% default (50).
    assert capex_normalization["maintenance_capex"] == pytest.approx(200.0)
    assert capex_normalization["maintenance_capex"] > 100.0  # > D&A
    assert capex_normalization["maintenance_capex"] > 0.05 * 1000.0  # > flat 5% default
    assert capex_normalization["growth_capex"] == pytest.approx(300.0)
    assert capex_normalization["raw_current_margin"] == pytest.approx(-0.60)
    assert capex_normalization["ops_current_margin"] == pytest.approx(-0.30)

    # The Turkish note fires under maintenance_capex_floor_note, naming the
    # sector percentage (20.0%) and the Damodaran source, since the sector
    # floor (not the 5% default) actually determined maintenance_capex.
    note = capex_normalization["maintenance_capex_floor_note"]
    assert "%20.0" in note
    assert "Damodaran Cap Ex/Sales" in note
    assert "varsayılan %5" in note


def test_sector_capex_sales_none_matches_omitted_default_byte_for_byte():
    # Same capex-heavy fixture as test_valuation_capex_normalization.py's
    # test_capex_heavy_applies_relief_hand_verified (revenue=1000, capex=500,
    # dep=100, raw_current_margin=-0.60) -- relief applies via the flat 5%
    # default in both calls below, since neither passes a usable sector value.
    normalized = _annual(Revenue={2023: 1000.0}, CapEx={2023: 500.0}, Depreciation={2023: 100.0})
    metrics = {"latest_fy": 2023}

    result_omitted = _maintenance_adjusted_margin(normalized, metrics, -0.60)
    result_explicit_none = _maintenance_adjusted_margin(normalized, metrics, -0.60, sector_capex_sales=None)

    assert result_omitted == result_explicit_none

    ops_margin, capex_normalization = result_omitted
    assert ops_margin == pytest.approx(-0.20)  # pre-existing flat-5%-floor behavior, unchanged
    assert capex_normalization is not None
    assert capex_normalization["maintenance_capex"] == pytest.approx(100.0)  # D&A wins vs. 5%*1000=50
    # The 5%-default path never adds the sector-floor note.
    assert "maintenance_capex_floor_note" not in capex_normalization


# ---------------------------------------------------------------------------
# Part B -- damodaran.py: capex_sales parsing/surfacing (CSV -> sector_medians)
# ---------------------------------------------------------------------------


def test_load_sector_data_and_sector_medians_surface_capex_sales_when_present(tmp_path):
    (tmp_path / "multiples.csv").write_text(
        "industry,pe,ps,pfcf,capex_sales\n"
        "Semiconductor,28.4,6.1,24.7,0.045\n",
        encoding="utf-8",
    )
    (tmp_path / "erp.csv").write_text(
        "region,erp\nUS,4.6\n",
        encoding="utf-8",
    )

    sector_data = load_sector_data(str(tmp_path))

    assert sector_data is not None
    assert sector_data["multiples"][0].get("capex_sales") == pytest.approx(0.045)

    result = sector_medians(sector_data, "SEMICONDUCTORS & RELATED DEVICES")
    assert result is not None
    assert result["industry"] == "Semiconductor"
    assert result.get("capex_sales") == pytest.approx(0.045)


def test_sector_medians_capex_sales_is_none_via_get_when_column_absent(tmp_path):
    # No capex_sales column in the CSV at all -- the KEY itself is absent
    # from the parsed row / result dict (per damodaran.py's docstrings), so
    # callers must read it with .get(), which degrades to None, rather than
    # assuming the key is always present.
    (tmp_path / "multiples.csv").write_text(
        "industry,pe,ps,pfcf\n"
        "Semiconductor,28.4,6.1,24.7\n",
        encoding="utf-8",
    )
    (tmp_path / "erp.csv").write_text(
        "region,erp\nUS,4.6\n",
        encoding="utf-8",
    )

    sector_data = load_sector_data(str(tmp_path))
    assert sector_data is not None
    assert "capex_sales" not in sector_data["multiples"][0]
    assert sector_data["multiples"][0].get("capex_sales") is None

    result = sector_medians(sector_data, "SEMICONDUCTORS & RELATED DEVICES")
    assert result is not None
    assert "capex_sales" not in result
    assert result.get("capex_sales") is None


def test_sector_medians_capex_sales_absent_when_row_value_blank(tmp_path):
    # capex_sales column present in the header but blank/unparseable for this
    # row -- _to_float returns None, so the key is still omitted (not
    # present-with-None), per _parse_multiples_rows's documented behavior.
    (tmp_path / "multiples.csv").write_text(
        "industry,pe,ps,pfcf,capex_sales\n"
        "Semiconductor,28.4,6.1,24.7,\n",
        encoding="utf-8",
    )
    (tmp_path / "erp.csv").write_text(
        "region,erp\nUS,4.6\n",
        encoding="utf-8",
    )

    sector_data = load_sector_data(str(tmp_path))
    assert sector_data is not None
    assert "capex_sales" not in sector_data["multiples"][0]
    assert sector_data["multiples"][0].get("capex_sales") is None
