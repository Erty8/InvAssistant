"""Unit tests for sec_analyzer.report.generator.

``generate_report`` doesn't depend on any live data source -- it only needs
a ``result`` dict shaped like ``interpret()``'s output (see
``sec_analyzer.interpret.analyzer``), plus the optional metrics/technical/
flags side data. These fixtures build that shape directly, matching the
style of the other ``tests/test_*.py`` modules.
"""

from sec_analyzer.report.generator import generate_report

_SCENARIO = {
    "lo": 90.0,
    "hi": 110.0,
    "growth": "%8 büyüme",
    "discount_rate": "%12",
    "note": "İki aşamalı DCF, FCF/hisse çapası.",
}


def _success_result():
    return {
        "fair_value_range": {
            "bear": {**_SCENARIO, "lo": 70.0, "hi": 95.0},
            "base": {**_SCENARIO},
            "bull": {**_SCENARIO, "lo": 115.0, "hi": 140.0},
        },
        "fundamental_verdict": "PAHALI",
        "technical_verdict": "AŞIRI ALIM (RSI 74, SMA50 +%12)",
        "profile_fit": {
            "verdict": "KISMEN",
            "reason": "growth stiline uygun, sektör limiti dolmak üzere",
        },
        "cyclical_risk": "low",
        "horizon_note": "1y ufkunda dengeli değerlendirme.",
        "key_risks": ["Marj daralması"],
        "red_flags_comment": "yok",
        "catalyst": "Q2 earnings ~27 Ağu",
        "summary": "NVDA güçlü büyüme gösteriyor ancak fiyat baz aralığın üzerinde.",
        "_provider": "script",
        "_model": "rule-based-v2",
        "_horizon": "1y",
        "_weights": {"fundamental": 0.5, "technical": 0.5},
    }


def _error_result():
    return {
        "error": "ollama_unreachable",
        "summary": "Local Ollama analysis is unavailable.",
        "_provider": "ollama",
    }


def _valuation():
    """A ``result["valuation"]`` dict matching the shape documented in
    ``sec_analyzer/valuation/SPEC.md`` Sec.11 (engine output), Sec.9
    (sensitivity), and Sec.10 (triangulation)."""
    return {
        "sector_type": "mature",
        "fcf0": 12_500_000_000.0,
        "fcf0_source": "ttm",
        "dcf": {
            "enabled": True,
            "disabled_reason": None,
            "scenarios": {
                "bear": {"per_share": 82.0, "lo": 73.8, "hi": 90.2},
                "base": {"per_share": 100.0, "lo": 90.0, "hi": 110.0},
                "bull": {"per_share": 127.0, "lo": 114.3, "hi": 139.7},
            },
            "normalized_variant": None,
        },
        "pb_roe": None,
        "fair_value_range": {
            "bear": {**_SCENARIO, "lo": 73.8, "hi": 90.2},
            "base": {**_SCENARIO},
            "bull": {**_SCENARIO, "lo": 114.3, "hi": 139.7},
        },
        "reverse_dcf": {
            "implied_growth": 0.19,
            "realized_cagr_5y": 0.14,
            "realized_label": "5y",
        },
        "multiples": {
            "history": [],
            "current": {"pe": 45.2, "ps": 12.1, "pfcf": 38.0},
            "pe_percentile": 88.0,
            "ps_percentile": 91.0,
            "pfcf_percentile": 80.0,
            "history_years": 8,
            "sector": {"available": False, "industry": None, "pe_median": None, "ps_median": None, "pfcf_median": None},
        },
        "sensitivity": {
            "growth_values": [0.06, 0.08, 0.10],
            "dr_values": [0.11, 0.12, 0.13],
            "matrix": [
                [95.0, 87.0, 80.0],
                [108.0, 100.0, 92.0],
                [122.0, 112.0, 103.0],
            ],
            "lo": 80.0,
            "hi": 122.0,
            "high_uncertainty": True,
        },
        "triangulation": {
            "signals": {"dcf": "pahali", "reverse_dcf": "pahali", "multiples": "pahali"},
            "confidence": "YÜKSEK",
            "direction": "pahali",
        },
        "assumptions": {
            "bear": {"growth_5y": 0.04, "terminal_growth": 0.025, "discount_rate": 0.13, "story": "Muhafazakar senaryo."},
            "base": {"growth_5y": 0.08, "terminal_growth": 0.025, "discount_rate": 0.12, "story": "Baz senaryo."},
            "bull": {"growth_5y": 0.12, "terminal_growth": 0.025, "discount_rate": 0.11, "story": "Optimist senaryo."},
        },
        "notes": ["fcf0 = TTM FCF kullanıldı."],
    }


def _scenario_returns():
    """A `result["scenario_returns"]` dict matching planning.compute_scenario_returns's shape."""
    return {
        "bear": {"ret_lo_pct": -42.5, "ret_hi_pct": -29.7},
        "base": {"ret_lo_pct": -29.9, "ret_hi_pct": -14.3},
        "bull": {"ret_lo_pct": -11.0, "ret_hi_pct": 8.8},
    }


def _entry_plan():
    """A `result["entry_plan"]` list matching planning.compute_entry_plan's tranche shape."""
    return [
        {
            "n": 1,
            "trigger": "Günlük kapanış 114.60 USD seviyesinin altına inerse (bölge 112.89-116.32 USD).",
            "price_zone": {"lo": 112.89, "hi": 116.32},
            "size_pct": 40.0,
            "invalidation": 85.5,
            "target": 139.7,
            "rr": 2.8,
            "note": None,
        },
        {
            "n": 2,
            "trigger": "Günlük kapanış 90.00 USD seviyesinin altına inerse (bölge 88.65-91.35 USD).",
            "price_zone": {"lo": 88.65, "hi": 91.35},
            "size_pct": 60.0,
            "invalidation": 85.5,
            "target": 139.7,
            "rr": 6.1,
            "note": None,
        },
    ]


def _stop_adding():
    """A `result["stop_adding"]` list matching planning.compute_stop_adding's signal shape."""
    return [
        {
            "code": "HIGH_UNCERTAINTY",
            "message": "Duyarlılık matrisi yüksek belirsizlik gösteriyor (bant genişliği baz hücrenin %60'ından fazla); pozisyon büyütmede temkinli olunmalı.",
        }
    ]


def _thesis_metric():
    """A `result["thesis_metric"]` dict matching planning.select_thesis_metric's shape."""
    return {
        "name": "Net Kâr Marjı",
        "latest_value": "%12.0",
        "trend": "iyileşiyor",
        "rationale": (
            "Olgun sektörlerde tezin sağlığı kâr marjının istikrarında görülür; bu yüzden net kâr marjı "
            "(hesaplanamıyorsa ROE) tek çapa metrik olarak izlenir. METODOLOJI §7 kuralı: bu metrik iki "
            "ardışık çeyrek boyunca tezin aksini gösterirse tez geçersiz sayılır ve bu açıkça belirtilir."
        ),
    }


def _success_result_with_valuation():
    """A phase-2 result dict (SPEC.md Sec.12) -- same base shape as
    :func:`_success_result` plus ``valuation``/``confidence``/
    ``reverse_dcf_comment``, and the four code-computed planning fields
    (``scenario_returns``/``entry_plan``/``stop_adding``/``thesis_metric``)
    that ``analyzer._postprocess_phase2_result`` always injects alongside
    ``fair_value_range``/``confidence`` (see sec_analyzer/interpret/planning.py)."""
    result = _success_result()
    result["valuation"] = _valuation()
    result["confidence"] = "YÜKSEK"
    result["reverse_dcf_comment"] = (
        "Fiyat, 10 yıl boyunca %19 büyüme ima ediyor; gerçekleşen 5y CAGR %14 ile "
        "karşılaştırıldığında iddialı."
    )
    result["scenario_returns"] = _scenario_returns()
    result["entry_plan"] = _entry_plan()
    result["stop_adding"] = _stop_adding()
    result["thesis_metric"] = _thesis_metric()
    return result


def _metrics():
    return {"price": 128.40, "pe": 45.2, "latest_fy": 2025}


def _technical():
    return {
        "price": 128.40,
        "as_of": "2026-07-11",
        "rsi14": 74.0,
        "sma50": 114.6,
        "verdict": "AŞIRI ALIM",
        "verdict_detail": "RSI 74, SMA50 +%12",
        "horizon_summary": "RSI 74.0 seviyesinde.",
        "horizon": "1y",
        "support_levels": [{"low": 118.0, "high": 120.0, "price": 119.0, "dist_pct": -7.3, "touches": 2, "fib": None, "last_touch": "2026-05-01", "is_52w_high": False, "is_52w_low": False}],
        "resistance_levels": [{"low": 135.0, "high": 137.0, "price": 136.0, "dist_pct": 5.9, "touches": 1, "fib": "38.2%", "last_touch": "2026-06-01", "is_52w_high": False, "is_52w_low": False}],
        "price_series": [{"t": "2026-01-02", "c": 110.0}, {"t": "2026-03-02", "c": 120.0}, {"t": "2026-07-11", "c": 128.4}],
    }


def _flags():
    return [
        {
            "code": "SBC_HIGH",
            "message": "Hisse bazlı ödemeler gelire göre yüksek",
            "detail": "SBC gelirin %12.0'i.",
        }
    ]


def test_generate_report_success_writes_file_with_expected_content(tmp_path):
    path = generate_report(
        "nvda",
        "1y",
        _success_result(),
        metrics=_metrics(),
        technical=_technical(),
        flags=_flags(),
        price=128.40,
        as_of="2026-07-11",
        out_dir=str(tmp_path),
    )

    assert path.startswith(str(tmp_path))
    assert path.endswith(".html")

    content = open(path, "r", encoding="utf-8").read()

    # The ticker is upper-cased for both the filename and the embedded data.
    assert "NVDA" in content
    # The template placeholder must have been fully substituted.
    assert "__DATA_JSON__" not in content
    # A Turkish verdict string from the fixture should survive JSON encoding
    # unescaped (ensure_ascii=False) and appear verbatim in the file.
    assert "AŞIRI" in content
    # The report should still be a well-formed, self-contained HTML document.
    assert "<!DOCTYPE html>" in content
    assert "</html>" in content


def test_generate_report_embeds_events_from_result(tmp_path):
    """Recent 8-K events attached as ``result["events"]`` (by cmd_analyze) are
    embedded in the payload and rendered by the template's events card."""
    result = _success_result()
    result["events"] = [
        {"date": "2026-06-15", "severity": "critical", "items": ["4.02"],
         "categories": ["Önceki finansal tablolara güvenilemez (restatement)"]},
        {"date": "2026-07-01", "severity": "warning", "items": ["5.02"],
         "categories": ["Üst düzey yönetici/kurul değişikliği"]},
    ]
    path = generate_report(
        "NVDA", "1y", result,
        metrics=_metrics(), technical=_technical(), flags=_flags(),
        out_dir=str(tmp_path),
    )
    content = open(path, "r", encoding="utf-8").read()

    # Event data (inside result) survives into the embedded JSON payload.
    assert "güvenilemez" in content
    assert "4.02" in content
    # The events renderer and its card title are present in the template.
    assert "eventsCardHtml" in content
    assert "Son Dosyalama Olayları" in content


def test_generate_report_filename_includes_ticker_horizon_and_date(tmp_path):
    path = generate_report(
        "AAPL", "5y", _success_result(), out_dir=str(tmp_path)
    )
    filename = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    assert filename.startswith("AAPL_")
    assert filename.endswith("_5y.html")


def test_generate_report_creates_missing_out_dir(tmp_path):
    nested = tmp_path / "nested" / "reports"
    path = generate_report("MSFT", "3m", _success_result(), out_dir=str(nested))
    assert nested.exists()
    assert path.startswith(str(nested))


def test_generate_report_error_result_still_writes_a_file(tmp_path):
    path = generate_report(
        "OLD", "1y", _error_result(), out_dir=str(tmp_path)
    )
    content = open(path, "r", encoding="utf-8").read()

    assert "OLD" in content
    assert "__DATA_JSON__" not in content
    assert "ollama_unreachable" in content
    assert "<!DOCTYPE html>" in content


def test_generate_report_with_valuation_embeds_the_full_valuation_payload(tmp_path):
    """When ``result["valuation"]`` (SPEC.md Sec.11) is present, the report's
    embedded JSON payload -- and therefore the client-side triangulation/
    sensitivity/reverse-DCF sections that read from it -- carry the new
    fields through verbatim."""
    path = generate_report(
        "NVDA",
        "1y",
        _success_result_with_valuation(),
        metrics=_metrics(),
        technical=_technical(),
        flags=_flags(),
        price=128.40,
        as_of="2026-07-11",
        out_dir=str(tmp_path),
    )
    content = open(path, "r", encoding="utf-8").read()

    assert '"valuation"' in content
    assert '"triangulation"' in content
    assert '"sensitivity"' in content
    assert '"high_uncertainty":true' in content.replace(" ", "")
    assert '"reverse_dcf_comment"' in content
    assert '"confidence":"YÜKSEK"' in content.replace(" ", "")
    # The template's JS-side section builders for the valuation-only
    # sections must be present so they can actually render this data.
    assert "triangulationRowHtml" in content
    assert "sensitivityTableHtml" in content


def test_generate_report_without_valuation_omits_it_from_the_payload(tmp_path):
    """The classic (pre-valuation-engine) result shape -- no ``"valuation"``
    key at all -- must still produce a well-formed report; the template
    degrades gracefully (no triangulation/sensitivity data to show), never
    a crash."""
    path = generate_report(
        "AAPL",
        "5y",
        _success_result(),
        metrics=_metrics(),
        technical=_technical(),
        flags=_flags(),
        price=128.40,
        as_of="2026-07-11",
        out_dir=str(tmp_path),
    )
    content = open(path, "r", encoding="utf-8").read()

    assert "<!DOCTYPE html>" in content and "</html>" in content
    assert '"valuation"' not in content
    assert '"reverse_dcf_comment"' not in content


# ---------------------------------------------------------------------------
# Mechanical planning fields (sec_analyzer/interpret/planning.py, injected by
# analyzer._postprocess_phase2_result for every provider): scenario_returns,
# entry_plan, stop_adding, thesis_metric. These tests assert the embedded
# JSON payload carries the new keys through verbatim, and that the
# client-side section builders/CSS markers the template renders them with
# are present in the shipped HTML -- the same substring-on-HTML style as
# the rest of this file.
# ---------------------------------------------------------------------------


def test_generate_report_embeds_the_four_planning_fields_as_json(tmp_path):
    path = generate_report(
        "NVDA", "1y", _success_result_with_valuation(),
        metrics=_metrics(), technical=_technical(), flags=_flags(),
        price=128.40, as_of="2026-07-11", out_dir=str(tmp_path),
    )
    content = open(path, "r", encoding="utf-8").read()

    assert '"scenario_returns"' in content
    assert '"entry_plan"' in content
    assert '"stop_adding"' in content
    assert '"thesis_metric"' in content
    # Sample values from the fixtures survive JSON encoding verbatim.
    assert '"Net Kâr Marjı"' in content
    assert '"HIGH_UNCERTAINTY"' in content


def test_generate_report_renders_planning_section_builders_and_markers(tmp_path):
    path = generate_report(
        "NVDA", "1y", _success_result_with_valuation(),
        metrics=_metrics(), technical=_technical(), flags=_flags(),
        price=128.40, as_of="2026-07-11", out_dir=str(tmp_path),
    )
    content = open(path, "r", encoding="utf-8").read()

    # The four new client-side section builder functions.
    assert "entryPlanHtml" in content
    assert "stopAddingHtml" in content
    assert "thesisMetricHtml" in content
    assert "scenarioReturnsListHtml" in content
    # Their CSS/structural markers.
    assert "entry-plan-table" in content
    assert "scenario-returns-list" in content
    # Pre-existing structural markers (fan chart / triangulation / sensitivity)
    # must still be present alongside the new planning sections.
    assert "fan-band" in content
    assert "triangulation-row" in content
    assert "sensitivity-table" in content
    # Price chart: builder + CSS markers present, and the price series is
    # carried through in the embedded JSON payload.
    assert "priceChartCardHtml" in content
    assert "pchart-line" in content
    assert '"price_series"' in content


def test_generate_report_without_valuation_still_degrades_planning_sections_gracefully(tmp_path):
    """The classic result shape (no scenario_returns/entry_plan/stop_adding/
    thesis_metric keys at all -- as produced before this change, or by a
    caller that hasn't been updated) must still render a well-formed report:
    the builders are None-safe (entryPlanHtml/stopAddingHtml degrade to a
    plain sentence, scenarioReturnsListHtml/thesisMetricHtml to an empty
    string) rather than crashing client-side."""
    path = generate_report(
        "AAPL", "5y", _success_result(),
        metrics=_metrics(), technical=_technical(), flags=_flags(),
        price=128.40, as_of="2026-07-11", out_dir=str(tmp_path),
    )
    content = open(path, "r", encoding="utf-8").read()

    assert "<!DOCTYPE html>" in content and "</html>" in content
    assert '"scenario_returns"' not in content
    assert '"entry_plan"' not in content
    # NOTE: check for the colon that only follows a real JSON dict key --
    # "stop_adding" (no trailing colon) also appears as a bare substring of
    # the static template's function/class names (stopAddingHtml, etc.).
    assert '"stop_adding":' not in content
    assert '"thesis_metric"' not in content
    # The builder functions themselves are still part of the static template.
    assert "entryPlanHtml" in content
    assert "stopAddingHtml" in content
