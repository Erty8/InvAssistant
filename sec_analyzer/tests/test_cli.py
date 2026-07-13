"""Unit tests for the terminal verdict card in sec_analyzer.cli.

Only ``_print_verdict_card`` is exercised here (via ``capsys``) -- it is the
one piece of ``cli.py`` that is both pure (stdout in, no network/DB) and has
meaningfully different behavior depending on whether ``result["valuation"]``
(the dict from ``sec_analyzer.valuation.engine.run_valuation``, SPEC.md
Sec.13) is present.
"""

from sec_analyzer.cli import _print_verdict_card


def _fair_value_range():
    return {
        "bear": {"lo": 70.0, "hi": 85.0, "growth": "%8 büyüme", "discount_rate": "%12", "note": "n"},
        "base": {"lo": 95.0, "hi": 115.0, "growth": "%10 büyüme", "discount_rate": "%10", "note": "n"},
        "bull": {"lo": 120.0, "hi": 140.0, "growth": "%15 büyüme", "discount_rate": "%9", "note": "n"},
    }


def _result_without_valuation():
    """A pre-valuation-engine-shaped result (or a legacy stored one) --
    no ``"valuation"`` key at all."""
    return {
        "fair_value_range": _fair_value_range(),
        "fundamental_verdict": "MAKUL",
        "technical_verdict": "NÖTR (yetersiz veri)",
        "profile_fit": {"verdict": "KISMEN", "reason": "PROFIL.md bulunamadı."},
        "red_flags_comment": "yok",
        "catalyst": "bilinmiyor",
        "summary": "A short summary.",
    }


def _valuation(dcf_enabled=True):
    return {
        "sector_type": "mature",
        "dcf": {"enabled": dcf_enabled, "disabled_reason": None},
        "reverse_dcf": {"implied_growth": 0.19, "realized_cagr_5y": 0.14, "realized_label": "5y"},
        "multiples": {
            "history_years": 8,
            "pe_percentile": 88.0,
            "ps_percentile": None,
            "pfcf_percentile": None,
        },
        "triangulation": {
            "signals": {"dcf": "pahali", "reverse_dcf": "pahali", "multiples": "pahali"},
            "confidence": "YÜKSEK",
            "direction": "pahali",
        },
        "sensitivity": {"lo": 87.0, "hi": 131.0, "high_uncertainty": False},
    }


def _result_with_valuation(dcf_enabled=True, high_uncertainty=False):
    result = _result_without_valuation()
    result["confidence"] = "YÜKSEK"
    valuation = _valuation(dcf_enabled=dcf_enabled)
    valuation["sensitivity"]["high_uncertainty"] = high_uncertainty
    result["valuation"] = valuation
    return result


# ---------------------------------------------------------------------------
# Without a "valuation" key: renders exactly as before the engine wiring.
# ---------------------------------------------------------------------------


def test_print_verdict_card_without_valuation_renders_old_shape(capsys):
    _print_verdict_card("AAPL", "1y", _result_without_valuation(), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Fair Value (base): $95–$115" in out
    # None of the new method-label/confidence/valuation-derived lines appear.
    assert "Güven:" not in out
    assert "Reverse DCF:" not in out
    assert "Multiples:" not in out
    assert "Üçgenleme:" not in out
    assert "Duyarlılık:" not in out
    assert "Fundamental:" in out
    assert "MAKUL" in out


def test_print_verdict_card_error_result_still_short_circuits(capsys):
    _print_verdict_card("AAPL", "1y", {"error": "boom", "summary": "no dice"}, metrics={"price": 100.0})
    out = capsys.readouterr().out
    assert "Analiz kullanılamıyor (boom): no dice" in out
    assert "Fair Value" not in out


# ---------------------------------------------------------------------------
# With a "valuation" dict: the SPEC.md Sec.13 additions.
# ---------------------------------------------------------------------------


def test_print_verdict_card_with_valuation_dcf_method_label(capsys):
    _print_verdict_card("AAPL", "1y", _result_with_valuation(dcf_enabled=True), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Fair Value (base, DCF): $95–$115   Güven: YÜKSEK" in out


def test_print_verdict_card_with_valuation_pb_roe_method_label_when_dcf_disabled(capsys):
    _print_verdict_card("AAPL", "1y", _result_with_valuation(dcf_enabled=False), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Fair Value (base, P/B×ROE): $95–$115   Güven: YÜKSEK" in out


def test_print_verdict_card_reverse_dcf_line_built_from_numbers(capsys):
    _print_verdict_card("AAPL", "1y", _result_with_valuation(), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Reverse DCF: fiyat 10y %19 CAGR ima ediyor (gerçekleşen 5y: %14)" in out


def test_print_verdict_card_reverse_dcf_line_prefers_comment(capsys):
    result = _result_with_valuation()
    result["reverse_dcf_comment"] = "Fiyat oldukça agresif bir büyüme ima ediyor."
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Reverse DCF: Fiyat oldukça agresif bir büyüme ima ediyor." in out
    assert "%19 CAGR" not in out


def test_print_verdict_card_multiples_line_primary_pe(capsys):
    _print_verdict_card("AAPL", "1y", _result_with_valuation(), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Multiples:   P/E kendi 8y medyanının 88. yüzdeliğinde" in out


def test_print_verdict_card_multiples_line_falls_back_and_veri_yetersiz(capsys):
    result = _result_with_valuation()
    result["valuation"]["multiples"] = {
        "history_years": 2,
        "pe_percentile": None,
        "ps_percentile": None,
        "pfcf_percentile": None,
    }
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Multiples:   veri yetersiz" in out


def test_print_verdict_card_triangulation_line_high_confidence_is_yon_net(capsys):
    _print_verdict_card("AAPL", "1y", _result_with_valuation(), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Üçgenleme:   DCF pahali · rDCF pahali · multiples pahali → yön net" in out


def test_print_verdict_card_triangulation_line_low_confidence_is_yon_karisik(capsys):
    result = _result_with_valuation()
    result["valuation"]["triangulation"]["confidence"] = "DÜŞÜK"
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "yön karışık" in out


def test_print_verdict_card_sensitivity_line_flags_high_uncertainty(capsys):
    _print_verdict_card("AAPL", "1y", _result_with_valuation(high_uncertainty=True), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Duyarlılık:  base $87–$131 (g±2pp, r±1pp) — yüksek belirsizlik" in out


def test_print_verdict_card_sensitivity_line_without_flag(capsys):
    _print_verdict_card("AAPL", "1y", _result_with_valuation(high_uncertainty=False), metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Duyarlılık:  base $87–$131 (g±2pp, r±1pp)" in out
    assert "yüksek belirsizlik" not in out
