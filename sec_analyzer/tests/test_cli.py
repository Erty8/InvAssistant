"""Unit tests for the terminal verdict card in sec_analyzer.cli.

Only ``_print_verdict_card`` is exercised here (via ``capsys``) -- it is the
one piece of ``cli.py`` that is both pure (stdout in, no network/DB) and has
meaningfully different behavior depending on whether ``result["valuation"]``
(the dict from ``sec_analyzer.valuation.engine.run_valuation``, SPEC.md
Sec.13) is present.
"""

from datetime import date, timedelta

import argparse

import pytest

from sec_analyzer.cli import _parse_as_of, _print_verdict_card, build_parser


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


def test_print_verdict_card_with_valuation_ffo_method_label_when_reit_populated(capsys):
    result = _result_with_valuation(dcf_enabled=False)
    result["valuation"]["ffo"] = {
        "scenarios": {"base": {"lo": 95.0, "hi": 115.0}},
        "ffo_per_share": 5.0,
        "implied_pffo": {"base": 15.0},
    }
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Fair Value (base, FFO): $95–$115   Güven: YÜKSEK" in out


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


def test_print_verdict_card_multiples_line_peg_applicable_not_mixed(capsys):
    result = _result_with_valuation()
    result["valuation"]["multiples"]["growth_adjusted"] = {
        "metric": "peg", "label": "PEG", "raw_label": "P/E",
        "value": 1.4, "percentile": 82.0, "raw_percentile": 88.0,
        "applicable": True, "reason": None, "base_growth_pct": 12.0, "sector_peg": None,
    }
    # triangulation multiples signal stays "pahali" (both components agree).
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Multiples:   P/E 88. pctile · PEG 1.40 (82. pctile)" in out
    assert "karışık sinyal" not in out


def test_print_verdict_card_multiples_line_peg_mixed_signal(capsys):
    result = _result_with_valuation()
    result["valuation"]["multiples"]["growth_adjusted"] = {
        "metric": "peg", "label": "PEG", "raw_label": "P/E",
        "value": 1.4, "percentile": 45.0, "raw_percentile": 88.0,
        "applicable": True, "reason": None, "base_growth_pct": 12.0, "sector_peg": None,
    }
    result["valuation"]["triangulation"]["signals"]["multiples"] = "karisik"
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Multiples:   P/E 88. pctile · PEG 1.40 (45. pctile) → karışık sinyal" in out


def test_print_verdict_card_multiples_line_reit_uses_pffo_no_peg(capsys):
    # A reit valuation with a populated pffo_percentile must render a P/FFO
    # line -- never P/E/P/FCF or a P/E-derived PEG, even when pe_percentile
    # and a growth_adjusted (PEG) block are both present in the payload.
    result = _result_with_valuation()
    result["valuation"]["sector_type"] = "reit"
    result["valuation"]["multiples"] = {
        "history_years": 8,
        "pe_percentile": 88.0,
        "ps_percentile": 60.0,
        "pfcf_percentile": None,
        "pffo_percentile": 72.0,
        "growth_adjusted": {
            "metric": "peg", "label": "PEG", "raw_label": "P/E",
            "value": 1.4, "percentile": 82.0, "raw_percentile": 88.0,
            "applicable": True, "reason": None, "base_growth_pct": 12.0, "sector_peg": None,
        },
    }
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Multiples:   P/FFO kendi 8y medyanının 72. yüzdeliğinde" in out
    assert "PEG" not in out
    assert "P/E" not in out


def test_print_verdict_card_multiples_line_reit_falls_back_to_ps(capsys):
    # pffo_percentile missing -> falls back to P/S (never P/E/P/FCF/PEG).
    result = _result_with_valuation()
    result["valuation"]["sector_type"] = "reit"
    result["valuation"]["multiples"] = {
        "history_years": 5,
        "pe_percentile": 88.0,
        "ps_percentile": 40.0,
        "pfcf_percentile": None,
        "pffo_percentile": None,
    }
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Multiples:   P/S kendi 5y medyanının 40. yüzdeliğinde" in out
    assert "PEG" not in out


def test_print_verdict_card_multiples_line_reit_pffo_none_degrades_to_veri_yetersiz(capsys):
    # Both pffo_percentile and ps_percentile missing -> the existing
    # "veri yetersiz" fallback, never a P/E-based line or PEG.
    result = _result_with_valuation()
    result["valuation"]["sector_type"] = "reit"
    result["valuation"]["multiples"] = {
        "history_years": 8,
        "pe_percentile": 88.0,
        "ps_percentile": None,
        "pfcf_percentile": None,
        "pffo_percentile": None,
        "growth_adjusted": {
            "metric": "peg", "label": "PEG", "raw_label": "P/E",
            "value": 1.4, "percentile": 82.0, "raw_percentile": 88.0,
            "applicable": True, "reason": None, "base_growth_pct": 12.0, "sector_peg": None,
        },
    }
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Multiples:   veri yetersiz" in out
    assert "PEG" not in out
    assert "P/E" not in out


def _technical_fixture():
    return {
        "price": 100.0, "rsi14": 59.0,
        "sma50_above_sma200": True, "golden_cross": True, "death_cross": False,
        "range_position_pct": 68.0,
        "return_1m_pct": 4.2, "return_3m_pct": 12.5, "return_6m_pct": -3.1,
        "macd": 1.234, "macd_signal": 0.9, "macd_hist": 0.334, "macd_cross": "bullish",
        "rel_volume": 1.3, "obv_trend": "up",
        "rsi_divergence": "bearish",
        "rsi_divergence_detail": {
            "type": "bearish", "price_prev": 140.0, "price_last": 142.0,
            "rsi_prev": 78.0, "rsi_last": 71.0, "last_date": "2025-05-20",
        },
        "verdict": "NÖTR", "verdict_detail": "RSI 59, SMA50 +%14",
        "resistance_levels": [
            {"low": 107.0, "high": 109.0, "price": 108.0, "dist_pct": 8.0, "strength": 4, "touches": 4, "fib": None, "last_touch": "2024-11-01", "is_52w_high": False, "is_52w_low": False},
            {"low": 117.0, "high": 119.0, "price": 118.0, "dist_pct": 18.0, "strength": 2, "touches": 2, "fib": None, "last_touch": "2024-06-01", "is_52w_high": True, "is_52w_low": False},
        ],
        "support_levels": [{"low": 91.0, "high": 94.0, "price": 92.5, "dist_pct": -7.5, "strength": 3, "touches": 3, "fib": None, "last_touch": "2025-01-15", "is_52w_high": False, "is_52w_low": False}],
    }


def test_print_verdict_card_technical_enriched_with_momentum_and_sr(capsys):
    result = _result_with_valuation()
    result["technical_verdict"] = "NÖTR"
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0}, technical=_technical_fixture())
    out = capsys.readouterr().out

    # Verdict line gains the detail suffix.
    assert "NÖTR (RSI 59, SMA50 +%14)" in out
    # Momentum sub-line with returns + trend.
    assert "Momentum:" in out
    assert "1a +%4" in out and "3a +%12" in out and "6a -%3" in out
    assert "Trend: yükseliş (GC)" in out
    # MACD + volume sub-line.
    assert "MACD/Hacim:" in out
    assert "MACD boğa (kesişim ↑)" in out
    assert "Hacim 1.3×" in out
    assert "OBV ↑" in out
    # RSI divergence sub-line.
    assert "RSI uyumsuzluğu (ayı)" in out
    assert "$140→$142" in out
    # Support/resistance sub-line: price ranges + combined evidence.
    assert "Destek/Direnç:" in out
    assert "Direnç $107–$109 (+%8 · 4×)" in out
    assert "$117–$119 (+%18 · 52h zirve + 2×)" in out   # 52w + touch evidence combined
    assert "Destek $91–$94 (-%8 · 3×)" in out


def test_print_verdict_card_technical_omitted_keeps_old_single_line(capsys):
    # No technical dict -> no momentum/SR sub-lines, verdict word only.
    result = _result_with_valuation()
    result["technical_verdict"] = "NÖTR"
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Teknik:" in out and "NÖTR" in out
    assert "Momentum:" not in out
    assert "MACD/Hacim:" not in out
    assert "RSI uyumsuzluğu" not in out
    assert "Destek/Direnç:" not in out


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


# ---------------------------------------------------------------------------
# "Olaylar:" line -- recent 8-K event signal attached as result["events"].
# ---------------------------------------------------------------------------


def test_print_verdict_card_events_line_summarizes_events(capsys):
    result = _result_without_valuation()
    result["events"] = [
        {"date": "2026-06-15", "severity": "critical", "items": ["4.02"],
         "categories": ["Önceki finansal tablolara güvenilemez (restatement)"]},
        {"date": "2026-07-01", "severity": "warning", "items": ["5.02"],
         "categories": ["Üst düzey yönetici/kurul değişikliği"]},
    ]
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    assert "Olaylar:" in out
    assert "1 kritik, 1 uyarı" in out
    # Most-severe-first: the restatement leads the listed events.
    assert "güvenilemez" in out


def test_print_verdict_card_events_line_reads_yok_when_empty(capsys):
    result = _result_without_valuation()
    result["events"] = []
    _print_verdict_card("AAPL", "1y", result, metrics={"price": 100.0})
    out = capsys.readouterr().out

    # The line is always present; empty events render "yok".
    assert "Olaylar:" in out
    events_line = [ln for ln in out.splitlines() if ln.startswith("Olaylar:")][0]
    assert events_line.strip().endswith("yok")


def test_print_verdict_card_events_line_absent_key_is_safe(capsys):
    # A legacy/stored result with no "events" key must still render "yok",
    # not crash.
    _print_verdict_card("AAPL", "1y", _result_without_valuation(), metrics={"price": 100.0})
    out = capsys.readouterr().out
    assert "Olaylar:" in out


# ---------------------------------------------------------------------------
# _parse_as_of -- argparse `type=` for --as-of: parses ISO dates, rejects a
# bad format or a future date.
# ---------------------------------------------------------------------------


def test_parse_as_of_accepts_a_valid_past_iso_date():
    result = _parse_as_of("2022-06-30")
    assert result == date(2022, 6, 30)


def test_parse_as_of_accepts_today():
    # "on or before today" -- today itself must be accepted, not rejected as
    # "in the future" (the check is a strict `>`, not `>=`).
    today_str = date.today().isoformat()
    assert _parse_as_of(today_str) == date.today()


def test_parse_as_of_rejects_malformed_date_string():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_as_of("not-a-date")


def test_parse_as_of_rejects_wrong_format():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_as_of("06/30/2022")


def test_parse_as_of_rejects_a_future_date():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_as_of(tomorrow)


# ---------------------------------------------------------------------------
# build_parser -- `analyze TICKER --as-of YYYY-MM-DD` wiring.
# ---------------------------------------------------------------------------


def test_build_parser_analyze_accepts_as_of_and_parses_to_a_date():
    parser = build_parser()
    args = parser.parse_args(["analyze", "AAPL", "--as-of", "2022-06-30"])
    assert args.as_of == date(2022, 6, 30)


def test_build_parser_analyze_defaults_as_of_to_none():
    parser = build_parser()
    args = parser.parse_args(["analyze", "AAPL"])
    assert args.as_of is None


def test_build_parser_analyze_rejects_a_future_as_of_date():
    parser = build_parser()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze", "AAPL", "--as-of", tomorrow])


def test_build_parser_calibrate_accepts_as_of_too():
    parser = build_parser()
    args = parser.parse_args(["calibrate", "--as-of", "2022-06-30"])
    assert args.as_of == date(2022, 6, 30)


# ---------------------------------------------------------------------------
# cmd_analyze -- as-of default is no-AI (SPEC's --no-ai / hindsight contract):
# with --as-of set and no explicit --provider, cmd_analyze must force
# provider="script" when calling interpret(), never falling back to
# Config.ANALYZER_PROVIDER (which could be "ollama"/"anthropic").
# ---------------------------------------------------------------------------


def _stub_offline_analyze_pipeline(monkeypatch, captured):
    """Stub every network/LLM boundary cmd_analyze touches with a cheap,
    offline substitute, and capture the ``provider`` kwarg interpret() is
    actually called with."""
    import sec_analyzer.cli as cli_module

    def _fake_fetch_normalize_store(args):
        normalized = {
            "annual": {"Revenue": [{"fy": 2020, "period_end": "2020-12-31", "value": 100.0}]},
            "quarterly": {},
            "missing": [],
        }
        return "723125", "Micron Technology, Inc.", normalized, [{"fy": 2020}]

    def _fake_fetch_price_and_technical(ticker, horizon, no_cache, as_of):
        return 100.0, as_of, None, None

    def _fake_fetch_risk_free_asof(as_of, no_cache):
        return None

    def _fake_fetch_submissions(cik, ticker, no_cache):
        return None

    def _fake_fetch_catalyst(submissions, ticker, as_of=None):
        return None

    def _fake_detect_filing_events(submissions, as_of=None):
        return []

    def _fake_fetch_analyst_targets(ticker, no_cache):
        # Only reached on a LIVE (as_of=None) run -- stubbed so that path
        # can't hit the network either.
        return None

    def _fake_interpret(*args, **kwargs):
        captured["provider"] = kwargs.get("provider")
        return {
            "fundamental_verdict": "MAKUL",
            "fair_value_range": {"bear": {}, "base": {}, "bull": {}},
            "technical_verdict": "VERİ YOK (fiyat verisi alınamadı)",
            "profile_fit": {"verdict": "KISMEN", "reason": "test"},
            "red_flags_comment": "yok",
            "catalyst": "bilinmiyor",
            "summary": "test",
        }

    monkeypatch.setattr(cli_module, "_fetch_normalize_store", _fake_fetch_normalize_store)
    monkeypatch.setattr(cli_module, "_fetch_price_and_technical", _fake_fetch_price_and_technical)
    monkeypatch.setattr(cli_module, "_fetch_risk_free_asof", _fake_fetch_risk_free_asof)
    monkeypatch.setattr(cli_module, "_fetch_submissions", _fake_fetch_submissions)
    monkeypatch.setattr(cli_module, "_fetch_catalyst", _fake_fetch_catalyst)
    monkeypatch.setattr(cli_module, "_detect_filing_events", _fake_detect_filing_events)
    monkeypatch.setattr(cli_module, "_fetch_analyst_targets", _fake_fetch_analyst_targets)
    monkeypatch.setattr(cli_module, "interpret", _fake_interpret)


def test_cmd_analyze_forces_script_provider_when_as_of_set_and_no_explicit_provider(tmp_path, monkeypatch):
    from sec_analyzer.config import Config

    monkeypatch.setattr(Config, "DB_PATH", str(tmp_path / "test.sqlite3"))
    captured = {}
    _stub_offline_analyze_pipeline(monkeypatch, captured)

    parser = build_parser()
    args = parser.parse_args(["analyze", "MU", "--as-of", "2022-06-30"])
    args.func(args)

    assert captured["provider"] == "script"


def test_cmd_analyze_respects_explicit_provider_even_with_as_of_set(tmp_path, monkeypatch):
    """An explicit --provider is honored (with a stderr warning printed by
    cmd_analyze) -- only the *default* (no --provider given) is forced to
    "script" in as-of mode."""
    from sec_analyzer.config import Config

    monkeypatch.setattr(Config, "DB_PATH", str(tmp_path / "test.sqlite3"))
    captured = {}
    _stub_offline_analyze_pipeline(monkeypatch, captured)

    parser = build_parser()
    args = parser.parse_args(["analyze", "MU", "--as-of", "2022-06-30", "--provider", "anthropic"])
    args.func(args)

    assert captured["provider"] == "anthropic"


def test_cmd_analyze_without_as_of_does_not_force_script_provider(tmp_path, monkeypatch):
    """A live (non as-of) run with no explicit --provider falls back to
    Config.ANALYZER_PROVIDER, not "script" -- the forcing is as-of-specific."""
    from sec_analyzer.config import Config

    monkeypatch.setattr(Config, "DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setattr(Config, "ANALYZER_PROVIDER", "script")
    captured = {}
    _stub_offline_analyze_pipeline(monkeypatch, captured)

    parser = build_parser()
    args = parser.parse_args(["analyze", "MU"])
    args.func(args)

    assert captured["provider"] == "script"
