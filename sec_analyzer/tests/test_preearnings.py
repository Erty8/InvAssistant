"""Unit tests for the pre-earnings briefing screener
(``sec_analyzer.screener.preearnings``).

Structural precedent: ``sec_analyzer/tests/test_screener.py`` (no-network
monkeypatching of the fetch layer) and ``test_swing_score.py`` (unit-testing
a pure helper -- here, ``_build_notes`` -- in isolation with minimal
fixtures).

No network anywhere in this module: every fetch function this screener
calls (``resolve_cik``, ``get_submissions``, ``estimate_next_earnings``,
``load_verdicts``, ``get_earnings_history``, ``detect_events``) is
monkeypatched as imported into ``sec_analyzer.screener.preearnings``'s own
namespace, and ``SecHttpClient`` is stubbed so the module never needs a real
``SEC_USER_AGENT``. Every scan-level test pins an explicit ``today`` so
assertions are exact and deterministic.
"""

import pytest

from sec_analyzer.screener import preearnings
from sec_analyzer.screener.preearnings import (
    DEFAULT_STALE_DAYS,
    DEFAULT_WITHIN_DAYS,
    _build_notes,
    format_surprise_pct,
    scan_preearnings,
)
from datetime import date

TODAY = date(2026, 8, 7)


# ---------------------------------------------------------------------------
# Fakes / installation helpers for the scan-level (integration) tests
# ---------------------------------------------------------------------------


class _DummyClient:
    """Stand-in for ``SecHttpClient`` -- never actually used by the fakes
    below (they ignore the ``client`` argument), but ``scan_preearnings``
    constructs one unconditionally, and the real class requires
    ``SEC_USER_AGENT`` to be configured."""


@pytest.fixture(autouse=True)
def _stub_http_client(monkeypatch):
    monkeypatch.setattr(preearnings, "SecHttpClient", _DummyClient)


def _catalyst(
    days_until,
    recently_reported=False,
    label=None,
    source="8-K 2.02",
    based_on="son 8 kazanç açıklamasının medyan aralığı (91 gün)",
    estimate_date="2026-08-21",
):
    """Build a fake ``estimate_next_earnings``-shaped dict."""
    return {
        "estimate_date": estimate_date,
        "label": label or f"Q earnings ~{estimate_date}",
        "based_on": based_on,
        "source": source,
        "last_report_date": "2026-05-01",
        "days_until": days_until,
        "recently_reported": recently_reported,
    }


def _verdict_row(**overrides):
    """Build a fake ``load_verdicts()[0]``-shaped row."""
    row = {
        "analyzed_at": "2026-07-02T10:11:12",
        "price": 100.0,
        "fundamental_verdict": "UCUZ",
        "technical_verdict": "NÖTR",
        "momentum_verdict": "POZİTİF",
        "confidence": "ORTA",
        "sector_type": "mature",
        "fv_base_lo": 90.0,
        "fv_base_hi": 110.0,
        "horizon": "1y",
        "provider": "rule_based",
    }
    row.update(overrides)
    return row


def _make_resolve_cik(overrides=None):
    overrides = overrides or {}

    def _fake(ticker, client, no_cache=False):
        cfg = overrides.get(ticker)
        if cfg and cfg.get("raise"):
            raise ValueError(cfg["raise"])
        cik = (cfg or {}).get("cik", ticker)
        name = (cfg or {}).get("name", f"{ticker} Inc.")
        return cik, name

    return _fake


def _fake_get_submissions(cik, client, no_cache=False):
    # Content is irrelevant -- estimate_next_earnings and detect_events are
    # faked separately and only need a way to identify which ticker this
    # submissions blob belongs to.
    return {"_cik": cik}


def _make_estimate_next_earnings(catalysts):
    def _fake(submissions, today=None):
        return catalysts.get(submissions.get("_cik"))

    return _fake


def _make_load_verdicts(verdicts):
    def _fake(ticker, db_path=None, limit=1, live_only=True):
        row = verdicts.get(ticker)
        return [row] if row is not None else []

    return _fake


def _make_get_earnings_history(histories):
    def _fake(ticker, no_cache=False):
        return histories.get(ticker)

    return _fake


def _make_detect_events(events_map):
    def _fake(submissions, lookback_days=None, min_severity=None, max_events=None, today=None):
        return events_map.get(submissions.get("_cik"), [])

    return _fake


def _install(monkeypatch, catalysts=None, resolve_overrides=None, verdicts=None, histories=None, events_map=None):
    monkeypatch.setattr(preearnings, "resolve_cik", _make_resolve_cik(resolve_overrides))
    monkeypatch.setattr(preearnings, "get_submissions", _fake_get_submissions)
    monkeypatch.setattr(preearnings, "estimate_next_earnings", _make_estimate_next_earnings(catalysts or {}))
    monkeypatch.setattr(preearnings, "load_verdicts", _make_load_verdicts(verdicts or {}))
    monkeypatch.setattr(preearnings, "get_earnings_history", _make_get_earnings_history(histories or {}))
    monkeypatch.setattr(preearnings, "detect_events", _make_detect_events(events_map or {}))


# ---------------------------------------------------------------------------
# Window filter / skip reasons
# ---------------------------------------------------------------------------


def test_ticker_inside_window_included_outside_window_skipped(monkeypatch):
    catalysts = {"IN": _catalyst(days_until=5), "OUT": _catalyst(days_until=30)}
    _install(monkeypatch, catalysts=catalysts)

    result = scan_preearnings(["IN", "OUT"], within_days=14, today=TODAY)

    assert [r["ticker"] for r in result["rows"]] == ["IN"]
    skip_reasons = {s["ticker"]: s["reason"] for s in result["skipped"]}
    assert "OUT" in skip_reasons
    assert "30" in skip_reasons["OUT"]
    assert "14" in skip_reasons["OUT"]


def test_include_all_keeps_out_of_window_ticker(monkeypatch):
    catalysts = {"OUT": _catalyst(days_until=30)}
    _install(monkeypatch, catalysts=catalysts)

    result = scan_preearnings(["OUT"], within_days=14, today=TODAY, include_all=True)

    assert [r["ticker"] for r in result["rows"]] == ["OUT"]
    assert result["skipped"] == []


def test_just_reported_sorts_first_despite_large_days_until(monkeypatch):
    catalysts = {
        "JR": _catalyst(
            days_until=90,
            recently_reported=True,
            label="7 Ağu tarihinde açıklandı · sonraki: Q3 earnings ~5 Kas",
        ),
        "SOON": _catalyst(days_until=5),
    }
    _install(monkeypatch, catalysts=catalysts)

    result = scan_preearnings(["SOON", "JR"], within_days=14, today=TODAY)

    assert [r["ticker"] for r in result["rows"]] == ["JR", "SOON"]


def test_default_within_days_applied_when_not_specified(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    _install(monkeypatch, catalysts=catalysts)

    result = scan_preearnings(["AAA"], today=TODAY)

    assert result["within_days"] == DEFAULT_WITHIN_DAYS


def test_result_shape_top_level_fields(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5), "OUT": _catalyst(days_until=99)}
    _install(monkeypatch, catalysts=catalysts)

    result = scan_preearnings(["AAA", "OUT"], today=TODAY, within_days=14)

    assert result["today"] == "2026-08-07"
    assert result["requested"] == 2
    assert result["count"] == 1
    assert set(result.keys()) == {"generated_at", "today", "within_days", "count", "requested", "rows", "skipped"}
    assert result["generated_at"].endswith("Z")


# ---------------------------------------------------------------------------
# fv_vs_price_pct arithmetic
# ---------------------------------------------------------------------------


def test_fv_vs_price_pct_hand_computed(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    verdicts = {"AAA": _verdict_row(price=210.5, fv_base_lo=240.0, fv_base_hi=280.0)}
    _install(monkeypatch, catalysts=catalysts, verdicts=verdicts)

    row = scan_preearnings(["AAA"], today=TODAY)["rows"][0]

    # mid = (240 + 280) / 2 = 260.0; pct = (260 / 210.5 - 1) * 100 = 23.515... -> 23.5
    assert row["fv_base_mid"] == pytest.approx(260.0)
    assert row["fv_vs_price_pct"] == pytest.approx(23.5)


def test_fv_vs_price_pct_none_when_price_missing(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    verdicts = {"AAA": _verdict_row(price=None, fv_base_lo=240.0, fv_base_hi=280.0)}
    _install(monkeypatch, catalysts=catalysts, verdicts=verdicts)

    row = scan_preearnings(["AAA"], today=TODAY)["rows"][0]

    assert row["fv_vs_price_pct"] is None


def test_fv_vs_price_pct_none_when_price_zero(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    verdicts = {"AAA": _verdict_row(price=0.0, fv_base_lo=240.0, fv_base_hi=280.0)}
    _install(monkeypatch, catalysts=catalysts, verdicts=verdicts)

    row = scan_preearnings(["AAA"], today=TODAY)["rows"][0]

    assert row["fv_vs_price_pct"] is None


def test_fv_vs_price_pct_none_when_band_missing(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    verdicts = {"AAA": _verdict_row(price=100.0, fv_base_lo=None, fv_base_hi=None)}
    _install(monkeypatch, catalysts=catalysts, verdicts=verdicts)

    row = scan_preearnings(["AAA"], today=TODAY)["rows"][0]

    assert row["fv_base_mid"] is None
    assert row["fv_vs_price_pct"] is None


# ---------------------------------------------------------------------------
# Beat/miss history arithmetic
# ---------------------------------------------------------------------------


def test_beat_streak_counts_only_consecutive_most_recent_positive_quarters(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    # Newest first: beat, beat, miss, beat -- the trailing beat must NOT
    # extend the streak past the miss that interrupts it.
    quarters = [
        {"period": "2026-06-30", "eps_estimate": 1.0, "eps_actual": 1.1, "surprise_pct": 10.0},
        {"period": "2026-03-31", "eps_estimate": 1.0, "eps_actual": 1.05, "surprise_pct": 5.0},
        {"period": "2025-12-31", "eps_estimate": 1.0, "eps_actual": 0.9, "surprise_pct": -10.0},
        {"period": "2025-09-30", "eps_estimate": 1.0, "eps_actual": 1.2, "surprise_pct": 20.0},
    ]
    histories = {"AAA": {"quarters": quarters, "source": "yfinance"}}
    _install(monkeypatch, catalysts=catalysts, histories=histories)

    row = scan_preearnings(["AAA"], today=TODAY)["rows"][0]

    assert row["beat_streak"] == 2
    assert row["beat_count"] == 3
    assert row["miss_count"] == 1


def test_avg_surprise_pct_ignores_none_quarters(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    quarters = [
        {"period": "p1", "eps_estimate": 1.0, "eps_actual": 1.1, "surprise_pct": 10.0},
        {"period": "p2", "eps_estimate": None, "eps_actual": None, "surprise_pct": None},
        {"period": "p3", "eps_estimate": 1.0, "eps_actual": 0.98, "surprise_pct": -2.0},
    ]
    histories = {"AAA": {"quarters": quarters, "source": "yfinance"}}
    _install(monkeypatch, catalysts=catalysts, histories=histories)

    row = scan_preearnings(["AAA"], today=TODAY)["rows"][0]

    assert row["avg_surprise_pct"] == pytest.approx(4.0)  # mean(10.0, -2.0)


# ---------------------------------------------------------------------------
# No stored verdict
# ---------------------------------------------------------------------------


def test_no_stored_verdict_yields_stale_and_none_fields(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5)}
    _install(monkeypatch, catalysts=catalysts)  # no verdicts dict -> load_verdicts returns []

    row = scan_preearnings(["AAA"], today=TODAY)["rows"][0]

    assert row["stale_verdict"] is True
    assert any("kayıtlı analiz yok" in n for n in row["notes"])
    for field in (
        "verdict", "technical_verdict", "momentum_verdict", "confidence", "sector_type",
        "horizon", "verdict_date", "verdict_age_days", "price", "fv_base_lo", "fv_base_hi",
        "fv_base_mid", "fv_vs_price_pct",
    ):
        assert row[field] is None, f"{field} should be None with no stored verdict"


# ---------------------------------------------------------------------------
# Resilience: one bad ticker / no estimate must not crash the scan
# ---------------------------------------------------------------------------


def test_resolve_cik_raises_lands_in_skipped_other_tickers_still_scan(monkeypatch):
    catalysts = {"GOOD": _catalyst(days_until=5)}
    _install(monkeypatch, catalysts=catalysts, resolve_overrides={"BAD": {"raise": "no cik"}})

    result = scan_preearnings(["BAD", "GOOD"], today=TODAY)

    assert [r["ticker"] for r in result["rows"]] == ["GOOD"]
    skip_reasons = {s["ticker"]: s["reason"] for s in result["skipped"]}
    assert skip_reasons.get("BAD") == "no cik"


def test_estimate_next_earnings_none_is_skipped_not_crashed(monkeypatch):
    _install(monkeypatch, catalysts={})  # AAA has no catalyst entry -> None

    result = scan_preearnings(["AAA"], today=TODAY)

    assert result["rows"] == []
    assert result["skipped"] == [{"ticker": "AAA", "reason": "kazanç tarihi tahmin edilemedi"}]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_scan_is_deterministic_across_runs(monkeypatch):
    catalysts = {"AAA": _catalyst(days_until=5), "BBB": _catalyst(days_until=2)}
    verdicts = {"AAA": _verdict_row()}
    quarters = [
        {"period": "2026-06-30", "eps_estimate": 1.0, "eps_actual": 1.1, "surprise_pct": 10.0},
        {"period": "2026-03-31", "eps_estimate": 1.0, "eps_actual": 0.9, "surprise_pct": -10.0},
        {"period": "2025-12-31", "eps_estimate": 1.0, "eps_actual": 1.05, "surprise_pct": 5.0},
    ]
    histories = {"AAA": {"quarters": quarters, "source": "yfinance"}}
    events_map = {
        "AAA": [
            {
                "date": "2026-07-01",
                "form": "8-K",
                "items": ["5.02"],
                "categories": ["Üst düzey yönetici/kurul değişikliği"],
                "severity": "warning",
                "accession": "0000002488-26-000115",
                "primary_doc": "aaa-20260701.htm",
            }
        ]
    }
    _install(monkeypatch, catalysts=catalysts, verdicts=verdicts, histories=histories, events_map=events_map)

    result1 = scan_preearnings(["AAA", "BBB"], today=TODAY)
    result2 = scan_preearnings(["AAA", "BBB"], today=TODAY)

    result1.pop("generated_at")
    result2.pop("generated_at")
    assert result1 == result2


def test_notes_never_contain_none_or_nan_strings(monkeypatch):
    catalysts = {
        "AAA": _catalyst(days_until=2),
        "BBB": _catalyst(
            days_until=90,
            recently_reported=True,
            label="1 Ağu tarihinde açıklandı · sonraki: Q1 earnings ~1 Kas",
        ),
    }
    verdicts = {"BBB": _verdict_row(confidence="DÜŞÜK", fundamental_verdict="PAHALI", momentum_verdict="POZİTİF")}
    quarters = [
        {"period": "p1", "eps_estimate": 1.0, "eps_actual": None, "surprise_pct": None},
        {"period": "p2", "eps_estimate": 1.0, "eps_actual": None, "surprise_pct": None},
        {"period": "p3", "eps_estimate": 1.0, "eps_actual": None, "surprise_pct": None},
    ]
    histories = {"BBB": {"quarters": quarters, "source": "yfinance"}}
    _install(monkeypatch, catalysts=catalysts, verdicts=verdicts, histories=histories)

    result = scan_preearnings(["AAA", "BBB"], today=TODAY)

    all_notes_text = " ".join(n for r in result["rows"] for n in r["notes"])
    assert "None" not in all_notes_text
    assert "nan" not in all_notes_text.lower()


# ---------------------------------------------------------------------------
# format_surprise_pct
# ---------------------------------------------------------------------------


def test_format_surprise_pct_positive():
    assert format_surprise_pct(8.4) == "+%8,4"


def test_format_surprise_pct_negative():
    assert format_surprise_pct(-2.1) == "-%2,1"


def test_format_surprise_pct_none():
    assert format_surprise_pct(None) == "—"


def test_format_surprise_pct_nan_and_inf():
    assert format_surprise_pct(float("nan")) == "—"
    assert format_surprise_pct(float("inf")) == "—"


# ---------------------------------------------------------------------------
# _build_notes -- each rule fires on its own minimal fixture, and only then
# ---------------------------------------------------------------------------


def _neutral_row(**overrides):
    """A row fixture for which none of the note rules should fire."""
    row = {
        "verdict_date": "2026-07-01T00:00:00",
        "verdict_age_days": 10,
        "verdict": None,
        "momentum_verdict": None,
        "just_reported": False,
        "days_until": 10,
        "catalyst_label": "Q earnings ~17 Ağu",
        "surprises": [],
        "beat_count": 0,
        "avg_surprise_pct": None,
        "events": [],
        "confidence": "ORTA",
    }
    row.update(overrides)
    return row


def test_notes_baseline_neutral_row_has_no_notes():
    assert _build_notes(_neutral_row()) == []


def test_note_rule1_no_stored_verdict():
    row = _neutral_row(verdict_date=None, verdict_age_days=None)
    assert _build_notes(row) == ["Bu isim için kayıtlı analiz yok — bilanço öncesi bir analiz çalıştırın."]


def test_note_rule2_stale_verdict_fires_past_threshold():
    row = _neutral_row(verdict_age_days=DEFAULT_STALE_DAYS + 1)
    notes = _build_notes(row)
    assert notes == [f"Kayıtlı analiz {DEFAULT_STALE_DAYS + 1} günlük — bilanço öncesi yenilenmeli."]


def test_note_rule2_does_not_fire_at_threshold_boundary():
    row = _neutral_row(verdict_age_days=DEFAULT_STALE_DAYS)
    assert _build_notes(row) == []


def test_note_rule3_just_reported():
    row = _neutral_row(
        just_reported=True,
        days_until=90,
        catalyst_label="7 Ağu tarihinde açıklandı · sonraki: Q3 earnings ~5 Kas",
    )
    assert _build_notes(row) == ["7 Ağu tarihinde açıklandı; sıradaki katalizör bir sonraki çeyrek."]


def test_note_rule4_imminent_earnings():
    row = _neutral_row(days_until=2)
    assert _build_notes(row) == ["Bilanço 2 gün içinde — yeni pozisyon açmak için zamanlama riski yüksek."]


def test_note_rule4_does_not_fire_above_threshold():
    row = _neutral_row(days_until=4)
    assert _build_notes(row) == []


def test_note_rule4_does_not_fire_when_just_reported_even_if_imminent():
    row = _neutral_row(
        days_until=1, just_reported=True, catalyst_label="6 Ağu tarihinde açıklandı · sonraki: ..."
    )
    notes = _build_notes(row)
    assert not any("gün içinde" in n for n in notes)


@pytest.mark.parametrize("momentum", ["POZİTİF", "GÜÇLÜ+"])
def test_note_rule5_ucuz_with_positive_momentum(momentum):
    row = _neutral_row(verdict="UCUZ", momentum_verdict=momentum)
    assert _build_notes(row) == ["UCUZ değerleme ile pozitif momentum aynı yönde — bilanço teyit edici olabilir."]


def test_note_rule5_ucuz_with_negative_momentum():
    row = _neutral_row(verdict="UCUZ", momentum_verdict="NEGATİF")
    assert _build_notes(row) == ["Model UCUZ diyor ama momentum negatif; bilanço bu ayrışmayı çözebilecek katalizör."]


def test_note_rule5_pahali_with_positive_momentum():
    row = _neutral_row(verdict="PAHALI", momentum_verdict="POZİTİF")
    assert _build_notes(row) == ["PAHALI değerleme momentumla taşınıyor; bilanço bir kırılma noktası."]


def test_note_rule5_pahali_with_negative_momentum_no_note():
    row = _neutral_row(verdict="PAHALI", momentum_verdict="NEGATİF")
    assert _build_notes(row) == []


def test_note_rule5_neutral_momentum_no_note():
    row = _neutral_row(verdict="UCUZ", momentum_verdict="NÖTR")
    assert _build_notes(row) == []


def test_note_rule5_unrecognized_momentum_no_note():
    row = _neutral_row(verdict="UCUZ", momentum_verdict="BILINMIYOR")
    assert _build_notes(row) == []


def test_note_rule5_no_note_when_verdict_unknown():
    row = _neutral_row(verdict=None, momentum_verdict="NEGATİF")
    assert _build_notes(row) == []


def test_note_rule6_beat_record():
    row = _neutral_row(surprises=[{}, {}, {}, {}], beat_count=3, avg_surprise_pct=8.4)
    assert _build_notes(row) == ["Son 4 çeyrekte 3 kez beklentiyi aştı (ort. +%8,4)."]


def test_note_rule6_all_miss():
    row = _neutral_row(surprises=[{}, {}, {}], beat_count=0, avg_surprise_pct=-5.0)
    assert _build_notes(row) == ["Son 3 çeyrekte hiç beklentiyi aşamadı."]


def test_note_rule6_does_not_fire_below_three_quarters():
    row = _neutral_row(surprises=[{}, {}], beat_count=1, avg_surprise_pct=5.0)
    assert _build_notes(row) == []


def test_note_rule7_material_events():
    row = _neutral_row(events=[{"categories": ["Üst düzey yönetici/kurul değişikliği"]}])
    assert _build_notes(row) == [
        f"Son {preearnings._EVENTS_LOOKBACK_DAYS} günde materyal dosyalama olayı var: "
        "Üst düzey yönetici/kurul değişikliği."
    ]


def test_note_rule7_does_not_fire_without_events():
    row = _neutral_row(events=[])
    assert _build_notes(row) == []


def test_note_rule8_low_confidence():
    row = _neutral_row(confidence="DÜŞÜK")
    assert _build_notes(row) == ["Değerleme güveni DÜŞÜK; bilanço sonrası bandın kayması olası."]


def test_note_rule8_does_not_fire_for_other_confidence():
    row = _neutral_row(confidence="YÜKSEK")
    assert _build_notes(row) == []
