"""Unit tests for the ``GET /report`` route in ``sec_analyzer.web.app``.

These use Flask's ``test_client()`` and monkeypatch the same boundary the
CLI tests treat as opaque: the network/DB-touching pipeline
(``_run_full_pipeline``) and the LLM/rule-based interpretation
(``interpret``, imported into ``sec_analyzer.web.app``'s own namespace).
Nothing here talks to SEC EDGAR, a price-data provider, an LLM, or the real
on-disk database -- ``save_verdict`` is stubbed out too.
"""

import sec_analyzer.web.app as web_app

_SCENARIO = {
    "lo": 90.0,
    "hi": 110.0,
    "growth": "%8 büyüme",
    "discount_rate": "%12",
    "note": "İki aşamalı DCF, FCF/hisse çapası.",
}


def _fair_value_range():
    return {
        "bear": {**_SCENARIO, "lo": 70.0, "hi": 95.0},
        "base": {**_SCENARIO},
        "bull": {**_SCENARIO, "lo": 115.0, "hi": 140.0},
    }


def _valuation():
    """A ``result["valuation"]`` dict matching SPEC.md Sec.11/9/10 -- just
    enough for the template's triangulation/sensitivity sections to render."""
    return {
        "sector_type": "mature",
        "dcf": {"enabled": True, "disabled_reason": None},
        "fair_value_range": _fair_value_range(),
        "reverse_dcf": {"implied_growth": 0.19, "realized_cagr_5y": 0.14, "realized_label": "5y"},
        "multiples": {
            "pe_percentile": 88.0, "ps_percentile": None, "pfcf_percentile": None, "history_years": 8,
        },
        "sensitivity": {
            "growth_values": [0.06, 0.08, 0.10],
            "dr_values": [0.11, 0.12, 0.13],
            "matrix": [[95.0, 87.0, 80.0], [108.0, 100.0, 92.0], [122.0, 112.0, 103.0]],
            "lo": 80.0, "hi": 122.0, "high_uncertainty": False,
        },
        "triangulation": {
            "signals": {"dcf": "pahali", "reverse_dcf": "pahali", "multiples": "pahali"},
            "confidence": "YÜKSEK",
            "direction": "pahali",
        },
    }


def _interpret_result():
    return {
        "fair_value_range": _fair_value_range(),
        "fundamental_verdict": "PAHALI",
        "technical_verdict": "NÖTR",
        "confidence": "YÜKSEK",
        "profile_fit": {"verdict": "KISMEN", "reason": "PROFIL.md bulunamadı."},
        "reverse_dcf_comment": "Fiyat 10y %19 CAGR ima ediyor.",
        "red_flags_comment": "yok",
        "catalyst": "bilinmiyor",
        "summary": "Test özeti.",
        "valuation": _valuation(),
        "_provider": "script",
        "_model": "rule-based-v2",
        "_horizon": "1y",
        "_weights": {"fundamental": 0.5, "technical": 0.5},
    }


def _run_full_pipeline_stub(ticker, years, no_cache, horizon):
    normalized = {
        "cik": "320193", "entity_name": "Apple Inc.", "currency": "USD",
        "annual": {}, "quarterly": {}, "missing": [],
    }
    ratios = []
    metrics = {"price": 128.40}
    technical = {
        "price": 128.40, "as_of": "2026-07-11",
        "verdict": "NÖTR", "verdict_detail": "", "horizon_summary": "",
    }
    flags = []
    catalyst = None
    price = 128.40
    submissions = None
    price_df = None
    return (
        "320193", "Apple Inc.", normalized, ratios, metrics, technical, flags,
        catalyst, price, submissions, price_df,
    )


def _fake_interpret(*args, **kwargs):
    return _interpret_result()


def test_report_happy_path_returns_full_report_html(monkeypatch):
    monkeypatch.setattr(web_app, "_run_full_pipeline", _run_full_pipeline_stub)
    monkeypatch.setattr(web_app, "interpret", _fake_interpret)
    monkeypatch.setattr(web_app, "save_verdict", lambda *a, **k: 1)

    client = web_app.app.test_client()
    resp = client.get("/report?ticker=AAPL&horizon=1y&provider=script&years=12")

    assert resp.status_code == 200
    assert resp.content_type.startswith("text/html")

    body = resp.get_data(as_text=True)
    # No Python traceback ever leaks into the response.
    assert "Traceback" not in body
    # The fan-chart / triangulation / sensitivity markers the report
    # template's client-side renderer keys off of.
    assert "fan-row-track" in body
    assert "triangulationRowHtml" in body
    assert "sensitivityTableHtml" in body
    assert "sensitivity-table" in body
    assert "AAPL" in body


def test_report_missing_ticker_returns_html_error_page():
    client = web_app.app.test_client()
    resp = client.get("/report?provider=script")

    assert resp.status_code == 400
    assert resp.content_type.startswith("text/html")
    body = resp.get_data(as_text=True)
    assert "Traceback" not in body
    assert "required" in body


def test_report_unresolvable_ticker_returns_graceful_html_error(monkeypatch):
    def _raise_value_error(ticker, years, no_cache, horizon):
        raise ValueError(f"Could not resolve ticker {ticker!r} to a CIK.")

    monkeypatch.setattr(web_app, "_run_full_pipeline", _raise_value_error)

    client = web_app.app.test_client()
    resp = client.get("/report?ticker=ZZINVALIDZZ&provider=script")

    assert resp.status_code == 404
    assert resp.content_type.startswith("text/html")
    body = resp.get_data(as_text=True)
    assert "Traceback" not in body
    assert "ZZINVALIDZZ" in body


def test_report_unexpected_error_returns_graceful_html_error_not_json(monkeypatch):
    def _raise_boom(ticker, years, no_cache, horizon):
        raise RuntimeError("boom")

    monkeypatch.setattr(web_app, "_run_full_pipeline", _raise_boom)

    client = web_app.app.test_client()
    resp = client.get("/report?ticker=AAPL&provider=script")

    assert resp.status_code == 500
    assert resp.content_type.startswith("text/html")
    body = resp.get_data(as_text=True)
    assert "Traceback" not in body
    assert "unexpected server error" in body.lower()


def test_report_passes_submissions_and_price_df_into_interpret(monkeypatch):
    """Regression guard for the web-UI parity fix: /report (and /api/analyze)
    must thread the once-fetched submissions/price_df into interpret(), not
    drop them on the floor."""
    captured = {}

    def _capturing_interpret(*args, **kwargs):
        captured.update(kwargs)
        return _interpret_result()

    def _pipeline_with_submissions(ticker, years, no_cache, horizon):
        base = list(_run_full_pipeline_stub(ticker, years, no_cache, horizon))
        base[9] = {"sic": "7372", "sicDescription": "Prepackaged Software"}  # submissions
        base[10] = "not-a-real-dataframe-but-non-none"  # price_df
        return tuple(base)

    monkeypatch.setattr(web_app, "_run_full_pipeline", _pipeline_with_submissions)
    monkeypatch.setattr(web_app, "interpret", _capturing_interpret)
    monkeypatch.setattr(web_app, "save_verdict", lambda *a, **k: 1)

    client = web_app.app.test_client()
    resp = client.get("/report?ticker=AAPL&provider=script")

    assert resp.status_code == 200
    assert captured.get("submissions") == {"sic": "7372", "sicDescription": "Prepackaged Software"}
    assert captured.get("price_df") == "not-a-real-dataframe-but-non-none"
