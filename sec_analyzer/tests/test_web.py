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


def _run_full_pipeline_stub(ticker, years, no_cache, horizon, as_of=None):
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
    fred_rate = {"value_pct": 2.98, "date": "2022-06-29", "series": "DGS10"} if as_of is not None else None
    return (
        "320193", "Apple Inc.", normalized, ratios, metrics, technical, flags,
        catalyst, price, submissions, price_df, fred_rate,
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
    def _raise_value_error(ticker, years, no_cache, horizon, as_of=None):
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
    def _raise_boom(ticker, years, no_cache, horizon, as_of=None):
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

    def _pipeline_with_submissions(ticker, years, no_cache, horizon, as_of=None):
        base = list(_run_full_pipeline_stub(ticker, years, no_cache, horizon, as_of))
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


# ---------------------------------------------------------------------------
# Point-in-time ("as-of") mode: POST /api/analyze accepts an optional
# "as_of" body field, 400s on invalid/future, threads it through, echoes it
# back, and suppresses the (undated) analyst-consensus cross-check.
# ---------------------------------------------------------------------------


def test_api_analyze_invalid_as_of_returns_400(monkeypatch):
    monkeypatch.setattr(web_app, "_run_full_pipeline", _run_full_pipeline_stub)
    monkeypatch.setattr(web_app, "interpret", _fake_interpret)

    client = web_app.app.test_client()
    resp = client.post("/api/analyze", json={"ticker": "AAPL", "as_of": "not-a-date"})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "as_of" in body["error"]


def test_api_analyze_future_as_of_returns_400(monkeypatch):
    monkeypatch.setattr(web_app, "_run_full_pipeline", _run_full_pipeline_stub)
    monkeypatch.setattr(web_app, "interpret", _fake_interpret)

    client = web_app.app.test_client()
    # Comfortably in the future relative to any run of this suite.
    resp = client.post("/api/analyze", json={"ticker": "AAPL", "as_of": "2999-01-01"})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "future" in body["error"]


def test_api_analyze_valid_as_of_threads_through_and_is_echoed(monkeypatch):
    captured_pipeline_args = {}
    captured_interpret_kwargs = {}
    captured_save_verdict_kwargs = {}

    def _capturing_pipeline(ticker, years, no_cache, horizon, as_of=None):
        captured_pipeline_args["as_of"] = as_of
        return _run_full_pipeline_stub(ticker, years, no_cache, horizon, as_of)

    def _capturing_interpret(*args, **kwargs):
        captured_interpret_kwargs.update(kwargs)
        return _interpret_result()

    def _capturing_save_verdict(*args, **kwargs):
        captured_save_verdict_kwargs.update(kwargs)
        return 1

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("analyst consensus must be suppressed in as-of mode")

    monkeypatch.setattr(web_app, "_run_full_pipeline", _capturing_pipeline)
    monkeypatch.setattr(web_app, "interpret", _capturing_interpret)
    monkeypatch.setattr(web_app, "save_verdict", _capturing_save_verdict)
    monkeypatch.setattr(web_app, "_fetch_analyst_targets", _fail_if_called)

    client = web_app.app.test_client()
    resp = client.post("/api/analyze", json={"ticker": "AAPL", "as_of": "2022-06-30"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["as_of"] == "2022-06-30"
    assert body["analyst"] is None

    from datetime import date
    assert captured_pipeline_args["as_of"] == date(2022, 6, 30)
    assert captured_interpret_kwargs.get("as_of") == date(2022, 6, 30)
    assert captured_interpret_kwargs.get("fred_rate") == {
        "value_pct": 2.98, "date": "2022-06-29", "series": "DGS10",
    }
    assert captured_save_verdict_kwargs.get("as_of") == "2022-06-30"


def test_api_analyze_without_as_of_still_fetches_analyst_targets(monkeypatch):
    monkeypatch.setattr(web_app, "_run_full_pipeline", _run_full_pipeline_stub)
    monkeypatch.setattr(web_app, "interpret", _fake_interpret)
    monkeypatch.setattr(web_app, "save_verdict", lambda *a, **k: 1)
    monkeypatch.setattr(web_app, "_fetch_analyst_targets", lambda ticker, no_cache: {"target": 200.0})

    client = web_app.app.test_client()
    resp = client.post("/api/analyze", json={"ticker": "AAPL"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["as_of"] is None
    assert body["analyst"] == {"target": 200.0}


# ---------------------------------------------------------------------------
# GET /history?ticker=X -- verdict-history screen (mode:"history" payload,
# no network/analysis).
# ---------------------------------------------------------------------------


def _history_rows():
    return [
        {
            "id": 1, "cik": "320193", "ticker": "AAPL",
            "analyzed_at": "2026-07-01T10:00:00", "as_of": None,
            "horizon": "1y", "provider": "script", "price": 128.4,
            "fundamental_verdict": "MAKUL", "technical_verdict": "NÖTR",
            "profile_fit": "KISMEN",
            "fv_bear_lo": 100.0, "fv_bear_hi": 110.0,
            "fv_base_lo": 120.0, "fv_base_hi": 140.0,
            "fv_bull_lo": 150.0, "fv_bull_hi": 170.0,
            "confidence": "YÜKSEK", "sector_type": "mature",
            "implied_growth": 0.12, "watch_note": None,
        },
    ]


def test_history_with_rows_renders_200(monkeypatch):
    monkeypatch.setattr(web_app, "load_verdicts", lambda ticker, db_path=None: _history_rows())
    monkeypatch.setattr(
        web_app, "load_latest_stored_price",
        lambda ticker, db_path=None: {"date": "2026-07-15", "close": 130.0},
    )

    client = web_app.app.test_client()
    resp = client.get("/history?ticker=AAPL")

    assert resp.status_code == 200
    assert resp.content_type.startswith("text/html")
    body = resp.get_data(as_text=True)
    assert "AAPL" in body
    assert "Traceback" not in body


def test_history_without_ticker_returns_400():
    client = web_app.app.test_client()
    resp = client.get("/history")

    assert resp.status_code == 400
    assert resp.content_type.startswith("text/html")
    body = resp.get_data(as_text=True)
    assert "required" in body
    assert "Traceback" not in body


def test_history_empty_rows_still_renders_200(monkeypatch):
    monkeypatch.setattr(web_app, "load_verdicts", lambda ticker, db_path=None: [])
    monkeypatch.setattr(web_app, "load_latest_stored_price", lambda ticker, db_path=None: None)

    client = web_app.app.test_client()
    resp = client.get("/history?ticker=ZZZZ")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "ZZZZ" in body
