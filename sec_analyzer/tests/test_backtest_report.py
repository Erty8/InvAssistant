"""Unit tests for ``sec_analyzer.backtest.report`` (hit-rate / calibration /
divergence tables + terminal/HTML rendering)."""

import json

import pytest

from sec_analyzer.backtest import BACKTEST_DISCLAIMER, report
from sec_analyzer.store import database


def _dcf_valuation():
    """A valuation dict whose ``calibrate._method_slug`` route is "dcf"."""
    return {"dcf": {"enabled": True, "scenarios": {"base": {"per_share": 100.0}}}}


def _save_hit_rate_verdict(db_path, ticker, fundamental_verdict, as_of="2020-06-30"):
    """A verdict with NO fair_value_range, so it can never leak into the
    calibration list -- used purely to attach a verdict_outcomes row for the
    hit-rate table."""
    result = {"fundamental_verdict": fundamental_verdict}
    return database.save_verdict(
        ticker, "1", "1y", "script", 100.0, result,
        db_path=db_path, as_of=as_of, valuation=_dcf_valuation(),
    )


def _save_outcome(db_path, verdict_id, horizon, hit, rel_return=0.1, referee_note=None):
    database.save_outcome(
        verdict_id=verdict_id, horizon=horizon,
        ref_date="2020-06-30", ref_price=100.0,
        fwd_date="2021-06-30", fwd_price=110.0,
        abs_return=0.1, rel_return=rel_return, hit=hit,
        evaluated_at="2021-07-01", referee_note=referee_note, db_path=db_path,
    )


# ---------------------------------------------------------------------------
# build_report_data -- hit-rate table
#
# Bucket A: ("UCUZ", "1y", "dcf")   -- 10 outcomes, 7 hits -> rate = 7/10 = 0.7;
#           n = 10 is NOT < 10, so insufficient=False.
# Bucket B: ("PAHALI", "1y", "dcf") -- 3 outcomes, 1 hit  -> rate = 1/3;
#           n = 3 < 10, so insufficient=True.
# ---------------------------------------------------------------------------


def test_build_report_data_hit_rate_groups_by_type_horizon_route_and_flags_small_n(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")

    for i in range(10):
        vid = _save_hit_rate_verdict(db_path, f"T{i}", "UCUZ")
        _save_outcome(db_path, vid, "1y", hit=(i < 7))  # 7 True, 3 False

    for i in range(3):
        vid = _save_hit_rate_verdict(db_path, f"P{i}", "PAHALI")
        _save_outcome(db_path, vid, "1y", hit=(i < 1))  # 1 True, 2 False

    data = report.build_report_data(db_path)

    by_key = {(r["verdict_type"], r["horizon"], r["route"]): r for r in data["hit_rate"]}

    cheap = by_key[("UCUZ", "1y", "dcf")]
    assert cheap["n"] == 10
    assert cheap["hits"] == 7
    assert cheap["rate"] == pytest.approx(0.7)
    assert cheap["insufficient"] is False

    expensive = by_key[("PAHALI", "1y", "dcf")]
    assert expensive["n"] == 3
    assert expensive["hits"] == 1
    assert expensive["rate"] == pytest.approx(1 / 3)
    assert expensive["insufficient"] is True


def test_build_report_data_hit_rate_excludes_outcomes_with_no_binary_hit(tmp_path):
    """A MAKUL (neutral) or referee-labeled outcome has hit=None and must not
    appear in any hit-rate bucket at all."""
    db_path = str(tmp_path / "test.sqlite3")
    vid = _save_hit_rate_verdict(db_path, "X", "MAKUL")
    _save_outcome(db_path, vid, "1y", hit=None)

    data = report.build_report_data(db_path)

    assert data["hit_rate"] == []


# ---------------------------------------------------------------------------
# build_report_data -- calibration time series.
#
# Hand-verified median: ratios = mid(base)/price for 3 verdicts sharing the
# same reference date:
#   A: (70+90)/2   / 100 = 0.80
#   B: (90+110)/2  / 100 = 1.00
#   C: (110+130)/2 / 100 = 1.20
# median([0.80, 1.00, 1.20]) = 1.00
# ---------------------------------------------------------------------------


def _save_calibration_verdict(db_path, ticker, lo, hi, price=100.0, as_of="2021-01-01"):
    result = {
        "fundamental_verdict": "MAKUL",
        "fair_value_range": {"bear": {}, "base": {"lo": lo, "hi": hi}, "bull": {}},
    }
    return database.save_verdict(
        ticker, "1", "1y", "script", price, result, db_path=db_path, as_of=as_of,
    )


def test_build_report_data_calibration_median_hand_verified(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    _save_calibration_verdict(db_path, "A", 70.0, 90.0)
    _save_calibration_verdict(db_path, "B", 90.0, 110.0)
    _save_calibration_verdict(db_path, "C", 110.0, 130.0)

    data = report.build_report_data(db_path)

    assert len(data["calibration"]) == 1
    entry = data["calibration"][0]
    assert entry["date"] == "2021-01-01"
    assert entry["n"] == 3
    assert entry["median_ratio"] == pytest.approx(1.00)


def test_build_report_data_calibration_skips_verdicts_missing_price_or_range(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    # No fair_value_range at all -> lo/hi are None -> excluded.
    _save_hit_rate_verdict(db_path, "NOFV", "MAKUL")

    data = report.build_report_data(db_path)

    assert data["calibration"] == []


# ---------------------------------------------------------------------------
# build_report_data -- divergence / referee cases.
# ---------------------------------------------------------------------------


def test_build_report_data_divergence_picks_up_model_market_divergence_verdict(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    vid = database.save_verdict(
        "TSLA", "1", "1y", "script", 100.0,
        {"fundamental_verdict": "MODEL-PİYASA AYRIŞMASI"},
        db_path=db_path, as_of="2021-03-01",
    )
    _save_outcome(db_path, vid, "1y", hit=None, rel_return=0.5, referee_note="hakem notu")

    data = report.build_report_data(db_path)

    assert len(data["divergence"]) == 1
    case = data["divergence"][0]
    assert case["ticker"] == "TSLA"
    assert case["ref_date"] == "2021-03-01"
    assert case["fundamental_verdict"] == "MODEL-PİYASA AYRIŞMASI"
    assert case["horizon"] == "1y"
    assert case["rel_return"] == pytest.approx(0.5)
    assert case["referee_note"] == "hakem notu"


def test_build_report_data_divergence_excludes_ordinary_verdicts(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    vid = _save_hit_rate_verdict(db_path, "AAPL", "UCUZ")
    _save_outcome(db_path, vid, "1y", hit=True)

    data = report.build_report_data(db_path)

    assert data["divergence"] == []


def test_build_report_data_always_carries_the_disclaimer(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    data = report.build_report_data(db_path)
    assert data["disclaimer"] == BACKTEST_DISCLAIMER


# ---------------------------------------------------------------------------
# _route_of -- valuation_json -> calibrate._method_slug route.
# ---------------------------------------------------------------------------


def test_route_of_parses_valuation_json_via_method_slug():
    valuation_json = json.dumps({"dcf": {"enabled": True}})
    assert report._route_of(valuation_json) == "dcf"


def test_route_of_returns_dash_for_none_or_malformed_json():
    assert report._route_of(None) == "—"
    assert report._route_of("not json at all") == "—"


# ---------------------------------------------------------------------------
# render_terminal / render_html
# ---------------------------------------------------------------------------


def test_render_terminal_contains_disclaimer_and_section_headers(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    data = report.build_report_data(db_path)  # empty DB -> empty lists, still renders headers

    text = report.render_terminal(data)

    assert BACKTEST_DISCLAIMER in text
    assert "Hit-rate" in text
    assert "Kalibrasyon" in text
    assert "Ayrışma" in text


def test_render_html_is_a_full_document_and_contains_disclaimer(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    data = report.build_report_data(db_path)

    html = report.render_html(data, "2026-07-21")

    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert BACKTEST_DISCLAIMER in html
    assert "2026-07-21" in html
