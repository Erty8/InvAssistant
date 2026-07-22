"""Tests for the momentum context layer: composite price momentum, the
quarterly-series reconstruction, fundamental/verdict momentum, the synthesis
and cross-signals, and the entry-plan stabilization note.

All deterministic and network-free -- momentum is a pure, no-LLM context
signal. Store-backed tests use a temporary SQLite db_path.
"""

import json

import pytest

from sec_analyzer.technical.momentum import (
    compute_price_momentum,
    sector_etf_for_sic,
)
from sec_analyzer.normalize.normalizer import to_quarterly_series
from sec_analyzer.signals.momentum import (
    compute_fundamental_momentum,
    compute_verdict_momentum,
    synthesize_momentum,
    _yoy_growth_series,
    _classify_accel,
)
from sec_analyzer.interpret.planning import apply_stabilization_condition
from sec_analyzer.store.database import (
    save_verdict,
    load_verdicts,
    load_prior_live_verdict,
)


# --------------------------------------------------------------------------
# compute_price_momentum
# --------------------------------------------------------------------------
def _strong_up_indicators(**overrides):
    ind = {
        "return_3m_pct": 25.0, "return_6m_pct": 40.0, "mom_12_1_pct": 60.0,
        "return_1m_pct": 12.0, "volatility_20d": 0.35,
        "sma50_slope_pct": 6.0, "sma200_slope_pct": 4.0, "dist_52w_high_pct": -3.0,
        "updown_volume_ratio": 1.8, "rsi14": 68.0, "macd_hist": 0.5, "macd_cross": None,
        "relative_strength": {"rs_3m_pct": 15.0},
    }
    ind.update(overrides)
    return ind


def test_price_momentum_strong_up_is_high_and_labeled():
    m = compute_price_momentum(_strong_up_indicators())
    assert 70 <= m["score"] <= 100
    assert m["direction"] == "up"
    assert m["label"] in ("YUKARI MOMENTUM", "GÜÇLÜ YUKARI MOMENTUM")
    assert m["accel"] in ("hızlanıyor", "sabit", "yavaşlıyor")


def test_price_momentum_strong_down_is_low():
    ind = _strong_up_indicators(
        return_3m_pct=-25.0, return_6m_pct=-40.0, mom_12_1_pct=-50.0, return_1m_pct=-15.0,
        sma50_slope_pct=-6.0, sma200_slope_pct=-4.0, dist_52w_high_pct=-40.0,
        updown_volume_ratio=0.5, rsi14=32.0, macd_hist=-0.5,
        relative_strength={"rs_3m_pct": -15.0},
    )
    m = compute_price_momentum(ind)
    assert m["score"] < 30
    assert m["direction"] == "down"
    assert m["label"] in ("AŞAĞI MOMENTUM", "GÜÇLÜ AŞAĞI MOMENTUM")


def test_price_momentum_deterministic_and_json_native():
    ind = _strong_up_indicators()
    a = compute_price_momentum(ind)
    b = compute_price_momentum(ind)
    assert a == b
    json.dumps(a)  # must not raise -- all JSON-native
    assert isinstance(a["score"], int)


def test_price_momentum_none_when_no_components():
    assert compute_price_momentum({}) is None
    assert compute_price_momentum(None) is None


def test_price_momentum_renormalizes_over_available_components():
    # Only a single returns figure available -> still produces a score,
    # weights renormalized so the one present component carries it.
    m = compute_price_momentum({"return_3m_pct": 30.0})
    assert m is not None
    assert m["direction"] == "up"
    # Exactly one contributing component.
    assert len(m["components"]) == 1
    assert m["components"][0]["key"] == "returns"


def test_price_momentum_flat_band():
    m = compute_price_momentum({"return_3m_pct": 0.0, "return_6m_pct": 0.0})
    assert m["direction"] == "flat"
    assert m["label"] == "YATAY MOMENTUM"
    assert m["score"] == 50


def test_price_momentum_points_sum_to_score_minus_50():
    m = compute_price_momentum(_strong_up_indicators())
    total_points = sum(c["points"] for c in m["components"])
    assert m["score"] == pytest.approx(50 + total_points, abs=1.0)


# --------------------------------------------------------------------------
# sector_etf_for_sic
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sic,expected", [
    (3674, "SMH"),       # semiconductors (single)
    ("3674", "SMH"),     # string form
    (7372, "XLK"),       # prepackaged software
    (2911, "XLE"),       # petroleum refining (single)
    (6798, "XLRE"),      # REIT (single)
    (6021, "XLF"),       # national commercial banks
    (2834, "XLV"),       # pharmaceutical preparations
    (4911, "XLU"),       # electric services
    (9999, None),        # unmapped
    (None, None),
    ("abc", None),
])
def test_sector_etf_for_sic(sic, expected):
    assert sector_etf_for_sic(sic) == expected


# --------------------------------------------------------------------------
# to_quarterly_series
# --------------------------------------------------------------------------
def _q(pe, val, start):
    return {"period_end": pe, "value": val, "start": start, "fy": int(pe[:4]), "form": "10-Q"}


def test_quarterly_series_quarter_only_and_q4_derivation():
    norm = {
        "quarterly": {"Revenue": [
            _q("2024-03-31", 100, "2024-01-01"),
            _q("2024-06-30", 110, "2024-04-01"),
            _q("2024-09-30", 120, "2024-07-01"),
        ]},
        "annual": {"Revenue": [{"fy": 2024, "period_end": "2024-12-31", "value": 460}]},
    }
    s = to_quarterly_series(norm, "Revenue")
    values = [round(x["value"], 1) for x in s]
    assert values == [100, 110, 120, 130]        # Q4 = 460 - 330
    assert s[-1]["derived"] is True
    assert all(x["derived"] is False for x in s[:3])


def test_quarterly_series_ytd_differencing():
    norm = {
        "quarterly": {"OperatingCashFlow": [
            {"period_end": "2024-03-31", "value": 50, "start": "2024-01-01", "fy": 2024, "form": "10-Q"},
            {"period_end": "2024-06-30", "value": 120, "start": "2024-01-01", "fy": 2024, "form": "10-Q"},
            {"period_end": "2024-09-30", "value": 200, "start": "2024-01-01", "fy": 2024, "form": "10-Q"},
        ]},
        "annual": {},
    }
    s = to_quarterly_series(norm, "OperatingCashFlow")
    values = [round(x["value"], 1) for x in s]
    assert values == [50, 70, 80]                # 50, 120-50, 200-120
    assert s[0]["derived"] is False and s[1]["derived"] is True


def test_quarterly_series_empty_when_missing():
    assert to_quarterly_series({"quarterly": {}, "annual": {}}, "Revenue") == []
    assert to_quarterly_series({}, "Revenue") == []


# --------------------------------------------------------------------------
# YoY growth + acceleration
# --------------------------------------------------------------------------
def _build_revenue(quarters_by_year):
    # Per-quarter start dates (~90-day spans) so each row reads as quarter-only
    # and to_quarterly_series takes the values as-is (no YTD differencing).
    starts = {"03-31": "01-01", "06-30": "04-01", "09-30": "07-01", "12-31": "10-01"}
    rev = []
    for yr, vals in quarters_by_year.items():
        for m, v in zip(("03-31", "06-30", "09-30", "12-31"), vals):
            rev.append(_q(f"{yr}-{m}", v, f"{yr}-{starts[m]}"))
    rev.sort(key=lambda x: x["period_end"])
    return rev


def test_yoy_and_acceleration_direction():
    rev = _build_revenue({
        2023: [100, 100, 100, 100],
        2024: [110, 113, 117, 120],
        2025: [143, 150, 160, 170],
    })
    yoy = _yoy_growth_series([{"period_end": r["period_end"], "value": r["value"]} for r in rev])
    ys = [y["yoy_pct"] for y in yoy]
    # YoY should be increasing (accelerating).
    assert ys[-1] > ys[0]
    m = compute_fundamental_momentum({"quarterly": {"Revenue": rev}, "annual": {}})
    assert m["revenue_accel"]["word"] == "hızlanıyor"
    assert m["label"] == "POZİTİF"
    # Continuous score exposed for the quadrant, consistent with the label.
    assert -1.0 <= m["s"] <= 1.0 and m["s"] > 0
    assert m["score"] == pytest.approx(50 + m["s"] * 50, abs=1.0)
    # The detail must mark the growth read as QUARTERLY, so it never reads as
    # contradicting the thesis card's annual "Yıllık Gelir Büyümesi (YoY)".
    assert "Çeyreklik gelir büyümesi" in m["detail"]
    assert "son çeyrek YoY" in m["detail"]


def test_fundamental_none_when_sparse():
    assert compute_fundamental_momentum({"quarterly": {}, "annual": {}}) is None
    assert compute_fundamental_momentum(None) is None


@pytest.mark.parametrize("series,expected", [
    ([10.0, 13.0, 17.0, 20.0], (1, True)),      # clean accel -> confirmed
    ([40.0, 30.0, 22.0, 15.0], (-1, True)),     # clean decel -> confirmed
    ([36.0, 48.0, 35.7, 63.5], (1, False)),     # RKLB: up but single-quarter -> unconfirmed
    ([36.0, 48.0, 63.5, 35.7], (0, False)),     # mean-up but latest crashed -> neutralized
    ([20.0, 21.0, 20.0, 21.0], (0, False)),     # flat
])
def test_classify_accel_guards(series, expected):
    assert _classify_accel(series) == expected


def test_single_quarter_accel_is_flagged_and_half_weighted():
    # A single-quarter acceleration (RKLB-shaped) is marked unconfirmed and
    # carries half weight, so it can't alone push the label to POZİTİF.
    rev = _build_revenue({
        2023: [100, 100, 100, 100],
        2024: [136, 148, 135, 163],   # noisy YoY, last quarter jumps
    })
    m = compute_fundamental_momentum({"quarterly": {"Revenue": rev}, "annual": {}})
    assert m["revenue_accel"]["word"] == "hızlanıyor"
    assert m["revenue_accel"]["confirmed"] is False
    assert "teyit bekliyor" in m["detail"]


# --------------------------------------------------------------------------
# model-based surprise
# --------------------------------------------------------------------------
def _prior_verdict_with_path(base_revenue, path, ref_date):
    valuation = {"hyper_growth_detail": {"scenarios": {"base": {
        "base_revenue": base_revenue, "revenue_path": path, "steady_state_year": 8,
    }}}}
    return {"valuation": valuation, "ref_date": ref_date}


def test_model_surprise_beat_and_miss():
    # Prior base projected year1=110, year2=121 from base 100 at 2024-01-01.
    prior = _prior_verdict_with_path(100.0, [110.0, 121.0, 133.0], "2024-01-01")
    # ~1 year later, actual TTM = 130 (4 quarters of ~32.5) -> beat vs implied ~110.
    rev = [_q("2024-12-31", 32.5, "2024-10-01"), _q("2024-09-30", 32.5, "2024-07-01"),
           _q("2024-06-30", 32.5, "2024-04-01"), _q("2024-03-31", 32.5, "2024-01-01")]
    norm = {"quarterly": {"Revenue": rev}, "annual": {}}
    m = compute_fundamental_momentum(norm, prior)
    assert m["model_surprise"] is not None
    assert m["model_surprise"]["direction"] == "beat"
    assert m["model_surprise"]["surprise_pct"] > 0


def test_model_surprise_none_for_legacy_blob_without_path():
    # Old stored verdict lacking revenue_path -> surprise degrades to None.
    prior = {"valuation": {"hyper_growth_detail": {"scenarios": {"base": {}}}}, "ref_date": "2024-01-01"}
    rev = _build_revenue({2023: [100, 100, 100, 100], 2024: [120, 120, 120, 120]})
    norm = {"quarterly": {"Revenue": rev}, "annual": {}}
    m = compute_fundamental_momentum(norm, prior)
    assert m is not None
    assert m["model_surprise"] is None


# --------------------------------------------------------------------------
# verdict momentum
# --------------------------------------------------------------------------
def test_verdict_momentum_opportunity_when_ratio_rises_price_falls():
    hist = [
        {"analyzed_at": "2025-07-01", "price": 80, "fv_base_lo": 118, "fv_base_hi": 138},
        {"analyzed_at": "2025-04-01", "price": 90, "fv_base_lo": 115, "fv_base_hi": 135},
        {"analyzed_at": "2025-01-01", "price": 100, "fv_base_lo": 110, "fv_base_hi": 130},
    ]
    vm = compute_verdict_momentum(hist)
    assert vm["direction"] == "up"
    assert vm["label"] == "POZİTİF"
    # series ascending by date
    assert vm["series"][0]["date"] < vm["series"][-1]["date"]


def test_verdict_momentum_weakening_when_fv_erodes():
    hist = [
        {"analyzed_at": "2025-01-01", "price": 100, "fv_base_lo": 130, "fv_base_hi": 150},
        {"analyzed_at": "2025-07-01", "price": 100, "fv_base_lo": 95, "fv_base_hi": 110},
    ]
    vm = compute_verdict_momentum(hist)
    assert vm["direction"] == "down"
    assert vm["label"] == "NEGATİF"


def test_verdict_momentum_none_with_one_point():
    assert compute_verdict_momentum([{"analyzed_at": "2025-01-01", "price": 100,
                                       "fv_base_lo": 110, "fv_base_hi": 130}]) is None
    assert compute_verdict_momentum([]) is None


# --------------------------------------------------------------------------
# synthesis + cross-signals
# --------------------------------------------------------------------------
def test_synthesis_falling_knife_when_cheap_and_down():
    price_m = {"label": "AŞAĞI MOMENTUM", "direction": "down", "score": 30}
    syn = synthesize_momentum(price_m, {"label": "NÖTR", "model_surprise": None}, None, "UCUZ")
    assert syn["falling_knife"] is True
    assert any(c["type"] == "falling_knife" for c in syn["cross_signals"])
    assert syn["verdict"] in ("NEGATİF", "NÖTR")


def test_synthesis_profile_guardrail_when_expensive_and_strong():
    price_m = {"label": "GÜÇLÜ YUKARI MOMENTUM", "direction": "up", "score": 90}
    syn = synthesize_momentum(price_m, {"label": "POZİTİF", "model_surprise": {"direction": "beat"}}, None, "PAHALI")
    assert syn["verdict"] == "GÜÇLÜ+"
    assert any(c["type"] == "profile_guardrail" for c in syn["cross_signals"])


def test_synthesis_strong_combo_when_cheap_and_positive():
    price_m = {"label": "YUKARI MOMENTUM", "direction": "up", "score": 65}
    syn = synthesize_momentum(price_m, {"label": "POZİTİF", "model_surprise": {"direction": "beat"}}, None, "UCUZ")
    types = [c["type"] for c in syn["cross_signals"]]
    assert "strong_combo" in types


def test_synthesis_none_when_no_layers():
    assert synthesize_momentum(None, None, None, "UCUZ") is None


def test_momentum_functions_do_not_mutate_inputs():
    # Invariant guard: momentum is a read-only context signal. The synthesis
    # must not mutate the price/fundamental/verdict dicts it is handed (they are
    # shared with the technical payload / valuation-adjacent structures).
    price_m = {"label": "YUKARI MOMENTUM", "direction": "up", "score": 65}
    fund_m = {"label": "POZİTİF", "model_surprise": {"direction": "beat"}}
    price_copy, fund_copy = json.loads(json.dumps(price_m)), json.loads(json.dumps(fund_m))
    synthesize_momentum(price_m, fund_m, None, "UCUZ")
    assert price_m == price_copy and fund_m == fund_copy


# --------------------------------------------------------------------------
# entry-plan stabilization note
# --------------------------------------------------------------------------
def test_stabilization_appends_to_dip_only():
    plan = [{"n": 1, "kind": "dip", "note": None}, {"n": 2, "kind": "breakout", "note": "x"}]
    apply_stabilization_condition(plan, active=True)
    assert "Stabilizasyon" in plan[0]["note"]
    assert plan[1]["note"] == "x"   # breakout untouched


def test_stabilization_noop_when_inactive():
    plan = [{"n": 1, "kind": "dip", "note": None}]
    apply_stabilization_condition(plan, active=False)
    assert plan[0]["note"] is None


def test_stabilization_appends_not_overwrites_existing_note():
    plan = [{"n": 1, "kind": "dip", "note": "R:R sırası ters"}]
    apply_stabilization_condition(plan, active=True)
    assert plan[0]["note"].startswith("R:R sırası ters")
    assert "Stabilizasyon" in plan[0]["note"]


# --------------------------------------------------------------------------
# store: momentum_verdict column, live_only, load_prior_live_verdict
# --------------------------------------------------------------------------
def _min_result(momentum_verdict=None, fv=(90, 110)):
    result = {
        "fundamental_verdict": "UCUZ",
        "fair_value_range": {"base": {"lo": fv[0], "hi": fv[1]}},
    }
    if momentum_verdict is not None:
        result["momentum"] = {"verdict": momentum_verdict}
    return result


def test_store_persists_momentum_verdict_and_roundtrips(tmp_path):
    db = str(tmp_path / "t.db")
    save_verdict("AAPL", 320193, "1y", "script", 100.0, _min_result("POZİTİF"),
                 db_path=db, analyzed_at="2025-01-01T00:00:00")
    rows = load_verdicts("AAPL", db_path=db)
    assert rows and rows[0]["momentum_verdict"] == "POZİTİF"


def test_store_live_only_excludes_asof(tmp_path):
    db = str(tmp_path / "t.db")
    save_verdict("AAPL", 320193, "1y", "script", 100.0, _min_result("POZİTİF"),
                 db_path=db, analyzed_at="2025-01-01T00:00:00")
    save_verdict("AAPL", 320193, "1y", "script", 95.0, _min_result("NÖTR"),
                 db_path=db, analyzed_at="2025-02-01T00:00:00", as_of="2020-01-01")
    all_rows = load_verdicts("AAPL", db_path=db)
    live_rows = load_verdicts("AAPL", db_path=db, live_only=True)
    assert len(all_rows) == 2
    assert len(live_rows) == 1
    assert live_rows[0]["as_of"] is None


def test_load_prior_live_verdict_returns_latest_with_valuation(tmp_path):
    db = str(tmp_path / "t.db")
    valuation = {"sector_type": "growth_unprofitable",
                 "hyper_growth_detail": {"scenarios": {"base": {"base_revenue": 100.0, "revenue_path": [110.0]}}}}
    result = _min_result("POZİTİF")
    save_verdict("NVDA", 1045810, "1y", "script", 100.0, result,
                 db_path=db, analyzed_at="2025-01-01T00:00:00", valuation=valuation)
    prior = load_prior_live_verdict("NVDA", db_path=db)
    assert prior is not None
    assert prior["valuation"]["hyper_growth_detail"]["scenarios"]["base"]["base_revenue"] == 100.0


def test_load_prior_live_verdict_respects_before_and_skips_asof(tmp_path):
    db = str(tmp_path / "t.db")
    save_verdict("NVDA", 1045810, "1y", "script", 100.0, _min_result("POZİTİF"),
                 db_path=db, analyzed_at="2025-01-01T00:00:00", valuation={"a": 1})
    # An as-of run must be ignored even though it's newer.
    save_verdict("NVDA", 1045810, "1y", "script", 90.0, _min_result("NÖTR"),
                 db_path=db, analyzed_at="2025-03-01T00:00:00", as_of="2019-01-01", valuation={"a": 2})
    prior = load_prior_live_verdict("NVDA", before="2025-06-01T00:00:00", db_path=db)
    assert prior["analyzed_at"] == "2025-01-01T00:00:00"
    # Nothing strictly before the first live run.
    assert load_prior_live_verdict("NVDA", before="2024-01-01T00:00:00", db_path=db) is None
