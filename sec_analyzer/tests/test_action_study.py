"""Tests for sec_analyzer.backtest.action_study (ACTION_STUDY_SPEC.md).

Network-free: the price layer is monkeypatched onto both action_study (ticker
frames) and swing_study (the shared SPY benchmark cache). End-to-end runs
force the thread-pool fallback so the patches are visible to workers.
"""

import concurrent.futures
import json
import math

import pandas as pd
import pytest

from sec_analyzer.backtest import BACKTEST_DISCLAIMER, action_study, swing_study
from sec_analyzer.fetch.prices import slice_asof
from sec_analyzer.technical.action import compute_technical_action
from sec_analyzer.technical.indicators import compute_indicators, relative_strength
from sec_analyzer.technical.momentum import compute_price_momentum


@pytest.fixture(autouse=True)
def _reset_benchmark_cache():
    swing_study._benchmark_cache.clear()
    yield
    swing_study._benchmark_cache.clear()


@pytest.fixture
def force_thread_pool(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise OSError("process pools disabled for this test")
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _Boom)


def _df(closes, start="2021-01-04"):
    idx = pd.bdate_range(start=start, periods=len(closes))
    idx.name = "Date"
    return pd.DataFrame({"Open": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
                         "Close": closes, "Volume": [1_000_000 + (i % 7) * 50_000 for i in range(len(closes))]},
                        index=idx)


def _wave(n=500, base=100.0, drift=0.05, amp=8.0):
    return [base + drift * i + amp * math.sin(i / 25.0) for i in range(n)]


def _patch_prices(monkeypatch, frames):
    def fake(symbol, *a, **k):
        if symbol not in frames:
            raise RuntimeError("no data")
        return frames[symbol], "stub"
    monkeypatch.setattr(action_study, "get_price_history", fake)
    monkeypatch.setattr(swing_study, "get_price_history", fake)


def test_point_in_time_future_outliers_do_not_change_the_action(monkeypatch):
    closes = _wave()
    d_pos = 400
    spoiled = closes[: d_pos + 1] + [c * 50 for c in closes[d_pos + 1:]]
    df_clean, df_spoiled = _df(closes), _df(spoiled)
    spy = _df(_wave(drift=0.02, amp=3.0))
    d = df_clean.index[d_pos].date()

    _patch_prices(monkeypatch, {"AAA": df_spoiled, "SPY": spy})
    r = action_study._action_ticker_task({"ticker": "AAA", "dates": [d.isoformat()]})
    assert r["status"] == "ok" and r["observations"]

    sliced = slice_asof(df_clean, d)
    ind = compute_indicators(sliced)
    ind["relative_strength"] = relative_strength(sliced["Close"], slice_asof(spy, d)["Close"], benchmark="SPY")
    ind["momentum"] = compute_price_momentum(ind)
    direct = compute_technical_action(ind, "1y")
    assert r["observations"][0]["action"] == direct["action"]
    assert r["observations"][0]["regime"] == direct["regime"]


def test_forward_window_starts_at_next_bar(monkeypatch):
    closes = _wave()
    d_pos = 300
    closes[d_pos + 1] = closes[d_pos] * 2  # jump on d+1 must NOT count
    df = _df(closes)
    _patch_prices(monkeypatch, {"AAA": df, "SPY": _df(_wave(drift=0.02, amp=3.0))})
    d = df.index[d_pos].date().isoformat()
    obs = action_study._action_ticker_task({"ticker": "AAA", "dates": [d]})["observations"][0]
    expected = (closes[d_pos + 1 + 10] / closes[d_pos + 1] - 1) * 100
    assert obs["fwd_10"] == pytest.approx(expected, abs=1e-3)


def test_missing_forward_bars_are_dropped_and_counted(monkeypatch):
    df = _df(_wave(n=320))
    _patch_prices(monkeypatch, {"AAA": df, "SPY": _df(_wave(n=320, drift=0.02))})
    d = df.index[-15].date().isoformat()  # 10d ok, 21d/63d missing
    r = action_study._action_ticker_task({"ticker": "AAA", "dates": [d]})
    assert r["dropped_no_forward"] == 2
    assert r["observations"][0]["fwd_10"] is not None
    assert r["observations"][0]["fwd_63"] is None


def test_demeaning_is_per_date():
    obs = [{"date": "d1", "fwd_10": 10.0, "fwd_21": None, "fwd_63": None},
           {"date": "d1", "fwd_10": 0.0, "fwd_21": None, "fwd_63": None},
           {"date": "d2", "fwd_10": 100.0, "fwd_21": None, "fwd_63": None}]
    action_study._add_demeaned(obs)
    assert [o["rel_10"] for o in obs] == [5.0, -5.0, 0.0]
    assert obs[0]["rel_21"] is None


def test_contrast_bootstrap_is_deterministic_and_brackets_diff():
    obs = []
    for i in range(40):
        obs.append({"date": f"d{i}", "g": "A", "rel_21": 2.0 + (i % 5) * 0.1})
        obs.append({"date": f"d{i}", "g": "B", "rel_21": -1.0 + (i % 3) * 0.1})
    pa, pb = (lambda o: o["g"] == "A"), (lambda o: o["g"] == "B")
    c1 = action_study._contrast(obs, pa, pb, "rel_21")
    c2 = action_study._contrast(obs, pa, pb, "rel_21")
    assert c1 == c2
    assert c1["ci_low"] <= c1["diff"] <= c1["ci_high"]
    assert c1["ci_low"] > 0


def _by_h(diff21, lo, hi, d10, d63, n=100):
    return {"10": {"diff": d10, "n_a": n, "n_b": n, "ci_low": None, "ci_high": None},
            "21": {"diff": diff21, "n_a": n, "n_b": n, "ci_low": lo, "ci_high": hi},
            "63": {"diff": d63, "n_a": n, "n_b": n, "ci_low": None, "ci_high": None}}


def test_verdict_rules():
    assert action_study._verdict(_by_h(1.0, 0.2, 1.8, 0.5, 2.0), +1) == "supported"
    assert action_study._verdict(_by_h(1.0, 0.2, 1.8, -0.5, 2.0), +1) == "inconclusive"  # 10d disagrees
    assert action_study._verdict(_by_h(-1.0, -1.8, -0.2, -1, -1), +1) == "contradicted"
    assert action_study._verdict(_by_h(0.3, -0.2, 0.8, 0.1, 0.1), +1) == "inconclusive"
    assert action_study._verdict(_by_h(-1.0, -1.8, -0.2, -0.5, -2.0), -1) == "supported"
    assert action_study._verdict(_by_h(1.0, 0.2, 1.8, 0.5, 2.0, n=10), +1) == "inconclusive"


def test_checkpoint_resume_matches_fresh_run_and_skips_done_tickers(monkeypatch, force_thread_pool, tmp_path):
    frames = {"AAA": _df(_wave(drift=0.08)), "BBB": _df(_wave(drift=-0.05, amp=10)),
              "SPY": _df(_wave(drift=0.02, amp=3.0))}
    _patch_prices(monkeypatch, frames)
    ckpt = str(tmp_path / "ckpt.jsonl")
    kw = dict(tickers=["AAA", "BBB"], start="2021-12-01", max_workers=2)
    fresh = action_study.run_action_study(**kw)
    first = action_study.run_action_study(checkpoint_path=ckpt, **kw)

    calls = []
    real = action_study._action_ticker_task
    monkeypatch.setattr(action_study, "_action_ticker_task", lambda task: calls.append(task["ticker"]) or real(task))
    resumed = action_study.run_action_study(checkpoint_path=ckpt, **kw)
    assert calls == []  # everything came from the checkpoint
    for r in (fresh, first, resumed):
        r.pop("generated_at")
    assert fresh == first == resumed


def test_end_to_end_shape_json_and_determinism(monkeypatch, force_thread_pool):
    frames = {
        "AAA": _df(_wave(drift=0.08)), "BBB": _df(_wave(drift=-0.05, amp=10)),
        "CCC": _df(_wave(drift=0.0, amp=15)), "SPY": _df(_wave(drift=0.02, amp=3.0)),
    }
    _patch_prices(monkeypatch, frames)
    kw = dict(tickers=["AAA", "BBB", "CCC", "ZZZ"], start="2021-12-01", max_workers=2)
    r1 = action_study.run_action_study(**kw)
    r2 = action_study.run_action_study(**kw)
    r1.pop("generated_at"); r2.pop("generated_at")  # noqa: E702
    assert r1 == r2
    json.loads(json.dumps(r1))
    assert r1["observations"] > 0
    assert [s["ticker"] for s in r1["skipped"]] == ["ZZZ"]
    assert set(r1["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5"}
    for h in r1["hypotheses"].values():
        assert set(h["by_horizon"]) == {"10", "21", "63"}
        assert h["verdict"] in {"supported", "contradicted", "inconclusive"}
    assert any("Survivorship" in s for s in r1["limitations"])
    assert r1["disclaimer"] == BACKTEST_DISCLAIMER
