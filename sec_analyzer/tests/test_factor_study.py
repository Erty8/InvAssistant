"""Tests for sec_analyzer.backtest.factor_study (FACTOR_STUDY_SPEC.md).

Network-free: the price layer is monkeypatched. The invariants most likely
to silently invalidate the study get tests built to fail if the bug returns:
future leakage, same-bar forward measurement, and the quintile direction.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from sec_analyzer.backtest import BACKTEST_DISCLAIMER, factor_study, swing_study


@pytest.fixture(autouse=True)
def _reset_benchmark_cache():
    swing_study._benchmark_cache.clear()
    yield
    swing_study._benchmark_cache.clear()


def _series(values, start="2015-01-02"):
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_features_use_only_past_bars():
    base = [100 + 0.1 * i + 5 * math.sin(i / 17) for i in range(600)]
    spoiled = base[:400] + [v * 30 for v in base[400:]]
    a = factor_study._ticker_features(_series(base)).iloc[399]
    b = factor_study._ticker_features(_series(spoiled)).iloc[399]
    for col in ("mom_12_1", "ret_1m", "high52", "vol_63"):
        assert a[col] == pytest.approx(b[col])


def test_feature_definitions_and_forward_from_next_bar():
    closes = [100.0 + i for i in range(400)]
    closes[301] = closes[300] * 2  # a jump on bar d+1 must not count
    f = factor_study._ticker_features(_series(closes)).iloc[300]
    assert f["mom_12_1"] == pytest.approx(closes[279] / closes[48] - 1)
    assert f["ret_1m"] == pytest.approx(closes[300] / closes[279] - 1)
    assert f["high52"] == pytest.approx(closes[300] / max(closes[49:301]))
    assert f["fwd_21"] == pytest.approx((closes[322] / closes[301] - 1) * 100)


def test_quintiles_put_predicted_best_in_q5_and_break_ties_by_ticker():
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-31")] * 10,
        "ticker": list("JIHGFEDCBA"),
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
    })
    hi = factor_study._quintiles(panel, "x", +1)
    lo = factor_study._quintiles(panel, "x", -1)
    assert list(hi[panel["x"] == 1]) == [1] and list(lo[panel["x"] == 1]) == [5]
    # Tie at x=9 between A and B: A sorts first, so B gets the higher rank.
    assert hi[panel["ticker"] == "B"].iloc[0] == 5
    assert hi.tolist() == factor_study._quintiles(panel, "x", +1).tolist()


def test_block_bootstrap_deterministic_and_brackets_mean():
    s = [0.5 + 0.3 * math.sin(i) for i in range(120)]
    a = factor_study._block_bootstrap_ci(s, 3)
    assert a == factor_study._block_bootstrap_ci(s, 3)
    assert a[0] < statistics_mean(s) < a[1]
    assert a[0] > 0


def statistics_mean(v):
    return sum(v) / len(v)


def test_verdict_rules():
    m = lambda lo, hi, e, l: {"ci_low": lo, "ci_high": hi, "sub_2005_2014": e, "sub_2015_end": l}  # noqa: E731
    assert factor_study._verdict(m(0.1, 0.9, 0.4, 0.5)) == "supported"
    assert factor_study._verdict(m(0.1, 0.9, 0.4, -0.1)) == "inconclusive"   # a subperiod flips
    assert factor_study._verdict(m(-0.9, -0.1, -0.4, -0.5)) == "contradicted"
    assert factor_study._verdict(m(-0.2, 0.6, 0.4, 0.5)) == "inconclusive"
    assert factor_study._verdict(m(None, None, None, None)) == "inconclusive"


def test_turnover_cost():
    d1, d2 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")
    panel = pd.DataFrame({"date": [d1] * 5 + [d2] * 5, "ticker": list("ABCDE") + list("ABCDF")})
    q = pd.Series([5, 1, 3, 3, 3, 5, 3, 3, 3, 1])  # best stays A; worst B -> F
    cost = factor_study._turnover_cost(panel, q)
    assert cost == pytest.approx((0 + 1) * 2 * 10 / 100)


def _frame(close):
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6})


def test_pure_noise_is_not_supported(monkeypatch):
    """Null control: driftless random walks carry no factor effect, so no
    factor may come out 'supported'."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2005-01-03", periods=4000)
    frames = {f"N{k:03d}": (_frame(pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 4000))), index=idx)),
                            "yfinance") for k in range(160)}
    frames["SPY"] = (_frame(pd.Series(np.linspace(100, 300, 4000), index=idx)), "yfinance")
    monkeypatch.setattr(factor_study, "get_price_history", lambda s, *a, **k: frames[s])
    monkeypatch.setattr(swing_study, "get_price_history", lambda s, *a, **k: frames[s])
    r = factor_study.run_factor_study(start="2006-01-01", tickers=[t for t in frames if t != "SPY"])
    verdicts = {fid: f["verdict"] for fid, f in r["confirmation"]["factors"].items()}
    assert "supported" not in verdicts.values(), verdicts


def test_pit_uses_membership_on_each_date_and_reports_coverage(monkeypatch):
    from datetime import date as _date
    from sec_analyzer.fetch.sp500_history import MembershipHistory

    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2005-01-03", periods=4000)
    cache = {f"P{k:03d}": (_frame(pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 4000))), index=idx)),
                           "cache(yfinance)") for k in range(160)}
    cache["SPY"] = (_frame(pd.Series(np.linspace(100, 300, 4000), index=idx)), "cache(yfinance)")
    early = frozenset(f"P{k:03d}" for k in range(120)) | {"DEAD"}   # DEAD: member, no prices
    late = frozenset(f"P{k:03d}" for k in range(40, 160))           # 0-39 leave, 120-159 join
    membership = MembershipHistory([(_date(2004, 1, 2), early), (_date(2012, 1, 3), late)], sha256="x")

    def no_network(*a, **k):
        raise AssertionError("PIT mode must not fetch")
    monkeypatch.setattr(factor_study, "get_price_history", no_network)
    monkeypatch.setattr(factor_study, "load_cached_history", lambda s: cache.get(s))
    monkeypatch.setattr(swing_study, "get_price_history", lambda s, *a, **k: cache[s])

    r = factor_study.run_factor_study(start="2006-01-01", pit=True, membership=membership)
    pit = r["pit"]
    assert pit["tickers_ever_members"] == 161
    assert pit["tickers_without_cached_prices"] == 1
    # 2008: 120 covered of 121 members; 2015: all 120 covered.
    assert pit["coverage"]["by_year"]["2008"] == pytest.approx(120 / 121, abs=1e-3)
    assert pit["coverage"]["by_year"]["2015"] == pytest.approx(1.0)
    assert pit["primary"]["tickers"] == 160
    assert set(pit["primary"]["factors"]) == {"F1", "F2", "F3", "F4", "F5"}
    assert all("robust" in f for f in pit["primary"]["factors"].values())
    json.loads(json.dumps(r))
    r2 = factor_study.run_factor_study(start="2006-01-01", pit=True, membership=membership)
    r.pop("generated_at"); r2.pop("generated_at")  # noqa: E702
    assert r == r2


def test_pit_panel_never_includes_a_non_member(monkeypatch):
    from datetime import date as _date
    from sec_analyzer.fetch.sp500_history import MembershipHistory

    idx = pd.bdate_range("2005-01-03", periods=3000)
    cache = {f"Q{k:03d}": (_frame(pd.Series(np.linspace(50, 50 + k, 3000), index=idx)), "cache(yfinance)")
             for k in range(130)}
    cache["SPY"] = (_frame(pd.Series(np.linspace(100, 300, 3000), index=idx)), "cache(yfinance)")
    members = frozenset(f"Q{k:03d}" for k in range(110))
    later = members | {"Q120"}
    membership = MembershipHistory([(_date(2004, 1, 2), members), (_date(2010, 6, 1), later)])
    seen = []
    real = factor_study._evaluate
    monkeypatch.setattr(factor_study, "_evaluate", lambda sub, with_verdict: seen.append(sub) or real(sub, with_verdict))
    monkeypatch.setattr(factor_study, "load_cached_history", lambda s: cache.get(s))
    monkeypatch.setattr(swing_study, "get_price_history", lambda s, *a, **k: cache[s])
    factor_study._run_factor_study_pit("2006-01-01", None, membership)
    primary = seen[0]
    q120 = primary[primary["ticker"] == "Q120"]
    assert not q120.empty and q120["date"].min() >= pd.Timestamp("2010-06-01")
    assert "Q125" not in set(primary["ticker"])  # never a member


def test_end_to_end_detects_a_planted_momentum_effect(monkeypatch):
    """Winners over 12-1 months keep winning by construction; F1 must come out
    supported and positive. Also checks exclusion, JSON and determinism."""
    rng = np.random.default_rng(0)
    # ~2005-2020 so both subperiods exist; confirmation = 3/4 of names, must
    # clear MIN_CROSS_SECTION.
    n_bars, n_tick = 4000, 160
    idx = pd.bdate_range("2005-01-03", periods=n_bars)
    drifts = np.linspace(-0.0008, 0.0012, n_tick)  # persistent drift -> momentum persists
    frames = {}
    for k in range(n_tick):
        rets = drifts[k] + rng.normal(0, 0.01, n_bars)
        frames[f"T{k:03d}"] = (_frame(pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)), "yfinance")
    frames["NASD"] = (_frame(pd.Series(np.linspace(10, 20, n_bars), index=idx)), "nasdaq")
    frames["SPY"] = (_frame(pd.Series(np.linspace(100, 300, n_bars), index=idx)), "yfinance")

    def fake(symbol, *a, **k):
        if symbol not in frames:
            raise RuntimeError("no data")
        return frames[symbol]
    monkeypatch.setattr(factor_study, "get_price_history", fake)
    monkeypatch.setattr(swing_study, "get_price_history", fake)

    tickers = sorted(t for t in frames if t != "SPY") + ["ZZZ"]
    r1 = factor_study.run_factor_study(start="2006-01-01", tickers=tickers)
    r2 = factor_study.run_factor_study(start="2006-01-01", tickers=tickers)
    r1.pop("generated_at"); r2.pop("generated_at")  # noqa: E702
    assert r1 == r2
    json.loads(json.dumps(r1))
    assert [e["ticker"] for e in r1["excluded_non_total_return"]] == ["NASD"]
    assert [f["ticker"] for f in r1["failed"]] == ["ZZZ"]
    f1 = r1["confirmation"]["factors"]["F1"]
    assert f1["by_horizon"]["63"]["mean_spread_pct"] > 0
    assert f1["verdict"] == "supported"
    assert "verdict" not in r1["discovery"]["factors"]["F1"]
    assert set(r1["confirmation"]["factors"]) == {"F1", "F2", "F3", "F4", "F5"}
    assert r1["disclaimer"] == BACKTEST_DISCLAIMER
    assert any("Survivorship" in s for s in r1["limitations"])
