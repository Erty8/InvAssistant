"""Unit tests for ``sec_analyzer.backtest.swing_study`` (SWING_STUDY_SPEC.md).

Binding contract: SWING_STUDY_SPEC.md Sec.7 lists the invariants this module
targets; Sec.3 (point-in-time correctness) and Sec.2 (per-date deciles) are
the ones most likely to silently invalidate the whole study, so those get
tests specifically engineered to FAIL if the bug they guard against were
reintroduced (see the module docstrings on each test below).

No network access anywhere in this module: the price layer
(``sec_analyzer.fetch.prices.get_price_history``) is monkeypatched onto
``swing_study.get_price_history`` -- the exact name both
``_score_ticker_task`` and ``_get_benchmark_frame`` look up at call time,
mirroring ``test_screener.py``'s established monkeypatch convention.

A ``ProcessPoolExecutor`` cannot see a parent-process monkeypatch (spawned
workers re-import modules fresh), so every end-to-end ``run_swing_study``
test forces the thread-pool fallback via the ``force_thread_pool`` fixture
below -- which itself directly exercises the fallback path required by
invariant 10 (a genuine multi-process integration test would either miss the
monkeypatch entirely or require real disk-cache fixtures across a process
boundary; forcing the fallback plus a dedicated ``_make_executor`` unit test
covers the same code path deterministically and fast).
"""

import concurrent.futures

import pandas as pd
import pytest

from sec_analyzer.backtest import BACKTEST_DISCLAIMER, swing_study
from sec_analyzer.store import database
from sec_analyzer.technical.swing import compute_swing_score


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_benchmark_cache():
    """The SPY benchmark cache is a module-level dict (one per *process*,
    per SWING_STUDY_SPEC.md Sec.6) -- reset it around every test so one
    test's stubbed SPY frame can never leak into the next."""
    swing_study._benchmark_cache.clear()
    yield
    swing_study._benchmark_cache.clear()


@pytest.fixture
def force_thread_pool(monkeypatch):
    """Force ``_make_executor`` to fall back to a ``ThreadPoolExecutor``.

    A real ``ProcessPoolExecutor`` spawns fresh interpreters that never see
    this test module's monkeypatches, so every end-to-end
    ``run_swing_study`` test needs this to actually exercise the stubbed
    price layer (and, as a side effect, this directly tests the "process
    pool cannot start -> falls back" behaviour of SWING_STUDY_SPEC.md Sec.6 /
    invariant 10).
    """

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("process pools disabled for this test")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _Boom)


def _df_from_closes(closes: "list[float]", start: str = "2022-01-03") -> pd.DataFrame:
    """A synthetic daily OHLCV frame from an explicit ``closes`` list."""
    n = len(closes)
    idx = pd.bdate_range(start=start, periods=n)
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.005 for c in closes],
            "Low": [c * 0.995 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


def _trend_df(n: int = 260, slope: float = 0.15, start: str = "2022-01-03") -> pd.DataFrame:
    """A steadily-trending frame -- long enough (>=260 bars at the default)
    for SMA200 + its slope lookback, the Bollinger-squeeze percentile, and
    ATR14 to all be computable, so a realistic ``compute_swing_score`` run
    clears the SWING_SPEC.md Sec.3.0 coverage floor (mirrors
    ``test_screener.py``'s proven ``_swing_df`` fixture)."""
    closes = [100.0 + i * slope for i in range(n)]
    return _df_from_closes(closes, start=start)


def _make_price_stub(price_map: dict):
    """A ``get_price_history``-shaped stub keyed by symbol; unknown symbols
    raise (an unwired ticker looks exactly like a delisted/no-data one)."""

    def _fake(symbol, no_cache=False):
        if symbol not in price_map:
            raise ValueError(f"no stubbed price data for {symbol}")
        return price_map[symbol], "stooq"

    return _fake


def _multi_ticker_price_map() -> dict:
    """A small, varied basket (including SPY) for end-to-end study runs:
    one steep uptrend, one mild uptrend, one downtrend, so scores spread out
    across the decile range."""
    return {
        "SPY": _trend_df(n=300, slope=0.05),
        "AAA": _trend_df(n=300, slope=0.30),
        "BBB": _trend_df(n=300, slope=0.10),
        "CCC": _trend_df(n=300, slope=-0.05),
    }


def _run_small_study(monkeypatch, **kwargs):
    price_map = _multi_ticker_price_map()
    monkeypatch.setattr(swing_study, "get_price_history", _make_price_stub(price_map))
    tickers = [t for t in price_map if t != "SPY"]
    return swing_study.run_swing_study(tickers=tickers, max_workers=2, **kwargs)


# ---------------------------------------------------------------------------
# Invariant 1: point-in-time correctness -- no future leak into the score.
# ---------------------------------------------------------------------------


def test_score_at_d_matches_direct_slice_even_with_extreme_future_bars(monkeypatch):
    """A study run whose price frame has extreme outlier bars AFTER date
    ``d`` must produce the SAME score at ``d`` as a direct
    ``compute_indicators(slice_asof(df, d))`` + ``compute_swing_score`` call.

    If the implementation ever computed indicators on the FULL frame instead
    of ``slice_asof(df, d)``, the 52-week high, SMA200, and Bollinger
    percentile would all be dominated by the 100000+ "future" outlier
    values injected below, and the resulting score would be wildly
    different from the direct (correctly point-in-time) computation -- this
    test would then fail.
    """
    n_normal = 260
    normal = [100.0 + i * 0.15 for i in range(n_normal)]
    tail = [100_000.0 + i for i in range(60)]  # deliberately extreme "future" bars
    df = _df_from_closes(normal + tail)
    pos = n_normal - 1
    d = df.index[pos].date()

    # Ground truth: compute directly on the point-in-time slice.
    sliced = swing_study.slice_asof(df, d)
    assert len(sliced) == n_normal  # sanity: exactly the normal part, no leak
    direct_indicators = swing_study.compute_indicators(sliced)
    direct_score = compute_swing_score(direct_indicators)
    assert direct_score is not None

    monkeypatch.setattr(swing_study, "get_price_history", _make_price_stub({"AAA": df}))
    monkeypatch.setattr(swing_study, "_get_benchmark_frame", lambda: None)

    result = swing_study._score_ticker_task({"ticker": "AAA", "dates": [d.isoformat()]})

    assert result["status"] == "ok"
    assert len(result["observations"]) == 1
    obs = result["observations"][0]
    assert obs["score"] == direct_score["score"]
    assert obs["setup"] == direct_score["setup"]


# ---------------------------------------------------------------------------
# Invariant 2: the forward window starts at bar d+1, never bar d.
# ---------------------------------------------------------------------------


def test_forward_return_measured_from_bar_d_plus_1_not_bar_d(monkeypatch):
    """With a synthetic frame where bar ``d+1`` jumps to a known value, the
    observation's ``fwd_10_pct``/``fwd_21_pct`` must reflect that jump
    measured from bar ``d+1``'s close -- not from bar ``d``'s own close
    (which would give a wildly different, easily distinguishable number).
    """
    n_normal = 260
    normal = [100.0 + i * 0.15 for i in range(n_normal)]
    tail = [5000.0] * 40
    tail[0] = 5000.0    # index 260 = bar d+1
    tail[10] = 6000.0   # index 270 = bar d+1+10
    tail[21] = 7500.0   # index 281 = bar d+1+21
    df = _df_from_closes(normal + tail)
    pos = n_normal - 1
    d = df.index[pos].date()
    close_at_d = normal[pos]  # ~138.85 -- what a buggy "measure from d" bug would use

    monkeypatch.setattr(swing_study, "get_price_history", _make_price_stub({"AAA": df}))
    monkeypatch.setattr(swing_study, "_get_benchmark_frame", lambda: None)

    result = swing_study._score_ticker_task({"ticker": "AAA", "dates": [d.isoformat()]})

    assert result["status"] == "ok"
    assert len(result["observations"]) == 1
    obs = result["observations"][0]

    expected_fwd_10 = round((6000.0 / 5000.0 - 1.0) * 100.0, 3)
    expected_fwd_21 = round((7500.0 / 5000.0 - 1.0) * 100.0, 3)
    assert obs["fwd_10_pct"] == expected_fwd_10
    assert obs["fwd_21_pct"] == expected_fwd_21

    wrong_fwd_10_from_d = round((6000.0 / close_at_d - 1.0) * 100.0, 3)
    assert obs["fwd_10_pct"] != wrong_fwd_10_from_d


# ---------------------------------------------------------------------------
# Invariant 3: deciles assigned WITHIN each rebalance date.
# ---------------------------------------------------------------------------


def test_deciles_assigned_within_date_two_disjoint_dates_each_get_full_spread():
    """Two rebalance dates with disjoint score ranges must each get a full
    1..10 decile spread -- pooling scores across dates would instead let the
    higher-scoring date fill the top deciles and the lower-scoring date fill
    the bottom ones."""
    obs = []
    for i in range(10):
        obs.append({"ticker": f"H{i}", "date": "2024-01-31", "score": 80 + i})
    for i in range(10):
        obs.append({"ticker": f"L{i}", "date": "2024-02-29", "score": 10 + i})

    out = swing_study._assign_deciles(obs)

    by_date: dict = {}
    for o in out:
        by_date.setdefault(o["date"], []).append(o["decile"])

    assert sorted(by_date["2024-01-31"]) == list(range(1, 11))
    assert sorted(by_date["2024-02-29"]) == list(range(1, 11))

    jan = {o["ticker"]: o["decile"] for o in out if o["date"] == "2024-01-31"}
    feb = {o["ticker"]: o["decile"] for o in out if o["date"] == "2024-02-29"}
    # Highest score in EACH date lands in decile 10, lowest in decile 1 --
    # independent of the other date's absolute score range.
    assert jan["H9"] == 10 and jan["H0"] == 1
    assert feb["L9"] == 10 and feb["L0"] == 1


# ---------------------------------------------------------------------------
# Invariant 4: dropped_no_score / dropped_no_forward accounting.
# ---------------------------------------------------------------------------


def test_none_score_is_counted_as_dropped_no_score(monkeypatch):
    short_df = _trend_df(n=5)  # far too little history for any sub-score
    d = short_df.index[-1].date()

    monkeypatch.setattr(swing_study, "get_price_history", _make_price_stub({"AAA": short_df}))
    monkeypatch.setattr(swing_study, "_get_benchmark_frame", lambda: None)

    result = swing_study._score_ticker_task({"ticker": "AAA", "dates": [d.isoformat()]})

    assert result["observations"] == []
    assert result["dropped_no_score"] == 1
    assert result["dropped_no_forward"] == 0


def test_no_forward_bars_at_all_drops_observation_and_counts_both_horizons(monkeypatch):
    """``d`` is the very LAST bar of the frame -- the score is computable,
    but neither horizon has any forward data at all, so the observation is
    dropped entirely (never padded/extrapolated) and both configured
    horizons count toward ``dropped_no_forward``."""
    df = _trend_df(n=260, slope=0.15)
    d = df.index[-1].date()

    monkeypatch.setattr(swing_study, "get_price_history", _make_price_stub({"AAA": df}))
    monkeypatch.setattr(swing_study, "_get_benchmark_frame", lambda: None)

    result = swing_study._score_ticker_task({"ticker": "AAA", "dates": [d.isoformat()]})

    assert result["observations"] == []
    assert result["dropped_no_score"] == 0  # the score itself WAS computable
    assert result["dropped_no_forward"] == 2  # both FORWARD_HORIZONS missing


def test_partial_forward_data_keeps_observation_with_missing_horizon_as_none(monkeypatch):
    """261 bars after ``d`` (0-indexed bar 259 of 271) gives exactly enough
    room for the +10 horizon (bar 270 is the last valid index) but not the
    +21 horizon (would need bar 281) -- the observation is kept (score was
    computable and at least one horizon succeeded), with ``fwd_21_pct`` /
    ``excess_21_pct`` left ``None`` and one ``dropped_no_forward`` counted."""
    df = _trend_df(n=271, slope=0.15)
    d = df.index[259].date()

    monkeypatch.setattr(swing_study, "get_price_history", _make_price_stub({"AAA": df}))
    monkeypatch.setattr(swing_study, "_get_benchmark_frame", lambda: None)

    result = swing_study._score_ticker_task({"ticker": "AAA", "dates": [d.isoformat()]})

    assert result["dropped_no_score"] == 0
    assert result["dropped_no_forward"] == 1  # only the +21 horizon missing
    assert len(result["observations"]) == 1
    obs = result["observations"][0]
    assert obs["fwd_10_pct"] is not None
    assert obs["fwd_21_pct"] is None
    assert obs["excess_21_pct"] is None


# ---------------------------------------------------------------------------
# Invariant 5 / 6 / 7: determinism, JSON round-trip, limitations/disclaimer.
# ---------------------------------------------------------------------------


def test_run_swing_study_is_deterministic(monkeypatch, force_thread_pool):
    result1 = _run_small_study(monkeypatch)
    swing_study._benchmark_cache.clear()  # simulate a fully independent second run
    result2 = _run_small_study(monkeypatch)

    d1 = dict(result1)
    d2 = dict(result2)
    assert d1.pop("generated_at") and d2.pop("generated_at")
    assert d1 == d2


def test_run_swing_study_result_is_json_round_trippable(monkeypatch, force_thread_pool):
    import json

    result = _run_small_study(monkeypatch)
    assert json.loads(json.dumps(result)) == result


def test_run_swing_study_carries_limitations_and_disclaimer(monkeypatch, force_thread_pool):
    result = _run_small_study(monkeypatch)

    assert result["disclaimer"] == BACKTEST_DISCLAIMER
    assert result["limitations"]
    assert "hayatta kal" in result["limitations"][0].lower()  # survivorship bias, listed first/dominant


def test_run_swing_study_never_raises_for_a_bad_ticker(monkeypatch, force_thread_pool):
    price_map = _multi_ticker_price_map()
    monkeypatch.setattr(swing_study, "get_price_history", _make_price_stub(price_map))

    result = swing_study.run_swing_study(tickers=["AAA", "BBB", "GHOST"], max_workers=2)

    assert any(s["ticker"] == "GHOST" for s in result["skipped"])
    assert result["skipped"][0]["reason"]  # non-empty Turkish-ish reason string
    assert result["tickers_scanned"] == 3


# ---------------------------------------------------------------------------
# Invariant 8: low_sample is True exactly when a decile's n < 30.
# ---------------------------------------------------------------------------


def test_low_sample_flag_exactly_at_the_n_30_threshold():
    obs = []
    for i in range(29):
        obs.append({"ticker": f"A{i}", "score": 10, "decile": 1, "excess_10_pct": 1.0, "excess_21_pct": 1.0})
    for i in range(30):
        obs.append({"ticker": f"B{i}", "score": 20, "decile": 2, "excess_10_pct": 2.0, "excess_21_pct": 2.0})

    deciles, _spread, _mono = swing_study._build_decile_tables(obs)

    d1 = next(e for e in deciles["10"] if e["decile"] == 1)
    d2 = next(e for e in deciles["10"] if e["decile"] == 2)
    assert d1["n"] == 29 and d1["low_sample"] is True
    assert d2["n"] == 30 and d2["low_sample"] is False


# ---------------------------------------------------------------------------
# Invariant 9: save_swing_study / load_latest_swing_study persistence.
# ---------------------------------------------------------------------------


def _sample_study_payload(universe: str = "SP500", generated_at: str = "2026-07-01T00:00:00Z") -> dict:
    return {
        "generated_at": generated_at,
        "index": universe,
        "index_label": "S&P 500" if universe == "SP500" else "Nasdaq 100",
        "start": "2024-01-31",
        "end": "2024-12-31",
        "rebalance_dates": 12,
        "observations": 500,
        "tickers_scanned": 50,
        "skipped": [],
        "dropped_no_score": 3,
        "dropped_no_forward": 7,
        "deciles": {"10": [], "21": []},
        "spread": {
            "10": {"mean_excess_pct": 1.0, "ci_low": 0.1, "ci_high": 2.0},
            "21": {"mean_excess_pct": None, "ci_low": None, "ci_high": None},
        },
        "monotonicity": {"10": 0.5, "21": 0.0},
        "by_setup": {},
        "limitations": ["test limitation"],
        "disclaimer": BACKTEST_DISCLAIMER,
    }


def test_save_and_load_swing_study_round_trips_through_temp_db(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    payload = _sample_study_payload()

    row_id = database.save_swing_study(payload, db_path=db_path)
    assert row_id and row_id > 0

    loaded = database.load_latest_swing_study(db_path=db_path)
    assert loaded == payload


def test_load_latest_swing_study_returns_none_on_empty_db(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    assert database.load_latest_swing_study(db_path=db_path) is None


def test_load_latest_swing_study_per_universe_scoping(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    sp500 = _sample_study_payload(universe="SP500", generated_at="2026-07-01T00:00:00Z")
    ndx = _sample_study_payload(universe="NDX", generated_at="2026-07-02T00:00:00Z")

    database.save_swing_study(sp500, db_path=db_path)
    database.save_swing_study(ndx, db_path=db_path)

    assert database.load_latest_swing_study(db_path=db_path, universe="SP500") == sp500
    assert database.load_latest_swing_study(db_path=db_path, universe="NDX") == ndx
    assert database.load_latest_swing_study(db_path=db_path) == ndx  # most recent overall
    assert database.load_latest_swing_study(db_path=db_path, universe="NDX", ) is not None


# ---------------------------------------------------------------------------
# Invariant 10: process-pool-unavailable falls back to a thread pool.
# ---------------------------------------------------------------------------


def test_make_executor_falls_back_to_thread_pool_when_process_pool_unavailable(monkeypatch):
    class _Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("no process pools in this sandbox")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _Boom)

    executor, kind = swing_study._make_executor(2)
    try:
        assert kind == "thread"
        assert isinstance(executor, concurrent.futures.ThreadPoolExecutor)
    finally:
        executor.shutdown(wait=True)


def test_make_executor_uses_process_pool_when_available():
    executor, kind = swing_study._make_executor(2)
    try:
        assert kind == "process"
        assert isinstance(executor, concurrent.futures.ProcessPoolExecutor)
    finally:
        executor.shutdown(wait=True)
