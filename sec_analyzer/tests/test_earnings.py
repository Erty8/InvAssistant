"""Unit tests for the display-only earnings-surprise (beat/miss) feature:

* ``sec_analyzer.fetch.earnings.get_earnings_history`` (and its small helpers)
  -- fetches/caches recent quarterly EPS estimate-vs-actual history from the
  optional ``yfinance`` package's ``Ticker.get_earnings_history()``.

No real network access is used anywhere in this module: ``yfinance`` is
monkeypatched via ``sys.modules`` (mirroring ``test_analyst.py``), and
``Config.RAW_DIR`` is pointed at pytest's ``tmp_path`` so nothing touches the
package's real cache directory.

This is a display-only cross-check (see ``sec_analyzer/fetch/earnings.py``'s
module docstring): it never feeds the valuation engine, so these tests only
check plumbing (parsing, surprise math, NaN/None-safety, caching).
"""

import json
import math
import sys

import pandas as pd
import pytest

from sec_analyzer.config import Config
from sec_analyzer.fetch import earnings


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _history_df(estimates, actuals, dates):
    """Build a yfinance-shaped ``earnings_history`` DataFrame."""
    return pd.DataFrame(
        {"epsEstimate": estimates, "epsActual": actuals},
        index=pd.to_datetime(dates),
    )


class _FakeTicker:
    """Minimal stand-in for ``yfinance.Ticker`` used by the fetch tests."""

    call_count = 0
    df_to_return = None

    def __init__(self, ticker):
        self.ticker = ticker

    def get_earnings_history(self, as_dict=False):
        _FakeTicker.call_count += 1
        return _FakeTicker.df_to_return


class _FakeYfModule:
    Ticker = _FakeTicker


def _install_fake_yfinance(monkeypatch, df):
    _FakeTicker.call_count = 0
    _FakeTicker.df_to_return = df
    monkeypatch.setitem(sys.modules, "yfinance", _FakeYfModule)


# ---------------------------------------------------------------------------
# Helper-level tests: surprise math
# ---------------------------------------------------------------------------


def test_surprise_pct_beat_and_miss():
    assert earnings._surprise_pct(1.35, 1.20) == pytest.approx(12.5)
    assert earnings._surprise_pct(0.85, 0.90) == pytest.approx(-5.5555, rel=1e-3)


def test_surprise_pct_negative_estimate_uses_abs_denominator():
    # Expected loss of -1.00, actual loss of only -0.50 -> a positive surprise.
    assert earnings._surprise_pct(-0.50, -1.00) == pytest.approx(50.0)


@pytest.mark.parametrize(
    "actual,estimate", [(None, 1.0), (1.0, None), (1.0, 0.0)]
)
def test_surprise_pct_none_when_unusable(actual, estimate):
    assert earnings._surprise_pct(actual, estimate) is None


def test_coerce_float_rejects_nan_and_inf():
    assert earnings._coerce_float(float("nan")) is None
    assert earnings._coerce_float(float("inf")) is None
    assert earnings._coerce_float("not-a-number") is None
    assert earnings._coerce_float("1.5") == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# get_earnings_history -- happy path / ordering / surprise
# ---------------------------------------------------------------------------


def test_get_earnings_history_happy_path_sorted_newest_first(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    df = _history_df(
        estimates=[1.20, 0.90],
        actuals=[1.35, 0.85],
        dates=["2024-06-30", "2024-09-30"],
    )
    _install_fake_yfinance(monkeypatch, df)

    result = earnings.get_earnings_history("FAKE", no_cache=True)

    assert result is not None
    assert result["source"] == "yfinance"
    quarters = result["quarters"]
    assert [q["period"] for q in quarters] == ["2024-09-30", "2024-06-30"]
    # Newest quarter: (0.85 - 0.90) / 0.90 * 100 = -5.56% (a miss).
    assert quarters[0]["surprise_pct"] == pytest.approx(-5.5555, rel=1e-3)
    # Older quarter: (1.35 - 1.20) / 1.20 * 100 = +12.5% (a beat).
    assert quarters[1]["surprise_pct"] == pytest.approx(12.5)


def test_get_earnings_history_drops_rows_missing_both_eps(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    df = _history_df(
        estimates=[1.20, math.nan],
        actuals=[1.35, math.nan],
        dates=["2024-06-30", "2024-09-30"],
    )
    _install_fake_yfinance(monkeypatch, df)

    result = earnings.get_earnings_history("FAKE", no_cache=True)

    assert result is not None
    assert [q["period"] for q in result["quarters"]] == ["2024-06-30"]


def test_get_earnings_history_nan_estimate_yields_none_not_nan(monkeypatch, tmp_path):
    """A missing estimate arrives as NaN; it must serialize as JSON null
    (None), never a raw NaN that the browser's JSON.parse would reject."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    df = _history_df(estimates=[math.nan], actuals=[1.35], dates=["2024-06-30"])
    _install_fake_yfinance(monkeypatch, df)

    result = earnings.get_earnings_history("FAKE", no_cache=True)

    q = result["quarters"][0]
    assert q["eps_estimate"] is None
    assert q["eps_actual"] == pytest.approx(1.35)
    assert q["surprise_pct"] is None
    # The whole payload must be strict-JSON serializable (no NaN literal).
    assert "NaN" not in json.dumps(result)


# ---------------------------------------------------------------------------
# get_earnings_history -- empty / None responses -> None
# ---------------------------------------------------------------------------


def test_get_earnings_history_none_when_df_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, None)

    assert earnings.get_earnings_history("FAKE", no_cache=True) is None


def test_get_earnings_history_none_when_df_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, _history_df([], [], []))

    assert earnings.get_earnings_history("FAKE", no_cache=True) is None


# ---------------------------------------------------------------------------
# get_earnings_history -- error paths never raise
# ---------------------------------------------------------------------------


def test_get_earnings_history_none_when_yfinance_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))

    class _RaisingTicker:
        def __init__(self, ticker):
            pass

        def get_earnings_history(self, as_dict=False):
            raise RuntimeError("boom")

    class _RaisingYfModule:
        Ticker = _RaisingTicker

    monkeypatch.setitem(sys.modules, "yfinance", _RaisingYfModule)

    assert earnings.get_earnings_history("FAKE", no_cache=True) is None


def test_get_earnings_history_none_when_yfinance_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "yfinance", None)

    assert earnings.get_earnings_history("FAKE", no_cache=True) is None


# ---------------------------------------------------------------------------
# get_earnings_history -- caching
# ---------------------------------------------------------------------------


def test_get_earnings_history_writes_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    df = _history_df([1.20], [1.35], ["2024-06-30"])
    _install_fake_yfinance(monkeypatch, df)

    earnings.get_earnings_history("FAKE", no_cache=True)

    cache_file = tmp_path / "earnings_FAKE.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["quarters"][0]["eps_actual"] == pytest.approx(1.35)


def test_get_earnings_history_second_call_hits_cache_not_yfinance(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    df = _history_df([1.20], [1.35], ["2024-06-30"])
    _install_fake_yfinance(monkeypatch, df)

    first = earnings.get_earnings_history("FAKE", no_cache=False)
    assert _FakeTicker.call_count == 1

    _FakeTicker.df_to_return = _history_df([9.0], [9.9], ["2024-09-30"])
    second = earnings.get_earnings_history("FAKE", no_cache=False)

    assert _FakeTicker.call_count == 1  # served from cache, no second call
    assert second == first


def test_get_earnings_history_corrupt_cache_triggers_refetch(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    Config.ensure_dirs()
    (tmp_path / "earnings_FAKE.json").write_text("{not valid json", encoding="utf-8")

    _install_fake_yfinance(monkeypatch, _history_df([1.20], [1.35], ["2024-06-30"]))

    result = earnings.get_earnings_history("FAKE", no_cache=False)

    assert result is not None
    assert _FakeTicker.call_count == 1
