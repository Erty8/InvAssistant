"""Unit tests for the display-only analyst-consensus feature:

* ``sec_analyzer.fetch.analyst.get_analyst_targets`` (and its small helpers)
  -- fetches/caches consensus analyst price targets from the optional
  ``yfinance`` package's ``Ticker.info`` dict.
* ``sec_analyzer.cli._analyst_line`` -- renders the Turkish verdict-card line
  from the dict ``get_analyst_targets`` returns.

No real network access is used anywhere in this module: ``yfinance`` is
monkeypatched via ``sys.modules`` (mirroring the established pattern in
``test_technical.py``'s price-fetching tests), and ``Config.RAW_DIR`` is
pointed at pytest's ``tmp_path`` so nothing touches the package's real cache
directory.

This is a display-only cross-check (see ``sec_analyzer/fetch/analyst.py``'s
module docstring): it never feeds the valuation engine, so there is nothing
here to hand-verify numerically -- these tests only check plumbing (coercion,
caching, None-safety) and string rendering.
"""

import json
import sys

import pytest

from sec_analyzer.cli import _analyst_line
from sec_analyzer.config import Config
from sec_analyzer.fetch import analyst


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _full_info(**overrides):
    """A fully-populated fake yfinance ``Ticker(...).info`` dict."""
    info = {
        "targetMeanPrice": 128.0,
        "targetHighPrice": 160.0,
        "targetLowPrice": 95.0,
        "targetMedianPrice": 130.0,
        "numberOfAnalystOpinions": 34,
        "currency": "USD",
        "recommendationKey": "buy",
    }
    info.update(overrides)
    return info


class _FakeTicker:
    """Minimal stand-in for ``yfinance.Ticker`` used by the fetch tests."""

    call_count = 0

    def __init__(self, ticker):
        self.ticker = ticker

    @property
    def info(self):
        _FakeTicker.call_count += 1
        return _FakeTicker.info_to_return


class _FakeYfModule:
    """Minimal stand-in for the ``yfinance`` module itself."""

    Ticker = _FakeTicker


def _install_fake_yfinance(monkeypatch, info_dict):
    """Monkeypatch ``sys.modules['yfinance']`` so ``import yfinance`` inside
    ``get_analyst_targets`` resolves to a fake module whose ``Ticker(...).info``
    returns ``info_dict``. Resets the call counter so each test starts fresh."""
    _FakeTicker.call_count = 0
    _FakeTicker.info_to_return = info_dict
    monkeypatch.setitem(sys.modules, "yfinance", _FakeYfModule)


def _install_raising_yfinance(monkeypatch, exc=RuntimeError("boom")):
    """Monkeypatch ``yfinance`` so ``Ticker(...).info`` raises ``exc``."""

    class _RaisingTicker:
        def __init__(self, ticker):
            pass

        @property
        def info(self):
            raise exc

    class _RaisingYfModule:
        Ticker = _RaisingTicker

    monkeypatch.setitem(sys.modules, "yfinance", _RaisingYfModule)


# ---------------------------------------------------------------------------
# get_analyst_targets -- happy path / coercion
# ---------------------------------------------------------------------------


def test_get_analyst_targets_happy_path_coerces_all_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, _full_info())

    result = analyst.get_analyst_targets("FAKE", no_cache=True)

    assert result == {
        "target_mean": 128.0,
        "target_high": 160.0,
        "target_low": 95.0,
        "target_median": 130.0,
        "num_analysts": 34,
        "currency": "USD",
        "recommendation": "buy",
        "source": "yfinance",
    }
    # Types, not just values -- floats/int must actually be coerced.
    assert isinstance(result["target_mean"], float)
    assert isinstance(result["num_analysts"], int)


def test_get_analyst_targets_coerces_float_analyst_count_to_int(monkeypatch, tmp_path):
    """``numberOfAnalystOpinions`` arriving as a float (e.g. ``34.0``, as
    some yfinance responses shape it) must be coerced to a plain int."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, _full_info(numberOfAnalystOpinions=34.0))

    result = analyst.get_analyst_targets("FAKE", no_cache=True)

    assert result["num_analysts"] == 34
    assert isinstance(result["num_analysts"], int)


def test_get_analyst_targets_coerces_numeric_strings(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(
        monkeypatch,
        _full_info(targetMeanPrice="128.0", targetHighPrice="160", numberOfAnalystOpinions="34"),
    )

    result = analyst.get_analyst_targets("FAKE", no_cache=True)

    assert result["target_mean"] == pytest.approx(128.0)
    assert result["target_high"] == pytest.approx(160.0)
    assert result["num_analysts"] == 34


# ---------------------------------------------------------------------------
# get_analyst_targets -- missing/invalid targetMeanPrice -> None
# ---------------------------------------------------------------------------


def test_get_analyst_targets_none_when_target_mean_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    info = _full_info()
    del info["targetMeanPrice"]
    _install_fake_yfinance(monkeypatch, info)

    assert analyst.get_analyst_targets("FAKE", no_cache=True) is None


@pytest.mark.parametrize("bad_mean", [0, -5.0, "not-a-number"])
def test_get_analyst_targets_none_when_target_mean_non_positive_or_unparseable(
    monkeypatch, tmp_path, bad_mean
):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, _full_info(targetMeanPrice=bad_mean))

    assert analyst.get_analyst_targets("FAKE", no_cache=True) is None


# ---------------------------------------------------------------------------
# get_analyst_targets -- partial info
# ---------------------------------------------------------------------------


def test_get_analyst_targets_partial_info_only_mean_present(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, {"targetMeanPrice": 128.0})

    result = analyst.get_analyst_targets("FAKE", no_cache=True)

    assert result is not None
    assert result["target_mean"] == 128.0
    for key in ("target_high", "target_low", "target_median", "num_analysts", "currency", "recommendation"):
        assert result[key] is None
    assert result["source"] == "yfinance"


def test_get_analyst_targets_non_string_currency_and_recommendation_become_none(monkeypatch, tmp_path):
    """A non-string ``currency``/``recommendationKey`` (e.g. a stray numeric
    or dict from a malformed response) must not be passed through as-is."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(
        monkeypatch, _full_info(currency=123, recommendationKey=["buy"])
    )

    result = analyst.get_analyst_targets("FAKE", no_cache=True)

    assert result["currency"] is None
    assert result["recommendation"] is None


# ---------------------------------------------------------------------------
# get_analyst_targets -- error paths never raise
# ---------------------------------------------------------------------------


def test_get_analyst_targets_none_when_yfinance_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_raising_yfinance(monkeypatch)

    assert analyst.get_analyst_targets("FAKE", no_cache=True) is None


def test_get_analyst_targets_none_when_yfinance_not_installed(monkeypatch, tmp_path):
    """Simulate yfinance being absent: setting the sys.modules entry to None
    makes any subsequent `import yfinance` raise ImportError immediately,
    without needing the package to be uninstalled from the environment."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "yfinance", None)

    assert analyst.get_analyst_targets("FAKE", no_cache=True) is None


# ---------------------------------------------------------------------------
# get_analyst_targets -- caching
# ---------------------------------------------------------------------------


def test_get_analyst_targets_writes_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, _full_info())

    analyst.get_analyst_targets("FAKE", no_cache=True)

    cache_file = tmp_path / "analyst_FAKE.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["target_mean"] == 128.0


def test_get_analyst_targets_second_call_hits_cache_not_yfinance(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, _full_info())

    first = analyst.get_analyst_targets("FAKE", no_cache=False)
    assert _FakeTicker.call_count == 1

    # Mutate what yfinance would return; a cache hit must NOT reflect this.
    _FakeTicker.info_to_return = _full_info(targetMeanPrice=999.0)
    second = analyst.get_analyst_targets("FAKE", no_cache=False)

    assert _FakeTicker.call_count == 1  # no second network call
    assert second == first
    assert second["target_mean"] == 128.0


def test_get_analyst_targets_no_cache_true_refetches(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    _install_fake_yfinance(monkeypatch, _full_info())

    analyst.get_analyst_targets("FAKE", no_cache=False)
    assert _FakeTicker.call_count == 1

    _FakeTicker.info_to_return = _full_info(targetMeanPrice=999.0)
    result = analyst.get_analyst_targets("FAKE", no_cache=True)

    assert _FakeTicker.call_count == 2
    assert result["target_mean"] == 999.0


def test_get_analyst_targets_corrupt_cache_triggers_refetch_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    Config.ensure_dirs()
    cache_path = tmp_path / "analyst_FAKE.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    _install_fake_yfinance(monkeypatch, _full_info())

    result = analyst.get_analyst_targets("FAKE", no_cache=False)

    assert result is not None
    assert result["target_mean"] == 128.0
    assert _FakeTicker.call_count == 1


# ---------------------------------------------------------------------------
# _analyst_line
# ---------------------------------------------------------------------------


def _full_analyst_dict(**overrides):
    d = {
        "target_mean": 128.0,
        "target_high": 160.0,
        "target_low": 95.0,
        "target_median": 130.0,
        "num_analysts": 34,
        "currency": "USD",
        "recommendation": "buy",
        "source": "yfinance",
    }
    d.update(overrides)
    return d


def test_analyst_line_full_dict_upside_when_mean_above_price():
    # target_mean=128, price=100 -> upside = 128/100 - 1 = +28% -> "+%28"
    line = _analyst_line(_full_analyst_dict(), 100.0)

    assert line is not None
    assert "ort" in line
    assert "34 analist" in line
    assert "+%28" in line
    assert "aralık" in line
    assert "$95" in line and "$160" in line


def test_analyst_line_downside_when_mean_below_price():
    # target_mean=128, price=200 -> upside = 128/200 - 1 = -36% -> "-%36"
    line = _analyst_line(_full_analyst_dict(), 200.0)

    assert line is not None
    assert "-%36" in line


def test_analyst_line_none_when_analyst_falsy():
    assert _analyst_line(None, 100.0) is None
    assert _analyst_line({}, 100.0) is None


def test_analyst_line_none_when_target_mean_missing():
    analyst_dict = _full_analyst_dict()
    analyst_dict["target_mean"] = None
    assert _analyst_line(analyst_dict, 100.0) is None


@pytest.mark.parametrize("bad_price", [None, 0, -50.0])
def test_analyst_line_no_upside_segment_when_price_unusable(bad_price):
    line = _analyst_line(_full_analyst_dict(), bad_price)

    assert line is not None
    assert "ort" in line
    # No upside marker ("+%" or "-%") should appear when price is unusable,
    # while the unrelated range segment ("aralık") is still present.
    assert "+%" not in line
    assert "-%" not in line
    assert "aralık" in line


def test_analyst_line_no_analyst_count_segment_when_missing():
    analyst_dict = _full_analyst_dict()
    analyst_dict["num_analysts"] = None
    line = _analyst_line(analyst_dict, 100.0)

    assert line is not None
    assert "analist" not in line


def test_analyst_line_no_range_segment_when_low_or_high_missing():
    only_low_missing = _full_analyst_dict()
    only_low_missing["target_low"] = None
    line = _analyst_line(only_low_missing, 100.0)
    assert line is not None
    assert "aralık" not in line

    only_high_missing = _full_analyst_dict()
    only_high_missing["target_high"] = None
    line = _analyst_line(only_high_missing, 100.0)
    assert line is not None
    assert "aralık" not in line


def test_analyst_line_never_contains_none_or_nan_text():
    """Regardless of which optional fields are missing, the rendered line
    must never leak a literal 'None'/'NaN' substring -- missing pieces are
    simply omitted segments, never a printed sentinel."""
    sparse = {"target_mean": 128.0}
    line = _analyst_line(sparse, None)

    assert line is not None
    assert "None" not in line
    assert "NaN" not in line
