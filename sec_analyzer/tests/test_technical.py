"""Unit tests for sec_analyzer.technical (indicators + verdict) and the
Stooq/yfinance price-fetching layer in sec_analyzer.fetch.prices.

No real network access is used anywhere in this module: the price-history
tests monkeypatch ``requests.get`` (and, for the fallback path, force
``import yfinance`` to fail via ``sys.modules``) rather than hitting Stooq
or Yahoo Finance.
"""

import io
import sys

import pandas as pd
import pytest

from sec_analyzer.config import Config
from sec_analyzer.fetch import prices
from sec_analyzer.technical.indicators import compute_indicators
from sec_analyzer.technical.verdict import technical_verdict

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_df(closes, start="2023-01-02"):
    """Build a minimal OHLCV price-history DataFrame matching the shape
    ``get_price_history`` returns (Date index, ascending, Open/High/Low/
    Close/Volume columns). Open/High/Low are just set equal to Close since
    none of the code under test here reads them."""
    idx = pd.bdate_range(start=start, periods=len(closes))
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------


def test_compute_indicators_rising_series_is_strong_and_near_high():
    """A steadily rising Close series: RSI pegs high (all gains, no
    losses), price sits above SMA50, and the price is at (or essentially
    at) the top of its own 52-week range."""
    closes = [100 + i for i in range(100)]
    df = _make_df(closes)

    ind = compute_indicators(df)

    assert ind["rsi14"] is not None and ind["rsi14"] > 60
    assert ind["sma50"] is not None and ind["sma50"] < ind["price"]
    assert ind["dist_sma50_pct"] is not None and ind["dist_sma50_pct"] > 0
    assert ind["range_position_pct"] is not None and ind["range_position_pct"] >= 95
    assert ind["price"] == 199.0
    assert ind["as_of"] == df.index[-1].strftime("%Y-%m-%d")


def test_compute_indicators_falling_series_is_oversold():
    """A steadily falling Close series: RSI bottoms out (all losses, no
    gains), which the verdict rule must read as oversold regardless of the
    SMA50 relationship."""
    closes = [300 - i for i in range(100)]
    df = _make_df(closes)

    ind = compute_indicators(df)
    assert ind["rsi14"] is not None and ind["rsi14"] < 30

    verdict = technical_verdict(ind)
    assert verdict["verdict"] == "AŞIRI SATIM"


def test_technical_verdict_overbought_when_rsi_high_and_price_above_sma50():
    """RSI > 70 and price > SMA50 together must trigger the overbought
    verdict."""
    closes = [100 + i for i in range(100)]
    df = _make_df(closes)
    ind = compute_indicators(df)

    verdict = technical_verdict(ind)

    assert verdict["verdict"] == "AŞIRI ALIM"
    assert "RSI" in verdict["verdict_detail"]


def test_compute_indicators_short_flat_series_has_no_exceptions():
    """A short (20-row), perfectly flat series must not raise anywhere in
    the pipeline: SMA200/SMA50 need more history than is available, RSI is
    undefined for a zero-volatility series (0/0), and the verdict must
    fall back cleanly to neutral/insufficient-data."""
    closes = [100.0] * 20
    df = _make_df(closes)

    ind = compute_indicators(df)

    assert ind["sma50"] is None
    assert ind["sma200"] is None
    assert ind["rsi14"] is None
    assert ind["golden_cross"] is None
    assert ind["death_cross"] is None

    verdict = technical_verdict(ind)
    assert verdict["verdict"] == "NÖTR"
    assert verdict["verdict_detail"] == "yetersiz veri"


def test_compute_indicators_golden_cross_detected():
    """Engineer a long decline followed by a strong, sustained rise so
    that SMA50 is clearly below SMA200 sixty trading days ago and clearly
    above it now -- the textbook golden-cross setup.

    The values below are plain arithmetic (linear) sequences chosen so the
    SMA50/SMA200 values at both the "now" and "60 days ago" points can be
    (and were) verified by hand via the arithmetic-series average formula
    ``(first + last) / 2``:

    * Now (last row): SMA50 ~= 577, SMA200 ~= 374.5 -> SMA50 above SMA200.
    * 60 trading days earlier: SMA50 ~= 226, SMA200 ~= 403 -> SMA50 below
      SMA200.
    """
    decline = [1000 - 4 * i for i in range(220)]
    rise = [130 + 6 * j for j in range(100)]
    closes = decline + rise
    df = _make_df(closes)

    ind = compute_indicators(df)

    assert ind["sma50_above_sma200"] is True
    assert ind["golden_cross"] is True
    assert ind["death_cross"] is False


# ---------------------------------------------------------------------------
# technical_verdict
# ---------------------------------------------------------------------------


def _sample_indicators(**overrides):
    base = {
        "rsi14": 74.0,
        "price": 112.0,
        "sma50": 100.0,
        "sma200": 90.0,
        "dist_sma50_pct": 12.0,
        "dist_sma200_pct": 24.4,
        "high_52w": 120.0,
        "low_52w": 80.0,
        "range_position_pct": 80.0,
        "volatility_20d": 0.32,
        "golden_cross": False,
        "death_cross": False,
        "sma50_above_sma200": True,
        "as_of": "2024-01-01",
    }
    base.update(overrides)
    return base


def test_verdict_detail_format_matches_spec_example():
    ind = _sample_indicators()
    result = technical_verdict(ind)
    assert result["verdict_detail"] == "RSI 74, SMA50 +%12"


def test_verdict_neutral_when_rsi_missing():
    ind = _sample_indicators(rsi14=None)
    result = technical_verdict(ind)
    assert result["verdict"] == "NÖTR"
    assert result["verdict_detail"] == "yetersiz veri"


def test_horizon_summary_3m_mentions_rsi():
    ind = _sample_indicators()
    result = technical_verdict(ind, horizon="3m")
    assert result["horizon"] == "3m"
    assert "RSI" in result["horizon_summary"]


def test_horizon_summary_1y_is_balanced():
    ind = _sample_indicators()
    result = technical_verdict(ind, horizon="1y")
    assert result["horizon"] == "1y"
    assert "RSI" in result["horizon_summary"]
    assert "SMA" in result["horizon_summary"]


def test_horizon_summary_5y_notes_rsi_not_relevant_and_frames_sma200_as_entry_timing():
    ind = _sample_indicators()
    result = technical_verdict(ind, horizon="5y")
    assert result["horizon"] == "5y"
    assert "RSI" in result["horizon_summary"]
    assert "SMA200" in result["horizon_summary"]
    assert "giriş" in result["horizon_summary"].lower()


# ---------------------------------------------------------------------------
# get_price_history / latest_price (network mocked out)
# ---------------------------------------------------------------------------


def _fake_stooq_csv(n=40, start_price=100.0):
    """Build a small, valid Stooq-shaped CSV payload as plain text."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    lines = ["Date,Open,High,Low,Close,Volume"]
    price = start_price
    for d in dates:
        price += 0.5
        lines.append(
            f"{d.date()},{price - 0.3:.2f},{price + 0.5:.2f},{price - 0.5:.2f},{price:.2f},1000000"
        )
    return "\n".join(lines) + "\n"


class _FakeResponse:
    """Minimal stand-in for requests.Response used by the fetch tests."""

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_get_price_history_stooq_success_writes_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    csv_text = _fake_stooq_csv(40)
    captured_url = {}

    def _fake_get(url, *a, **k):
        captured_url["url"] = url
        return _FakeResponse(csv_text)

    monkeypatch.setattr(prices.requests, "get", _fake_get)

    df, source = prices.get_price_history("FAKE", no_cache=True)

    assert source == "stooq"
    assert len(df) == 40
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    # The Stooq request must not carry any d1/d2 date-range params -- those
    # would narrow the response to a fixed lookback window instead of
    # Stooq's default behavior of returning the full available history.
    assert "d1=" not in captured_url["url"]
    assert "d2=" not in captured_url["url"]

    cache_file = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    assert cache_file.exists()


def test_get_price_history_cache_hit_skips_network(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    csv_text = _fake_stooq_csv(40)
    cached_df = (
        pd.read_csv(io.StringIO(csv_text), parse_dates=["Date"]).set_index("Date").sort_index()
    )
    cache_path = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    cached_df.to_csv(cache_path, index_label="Date")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("network should not be hit on a fresh cache hit")

    monkeypatch.setattr(prices.requests, "get", _fail_if_called)

    df, source = prices.get_price_history("FAKE")

    assert source == "cache(stooq)"
    assert len(df) == 40


def test_get_price_history_ignores_stale_narrow_cache_from_old_filename(monkeypatch, tmp_path):
    """A cache file written under the pre-fix filename (unsuffixed,
    ``prices_FAKE.csv``) -- e.g. one narrowed to ~2 years by yfinance's old
    hardcoded ``period="2y"`` -- must simply be ignored (different
    filename), not read as if it were the new full-history cache."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    stale_csv = _fake_stooq_csv(40)
    stale_df = (
        pd.read_csv(io.StringIO(stale_csv), parse_dates=["Date"]).set_index("Date").sort_index()
    )
    old_style_path = tmp_path / "prices_FAKE.csv"
    stale_df.to_csv(old_style_path, index_label="Date")

    fresh_csv = _fake_stooq_csv(80)
    monkeypatch.setattr(prices.requests, "get", lambda *a, **k: _FakeResponse(fresh_csv))

    df, source = prices.get_price_history("FAKE")

    assert source == "stooq"
    assert len(df) == 80


def test_fetch_yfinance_defaults_to_max_period(monkeypatch):
    """The yfinance fallback must default to the full available history
    (``period="max"``), not the old hardcoded ``"2y"`` window."""
    captured = {}

    class _FakeYf:
        @staticmethod
        def download(ticker, period=None, interval=None, progress=None):
            captured["period"] = period
            idx = pd.bdate_range("2023-01-02", periods=40)
            return pd.DataFrame(
                {
                    "Open": [1.0] * 40,
                    "High": [1.0] * 40,
                    "Low": [1.0] * 40,
                    "Close": [1.0] * 40,
                    "Volume": [1.0] * 40,
                },
                index=idx,
            )

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYf)

    df = prices._fetch_yfinance("FAKE")

    assert captured["period"] == "max"
    assert len(df) == 40


def test_get_price_history_raises_when_stooq_and_yfinance_both_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    # An HTML error page instead of CSV -- Stooq's "no data" / blocked shape.
    monkeypatch.setattr(prices.requests, "get", lambda *a, **k: _FakeResponse("<html>error</html>"))
    # Force `import yfinance` inside _fetch_yfinance to raise ImportError:
    # setting a sys.modules entry to None makes any subsequent `import` of
    # that name raise ImportError immediately, without needing the package
    # to be absent from the environment.
    monkeypatch.setitem(sys.modules, "yfinance", None)

    with pytest.raises(prices.PriceDataError):
        prices.get_price_history("FAKE", no_cache=True)


def test_latest_price_returns_last_close_and_its_date():
    idx = pd.bdate_range("2023-01-02", periods=5)
    df = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0, 14.5]}, index=idx)

    price, as_of = prices.latest_price(df)

    assert price == 14.5
    assert as_of == idx[-1].strftime("%Y-%m-%d")
