"""Unit tests for ``sec_analyzer.fetch.prices``: ``slice_asof`` (point-in-time
price slicing) and ``_drop_unusable_bars`` (the swing-screener price-cleaning
regression, SWING_SPEC.md).

No network access -- these build a small in-memory OHLCV DataFrame directly,
matching the ``Date``-indexed, ascending shape ``get_price_history`` returns;
the one test that exercises ``get_price_history`` itself monkeypatches
``requests.get`` rather than hitting Stooq.
"""

from datetime import date

import pandas as pd
import pytest

from sec_analyzer.config import Config
from sec_analyzer.fetch import prices
from sec_analyzer.fetch.prices import _drop_unusable_bars, slice_asof


def _price_df():
    """A small ascending daily frame with a deliberate weekend gap:
    Fri 2023-01-06, then Mon 2023-01-09 (no Sat/Sun rows, as real market
    data has), then a few more trading days."""
    dates = pd.to_datetime(
        ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09", "2023-01-10"]
    )
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "High": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "Low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
            "Close": [10.2, 11.2, 12.2, 13.2, 14.2, 15.2],
            "Volume": [100, 200, 300, 400, 500, 600],
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_slice_asof_none_returns_df_unchanged():
    df = _price_df()
    result = slice_asof(df, None)
    assert result is df  # unchanged -- not even a copy is required by contract
    pd.testing.assert_frame_equal(result, df)


def test_slice_asof_keeps_rows_on_or_before_cutoff():
    df = _price_df()
    result = slice_asof(df, "2023-01-05")
    assert list(result.index.strftime("%Y-%m-%d")) == ["2023-01-03", "2023-01-04", "2023-01-05"]
    assert result["Close"].tolist() == [10.2, 11.2, 12.2]


def test_slice_asof_accepts_a_date_object():
    df = _price_df()
    result = slice_asof(df, date(2023, 1, 5))
    assert list(result.index.strftime("%Y-%m-%d")) == ["2023-01-03", "2023-01-04", "2023-01-05"]


def test_slice_asof_exact_row_date_is_inclusive():
    """as_of landing exactly on a trading day includes that day's row."""
    df = _price_df()
    result = slice_asof(df, "2023-01-06")
    assert result.index[-1].strftime("%Y-%m-%d") == "2023-01-06"


def test_slice_asof_weekend_gap_falls_back_to_last_prior_trading_day():
    """as_of = Sunday 2023-01-08 (no row that day) must return everything up
    to the last prior trading day, Friday 2023-01-06 -- NOT the following
    Monday's row."""
    df = _price_df()
    result = slice_asof(df, "2023-01-08")
    assert result.index[-1].strftime("%Y-%m-%d") == "2023-01-06"
    assert "2023-01-09" not in result.index.strftime("%Y-%m-%d").tolist()


def test_slice_asof_before_all_data_returns_empty_frame():
    df = _price_df()
    result = slice_asof(df, "2020-01-01")
    assert result.empty
    # Columns are preserved even when empty.
    assert list(result.columns) == list(df.columns)


def test_slice_asof_after_all_data_returns_everything():
    df = _price_df()
    result = slice_asof(df, "2030-01-01")
    assert len(result) == len(df)


def test_slice_asof_never_mutates_the_input_frame():
    df = _price_df()
    original = df.copy()
    slice_asof(df, "2023-01-05")
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# _drop_unusable_bars -- regression coverage for the trailing/interior-NaN
# price bug: Stooq (and occasionally yfinance) can hand back a bar for an
# in-progress/unsettled session that carries a Volume estimate but NaN for
# Open/High/Low/Close. Left un-dropped, that single NaN Close poisons every
# rolling-window indicator downstream (compute_indicators reads
# close.iloc[-1] and rolling sma50/sma200 windows), which is exactly what
# silently forced price/sma50/sma200 to None across the whole app before this
# fix. See sec_analyzer/fetch/prices.py::_drop_unusable_bars.
# ---------------------------------------------------------------------------


def _ohlcv_df(closes, volumes, start="2023-01-02"):
    idx = pd.bdate_range(start=start, periods=len(closes))
    idx.name = "Date"
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": volumes},
        index=idx,
    )


def test_drop_unusable_bars_removes_trailing_nan_close_row_but_keeps_earlier_rows():
    closes = [100.0 + i for i in range(10)] + [float("nan")]
    volumes = [1_000_000] * 10 + [2_000_000]   # the unsettled bar still carries a Volume estimate
    df = _ohlcv_df(closes, volumes)

    result = _drop_unusable_bars(df, "FAKE")

    assert len(result) == 10
    assert not result["Close"].isna().any()
    assert result["Close"].iloc[-1] == 109.0
    # The dropped row's index is gone; every earlier row is untouched.
    assert result.index[-1] == df.index[-2]


def test_drop_unusable_bars_removes_an_interior_nan_close_row():
    closes = [100.0, 101.0, float("nan"), 103.0, 104.0]
    volumes = [1_000_000] * 5
    df = _ohlcv_df(closes, volumes)

    result = _drop_unusable_bars(df, "FAKE")

    assert len(result) == 4
    assert not result["Close"].isna().any()
    assert result["Close"].tolist() == [100.0, 101.0, 103.0, 104.0]


def test_drop_unusable_bars_no_bad_rows_returns_frame_unchanged():
    df = _ohlcv_df([100.0, 101.0, 102.0], [1_000_000] * 3)
    result = _drop_unusable_bars(df, "FAKE")
    pd.testing.assert_frame_equal(result, df)


def test_drop_unusable_bars_none_and_empty_frame_pass_through_unchanged():
    assert _drop_unusable_bars(None) is None

    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    result = _drop_unusable_bars(empty)
    assert result.empty
    assert list(result.columns) == list(empty.columns)


def test_drop_unusable_bars_degrades_cleanly_with_no_close_column():
    # No "Close" column at all -- nothing to drop; must not crash, and the
    # missing-column case is left for _validate_frame downstream to reject.
    df = pd.DataFrame({"Open": [1.0, 2.0], "Volume": [100, 200]})
    result = _drop_unusable_bars(df, "FAKE")
    pd.testing.assert_frame_equal(result, df)


class _FakeResponse:
    """Minimal stand-in for requests.Response used by the fetch tests."""

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_get_price_history_drops_trailing_unsettled_bar_from_stooq_and_indicators_survive(
    monkeypatch, tmp_path
):
    """End-to-end regression: a Stooq response shaped exactly like the real
    failure (259 good daily bars, then one trailing bar for an unsettled
    session with a Volume estimate but blank OHLC) must come back from
    ``get_price_history`` with that row dropped -- and, critically, with
    enough clean history left that ``compute_indicators`` can still compute
    ``price``/``sma50``/``sma200`` instead of the whole set silently going
    ``None``."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))

    n_good = 259
    dates = pd.bdate_range("2022-01-03", periods=n_good + 1)
    lines = ["Date,Open,High,Low,Close,Volume"]
    price = 100.0
    for d in dates[:n_good]:
        price += 0.2
        lines.append(f"{d.date()},{price - 0.1:.2f},{price + 0.3:.2f},{price - 0.3:.2f},{price:.2f},1000000")
    # The trailing unsettled-session bar: Volume estimate present, OHLC blank.
    unsettled_date = dates[-1]
    lines.append(f"{unsettled_date.date()},,,,,2000000")
    csv_text = "\n".join(lines) + "\n"

    monkeypatch.setattr(prices.requests, "get", lambda *a, **k: _FakeResponse(csv_text))

    df, source = prices.get_price_history("FAKE", no_cache=True)

    assert source == "stooq"
    assert len(df) == n_good
    assert not df["Close"].isna().any()
    assert df.index[-1] == dates[n_good - 1]   # the unsettled trailing bar is gone
    last_good_close = round(price, 2)
    assert df["Close"].iloc[-1] == last_good_close

    from sec_analyzer.technical.indicators import compute_indicators

    ind = compute_indicators(df)
    assert ind["price"] == last_good_close
    assert ind["sma50"] is not None
    assert ind["sma200"] is not None
