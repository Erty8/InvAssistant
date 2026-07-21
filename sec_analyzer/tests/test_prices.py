"""Unit tests for ``fetch.prices.slice_asof`` (point-in-time price slicing).

No network access -- these build a small in-memory OHLCV DataFrame directly,
matching the ``Date``-indexed, ascending shape ``get_price_history`` returns.
"""

from datetime import date

import pandas as pd
import pytest

from sec_analyzer.fetch.prices import slice_asof


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
