"""Unit tests for ``sec_analyzer.fetch.prices``: ``slice_asof`` (point-in-time
price slicing) and ``_drop_unusable_bars`` (the swing-screener price-cleaning
regression, SWING_SPEC.md).

No network access -- these build a small in-memory OHLCV DataFrame directly,
matching the ``Date``-indexed, ascending shape ``get_price_history`` returns;
the tests that exercise ``get_price_history`` itself replace the lazily-
imported ``yfinance`` module via ``sys.modules`` (see ``_fake_yfinance``)
rather than hitting Yahoo Finance.
"""

import os
import sys
from datetime import date, datetime, timedelta, timezone

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


def _fake_yfinance(frame):
    """A stand-in ``yfinance`` module whose ``download`` returns ``frame``.

    ``_fetch_yfinance`` imports yfinance lazily, so injecting the name into
    ``sys.modules`` is the seam for every fetch-path test. It replaced
    monkeypatching ``prices.requests.get``, which stopped existing when the
    Stooq path was removed (Stooq began serving a JS bot-check in Aug 2026).
    """

    class _FakeYf:
        @staticmethod
        def download(_ticker, period=None, interval=None, progress=None):
            return frame.copy()

    return _FakeYf


def _no_network_yfinance():
    """A stand-in ``yfinance`` that fails the test if anything fetches."""

    class _FakeYf:
        @staticmethod
        def download(*_a, **_k):
            raise AssertionError("network should not be hit on a fresh cache hit")

    return _FakeYf


def test_get_price_history_drops_trailing_nan_bar_and_indicators_survive(
    monkeypatch, tmp_path
):
    """End-to-end regression: an upstream response shaped exactly like the
    real failure (259 good daily bars, then one trailing bar for an unsettled
    session with a Volume estimate but blank OHLC) must come back from
    ``get_price_history`` with that row dropped -- and, critically, with
    enough clean history left that ``compute_indicators`` can still compute
    ``price``/``sma50``/``sma200`` instead of the whole set silently going
    ``None``."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))

    n_good = 259
    dates = pd.bdate_range("2022-01-03", periods=n_good + 1)
    price = 100.0
    rows = []
    for _d in dates[:n_good]:
        price += 0.2
        rows.append({"Open": price - 0.1, "High": price + 0.3,
                     "Low": price - 0.3, "Close": price, "Volume": 1000000})
    # The trailing unsettled-session bar: Volume estimate present, OHLC blank.
    rows.append({"Open": float("nan"), "High": float("nan"), "Low": float("nan"),
                 "Close": float("nan"), "Volume": 2000000})
    fetched = pd.DataFrame(rows, index=dates)

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(fetched))

    df, source = prices.get_price_history("FAKE", no_cache=True)

    assert source == "yfinance"
    assert len(df) == n_good
    assert not df["Close"].isna().any()
    assert df.index[-1] == dates[n_good - 1]   # the unsettled trailing bar is gone
    assert df["Close"].iloc[-1] == pytest.approx(price)

    from sec_analyzer.technical.indicators import compute_indicators

    ind = compute_indicators(df)
    assert ind["price"] == pytest.approx(price)
    assert ind["sma50"] is not None
    assert ind["sma200"] is not None


# ---------------------------------------------------------------------------
# The in-progress-bar cache poisoning regression (ORCL 2026-08-03).
#
# A run during market hours cached a bar stamped with that day's date but
# holding a mid-session Close. From the next day on, that date made the cache
# look fresh forever, so the real close was never fetched: ORCL's report
# showed $127.89 (a Friday 14:48 UTC snapshot, 13.2M shares) where the actual
# 07-31 close was $129.87 on 34.4M shares.
#
# The invariant under test: a frame handed to a caller may end in an unsettled
# bar (during market hours that is the live price), but the on-disk cache must
# only ever hold settled sessions.
# See prices.drop_unsettled_bars / prices._drop_partial_trailing_bar.
# ---------------------------------------------------------------------------


def _settled_through_last_session(n=40, start_price=100.0):
    """An OHLCV frame whose newest bar IS the last completed session."""
    dates = pd.bdate_range(end=pd.Timestamp(prices.last_completed_session()), periods=n)
    rows = []
    price = start_price
    for d in dates:
        price += 0.5
        rows.append({"Date": d, "Open": price - 0.3, "High": price + 0.5,
                     "Low": price - 0.5, "Close": price, "Volume": 1_000_000})
    return pd.DataFrame(rows).set_index("Date").sort_index()


def _with_unsettled_bar(df, close=999.0, volume=13_224_098):
    """Append a bar for a session that has not published yet -- the shape of a
    mid-session snapshot: a real (but partial) Close on thin volume."""
    unsettled = pd.Timestamp(prices.last_completed_session() + timedelta(days=1))
    extra = pd.DataFrame(
        [{"Open": close - 2, "High": close + 2, "Low": close - 3,
          "Close": close, "Volume": volume}],
        index=pd.DatetimeIndex([unsettled], name="Date"),
    )
    return pd.concat([df, extra])


def _csv_text(df):
    """Render a Date-indexed OHLCV frame the way Stooq's CSV endpoint does."""
    lines = ["Date,Open,High,Low,Close,Volume"]
    for d, row in df.iterrows():
        lines.append(
            f"{d.date()},{row.Open:.2f},{row.High:.2f},{row.Low:.2f},"
            f"{row.Close:.2f},{int(row.Volume)}"
        )
    return "\n".join(lines) + "\n"


def test_drop_unsettled_bars_removes_a_bar_for_an_unpublished_session():
    settled = _settled_through_last_session(10)
    df = _with_unsettled_bar(settled)

    result = prices.drop_unsettled_bars(df, "FAKE")

    assert len(result) == 10
    assert result.index.max().date() == prices.last_completed_session()


def test_drop_unsettled_bars_leaves_a_fully_settled_frame_untouched():
    settled = _settled_through_last_session(10)

    result = prices.drop_unsettled_bars(settled, "FAKE")

    assert result.index.equals(settled.index)


def test_drop_unsettled_bars_none_and_empty_pass_through():
    assert prices.drop_unsettled_bars(None) is None
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    assert prices.drop_unsettled_bars(empty).empty


def test_drop_unsettled_bars_keeps_frame_when_every_bar_is_unsettled():
    """An all-unsettled frame would be rejected by _validate_frame anyway;
    emptying it here would write a useless zero-row cache."""
    only_unsettled = _with_unsettled_bar(
        pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]).astype(float)
    )

    result = prices.drop_unsettled_bars(only_unsettled, "FAKE")

    assert len(result) == 1


def test_cache_never_persists_the_in_progress_bar_the_caller_receives(
    monkeypatch, tmp_path
):
    """The core ORCL fix: the returned frame keeps today's live bar, the cache
    written to disk stops at the last settled session."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    fetched = _with_unsettled_bar(_settled_through_last_session(40), close=134.95)
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(fetched))

    df, source = prices.get_price_history("FAKE", no_cache=True)

    assert source == "yfinance"
    # The caller still sees the live intraday bar...
    assert df.index.max().date() > prices.last_completed_session()
    assert df["Close"].iloc[-1] == pytest.approx(134.95)

    # ...but it was not persisted.
    on_disk = pd.read_csv(
        tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv", parse_dates=["Date"]
    ).set_index("Date")
    assert on_disk.index.max().date() == prices.last_completed_session()
    assert 134.95 not in set(on_disk["Close"])


def test_legacy_cache_written_mid_session_is_healed_and_refetched(
    monkeypatch, tmp_path
):
    """A cache file already on disk whose newest bar was captured mid-session
    must not be trusted: its date says "fresh", its mtime proves it partial.
    Without this the file never heals -- it looks fresh every single day."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    poisoned = _settled_through_last_session(40)
    poisoned.to_csv(cache_path, index_label="Date")

    # Written during the newest bar's own session, hours before its close
    # published -- exactly how the ORCL cache was created.
    session = prices.last_completed_session()
    mid_session = datetime(
        session.year, session.month, session.day,
        prices._SESSION_PUBLISHED_HOUR_UTC - 7, tzinfo=timezone.utc,
    ).timestamp()
    os.utime(cache_path, (mid_session, mid_session))

    monkeypatch.setitem(
        sys.modules, "yfinance", _fake_yfinance(_settled_through_last_session(40))
    )

    _df, source = prices.get_price_history("FAKE")

    assert source == "yfinance", "a mid-session cache must be discarded, not served"


def test_cache_written_after_the_close_is_still_a_fresh_hit(monkeypatch, tmp_path):
    """The healing check must not fire on a legitimately post-close cache --
    that would re-fetch every ticker on every run."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    _settled_through_last_session(40).to_csv(cache_path, index_label="Date")

    session = prices.last_completed_session()
    after_close = datetime(
        session.year, session.month, session.day,
        prices._SESSION_PUBLISHED_HOUR_UTC, tzinfo=timezone.utc,
    ).timestamp() + 60
    os.utime(cache_path, (after_close, after_close))

    monkeypatch.setitem(sys.modules, "yfinance", _no_network_yfinance())

    _df, source = prices.get_price_history("FAKE")

    assert source == "cache(unknown)"


def test_healed_cache_is_rewritten_so_the_partial_bar_stops_coming_back(
    monkeypatch, tmp_path
):
    """Healing on read is not enough on its own: if the file keeps its partial
    bar, every later read re-discards it. The served frame and the file must
    agree."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    settled = _settled_through_last_session(40)
    _with_unsettled_bar(settled, close=134.95).to_csv(cache_path, index_label="Date")

    # Written mid-session on the unsettled bar's own day -- stamped explicitly
    # rather than from the real clock, so the test does not change meaning
    # depending on the hour it runs at.
    partial_day = prices.last_completed_session() + timedelta(days=1)
    mid_session = datetime(
        partial_day.year, partial_day.month, partial_day.day,
        prices._SESSION_PUBLISHED_HOUR_UTC - 7, tzinfo=timezone.utc,
    ).timestamp()
    os.utime(cache_path, (mid_session, mid_session))

    monkeypatch.setitem(sys.modules, "yfinance", _no_network_yfinance())

    df, source = prices.get_price_history("FAKE")

    assert source == "cache(unknown)"
    assert df.index.max().date() == prices.last_completed_session()

    on_disk = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
    assert on_disk.index.max().date() == prices.last_completed_session()
    assert 134.95 not in set(on_disk["Close"])

    # Second read: the file is now clean, so it is a plain hit with nothing
    # left to heal.
    df2, source2 = prices.get_price_history("FAKE")
    assert source2 == "cache(unknown)"
    assert len(df2) == len(df)


# ---------------------------------------------------------------------------
# prefer_live -- the other half of the ORCL 2026-08-03 complaint.
#
# Once the cache holds the last SETTLED session it is fresh by design, so a
# run at midday kept reporting the previous close ($129.87) while the stock
# was trading at $138.21. That is the right answer for anything ranking on
# daily bars (screener, backtests) and the wrong one for a report printing a
# current price -- hence an opt-in flag rather than a global behavior change.
# ---------------------------------------------------------------------------


def test_session_in_progress_is_false_on_weekends():
    saturday = datetime(2026, 8, 1, 17, 0, tzinfo=timezone.utc)
    assert prices.session_in_progress(saturday) is False


def test_session_in_progress_spans_the_us_cash_session():
    day = date(2026, 8, 3)  # a Monday

    def at(hour):
        return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)

    assert prices.session_in_progress(at(9)) is False    # pre-open
    assert prices.session_in_progress(at(14)) is True    # mid-session
    assert prices.session_in_progress(at(21)) is True    # closed but unpublished
    assert prices.session_in_progress(at(22)) is False   # published


def test_prefer_live_refetches_a_settled_cache_while_a_session_is_open(
    monkeypatch, tmp_path
):
    """The user-visible ORCL symptom: cache fresh through Friday, market open
    on Monday, and the report kept printing Friday's close."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv"
    settled = _settled_through_last_session(40)
    prices._write_cache(str(cache_path), settled, source="yfinance")

    live = _with_unsettled_bar(settled, close=138.21)
    monkeypatch.setattr(prices, "session_in_progress", lambda *_a, **_k: True)
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(live))

    df, source = prices.get_price_history("FAKE", prefer_live=True)

    assert source == "yfinance"
    assert df["Close"].iloc[-1] == pytest.approx(138.21)

    # Still settled-only on disk -- prefer_live changes what is served, never
    # what is stored.
    on_disk = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
    assert on_disk.index.max().date() == prices.last_completed_session()


def test_prefer_live_is_a_plain_cache_hit_outside_market_hours(monkeypatch, tmp_path):
    """No session in progress means there is no live bar to go get."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = str(tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv")
    prices._write_cache(cache_path, _settled_through_last_session(40), source="yfinance")

    monkeypatch.setattr(prices, "session_in_progress", lambda *_a, **_k: False)
    monkeypatch.setitem(sys.modules, "yfinance", _no_network_yfinance())

    _df, source = prices.get_price_history("FAKE", prefer_live=True)

    assert source == "cache(yfinance)"


def test_default_still_serves_the_settled_cache_during_market_hours(
    monkeypatch, tmp_path
):
    """The screener and backtests must keep ranking on settled closes; without
    prefer_live an open session changes nothing."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = str(tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv")
    prices._write_cache(cache_path, _settled_through_last_session(40), source="yfinance")

    monkeypatch.setattr(prices, "session_in_progress", lambda *_a, **_k: True)
    monkeypatch.setitem(sys.modules, "yfinance", _no_network_yfinance())

    _df, source = prices.get_price_history("FAKE")

    assert source == "cache(yfinance)"


def test_prefer_live_falls_back_to_the_settled_cache_when_the_fetch_fails(
    monkeypatch, tmp_path
):
    """Wanting a live price must not turn a working cached price into an
    error: the settled close is a fine answer when the upstream is down."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = str(tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv")
    prices._write_cache(cache_path, _settled_through_last_session(40), source="yfinance")

    monkeypatch.setattr(prices, "session_in_progress", lambda *_a, **_k: True)
    monkeypatch.setitem(sys.modules, "yfinance", None)  # ImportError on import

    df, source = prices.get_price_history("FAKE", prefer_live=True)

    assert source == "stale-cache(yfinance)"
    assert df.index.max().date() == prices.last_completed_session()


# ---------------------------------------------------------------------------
# The fallback chain: yfinance -> Yahoo chart (identical basis) -> Nasdaq
# (split-adjusted only). Stooq's removal left no fallback at all; these cover
# the replacement, and above all the guard that keeps Nasdaq's different price
# basis out of the valuation layer.
# ---------------------------------------------------------------------------


def _yahoo_chart_payload(n=40, start_price=100.0, adj_ratio=0.9):
    """A Yahoo chart-endpoint payload. ``adj_ratio`` scales adjclose below
    close, the way a dividend history does."""
    dates = pd.bdate_range(end=pd.Timestamp(prices.last_completed_session()), periods=n)
    stamps = [int(pd.Timestamp(d).timestamp()) + 13 * 3600 for d in dates]
    closes, opens, highs, lows, vols = [], [], [], [], []
    price = start_price
    for _d in dates:
        price += 0.5
        closes.append(price)
        opens.append(price - 0.3)
        highs.append(price + 0.5)
        lows.append(price - 0.5)
        vols.append(1_000_000)
    return {
        "chart": {
            "result": [
                {
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [{"open": opens, "high": highs, "low": lows,
                                   "close": closes, "volume": vols}],
                        "adjclose": [{"adjclose": [c * adj_ratio for c in closes]}],
                    },
                }
            ],
            "error": None,
        }
    }


def _nasdaq_payload(n=40, start_price=100.0):
    """A Nasdaq historical payload, including its display formatting."""
    dates = pd.bdate_range(end=pd.Timestamp(prices.last_completed_session()), periods=n)
    rows = []
    price = start_price
    for d in dates:
        price += 0.5
        rows.append({
            "date": d.strftime("%m/%d/%Y"),
            "close": f"${price:,.2f}",
            "volume": f"{1_234_567:,}",
            "open": f"${price - 0.3:,.2f}",
            "high": f"${price + 0.5:,.2f}",
            "low": f"${price - 0.5:,.2f}",
        })
    rows.reverse()  # Nasdaq returns newest-first
    return {"data": {"symbol": "FAKE", "totalRecords": len(rows),
                     "tradesTable": {"rows": rows}}, "status": {"rCode": 200}}


class _JsonResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _broken_yfinance():
    class _FakeYf:
        @staticmethod
        def download(*_a, **_k):
            raise RuntimeError("yfinance internals changed again")

    return _FakeYf


def _raise_request_error(*_a, **_k):
    raise prices.requests.RequestException("down")


# --- source classification -------------------------------------------------


def test_upstream_of_unwraps_every_source_form():
    assert prices.upstream_of("yfinance") == "yfinance"
    assert prices.upstream_of("cache(nasdaq)") == "nasdaq"
    assert prices.upstream_of("stale-cache(yahoo-chart)") == "yahoo-chart"
    assert prices.upstream_of(None) == ""


def test_is_total_return_basis_accepts_the_yahoo_pair_and_rejects_nasdaq():
    assert prices.is_total_return_basis("yfinance") is True
    assert prices.is_total_return_basis("yahoo-chart") is True
    assert prices.is_total_return_basis("cache(yahoo-chart)") is True
    # Split-adjusted only -- the ORCL measurement was -14% ten years back.
    assert prices.is_total_return_basis("nasdaq") is False
    assert prices.is_total_return_basis("cache(nasdaq)") is False


def test_is_total_return_basis_is_false_for_unrecorded_sources():
    """Guessing "probably fine" would fail silently and expensively."""
    assert prices.is_total_return_basis("unknown") is False
    assert prices.is_total_return_basis(None) is False
    assert prices.is_total_return_basis("some-new-source") is False


# --- the Yahoo chart fallback ----------------------------------------------


def test_yahoo_chart_reproduces_yfinance_auto_adjust(monkeypatch):
    """Close becomes adjclose, OHLC scale by the same ratio, Volume is left
    alone -- verified against yfinance on real ORCL data (2026-08-03)."""
    payload = _yahoo_chart_payload(40, adj_ratio=0.9)
    monkeypatch.setattr(prices.requests, "get", lambda *a, **k: _JsonResponse(payload))

    df = prices._fetch_yahoo_chart("FAKE")

    raw_close = payload["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    raw_open = payload["chart"]["result"][0]["indicators"]["quote"][0]["open"]
    assert df["Close"].iloc[-1] == pytest.approx(raw_close[-1] * 0.9)
    assert df["Open"].iloc[-1] == pytest.approx(raw_open[-1] * 0.9)
    assert df["Volume"].iloc[-1] == 1_000_000  # untouched by the adjustment
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_yahoo_chart_url_never_uses_the_range_max_granularity_trap(monkeypatch):
    """``range=max&interval=1d`` silently returns ~quarterly bars (measured:
    163 bars, 92-day median gap). The request must use period1/period2."""
    seen = {}

    def _capture(url, *a, **k):
        seen["url"] = url
        return _JsonResponse(_yahoo_chart_payload(40))

    monkeypatch.setattr(prices.requests, "get", _capture)
    prices._fetch_yahoo_chart("FAKE")

    assert "range=" not in seen["url"]
    assert "period1=" in seen["url"] and "period2=" in seen["url"]
    assert "interval=1d" in seen["url"]


def test_yahoo_chart_raises_on_an_empty_result(monkeypatch):
    monkeypatch.setattr(
        prices.requests, "get",
        lambda *a, **k: _JsonResponse({"chart": {"result": [], "error": "Not Found"}}),
    )
    with pytest.raises(prices.PriceDataError):
        prices._fetch_yahoo_chart("NOSUCH")


# --- the Nasdaq fallback ---------------------------------------------------


def test_nasdaq_parses_display_formatted_numbers(monkeypatch):
    monkeypatch.setattr(
        prices.requests, "get", lambda *a, **k: _JsonResponse(_nasdaq_payload(40))
    )

    df = prices._fetch_nasdaq("FAKE")

    assert len(df) == 40
    assert df.index.is_monotonic_increasing, "Nasdaq returns newest-first"
    assert df["Volume"].iloc[-1] == pytest.approx(1_234_567)
    assert df["Close"].iloc[-1] > 0


def test_parse_nasdaq_number_handles_junk():
    assert prices._parse_nasdaq_number("$1,234.56") == pytest.approx(1234.56)
    assert pd.isna(prices._parse_nasdaq_number(None))
    assert pd.isna(prices._parse_nasdaq_number("N/A"))


def test_nasdaq_raises_on_the_rows_none_shape(monkeypatch):
    """Nasdaq answers HTTP 200 with rows=None for an unknown symbol or a
    window it will not serve."""
    monkeypatch.setattr(
        prices.requests, "get",
        lambda *a, **k: _JsonResponse(
            {"data": {"totalRecords": 0, "tradesTable": {"rows": None}},
             "status": {"rCode": 200}}
        ),
    )
    with pytest.raises(prices.PriceDataError):
        prices._fetch_nasdaq("NOSUCH")


# --- chain order and degradation ------------------------------------------


def test_chain_falls_through_to_yahoo_chart_when_the_package_breaks(
    monkeypatch, tmp_path
):
    """The failure this fallback exists for: the yfinance package breaking
    while Yahoo itself is perfectly reachable."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "yfinance", _broken_yfinance())
    monkeypatch.setattr(
        prices.requests, "get", lambda *a, **k: _JsonResponse(_yahoo_chart_payload(40))
    )

    _df, source = prices.get_price_history("FAKE", no_cache=True)

    assert source == "yahoo-chart"
    assert prices.is_total_return_basis(source) is True


def test_chain_falls_through_to_nasdaq_when_both_yahoo_paths_fail(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "yfinance", _broken_yfinance())

    def _yahoo_down_nasdaq_up(url, *a, **k):
        if "nasdaq.com" in url:
            return _JsonResponse(_nasdaq_payload(40))
        raise prices.requests.RequestException("Yahoo unreachable")

    monkeypatch.setattr(prices.requests, "get", _yahoo_down_nasdaq_up)

    _df, source = prices.get_price_history("FAKE", no_cache=True)

    assert source == "nasdaq"
    assert prices.is_total_return_basis(source) is False


def test_chain_raises_with_every_error_when_all_sources_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "yfinance", _broken_yfinance())
    monkeypatch.setattr(prices.requests, "get", _raise_request_error)

    with pytest.raises(prices.PriceDataError) as excinfo:
        prices.get_price_history("FAKE", no_cache=True)

    message = str(excinfo.value)
    for source in ("yfinance", "yahoo-chart", "nasdaq"):
        assert source in message, "the error must say what was tried"


def test_a_nasdaq_cache_is_never_a_plain_hit(monkeypatch, tmp_path):
    """A last-resort cache must not become permanent just because it covers
    the last session: the preferred sources get retried."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = str(tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv")
    prices._write_cache(cache_path, _settled_through_last_session(40), source="nasdaq")

    monkeypatch.setattr(prices, "session_in_progress", lambda *_a, **_k: False)
    monkeypatch.setitem(sys.modules, "yfinance", _broken_yfinance())
    monkeypatch.setattr(
        prices.requests, "get", lambda *a, **k: _JsonResponse(_yahoo_chart_payload(40))
    )

    _df, source = prices.get_price_history("FAKE")

    assert source == "yahoo-chart", "a degraded cache must trigger a retry"


def test_a_yahoo_chart_cache_is_a_plain_hit(monkeypatch, tmp_path):
    """yahoo-chart is basis-identical to yfinance, so its cache is good data
    and must NOT trigger a pointless retry on every run."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = str(tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv")
    prices._write_cache(cache_path, _settled_through_last_session(40), source="yahoo-chart")

    monkeypatch.setattr(prices, "session_in_progress", lambda *_a, **_k: False)
    monkeypatch.setitem(sys.modules, "yfinance", _no_network_yfinance())

    _df, source = prices.get_price_history("FAKE")

    assert source == "cache(yahoo-chart)"


def test_a_nasdaq_cache_still_serves_when_every_retry_fails(monkeypatch, tmp_path):
    """Retrying the preferred sources must not turn a usable degraded cache
    into an outright failure."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    cache_path = str(tmp_path / f"prices_FAKE{prices._CACHE_SUFFIX}.csv")
    prices._write_cache(cache_path, _settled_through_last_session(40), source="nasdaq")

    monkeypatch.setattr(prices, "session_in_progress", lambda *_a, **_k: False)
    monkeypatch.setitem(sys.modules, "yfinance", _broken_yfinance())
    monkeypatch.setattr(prices.requests, "get", _raise_request_error)

    df, source = prices.get_price_history("FAKE")

    assert source == "stale-cache(nasdaq)"
    assert len(df) == 40
