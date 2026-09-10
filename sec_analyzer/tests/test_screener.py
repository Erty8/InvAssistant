"""Unit tests for the S&P 500 swing screener's universe loading, scan
orchestration, and persistence (SWING_SPEC.md Sec.1, 4, 5).

No network access anywhere in this module: ``scan_swing``'s price layer
(``sec_analyzer.fetch.prices.get_price_history``) and its universe loader are
monkeypatched, and persistence tests use a fresh SQLite file under
pytest's ``tmp_path`` (mirroring ``test_store.py``'s convention).
"""

import pandas as pd
import pytest

from sec_analyzer.fetch.prices import PriceDataError
from sec_analyzer.screener import swing_scan
from sec_analyzer.screener.swing_scan import scan_swing
from sec_analyzer.screener.universe import (
    NASDAQ100_CSV_PATH,
    SP500_CSV_PATH,
    load_universe,
    normalize_index,
    price_symbol,
    universe_label,
)
from sec_analyzer.store import database


# ---------------------------------------------------------------------------
# price_symbol (invariant 6)
# ---------------------------------------------------------------------------


def test_price_symbol_dotted_class_share_becomes_dashed():
    assert price_symbol("BRK.B") == "BRK-B"
    assert price_symbol("BF.B") == "BF-B"


def test_price_symbol_strips_and_uppercases_plain_ticker():
    assert price_symbol(" aapl ") == "AAPL"


def test_price_symbol_none_or_blank_returns_empty_string():
    assert price_symbol(None) == ""
    assert price_symbol("") == ""
    assert price_symbol("   ") == ""


# ---------------------------------------------------------------------------
# load_universe (invariant 7)
# ---------------------------------------------------------------------------


def test_load_universe_bundled_csv_meets_size_uniqueness_and_order_invariants():
    rows = load_universe()

    assert len(rows) >= 490
    tickers = [r["ticker"] for r in rows]
    assert len(tickers) == len(set(tickers))          # unique
    assert tickers == sorted(tickers)                  # ascending
    for row in rows:
        assert set(row) == {"ticker", "name", "sector", "cik"}


def test_load_universe_default_path_matches_sp500_csv_path_constant():
    assert load_universe() == load_universe(path=SP500_CSV_PATH)


def test_load_universe_dedupes_case_insensitively_first_occurrence_wins(tmp_path):
    csv_path = tmp_path / "sp500.csv"
    csv_path.write_text(
        "ticker,name,sector,cik\n"
        "MSFT,Microsoft,Tech,0000789019\n"
        "AAPL,Apple,Tech,0000320193\n"
        "aapl,Apple Duplicate,Tech,0000320193\n"
        "BRK.B,Berkshire,Financials,0001067983\n",
        encoding="utf-8",
    )

    rows = load_universe(path=str(csv_path))

    assert [r["ticker"] for r in rows] == ["AAPL", "BRK.B", "MSFT"]
    # First occurrence wins -- the later "aapl" duplicate row is dropped, not
    # overwriting the earlier "Apple" name.
    aapl = next(r for r in rows if r["ticker"] == "AAPL")
    assert aapl["name"] == "Apple"


def test_load_universe_skips_blank_and_short_lines(tmp_path):
    csv_path = tmp_path / "sp500.csv"
    csv_path.write_text(
        "ticker,name,sector,cik\n"
        "AAPL,Apple,Tech,0000320193\n"
        "\n"
        "TOOFEW\n"
        "MSFT,Microsoft,Tech,0000789019\n",
        encoding="utf-8",
    )

    rows = load_universe(path=str(csv_path))

    assert [r["ticker"] for r in rows] == ["AAPL", "MSFT"]


def test_load_universe_raises_oserror_for_missing_file():
    with pytest.raises(OSError):
        load_universe(path="Z:/definitely/does/not/exist/sp500.csv")


# ---------------------------------------------------------------------------
# Multi-index universes: NDX support + normalize_index (invariant 13)
# ---------------------------------------------------------------------------


def test_load_universe_no_args_still_returns_sp500_rows():
    # Backward compatibility: existing callers with no arguments must keep
    # getting the full S&P 500 list, unaffected by adding NDX support.
    rows = load_universe()
    assert len(rows) == 503
    tickers = [r["ticker"] for r in rows]
    assert len(tickers) == len(set(tickers))
    assert tickers == sorted(tickers)


def test_load_universe_ndx_returns_103_unique_ascending_rows():
    rows = load_universe("NDX")
    assert len(rows) == 103
    tickers = [r["ticker"] for r in rows]
    assert len(tickers) == len(set(tickers))
    assert tickers == sorted(tickers)
    for row in rows:
        assert set(row) == {"ticker", "name", "sector", "cik"}


def test_load_universe_ndx_matches_nasdaq100_csv_path_constant():
    assert load_universe("NDX") == load_universe(path=NASDAQ100_CSV_PATH)


def test_normalize_index_blank_and_unrecognized_fall_back_to_sp500():
    assert normalize_index(None) == "SP500"
    assert normalize_index("") == "SP500"
    assert normalize_index("   ") == "SP500"
    assert normalize_index("garbage") == "SP500"


def test_normalize_index_accepts_ndx_and_its_aliases_case_insensitively():
    assert normalize_index("ndx") == "NDX"
    assert normalize_index("NDX") == "NDX"
    assert normalize_index("nasdaq100") == "NDX"
    assert normalize_index("NASDAQ100") == "NDX"
    assert normalize_index("Nasdaq-100") == "NDX"
    assert normalize_index("ndx100") == "NDX"


def test_universe_label_known_and_unknown_codes():
    assert universe_label("SP500") == "S&P 500"
    assert universe_label("NDX") == "Nasdaq 100"
    # Unknown codes are returned unchanged rather than raising (display-only).
    assert universe_label("XYZ") == "XYZ"


# ---------------------------------------------------------------------------
# scan_swing orchestration (invariants 8, 9; progress_cb / benchmark rules)
# ---------------------------------------------------------------------------


def _swing_df(slope, n=260, start="2022-01-03"):
    """A long, steadily-trending OHLCV frame -- long enough (260 bars) for
    SMA200 + its 21-bar slope lookback, Bollinger-squeeze percentile ranking,
    and ATR14 to all be computable, so a realistic ``compute_swing_score``
    run over this data clears the Sec.3.0 coverage floor."""
    idx = pd.bdate_range(start=start, periods=n)
    idx.name = "Date"
    closes = [100.0 + i * slope for i in range(n)]
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


@pytest.fixture
def stub_price_layer(monkeypatch):
    """Stub the price layer + universe for scan_swing: SPY plus three good
    tickers (AAA on a steep uptrend; BBB and CCC fed byte-identical price
    histories, so determinism guarantees a tied score for the sort/tie-break
    test) and one ticker (ZZZ) that always raises -- exactly what a
    delisted/no-data ticker looks like through ``get_price_history``.

    Returns the per-symbol call-count dict so a test can assert the SPY
    benchmark is fetched exactly once.
    """
    calls: "dict[str, int]" = {}
    spy_df = _swing_df(0.15)
    aaa_df = _swing_df(1.0)
    tie_df = _swing_df(0.3)

    def _fake_get_price_history(symbol, no_cache=False):
        calls[symbol] = calls.get(symbol, 0) + 1
        if symbol == "ZZZ":
            raise PriceDataError("no usable data for ZZZ")
        if symbol == "SPY":
            return spy_df, "yfinance"
        if symbol == "AAA":
            return aaa_df, "yfinance"
        if symbol in ("BBB", "CCC"):
            return tie_df, "yfinance"
        raise PriceDataError(f"unexpected symbol {symbol}")

    monkeypatch.setattr(swing_scan, "get_price_history", _fake_get_price_history)
    monkeypatch.setattr(
        swing_scan,
        "load_universe",
        # Accepts the ``index`` kwarg scan_swing now passes through (SWING_SPEC.md
        # Sec.4/Sec.1) but ignores it -- this fixture only ever stands in for a
        # single small universe, regardless of which index code is requested.
        lambda *args, **kwargs: [
            {"ticker": "AAA", "name": "Aaa Co", "sector": "Tech", "cik": "1"},
            {"ticker": "BBB", "name": "Bbb Co", "sector": "Tech", "cik": "2"},
            {"ticker": "CCC", "name": "Ccc Co", "sector": "Tech", "cik": "3"},
            {"ticker": "ZZZ", "name": "Zzz Co", "sector": "Tech", "cik": "4"},
        ],
    )
    return calls


def test_scan_swing_bad_ticker_lands_in_skipped_others_still_score(stub_price_layer):
    result = scan_swing(max_workers=2)

    assert result["total"] == 4
    assert result["count"] == 3
    assert {r["ticker"] for r in result["rows"]} == {"AAA", "BBB", "CCC"}
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["ticker"] == "ZZZ"
    assert result["skipped"][0]["reason"]   # non-empty reason string


def test_scan_swing_rows_sorted_score_desc_then_ticker_asc(stub_price_layer):
    result = scan_swing(max_workers=2)
    rows = result["rows"]

    assert rows == sorted(rows, key=lambda r: (-r["score"], r["ticker"]))

    # BBB and CCC were fed byte-identical price histories -> by the
    # determinism invariant their scores must be exactly equal, so the sort
    # can only be resolving their order via the ticker-ascending tie-break.
    bbb = next(r for r in rows if r["ticker"] == "BBB")
    ccc = next(r for r in rows if r["ticker"] == "CCC")
    assert bbb["score"] == ccc["score"]
    assert rows.index(bbb) < rows.index(ccc)


def test_scan_swing_progress_cb_called_once_per_ticker_with_final_totals(stub_price_layer):
    calls = []
    scan_swing(max_workers=2, progress_cb=lambda done, total, ticker: calls.append((done, total, ticker)))

    assert len(calls) == 4
    assert [c[0] for c in calls] == [1, 2, 3, 4]      # done increments 1..total
    assert all(c[1] == 4 for c in calls)               # total is stable throughout
    assert {c[2] for c in calls} == {"AAA", "BBB", "CCC", "ZZZ"}


def test_scan_swing_broken_progress_cb_does_not_break_the_scan(stub_price_layer):
    def _broken_cb(done, total, ticker):
        raise RuntimeError("boom")

    result = scan_swing(max_workers=2, progress_cb=_broken_cb)

    assert result["count"] == 3
    assert len(result["skipped"]) == 1


def test_scan_swing_fetches_spy_benchmark_exactly_once(stub_price_layer):
    scan_swing(max_workers=2)
    assert stub_price_layer["SPY"] == 1


def test_scan_swing_index_ndx_scans_nasdaq100_universe_and_tags_result(monkeypatch):
    """Invariant 15: scan_swing(index="NDX") requests the Nasdaq-100 universe
    (not the S&P 500 default) and the returned payload carries the resolved
    index code and its display label."""
    load_universe_calls = []
    price_df = _swing_df(1.0)

    def _fake_get_price_history(symbol, no_cache=False):
        return price_df, "yfinance"

    def _fake_load_universe(index="SP500", path=None):
        load_universe_calls.append(index)
        return [{"ticker": "NVDA", "name": "Nvidia", "sector": "Tech", "cik": "1"}]

    monkeypatch.setattr(swing_scan, "get_price_history", _fake_get_price_history)
    monkeypatch.setattr(swing_scan, "load_universe", _fake_load_universe)

    result = scan_swing(index="NDX", max_workers=2)

    assert load_universe_calls == ["NDX"]
    assert result["universe"] == "NDX"
    assert result["universe_label"] == "Nasdaq 100"
    assert result["total"] == 1


def test_scan_swing_tickers_filter_restricts_universe_and_allows_unknown(stub_price_layer):
    # An unknown ticker not in the (stubbed) universe is still scanned, with
    # name/sector left None (SWING_SPEC.md Sec.4 step 1) -- but here it also
    # isn't wired into the fake price layer, so it lands in skipped.
    result = scan_swing(tickers=["AAA", "UNKNOWN"], max_workers=2)

    assert result["total"] == 2
    row_tickers = {r["ticker"] for r in result["rows"]}
    skipped_tickers = {s["ticker"] for s in result["skipped"]}
    assert row_tickers | skipped_tickers == {"AAA", "UNKNOWN"}
    if "AAA" in row_tickers:
        aaa_row = next(r for r in result["rows"] if r["ticker"] == "AAA")
        assert aaa_row["name"] == "Aaa Co"


# ---------------------------------------------------------------------------
# save_swing_scan / load_latest_swing_scan (invariant 10)
# ---------------------------------------------------------------------------


def _sample_scan_payload():
    return {
        "generated_at": "2026-07-27T09:12:03Z",
        "price_as_of": "2026-07-24",
        "universe": "SP500",
        "count": 2,
        "total": 2,
        "skipped": [],
        "rows": [
            {"ticker": "AAA", "score": 80, "label": "FIRSAT"},
            {"ticker": "BBB", "score": 60, "label": "NÖTR"},
        ],
    }


def test_save_and_load_swing_scan_round_trips_through_temp_db(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    payload = _sample_scan_payload()

    row_id = database.save_swing_scan(payload, db_path=db_path)
    assert row_id and row_id > 0

    loaded = database.load_latest_swing_scan(db_path=db_path)
    assert loaded == payload


def test_load_latest_swing_scan_returns_none_on_empty_db(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    assert database.load_latest_swing_scan(db_path=db_path) is None


def test_load_latest_swing_scan_returns_most_recently_saved_payload(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    first = _sample_scan_payload()
    second = _sample_scan_payload()
    second["generated_at"] = "2026-07-27T10:00:00Z"
    second["rows"] = [{"ticker": "CCC", "score": 99, "label": "GÜÇLÜ FIRSAT"}]

    database.save_swing_scan(first, db_path=db_path)
    database.save_swing_scan(second, db_path=db_path)

    loaded = database.load_latest_swing_scan(db_path=db_path)
    assert loaded == second


def test_load_latest_swing_scan_per_index_persistence(tmp_path):
    """Invariant 14: each index keeps its own independent scan history --
    persisting an NDX scan must never make the stored SP500 scan
    unreachable, and universe-filtered lookups must not cross-contaminate."""
    db_path = str(tmp_path / "test.sqlite3")
    sp500_payload = _sample_scan_payload()
    ndx_payload = _sample_scan_payload()
    ndx_payload["universe"] = "NDX"
    ndx_payload["generated_at"] = "2026-07-27T10:00:00Z"
    ndx_payload["rows"] = [{"ticker": "NVDA", "score": 90, "label": "GÜÇLÜ FIRSAT"}]

    database.save_swing_scan(sp500_payload, db_path=db_path)
    database.save_swing_scan(ndx_payload, db_path=db_path)

    assert database.load_latest_swing_scan(db_path=db_path, universe="SP500") == sp500_payload
    assert database.load_latest_swing_scan(db_path=db_path, universe="NDX") == ndx_payload
    # No filter -> most recent of either index (original single-index behaviour).
    assert database.load_latest_swing_scan(db_path=db_path) == ndx_payload


def test_load_latest_swing_scan_universe_filter_returns_none_when_that_index_absent(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    database.save_swing_scan(_sample_scan_payload(), db_path=db_path)  # SP500 only

    assert database.load_latest_swing_scan(db_path=db_path, universe="NDX") is None


def test_save_swing_scan_on_unwritable_db_path_returns_zero_not_raises(tmp_path):
    # A path inside a file (not a directory) can never be opened as a SQLite
    # database -- this must be swallowed and reported as 0, mirroring
    # save_prices' non-fatal posture (SWING_SPEC.md Sec.5), never raise.
    not_a_directory = tmp_path / "not_a_directory"
    not_a_directory.write_text("i am a file, not a directory", encoding="utf-8")
    bad_db_path = str(not_a_directory / "sub" / "test.sqlite3")

    result = database.save_swing_scan(_sample_scan_payload(), db_path=bad_db_path)

    assert result == 0
