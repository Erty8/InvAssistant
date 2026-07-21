"""Unit tests for the new sec_analyzer.store.database tables: `prices` and
`verdicts`.

Each test uses a fresh SQLite file under pytest's `tmp_path`, so nothing
here touches the package's default database or any other test's state.
"""

import json

import pytest

from sec_analyzer.store import database


def _price_row(date, price):
    return {
        "date": date,
        "open": price - 0.5,
        "high": price + 0.5,
        "low": price - 1.0,
        "close": price,
        "volume": 1_000_000,
    }


# ---------------------------------------------------------------------------
# save_prices
# ---------------------------------------------------------------------------


def test_save_prices_writes_rows_and_creates_schema(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    rows = [_price_row("2023-01-02", 100.0), _price_row("2023-01-03", 101.5)]

    written = database.save_prices("320193", rows, db_path=db_path)

    assert written == 2
    conn = database.get_connection(db_path)
    try:
        stored = conn.execute("SELECT * FROM prices ORDER BY date").fetchall()
    finally:
        conn.close()
    assert len(stored) == 2
    assert stored[0]["cik"] == "320193"
    assert stored[0]["date"] == "2023-01-02"
    assert stored[0]["close"] == 100.0
    assert stored[1]["close"] == 101.5


def test_save_prices_upsert_is_idempotent(tmp_path):
    """Saving the same rows twice must not duplicate them (same row count
    after the second call as after the first), and re-saving with a
    changed close must overwrite the old value in place."""
    db_path = str(tmp_path / "test.sqlite3")
    rows = [_price_row("2023-01-02", 100.0), _price_row("2023-01-03", 101.5)]

    first_count = database.save_prices("320193", rows, db_path=db_path)
    second_count = database.save_prices("320193", rows, db_path=db_path)
    assert first_count == second_count == 2

    conn = database.get_connection(db_path)
    try:
        total_rows = conn.execute("SELECT COUNT(*) AS n FROM prices").fetchone()["n"]
    finally:
        conn.close()
    assert total_rows == 2  # no duplicates from the second, identical save

    # Re-saving one of the same dates with a different close overwrites it
    # rather than inserting a second row for that (cik, date).
    updated_rows = [_price_row("2023-01-02", 999.0)]
    database.save_prices("320193", updated_rows, db_path=db_path)

    conn = database.get_connection(db_path)
    try:
        total_rows_after_update = conn.execute("SELECT COUNT(*) AS n FROM prices").fetchone()["n"]
        updated = conn.execute(
            "SELECT close FROM prices WHERE cik = ? AND date = ?", ("320193", "2023-01-02")
        ).fetchone()
    finally:
        conn.close()
    assert total_rows_after_update == 2
    assert updated["close"] == 999.0


def test_save_prices_skips_rows_without_a_date(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    rows = [_price_row("2023-01-02", 100.0), {"date": None, "close": 5.0}]

    written = database.save_prices("320193", rows, db_path=db_path)

    assert written == 1


def test_save_prices_empty_input_writes_nothing(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    written = database.save_prices("320193", [], db_path=db_path)
    assert written == 0


# ---------------------------------------------------------------------------
# save_verdict
# ---------------------------------------------------------------------------


def _sample_result():
    return {
        "fair_value_range": {
            "bear": {"lo": 80.0, "hi": 90.0, "growth": "%5 büyüme", "discount_rate": "%12", "note": "n"},
            "base": {"lo": 95.0, "hi": 110.0, "growth": "%10 büyüme", "discount_rate": "%10", "note": "n"},
            "bull": {"lo": 120.0, "hi": 140.0, "growth": "%15 büyüme", "discount_rate": "%9", "note": "n"},
        },
        "fundamental_verdict": "MAKUL",
        "technical_verdict": "NÖTR (yetersiz veri)",
        "profile_fit": {"verdict": "KISMEN", "reason": "PROFIL.md bulunamadı."},
        "cyclical_risk": "low cyclicality.",
        "horizon_note": "1y ufukta dengeli.",
        "key_risks": ["Liquidity"],
        "red_flags_comment": "yok",
        "catalyst": "bilinmiyor",
        "summary": "A short summary.",
        "_provider": "script",
        "_model": "rule-based-v2",
        "_horizon": "1y",
        "_weights": {"fundamental": 0.5, "technical": 0.5},
    }


def test_save_verdict_inserts_row_with_extracted_band_values(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()

    verdict_id = database.save_verdict(
        "AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path
    )

    assert verdict_id is not None
    conn = database.get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM verdicts WHERE id = ?", (verdict_id,)).fetchone()
    finally:
        conn.close()

    assert row["ticker"] == "AAPL"
    assert row["cik"] == "320193"
    assert row["horizon"] == "1y"
    assert row["provider"] == "script"
    assert row["price"] == 100.0
    assert row["fundamental_verdict"] == "MAKUL"
    assert row["technical_verdict"] == "NÖTR (yetersiz veri)"
    assert row["profile_fit"] == "KISMEN"
    assert row["fv_bear_lo"] == 80.0
    assert row["fv_bear_hi"] == 90.0
    assert row["fv_base_lo"] == 95.0
    assert row["fv_base_hi"] == 110.0
    assert row["fv_bull_lo"] == 120.0
    assert row["fv_bull_hi"] == 140.0
    assert row["analyzed_at"]  # a timestamp was stamped


def test_save_verdict_result_json_round_trips(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()

    verdict_id = database.save_verdict(
        "AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path
    )

    conn = database.get_connection(db_path)
    try:
        row = conn.execute("SELECT result_json FROM verdicts WHERE id = ?", (verdict_id,)).fetchone()
    finally:
        conn.close()

    round_tripped = json.loads(row["result_json"])
    assert round_tripped == result


def test_save_verdict_is_append_only_history_not_an_upsert(tmp_path):
    """Every call to save_verdict inserts a new row, even for the same
    ticker/horizon -- unlike financials/ratios/prices, this table is meant
    to preserve every past analysis run for future backtesting."""
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()

    first_id = database.save_verdict("AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path)
    second_id = database.save_verdict("AAPL", "320193", "1y", "script", 105.0, result, db_path=db_path)

    assert first_id != second_id
    conn = database.get_connection(db_path)
    try:
        total_rows = conn.execute("SELECT COUNT(*) AS n FROM verdicts").fetchone()["n"]
    finally:
        conn.close()
    assert total_rows == 2


# ---------------------------------------------------------------------------
# save_verdict -- valuation kwarg / new columns (SPEC.md Sec.14)
# ---------------------------------------------------------------------------


def _sample_valuation():
    return {
        "sector_type": "mature",
        "fcf0": 100.0,
        "fcf0_source": "ttm",
        "dcf": {"enabled": True, "disabled_reason": None, "scenarios": {}, "normalized_variant": None},
        "pb_roe": None,
        "fair_value_range": {
            "bear": {"lo": 80.0, "hi": 90.0, "growth": "%5 büyüme", "discount_rate": "%12", "note": "n"},
            "base": {"lo": 95.0, "hi": 110.0, "growth": "%10 büyüme", "discount_rate": "%10", "note": "n"},
            "bull": {"lo": 120.0, "hi": 140.0, "growth": "%15 büyüme", "discount_rate": "%9", "note": "n"},
        },
        "reverse_dcf": {"implied_growth": 0.19, "realized_cagr_5y": 0.14, "realized_label": "5y"},
        "multiples": {"history": [], "current": {"pe": None, "ps": None, "pfcf": None}},
        "sensitivity": {"lo": 87.0, "hi": 131.0, "high_uncertainty": False},
        "triangulation": {
            "signals": {"dcf": "pahali", "reverse_dcf": "pahali", "multiples": "pahali"},
            "confidence": "YÜKSEK",
            "direction": "pahali",
        },
        "assumptions": {},
        "notes": [],
    }


def test_save_verdict_populates_valuation_derived_columns_when_given(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()
    result["confidence"] = "YÜKSEK"
    valuation = _sample_valuation()

    verdict_id = database.save_verdict(
        "AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path, valuation=valuation
    )

    conn = database.get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM verdicts WHERE id = ?", (verdict_id,)).fetchone()
    finally:
        conn.close()

    assert row["confidence"] == "YÜKSEK"
    assert row["sector_type"] == "mature"
    assert row["implied_growth"] == pytest.approx(0.19)
    assert json.loads(row["fair_value_json"]) == valuation["fair_value_range"]
    assert json.loads(row["valuation_json"]) == valuation


def test_save_verdict_without_valuation_leaves_new_columns_none(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()

    verdict_id = database.save_verdict("AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path)

    conn = database.get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM verdicts WHERE id = ?", (verdict_id,)).fetchone()
    finally:
        conn.close()

    assert row["sector_type"] is None
    assert row["implied_growth"] is None
    assert row["fair_value_json"] is None
    assert row["valuation_json"] is None
    # confidence still comes straight from the result dict even without a
    # valuation dict attached (e.g. a legacy caller that hasn't started
    # passing valuation= yet, but the result already carries "confidence").
    assert row["confidence"] is None


def test_save_verdict_new_columns_exist_after_init_db_migration(tmp_path):
    """A pre-existing database file (created before the Sec.14 schema
    additions) gets the new ``verdicts`` columns backfilled by
    ``_ensure_columns`` the next time ``init_db``/``save_verdict`` runs."""
    db_path = str(tmp_path / "test.sqlite3")
    database.init_db(db_path)

    conn = database.get_connection(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(verdicts)").fetchall()}
    finally:
        conn.close()

    for expected in ("confidence", "sector_type", "implied_growth", "fair_value_json", "valuation_json"):
        assert expected in columns


# ---------------------------------------------------------------------------
# Point-in-time ("as-of") mode: save_verdict's new "as_of" column + the
# load_verdicts/load_latest_stored_price read helpers.
# ---------------------------------------------------------------------------


def test_save_verdict_as_of_roundtrips_through_load_verdicts(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()

    database.save_verdict(
        "AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path, as_of="2022-06-30",
    )

    rows = database.load_verdicts("AAPL", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["as_of"] == "2022-06-30"


def test_save_verdict_live_run_has_as_of_none(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()

    database.save_verdict("AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path)

    rows = database.load_verdicts("AAPL", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["as_of"] is None


def test_save_verdict_as_of_column_is_backfilled_on_a_legacy_database(tmp_path):
    """A pre-existing database created before the as-of column was added
    still gets the ``as_of`` column backfilled by ``_ensure_columns`` the
    next time ``init_db``/``save_verdict`` runs -- mirrors
    test_save_verdict_new_columns_exist_after_init_db_migration above, for
    the newer as-of-specific column."""
    db_path = str(tmp_path / "test.sqlite3")
    database.init_db(db_path)

    conn = database.get_connection(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(verdicts)").fetchall()}
    finally:
        conn.close()

    assert "as_of" in columns


def test_load_verdicts_orders_newest_analyzed_at_first(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()

    database.save_verdict(
        "AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path,
        analyzed_at="2026-01-01T10:00:00",
    )
    database.save_verdict(
        "AAPL", "320193", "1y", "script", 105.0, result, db_path=db_path,
        analyzed_at="2026-03-01T10:00:00",
    )
    database.save_verdict(
        "AAPL", "320193", "1y", "script", 102.0, result, db_path=db_path,
        analyzed_at="2026-02-01T10:00:00",
    )

    rows = database.load_verdicts("AAPL", db_path=db_path)
    assert [r["analyzed_at"] for r in rows] == [
        "2026-03-01T10:00:00", "2026-02-01T10:00:00", "2026-01-01T10:00:00",
    ]


def test_load_verdicts_respects_limit(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()
    for i in range(5):
        database.save_verdict(
            "AAPL", "320193", "1y", "script", 100.0 + i, result, db_path=db_path,
            analyzed_at=f"2026-01-0{i + 1}T10:00:00",
        )

    rows = database.load_verdicts("AAPL", db_path=db_path, limit=2)
    assert len(rows) == 2
    # Still newest-first even when truncated by the limit.
    assert rows[0]["analyzed_at"] == "2026-01-05T10:00:00"


def test_load_verdicts_matches_ticker_case_insensitively(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()
    database.save_verdict("AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path)

    assert len(database.load_verdicts("aapl", db_path=db_path)) == 1
    assert len(database.load_verdicts("AaPl", db_path=db_path)) == 1
    assert len(database.load_verdicts("MSFT", db_path=db_path)) == 0


def test_load_verdicts_row_shape_excludes_blobs_and_includes_scalars(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = _sample_result()
    valuation = _sample_valuation()
    database.save_verdict(
        "AAPL", "320193", "1y", "script", 100.0, result, db_path=db_path, valuation=valuation,
    )

    rows = database.load_verdicts("AAPL", db_path=db_path)
    row = rows[0]
    for blob_key in ("result_json", "fair_value_json", "valuation_json"):
        assert blob_key not in row
    for scalar_key in ("id", "cik", "ticker", "analyzed_at", "as_of", "horizon", "provider",
                       "price", "fv_base_lo", "fv_base_hi", "confidence", "sector_type",
                       "implied_growth"):
        assert scalar_key in row


def test_load_verdicts_returns_empty_list_for_unknown_ticker(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    assert database.load_verdicts("NOSUCHTICKER", db_path=db_path) == []


def _price_row_for_cik(date_str, price, cik="320193"):
    return {
        "date": date_str,
        "open": price - 0.5, "high": price + 0.5, "low": price - 1.0,
        "close": price, "volume": 1_000_000,
    }


def test_load_latest_stored_price_returns_the_latest_bar(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    database.init_db(db_path)
    conn = database.get_connection(db_path)
    try:
        with conn:
            database.upsert_company(conn, "320193", "AAPL", "Apple Inc.")
    finally:
        conn.close()

    database.save_prices(
        "320193",
        [
            _price_row_for_cik("2023-01-02", 100.0),
            _price_row_for_cik("2023-01-05", 103.0),
            _price_row_for_cik("2023-01-04", 101.5),
        ],
        db_path=db_path,
    )

    result = database.load_latest_stored_price("AAPL", db_path=db_path)
    assert result == {"date": "2023-01-05", "close": 103.0}


def test_load_latest_stored_price_matches_ticker_case_insensitively(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    database.init_db(db_path)
    conn = database.get_connection(db_path)
    try:
        with conn:
            database.upsert_company(conn, "320193", "AAPL", "Apple Inc.")
    finally:
        conn.close()
    database.save_prices("320193", [_price_row_for_cik("2023-01-02", 100.0)], db_path=db_path)

    assert database.load_latest_stored_price("aapl", db_path=db_path) == {
        "date": "2023-01-02", "close": 100.0,
    }


def test_load_latest_stored_price_returns_none_for_unknown_ticker(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    database.init_db(db_path)
    assert database.load_latest_stored_price("NOSUCHTICKER", db_path=db_path) is None


def test_load_latest_stored_price_returns_none_when_no_prices_stored(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    database.init_db(db_path)
    conn = database.get_connection(db_path)
    try:
        with conn:
            database.upsert_company(conn, "320193", "AAPL", "Apple Inc.")
    finally:
        conn.close()

    assert database.load_latest_stored_price("AAPL", db_path=db_path) is None


def test_save_verdict_handles_missing_band_values_none_safely(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    result = {
        "fair_value_range": {
            "bear": {"lo": None, "hi": None, "growth": "%0 büyüme", "discount_rate": "%12", "note": "n"},
            "base": {"lo": None, "hi": None, "growth": "%0 büyüme", "discount_rate": "%10", "note": "n"},
            "bull": {"lo": None, "hi": None, "growth": "%0 büyüme", "discount_rate": "%9", "note": "n"},
        },
        "fundamental_verdict": "MAKUL",
        "technical_verdict": "VERİ YOK (fiyat verisi alınamadı)",
        "profile_fit": {"verdict": "KISMEN", "reason": "..."},
        "cyclical_risk": "insufficient history.",
        "horizon_note": "...",
        "key_risks": [],
        "red_flags_comment": "yok",
        "catalyst": "bilinmiyor",
        "summary": "...",
        "_provider": "script",
        "_model": "rule-based-v2",
    }

    verdict_id = database.save_verdict("XYZ", "1", "1y", "script", None, result, db_path=db_path)

    conn = database.get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM verdicts WHERE id = ?", (verdict_id,)).fetchone()
    finally:
        conn.close()
    assert row["price"] is None
    assert row["fv_bear_lo"] is None
    assert row["fv_base_hi"] is None
