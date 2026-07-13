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
