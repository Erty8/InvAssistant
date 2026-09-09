"""Tests for sec_analyzer.store.thesis_anchors -- the persisted day-1
thesis-anchor direction backing METODOLOJI.md Sec.7's quarterly-invalidation
rule.

Mirrors test_assumptions_cache.py's conventions: an isolated throwaway
SQLite file per test (via ``Config.DB_PATH`` monkeypatched), plain
function-in/value-out assertions, no mocks of the code under test.
"""

import pytest

from sec_analyzer.config import Config
from sec_analyzer.store import thesis_anchors as TA
from sec_analyzer.store.database import get_connection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A throwaway SQLite path, also installed as Config.DB_PATH so code
    paths that read the default (get_anchor/set_anchor with no db_path
    override) hit the same isolated file."""
    p = str(tmp_path / "test.sqlite3")
    monkeypatch.setattr(Config, "DB_PATH", p)
    return p


# ---------------------------------------------------------------------------
# get_anchor -- no row yet
# ---------------------------------------------------------------------------

def test_get_anchor_returns_none_when_no_row_exists(db_path):
    assert TA.get_anchor("0000320193") is None


# ---------------------------------------------------------------------------
# set_anchor -- establish, then get_anchor reads it back
# ---------------------------------------------------------------------------

def test_set_anchor_establishes_and_get_anchor_returns_it(db_path):
    row = TA.set_anchor(
        cik="0000320193",
        metric_key="net_margin",
        direction="improving",
        established_fy=2023,
        ticker="AAPL",
    )
    assert row is not None
    assert row["cik"] == "0000320193"
    assert row["ticker"] == "AAPL"
    assert row["metric_key"] == "net_margin"
    assert row["direction"] == "improving"
    assert row["established_fy"] == 2023
    # established_at is stamped audit metadata, always present after a write.
    assert row["established_at"] is not None

    fetched = TA.get_anchor("0000320193")
    assert fetched == row


def test_set_anchor_established_fy_none_is_allowed(db_path):
    row = TA.set_anchor(
        cik="7", metric_key="roe", direction="deteriorating", established_fy=None
    )
    assert row is not None
    assert row["established_fy"] is None
    assert TA.get_anchor("7")["established_fy"] is None


# ---------------------------------------------------------------------------
# set_anchor -- re-calling with a new metric_key overwrites (upsert) in place
# ---------------------------------------------------------------------------

def test_set_anchor_upserts_overwrite_in_place_not_append(db_path):
    TA.set_anchor(cik="1", metric_key="net_margin", direction="improving", established_fy=2022)
    second = TA.set_anchor(
        cik="1", metric_key="gross_margin", direction="deteriorating", established_fy=2023, ticker="X"
    )
    assert second["metric_key"] == "gross_margin"
    assert second["direction"] == "deteriorating"
    assert second["established_fy"] == 2023
    assert second["ticker"] == "X"

    # Upsert on cik, not append: exactly one row survives for this cik.
    conn = get_connection(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM thesis_anchors WHERE cik = ?", ("1",)
        ).fetchone()["c"]
    finally:
        conn.close()
    assert count == 1

    fetched = TA.get_anchor("1")
    assert fetched["metric_key"] == "gross_margin"
    assert fetched["direction"] == "deteriorating"


def test_set_anchor_upserts_same_metric_key_direction_change(db_path):
    """Re-calling set_anchor for the SAME metric_key (e.g. a later run whose
    direction was deliberately re-seeded) still overwrites in place -- the
    upsert doesn't care whether metric_key changed, only planning.py's
    caller-side policy (_establish_or_load_anchor) decides WHEN to call
    set_anchor again."""
    TA.set_anchor(cik="2", metric_key="net_margin", direction="improving", established_fy=2022)
    TA.set_anchor(cik="2", metric_key="net_margin", direction="deteriorating", established_fy=2024)

    fetched = TA.get_anchor("2")
    assert fetched["metric_key"] == "net_margin"
    assert fetched["direction"] == "deteriorating"
    assert fetched["established_fy"] == 2024


# ---------------------------------------------------------------------------
# DB round-trip via a separate direct connection (not just get_anchor)
# ---------------------------------------------------------------------------

def test_set_anchor_persists_to_the_underlying_sqlite_file(db_path):
    TA.set_anchor(
        cik="42", metric_key="roe", direction="improving", established_fy=2021, db_path=db_path
    )
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM thesis_anchors WHERE cik = ?", ("42",)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["metric_key"] == "roe"
    assert row["direction"] == "improving"
    assert row["established_fy"] == 2021


def test_cik_is_stringified_for_storage_and_lookup(db_path):
    TA.set_anchor(cik=320193, metric_key="net_margin", direction="improving", established_fy=2023)
    # Lookup with either an int or a string cik resolves to the same row.
    by_int = TA.get_anchor(320193)
    by_str = TA.get_anchor("320193")
    assert by_int is not None and by_str is not None
    assert by_int["cik"] == "320193"
    assert by_int == by_str


# ---------------------------------------------------------------------------
# Explicit db_path override on both read and write paths
# ---------------------------------------------------------------------------

def test_explicit_db_path_overrides_config_db_path(tmp_path, monkeypatch):
    # Config.DB_PATH points somewhere else entirely; the explicit db_path
    # argument must still be honored on both set_anchor and get_anchor.
    monkeypatch.setattr(Config, "DB_PATH", str(tmp_path / "unused.sqlite3"))
    other_path = str(tmp_path / "explicit.sqlite3")

    TA.set_anchor(
        cik="99", metric_key="fcf_margin", direction="improving", established_fy=2020, db_path=other_path
    )
    assert TA.get_anchor("99", db_path=other_path) is not None
    # The default (Config.DB_PATH) file never got this row.
    assert TA.get_anchor("99") is None
