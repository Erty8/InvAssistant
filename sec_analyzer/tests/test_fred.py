"""Unit tests for ``fetch.fred`` (FRED DGS10 point-in-time risk-free rate).

No real network access anywhere: ``_parse_asof`` is exercised directly on
hand-built CSV text, and ``get_risk_free_asof`` end-to-end tests monkeypatch
``Config.RAW_DIR`` (to a pytest ``tmp_path``) and ``fred._fetch_csv`` so the
on-disk cache path is real but no HTTP request is ever made.
"""

import os

import pytest

from sec_analyzer.fetch import fred

_CANNED_CSV_LEGACY_HEADER = (
    "DATE,DGS10\n"
    "2022-06-27,3.13\n"
    "2022-06-28,3.20\n"
    "2022-06-29,3.10\n"
    "2022-06-30,2.98\n"
    "2022-07-01,2.88\n"
)

_CANNED_CSV_CURRENT_HEADER = (
    "observation_date,DGS10\n"
    "2022-06-27,3.13\n"
    "2022-06-28,3.20\n"
    "2022-06-29,3.10\n"
    "2022-06-30,2.98\n"
    "2022-07-01,2.88\n"
)


# ---------------------------------------------------------------------------
# _parse_asof
# ---------------------------------------------------------------------------


def test_parse_asof_handles_legacy_date_header():
    result = fred._parse_asof(_CANNED_CSV_LEGACY_HEADER, "DGS10", "2022-06-30")
    assert result == {
        "value_pct": 2.98, "date": "2022-06-30", "series": "DGS10", "source": "FRED DGS10",
    }


def test_parse_asof_handles_current_observation_date_header():
    result = fred._parse_asof(_CANNED_CSV_CURRENT_HEADER, "DGS10", "2022-06-30")
    assert result == {
        "value_pct": 2.98, "date": "2022-06-30", "series": "DGS10", "source": "FRED DGS10",
    }


def test_parse_asof_skips_missing_value_dot_rows():
    """FRED marks a holiday/no-observation day with a literal '.' value; it
    must be skipped, falling back to the last real observation on/before
    the cutoff, not crash trying to float('.')."""
    text = (
        "DATE,DGS10\n"
        "2022-06-30,2.98\n"
        "2022-07-01,.\n"  # holiday marker
        "2022-07-04,3.05\n"
    )
    result = fred._parse_asof(text, "DGS10", "2022-07-01")
    assert result["value_pct"] == pytest.approx(2.98)
    assert result["date"] == "2022-06-30"


def test_parse_asof_on_or_before_selection_including_weekend_asof():
    """Hand-verified: rows 2022-06-29->3.10, 2022-06-30->2.98,
    2022-07-01->2.88 (a Friday). as_of='2022-07-02' (a Saturday, no row that
    day) must walk back to the last observation on/before it: 2022-07-01,
    value 2.88."""
    result = fred._parse_asof(_CANNED_CSV_LEGACY_HEADER, "DGS10", "2022-07-02")
    assert result == {
        "value_pct": 2.88, "date": "2022-07-01", "series": "DGS10", "source": "FRED DGS10",
    }


def test_parse_asof_exact_date_match_is_inclusive():
    result = fred._parse_asof(_CANNED_CSV_LEGACY_HEADER, "DGS10", "2022-06-29")
    assert result["value_pct"] == pytest.approx(3.10)
    assert result["date"] == "2022-06-29"


def test_parse_asof_before_first_row_returns_none():
    result = fred._parse_asof(_CANNED_CSV_LEGACY_HEADER, "DGS10", "2020-01-01")
    assert result is None


def test_parse_asof_empty_or_header_only_text_returns_none():
    assert fred._parse_asof("", "DGS10", "2022-07-01") is None
    assert fred._parse_asof("DATE,DGS10\n", "DGS10", "2022-07-01") is None


def test_parse_asof_malformed_csv_does_not_raise():
    # Not valid CSV in a way that would upset csv.reader badly -- just make
    # sure garbage input degrades to None rather than raising.
    assert fred._parse_asof("not,a,real\ncsv\x00file", "DGS10", "2022-07-01") is None


# ---------------------------------------------------------------------------
# get_risk_free_asof -- end-to-end with a monkeypatched cache dir + fetch.
# ---------------------------------------------------------------------------


def test_get_risk_free_asof_fetches_writes_cache_and_returns_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    result = fred.get_risk_free_asof("2022-06-30")

    assert result == {
        "value_pct": 2.98, "date": "2022-06-30", "series": "DGS10", "source": "FRED DGS10",
    }
    cache_path = os.path.join(str(tmp_path), "fred_DGS10.csv")
    assert os.path.isfile(cache_path)
    with open(cache_path, encoding="utf-8") as fh:
        assert fh.read() == _CANNED_CSV_LEGACY_HEADER


def test_get_risk_free_asof_accepts_date_object(tmp_path, monkeypatch):
    from datetime import date

    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    result = fred.get_risk_free_asof(date(2022, 6, 30))
    assert result["value_pct"] == pytest.approx(2.98)


def test_get_risk_free_asof_uses_fresh_cache_without_refetching(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    cache_path = os.path.join(str(tmp_path), "fred_DGS10.csv")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(_CANNED_CSV_LEGACY_HEADER)

    calls = []

    def _boom(series):
        calls.append(series)
        raise AssertionError("should not hit the network when the cache is fresh")

    monkeypatch.setattr(fred, "_fetch_csv", _boom)

    result = fred.get_risk_free_asof("2022-06-30")
    assert result["value_pct"] == pytest.approx(2.98)
    assert calls == []


def test_get_risk_free_asof_network_failure_with_no_cache_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)

    # No cache file exists in the fresh tmp_path -- must return None, never raise.
    result = fred.get_risk_free_asof("2022-06-30")
    assert result is None


def test_get_risk_free_asof_network_failure_falls_back_to_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    cache_path = os.path.join(str(tmp_path), "fred_DGS10.csv")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(_CANNED_CSV_LEGACY_HEADER)
    # Make the cache look stale (>24h) so a re-fetch is attempted.
    old_time = os.path.getmtime(cache_path) - (25 * 60 * 60)
    os.utime(cache_path, (old_time, old_time))

    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)

    result = fred.get_risk_free_asof("2022-06-30")
    # Falls back to the stale cache rather than failing outright.
    assert result["value_pct"] == pytest.approx(2.98)


def test_get_risk_free_asof_no_cache_flag_bypasses_and_still_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    cache_path = os.path.join(str(tmp_path), "fred_DGS10.csv")
    os.makedirs(str(tmp_path), exist_ok=True)
    # Pre-existing (fresh) cache with a DIFFERENT value than the fetch below,
    # to prove no_cache=True bypasses reading it.
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write("DATE,DGS10\n2022-06-30,9.99\n")

    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    result = fred.get_risk_free_asof("2022-06-30", no_cache=True)
    assert result["value_pct"] == pytest.approx(2.98)


def test_get_risk_free_asof_never_raises_on_unexpected_internal_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fred, "_parse_asof", _boom)
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    assert fred.get_risk_free_asof("2022-06-30", no_cache=True) is None
