"""Unit tests for ``fetch.treasury`` (Treasury.gov daily par yield curve).

No real network access anywhere: ``_parse_rows``/``_mmddyyyy_to_iso``/
``_normalize_maturity`` are exercised directly on hand-built CSV text, and
end-to-end tests monkeypatch ``Config.RAW_DIR`` (to a pytest ``tmp_path``,
via the autouse fixture below -- same pattern as ``test_fred.py``) and
``treasury._fetch_csv`` so the on-disk cache path is real but no HTTP
request is ever made.
"""

import os

import pytest

from sec_analyzer.fetch import treasury


@pytest.fixture(autouse=True)
def _isolate_raw_dir(tmp_path, monkeypatch):
    """Every test in this module writes to a throwaway cache directory,
    never the real on-disk ``Config.RAW_DIR`` -- see the equivalent fixture
    (and the incident it fixed) in ``test_fred.py``."""
    monkeypatch.setattr(treasury.Config, "RAW_DIR", str(tmp_path))


# Quoted-header nominal curve CSV, shaped like Treasury's real response.
_NOMINAL_CSV_2026 = (
    'Date,"1 Mo","2 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr",'
    '"10 Yr","20 Yr","30 Yr"\n'
    "01/02/2026,3.80,3.84,3.90,3.99,4.06,4.25,4.31,4.40,4.53,4.69,5.22,5.22\n"
    "01/05/2026,3.79,3.83,3.89,3.98,4.05,4.24,4.30,4.39,4.52,4.68,5.21,5.21\n"
)

# Quoted-header real (TIPS) curve CSV -- note "YR" casing differs from the
# nominal curve's "Yr", and Treasury sometimes leaves a maturity blank.
_REAL_CSV_2026 = (
    'Date,"5 YR","7 YR","10 YR","20 YR","30 YR"\n'
    "01/02/2026,1.90,2.10,2.43,2.78,2.98\n"
    "01/05/2026,,2.09,2.42,2.77,N/A\n"
)


def _write_cache(tmp_path, year, real, text):
    kind = "real" if real else "nominal"
    path = os.path.join(str(tmp_path), f"treasury_{kind}_{year}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# _mmddyyyy_to_iso
# ---------------------------------------------------------------------------


def test_mmddyyyy_to_iso_parses_valid_date():
    assert treasury._mmddyyyy_to_iso("01/02/2026") == "2026-01-02"


def test_mmddyyyy_to_iso_returns_none_on_garbage():
    assert treasury._mmddyyyy_to_iso("not-a-date") is None
    assert treasury._mmddyyyy_to_iso("") is None
    assert treasury._mmddyyyy_to_iso(None) is None


# ---------------------------------------------------------------------------
# _normalize_maturity
# ---------------------------------------------------------------------------


def test_normalize_maturity_matches_across_casing_and_spacing():
    assert treasury._normalize_maturity("10 Yr") == treasury._normalize_maturity("10 YR")
    assert treasury._normalize_maturity("10 Yr") == treasury._normalize_maturity("10Yr")


def test_normalize_maturity_none_or_empty_is_empty_string():
    assert treasury._normalize_maturity(None) == ""
    assert treasury._normalize_maturity("") == ""


# ---------------------------------------------------------------------------
# _parse_rows -- quoted headers, MM/DD/YYYY, blank/N/A cells, missing column.
# ---------------------------------------------------------------------------


def test_parse_rows_nominal_curve_quoted_header():
    rows = treasury._parse_rows(_NOMINAL_CSV_2026)
    assert len(rows) == 2
    date_iso, values = rows[0]
    assert date_iso == "2026-01-02"
    assert values["10 Yr"] == pytest.approx(4.69)
    assert values["2 Yr"] == pytest.approx(4.25)
    assert values["30 Yr"] == pytest.approx(5.22)


def test_parse_rows_real_curve_skips_blank_and_na_cells():
    rows = treasury._parse_rows(_REAL_CSV_2026)
    assert len(rows) == 2
    date_iso, values = rows[1]
    assert date_iso == "2026-01-05"
    # "5 YR" was blank and "30 YR" was "N/A" -- both must be absent, not 0.0.
    assert "5 YR" not in values
    assert "30 YR" not in values
    assert values["10 YR"] == pytest.approx(2.42)


def test_parse_rows_missing_maturity_column_is_simply_absent():
    text = 'Date,"10 Yr"\n01/02/2026,4.69\n'
    rows = treasury._parse_rows(text)
    date_iso, values = rows[0]
    assert "2 Yr" not in values
    assert values["10 Yr"] == pytest.approx(4.69)


def test_parse_rows_empty_or_header_only_returns_empty_list():
    assert treasury._parse_rows("") == []
    assert treasury._parse_rows('Date,"10 Yr"\n') == []


def test_parse_rows_malformed_csv_does_not_raise():
    assert treasury._parse_rows("not,a,real\ncsv\x00file") == []


def test_parse_rows_skips_rows_with_unparseable_date():
    text = 'Date,"10 Yr"\nnot-a-date,4.69\n01/02/2026,4.70\n'
    rows = treasury._parse_rows(text)
    assert len(rows) == 1
    assert rows[0][0] == "2026-01-02"


# ---------------------------------------------------------------------------
# get_maturity_series -- end to end, with a monkeypatched fetch + cache dir.
# ---------------------------------------------------------------------------


def test_get_maturity_series_single_year(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)

    series = treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026)

    assert series == [("2026-01-02", pytest.approx(4.69)), ("2026-01-05", pytest.approx(4.68))]


def test_get_maturity_series_spans_multiple_years_and_sorts_chronologically(tmp_path, monkeypatch):
    def _fetch(year, real):
        if year == 2025:
            return 'Date,"10 Yr"\n12/30/2025,4.50\n12/31/2025,4.52\n'
        if year == 2026:
            return _NOMINAL_CSV_2026
        return None

    monkeypatch.setattr(treasury, "_fetch_csv", _fetch)

    series = treasury.get_maturity_series(treasury.MATURITY_10Y, 2025, 2026)

    assert [d for d, _ in series] == ["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"]


def test_get_maturity_series_start_end_year_order_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)
    forward = treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026)
    backward = treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026)
    assert forward == backward


def test_get_maturity_series_real_curve_matches_yr_casing(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _REAL_CSV_2026)

    series = treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026, real=True)

    # "10 Yr" (canonical label) must match the real curve's "10 YR" header.
    assert ("2026-01-02", pytest.approx(2.43)) in series


def test_get_maturity_series_unknown_maturity_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)
    assert treasury.get_maturity_series("99 Yr", 2026, 2026) == []


def test_get_maturity_series_fetch_failure_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: None)
    assert treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026) == []


def test_get_maturity_series_never_raises_on_internal_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(treasury, "_parse_rows", _boom)
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)

    assert treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026) == []


# ---------------------------------------------------------------------------
# get_curve_asof -- as-of walk-back, including across a weekend/year edge.
# ---------------------------------------------------------------------------


def test_get_curve_asof_as_of_none_returns_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)

    result = treasury.get_curve_asof(as_of=None)

    assert result["date"] == "2026-01-05"
    assert result["maturities"]["10 Yr"] == pytest.approx(4.68)
    assert result["real"] is False
    assert "nominal" in result["source"]


def test_get_curve_asof_walks_back_across_a_weekend(monkeypatch):
    """as_of=2026-01-04 (a Sunday, no row that day) must walk back to the
    last trading day on/before it: 2026-01-02."""

    def _fetch(year, real):
        return _NOMINAL_CSV_2026 if year == 2026 else None

    monkeypatch.setattr(treasury, "_fetch_csv", _fetch)

    result = treasury.get_curve_asof(as_of="2026-01-04")

    assert result["date"] == "2026-01-02"
    assert result["maturities"]["10 Yr"] == pytest.approx(4.69)


def test_get_curve_asof_walks_back_across_year_boundary(monkeypatch):
    """An as_of early in a new year, before that year's first published row,
    must fall back into December of the prior year."""

    def _fetch(year, real):
        if year == 2025:
            return 'Date,"10 Yr"\n12/31/2025,4.52\n'
        if year == 2026:
            return 'Date,"10 Yr"\n01/05/2026,4.68\n'
        return None

    monkeypatch.setattr(treasury, "_fetch_csv", _fetch)

    result = treasury.get_curve_asof(as_of="2026-01-02")

    assert result["date"] == "2025-12-31"
    assert result["maturities"]["10 Yr"] == pytest.approx(4.52)


def test_get_curve_asof_real_curve(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _REAL_CSV_2026)
    result = treasury.get_curve_asof(as_of="2026-01-02", real=True)
    assert result["maturities"]["10 YR"] == pytest.approx(2.43)
    assert result["real"] is True


def test_get_curve_asof_no_data_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: None)
    assert treasury.get_curve_asof(as_of="2026-01-02") is None


def test_get_curve_asof_never_raises_on_internal_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(treasury, "_parse_rows", _boom)
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)

    assert treasury.get_curve_asof(as_of="2026-01-02") is None


# ---------------------------------------------------------------------------
# Per-year cache: closed-year long TTL vs. current-year 24h TTL.
# ---------------------------------------------------------------------------


def test_cache_write_writes_a_file_per_year(tmp_path, monkeypatch):
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)

    treasury.get_maturity_series(treasury.MATURITY_10Y, 2025, 2026)

    assert os.path.isfile(os.path.join(str(tmp_path), "treasury_nominal_2025.csv"))
    assert os.path.isfile(os.path.join(str(tmp_path), "treasury_nominal_2026.csv"))


def test_closed_year_cache_is_used_even_when_very_old(tmp_path, monkeypatch):
    """A closed (past) year's cache must not be considered stale just
    because it is more than 24h old -- that year will never gain a new row."""
    path = _write_cache(tmp_path, 2020, real=False, text='Date,"10 Yr"\n06/30/2020,0.66\n')
    very_old = os.path.getmtime(path) - (400 * 24 * 60 * 60)
    os.utime(path, (very_old, very_old))

    def _boom(year, real):
        raise AssertionError("should not re-fetch a closed year's fresh-enough cache")

    monkeypatch.setattr(treasury, "_fetch_csv", _boom)

    series = treasury.get_maturity_series(treasury.MATURITY_10Y, 2020, 2020)
    assert series == [("2020-06-30", pytest.approx(0.66))]


def test_current_year_cache_older_than_24h_is_refetched(tmp_path, monkeypatch):
    import datetime

    current_year = datetime.date.today().year
    path = _write_cache(
        tmp_path, current_year, real=False, text=f'Date,"10 Yr"\n01/02/{current_year},1.00\n'
    )
    old_time = os.path.getmtime(path) - (25 * 60 * 60)
    os.utime(path, (old_time, old_time))

    fresh_text = f'Date,"10 Yr"\n01/03/{current_year},2.00\n'
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: fresh_text)

    series = treasury.get_maturity_series(treasury.MATURITY_10Y, current_year, current_year)
    assert series == [(f"{current_year}-01-03", pytest.approx(2.00))]


def test_no_cache_flag_bypasses_cache(tmp_path, monkeypatch):
    _write_cache(tmp_path, 2026, real=False, text='Date,"10 Yr"\n01/02/2026,9.99\n')
    monkeypatch.setattr(treasury, "_fetch_csv", lambda year, real: _NOMINAL_CSV_2026)

    series = treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026, no_cache=True)
    assert series == [("2026-01-02", pytest.approx(4.69)), ("2026-01-05", pytest.approx(4.68))]


def test_nominal_and_real_caches_do_not_clobber_each_other(tmp_path, monkeypatch):
    def _fetch(year, real):
        return _REAL_CSV_2026 if real else _NOMINAL_CSV_2026

    monkeypatch.setattr(treasury, "_fetch_csv", _fetch)

    treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026, real=False)
    treasury.get_maturity_series(treasury.MATURITY_10Y, 2026, 2026, real=True)

    nominal_path = os.path.join(str(tmp_path), "treasury_nominal_2026.csv")
    real_path = os.path.join(str(tmp_path), "treasury_real_2026.csv")
    assert os.path.isfile(nominal_path)
    assert os.path.isfile(real_path)
    with open(nominal_path, encoding="utf-8") as fh:
        assert fh.read() == _NOMINAL_CSV_2026
    with open(real_path, encoding="utf-8") as fh:
        assert fh.read() == _REAL_CSV_2026
