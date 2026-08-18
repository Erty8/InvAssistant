"""Unit tests for ``fetch.fred`` (FRED macro series, point-in-time and live).

No real network access anywhere: ``_parse_asof``/``_compute_percentile`` are
exercised directly on hand-built CSV text, and end-to-end tests monkeypatch
``Config.RAW_DIR`` (to a pytest ``tmp_path``) and ``fred._fetch_csv`` so the
on-disk cache path is real but no HTTP request is ever made.

``Config.RAW_DIR`` is redirected to ``tmp_path`` by an autouse fixture below
for every test in this module -- not just the ones that obviously write a
cache file. A prior version of this suite had one test
(``test_get_risk_free_asof_never_raises_on_unexpected_internal_error``) that
forgot the per-test override; since it exercises a real cache-writing path
(``no_cache=True`` forces a fetch, and the fetch result gets written to
``Config.RAW_DIR`` before the monkeypatched ``_parse_asof`` blows up), it
silently wrote a hand-built 2022 fixture CSV into the real production cache
directory. That file would then have been served back by
``get_risk_free_asof(None)`` as if it were *today's* risk-free rate on the
next live run. The autouse fixture closes that hole for every test, present
and future, rather than relying on each test remembering to opt in.
"""

import os

import pytest

from sec_analyzer.fetch import fred

#: The real, un-monkeypatched cache directory -- captured at import time,
#: before the autouse fixture below ever runs -- so the regression guard at
#: the bottom of this module can prove nothing gets written there.
_REAL_RAW_DIR = fred.Config.RAW_DIR


@pytest.fixture(autouse=True)
def _isolate_raw_dir(tmp_path, monkeypatch):
    """Force every test in this module to use a throwaway cache directory.

    Autouse so a new test can't reintroduce the leak above by simply
    forgetting to monkeypatch ``Config.RAW_DIR`` itself. Tests below still
    set it explicitly too in most cases (harmless -- same ``tmp_path``
    instance), which keeps them readable as self-contained examples.
    """
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))


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

    # get_risk_free_asof's return is now a superset (percentile fields added
    # on top) of the original 4-key shape -- check those 4 keys individually
    # rather than exact dict equality, per fetch/fred.py's docstring.
    assert result["value_pct"] == pytest.approx(2.98)
    assert result["date"] == "2022-06-30"
    assert result["series"] == "DGS10"
    assert result["source"] == "FRED DGS10"
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
    # FRED failing now falls back to Treasury (see test_fred_failure_falls_back_to_treasury
    # below) -- block that path too here so this test exercises "no provider at all".
    monkeypatch.setattr(fred.treasury, "_fetch_csv", lambda year, real: None)

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


# ---------------------------------------------------------------------------
# get_series_asof / get_risk_free_asof with as_of=None -- live-mode "latest".
# ---------------------------------------------------------------------------


def test_get_series_asof_as_of_none_returns_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    result = fred.get_series_asof("DGS10", as_of=None)

    assert result["date"] == "2022-07-01"
    assert result["value_pct"] == pytest.approx(2.88)
    assert result["series"] == "DGS10"
    assert result["source"] == "FRED DGS10"


def test_get_risk_free_asof_none_returns_latest(tmp_path, monkeypatch):
    """as_of=None must not fail -- this is the live-mode path that used to
    be unsupported (get_risk_free_asof previously required a real as_of)."""
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    result = fred.get_risk_free_asof(None)

    assert result["date"] == "2022-07-01"
    assert result["value_pct"] == pytest.approx(2.88)
    assert result["series"] == "DGS10"


# ---------------------------------------------------------------------------
# _compute_percentile -- self-anchored percentile within a trailing window.
# ---------------------------------------------------------------------------


def test_compute_percentile_hand_computed_with_midrank_tie():
    """30 daily observations 2020-01-02..2020-01-31 valued 1..30, except the
    15th day is also set to 30.0 (a tie with the reference/current value).
    less_count = values 1..14 and 16..29 (28 values); equal_count = 2
    (day 15 and day 30). pct = (28 + 0.5*2) / 30 * 100 = 96.7.
    A row dated 2020-02-01 (after the reference date) must not enter the
    window at all."""
    lines = ["DATE,DGS10"]
    for i in range(1, 31):
        value = 30.0 if i == 15 else float(i)
        lines.append(f"2020-01-{i + 1:02d},{value}")
    lines.append("2020-02-01,999")  # after reference date -- must be excluded
    text = "\n".join(lines) + "\n"

    result = fred._compute_percentile(text, "DGS10", 30.0, "2020-01-31", 10)

    assert result["window_n"] == 30
    assert result["window_start"] == "2010-01-31"
    assert result["min_pct"] == pytest.approx(1.0)
    assert result["max_pct"] == pytest.approx(30.0)
    assert result["percentile"] == pytest.approx(round((28 + 0.5 * 2) / 30 * 100, 1))
    assert result["percentile"] == pytest.approx(96.7)


def test_compute_percentile_none_below_min_sample_size():
    lines = ["DATE,DGS10"]
    for i in range(1, 6):  # only 5 observations -- below the 30 minimum
        lines.append(f"2020-01-{i:02d},{float(i)}")
    text = "\n".join(lines) + "\n"

    result = fred._compute_percentile(text, "DGS10", 5.0, "2020-01-05", 10)

    assert result["window_n"] == 5
    assert result["percentile"] is None
    assert result["min_pct"] == pytest.approx(1.0)
    assert result["max_pct"] == pytest.approx(5.0)


def test_compute_percentile_window_bounded_by_percentile_years():
    """The window for reference date 2020-01-31 with percentile_years=10
    starts exactly at 2010-01-31 (inclusive). A row one day earlier
    (2010-01-30) must fall outside the window."""
    text = "DATE,DGS10\n2010-01-30,1.0\n2010-01-31,2.0\n2020-01-31,3.0\n"

    result = fred._compute_percentile(text, "DGS10", 3.0, "2020-01-31", 10)

    assert result["window_n"] == 2
    assert result["min_pct"] == pytest.approx(2.0)


def test_compute_percentile_empty_series_returns_none_fields():
    result = fred._compute_percentile("DATE,DGS10\n", "DGS10", 3.0, "2020-01-31", 10)
    assert result["window_n"] == 0
    assert result["percentile"] is None
    assert result["min_pct"] is None
    assert result["max_pct"] is None


# ---------------------------------------------------------------------------
# get_macro_panel -- one series failing must not affect the others.
# ---------------------------------------------------------------------------


def test_get_macro_panel_keys_every_requested_series_partial_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    # Block the Treasury fallback too, so a FRED failure for DGS2 has no
    # provider left and genuinely comes back None (see the dedicated
    # FRED-down/Treasury-up fallback tests below for the mixed case).
    monkeypatch.setattr(fred.treasury, "_fetch_csv", lambda year, real: None)

    def _fetch(series):
        if series == fred.SERIES_DGS2:
            return None  # simulate this one series failing
        return _CANNED_CSV_LEGACY_HEADER

    monkeypatch.setattr(fred, "_fetch_csv", _fetch)

    panel = fred.get_macro_panel(
        as_of="2022-06-30",
        series=(fred.SERIES_DGS10, fred.SERIES_DGS2, fred.SERIES_T10YIE),
    )

    assert set(panel.keys()) == {fred.SERIES_DGS10, fred.SERIES_DGS2, fred.SERIES_T10YIE}
    assert panel[fred.SERIES_DGS2] is None
    assert panel[fred.SERIES_DGS10]["value_pct"] == pytest.approx(2.98)
    assert panel[fred.SERIES_T10YIE]["value_pct"] == pytest.approx(2.98)


def test_get_macro_panel_never_raises_returns_dict_of_none_on_total_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)
    monkeypatch.setattr(fred.treasury, "_fetch_csv", lambda year, real: None)

    panel = fred.get_macro_panel(as_of="2022-06-30", series=(fred.SERIES_DGS10, fred.SERIES_DGS2))

    assert panel == {fred.SERIES_DGS10: None, fred.SERIES_DGS2: None}


def test_get_macro_panel_writes_a_cache_file_per_series_no_clobbering(tmp_path, monkeypatch):
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    fred.get_macro_panel(as_of="2022-06-30", series=(fred.SERIES_DGS10, fred.SERIES_DGS2))

    dgs10_path = os.path.join(str(tmp_path), "fred_DGS10.csv")
    dgs2_path = os.path.join(str(tmp_path), "fred_DGS2.csv")
    assert os.path.isfile(dgs10_path)
    assert os.path.isfile(dgs2_path)
    with open(dgs10_path, encoding="utf-8") as fh:
        assert fh.read() == _CANNED_CSV_LEGACY_HEADER
    with open(dgs2_path, encoding="utf-8") as fh:
        assert fh.read() == _CANNED_CSV_LEGACY_HEADER


# ---------------------------------------------------------------------------
# FRED -> Treasury fallback (fred.stlouisfed.org is unreachable on some
# networks entirely -- see fred.py's module docstring).
# ---------------------------------------------------------------------------

#: A one-year nominal curve CSV with 30 trading days -- enough to clear
#: _MIN_PERCENTILE_WINDOW_N (30) on its own, for the percentile-through-
#: fallback test below. Values are simply 1..30 by day.
_TREASURY_NOMINAL_2026_30_ROWS = "Date,\"2 Yr\",\"10 Yr\",\"30 Yr\"\n" + "".join(
    f"01/{i + 1:02d}/2026,{float(i)},{float(i)},{float(i)}\n" for i in range(1, 31)
)
#: Matching 30-row real (TIPS) curve, offset by a constant 1.5pp so the
#: derived breakeven (nominal - real) is a steady 1.5 across the window --
#: exercises the T10YIE-via-Treasury path without needing an interesting
#: number, just a non-None one with the right provider/source.
_TREASURY_REAL_2026_30_ROWS = "Date,\"10 YR\"\n" + "".join(
    f"01/{i + 1:02d}/2026,{float(i) - 1.5}\n" for i in range(1, 31)
)


def test_fred_failure_falls_back_to_treasury_for_dgs10(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)  # FRED unreachable
    monkeypatch.setattr(
        fred.treasury, "_fetch_csv", lambda year, real: _TREASURY_NOMINAL_2026_30_ROWS
    )

    result = fred.get_risk_free_asof("2026-01-31")

    assert result is not None
    assert result["provider"] == "Treasury"
    assert result["source"] == "Treasury CMT 10 Yr"
    assert result["value_pct"] == pytest.approx(30.0)
    assert result["date"] == "2026-01-31"


def test_fred_failure_falls_back_to_treasury_for_dgs2_and_dgs30(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)
    monkeypatch.setattr(
        fred.treasury, "_fetch_csv", lambda year, real: _TREASURY_NOMINAL_2026_30_ROWS
    )

    dgs2 = fred.get_series_asof(fred.SERIES_DGS2, as_of="2026-01-31")
    dgs30 = fred.get_series_asof(fred.SERIES_DGS30, as_of="2026-01-31")

    assert dgs2["provider"] == "Treasury"
    assert dgs2["source"] == "Treasury CMT 2 Yr"
    assert dgs30["provider"] == "Treasury"
    assert dgs30["source"] == "Treasury CMT 30 Yr"


def test_fred_failure_falls_back_to_treasury_for_t10yie_breakeven(tmp_path, monkeypatch):
    nominal_csv = "Date,\"10 Yr\"\n01/02/2026,4.69\n"
    real_csv = "Date,\"10 YR\"\n01/02/2026,2.43\n"

    def _fetch(series):
        return None  # FRED unreachable

    def _treasury_fetch(year, real):
        return real_csv if real else nominal_csv

    monkeypatch.setattr(fred, "_fetch_csv", _fetch)
    monkeypatch.setattr(fred.treasury, "_fetch_csv", _treasury_fetch)

    result = fred.get_series_asof(fred.SERIES_T10YIE, as_of="2026-01-02")

    assert result is not None
    assert result["provider"] == "Treasury"
    assert "breakeven" in result["source"].lower()
    assert result["value_pct"] == pytest.approx(4.69 - 2.43)


def test_fred_success_never_calls_treasury(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)

    calls = []

    def _boom(year, real):
        calls.append((year, real))
        raise AssertionError("Treasury must not be called when FRED already answered")

    monkeypatch.setattr(fred.treasury, "_fetch_csv", _boom)

    result = fred.get_risk_free_asof("2022-06-30")

    assert result["provider"] == "FRED"
    assert calls == []


def test_credit_spreads_have_no_treasury_substitute_and_stay_none(tmp_path, monkeypatch):
    """BAA10Y and BAMLH0A0HYM2 have no Treasury equivalent (Treasury does not
    publish Moody's Baa or ICE BofA HY data). When FRED is down, both must
    come back None -- never a fabricated proxy that would silently corrupt
    signals.macro.build_macro_context's credit_regime."""
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)  # FRED unreachable

    calls = []

    def _record(year, real):
        calls.append((year, real))
        return None

    monkeypatch.setattr(fred.treasury, "_fetch_csv", _record)

    baa = fred.get_series_asof(fred.SERIES_BAA10Y, as_of="2026-01-31")
    hy = fred.get_series_asof(fred.SERIES_HY_OAS, as_of="2026-01-31")

    assert baa is None
    assert hy is None
    # Neither series should even attempt a Treasury fetch -- there is no
    # mapping for them, unlike the rate series above.
    assert calls == []


def test_get_macro_panel_mixed_providers_rates_from_treasury_spreads_none(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)  # FRED unreachable
    monkeypatch.setattr(
        fred.treasury,
        "_fetch_csv",
        lambda year, real: _TREASURY_REAL_2026_30_ROWS if real else _TREASURY_NOMINAL_2026_30_ROWS,
    )

    panel = fred.get_macro_panel(as_of="2026-01-31")

    # All four rate/breakeven series have a faithful Treasury substitute.
    assert panel[fred.SERIES_DGS10]["provider"] == "Treasury"
    assert panel[fred.SERIES_DGS2]["provider"] == "Treasury"
    assert panel[fred.SERIES_DGS30]["provider"] == "Treasury"
    assert panel[fred.SERIES_T10YIE]["provider"] == "Treasury"
    assert panel[fred.SERIES_T10YIE]["value_pct"] == pytest.approx(1.5)
    # The two credit spreads have no substitute -- must stay None, never a
    # fabricated proxy, and the panel as a whole must not raise.
    assert panel[fred.SERIES_BAA10Y] is None
    assert panel[fred.SERIES_HY_OAS] is None


def test_percentile_computed_over_multiyear_treasury_window(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: None)  # FRED unreachable

    calls = []

    def _treasury_fetch(year, real):
        calls.append(year)
        return _TREASURY_NOMINAL_2026_30_ROWS if year == 2026 else None

    monkeypatch.setattr(fred.treasury, "_fetch_csv", _treasury_fetch)

    result = fred.get_series_asof(fred.SERIES_DGS10, as_of="2026-01-31", percentile_years=10)

    # start_year = cutoff_year(2026) - percentile_years(10) - 1 = 2015, so
    # years 2015..2026 (12 years) must all have been attempted.
    assert set(calls) == set(range(2015, 2027))
    assert result["window_n"] == 30
    assert result["percentile"] is not None


# ---------------------------------------------------------------------------
# Regression guard -- a cache write must never reach the real Config.RAW_DIR.
# ---------------------------------------------------------------------------


def test_cache_write_never_leaks_into_the_real_raw_dir(tmp_path, monkeypatch):
    """Directly exercises the exact path that leaked before (see module
    docstring): a forced fetch (``no_cache=True``) that writes a cache file
    and then hits a downstream failure. Asserts the REAL cache directory
    (``_REAL_RAW_DIR``, captured before this module's autouse fixture ever
    ran) gets nothing written to it, regardless of what ``tmp_path`` this
    particular test run happens to redirect to.
    """
    monkeypatch.setattr(fred.Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fred, "_fetch_csv", lambda series: _CANNED_CSV_LEGACY_HEADER)
    monkeypatch.setattr(fred, "_parse_asof", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    assert fred.get_risk_free_asof("2022-06-30", no_cache=True) is None

    if os.path.isdir(_REAL_RAW_DIR):
        leaked = [f for f in os.listdir(_REAL_RAW_DIR) if f.startswith("fred_")]
        assert leaked == [], f"cache write leaked into the real RAW_DIR: {leaked}"
