"""Unit tests for ``sec_analyzer.screener.overview.build_overview``.

Binding spec: the overview task description (portfolio/sector overview
aggregation). Pure unit tests -- ``_build_universe_lookup`` is monkeypatched
in every test so no bundled CSV is ever read, and ``today`` is always passed
explicitly for deterministic date arithmetic (mirrors
``test_swing_score.py``'s convention of unit-testing a pure function with
hand-built inputs).
"""

import json
import re
from datetime import date

import pytest

from sec_analyzer.screener import overview
from sec_analyzer.screener.overview import (
    CHEAP_THRESHOLD_PCT,
    EXPENSIVE_THRESHOLD_PCT,
    SECTOR_TYPE_LABELS,
    build_overview,
    sector_type_label,
)

TODAY = date(2026, 8, 7)


def _empty_lookup():
    return {}


@pytest.fixture(autouse=True)
def _no_csv(monkeypatch):
    """Every test in this module runs with no bundled universe lookup unless
    it explicitly overrides ``overview._build_universe_lookup`` itself."""
    monkeypatch.setattr(overview, "_build_universe_lookup", _empty_lookup)


def _row(ticker="AAA", **kwargs) -> dict:
    """A minimal valid input row, overridable per test."""
    base = {
        "id": 1,
        "cik": "0000000001",
        "ticker": ticker,
        "analyzed_at": "2026-08-01T10:00:00",
        "as_of": None,
        "horizon": "1y",
        "provider": "script",
        "price": 100.0,
        "fundamental_verdict": None,
        "technical_verdict": None,
        "profile_fit": None,
        "momentum_verdict": None,
        "fv_bear_lo": None,
        "fv_bear_hi": None,
        "fv_base_lo": None,
        "fv_base_hi": None,
        "fv_bull_lo": None,
        "fv_bull_hi": None,
        "confidence": None,
        "sector_type": None,
        "implied_growth": None,
        "watch_note": None,
    }
    base.update(kwargs)
    return base


def _row_by_ticker(result: dict, ticker: str) -> dict:
    for row in result["rows"]:
        if row["ticker"] == ticker:
            return row
    raise AssertionError(f"{ticker} not found in result rows")


# ---------------------------------------------------------------------------
# fv_base_mid / fv_vs_price_pct
# ---------------------------------------------------------------------------


def test_fv_vs_price_pct_hand_computed():
    row = _row(price=200.0, fv_base_lo=220.0, fv_base_hi=260.0)
    result = build_overview([row], today=TODAY)
    enriched = result["rows"][0]
    assert enriched["fv_base_mid"] == 240.0
    assert enriched["fv_vs_price_pct"] == 20.0


def test_fv_vs_price_pct_none_when_price_zero():
    row = _row(price=0.0, fv_base_lo=100.0, fv_base_hi=120.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_base_mid"] == 110.0
    assert enriched["fv_vs_price_pct"] is None


def test_fv_vs_price_pct_none_when_price_missing():
    row = _row(price=None, fv_base_lo=100.0, fv_base_hi=120.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_vs_price_pct"] is None


def test_fv_base_mid_none_when_band_missing():
    row = _row(price=100.0, fv_base_lo=100.0, fv_base_hi=None)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_base_mid"] is None
    assert enriched["fv_vs_price_pct"] is None


# ---------------------------------------------------------------------------
# value_bucket boundaries
# ---------------------------------------------------------------------------


def test_value_bucket_cheap_boundary_inclusive():
    row = _row(price=100.0, fv_base_lo=115.0, fv_base_hi=115.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_vs_price_pct"] == CHEAP_THRESHOLD_PCT
    assert enriched["value_bucket"] == "ucuz"


def test_value_bucket_expensive_boundary_inclusive():
    row = _row(price=100.0, fv_base_lo=85.0, fv_base_hi=85.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_vs_price_pct"] == EXPENSIVE_THRESHOLD_PCT
    assert enriched["value_bucket"] == "pahali"


def test_value_bucket_fair_between_thresholds():
    row = _row(price=100.0, fv_base_lo=100.0, fv_base_hi=100.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_vs_price_pct"] == 0.0
    assert enriched["value_bucket"] == "makul"


def test_value_bucket_none_when_pct_unknown():
    row = _row(price=None, fv_base_lo=100.0, fv_base_hi=100.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["value_bucket"] is None


# ---------------------------------------------------------------------------
# heat
# ---------------------------------------------------------------------------


def test_heat_saturates_beyond_positive_40_pct():
    # price 50, fv_base_mid 100 -> pct = 100.0, well beyond +40.
    row = _row(price=50.0, fv_base_lo=100.0, fv_base_hi=100.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_vs_price_pct"] == 100.0
    assert enriched["heat"] == 1.0


def test_heat_saturates_beyond_negative_40_pct():
    # price 100, fv_base_mid 10 -> pct = -90.0, well beyond -40.
    row = _row(price=100.0, fv_base_lo=10.0, fv_base_hi=10.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_vs_price_pct"] == -90.0
    assert enriched["heat"] == -1.0


def test_heat_scales_linearly_within_range():
    # price 100, fv_base_mid 120 -> pct = 20.0 -> heat = 20/40 = 0.5
    row = _row(price=100.0, fv_base_lo=120.0, fv_base_hi=120.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["fv_vs_price_pct"] == 20.0
    assert enriched["heat"] == 0.5


def test_heat_none_when_pct_missing():
    row = _row(price=None, fv_base_lo=100.0, fv_base_hi=100.0)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["heat"] is None


# ---------------------------------------------------------------------------
# verdict_drift
# ---------------------------------------------------------------------------


def test_verdict_drift_true_when_stored_pahali_but_bucket_ucuz():
    row = _row(price=100.0, fv_base_lo=120.0, fv_base_hi=120.0, fundamental_verdict="PAHALI")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["value_bucket"] == "ucuz"
    assert enriched["verdict_drift"] is True


def test_verdict_drift_false_when_stored_matches_bucket():
    row = _row(price=100.0, fv_base_lo=120.0, fv_base_hi=120.0, fundamental_verdict="UCUZ")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["value_bucket"] == "ucuz"
    assert enriched["verdict_drift"] is False


def test_verdict_drift_false_when_verdict_unknown():
    row = _row(price=100.0, fv_base_lo=120.0, fv_base_hi=120.0, fundamental_verdict=None)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["value_bucket"] == "ucuz"
    assert enriched["verdict_drift"] is False


def test_verdict_drift_false_when_bucket_unknown():
    row = _row(price=None, fv_base_lo=120.0, fv_base_hi=120.0, fundamental_verdict="PAHALI")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["value_bucket"] is None
    assert enriched["verdict_drift"] is False


# ---------------------------------------------------------------------------
# age_days / stale
# ---------------------------------------------------------------------------


def test_age_days_and_stale_full_iso_timestamp():
    analyzed = date(2026, 4, 1)
    row = _row(analyzed_at="2026-04-01T09:30:00")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    expected_age = (TODAY - analyzed).days
    assert enriched["age_days"] == expected_age
    assert expected_age > 90
    assert enriched["stale"] is True


def test_age_days_and_stale_bare_date_string():
    analyzed = date(2026, 8, 1)
    row = _row(analyzed_at="2026-08-01")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    expected_age = (TODAY - analyzed).days
    assert enriched["age_days"] == expected_age
    assert enriched["stale"] is False


def test_age_days_none_and_not_stale_when_unparseable():
    row = _row(analyzed_at="not-a-date")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["age_days"] is None
    assert enriched["stale"] is False


def test_stale_respects_custom_stale_days():
    row = _row(analyzed_at="2026-07-01T00:00:00")  # 37 days before TODAY
    enriched = build_overview([row], today=TODAY, stale_days=30)["rows"][0]
    assert enriched["age_days"] == 37
    assert enriched["stale"] is True
    enriched2 = build_overview([row], today=TODAY, stale_days=60)["rows"][0]
    assert enriched2["stale"] is False


# ---------------------------------------------------------------------------
# days_until_earnings
# ---------------------------------------------------------------------------


def test_days_until_earnings_negative_for_past_catalyst():
    row = _row(catalyst_date="2026-08-01")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["days_until_earnings"] == -6


def test_days_until_earnings_positive_for_future_catalyst():
    row = _row(catalyst_date="2026-08-21")
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["days_until_earnings"] == 14


def test_days_until_earnings_none_when_key_absent():
    row = _row()
    assert "catalyst_date" not in row
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["days_until_earnings"] is None


# ---------------------------------------------------------------------------
# Legacy row (no insider_verdict / catalyst_date keys at all)
# ---------------------------------------------------------------------------


def test_legacy_row_missing_insider_and_catalyst_keys():
    row = _row(ticker="LEGACY")
    assert "insider_verdict" not in row
    assert "catalyst_date" not in row
    result = build_overview([row], today=TODAY)
    enriched = result["rows"][0]
    assert enriched["days_until_earnings"] is None
    assert result["insider_counts"]["bilinmiyor"] == 1
    assert result["insider_counts"]["ALIM"] == 0


# ---------------------------------------------------------------------------
# Sector grouping
# ---------------------------------------------------------------------------


def test_sector_grouping_medians_counts_and_unclassified_bucket(monkeypatch):
    monkeypatch.setattr(
        overview,
        "_build_universe_lookup",
        lambda: {
            "AAA": ("Alpha Co", "Information Technology"),
            "BBB": ("Beta Co", "Information Technology"),
            "CCC": ("Gamma Co", None),
        },
    )
    rows = [
        _row(ticker="AAA", price=100.0, fv_base_lo=120.0, fv_base_hi=120.0),  # pct 20 -> ucuz
        _row(ticker="BBB", price=100.0, fv_base_lo=80.0, fv_base_hi=80.0),  # pct -20 -> pahali
        _row(ticker="CCC", price=100.0, fv_base_lo=100.0, fv_base_hi=100.0),  # pct 0 -> makul, no sector
    ]
    result = build_overview(rows, today=TODAY)
    sectors = {s["sector"]: s for s in result["sectors"]}

    it_sector = sectors["Information Technology"]
    assert it_sector["label"] == "Information Technology"
    assert it_sector["n"] == 2
    assert it_sector["median_fv_vs_price_pct"] == 0.0  # median of +20 and -20
    assert it_sector["cheap_n"] == 1
    assert it_sector["expensive_n"] == 1
    assert it_sector["fair_n"] == 0
    assert it_sector["tickers"] == ["AAA", "BBB"]

    unclassified = sectors[None]
    assert unclassified["label"] == "Sınıflandırılmamış"
    assert unclassified["n"] == 1
    assert unclassified["fair_n"] == 1
    assert unclassified["tickers"] == ["CCC"]


# ---------------------------------------------------------------------------
# Sector vocabulary aliasing (Defect 1)
# ---------------------------------------------------------------------------


def test_canonicalize_sector_aliases_case_insensitive():
    assert overview._canonicalize_sector("Technology") == "Information Technology"
    assert overview._canonicalize_sector("technology") == "Information Technology"
    assert overview._canonicalize_sector("  TECHNOLOGY  ") == "Information Technology"
    assert overview._canonicalize_sector("Basic Materials") == "Materials"
    assert overview._canonicalize_sector("Telecommunications") == "Communication Services"


def test_canonicalize_sector_unknown_passes_through_unchanged():
    assert overview._canonicalize_sector("Some Made Up Sector") == "Some Made Up Sector"
    assert overview._canonicalize_sector(None) is None
    assert overview._canonicalize_sector("") == ""


def test_aliasing_merges_two_csv_vocabularies_into_one_sector_tile(monkeypatch):
    # One row's sector came from nasdaq100.csv's "Technology", the other from
    # sp500.csv's "Information Technology" -- _build_universe_lookup is
    # responsible for canonicalizing both onto the same string before this
    # module ever sees them, so they must land in the SAME heat-map tile.
    monkeypatch.setattr(
        overview,
        "_build_universe_lookup",
        lambda: {
            "NDXCO": ("Nasdaq Co", overview._canonicalize_sector("Technology")),
            "SPCO": ("SP500 Co", "Information Technology"),
        },
    )
    rows = [_row(ticker="NDXCO"), _row(ticker="SPCO")]
    result = build_overview(rows, today=TODAY)
    tech_sectors = [s for s in result["sectors"] if s["sector"] == "Information Technology"]
    assert len(tech_sectors) == 1
    assert tech_sectors[0]["n"] == 2
    assert tech_sectors[0]["tickers"] == ["NDXCO", "SPCO"]


# ---------------------------------------------------------------------------
# Sector fallback chain: index lookup -> SIC -> None (Defect 2)
# ---------------------------------------------------------------------------


def test_sector_from_index_lookup_wins_over_sic(monkeypatch):
    # In the universe lookup AND carries a sic that would map elsewhere --
    # the index sector must still win, with sector_source == "index".
    monkeypatch.setattr(overview, "_build_universe_lookup", lambda: {"AAA": ("Alpha Co", "Health Care")})
    row = _row(ticker="AAA", sic="7370")  # would map to XLK / Information Technology
    enriched = build_overview([row], today=TODAY)["rows"][0]

    assert enriched["sector"] == "Health Care"
    assert enriched["sector_source"] == "index"


def test_sector_from_sic_xlk_range_when_not_in_index():
    row = _row(ticker="ZZZ", sic="7370")  # software/IT services -> XLK
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["sector"] == "Information Technology"
    assert enriched["sector_source"] == "sic"


def test_sector_from_sic_xlf_range_when_not_in_index():
    row = _row(ticker="ZZZ", sic="6022")  # state commercial banks -> XLF
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["sector"] == "Financials"
    assert enriched["sector_source"] == "sic"


def test_sector_from_sic_single_semiconductor_code():
    row = _row(ticker="ZZZ", sic="3674")  # semiconductors -> SMH
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["sector"] == "Information Technology"
    assert enriched["sector_source"] == "sic"


def test_sector_none_and_source_none_when_neither_index_nor_sic():
    row = _row(ticker="NOBODY")  # not in lookup, no sic key at all
    result = build_overview([row], today=TODAY)
    enriched = result["rows"][0]
    assert enriched["sector"] is None
    assert enriched["sector_source"] is None
    unclassified = [s for s in result["sectors"] if s["sector"] is None]
    assert len(unclassified) == 1
    assert unclassified[0]["label"] == "Sınıflandırılmamış"
    assert "NOBODY" in unclassified[0]["tickers"]


@pytest.mark.parametrize("bad_sic", ["", "abc", None])
def test_sector_from_sic_degrades_gracefully_on_unparseable_sic(bad_sic):
    row = _row(ticker="ZZZ", sic=bad_sic)
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["sector"] is None
    assert enriched["sector_source"] is None


def test_sector_from_sic_missing_key_entirely():
    row = _row(ticker="ZZZ")
    assert "sic" not in row
    enriched = build_overview([row], today=TODAY)["rows"][0]
    assert enriched["sector"] is None
    assert enriched["sector_source"] is None


# ---------------------------------------------------------------------------
# Routes grouping
# ---------------------------------------------------------------------------


def test_routes_grouping_uses_turkish_labels_and_unknown_bucket():
    rows = [
        _row(ticker="AAA", sector_type="mature"),
        _row(ticker="BBB", sector_type="cyclical"),
        _row(ticker="CCC", sector_type=None),
        _row(ticker="DDD", sector_type="not_a_real_route"),
    ]
    result = build_overview(rows, today=TODAY)
    routes = {r["sector_type"]: r for r in result["routes"]}

    assert routes["mature"]["label"] == SECTOR_TYPE_LABELS["mature"]
    assert routes["cyclical"]["label"] == SECTOR_TYPE_LABELS["cyclical"]
    assert routes[None]["label"] == "Bilinmiyor"
    assert routes[None]["tickers"] == ["CCC"]
    assert routes["not_a_real_route"]["label"] == "Bilinmiyor"

    assert sector_type_label(None) == "Bilinmiyor"
    assert sector_type_label("garbage") == "Bilinmiyor"
    assert sector_type_label("reit") == "GYO"


# ---------------------------------------------------------------------------
# Row sort order
# ---------------------------------------------------------------------------


def test_row_sort_order_descending_pct_none_last_ticker_tiebreak():
    rows = [
        _row(ticker="ZZZ", price=100.0, fv_base_lo=100.0, fv_base_hi=100.0),  # pct 0
        _row(ticker="AAA", price=100.0, fv_base_lo=100.0, fv_base_hi=100.0),  # pct 0, tie -> ticker asc
        _row(ticker="HIGH", price=100.0, fv_base_lo=150.0, fv_base_hi=150.0),  # pct 50
        _row(ticker="NOPCT", price=None, fv_base_lo=100.0, fv_base_hi=100.0),  # pct None -> last
        _row(ticker="LOW", price=100.0, fv_base_lo=50.0, fv_base_hi=50.0),  # pct -50
    ]
    result = build_overview(rows, today=TODAY)
    tickers_in_order = [r["ticker"] for r in result["rows"]]
    assert tickers_in_order == ["HIGH", "AAA", "ZZZ", "LOW", "NOPCT"]


# ---------------------------------------------------------------------------
# Count dicts always carry every known key
# ---------------------------------------------------------------------------


def test_count_dicts_carry_all_known_keys_even_when_zero():
    row = _row(fundamental_verdict="UCUZ", confidence="YÜKSEK", momentum_verdict="POZİTİF", insider_verdict="ALIM")
    result = build_overview([row], today=TODAY)

    assert result["verdict_counts"] == {"UCUZ": 1, "MAKUL": 0, "PAHALI": 0, "bilinmiyor": 0}
    assert result["confidence_counts"] == {"YÜKSEK": 1, "ORTA": 0, "DÜŞÜK": 0, "bilinmiyor": 0}
    assert result["momentum_counts"] == {"GÜÇLÜ+": 0, "POZİTİF": 1, "NÖTR": 0, "NEGATİF": 0, "bilinmiyor": 0}
    assert result["insider_counts"] == {
        "GÜÇLÜ ALIM": 0,
        "ALIM": 1,
        "NÖTR": 0,
        "SATIŞ": 0,
        "YOĞUN SATIŞ": 0,
        "bilinmiyor": 0,
    }
    # bucket_counts is derived (value_bucket unset here -> "bilinmiyor")
    assert result["bucket_counts"]["bilinmiyor"] == 1


# ---------------------------------------------------------------------------
# Upcoming earnings / drift lists
# ---------------------------------------------------------------------------


def test_upcoming_earnings_window_and_sort():
    rows = [
        _row(ticker="SOON", catalyst_date="2026-08-10"),  # +3
        _row(ticker="TODAYX", catalyst_date="2026-08-07"),  # 0
        _row(ticker="LATER", catalyst_date="2026-09-01"),  # outside default 21-day window
        _row(ticker="PAST", catalyst_date="2026-08-01"),  # negative, excluded
    ]
    result = build_overview(rows, today=TODAY)
    tickers = [r["ticker"] for r in result["upcoming_earnings"]]
    assert tickers == ["TODAYX", "SOON"]


def test_drift_list_sorted_by_abs_pct_desc():
    rows = [
        _row(ticker="SMALL", price=100.0, fv_base_lo=116.0, fv_base_hi=116.0, fundamental_verdict="PAHALI"),  # +16
        _row(ticker="BIG", price=100.0, fv_base_lo=140.0, fv_base_hi=140.0, fundamental_verdict="PAHALI"),  # +40
        _row(ticker="AGREE", price=100.0, fv_base_lo=120.0, fv_base_hi=120.0, fundamental_verdict="UCUZ"),  # no drift
    ]
    result = build_overview(rows, today=TODAY)
    tickers = [r["ticker"] for r in result["drift"]]
    assert tickers == ["BIG", "SMALL"]


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_build_overview_empty_rows():
    result = build_overview([], today=TODAY)
    assert result["count"] == 0
    assert result["rows"] == []
    assert result["sectors"] == []
    assert result["routes"] == []
    assert result["stale"] == []
    assert result["stale_count"] == 0
    assert result["upcoming_earnings"] == []
    assert result["drift"] == []
    assert result["median_fv_vs_price_pct"] is None
    assert result["verdict_counts"] == {"UCUZ": 0, "MAKUL": 0, "PAHALI": 0, "bilinmiyor": 0}
    assert result["today"] == "2026-08-07"


def test_build_overview_none_input_does_not_raise():
    result = build_overview(None, today=TODAY)
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_input_same_output_ignoring_generated_at():
    rows = [
        _row(ticker="AAA", price=100.0, fv_base_lo=120.0, fv_base_hi=130.0, catalyst_date="2026-08-15"),
        _row(ticker="BBB", price=50.0, fv_base_lo=40.0, fv_base_hi=45.0, analyzed_at="2026-01-01"),
    ]
    result1 = build_overview(rows, today=TODAY)
    result2 = build_overview(rows, today=TODAY)
    result1.pop("generated_at")
    result2.pop("generated_at")
    assert result1 == result2


# ---------------------------------------------------------------------------
# No "None"/"nan" leakage into rendered strings
# ---------------------------------------------------------------------------


def _collect_strings(value, out):
    if isinstance(value, dict):
        for v in value.values():
            _collect_strings(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_strings(v, out)
    elif isinstance(value, str):
        out.append(value)


def test_no_none_or_nan_substrings_in_rendered_strings():
    """Every *string* value in the payload must be free of the literal
    tokens "None"/"nan" as whole words -- missing data is the Python
    ``None`` object (which ``json.dumps`` renders as ``null``, never the
    string "None"), not a stringified placeholder. Checked as whole tokens
    (not raw substring containment) because legitimate Turkish labels can
    contain "nan" as a substring (e.g. "Finansal")."""
    rows = [
        _row(ticker="AAA"),  # mostly-None row: nothing computable
        _row(ticker="BBB", price=100.0, fv_base_lo=120.0, fv_base_hi=130.0, fundamental_verdict="UCUZ"),
        _row(ticker="CCC", sector_type="financial"),  # exercises the "Finansal" label
    ]
    result = build_overview(rows, today=TODAY)
    strings = []
    _collect_strings(result, strings)
    for s in strings:
        tokens = re.split(r"[^A-Za-z]+", s)
        for token in tokens:
            assert token.lower() not in ("none", "nan"), f"forbidden token in rendered string: {s!r}"

    # And the JSON-serialized form (what the web API / CLI would actually
    # emit) never contains the bare "None" token either -- `None` serializes
    # to `null`, not the string "None".
    blob = json.dumps(result, ensure_ascii=False, default=str)
    assert "None" not in blob
