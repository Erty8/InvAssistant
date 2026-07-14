"""Tests for ``valuation.damodaran`` (SIC->industry matcher + CSV loader).

``sector_medians()`` was hardened with an alias table (``_ALIASES``,
checked first, most-specific-first) plus a fuzzy token-overlap fallback
(``_tokenize``/``_STOPWORDS``) that requires at least one shared
*distinctive* token, so generic connector words (e.g. "related", "services")
can no longer manufacture a false match on their own. These tests build a
small in-memory ``sector_data`` fixture (the shape ``load_sector_data``
returns) rather than depending on the real CSVs in ``data/damodaran/``, so
they stay correct regardless of what an operator has dropped in that folder.
"""

import pytest

from sec_analyzer.valuation.damodaran import load_sector_data, sector_medians

# ---------------------------------------------------------------------------
# Fixture: a small in-memory sector_data dict covering every industry the
# acceptance cases below need, plus a "Coal & Related Energy" row that exists
# solely to prove the old "related" false-match bug stays fixed, and a
# "Steel" row that exists solely to exercise the fuzzy fallback path (no
# alias covers steel).
# ---------------------------------------------------------------------------

_INDUSTRIES = [
    ("Semiconductor", 28.4, 6.1, 24.7),
    ("Coal & Related Energy", 9.1, 0.8, 8.0),
    ("Software (System & Application)", 32.0, 8.2, 27.5),
    ("Drugs (Pharmaceutical)", 19.5, 4.0, 18.0),
    ("Drugs (Biotechnology)", 22.0, 5.5, 20.0),
    ("Retail (General)", 18.6, 1.1, 17.2),
    ("Retail (Grocery and Food)", 16.0, 0.4, 15.0),
    ("Restaurant/Dining", 24.0, 2.1, 21.0),
    ("Banks (Regional)", 11.0, 2.9, 12.0),
    ("Oil/Gas (Production and Exploration)", 10.5, 1.5, 9.0),
    ("Oil/Gas (Integrated)", 8.0, 0.9, 7.5),
    ("Auto & Truck", 9.8, 0.6, 10.2),
    ("Aerospace/Defense", 20.1, 1.4, 19.0),
    ("Computers/Peripherals", 21.3, 2.6, 20.5),
    ("Telecom. Services", 15.0, 1.9, 14.0),
    ("Telecom (Wireless)", 14.2, 2.0, 13.5),
    ("R.E.I.T.", 17.0, 6.0, 16.0),
    ("Computer Services", 23.0, 3.2, 22.0),
    ("Advertising", 20.0, 1.8, 19.5),
    ("Steel", 12.0, 1.0, 11.0),
]


def _sector_data():
    return {
        "multiples": [
            {"industry": name, "pe": pe, "ps": ps, "pfcf": pfcf}
            for name, pe, ps, pfcf in _INDUSTRIES
        ],
        "erp": 4.6,
    }


# ---------------------------------------------------------------------------
# Acceptance table: sicDescription -> expected Damodaran industry.
# ---------------------------------------------------------------------------

_ACCEPTANCE_CASES = [
    ("SEMICONDUCTORS & RELATED DEVICES", "Semiconductor"),
    ("SERVICES-PREPACKAGED SOFTWARE", "Software (System & Application)"),
    ("PHARMACEUTICAL PREPARATIONS", "Drugs (Pharmaceutical)"),
    ("BIOLOGICAL PRODUCTS, (NO DIAGNOSTIC SUBSTANCES)", "Drugs (Biotechnology)"),
    ("RETAIL-VARIETY STORES", "Retail (General)"),
    ("RETAIL-GROCERY STORES", "Retail (Grocery and Food)"),
    ("RETAIL-EATING PLACES", "Restaurant/Dining"),
    ("NATIONAL COMMERCIAL BANKS", "Banks (Regional)"),
    ("STATE COMMERCIAL BANKS", "Banks (Regional)"),
    ("CRUDE PETROLEUM & NATURAL GAS", "Oil/Gas (Production and Exploration)"),
    ("PETROLEUM REFINING", "Oil/Gas (Integrated)"),
    ("MOTOR VEHICLES & PASSENGER CAR BODIES", "Auto & Truck"),
    ("AIRCRAFT", "Aerospace/Defense"),
    ("ELECTRONIC COMPUTERS", "Computers/Peripherals"),
    ("TELEPHONE COMMUNICATIONS (NO RADIOTELEPHONE)", "Telecom. Services"),
    ("RADIOTELEPHONE COMMUNICATIONS", "Telecom (Wireless)"),
    ("REAL ESTATE INVESTMENT TRUSTS", "R.E.I.T."),
    ("SERVICES-COMPUTER PROGRAMMING, DATA PROCESSING, ETC.", "Computer Services"),
    ("ADVERTISING SERVICES", "Advertising"),
]


@pytest.mark.parametrize("sic_description, expected_industry", _ACCEPTANCE_CASES)
def test_sector_medians_alias_acceptance_table(sic_description, expected_industry):
    result = sector_medians(_sector_data(), sic_description)
    assert result is not None
    assert result["industry"] == expected_industry


def test_sector_medians_alias_table_returns_full_row_shape():
    # Spot-check the full row shape (not just "industry") for one case.
    result = sector_medians(_sector_data(), "PHARMACEUTICAL PREPARATIONS")
    assert result == {
        "industry": "Drugs (Pharmaceutical)", "pe": 19.5, "ps": 4.0, "pfcf": 18.0,
        "growth": None, "peg": None,
    }


# ---------------------------------------------------------------------------
# Regression test for the bug this hardening fixed: the shared generic word
# "related" must never manufacture a match between "SEMICONDUCTORS & RELATED
# DEVICES" and "Coal & Related Energy". The alias table resolves this SIC
# description to "Semiconductor" (see acceptance table above); this test
# additionally asserts the old wrong answer is not produced, with the
# "Coal & Related Energy" row present in the fixture specifically so this
# assertion is meaningful (it would be trivially true if that row didn't
# exist to tempt a match).
# ---------------------------------------------------------------------------


def test_sector_medians_does_not_false_match_on_shared_word_related():
    result = sector_medians(_sector_data(), "SEMICONDUCTORS & RELATED DEVICES")
    assert result is not None
    assert result["industry"] == "Semiconductor"
    assert result["industry"] != "Coal & Related Energy"


# ---------------------------------------------------------------------------
# Fuzzy fallback (stage 2): no alias covers "Steel", so a genuine distinctive
# token overlap ("steel") must still resolve it; pure nonsense must not.
# ---------------------------------------------------------------------------


def test_sector_medians_fuzzy_fallback_resolves_via_distinctive_token_overlap():
    result = sector_medians(_sector_data(), "STEEL WORKS, BLAST FURNACES & ROLLING MILLS")
    assert result is not None
    assert result["industry"] == "Steel"


def test_sector_medians_fuzzy_fallback_returns_none_for_nonsense_string():
    result = sector_medians(_sector_data(), "ZZZ QReW")
    assert result is None


# ---------------------------------------------------------------------------
# Negative / edge cases.
# ---------------------------------------------------------------------------


def test_sector_medians_returns_none_when_sector_data_is_none():
    assert sector_medians(None, "SEMICONDUCTORS & RELATED DEVICES") is None


def test_sector_medians_returns_none_when_multiples_missing():
    assert sector_medians({"erp": 4.6}, "SEMICONDUCTORS & RELATED DEVICES") is None


def test_sector_medians_returns_none_when_multiples_empty_list():
    assert sector_medians({"multiples": [], "erp": 4.6}, "SEMICONDUCTORS & RELATED DEVICES") is None


def test_sector_medians_returns_none_when_sic_description_is_none():
    assert sector_medians(_sector_data(), None) is None


def test_sector_medians_returns_none_when_sic_description_is_whitespace():
    assert sector_medians(_sector_data(), "   ") is None


# ---------------------------------------------------------------------------
# load_sector_data: CSV loading from a directory, per data/damodaran/README.md
# (headers: "industry,pe,ps,pfcf" and "region,erp"; only the US erp row is
# used).
# ---------------------------------------------------------------------------


def test_load_sector_data_parses_multiples_and_us_erp_from_csvs(tmp_path):
    multiples_csv = tmp_path / "multiples.csv"
    multiples_csv.write_text(
        "industry,pe,ps,pfcf\n"
        "Semiconductor,28.4,6.1,24.7\n"
        "Retail (General),18.6,1.1,17.2\n",
        encoding="utf-8",
    )
    erp_csv = tmp_path / "erp.csv"
    erp_csv.write_text(
        "region,erp\n"
        "US,4.6\n"
        "Europe,5.1\n",
        encoding="utf-8",
    )

    result = load_sector_data(str(tmp_path))

    assert result is not None
    assert result["erp"] == pytest.approx(4.6)
    assert isinstance(result["erp"], float)
    assert result["multiples"] == [
        {"industry": "Semiconductor", "pe": 28.4, "ps": 6.1, "pfcf": 24.7, "growth": None, "peg": None},
        {"industry": "Retail (General)", "pe": 18.6, "ps": 1.1, "pfcf": 17.2, "growth": None, "peg": None},
    ]


def test_load_sector_data_returns_none_for_missing_directory(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    assert load_sector_data(str(missing_dir)) is None
