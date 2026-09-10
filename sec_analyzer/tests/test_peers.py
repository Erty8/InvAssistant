"""Unit tests for :mod:`sec_analyzer.screener.peers` (cross-sectional peer
comparison built on SEC XBRL Frames).

Pure/unit throughout: no network, no real universe CSV. ``load_universe``
and ``get_frame_with_aliases`` are monkeypatched directly on the
``sec_analyzer.screener.peers`` module (the names it imported them under),
and an AUTOUSE fixture redirects ``Config.RAW_DIR`` to ``tmp_path`` for every
test in this file regardless of whether a given test happens to touch the
cache -- opt-out isolation, not opt-in, so a future test added here without
remembering to ask for isolation still cannot write into the real,
multi-gigabyte ``sec_analyzer/raw`` cache.

Fixture data (see ``_FAKE_UNIVERSE`` / ``_FAKE_FRAMES`` below): a 7-company,
2-sector peer set built from hand-computed inputs -- every metric's expected
value, and every sector's median/p25/p75, is checked against a value
verified independently via ``statistics.median``/``statistics.quantiles``
(the exact functions ``peers._stats`` uses), documented inline.
"""

import pytest

from sec_analyzer.config import Config
from sec_analyzer.screener import peers

_YEAR = 2024


@pytest.fixture(autouse=True)
def isolate_raw_dir(tmp_path, monkeypatch):
    """Redirect Config.RAW_DIR for every test in this file (autouse, not
    opt-in) -- this module's tests are pure/unit and should touch no cache
    at all, but the guard costs nothing and matches test_frames.py."""
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

# Seven companies, two sectors. CIKs 1-5 are "Information Technology" (three
# spelled that way in the fake SP500 universe, two spelled "Technology" in
# the fake NDX universe -- the canonicalization case). CIKs 6-7 are "Health
# Care" (n=2, deliberately below the n>=5 percentile-reporting floor).
#
#           revenue  rev_prior  net_income  op_income  gross_profit  ocf  capex  equity  assets  liabilities
# cik 1        1000        800         100        150           600  140      0     500    2000           300
# cik 2        1000        800         150        200           600  190      0     500    2000           300
# cik 3        1000        800         200        250           600  240      0     500    2000           300
# cik 4        1000        800         250        300           600  290      0     500    2000           300
# cik 5        1000        800         300        350           600  340      0    -100    2000           400  <- negative equity trap
# cik 6         500        400          50         70           300   90      0     200    1000           100
# cik 7         600        480          60         84           360  108      0     240    1200           120
_RAW = {
    1: dict(revenue=1000.0, revenue_prior=800.0, net_income=100.0, operating_income=150.0,
            gross_profit=600.0, ocf=140.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0),
    2: dict(revenue=1000.0, revenue_prior=800.0, net_income=150.0, operating_income=200.0,
            gross_profit=600.0, ocf=190.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0),
    3: dict(revenue=1000.0, revenue_prior=800.0, net_income=200.0, operating_income=250.0,
            gross_profit=600.0, ocf=240.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0),
    4: dict(revenue=1000.0, revenue_prior=800.0, net_income=250.0, operating_income=300.0,
            gross_profit=600.0, ocf=290.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0),
    5: dict(revenue=1000.0, revenue_prior=800.0, net_income=300.0, operating_income=350.0,
            gross_profit=600.0, ocf=340.0, capex=0.0, equity=-100.0, assets=2000.0, liabilities=400.0),
    6: dict(revenue=500.0, revenue_prior=400.0, net_income=50.0, operating_income=70.0,
            gross_profit=300.0, ocf=90.0, capex=0.0, equity=200.0, assets=1000.0, liabilities=100.0),
    7: dict(revenue=600.0, revenue_prior=480.0, net_income=60.0, operating_income=84.0,
            gross_profit=360.0, ocf=108.0, capex=0.0, equity=240.0, assets=1200.0, liabilities=120.0),
}

_FAKE_UNIVERSE = {
    "SP500": [
        {"ticker": "AAA", "name": "Company AAA", "sector": "Information Technology", "cik": "1"},
        {"ticker": "BBB", "name": "Company BBB", "sector": "Information Technology", "cik": "2"},
        {"ticker": "CCC", "name": "Company CCC", "sector": "Information Technology", "cik": "3"},
        {"ticker": "DDD", "name": "Company DDD", "sector": "Health Care", "cik": "6"},
        {"ticker": "EEE", "name": "Company EEE", "sector": "Health Care", "cik": "7"},
        {"ticker": "GGG", "name": "Company GGG", "sector": "Information Technology", "cik": None},
    ],
    "NDX": [
        # "Technology" (nasdaq100.csv's spelling) must land in the SAME
        # bucket as "Information Technology" above.
        {"ticker": "HHH", "name": "Company HHH", "sector": "Technology", "cik": "4"},
        {"ticker": "III", "name": "Company III", "sector": "Technology", "cik": "5"},
    ],
}


def _fake_load_universe(index="SP500", path=None):
    return list(_FAKE_UNIVERSE.get(index, []))


# Concept -> tag used as the dict key below (matches CONCEPTS[...][0] for
# every concept peers.py fetches, so keying on tags[0] unambiguously
# identifies which concept is being requested).
_TAG_TO_FIELD_AND_YEAR = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("revenue", _YEAR),
    "NetIncomeLoss": ("net_income", _YEAR),
    "OperatingIncomeLoss": ("operating_income", _YEAR),
    "GrossProfit": ("gross_profit", _YEAR),
    "NetCashProvidedByUsedInOperatingActivities": ("ocf", _YEAR),
    "PaymentsToAcquirePropertyPlantAndEquipment": ("capex", _YEAR),
    "StockholdersEquity": ("equity", _YEAR),
    "Assets": ("assets", _YEAR),
    "Liabilities": ("liabilities", _YEAR),
}


def _fake_get_frame_with_aliases(tags, period, client, unit="USD", taxonomy="us-gaap", no_cache=False):
    tag = tags[0]
    if tag == "RevenueFromContractWithCustomerExcludingAssessedTax" and period == f"CY{_YEAR - 1}":
        return {cik: row["revenue_prior"] for cik, row in _RAW.items()}
    field, year = _TAG_TO_FIELD_AND_YEAR.get(tag, (None, None))
    if field is None:
        return {}
    return {cik: row[field] for cik, row in _RAW.items()}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(peers, "load_universe", _fake_load_universe)
    monkeypatch.setattr(peers, "get_frame_with_aliases", _fake_get_frame_with_aliases)
    return None


# ---------------------------------------------------------------------------
# _derive_company_metrics -- hand-computed arithmetic
# ---------------------------------------------------------------------------


def test_derive_metrics_hand_computed_case():
    m = peers._derive_company_metrics(
        revenue=1000.0, revenue_prior=800.0, net_income=200.0, operating_income=250.0,
        gross_profit=600.0, ocf=240.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0,
    )
    assert m["revenue"] == 1000.0
    assert m["net_margin"] == pytest.approx(0.2)
    assert m["operating_margin"] == pytest.approx(0.25)
    assert m["gross_margin"] == pytest.approx(0.6)
    assert m["fcf_margin"] == pytest.approx(0.24)
    assert m["roe"] == pytest.approx(0.4)
    assert m["roa"] == pytest.approx(0.1)
    assert m["debt_to_equity"] == pytest.approx(0.6)
    assert m["revenue_growth"] == pytest.approx(0.25)


def test_roe_and_debt_to_equity_are_none_for_negative_equity():
    m = peers._derive_company_metrics(
        revenue=1000.0, revenue_prior=800.0, net_income=300.0, operating_income=350.0,
        gross_profit=600.0, ocf=340.0, capex=0.0, equity=-100.0, assets=2000.0, liabilities=400.0,
    )
    assert m["roe"] is None
    assert m["debt_to_equity"] is None
    # Everything else is still computed -- one bad denominator must not
    # blank out the whole row.
    assert m["net_margin"] == pytest.approx(0.3)


def test_roe_and_debt_to_equity_are_none_for_zero_equity():
    m = peers._derive_company_metrics(
        revenue=1000.0, revenue_prior=800.0, net_income=100.0, operating_income=150.0,
        gross_profit=600.0, ocf=140.0, capex=0.0, equity=0.0, assets=2000.0, liabilities=300.0,
    )
    assert m["roe"] is None
    assert m["debt_to_equity"] is None


def test_revenue_growth_none_when_prior_missing():
    m = peers._derive_company_metrics(
        revenue=1000.0, revenue_prior=None, net_income=100.0, operating_income=150.0,
        gross_profit=600.0, ocf=140.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0,
    )
    assert m["revenue_growth"] is None


def test_revenue_growth_none_when_prior_non_positive():
    m = peers._derive_company_metrics(
        revenue=1000.0, revenue_prior=0.0, net_income=100.0, operating_income=150.0,
        gross_profit=600.0, ocf=140.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0,
    )
    assert m["revenue_growth"] is None


def test_missing_revenue_blanks_every_margin():
    m = peers._derive_company_metrics(
        revenue=None, revenue_prior=800.0, net_income=100.0, operating_income=150.0,
        gross_profit=600.0, ocf=140.0, capex=0.0, equity=500.0, assets=2000.0, liabilities=300.0,
    )
    assert m["net_margin"] is None
    assert m["operating_margin"] is None
    assert m["gross_margin"] is None
    assert m["fcf_margin"] is None
    # roe/roa/debt_to_equity do not depend on revenue.
    assert m["roe"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _stats -- hand-checked median/p25/p75
# ---------------------------------------------------------------------------


def test_stats_hand_checked_quantiles():
    # Verified independently via statistics.median/statistics.quantiles
    # (n=4, method="inclusive") on this exact list.
    stats = peers._stats([0.30, 0.10, 0.20, 0.25, 0.15])
    assert stats["n"] == 5
    assert stats["values"] == [0.10, 0.15, 0.20, 0.25, 0.30]
    assert stats["median"] == pytest.approx(0.2)
    assert stats["p25"] == pytest.approx(0.15)
    assert stats["p75"] == pytest.approx(0.25)


def test_stats_empty_list_is_none():
    assert peers._stats([]) is None


def test_stats_constant_list_has_equal_median_and_quartiles():
    stats = peers._stats([0.6, 0.6, 0.6, 0.6, 0.6])
    assert stats["median"] == pytest.approx(0.6)
    assert stats["p25"] == pytest.approx(0.6)
    assert stats["p75"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# build_peer_snapshot
# ---------------------------------------------------------------------------


def test_build_peer_snapshot_shape_and_coverage(patched):
    snap = peers.build_peer_snapshot(_YEAR, client=object())

    assert snap["year"] == _YEAR
    assert set(snap["indexes"]) == {"SP500", "NDX"}
    assert snap["universe_size"] == 8  # AAA..GGG (6) + HHH, III (2)
    assert snap["covered"] == 7  # everyone except GGG (no cik) has metrics
    assert snap["skipped"] == [{"ticker": "GGG", "reason": "Kullanılabilir CIK yok"}]
    assert snap["frames_missing"] == []
    assert snap["frames_fetched"] == 10  # 2x Revenue (cur+prior) + 8 other concepts


def test_sector_canonicalization_merges_technology_alias(patched):
    """'Technology' (NDX spelling) and 'Information Technology' (SP500
    spelling) must land in ONE bucket, not two."""
    snap = peers.build_peer_snapshot(_YEAR, client=object())

    assert "Technology" not in snap["sectors"]
    assert "Information Technology" in snap["sectors"]
    assert snap["sectors"]["Information Technology"]["n"] == 5


def test_sector_metrics_hand_checked(patched):
    snap = peers.build_peer_snapshot(_YEAR, client=object())
    it = snap["sectors"]["Information Technology"]["metrics"]

    # net_margin values: 0.10, 0.15, 0.20, 0.25, 0.30 (ciks 1-5).
    assert it["net_margin"]["n"] == 5
    assert it["net_margin"]["median"] == pytest.approx(0.2)
    assert it["net_margin"]["p25"] == pytest.approx(0.15)
    assert it["net_margin"]["p75"] == pytest.approx(0.25)
    assert it["net_margin"]["values"] == pytest.approx([0.10, 0.15, 0.20, 0.25, 0.30])

    # roe only has 4 usable values -- cik 5's negative equity drops it.
    assert it["roe"]["n"] == 4
    assert it["debt_to_equity"]["n"] == 4

    # gross_margin is constant (0.6) across all 5 IT companies.
    assert it["gross_margin"]["median"] == pytest.approx(0.6)
    assert it["gross_margin"]["p25"] == pytest.approx(0.6)
    assert it["gross_margin"]["p75"] == pytest.approx(0.6)


def test_health_care_sector_has_two_companies(patched):
    snap = peers.build_peer_snapshot(_YEAR, client=object())
    hc = snap["sectors"]["Health Care"]
    assert hc["n"] == 2
    assert hc["metrics"]["net_margin"]["n"] == 2


def test_companies_dict_uses_string_cik_keys(patched):
    snap = peers.build_peer_snapshot(_YEAR, client=object())
    assert "1" in snap["companies"]
    assert 1 not in snap["companies"]
    assert snap["companies"]["1"]["ticker"] == "AAA"
    assert snap["companies"]["1"]["sector"] == "Information Technology"
    assert snap["companies"]["1"]["metrics"]["net_margin"] == pytest.approx(0.1)


def test_every_frame_missing_returns_well_formed_snapshot_with_misses_recorded(monkeypatch):
    monkeypatch.setattr(peers, "load_universe", _fake_load_universe)
    monkeypatch.setattr(peers, "get_frame_with_aliases", lambda *a, **k: {})

    snap = peers.build_peer_snapshot(_YEAR, client=object())

    assert snap["universe_size"] == 8
    assert snap["covered"] == 0
    assert snap["frames_fetched"] == 10
    assert len(snap["frames_missing"]) == 10
    assert "Revenue@CY2024" in snap["frames_missing"]
    assert "Revenue@CY2023" in snap["frames_missing"]
    # Companies are still enumerated, just with every metric None.
    assert snap["companies"]["1"]["metrics"]["net_margin"] is None


def test_build_peer_snapshot_never_raises_on_broken_universe(monkeypatch):
    def _boom(index="SP500", path=None):
        raise RuntimeError("csv exploded")

    monkeypatch.setattr(peers, "load_universe", _boom)

    snap = peers.build_peer_snapshot(_YEAR, client=object())

    assert snap["universe_size"] == 0
    assert snap["companies"] == {}


def test_build_peer_snapshot_determinism(patched):
    snap1 = peers.build_peer_snapshot(_YEAR, client=object())
    snap2 = peers.build_peer_snapshot(_YEAR, client=object())

    snap1.pop("generated_at")
    snap2.pop("generated_at")
    assert snap1 == snap2


def test_build_peer_snapshot_default_indexes_is_every_universe(patched, monkeypatch):
    from sec_analyzer.screener.universe import UNIVERSES

    snap = peers.build_peer_snapshot(_YEAR, client=object(), indexes=None)
    assert set(snap["indexes"]) == set(UNIVERSES)


# ---------------------------------------------------------------------------
# rank_against_peers
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot(patched):
    return peers.build_peer_snapshot(_YEAR, client=object())


def test_rank_against_peers_hand_computed_percentile_between_values(snapshot):
    # Peer net_margin = [0.10, 0.15, 0.20, 0.25, 0.30]. Own value 0.28 is
    # strictly greater than 4 of 5 (0.10, 0.15, 0.20, 0.25) and equal to
    # none: percentile = 4/5*100 = 80.0.
    result = peers.rank_against_peers({"net_margin": 0.28}, "Information Technology", snapshot)

    assert result["sector"] == "Information Technology"
    assert result["peer_n"] == 5
    assert result["metrics"]["net_margin"]["value"] == 0.28
    assert result["metrics"]["net_margin"]["percentile"] == pytest.approx(80.0)
    assert result["metrics"]["net_margin"]["median"] == pytest.approx(0.2)
    assert result["metrics"]["net_margin"]["delta_vs_median"] == pytest.approx(0.08)


def test_rank_against_peers_midrank_tie_rule(snapshot):
    # Own value 0.20 TIES with one peer (cik 3) and beats two others (0.10,
    # 0.15): percentile = (2 + 0.5*1) / 5 * 100 = 50.0.
    result = peers.rank_against_peers({"net_margin": 0.20}, "Information Technology", snapshot)

    assert result["metrics"]["net_margin"]["percentile"] == pytest.approx(50.0)


def test_rank_against_peers_none_for_metric_with_n_below_5(snapshot):
    # roe's peer sample is only 4 (cik 5's negative equity dropped it) --
    # below the min-sample-5 floor, so it must be OMITTED from the result,
    # even though the company's own roe value is present.
    result = peers.rank_against_peers({"roe": 0.35, "net_margin": 0.20}, "Information Technology", snapshot)

    assert "roe" not in result["metrics"]
    assert "net_margin" in result["metrics"]


def test_rank_against_peers_none_for_unknown_sector(snapshot):
    assert peers.rank_against_peers({"net_margin": 0.2}, "Utilities", snapshot) is None


def test_rank_against_peers_none_for_missing_sector_arg(snapshot):
    assert peers.rank_against_peers({"net_margin": 0.2}, None, snapshot) is None


def test_rank_against_peers_only_metrics_present_in_both_inputs(snapshot):
    result = peers.rank_against_peers(
        {"net_margin": 0.20, "some_unknown_metric": 42.0}, "Information Technology", snapshot,
    )
    assert set(result["metrics"]) == {"net_margin"}


def test_debt_to_equity_high_percentile_is_a_weakness_not_a_strength():
    # Build a snapshot by hand so debt_to_equity has a clean 0-100 spread.
    snap = {
        "year": _YEAR,
        "sectors": {
            "Widgets": {
                "n": 5,
                "metrics": {
                    "debt_to_equity": {
                        "n": 5, "median": 0.5, "p25": 0.3, "p75": 0.7,
                        "values": [0.1, 0.3, 0.5, 0.7, 0.9],
                    },
                },
            },
        },
    }
    # 0.85 beats 4 of 5 peer values -> percentile 80 (a HIGH leverage
    # percentile) -- must be a WEAKNESS, never a strength.
    result = peers.rank_against_peers({"debt_to_equity": 0.85}, "Widgets", snap)
    assert result["metrics"]["debt_to_equity"]["percentile"] == pytest.approx(80.0)
    assert "debt_to_equity" in result["weaknesses"]
    assert "debt_to_equity" not in result["strengths"]


def test_debt_to_equity_low_percentile_is_a_strength():
    snap = {
        "year": _YEAR,
        "sectors": {
            "Widgets": {
                "n": 5,
                "metrics": {
                    "debt_to_equity": {
                        "n": 5, "median": 0.5, "p25": 0.3, "p75": 0.7,
                        "values": [0.1, 0.3, 0.5, 0.7, 0.9],
                    },
                },
            },
        },
    }
    # 0.15 beats only 1 of 5 -> percentile 20 (LOW leverage percentile) --
    # must be a STRENGTH.
    result = peers.rank_against_peers({"debt_to_equity": 0.15}, "Widgets", snap)
    assert result["metrics"]["debt_to_equity"]["percentile"] == pytest.approx(20.0)
    assert "debt_to_equity" in result["strengths"]
    assert "debt_to_equity" not in result["weaknesses"]


def test_strengths_and_weaknesses_ordering_is_deterministic():
    snap = {
        "year": _YEAR,
        "sectors": {
            "Widgets": {
                "n": 5,
                "metrics": {
                    "fcf_margin": {"n": 5, "median": 0.1, "p25": 0.05, "p75": 0.15,
                                    "values": [0.01, 0.02, 0.03, 0.04, 0.05]},
                    "net_margin": {"n": 5, "median": 0.1, "p25": 0.05, "p75": 0.15,
                                    "values": [0.01, 0.02, 0.03, 0.04, 0.05]},
                    "revenue_growth": {"n": 5, "median": 0.1, "p25": 0.05, "p75": 0.15,
                                        "values": [0.30, 0.40, 0.50, 0.60, 0.70]},
                },
            },
        },
    }
    # fcf_margin (value beats all 5 -> 100) and net_margin (beats 4 of 5 ->
    # 80) are both strengths; revenue_growth (beats none -> 0) is a
    # weakness.
    metrics = {"fcf_margin": 0.10, "net_margin": 0.045, "revenue_growth": 0.10}
    result = peers.rank_against_peers(metrics, "Widgets", snap)

    assert result["strengths"] == ["fcf_margin", "net_margin"]  # sorted desc by percentile
    assert result["weaknesses"] == ["revenue_growth"]


def test_note_neutral_fallback_when_nothing_clears_either_band():
    snap = {
        "year": _YEAR,
        "sectors": {
            "Widgets": {
                "n": 40,
                "metrics": {
                    "net_margin": {"n": 40, "median": 0.1, "p25": 0.05, "p75": 0.15,
                                    "values": [0.01 * i for i in range(1, 41)]},
                },
            },
        },
    }
    # 0.205 sits right at the middle of [0.01..0.40] -> ~50th percentile,
    # inside both bands, so nothing qualifies as a strength or weakness.
    result = peers.rank_against_peers({"net_margin": 0.205}, "Widgets", snap)
    assert result["strengths"] == []
    assert result["weaknesses"] == []
    assert result["note"] == "40 emsale göre belirgin bir ayrışma yok."


def test_note_mentions_sector_and_standout_metrics(snapshot):
    # Peer net_margin = [0.10, 0.15, 0.20, 0.25, 0.30]; own 0.28 beats 4 of
    # 5 -> percentile 80. Peer fcf_margin = [0.14, 0.19, 0.24, 0.29, 0.34];
    # own 0.31 beats 4 of 5 -> percentile 80 too. Both comfortably clear the
    # strength band (>=75); nothing else is passed in, so there is no
    # weakness clause.
    metrics = {"net_margin": 0.28, "fcf_margin": 0.31}
    result = peers.rank_against_peers(metrics, "Information Technology", snapshot)

    assert result["strengths"] == ["fcf_margin", "net_margin"]  # tie at 80.0, alphabetical tiebreak
    assert result["weaknesses"] == []
    assert result["note"] == (
        "Information Technology sektöründeki 5 emsale göre "
        "FCF marjı (80. yüzdelik) ve net marj (80. yüzdelik) güçlü."
    )
    assert "None" not in result["note"]
    assert "nan" not in result["note"].lower()


def test_rank_against_peers_returns_none_on_missing_snapshot():
    assert peers.rank_against_peers({"net_margin": 0.2}, "Widgets", None) is None


def test_rank_against_peers_returns_none_on_missing_metrics():
    assert peers.rank_against_peers(None, "Widgets", {"sectors": {}}) is None


# ---------------------------------------------------------------------------
# rank_against_peers -- fiscal-year mismatch disclosure
# ---------------------------------------------------------------------------


def test_rank_against_peers_no_mismatch_when_years_match(snapshot):
    # snapshot fixture was built for _YEAR (2024); metrics["fy"] == 2024 too.
    metrics = {"net_margin": 0.28, "fy": _YEAR}
    result = peers.rank_against_peers(metrics, "Information Technology", snapshot)

    assert result["year"] == _YEAR
    assert result["company_fy"] == _YEAR
    assert result["fy_mismatch"] is False
    assert "yıllar farklı" not in result["note"]


def test_rank_against_peers_flags_mismatch_when_company_fy_is_newer(snapshot):
    metrics = {"net_margin": 0.28, "fy": 2026}
    result = peers.rank_against_peers(metrics, "Information Technology", snapshot)

    assert result["year"] == _YEAR  # 2024, the peer snapshot's fixed year
    assert result["company_fy"] == 2026
    assert result["fy_mismatch"] is True
    assert result["note"].endswith("(şirket FY2026, emsal kesiti FY2024 — yıllar farklı).")


def test_rank_against_peers_no_mismatch_flag_when_company_fy_unknown(snapshot):
    # metrics carries no "fy" key at all (e.g. a hand-built dict, or the
    # caller never resolved a fiscal year) -- unresolvable is "unknown",
    # not "mismatched", so the flag stays False and no caveat is added.
    metrics = {"net_margin": 0.28}
    result = peers.rank_against_peers(metrics, "Information Technology", snapshot)

    assert result["company_fy"] is None
    assert result["fy_mismatch"] is False
    assert "yıllar farklı" not in result["note"]


def test_rank_against_peers_mismatch_caveat_appended_to_neutral_note():
    snap = {
        "year": 2024,
        "sectors": {
            "Widgets": {
                "n": 40,
                "metrics": {
                    "net_margin": {"n": 40, "median": 0.1, "p25": 0.05, "p75": 0.15,
                                    "values": [0.01 * i for i in range(1, 41)]},
                },
            },
        },
    }
    # Same neutral-band value as the earlier "belirgin bir ayrışma yok" test,
    # but this time the filer's own fy is newer than the snapshot's -- the
    # caveat must still be appended even when neither band fired.
    result = peers.rank_against_peers({"net_margin": 0.205, "fy": 2025}, "Widgets", snap)

    assert result["strengths"] == []
    assert result["weaknesses"] == []
    assert result["fy_mismatch"] is True
    assert result["note"] == (
        "40 emsale göre belirgin bir ayrışma yok. "
        "(şirket FY2025, emsal kesiti FY2024 — yıllar farklı)."
    )


def test_rank_against_peers_fy_metadata_key_is_never_treated_as_a_metric(snapshot):
    # "fy" sitting in the metrics dict must never show up as a ranked
    # metric, strength, or weakness -- only METRIC_KEYS entries are ranked.
    metrics = {"net_margin": 0.28, "fy": 2026}
    result = peers.rank_against_peers(metrics, "Information Technology", snapshot)

    assert "fy" not in result["metrics"]
    assert "fy" not in result["strengths"]
    assert "fy" not in result["weaknesses"]


# ---------------------------------------------------------------------------
# metrics_from_normalized
# ---------------------------------------------------------------------------


def _normalized_with_revenue_and_equity(revenue_by_fy, equity_by_fy):
    return {
        "annual": {
            "Revenue": [{"fy": fy, "value": v} for fy, v in revenue_by_fy.items()],
            "StockholdersEquity": [{"fy": fy, "value": v} for fy, v in equity_by_fy.items()],
        },
    }


def test_metrics_from_normalized_reuses_ratio_row_fields():
    normalized = _normalized_with_revenue_and_equity({2024: 1000.0}, {2024: 500.0})
    ratios = [{
        "fy": 2024, "net_margin": 0.2, "operating_margin": 0.25, "gross_margin": 0.6,
        "fcf_margin": 0.24, "roe": 0.4, "roa": 0.1, "debt_to_equity": 0.6,
        "yoy_revenue_growth": 0.25,
    }]
    metrics = {"latest_fundamental_fy": 2024}

    result = peers.metrics_from_normalized(normalized, ratios, metrics)

    assert result["revenue"] == 1000.0
    assert result["net_margin"] == 0.2
    assert result["roe"] == 0.4
    assert result["debt_to_equity"] == 0.6
    assert result["revenue_growth"] == 0.25
    assert result["fy"] == 2024
    assert "fy" not in peers.METRIC_KEYS  # metadata, not one of the ranked metrics


def test_metrics_from_normalized_masks_roe_for_negative_equity():
    normalized = _normalized_with_revenue_and_equity({2024: 1000.0}, {2024: -100.0})
    ratios = [{"fy": 2024, "net_margin": 0.3, "roe": -3.0, "debt_to_equity": -4.0}]
    metrics = {"latest_fundamental_fy": 2024}

    result = peers.metrics_from_normalized(normalized, ratios, metrics)

    # ratios.compute_ratios would have reported roe=-3.0 / debt_to_equity
    # =-4.0 (it only guards against a ZERO denominator, not a negative
    # one) -- metrics_from_normalized must re-mask both to None so the
    # comparison stays apples-to-apples with the Frames-derived peer side,
    # which never reports a negative-equity ROE at all.
    assert result["roe"] is None
    assert result["debt_to_equity"] is None
    assert result["net_margin"] == 0.3


def test_metrics_from_normalized_falls_back_to_latest_ratio_row_without_metrics_dict():
    normalized = _normalized_with_revenue_and_equity({2023: 900.0, 2024: 1000.0}, {2024: 500.0})
    ratios = [
        {"fy": 2024, "net_margin": 0.2, "roe": 0.4},
        {"fy": 2023, "net_margin": 0.1, "roe": 0.2},
    ]
    result = peers.metrics_from_normalized(normalized, ratios, metrics=None)
    assert result["net_margin"] == 0.2
    assert result["revenue"] == 1000.0
    assert result["fy"] == 2024  # falls back to ratios[0]'s fy (latest, since compute_ratios sorts descending)


def test_metrics_from_normalized_empty_input_returns_all_none_dict():
    # No exception is raised for empty input -- it degrades to a
    # well-formed dict with every METRIC_KEYS entry None (plus the "fy"
    # metadata key, also None), not an empty {} (that fallback is reserved
    # for a genuine internal failure).
    expected = {k: None for k in peers.METRIC_KEYS}
    expected["fy"] = None
    assert peers.metrics_from_normalized(None, None, None) == expected
    assert peers.metrics_from_normalized({}, [], {}) == expected


def test_metrics_from_normalized_never_raises_on_internal_failure(monkeypatch):
    monkeypatch.setattr(peers, "_metrics_from_normalized", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert peers.metrics_from_normalized({"annual": {}}, [], {}) == {}


# ---------------------------------------------------------------------------
# resolve_sector
# ---------------------------------------------------------------------------


def test_resolve_sector_rung1_snapshot_wins_on_disagreement(patched):
    # The fake universe lists AAA (cik 1) under "Information Technology",
    # but the snapshot's OWN row for cik 1 was bucketed under a different
    # sector -- rung 1 must win: a filer can never be ranked against a
    # sector it was not counted in.
    snap = {"companies": {"1": {"ticker": "AAA", "sector": "Snapshot Sector", "metrics": {}}}}

    result = peers.resolve_sector(ticker="AAA", cik=1, sic=None, snapshot=snap)

    assert result == "Snapshot Sector"


def test_resolve_sector_rung1_accepts_a_padded_cik_string(patched):
    snap = {"companies": {"1": {"ticker": "AAA", "sector": "Snapshot Sector", "metrics": {}}}}

    result = peers.resolve_sector(cik="0000000001", snapshot=snap)

    assert result == "Snapshot Sector"


def test_resolve_sector_rung2_bundled_index_lookup_when_no_snapshot(patched):
    # HHH (cik 4) is spelled "Technology" in the fake NDX universe --
    # resolve_sector must return the CANONICALIZED name.
    result = peers.resolve_sector(ticker="HHH", cik=None, sic=None, snapshot=None)

    assert result == "Information Technology"


def test_resolve_sector_falls_to_rung2_when_cik_missing_from_snapshot(patched):
    # A snapshot IS given, but this cik isn't one of its peer rows (e.g. a
    # narrower `indexes` selection than the filer belongs to) -- falls
    # through to the bundled-index lookup by ticker.
    snap = {"companies": {}}

    result = peers.resolve_sector(ticker="AAA", cik=999, sic=None, snapshot=snap)

    assert result == "Information Technology"


def test_resolve_sector_rung3_sic_fallback(patched, monkeypatch):
    # Ticker not in the (fake) bundled universe at all -- falls through to
    # the SIC-derived fallback.
    monkeypatch.setattr(peers, "sector_etf_for_sic", lambda sic: "XLK")

    result = peers.resolve_sector(ticker="NOTINUNIVERSE", cik=None, sic="7372", snapshot=None)

    assert result == "Information Technology"


def test_resolve_sector_rung3_only_used_when_ticker_and_snapshot_absent(monkeypatch):
    # No universe lookup needed at all when only a SIC is supplied.
    monkeypatch.setattr(peers, "sector_etf_for_sic", lambda sic: "XLRE")

    result = peers.resolve_sector(sic="6798")  # a REIT SIC code

    assert result == "Real Estate"


def test_resolve_sector_rung4_none_when_nothing_resolves(patched, monkeypatch):
    monkeypatch.setattr(peers, "sector_etf_for_sic", lambda sic: None)

    result = peers.resolve_sector(ticker="NOTINUNIVERSE", cik=None, sic="9999999", snapshot=None)

    assert result is None


def test_resolve_sector_none_for_completely_empty_call():
    assert peers.resolve_sector() is None


def test_resolve_sector_never_raises_on_malformed_snapshot():
    # snapshot is not even a dict -- resolve_sector must degrade to None,
    # never propagate the AttributeError.
    assert peers.resolve_sector(ticker="AAA", cik=1, snapshot=12345) is None


def test_resolve_sector_sic_fallback_disabled_when_momentum_import_missing(monkeypatch):
    monkeypatch.setattr(peers, "sector_etf_for_sic", None)
    assert peers.resolve_sector(sic="7372") is None


# ---------------------------------------------------------------------------
# Regression guard: no leak into the real cache directory
# ---------------------------------------------------------------------------


def test_build_peer_snapshot_never_touches_raw_dir(patched):
    """This module's build path goes through get_frame_with_aliases, which
    is stubbed out entirely in these tests -- so no cache directory should
    be created at all under the (already tmp_path-redirected) RAW_DIR."""
    import os

    peers.build_peer_snapshot(_YEAR, client=object())

    frames_dir = os.path.join(Config.RAW_DIR, "frames")
    assert not os.path.exists(frames_dir)
