"""Unit tests for :mod:`sec_analyzer.fetch.frames` (SEC XBRL Frames API).

No real network access is used anywhere in this module: a small fake HTTP
client stands in for ``SecHttpClient`` (mirroring ``test_earnings_catalyst.py``'s
``_StubClient``), and ``Config.RAW_DIR`` is redirected to pytest's ``tmp_path``
via an AUTOUSE fixture -- every test in this file gets isolation whether or
not it remembers to ask for it. That matters here specifically: a sibling
concurrent-agent's test once wrote a fixture straight into the real,
multi-gigabyte ``sec_analyzer/raw`` cache because its redirect was opt-in
rather than automatic. See ``test_no_cache_write_leaks_into_the_real_raw_dir``
below for the regression guard.
"""

import json
import os
import time

import pytest
import requests

from sec_analyzer.config import Config
from sec_analyzer.fetch import frames

_REAL_RAW_DIR = Config.RAW_DIR


@pytest.fixture(autouse=True)
def isolate_raw_dir(tmp_path, monkeypatch):
    """Redirect every cache read/write in this file to ``tmp_path``.

    Autouse, not opt-in: a test that forgets to request isolation must not
    be able to touch the real cache.
    """
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "ensure_dirs", classmethod(lambda cls: None))
    return tmp_path


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


class _StubClient:
    """Fake ``SecHttpClient``. ``responses`` is a list consumed in order (one
    entry per call to ``get_json``); each entry is either a payload dict or
    an exception instance/class to raise."""

    def __init__(self, responses=None):
        self.responses = list(responses) if responses is not None else []
        self.calls = 0
        self.urls = []

    def get_json(self, url):
        self.calls += 1
        self.urls.append(url)
        if not self.responses:
            raise AssertionError("_StubClient.get_json called more times than responses were queued")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, type) and issubclass(response, Exception):
            raise response()
        return response


def _http_404():
    return requests.HTTPError("404 error", response=_Resp(404))


def _http_500():
    return requests.HTTPError("500 error", response=_Resp(500))


def _frame_payload(entries, pts=None):
    """Build a well-formed SEC Frames response from ``[(cik, val), ...]``."""
    data = [
        {
            "accn": "0000000000-00-000000",
            "cik": cik,
            "entityName": f"Company {cik}",
            "loc": "US-CA",
            "start": "2024-01-01",
            "end": "2024-12-31",
            "val": val,
        }
        for cik, val in entries
    ]
    return {
        "taxonomy": "us-gaap",
        "tag": "Revenues",
        "ccp": "CY2024",
        "uom": "USD",
        "pts": pts if pts is not None else len(entries),
        "data": data,
    }


def _seed_cache(path, payload, age_hours=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


def test_annual_period_format():
    assert frames.annual_period(2024) == "CY2024"


def test_instant_period_format_defaults_to_q4():
    assert frames.instant_period(2024) == "CY2024Q4I"


def test_instant_period_format_explicit_quarter():
    assert frames.instant_period(2024, quarter=2) == "CY2024Q2I"


# ---------------------------------------------------------------------------
# get_frame -- basic shape, reduction, dedup
# ---------------------------------------------------------------------------


def test_well_formed_response_reduces_to_cik_value_map():
    payload = _frame_payload([(320193, 391035000000.0), (789019, 245122000000.0)], pts=2)
    client = _StubClient([payload])

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result is not None
    assert result["tag"] == "Revenues"
    assert result["period"] == "CY2024"
    assert result["unit"] == "USD"
    assert result["taxonomy"] == "us-gaap"
    assert result["pts"] == 2
    assert result["values"] == {320193: 391035000000.0, 789019: 245122000000.0}
    for cik in result["values"]:
        assert isinstance(cik, int)


def test_duplicate_cik_keeps_first_occurrence_deterministically():
    payload = _frame_payload([(320193, 100.0), (320193, 999.0), (789019, 50.0)])
    client = _StubClient([payload])

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result["values"][320193] == 100.0
    assert result["values"][789019] == 50.0


# ---------------------------------------------------------------------------
# get_frame -- error handling
# ---------------------------------------------------------------------------


def test_404_returns_none_without_raising():
    client = _StubClient([_http_404()])

    result = frames.get_frame("NonexistentTag", "CY2024", client)

    assert result is None
    assert client.calls == 1  # no retry storm


def test_other_exception_returns_none_without_raising():
    client = _StubClient([RuntimeError("network exploded")])

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result is None


def test_500_http_error_returns_none_without_raising():
    client = _StubClient([_http_500()])

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result is None


# ---------------------------------------------------------------------------
# get_frame -- caching
# ---------------------------------------------------------------------------


def test_second_call_hits_cache_no_network(tmp_path):
    payload = _frame_payload([(1, 10.0)])
    client = _StubClient([payload])

    first = frames.get_frame("Revenues", "CY2024", client)
    second = frames.get_frame("Revenues", "CY2024", client)

    assert first == second
    assert client.calls == 1


def test_no_cache_flag_forces_a_refetch(tmp_path):
    payload1 = _frame_payload([(1, 10.0)])
    payload2 = _frame_payload([(1, 20.0)])
    client = _StubClient([payload1, payload2])

    first = frames.get_frame("Revenues", "CY2024", client)
    second = frames.get_frame("Revenues", "CY2024", client, no_cache=True)

    assert first["values"][1] == 10.0
    assert second["values"][1] == 20.0
    assert client.calls == 2


def test_corrupt_cache_falls_through_to_a_refetch(tmp_path):
    cache_path = tmp_path / "frames" / "us-gaap_Revenues_USD_CY2024.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not valid json", encoding="utf-8")

    payload = _frame_payload([(1, 42.0)])
    client = _StubClient([payload])

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result["values"][1] == 42.0
    assert client.calls == 1


def test_stale_cache_triggers_a_refetch(tmp_path):
    cache_path = tmp_path / "frames" / "us-gaap_Revenues_USD_CY2024.json"
    _seed_cache(cache_path, {"pts": 1, "values": {"1": 1.0}}, age_hours=frames.FRAMES_CACHE_TTL_HOURS + 10)

    payload = _frame_payload([(1, 99.0)])
    client = _StubClient([payload])

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result["values"][1] == 99.0
    assert client.calls == 1


def test_stale_cache_is_used_when_the_refetch_fails(tmp_path):
    cache_path = tmp_path / "frames" / "us-gaap_Revenues_USD_CY2024.json"
    _seed_cache(cache_path, {"pts": 1, "values": {"1": 7.0}}, age_hours=frames.FRAMES_CACHE_TTL_HOURS + 10)

    client = _StubClient([RuntimeError("network down")])

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result["values"][1] == 7.0
    assert client.calls == 1


def test_fresh_cache_is_served_without_a_network_call(tmp_path):
    cache_path = tmp_path / "frames" / "us-gaap_Revenues_USD_CY2024.json"
    _seed_cache(cache_path, {"pts": 1, "values": {"1": 5.0}}, age_hours=1)

    client = _StubClient([])  # no responses queued -- a call would raise AssertionError

    result = frames.get_frame("Revenues", "CY2024", client)

    assert result["values"][1] == 5.0
    assert client.calls == 0


# ---------------------------------------------------------------------------
# get_frame_with_aliases
# ---------------------------------------------------------------------------


def test_alias_merge_earliest_tag_wins_for_a_shared_cik():
    # RevenueFromContractWithCustomerExcludingAssessedTax reports CIK 1 as
    # 100; Revenues (later alias) reports the SAME cik as 999 and a NEW
    # cik (2) as 50. The earlier alias must win for CIK 1; CIK 2 (only in
    # the later alias) must still come through.
    payload_a = _frame_payload([(1, 100.0)])
    payload_b = _frame_payload([(1, 999.0), (2, 50.0)])
    client = _StubClient([payload_a, payload_b])

    result = frames.get_frame_with_aliases(
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"], "CY2024", client,
    )

    assert result == {1: 100.0, 2: 50.0}


def test_alias_merge_skips_a_404_alias_and_returns_the_others():
    payload = _frame_payload([(1, 10.0)])
    client = _StubClient([_http_404(), payload])

    result = frames.get_frame_with_aliases(["MissingTag", "Revenues"], "CY2024", client)

    assert result == {1: 10.0}


def test_alias_merge_returns_empty_dict_when_every_alias_misses():
    client = _StubClient([_http_404(), _http_404()])

    result = frames.get_frame_with_aliases(["MissingTagA", "MissingTagB"], "CY2024", client)

    assert result == {}


def test_alias_merge_empty_tag_list_returns_empty_dict():
    client = _StubClient([])

    result = frames.get_frame_with_aliases([], "CY2024", client)

    assert result == {}
    assert client.calls == 0


def test_get_frame_with_aliases_never_raises_even_on_unexpected_internal_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(frames, "get_frame", _boom)

    result = frames.get_frame_with_aliases(["Revenues"], "CY2024", _StubClient([]))

    assert result == {}


# ---------------------------------------------------------------------------
# Regression guard: no leak into the real cache directory
# ---------------------------------------------------------------------------


def test_no_cache_write_leaks_into_the_real_raw_dir():
    """Exercise the cache-writing path, then assert the REAL (non-tmp_path)
    ``Config.RAW_DIR/frames`` was never touched by this test run.

    ``isolate_raw_dir`` redirects ``Config.RAW_DIR`` for the duration of
    this test, but the whole point of this guard is to check the ORIGINAL
    path recorded at import time (``_REAL_RAW_DIR``), independent of
    whatever the fixture patched it to.
    """
    payload = _frame_payload([(1, 1.0)])
    client = _StubClient([payload])

    result = frames.get_frame("RegressionGuardTag", "CY2024", client)
    assert result is not None  # sanity: the write path really did run

    real_frames_dir = os.path.join(_REAL_RAW_DIR, "frames")
    leaked_file = os.path.join(real_frames_dir, "us-gaap_RegressionGuardTag_USD_CY2024.json")
    assert not os.path.exists(leaked_file)
