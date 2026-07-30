"""Unit tests for the earnings-catalyst correctness work (SPEC.md Sec.21).

Three things are covered:

* 21a -- the estimate is projected from the 8-K item-2.02 earnings RELEASE
  series, not the 10-Q/10-K paperwork that follows it days later.
* 21b -- a quarter published a day ago is reported as published, and never
  as an upcoming catalyst; the planning signal is gated on proximity.
* 21c -- the SEC JSON caches expire, and a failed re-fetch degrades to the
  stale copy instead of raising.

The SoFi calendar used here is real: earnings 8-Ks on 2025-04-29, 2025-07-29,
2025-10-28, 2026-01-30, 2026-04-29 and 2026-07-29, with the matching 10-Q/10-K
landing about nine days later each time.
"""

import json
import os
import time
from datetime import date

import pytest

from sec_analyzer.config import Config
from sec_analyzer.fetch import companyfacts
from sec_analyzer.fetch.filings import estimate_next_earnings
from sec_analyzer.interpret import planning

# Real SoFi 8-K item-2.02 earnings-release dates.
_SOFI_RELEASES = [
    "2024-07-30", "2024-10-29", "2025-01-27", "2025-04-29",
    "2025-07-29", "2025-10-28", "2026-01-30", "2026-04-29", "2026-07-29",
]
# Real SoFi periodic filings -- each ~9 days after the matching release.
_SOFI_PERIODIC = [
    ("2024-08-06", "10-Q"), ("2024-11-07", "10-Q"), ("2025-02-24", "10-K"),
    ("2025-05-06", "10-Q"), ("2025-08-07", "10-Q"), ("2025-11-06", "10-Q"),
    ("2026-02-17", "10-K"), ("2026-05-07", "10-Q"),
]


def _sofi_submissions(with_items=True):
    """A submissions dict carrying both series, in SEC's parallel-array shape."""
    forms, dates, items = [], [], []
    for d in _SOFI_RELEASES:
        forms.append("8-K")
        dates.append(d)
        items.append("2.02,9.01")
    for d, form in _SOFI_PERIODIC:
        forms.append(form)
        dates.append(d)
        items.append("")
    # A non-earnings 8-K that must never be counted as a release.
    forms.append("8-K")
    dates.append("2026-06-18")
    items.append("5.07")

    recent = {"form": forms, "filingDate": dates}
    if with_items:
        recent["items"] = items
    return {"filings": {"recent": recent}}


# ---------------------------------------------------------------------------
# 21a. The release series drives the estimate
# ---------------------------------------------------------------------------


def test_estimate_uses_the_8k_release_series_not_the_10q_filing():
    result = estimate_next_earnings(_sofi_submissions(), today=date(2026, 7, 30))

    assert result["source"] == "8-K 2.02"
    assert result["last_report_date"] == "2026-07-29"
    # The 10-Q for the same quarter lands ~2026-08-07; that must NOT be the
    # estimate -- the catalyst already happened on 07-29.
    assert result["estimate_date"] == "2026-10-28"
    assert result["days_until"] == 90


def test_a_non_earnings_8k_is_not_counted_as_a_release():
    """The 2026-06-18 8-K carries item 5.07 (shareholder vote), not 2.02."""
    result = estimate_next_earnings(_sofi_submissions(), today=date(2026, 7, 30))

    assert result["last_report_date"] == "2026-07-29"
    assert "18 Haz" not in result["label"]


@pytest.mark.parametrize(
    "today, expected_quarter",
    [
        (date(2026, 2, 20), "Q1"),   # right after the FY 10-K
        (date(2026, 5, 20), "Q2"),   # one release since that 10-K
        (date(2026, 8, 20), "Q3"),   # two releases since
        (date(2025, 11, 20), "FY"),  # three releases since the 2025 10-K
    ],
)
def test_quarter_label_counts_releases_since_the_last_10k(today, expected_quarter):
    result = estimate_next_earnings(_sofi_submissions(), today=today)

    assert result["label"].startswith(expected_quarter)


def test_without_a_10k_the_label_omits_the_quarter():
    subs = _sofi_submissions()
    recent = subs["filings"]["recent"]
    keep = [i for i, f in enumerate(recent["form"]) if f != "10-K"]
    for key in ("form", "filingDate", "items"):
        recent[key] = [recent[key][i] for i in keep]

    result = estimate_next_earnings(subs, today=date(2026, 8, 20))

    assert result["label"].startswith("Sonraki bilanço ~")


def test_falls_back_to_periodic_filings_when_items_are_absent():
    """Older submissions dicts carry no `items` key at all -- the estimate must
    still work, using the pre-Sec.21 periodic-filing cadence."""
    result = estimate_next_earnings(_sofi_submissions(with_items=False),
                                    today=date(2026, 7, 30))

    assert result["source"] == "10-Q/10-K"
    assert result["last_report_date"] == "2026-05-07"
    assert "dosyalamanın" in result["based_on"]


def test_falls_back_when_too_few_releases_exist():
    subs = _sofi_submissions()
    recent = subs["filings"]["recent"]
    # Keep only two earnings releases -- below the 3-filing minimum.
    keep = [i for i, (f, it) in enumerate(zip(recent["form"], recent["items"]))
            if not (f == "8-K" and "2.02" in it)][:]
    releases = [i for i, (f, it) in enumerate(zip(recent["form"], recent["items"]))
                if f == "8-K" and "2.02" in it][:2]
    keep = sorted(keep + releases)
    for key in ("form", "filingDate", "items"):
        recent[key] = [recent[key][i] for i in keep]

    result = estimate_next_earnings(subs, today=date(2026, 7, 30))

    assert result["source"] == "10-Q/10-K"


def test_release_after_today_is_ignored_point_in_time():
    """A release dated after the as-of cutoff was not public yet."""
    before = estimate_next_earnings(_sofi_submissions(), today=date(2026, 7, 1))

    assert before["last_report_date"] == "2026-04-29"
    assert before["estimate_date"] == "2026-07-29"


# ---------------------------------------------------------------------------
# 21b. Already-reported is not an upcoming catalyst
# ---------------------------------------------------------------------------


def test_just_reported_quarter_is_labeled_as_reported():
    result = estimate_next_earnings(_sofi_submissions(), today=date(2026, 7, 30))

    assert result["recently_reported"] is True
    assert result["label"].startswith("29 Tem tarihinde açıklandı")
    assert "sonraki: Q3 earnings ~28 Eki" in result["label"]


def test_recently_reported_window_closes_after_three_days():
    on_the_edge = estimate_next_earnings(_sofi_submissions(), today=date(2026, 8, 1))
    past_it = estimate_next_earnings(_sofi_submissions(), today=date(2026, 8, 2))

    assert on_the_edge["recently_reported"] is True
    assert past_it["recently_reported"] is False
    assert past_it["label"] == "Q3 earnings ~28 Eki"


def test_estimate_date_is_far_enough_out_to_silence_the_report_badge():
    """The HTML report shows its earnings badge only within 21 days. With the
    correct next-release date the badge self-suppresses -- the old estimate
    (~2026-08-07, 8 days out) wrongly triggered it."""
    result = estimate_next_earnings(_sofi_submissions(), today=date(2026, 7, 30))

    assert result["days_until"] > 21


def test_days_until_is_measured_against_the_supplied_reference_date():
    result = estimate_next_earnings(_sofi_submissions(), today=date(2026, 10, 1))

    assert result["estimate_date"] == "2026-10-28"
    assert result["days_until"] == 27


# ---- the planning proximity gate ----


def _stop_adding(catalyst):
    return planning._compute_stop_adding({}, None, None, [], catalyst)


def _codes(signals):
    return [s["code"] for s in signals]


def test_planning_signal_fires_for_a_near_catalyst():
    catalyst = {"label": "Q3 earnings ~28 Eki", "days_until": 10,
                "recently_reported": False}

    assert "BINARY_CATALYST_NEAR" in _codes(_stop_adding(catalyst))


def test_planning_signal_is_silent_for_a_just_reported_quarter():
    catalyst = {"label": "29 Tem tarihinde açıklandı · sonraki: Q3 earnings ~28 Eki",
                "days_until": 90, "recently_reported": True}

    assert "BINARY_CATALYST_NEAR" not in _codes(_stop_adding(catalyst))


def test_planning_signal_is_silent_for_a_distant_catalyst():
    catalyst = {"label": "Q3 earnings ~28 Eki", "days_until": 90,
                "recently_reported": False}

    assert "BINARY_CATALYST_NEAR" not in _codes(_stop_adding(catalyst))


def test_planning_signal_keeps_old_behavior_without_days_until():
    """A hand-built or pre-Sec.21 catalyst dict must not silently lose the
    signal just because it lacks the new key."""
    catalyst = {"label": "Q3 earnings ~28 Eki"}

    assert "BINARY_CATALYST_NEAR" in _codes(_stop_adding(catalyst))


# ---------------------------------------------------------------------------
# 21c. Cache freshness
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self, payload=None, fail=False):
        self.payload = payload if payload is not None else {"fetched": True}
        self.fail = fail
        self.calls = 0

    def get_json(self, url):
        self.calls += 1
        if self.fail:
            raise RuntimeError("network down")
        return self.payload


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "ensure_dirs", classmethod(lambda cls: None))
    return tmp_path


def _seed(raw_dir, name, payload, age_hours=0.0):
    path = raw_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))
    return path


def test_fresh_cache_is_served_without_a_fetch(raw_dir, monkeypatch):
    monkeypatch.setattr(Config, "SUBMISSIONS_CACHE_TTL_HOURS", 24.0)
    _seed(raw_dir, "submissions_CIK0000001.json", {"cached": True}, age_hours=1)
    client = _StubClient()

    result = companyfacts.get_submissions("0000001", client)

    assert result == {"cached": True}
    assert client.calls == 0


def test_stale_cache_triggers_a_refetch(raw_dir, monkeypatch):
    monkeypatch.setattr(Config, "SUBMISSIONS_CACHE_TTL_HOURS", 24.0)
    path = _seed(raw_dir, "submissions_CIK0000001.json", {"cached": True}, age_hours=48)
    client = _StubClient({"fresh": True})

    result = companyfacts.get_submissions("0000001", client)

    assert result == {"fresh": True}
    assert client.calls == 1
    # The refreshed document replaced the stale one on disk.
    assert json.loads(path.read_text(encoding="utf-8")) == {"fresh": True}


def test_stale_cache_is_used_when_the_refetch_fails(raw_dir, monkeypatch):
    """A month-old document still supports an analysis; a raised exception
    would kill the whole run."""
    monkeypatch.setattr(Config, "SUBMISSIONS_CACHE_TTL_HOURS", 24.0)
    _seed(raw_dir, "submissions_CIK0000001.json", {"cached": True}, age_hours=720)
    client = _StubClient(fail=True)

    result = companyfacts.get_submissions("0000001", client)

    assert result == {"cached": True}
    assert client.calls == 1


def test_a_failed_fetch_with_no_cache_still_raises(raw_dir, monkeypatch):
    monkeypatch.setattr(Config, "SUBMISSIONS_CACHE_TTL_HOURS", 24.0)
    client = _StubClient(fail=True)

    with pytest.raises(RuntimeError):
        companyfacts.get_submissions("0000001", client)


def test_zero_ttl_restores_never_expires_behavior(raw_dir, monkeypatch):
    monkeypatch.setattr(Config, "SUBMISSIONS_CACHE_TTL_HOURS", 0.0)
    _seed(raw_dir, "submissions_CIK0000001.json", {"cached": True}, age_hours=10_000)
    client = _StubClient()

    result = companyfacts.get_submissions("0000001", client)

    assert result == {"cached": True}
    assert client.calls == 0


def test_no_cache_flag_bypasses_a_fresh_cache(raw_dir, monkeypatch):
    monkeypatch.setattr(Config, "SUBMISSIONS_CACHE_TTL_HOURS", 24.0)
    _seed(raw_dir, "submissions_CIK0000001.json", {"cached": True}, age_hours=1)
    client = _StubClient({"fresh": True})

    result = companyfacts.get_submissions("0000001", client, no_cache=True)

    assert result == {"fresh": True}
    assert client.calls == 1


def test_companyfacts_uses_its_own_longer_ttl(raw_dir, monkeypatch):
    monkeypatch.setattr(Config, "COMPANYFACTS_CACHE_TTL_HOURS", 168.0)
    _seed(raw_dir, "CIK0000001.json", {"cached": True}, age_hours=48)
    client = _StubClient({"fresh": True})

    # 48h old is stale for submissions (24h) but fresh for companyfacts (168h).
    result = companyfacts.get_company_facts("0000001", client)

    assert result == {"cached": True}
    assert client.calls == 0
