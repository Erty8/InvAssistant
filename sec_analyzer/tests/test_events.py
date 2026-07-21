"""Unit tests for sec_analyzer.signals.events.

No network access -- ``detect_events`` only needs the
``filings.recent.{form,filingDate,items,accessionNumber,primaryDocument}``
parallel-array shape of a real SEC submissions document, which these
fixtures build synthetically. A fixed ``today`` is passed everywhere so the
lookback window is deterministic and independent of the wall clock.
"""

from datetime import date

from sec_analyzer.signals.events import detect_events, summarize_events

_TODAY = date(2026, 7, 16)


def _submissions(rows):
    """Build a submissions dict from ``rows`` of
    ``(form, filingDate, items, accession, primaryDoc)`` tuples, in the
    newest-first order SEC uses.
    """
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "items": [r[2] for r in rows],
                "accessionNumber": [r[3] for r in rows],
                "primaryDocument": [r[4] for r in rows],
            }
        }
    }


def _codes(events):
    return [e["items"] for e in events]


def test_classifies_officer_departure_as_warning():
    subs = _submissions(
        [("8-K", "2026-07-01", "5.02", "acc-1", "d1.htm")]
    )
    events = detect_events(subs, today=_TODAY)

    assert len(events) == 1
    ev = events[0]
    assert ev["items"] == ["5.02"]
    assert ev["severity"] == "warning"
    assert ev["categories"] == ["Üst düzey yönetici/kurul değişikliği"]
    assert ev["accession"] == "acc-1"
    assert ev["primary_doc"] == "d1.htm"


def test_restatement_is_critical():
    subs = _submissions([("8-K", "2026-06-15", "4.02", "acc-2", "d2.htm")])
    events = detect_events(subs, today=_TODAY)
    assert events[0]["severity"] == "critical"


def test_event_severity_is_max_over_its_items():
    # A single 8-K carrying both a routine vote (5.07, info) and a
    # restatement (4.02, critical) must be classified critical.
    subs = _submissions([("8-K", "2026-06-01", "5.07,4.02,9.01", "acc-3", "d3.htm")])
    events = detect_events(subs, today=_TODAY)
    assert len(events) == 1
    assert events[0]["severity"] == "critical"
    assert events[0]["items"] == ["5.07", "4.02", "9.01"]
    # Category order follows item order and de-duplicates.
    assert events[0]["categories"][0] == "Genel kurul oylama sonuçları"


def test_non_8k_forms_are_ignored():
    subs = _submissions(
        [
            ("10-Q", "2026-07-05", "", "acc-q", "q.htm"),
            ("4", "2026-07-04", "", "acc-4", "f4.htm"),
            ("8-K", "2026-07-01", "2.02", "acc-8k", "d.htm"),
        ]
    )
    events = detect_events(subs, today=_TODAY)
    assert len(events) == 1
    assert events[0]["form"] == "8-K"


def test_8k_amendment_is_included():
    subs = _submissions([("8-K/A", "2026-07-02", "5.02", "acc-a", "a.htm")])
    events = detect_events(subs, today=_TODAY)
    assert len(events) == 1
    assert events[0]["form"] == "8-K/A"


def test_8k_without_item_codes_is_skipped():
    subs = _submissions([("8-K", "2026-07-01", "", "acc-x", "x.htm")])
    assert detect_events(subs, today=_TODAY) == []


def test_lookback_window_excludes_old_filings():
    subs = _submissions(
        [
            ("8-K", "2026-07-10", "5.02", "recent", "r.htm"),
            ("8-K", "2025-01-01", "5.02", "old", "o.htm"),  # >180 days back
        ]
    )
    events = detect_events(subs, lookback_days=180, today=_TODAY)
    assert [e["accession"] for e in events] == ["recent"]


def test_lookback_disabled_keeps_everything():
    subs = _submissions(
        [
            ("8-K", "2026-07-10", "5.02", "recent", "r.htm"),
            ("8-K", "2020-01-01", "5.02", "ancient", "a.htm"),
        ]
    )
    events = detect_events(subs, lookback_days=0, today=_TODAY)
    assert len(events) == 2


def test_min_severity_filters_out_info():
    subs = _submissions(
        [
            ("8-K", "2026-07-10", "2.02", "earnings", "e.htm"),  # info
            ("8-K", "2026-07-05", "5.02", "dep", "d.htm"),  # warning
            ("8-K", "2026-07-01", "4.02", "restate", "r.htm"),  # critical
        ]
    )
    warning_plus = detect_events(subs, min_severity="warning", today=_TODAY)
    assert {e["accession"] for e in warning_plus} == {"dep", "restate"}

    critical_only = detect_events(subs, min_severity="critical", today=_TODAY)
    assert {e["accession"] for e in critical_only} == {"restate"}


def test_results_are_sorted_most_recent_first():
    # Deliberately supply out-of-order dates to prove the explicit sort.
    subs = _submissions(
        [
            ("8-K", "2026-05-01", "5.02", "may", "m.htm"),
            ("8-K", "2026-07-01", "5.02", "jul", "j.htm"),
            ("8-K", "2026-06-01", "5.02", "jun", "n.htm"),
        ]
    )
    events = detect_events(subs, today=_TODAY)
    assert [e["accession"] for e in events] == ["jul", "jun", "may"]


def test_max_events_caps_after_sorting():
    subs = _submissions(
        [
            ("8-K", "2026-07-10", "5.02", "a", "a.htm"),
            ("8-K", "2026-07-05", "5.02", "b", "b.htm"),
            ("8-K", "2026-07-01", "5.02", "c", "c.htm"),
        ]
    )
    events = detect_events(subs, max_events=2, today=_TODAY)
    assert [e["accession"] for e in events] == ["a", "b"]


def test_unknown_item_code_is_surfaced_as_info():
    subs = _submissions([("8-K", "2026-07-01", "9.99", "acc", "d.htm")])
    events = detect_events(subs, today=_TODAY)
    assert len(events) == 1
    assert events[0]["severity"] == "info"
    assert "9.99" in events[0]["categories"][0]


def test_malformed_submissions_never_raise():
    assert detect_events(None) == []
    assert detect_events({}) == []
    assert detect_events({"filings": None}) == []
    assert detect_events({"filings": {"recent": None}}) == []
    # Ragged parallel arrays (items shorter than form) must not raise.
    ragged = {"filings": {"recent": {"form": ["8-K", "8-K"], "filingDate": ["2026-07-01"], "items": []}}}
    assert detect_events(ragged, today=_TODAY) == []


def test_bad_date_string_is_skipped():
    subs = _submissions([("8-K", "not-a-date", "5.02", "acc", "d.htm")])
    assert detect_events(subs, today=_TODAY) == []


# ---------------------------------------------------------------------------
# Point-in-time ("as-of") mode: `today` is the reference date the whole
# lookback window and "was this filing public yet" guard are computed
# against -- a filing dated after `today` must be excluded even though it
# would otherwise qualify (e.g. inside the lookback window relative to the
# real wall-clock date).
# ---------------------------------------------------------------------------


def test_filing_dated_after_today_is_excluded_even_if_within_lookback():
    as_of = date(2022, 6, 30)
    subs = _submissions(
        [
            ("8-K", "2022-06-15", "5.02", "before", "b.htm"),   # on/before as_of -> kept
            ("8-K", "2022-07-01", "4.02", "after", "a.htm"),    # after as_of -> point-in-time guard
        ]
    )
    events = detect_events(subs, today=as_of)
    assert [e["accession"] for e in events] == ["before"]


def test_filing_dated_exactly_on_today_is_included():
    as_of = date(2022, 6, 30)
    subs = _submissions([("8-K", "2022-06-30", "5.02", "same-day", "s.htm")])
    events = detect_events(subs, today=as_of)
    assert [e["accession"] for e in events] == ["same-day"]


def test_summarize_events_empty():
    assert summarize_events([]) == "yok"


def test_summarize_events_tally_and_order():
    events = detect_events(
        _submissions(
            [
                ("8-K", "2026-07-10", "2.02", "info1", "i.htm"),   # info
                ("8-K", "2026-07-08", "5.02", "warn1", "w.htm"),   # warning
                ("8-K", "2026-07-05", "4.02", "crit1", "c.htm"),   # critical
            ]
        ),
        today=_TODAY,
    )
    summary = summarize_events(events)
    assert summary.startswith("1 kritik, 1 uyarı, 1 bilgi")
    # Most-severe-first: the critical restatement leads the listed events.
    listed = summary.split("—", 1)[1]
    assert listed.index("güvenilemez") < listed.index("Kazanç")
