"""Unit tests for sec_analyzer.fetch.filings.

No network access -- ``estimate_next_earnings`` only needs the
``filings.recent.{form,filingDate}`` parallel-array shape of a real SEC
submissions document, which these fixtures build synthetically.
"""

from datetime import date, timedelta

from sec_analyzer.fetch.filings import estimate_next_earnings

_TURKISH_MONTH_ABBREVS = (
    "Oca", "Şub", "Mar", "Nis", "May", "Haz",
    "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara",
)


def _synthetic_submissions(anchor, forms, gap_days=91):
    """Build a submissions dict with ``len(forms)`` filings spaced
    ``gap_days`` apart, ending at ``anchor`` (the most recent filing date).
    """
    n = len(forms)
    dates = [anchor - timedelta(days=gap_days * (n - 1 - i)) for i in range(n)]
    return {
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": [d.strftime("%Y-%m-%d") for d in dates],
            }
        }
    }


def test_estimate_next_earnings_projects_median_gap_forward():
    # Last filing 5 days ago, so the projected next date (91 days later) is
    # comfortably in the future -- no roll-forward needed, isolating the
    # "project the median gap" behavior from the "roll forward to >= today"
    # behavior tested separately below.
    anchor = date.today() - timedelta(days=5)
    forms = ["10-K", "10-Q", "10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-Q", "10-K"]
    submissions = _synthetic_submissions(anchor, forms)

    result = estimate_next_earnings(submissions)

    assert result is not None
    expected_next = anchor + timedelta(days=91)
    assert result["estimate_date"] == expected_next.strftime("%Y-%m-%d")
    assert any(month in result["label"] for month in _TURKISH_MONTH_ABBREVS)
    assert "9" in result["based_on"]
    assert "91" in result["based_on"]


def test_estimate_next_earnings_labels_quarter_after_a_10k():
    """The most recent filing is a 10-K, so the next filing should be
    labeled Q1 (the first quarterly filing after an annual one)."""
    anchor = date.today() - timedelta(days=5)
    forms = ["10-Q", "10-Q", "10-Q", "10-K"]
    submissions = _synthetic_submissions(anchor, forms)

    result = estimate_next_earnings(submissions)

    assert result is not None
    assert result["label"].startswith("Q1")


def test_estimate_next_earnings_labels_quarter_after_one_10q():
    """One 10-Q has been filed since the last 10-K, so the next filing
    should be labeled Q2."""
    anchor = date.today() - timedelta(days=5)
    forms = ["10-Q", "10-Q", "10-Q", "10-K", "10-Q"]
    submissions = _synthetic_submissions(anchor, forms)

    result = estimate_next_earnings(submissions)

    assert result is not None
    assert result["label"].startswith("Q2")


def test_estimate_next_earnings_rolls_forward_past_stale_history():
    """If the projected date (from an old filing history) has already
    passed, the estimate keeps adding the median gap until it's >= today."""
    anchor = date(2015, 1, 15)  # long in the past relative to "today"
    forms = ["10-Q", "10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-Q", "10-K", "10-Q"]
    submissions = _synthetic_submissions(anchor, forms)

    result = estimate_next_earnings(submissions)

    assert result is not None
    estimated = date.fromisoformat(result["estimate_date"])
    assert estimated >= date.today()


def test_ignores_non_earnings_forms():
    anchor = date.today() - timedelta(days=5)
    submissions = _synthetic_submissions(anchor, ["10-Q", "10-Q", "10-Q"])
    # Sprinkle in some 8-Ks between the quarterly filings' dates.
    submissions["filings"]["recent"]["form"].append("8-K")
    submissions["filings"]["recent"]["filingDate"].append(
        (anchor - timedelta(days=30)).strftime("%Y-%m-%d")
    )

    result = estimate_next_earnings(submissions)
    assert result is not None  # the 8-K must not have broken the estimate


def test_returns_none_with_fewer_than_three_usable_filings():
    anchor = date.today()
    submissions = _synthetic_submissions(anchor, ["10-Q", "10-Q"])
    assert estimate_next_earnings(submissions) is None


def test_returns_none_for_empty_or_malformed_submissions():
    assert estimate_next_earnings({}) is None
    assert estimate_next_earnings(None) is None
    assert estimate_next_earnings({"filings": {}}) is None
    assert estimate_next_earnings({"filings": {"recent": {}}}) is None


# ---------------------------------------------------------------------------
# Point-in-time ("as-of") mode: `today` is both the reference date the
# projection walks forward from AND the point-in-time cutoff -- a filing
# dated after `today` was "not yet public" and must be ignored, not just
# used as a default for `date.today()`.
# ---------------------------------------------------------------------------


def test_estimate_next_earnings_uses_explicit_today_deterministically():
    """A fixed `today` (rather than the real wall clock) gives a fully
    deterministic estimate: last filing 2022-04-01, gap 91 days -> projected
    next date 2022-07-01, which is already >= today (2022-06-30), so no
    roll-forward is needed."""
    as_of = date(2022, 6, 30)
    anchor = date(2022, 4, 1)
    forms = ["10-K", "10-Q", "10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-Q", "10-K"]
    submissions = _synthetic_submissions(anchor, forms)

    result = estimate_next_earnings(submissions, today=as_of)

    assert result is not None
    assert result["estimate_date"] == "2022-07-01"


def test_estimate_next_earnings_ignores_a_filing_dated_after_today():
    """A filing dated after `today` must not affect the median-gap cadence
    projection at all -- it's excluded up front by the point-in-time guard,
    exactly as if it had never been filed. Appending a stray 10-Q AFTER the
    as-of cutoff must therefore leave the estimate byte-for-byte identical
    to the same history without that extra filing."""
    as_of = date(2022, 6, 30)
    anchor = date(2022, 4, 1)
    forms = ["10-K", "10-Q", "10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-Q", "10-K"]
    submissions = _synthetic_submissions(anchor, forms)

    baseline = estimate_next_earnings(submissions, today=as_of)

    # A future (post-as-of) filing appended to the same history.
    submissions_with_future = _synthetic_submissions(anchor, forms)
    submissions_with_future["filings"]["recent"]["form"].append("10-Q")
    submissions_with_future["filings"]["recent"]["filingDate"].append("2022-08-01")

    with_future = estimate_next_earnings(submissions_with_future, today=as_of)

    assert with_future == baseline


def test_estimate_next_earnings_rolls_forward_from_explicit_today_not_wall_clock():
    """The roll-forward loop (`while next_date < today`) must use the
    EXPLICIT `today` passed in, not `date.today()` -- otherwise an as-of run
    for a date far in the past would rocket the projection all the way up
    to the real current date instead of stopping at the as-of cutoff."""
    as_of = date(2015, 6, 1)
    anchor = date(2015, 1, 15)
    forms = ["10-Q", "10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-Q", "10-K", "10-Q"]
    submissions = _synthetic_submissions(anchor, forms)

    result = estimate_next_earnings(submissions, today=as_of)

    assert result is not None
    estimated = date.fromisoformat(result["estimate_date"])
    # Must have rolled forward to at least `as_of`, but must NOT have
    # rocketed all the way to the real wall-clock date (which is far later
    # than 2015 in this test suite's lifetime).
    assert estimated >= as_of
    assert estimated < date(2016, 1, 1)
