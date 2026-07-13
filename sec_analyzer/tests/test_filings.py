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
