"""Estimate a filer's next earnings-related filing date from its history.

SEC's ``submissions`` endpoint (see
``sec_analyzer.fetch.companyfacts.get_submissions``) lists every filing a
company has made, including the quarterly cadence of its 10-Q/10-K filings.
There is no official "next earnings date" field anywhere in SEC data -- but
that cadence is remarkably regular for most filers (roughly one filing every
quarter), so a simple median-gap projection from the filing history gives a
reasonable best-effort estimate without hitting any third-party calendar
API.

This is intentionally a rough heuristic, not a scheduling guarantee: actual
earnings releases (as opposed to the SEC filing itself) often precede the
10-Q/10-K filing by a few days to a couple of weeks, and a company can shift
its fiscal calendar. The estimate should be read as "around this date," not
"on this date."
"""

import logging
import statistics
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Date format used throughout SEC submissions data for filing dates.
_DATE_FMT = "%Y-%m-%d"

#: English 3-letter month abbreviations, indexed 0 (January) to 11 (December).
_MONTH_ABBREVIATIONS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

#: Minimum number of usable (form, filingDate) pairs required before an
#: estimate is attempted -- with fewer than this, a median gap isn't a
#: meaningful cadence.
_MIN_USABLE_FILINGS = 3

#: Only the most recent this-many quarterly filings are used to compute the
#: median gap, so a stale/older cadence (e.g. before a fiscal-year-end
#: change) doesn't skew the estimate.
_MAX_FILINGS_CONSIDERED = 9

#: Forms that mark a quarterly/annual periodic filing. These are the
#: FALLBACK series -- the paperwork that FOLLOWS an earnings release, not the
#: release itself (see ``_RELEASE_FORM``/``_RELEASE_ITEM`` below).
_EARNINGS_FORMS = ("10-Q", "10-K")

#: The earnings RELEASE itself: an 8-K carrying item 2.02, "Results of
#: Operations and Financial Condition". This is what actually moves the
#: stock, and it precedes the matching 10-Q/10-K by days to weeks (SoFi's
#: median lag is 9 days), so projecting from periodic filings systematically
#: points at a date AFTER the catalyst has already passed. SPEC.md Sec.21a.
_RELEASE_FORM = "8-K"
_RELEASE_ITEM = "2.02"

#: A report published within this many days of the reference date counts as
#: "just reported": the quarter is out, and the next release -- not the one
#: already published -- is the upcoming catalyst. SPEC.md Sec.21b.
_RECENTLY_REPORTED_DAYS = 3

#: Quarter label for a given count of 10-Qs already filed since the last
#: 10-K (0 = right after a 10-K, so the next filing is Q1). Any count at or
#: beyond 3 (i.e. Q1, Q2, Q3 already filed) means the next filing is the
#: annual 10-K.
_QUARTER_LABELS = {0: "Q1", 1: "Q2", 2: "Q3"}


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse a ``YYYY-MM-DD`` date string, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _DATE_FMT).date()
    except (ValueError, TypeError):
        return None


def _month_date(d: date) -> str:
    """Render a date as ``"<day> <month abbreviation>"``, e.g. ``"27 Aug"``."""
    return f"{d.day} {_MONTH_ABBREVIATIONS[d.month - 1]}"


def _next_quarter_label(pairs: List[Tuple[date, str]]) -> str:
    """Guess which fiscal quarter the *next* filing will report.

    Walks ``pairs`` (sorted ascending) backwards from the most recent
    filing, counting consecutive 10-Qs until (and not including) the most
    recent 10-K. That count is "how many quarterly filings have happened
    since the last annual filing," which maps directly to which quarter
    comes next (see ``_QUARTER_LABELS``): 0 10-Qs since the 10-K means the
    next filing is Q1, ..., 3 means the next filing is the annual 10-K
    itself.
    """
    q_count_since_10k = 0
    for _, form in reversed(pairs):
        if form == "10-K":
            break
        if form == "10-Q":
            q_count_since_10k += 1
    return _QUARTER_LABELS.get(q_count_since_10k, "FY")


def _next_quarter_label_from_releases(
    releases: List[date], last_10k: Optional[date]
) -> Optional[str]:
    """Guess which fiscal quarter the *next* earnings release will report.

    The 8-K series carries no form distinction between a quarterly and an
    annual release, so the count is anchored on the most recent ``10-K``
    FILING date instead: the number of releases published strictly after it
    maps straight through :data:`_QUARTER_LABELS` (0 -> Q1, ..., 3+ -> FY).

    Anchoring on the 10-K *filing* (rather than a fiscal-year end) is what
    makes this correct: a filer's Q4/FY release lands BEFORE the 10-K that
    reports the same year, so it is counted against the PREVIOUS annual
    filing and never inflates the current year's count.

    Returns ``None`` when the history carries no 10-K to anchor on -- the
    caller then omits the quarter from the label rather than guessing.
    """
    if last_10k is None:
        return None
    count = sum(1 for d in releases if d > last_10k)
    return _QUARTER_LABELS.get(count, "FY")


def estimate_next_earnings(submissions: dict, today: Optional[date] = None) -> Optional[dict]:
    """Best-effort estimate of a filer's next earnings release.

    Projects from the ``8-K`` item-2.02 ("Results of Operations") series --
    the earnings release itself -- falling back to the ``10-Q``/``10-K``
    cadence only when fewer than :data:`_MIN_USABLE_FILINGS` releases are
    available (SPEC.md Sec.21a). The periodic filing lands days to weeks
    AFTER the release it reports, so the fallback systematically points past
    the actual catalyst; it exists for filers whose 8-K items SEC has not
    populated, and for callers passing a submissions dict with no ``items``
    key at all.

    Args:
        today: Reference date the projection walks forward from; also the
            point-in-time cutoff (filings dated after ``today`` are ignored).
            Defaults to :meth:`date.today`. Exposed for as-of / testing.
        submissions: The dict returned by
            ``sec_analyzer.fetch.companyfacts.get_submissions``, or any dict
            with the same ``filings.recent.{form,filingDate}`` shape.

    Returns:
        ``None`` if neither series has :data:`_MIN_USABLE_FILINGS` usable
        dates, or on any unexpected internal error. Otherwise a dict::

            {
              "estimate_date": "YYYY-MM-DD",  # always the NEXT release
              "label": "Q3 earnings ~28 Oct",
              "based_on": "median gap of the last 8 earnings releases (91 days)",
              "source": "8-K 2.02" | "10-Q/10-K",
              "last_report_date": "YYYY-MM-DD",
              "days_until": 90,
              "recently_reported": False,
            }

        ``estimate_date`` is never the report that was just published: when
        the most recent release is within :data:`_RECENTLY_REPORTED_DAYS` of
        ``today``, ``recently_reported`` is ``True`` and ``label`` leads with
        that fact ("reported on 29 Jul · next: ...") instead of
        presenting a past event as an upcoming catalyst (SPEC.md Sec.21b).
        ``days_until`` is measured against the same ``today`` the projection
        used, so downstream consumers stay deterministic in as-of mode.

        This function never raises.
    """
    try:
        return _estimate_next_earnings(submissions or {}, today or date.today())
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("estimate_next_earnings() failed unexpectedly; returning None.")
        return None


def _project(dates: List[date], today: date) -> Optional[Tuple[date, date, float, int]]:
    """Median-gap projection shared by both series.

    Returns ``(last_date, next_date, median_gap, sample_size)``, or ``None``
    when the cadence isn't usable.
    """
    dates = dates[-_MAX_FILINGS_CONSIDERED:]
    gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        logger.debug("estimate_next_earnings: non-positive median gap (%s); returning None.", median_gap)
        return None

    last_date = dates[-1]
    next_date = last_date + timedelta(days=median_gap)
    while next_date < today:
        next_date += timedelta(days=median_gap)
    return last_date, next_date, median_gap, len(dates)


def _build_result(
    last_date: date,
    next_date: date,
    median_gap: float,
    sample_size: int,
    quarter_label: Optional[str],
    source: str,
    noun: str,
    today: date,
) -> dict:
    """Assemble the public estimate dict (SPEC.md Sec.21b)."""
    days_since_last = (today - last_date).days
    recently_reported = 0 <= days_since_last <= _RECENTLY_REPORTED_DAYS

    if quarter_label:
        next_label = f"{quarter_label} earnings ~{_month_date(next_date)}"
    else:
        next_label = f"Next earnings ~{_month_date(next_date)}"

    if recently_reported:
        # Lead with what actually happened; the projection is secondary. A
        # quarter published yesterday must never read as an upcoming event.
        label = f"Reported on {_month_date(last_date)} · next: {next_label}"
    else:
        label = next_label

    return {
        "estimate_date": next_date.strftime(_DATE_FMT),
        "label": label,
        "based_on": f"median gap of the last {sample_size} {noun} ({int(median_gap)} days)",
        "source": source,
        "last_report_date": last_date.strftime(_DATE_FMT),
        "days_until": (next_date - today).days,
        "recently_reported": recently_reported,
    }


def _estimate_next_earnings(submissions: dict, today: date) -> Optional[dict]:
    recent = ((submissions.get("filings") or {}).get("recent")) or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    items = recent.get("items") or []

    pairs: List[Tuple[date, str]] = []
    releases: List[date] = []
    for index, (form, filing_date) in enumerate(zip(forms, filing_dates)):
        parsed = _parse_date(filing_date)
        if parsed is None:
            continue
        if parsed > today:
            # Point-in-time guard: filing not yet public as of the reference date.
            continue
        if form in _EARNINGS_FORMS:
            pairs.append((parsed, form))
        elif form == _RELEASE_FORM:
            item_str = items[index] if index < len(items) else None
            if item_str and _RELEASE_ITEM in str(item_str):
                releases.append(parsed)

    pairs.sort(key=lambda p: p[0])
    releases.sort()

    # Primary series: the earnings RELEASE (8-K item 2.02). SPEC.md Sec.21a.
    if len(releases) >= _MIN_USABLE_FILINGS:
        projected = _project(releases, today)
        if projected is not None:
            last_date, next_date, median_gap, sample_size = projected
            last_10k = next((d for d, form in reversed(pairs) if form == "10-K"), None)
            return _build_result(
                last_date, next_date, median_gap, sample_size,
                _next_quarter_label_from_releases(releases, last_10k),
                source="8-K 2.02", noun="earnings releases", today=today,
            )

    # Fallback: periodic-filing cadence. Reached when SEC has not populated
    # 8-K items for this filer, or the history is too short -- behavior here
    # is unchanged from before Sec.21.
    if len(pairs) < _MIN_USABLE_FILINGS:
        logger.debug(
            "estimate_next_earnings: only %d usable 8-K 2.02 release dates and "
            "%d usable 10-Q/10-K filing dates found (need >= %d of either); "
            "returning None.",
            len(releases), len(pairs), _MIN_USABLE_FILINGS,
        )
        return None

    pairs = pairs[-_MAX_FILINGS_CONSIDERED:]
    projected = _project([d for d, _ in pairs], today)
    if projected is None:
        return None
    last_date, next_date, median_gap, sample_size = projected

    return _build_result(
        last_date, next_date, median_gap, sample_size,
        _next_quarter_label(pairs),
        source="10-Q/10-K", noun="filings", today=today,
    )
