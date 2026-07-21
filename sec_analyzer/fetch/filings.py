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

#: Turkish 3-letter month abbreviations, indexed 0 (January) to 11 (December).
_TURKISH_MONTHS = [
    "Oca", "Şub", "Mar", "Nis", "May", "Haz",
    "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara",
]

#: Minimum number of usable (form, filingDate) pairs required before an
#: estimate is attempted -- with fewer than this, a median gap isn't a
#: meaningful cadence.
_MIN_USABLE_FILINGS = 3

#: Only the most recent this-many quarterly filings are used to compute the
#: median gap, so a stale/older cadence (e.g. before a fiscal-year-end
#: change) doesn't skew the estimate.
_MAX_FILINGS_CONSIDERED = 9

#: Forms that mark a quarterly/annual earnings-related filing.
_EARNINGS_FORMS = ("10-Q", "10-K")

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


def _turkish_date(d: date) -> str:
    """Render a date as ``"<day> <Turkish month abbreviation>"``, e.g. ``"27 Ağu"``."""
    return f"{d.day} {_TURKISH_MONTHS[d.month - 1]}"


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


def estimate_next_earnings(submissions: dict, today: Optional[date] = None) -> Optional[dict]:
    """Best-effort estimate of the next 10-Q/10-K filing (a proxy for the
    next earnings release) from a filer's submissions history.

    Args:
        today: Reference date the projection walks forward from; also the
            point-in-time cutoff (filings dated after ``today`` are ignored).
            Defaults to :meth:`date.today`. Exposed for as-of / testing.
        submissions: The dict returned by
            ``sec_analyzer.fetch.companyfacts.get_submissions``, or any dict
            with the same ``filings.recent.{form,filingDate}`` shape.

    Returns:
        ``None`` if fewer than :data:`_MIN_USABLE_FILINGS` usable 10-Q/10-K
        filing dates are available, or on any unexpected internal error.
        Otherwise a dict::

            {
              "estimate_date": "YYYY-MM-DD",
              "label": "Q2 earnings ~27 Ağu",
              "based_on": "son 8 dosyalamanın medyan aralığı (91 gün)",
            }

        This function never raises.
    """
    try:
        return _estimate_next_earnings(submissions or {}, today or date.today())
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("estimate_next_earnings() failed unexpectedly; returning None.")
        return None


def _estimate_next_earnings(submissions: dict, today: date) -> Optional[dict]:
    recent = ((submissions.get("filings") or {}).get("recent")) or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []

    pairs: List[Tuple[date, str]] = []
    for form, filing_date in zip(forms, filing_dates):
        if form not in _EARNINGS_FORMS:
            continue
        parsed = _parse_date(filing_date)
        if parsed is None:
            continue
        if parsed > today:
            # Point-in-time guard: filing not yet public as of the reference date.
            continue
        pairs.append((parsed, form))

    pairs.sort(key=lambda p: p[0])

    if len(pairs) < _MIN_USABLE_FILINGS:
        logger.debug(
            "estimate_next_earnings: only %d usable 10-Q/10-K filing dates "
            "found (need >= %d); returning None.",
            len(pairs), _MIN_USABLE_FILINGS,
        )
        return None

    pairs = pairs[-_MAX_FILINGS_CONSIDERED:]
    dates = [d for d, _ in pairs]

    gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        logger.debug("estimate_next_earnings: non-positive median gap (%s); returning None.", median_gap)
        return None

    last_date, _ = pairs[-1]
    next_date = last_date + timedelta(days=median_gap)
    while next_date < today:
        next_date += timedelta(days=median_gap)

    quarter_label = _next_quarter_label(pairs)
    label = f"{quarter_label} earnings ~{_turkish_date(next_date)}"
    based_on = f"son {len(pairs)} dosyalamanın medyan aralığı ({int(median_gap)} gün)"

    return {
        "estimate_date": next_date.strftime(_DATE_FMT),
        "label": label,
        "based_on": based_on,
    }
