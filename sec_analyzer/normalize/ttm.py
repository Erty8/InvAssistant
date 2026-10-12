"""Trailing-twelve-month (TTM) helpers over an SEC companyfacts document.

Pure functions of the companyfacts dict -- no network, no wall clock, no
randomness. They serve the ``funding-screen`` (see
``sec_analyzer/screener/FUNDING_SPEC.md`` Sec.4) and deliberately live apart
from :mod:`sec_analyzer.normalize.normalizer`, which is fiscal-year based and
feeds the valuation engine.

Flow items: ``TTM = last 10-K annual + current-year YTD - prior-year YTD``.
Balance-sheet items are instants taken at the TTM end date.

Known data pitfalls handled here (spec Sec.4.1): 10-Qs that tag a
trailing-12-month duration must not count as a fiscal year (annual facts must
come from a ``10-K*`` form); January fiscal years and 52/53-week years shift
period ends by a few days, so prior-year matching is tolerance based; some
filers report one ``dei`` share row per share class.

All helpers are defensive: malformed rows are skipped and missing data yields
``None`` -- nothing here raises on a bad document.
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Forms the screen reads (spec Sec.3).
FORMS = ("10-K", "10-K/A", "10-KT", "10-Q", "10-Q/A")

#: Annual durations (days) accepted as a fiscal year.
ANNUAL_MIN_DAYS = 340
ANNUAL_MAX_DAYS = 390

#: Tolerances (days) for period matching (spec Sec.4.1).
YTD_START_TOL = 4
PRIOR_END_TOL = 10
PRIOR_DUR_TOL = 12
PRIOR_FY_TOL = 10

#: Quarter duration window (days) and end-date tolerances for dilution.
_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100
_DIL_CUR_TOL = 5
_DIL_PRIOR_TOL = 10

DILUTED_TAG = "WeightedAverageNumberOfDilutedSharesOutstanding"

LONG_TERM_DEBT_TAGS = [
    "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
    "LongTermNotesPayable", "LongTermNotesAndLoans", "SeniorLongTermNotes",
    "ConvertibleNotesPayable", "ConvertibleDebtNoncurrent",
]
CURRENT_DEBT_TAGS = [
    "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "DebtCurrent", "NotesPayableCurrent",
]
#: Combined tags that already include the current portion.
TOTAL_DEBT_TAGS = [
    "LongTermDebt", "DebtInstrumentCarryingAmount",
    "DebtLongtermAndShorttermCombinedAmount",
]
SHORT_TERM_DEBT_TAGS = ["CommercialPaper", "ShortTermBorrowings"]
CASH_TAGS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]
SHORT_TERM_INVEST_TAGS = [
    "ShortTermInvestments", "MarketableSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
]


# ---------------------------------------------------------------------------
# Low-level fact access
# ---------------------------------------------------------------------------

def _parse(d) -> Optional[date]:
    """Parse an ISO ``YYYY-MM-DD`` string; ``None`` if unusable."""
    try:
        return date.fromisoformat(str(d)[:10])
    except (TypeError, ValueError):
        return None


def _days_between(a: date, b: date) -> int:
    return (b - a).days


def _rows(facts: dict, tag: str, unit: str = "USD", ns: str = "us-gaap") -> List[dict]:
    """Facts of ``tag`` in the allowed forms with a numeric value and valid end."""
    try:
        raw = facts.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {}).get(unit, [])
    except AttributeError:
        return []
    out = []
    for r in raw or []:
        if not isinstance(r, dict) or r.get("form") not in FORMS:
            continue
        if not isinstance(r.get("val"), (int, float)) or isinstance(r.get("val"), bool):
            continue
        if _parse(r.get("end")) is None:
            continue
        out.append(r)
    return out


def _duration_facts(
    facts: dict, tag: str, unit: str = "USD", annual_only: bool = False
) -> Dict[Tuple[str, str], float]:
    """``{(start, end): value}`` for duration facts; most recently filed wins.

    Args:
        facts: companyfacts document.
        tag: us-gaap tag.
        unit: XBRL unit key.
        annual_only: Keep facts from ``10-K*`` forms only, so a 10-Q that tags
            a trailing-12-month duration is never mistaken for a fiscal year.
    """
    best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for r in _rows(facts, tag, unit):
        if "start" not in r or _parse(r["start"]) is None:
            continue
        if annual_only and not str(r.get("form", "")).startswith("10-K"):
            continue
        key = (r["start"], r["end"])
        filed = str(r.get("filed") or "")
        if key not in best or filed > best[key][1]:
            best[key] = (r["val"], filed)
    return {k: v[0] for k, v in best.items()}


def _dur(key: Tuple[str, str]) -> int:
    return _days_between(_parse(key[0]), _parse(key[1]))


def _near(a: str, b: date, tol: int) -> bool:
    da = _parse(a)
    return da is not None and abs(_days_between(b, da)) <= tol


def _prior_year_match(
    d: Dict[Tuple[str, str], float], end: date, dur: int
) -> Optional[float]:
    """Value of the period ending ~364 days before ``end`` with duration ~``dur``.

    Covers January fiscal years and 52/53-week years (spec Sec.4.1). When
    several facts qualify the closest end date wins, then the closest
    duration, then the later end date (deterministic).
    """
    target = end - timedelta(days=364)
    best = None
    for k, v in d.items():
        e = _parse(k[1])
        gap = abs(_days_between(target, e))
        dgap = abs(_dur(k) - dur)
        if gap <= PRIOR_END_TOL and dgap <= PRIOR_DUR_TOL:
            rank = (gap, dgap, _neg_ord(e))
            if best is None or rank < best[0]:
                best = (rank, v)
    return best[1] if best else None


def _neg_ord(d: date) -> int:
    return -d.toordinal()


# ---------------------------------------------------------------------------
# Flow items
# ---------------------------------------------------------------------------

def _flow_tag(facts: dict, tag: str, unit: str) -> Optional[dict]:
    """TTM info for a single tag, or ``None`` when it has no 10-K annual data."""
    d = _duration_facts(facts, tag, unit)
    annual: Dict[str, float] = {}
    for k, v in sorted(_duration_facts(facts, tag, unit, annual_only=True).items()):
        if ANNUAL_MIN_DAYS <= _dur(k) <= ANNUAL_MAX_DAYS:
            annual[k[1]] = v
    if not annual:
        return None
    ends = sorted(annual)
    fy_end = ends[-1]
    fy_date = _parse(fy_end)

    res = {
        "tag": tag, "fy_end": fy_end, "annual": annual, "ttm": annual[fy_end],
        "ttm_end": fy_end, "ttm_prior": None, "ytd": None, "ytd_prior": None,
        "ytd_growth": None,
    }

    # Prior FY: must end within +-10 days of FY end - 365 days.
    prior_fy_target = fy_date - timedelta(days=365)
    a_prev = None
    for e in reversed(ends[:-1]):
        if _near(e, prior_fy_target, PRIOR_FY_TOL):
            a_prev = annual[e]
            break

    # Current-year YTD: starts right after FY end, ends later, < 340 days.
    start_target = fy_date + timedelta(days=1)
    cands = []
    for k, v in d.items():
        s, e = _parse(k[0]), _parse(k[1])
        if (abs(_days_between(start_target, s)) <= YTD_START_TOL
                and e > fy_date and _dur(k) < ANNUAL_MIN_DAYS):
            cands.append((e, -abs(_days_between(start_target, s)), k, v))
    if not cands:
        res["ttm_prior"] = a_prev
        return res
    e_date, _, ykey, y_val = max(cands, key=lambda c: (c[0], c[1]))
    dur = _dur(ykey)
    p_val = _prior_year_match(d, e_date, dur)
    if p_val is None:
        res["ttm_prior"] = a_prev
        return res

    res.update(
        ttm=annual[fy_end] + y_val - p_val, ttm_end=ykey[1],
        ytd=y_val, ytd_prior=p_val,
    )
    # Prior TTM (ending ~1y earlier): A_{F-1} + P - P_{-1}.
    p_end = e_date - timedelta(days=364)
    p_prior = _prior_year_match(d, p_end, dur)
    if a_prev is not None and p_prior is not None:
        res["ttm_prior"] = a_prev + p_val - p_prior
    elif p_val:
        res["ytd_growth"] = y_val / p_val - 1
    return res


def ttm_flow(
    facts: dict, tags: List[str], unit: str = "USD", want_end: Optional[str] = None
) -> Optional[dict]:
    """Trailing-twelve-month value of a flow concept.

    For each tag (in list order) builds ``TTM = A_F + Y - P`` where ``A_F`` is
    the latest 10-K annual value, ``Y`` the current-year YTD and ``P`` the
    prior-year YTD (falling back to ``TTM = A_F`` when no YTD pair exists).

    Tag choice: with ``want_end`` the first tag whose TTM ends on that date
    wins; otherwise (or if none matches) the tag with the latest ``ttm_end``
    wins, ties going to list order.

    Args:
        facts: SEC companyfacts document.
        tags: Ordered us-gaap tags to try.
        unit: XBRL unit key (default ``"USD"``).
        want_end: Optional ISO date the TTM should end on.

    Returns:
        ``{tag, fy_end, annual: {end: value}, ttm, ttm_end, ttm_prior, ytd,
        ytd_prior, ytd_growth}`` (inapplicable keys ``None``), or ``None`` if
        no tag has 10-K annual data.
    """
    try:
        cands = []
        for t in tags:
            r = _flow_tag(facts, t, unit)
            if r:
                cands.append(r)
        if not cands:
            return None
        if want_end:
            for r in cands:
                if r["ttm_end"] == want_end:
                    return r
        best = cands[0]
        for r in cands[1:]:
            if r["ttm_end"] > best["ttm_end"]:
                best = r
        return best
    except Exception:  # noqa: BLE001 - never raise on odd data
        logger.warning("ttm_flow failed for tags %s", tags, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Instants / balance sheet
# ---------------------------------------------------------------------------

def instant(facts: dict, tags: List[str], end: str, unit: str = "USD") -> Optional[float]:
    """Value of the first tag (list order) with an instant fact ending on ``end``.

    Duration facts (those with a ``start``) are ignored. When several facts
    share the end date the most recently filed wins.

    Returns:
        The value, or ``None`` if no tag has one.
    """
    for t in tags:
        best = None
        for r in _rows(facts, t, unit):
            if "start" not in r and r["end"] == end:
                filed = str(r.get("filed") or "")
                if best is None or filed > best[1]:
                    best = (r["val"], filed)
        if best is not None:
            return best[0]
    return None


def balance(facts: dict, end: str) -> dict:
    """Debt, liquidity and net debt at ``end`` (spec Sec.4.4).

    ``debt = long_term + current + short_term``. If no long-term tag matches,
    a combined tag (already including the current portion) is used and
    ``current`` is 0 so nothing is double-counted. Missing pieces count as 0.

    Returns:
        ``{"debt", "liquidity", "net_debt", "debt_tag_found"}``: floats plus a
        bool that is True when ANY debt tag (long-term, current, combined
        fallback or short-term) matched at ``end``, so "no debt tags" is not
        silently read as "no debt".
    """
    lt = instant(facts, LONG_TERM_DEBT_TAGS, end)
    cur_raw = instant(facts, CURRENT_DEBT_TAGS, end)
    cur = cur_raw or 0.0
    total_raw = None
    if lt is None:
        total_raw = instant(facts, TOTAL_DEBT_TAGS, end)
        lt = total_raw or 0.0
        cur = 0.0
    short_raw = instant(facts, SHORT_TERM_DEBT_TAGS, end)
    short = short_raw or 0.0
    cash = instant(facts, CASH_TAGS, end) or 0.0
    sti = instant(facts, SHORT_TERM_INVEST_TAGS, end) or 0.0
    debt = lt + cur + short
    liquidity = cash + sti
    found = any(x is not None for x in (
        instant(facts, LONG_TERM_DEBT_TAGS, end), cur_raw, total_raw, short_raw))
    return {
        "debt": debt, "liquidity": liquidity, "net_debt": debt - liquidity,
        "debt_tag_found": found,
    }


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------

def diluted_share_change(facts: dict, end: str) -> Optional[float]:
    """Year-over-year change in diluted weighted shares, as a fraction.

    Primary: the quarter (80-100 days) ending within 5 days of ``end`` divided
    by the quarter ending within 10 days of ``end - 364`` days, minus 1.
    Fallback: latest annual / previous annual - 1. Otherwise ``None``.
    """
    try:
        end_d = _parse(end)
        if end_d is None:
            return None
        d = _duration_facts(facts, DILUTED_TAG, "shares")
        quarters = {k: v for k, v in d.items()
                    if _QUARTER_MIN_DAYS <= _dur(k) <= _QUARTER_MAX_DAYS}
        cur = _closest(quarters, end_d, _DIL_CUR_TOL)
        prev = _closest(quarters, end_d - timedelta(days=364), _DIL_PRIOR_TOL)
        if cur and prev:
            return cur / prev - 1
        ann = sorted(
            (k[1], v)
            for k, v in _duration_facts(facts, DILUTED_TAG, "shares", annual_only=True).items()
            if ANNUAL_MIN_DAYS <= _dur(k) <= ANNUAL_MAX_DAYS
        )
        if len(ann) > 1 and ann[-2][1]:
            return ann[-1][1] / ann[-2][1] - 1
    except Exception:  # noqa: BLE001
        logger.warning("diluted_share_change failed", exc_info=True)
    return None


def _closest(d: Dict[Tuple[str, str], float], target: date, tol: int) -> Optional[float]:
    best = None
    for k, v in d.items():
        gap = abs(_days_between(target, _parse(k[1])))
        if gap <= tol and (best is None or (gap, k[1]) < best[0]):
            best = ((gap, k[1]), v)
    return best[1] if best else None


def cover_shares(facts: dict) -> Tuple[Optional[float], Optional[str]]:
    """Latest cover-page share count, summing every share class of that filing.

    Takes the ``dei:EntityCommonStockSharesOutstanding`` filing that is latest
    by ``filed`` then ``end`` and sums every row with the same ``accn``
    (GOOGL reports one row per class).

    Returns:
        ``(shares, as_of_end)``; ``(None, None)`` when unavailable.
    """
    try:
        rows = _rows(facts, "EntityCommonStockSharesOutstanding", "shares", ns="dei")
        if not rows:
            return None, None
        last = max(rows, key=lambda r: (str(r.get("filed") or ""), str(r["end"])))
        accn = last.get("accn")
        same = [r for r in rows if r.get("accn") == accn] if accn else [last]
        return float(sum(r["val"] for r in same)), last["end"]
    except Exception:  # noqa: BLE001
        logger.warning("cover_shares failed", exc_info=True)
        return None, None
