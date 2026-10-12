"""Self-funded growth screen (``funding-screen``).

Contract: ``sec_analyzer/screener/FUNDING_SPEC.md`` (binding).

This is a **research filter, not a signal**. It narrows an index to companies
whose trailing-twelve-month (TTM) SEC filings show growth funded by their own
cash flow: high FCF margin, little or no net debt, low dilution. It has never
been backtested (a backtest on today's index membership would carry
survivorship bias), and passing is a statement about the filings, not about
the stock's future return. The optional valuation "price reading" is a
heuristic, not engine output, and must never be shown as a valuation verdict.

Determinism: results depend only on the companyfacts documents (and, for the
overlay, the price/assumptions passed in) -- never on the wall clock.
Nothing here raises: a failing ticker becomes an error row.
"""

import contextlib
import io
import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from sec_analyzer.normalize.concepts import CONCEPTS
from sec_analyzer.normalize.ttm import (
    balance,
    cover_shares,
    diluted_share_change,
    ttm_flow,
)
from sec_analyzer.valuation import reverse_dcf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (spec Sec.5.2, 6.4)
# ---------------------------------------------------------------------------

#: FCF margin must be at least this fraction of revenue.
FCF_MARGIN_MIN = 0.15
#: Net debt / EBITDA must stay below this (net cash always passes).
LEVERAGE_MAX = 1.0
#: EBIT / interest must be at least this (n/a for a net-cash company).
INTEREST_COV_MIN = 8.0
#: Revenue growth + FCF margin, in percentage points.
RULE_OF_40_MIN = 40.0
#: Year-over-year diluted share growth must not exceed this fraction.
DILUTION_MAX = 0.03
#: Price-reading band in percentage points (implied growth vs TTM growth).
PRICE_READING_GAP_PP = 10.0

TEST_IDS = ("fcf_margin", "leverage", "interest_cov", "rule_of_40", "dilution")

#: Extra revenue tag appended to ``CONCEPTS["Revenue"]`` (screener-local; the
#: shared CONCEPTS list is never mutated).
EXTRA_REVENUE_TAGS = ["RevenueFromContractWithCustomerIncludingAssessedTax"]
#: Interest-expense tags, in order (spec Sec.5.1).
INTEREST_TAGS = [
    "InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
    "InterestAndDebtExpense", "InterestExpenseNet",
]

#: Net-revenue guard (screener-local; mirrors valuation SPEC Sec.19).
NET_REVENUE_TAGS = ["RevenuesNetOfInterestExpense"]
NET_REVENUE_DIFF = 0.05
#: Interest FY fallback is used only if that FY ends within this many days of E.
INTEREST_FY_MAX_DAYS = 400
#: Overlay assumption bounds (the engine's clamp bounds).
MIN_DISCOUNT_RATE = 0.07
MAX_TERMINAL_GROWTH = 0.04

DISCLAIMER = (
    "Research filter, not a signal: it describes trailing-twelve-month SEC "
    "filings only. It has never been backtested (a backtest on today's index "
    "membership would carry survivorship bias) and says nothing about future "
    "returns. The price reading is a heuristic, not a valuation verdict. "
    "Implied growth is the constant 5-year growth rate of FCF minus SBC "
    "(fading to terminal growth by year 10) that the engine's discount rate "
    "and terminal growth need to justify the price; it is compared with one "
    "trailing year of sales growth, so the gap also absorbs margin change, "
    "SBC trend and the discount-rate choice (about 3pp of implied growth per "
    "1pp of discount rate). "
    "Known limitations: cover-page shares are basic, not diluted; a split "
    "after the latest filing, or share classes with unequal economics (e.g. "
    "BRK), distorts market cap; under FCFE-direct, cash counts only through "
    "its interest income in CFO, so net-cash names (which this screen "
    "selects for) show somewhat higher implied growth."
)

NO_10K_ERROR = "no 10-K annual data (foreign filer 20-F/40-F?)"

PRICE_READING_UP = "price implies FCF growth well above TTM sales growth"
PRICE_READING_DOWN = "price implies FCF growth well below TTM sales growth"
PRICE_READING_FLAT = "price-implied FCF growth near TTM sales growth"
PRICE_READING_ABOVE_BRACKET = "implied growth above +60% (off scale)"
PRICE_READING_BELOW_BRACKET = "implied growth below −20% (off scale)"
PRICE_READING_NA = "n/a"
YTD_SUFFIX = " (YTD growth basis)"

_DASH = "—"
_GROUPS = ("A", "B", "C")


def thresholds() -> dict:
    """The screen thresholds as a JSON-friendly dict."""
    return {
        "fcf_margin_min": FCF_MARGIN_MIN,
        "net_debt_ebitda_max": LEVERAGE_MAX,
        "interest_coverage_min": INTEREST_COV_MIN,
        "rule_of_40_min": RULE_OF_40_MIN,
        "dilution_max": DILUTION_MAX,
        "price_reading_gap_pp": PRICE_READING_GAP_PP,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(x) -> Optional[float]:
    """``x`` as a finite float, else ``None``."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _error_row(ticker, message: str) -> dict:
    return {"ticker": ticker, "status": "error", "error": str(message)}


def _flow_on(facts: dict, tags: List[str], end: str) -> Optional[dict]:
    """``ttm_flow`` restricted to a TTM that ends exactly on ``end``."""
    r = ttm_flow(facts, tags, want_end=end)
    return r if r and r["ttm_end"] == end else None


# ---------------------------------------------------------------------------
# Tests and groups (spec Sec.5.2)
# ---------------------------------------------------------------------------

def evaluate_tests(row: dict) -> Tuple[Dict[str, str], List[str]]:
    """Evaluate the five tests on a metrics row.

    Missing data fails the three core tests (``fcf_margin``, ``leverage``,
    ``rule_of_40``); for ``interest_cov`` and ``dilution`` it is ``n/a``.
    ``interest_cov`` is also ``n/a`` for a net-cash company.

    Args:
        row: A row with ``fcf_margin`` (fraction), ``net_debt``,
            ``net_debt_ebitda``, ``interest_cov``, ``rule_of_40`` (points)
            and ``dilution`` (fraction); absent keys count as missing.

    Returns:
        ``(tests, fails)``: ``tests`` maps each id in :data:`TEST_IDS` to
        ``"pass"``/``"fail"``/``"n/a"``; ``fails`` lists the failed ids in
        :data:`TEST_IDS` order.
    """
    row = row or {}
    # Round before comparing so float noise cannot flip a boundary case
    # (e.g. 103/100 - 1 must equal exactly 3%).
    def _r(key):
        v = _num(row.get(key))
        return None if v is None else round(v, 6)

    margin = _r("fcf_margin")
    net_debt = _r("net_debt")
    nd_ebitda = _r("net_debt_ebitda")
    cov = _r("interest_cov")
    r40 = _r("rule_of_40")
    dil = _r("dilution")

    tests: Dict[str, str] = {}
    tests["fcf_margin"] = "pass" if margin is not None and margin >= FCF_MARGIN_MIN else "fail"

    if net_debt is None:
        tests["leverage"] = "fail"
    elif net_debt <= 0:
        tests["leverage"] = "pass"
    elif nd_ebitda is None:  # EBITDA <= 0 or missing
        tests["leverage"] = "fail"
    else:
        tests["leverage"] = "pass" if nd_ebitda < LEVERAGE_MAX else "fail"

    if (net_debt is not None and net_debt <= 0) or cov is None:
        tests["interest_cov"] = "n/a"
    else:
        tests["interest_cov"] = "pass" if cov >= INTEREST_COV_MIN else "fail"

    tests["rule_of_40"] = "pass" if r40 is not None and r40 >= RULE_OF_40_MIN else "fail"

    if dil is None:
        tests["dilution"] = "n/a"
    else:
        tests["dilution"] = "pass" if dil <= DILUTION_MAX else "fail"

    fails = [t for t in TEST_IDS if tests[t] == "fail"]
    return tests, fails


def assign_group(n_fails: Optional[int]) -> Optional[str]:
    """Group A = 0 fails, B = 1 fail, C = 2 or more; ``None`` for no count."""
    if n_fails is None:
        return None
    if n_fails <= 0:
        return "A"
    return "B" if n_fails == 1 else "C"


def sort_rows(rows: List[dict]) -> List[dict]:
    """Sort rows: group A to C, then Rule of 40 descending (missing last),
    then ticker; error rows last, sorted by ticker. Returns a new list."""
    def key(r: dict):
        ticker = str(r.get("ticker") or "")
        if r.get("status") == "error":
            return (1, 0, 0, 0.0, ticker)
        grp = r.get("group")
        gi = _GROUPS.index(grp) if grp in _GROUPS else len(_GROUPS)
        r40 = _num(r.get("rule_of_40"))
        return (0, gi, 0 if r40 is not None else 1, -(r40 or 0.0), ticker)

    return sorted(rows, key=key)


# ---------------------------------------------------------------------------
# Per-ticker metrics (spec Sec.5.1)
# ---------------------------------------------------------------------------

def screen_facts(ticker: str, name: Optional[str], cik: Optional[str], facts: dict) -> dict:
    """Compute metrics, tests and group for one company from its companyfacts.

    Pure: no network. Never raises; any failure becomes an error row
    ``{ticker, status: "error", error}``. Ratios are stored as fractions
    (``rule_of_40`` in points), money raw in USD.

    Args:
        ticker: Ticker symbol.
        name: Company name (may be ``None``).
        cik: 10-digit CIK string (may be ``None``).
        facts: SEC companyfacts document.

    Returns:
        A row with ``status: "ok"``, metrics, ``tests``, ``fails``,
        ``group`` and ``notes``; or an error row.
    """
    try:
        return _screen_facts(ticker, name, cik, facts)
    except Exception as exc:  # noqa: BLE001 - one bad ticker must not abort the screen
        logger.warning("Funding screen failed for %s", ticker, exc_info=True)
        return _error_row(ticker, f"{type(exc).__name__}: {exc}")


def _screen_facts(ticker, name, cik, facts) -> dict:
    if not isinstance(facts, dict) or not facts.get("facts"):
        return _error_row(ticker, NO_10K_ERROR)

    revenue_tags = list(CONCEPTS["Revenue"]) + EXTRA_REVENUE_TAGS
    ocf = ttm_flow(facts, CONCEPTS["OperatingCashFlow"])
    if ocf is None:
        if ttm_flow(facts, revenue_tags) is None:
            return _error_row(ticker, NO_10K_ERROR)
        return _error_row(ticker, "no operating cash flow data")

    end = ocf["ttm_end"]
    fy_end = ocf["fy_end"]
    notes: List[str] = []

    rev = ttm_flow(facts, revenue_tags, want_end=end)
    if rev is None:
        return _error_row(ticker, "no revenue data")
    if rev["ttm_end"] != end:
        return _error_row(
            ticker, f"revenue TTM end {rev['ttm_end']} != cash-flow TTM end {end}"
        )
    revenue = rev["ttm"]
    if not revenue or revenue <= 0:
        return _error_row(ticker, "non-positive revenue")

    # Net-revenue guard (mirrors valuation SPEC Sec.19; screener-local).
    net_rev = _flow_on(facts, NET_REVENUE_TAGS, end)
    if (net_rev is not None and net_rev["ttm"] and net_rev["ttm"] > 0
            and abs(net_rev["ttm"] / revenue - 1) > NET_REVENUE_DIFF):
        notes.append(
            f"net-revenue guard: {net_rev['tag']} ({net_rev['ttm']:.0f}) used "
            f"instead of {rev['tag']} ({revenue:.0f})"
        )
        rev = net_rev
        revenue = rev["ttm"]

    rev_growth_basis = None
    if rev.get("ttm_prior") is not None and rev["ttm_prior"] > 0:
        rev_growth = revenue / rev["ttm_prior"] - 1
        rev_growth_basis = "ttm"
    else:
        rev_growth = rev.get("ytd_growth")
        if rev_growth is not None:
            rev_growth_basis = "ytd"
            notes.append("rev_growth: YTD growth used (no prior-year TTM)")

    capex = _flow_on(facts, CONCEPTS["CapEx"], end)
    capex_any = capex or ttm_flow(facts, CONCEPTS["CapEx"])
    if capex is not None:
        capex_val = abs(capex["ttm"])
        fcf = ocf["ttm"] - capex_val
    elif _has_any_tag(facts, CONCEPTS["CapEx"]):
        # A CapEx tag exists but no TTM ends on E: never a silent OCF-only FCF.
        capex_val = None
        fcf = None
        notes.append("CapEx tag present but no TTM ending on the cash-flow end; FCF n/a")
    else:
        capex_val = 0.0
        fcf = ocf["ttm"]
        notes.append("capex_missing: no CapEx tag in the filing; treated as 0")
    fcf_margin = fcf / revenue if fcf is not None else None

    # Margin trend [FY-1, FY, TTM]; TTM only when the TTM end is not the FY end.
    def fy_margin(fy: str) -> Optional[float]:
        rv = _annual_near(rev["annual"], fy)
        if not rv:
            return None
        cx = _annual_near(capex_any["annual"], fy) if capex_any else None
        return (ocf["annual"][fy] - abs(cx or 0.0)) / rv

    fy_date = date.fromisoformat(fy_end)
    prev_fy = next(
        (e for e in sorted(ocf["annual"], reverse=True)
         if abs((date.fromisoformat(e) - (fy_date - timedelta(days=365))).days) <= 10),
        None,
    )
    trend: List[Optional[float]] = [
        fy_margin(prev_fy) if prev_fy else None,
        fy_margin(fy_end),
    ]
    is_ttm = end != fy_end
    if is_ttm:
        trend.append(fcf_margin)  # None when FCF is unknown

    sbc_f = _flow_on(facts, CONCEPTS["SBC"], end)
    sbc = sbc_f["ttm"] if sbc_f else None
    if sbc is None:
        notes.append("SBC missing; FCF-SBC margin n/a")

    ebit_f = _flow_on(facts, CONCEPTS["OperatingIncome"], end)
    ebit = ebit_f["ttm"] if ebit_f else None
    da_f = _flow_on(facts, CONCEPTS["Depreciation"], end)
    if ebit is None:
        ebitda = None
        notes.append("operating income missing; EBITDA n/a")
    else:
        if da_f is None:
            notes.append("D&A missing; treated as 0")
        ebitda = ebit + (da_f["ttm"] if da_f else 0.0)

    ie = ttm_flow(facts, INTEREST_TAGS, want_end=end)
    interest = None
    if ie is not None:
        if ie["ttm_end"] == end:
            interest = ie["ttm"]
        else:
            fy_gap = (date.fromisoformat(end) - date.fromisoformat(ie["fy_end"])).days
            if 0 <= fy_gap <= INTEREST_FY_MAX_DAYS:
                interest = ie["annual"].get(ie["fy_end"])
                notes.append(f"interest: FY ending {ie['fy_end']} value used")
            else:
                notes.append(
                    f"interest: latest FY ({ie['fy_end']}) too old; coverage n/a"
                )

    bal = balance(facts, end)
    net_debt = bal["net_debt"]
    if not bal["debt_tag_found"] and interest is not None and interest > 0:
        net_debt = None
        notes.append("net_debt unknown: no debt tags but interest expense > 0")
    nd_ebitda = (
        net_debt / ebitda
        if net_debt is not None and ebitda is not None and ebitda > 0 else None
    )
    interest_cov = (
        ebit / interest if ebit is not None and interest is not None and interest > 0 else None
    )

    rule_of_40 = (
        None if rev_growth is None or fcf_margin is None
        else 100 * rev_growth + 100 * fcf_margin
    )
    dilution = diluted_share_change(facts, end)
    shares, shares_as_of = cover_shares(facts)

    row = {
        "ticker": ticker, "name": name, "cik": cik, "status": "ok",
        "ttm_end": end, "fy_end": fy_end, "is_ttm": is_ttm,
        "revenue_tag": rev["tag"],
        "revenue": revenue, "rev_growth": rev_growth,
        "rev_growth_basis": rev_growth_basis,
        "cfo": ocf["ttm"], "capex": capex_val,
        "fcf": fcf, "fcf_margin": fcf_margin, "fcf_margin_trend": trend,
        "sbc": sbc,
        "sbc_rev": sbc / revenue if sbc is not None else None,
        "fcf_sbc_margin": (
            (fcf - sbc) / revenue if sbc is not None and fcf is not None else None
        ),
        "ebit": ebit, "ebitda": ebitda,
        "debt": bal["debt"], "liquidity": bal["liquidity"], "net_debt": net_debt,
        "net_debt_ebitda": nd_ebitda,
        "interest_cov": interest_cov,
        "rule_of_40": rule_of_40,
        "dilution": dilution,
        "cover_shares": shares, "cover_shares_as_of": shares_as_of,
        "notes": notes,
    }
    if interest_cov is None and net_debt is not None and net_debt > 0:
        notes.append("interest coverage n/a: interest missing or non-positive")
    if dilution is None:
        notes.append("dilution n/a: diluted share data missing")

    tests, fails = evaluate_tests(row)
    row["tests"] = tests
    row["fails"] = fails
    row["group"] = assign_group(len(fails))
    return row


def _has_any_tag(facts: dict, tags: List[str]) -> bool:
    """True if any of ``tags`` exists (with any fact) in the us-gaap namespace."""
    try:
        gaap = facts["facts"]["us-gaap"]
        return any(
            any(units for units in (gaap.get(t) or {}).get("units", {}).values())
            for t in tags
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _annual_near(annual: Dict[str, float], end: str, tol: int = 5) -> Optional[float]:
    """Annual value whose end date is within ``tol`` days of ``end``."""
    try:
        target = date.fromisoformat(end)
        best = None
        for e, v in sorted(annual.items()):
            gap = abs((date.fromisoformat(e) - target).days)
            if gap <= tol and (best is None or gap < best[0]):
                best = (gap, v)
        return best[1] if best else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Valuation overlay (spec Sec.6)
# ---------------------------------------------------------------------------

def valuation_overlay(
    row: dict,
    price: Optional[float],
    price_as_of: Optional[str],
    base_assumptions: Optional[dict],
    metrics_shares: Optional[float],
    price_reliable: Optional[bool] = None,
    sector_type: Optional[str] = None,
    dcf_enabled: Optional[bool] = None,
    engine_reverse_dcf: Optional[dict] = None,
) -> dict:
    """Reverse-DCF overlay on one screen row (heuristic, not a verdict).

    Pure: all engine/pipeline inputs are passed in. Uses ``fcf0 = fcf - sbc``
    with the engine's base ``discount_rate``/``terminal_growth`` via
    :func:`reverse_dcf.implied_growth_with_status` (``dilution_rate=0``, as
    the engine does, since SBC is already expensed in ``fcf0``). Shares come
    from the cover page (``row["cover_shares"]``), falling back to
    ``metrics_shares``.

    Guards (spec Sec.6.4) run before the solve; any hit gives
    ``implied_growth=None``, ``bracket_status="no_data"`` and a ``reason``:
    unreliable price (also suppresses ``p_fcf``), financial/REIT sector or a
    disabled FCF-DCF, SBC missing or ``fcf - sbc <= 0``, untagged CapEx
    (``capex_missing`` note), or assumptions outside the engine's clamp bounds.

    Price reading: ``gap = implied_growth - rev_growth`` in pp; above +10pp /
    below -10pp / otherwise near (see the ``PRICE_READING_*`` labels); a
    non-``ok`` status is labelled instead. A ``(YTD growth basis)`` suffix is
    added when ``row["rev_growth_basis"] == "ytd"``.

    Args:
        row: A screen row (``fcf``, ``sbc``, ``rev_growth``, ``cover_shares``,
            ``notes``, ``rev_growth_basis``).
        price: Current price.
        price_as_of: Price date (passed through).
        base_assumptions: Engine ``valuation["assumptions"]["base"]``.
        metrics_shares: Engine share count (fallback).
        price_reliable: ``metrics["price_reliable"]``; only ``False`` blocks.
        sector_type: ``valuation["sector_type"]``.
        dcf_enabled: ``valuation["dcf"]["enabled"]``; only ``False`` blocks.
        engine_reverse_dcf: Engine ``valuation["reverse_dcf"]`` block (with
            optional ``fcf0``/``fcf0_source``) for the audit fields.

    Returns:
        ``{price, price_as_of, shares, shares_source, market_cap, p_fcf, fcf0,
        discount_rate, terminal_growth, implied_growth, bracket_status, reason,
        gap_pp, price_reading, engine_implied_growth, engine_fcf0,
        engine_fcf0_source, base_divergence}`` or ``{"error": str}``.
    """
    try:
        price = _num(price)
        if price is None or price <= 0:
            return {"error": "missing price"}
        shares = _num(row.get("cover_shares"))
        source = "cover"
        if not shares or shares <= 0:
            shares = _num(metrics_shares)
            source = "engine"
        if not shares or shares <= 0:
            shares, source = None, None

        fcf = _num(row.get("fcf"))
        sbc = _num(row.get("sbc"))
        fcf0 = fcf - sbc if fcf is not None and sbc is not None else None
        market_cap = price * shares if shares else None
        unreliable = price_reliable is False
        p_fcf = (
            market_cap / fcf
            if not unreliable and market_cap and fcf is not None and fcf > 0 else None
        )

        base = base_assumptions or {}
        dr = _num(base.get("discount_rate"))
        tg = _num(base.get("terminal_growth"))
        notes = row.get("notes") or []

        reason = None
        if unreliable:
            reason = "unreliable price"
        elif sector_type in ("financial", "reit") or dcf_enabled is False:
            reason = "FCF-DCF not applicable (financial/REIT)"
        elif fcf is None:
            reason = "FCF missing"
        elif sbc is None:
            reason = "SBC missing"
        elif fcf0 <= 0:
            reason = "FCF\u2212SBC \u2264 0"
        elif any("capex_missing" in str(n) for n in notes):
            reason = "CapEx not tagged"
        elif dr is None or tg is None:
            reason = "assumptions missing"
        elif dr < MIN_DISCOUNT_RATE or tg > MAX_TERMINAL_GROWTH:
            reason = "assumptions out of bounds"
        elif shares is None:
            reason = "shares missing"

        if reason is not None:
            implied, status = None, reverse_dcf.STATUS_NO_DATA
        else:
            implied, status = reverse_dcf.implied_growth_with_status(
                price, fcf0, tg, dr, shares, 0.0
            )

        rev_growth = _num(row.get("rev_growth"))
        suffix = YTD_SUFFIX if row.get("rev_growth_basis") == "ytd" else ""
        gap_pp = None
        if status == reverse_dcf.STATUS_ABOVE_BRACKET:
            reading = PRICE_READING_ABOVE_BRACKET
        elif status == reverse_dcf.STATUS_BELOW_BRACKET:
            reading = PRICE_READING_BELOW_BRACKET
        elif status != reverse_dcf.STATUS_OK:
            reading = PRICE_READING_NA
        elif implied is None or rev_growth is None:
            reading = None
        else:
            gap_pp = round(100 * (implied - rev_growth), 6)
            if gap_pp > PRICE_READING_GAP_PP:
                reading = PRICE_READING_UP + suffix
            elif gap_pp < -PRICE_READING_GAP_PP:
                reading = PRICE_READING_DOWN + suffix
            else:
                reading = PRICE_READING_FLAT + suffix

        eng = engine_reverse_dcf if isinstance(engine_reverse_dcf, dict) else {}
        eng_fcf0 = _num(eng.get("fcf0"))
        divergence = None
        if fcf0 is not None and eng_fcf0 and abs(fcf0 / eng_fcf0 - 1) > 0.5:
            divergence = (
                f"TTM FCF-SBC base {fcf0:.0f} differs from the engine's "
                f"fcf0 {eng_fcf0:.0f} by more than 50%"
            )

        return {
            "price": price, "price_as_of": price_as_of,
            "shares": shares, "shares_source": source,
            "market_cap": market_cap, "p_fcf": p_fcf, "fcf0": fcf0,
            "discount_rate": dr, "terminal_growth": tg,
            "implied_growth": implied, "bracket_status": status,
            "reason": reason,
            "gap_pp": gap_pp, "price_reading": reading,
            "engine_implied_growth": _num(eng.get("implied_growth")),
            "engine_fcf0": eng_fcf0,
            "engine_fcf0_source": eng.get("fcf0_source"),
            "base_divergence": divergence,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Valuation overlay failed for %s", row.get("ticker"), exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"}

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_funding_screen(
    tickers: List[str], valuation: bool = False, no_cache: bool = False
) -> List[dict]:
    """Screen ``tickers`` and return the sorted rows (errors last).

    Per ticker: ``resolve_cik`` + ``get_company_facts`` + :func:`screen_facts`,
    each inside try/except so a failure becomes an error row. With
    ``valuation`` a single FRED fetch is made for the run and each non-error
    row gets a ``"valuation"`` block (``{error}`` if its overlay fails).

    Args:
        tickers: Ticker symbols.
        valuation: Add the reverse-DCF overlay.
        no_cache: Bypass on-disk caches.
    """
    from sec_analyzer.fetch.companyfacts import get_company_facts
    from sec_analyzer.fetch.tickers import resolve_cik
    from sec_analyzer.http_client import SecHttpClient

    rows: List[dict] = []
    client = None
    try:
        client = SecHttpClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create HTTP client: %s", exc)

    total = len(tickers)
    for i, raw in enumerate(tickers, 1):
        ticker = str(raw).strip().upper()
        try:
            if client is None:
                raise RuntimeError("HTTP client unavailable")
            cik, name = resolve_cik(ticker, client, no_cache)
            facts = get_company_facts(cik, client, no_cache)
            row = screen_facts(ticker, name, cik, facts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Funding screen failed for %s: %s", ticker, exc,
                           exc_info=not isinstance(exc, ValueError))
            row = _error_row(ticker, str(exc))
        logger.info("Funding screen %d/%d: %s -> %s", i, total, ticker,
                    row.get("group") or row.get("status"))
        rows.append(row)

    if valuation:
        _attach_valuation(rows, no_cache)
    return sort_rows(rows)


def _attach_valuation(rows: List[dict], no_cache: bool) -> None:
    """Add a ``valuation`` block to every non-error row (never raises)."""
    targets = [r for r in rows if r.get("status") != "error"]
    if not targets:
        return
    fred_rate = None
    try:
        # Lazy imports: cli imports this package's siblings (import cycle).
        from sec_analyzer.cli import _fetch_risk_free_asof
        fred_rate = _fetch_risk_free_asof(None, no_cache)
    except Exception:  # noqa: BLE001
        logger.warning("Risk-free rate fetch failed; overlay uses engine defaults", exc_info=True)
    from sec_analyzer.calibrate import run_headless_pipeline

    for row in targets:
        ticker = row["ticker"]
        try:
            # The pipeline helpers print progress lines; keep the table clean.
            with contextlib.redirect_stdout(io.StringIO()):
                result, price, price_as_of, metrics = run_headless_pipeline(
                    ticker, 12, no_cache, None, fred_rate
                )
            valuation = (result or {}).get("valuation") or {}
            base = (valuation.get("assumptions") or {}).get("base") or {}
            metrics = metrics if isinstance(metrics, dict) else {}
            engine_rdcf = dict(valuation.get("reverse_dcf") or {})
            for key in ("fcf0", "fcf0_source"):
                if key not in engine_rdcf and key in valuation:
                    engine_rdcf[key] = valuation[key]
            row["valuation"] = valuation_overlay(
                row, price, price_as_of, base, metrics.get("shares"),
                price_reliable=metrics.get("price_reliable"),
                sector_type=valuation.get("sector_type"),
                dcf_enabled=(valuation.get("dcf") or {}).get("enabled"),
                engine_reverse_dcf=engine_rdcf,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Valuation overlay failed for %s", ticker, exc_info=True)
            row["valuation"] = {"error": str(exc)}


# ---------------------------------------------------------------------------
# Rendering / payload
# ---------------------------------------------------------------------------

def _pct(x, digits: int = 1) -> str:
    v = _num(x)
    return _DASH if v is None else f"{100 * v:.{digits}f}%"


def _fx(x, digits: int = 1, suffix: str = "") -> str:
    v = _num(x)
    return _DASH if v is None else f"{v:.{digits}f}{suffix}"


def _trend(trend) -> str:
    if not trend:
        return _DASH
    return "/".join(
        _DASH if _num(t) is None else f"{100 * t:.0f}" for t in trend
    )


def _text(x) -> str:
    return _DASH if x is None or (isinstance(x, str) and not x.strip()) else str(x)


_BASE_COLS = [
    ("Ticker", 7), ("TTM end", 11), ("Rev gr", 8), ("FCF m", 8), ("FCF-SBC", 8),
    ("Trend", 12), ("ND/EBITDA", 10), ("IntCov", 7), ("Rule40", 8), ("Dilut", 8),
    ("Failed", 0),
]
_VAL_COLS = [("P/FCF (pre-SBC)", 7), ("Impl. FCF-SBC g (5y)", 8), ("Price reading", 0)]


def _row_cells(row: dict, with_valuation: bool) -> List[str]:
    cells = [
        _text(row.get("ticker")),
        _text(row.get("ttm_end")),
        _pct(row.get("rev_growth")),
        _pct(row.get("fcf_margin")),
        _pct(row.get("fcf_sbc_margin")),
        _trend(row.get("fcf_margin_trend")),
        _fx(row.get("net_debt_ebitda"), 2, "x"),
        _fx(row.get("interest_cov"), 1, "x"),
        _fx(row.get("rule_of_40"), 1),
        _pct(row.get("dilution")),
        ", ".join(row.get("fails") or []) or _DASH,
    ]
    if with_valuation:
        v = row.get("valuation") or {}
        if "error" in v:
            cells += [_DASH, _DASH, f"error: {v['error']}"]
        else:
            reading = _text(v.get("price_reading"))
            if v.get("reason") and reading != _DASH:
                reading = f"{reading} ({v['reason']})"
            cells += [
                _fx(v.get("p_fcf"), 1, "x"),
                _pct(v.get("implied_growth")),
                reading,
            ]
    return cells


def _fmt_line(cells: List[str], widths: List[int]) -> str:
    return "  ".join(
        c if w == 0 else c.rjust(w) if i else c.ljust(w)
        for i, (c, w) in enumerate(zip(cells, widths))
    ).rstrip()


def render_table(rows: List[dict], with_valuation: bool = False) -> str:
    """Render the console report: one table per group, errors, disclaimer.

    Missing values print as ``—``, never ``None``/``nan``. The "IntCov" column
    is interest coverage (EBIT / interest).

    Args:
        rows: Rows as returned by :func:`run_funding_screen`.
        with_valuation: Add P/FCF, implied growth and price-reading columns.
    """
    cols = _BASE_COLS + (_VAL_COLS if with_valuation else [])
    headers = [c[0] for c in cols]
    widths = [c[1] for c in cols]
    # Widen columns to fit the data so nothing is misaligned.
    ok_rows = [r for r in rows if r.get("status") != "error"]
    err_rows = [r for r in rows if r.get("status") == "error"]
    cell_cache = {id(r): _row_cells(r, with_valuation) for r in ok_rows}
    for i, w in enumerate(widths):
        if w:
            widths[i] = max([w, len(headers[i])] + [len(cell_cache[id(r)][i]) for r in ok_rows])

    lines: List[str] = []
    for grp in _GROUPS:
        members = [r for r in ok_rows if r.get("group") == grp]
        label = {"A": "all tests passed", "B": "1 test failed", "C": "2+ tests failed"}[grp]
        lines.append(f"Group {grp} ({label}): {len(members)}")
        if members:
            lines.append(_fmt_line(headers, widths))
            lines.append("-" * len(_fmt_line(headers, widths)))
            for r in members:
                lines.append(_fmt_line(cell_cache[id(r)], widths))
        lines.append("")
    ungrouped = [r for r in ok_rows if r.get("group") not in _GROUPS]
    if ungrouped:
        lines.append(f"Ungrouped: {len(ungrouped)}")
        for r in ungrouped:
            lines.append(_fmt_line(cell_cache[id(r)], widths))
        lines.append("")
    if err_rows:
        lines.append(f"Errors: {len(err_rows)}")
        for r in err_rows:
            lines.append(f"  {_text(r.get('ticker'))}: {_text(r.get('error'))}")
        lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def _clean(obj):
    """Recursively replace non-finite floats with ``None`` (valid JSON)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def build_payload(rows: List[dict], universe: str) -> dict:
    """JSON payload ``{meta, rows}``; missing values are ``null``.

    Args:
        rows: Screen rows.
        universe: Universe label (e.g. ``"Nasdaq 100"`` or ``"custom"``).
    """
    return _clean({
        "meta": {
            "universe": universe,
            "generated_from": "SEC companyfacts",
            "disclaimer": DISCLAIMER,
            "thresholds": thresholds(),
        },
        "rows": rows,
    })
