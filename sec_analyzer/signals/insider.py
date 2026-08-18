"""Aggregate a filer's recent Form 4 insider transactions into a buy/sell signal.

:mod:`sec_analyzer.fetch.insider` turns SEC's raw per-filing ownership XML
into a flat list of transactions; this module is its pure, deterministic
aggregation sibling -- no network, no I/O -- exactly the same split as
:mod:`sec_analyzer.fetch.filings` (fetch/estimate) and
:mod:`sec_analyzer.signals.events` (fetch/classify).

The core judgment call this module encodes: **insider buying is a much
stronger signal than insider selling.** An insider buys stock on the open
market for essentially one reason -- they believe it is undervalued, using
their own money. They sell for many reasons that have nothing to do with a
negative view of the price: portfolio diversification, a scheduled 10b5-1
plan, funding a tax bill, or plain liquidity needs. The empirical literature
(Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser 2003) backs this up:
insider buying predicts subsequent returns, insider selling barely does.
Routine selling is the base rate for any large, long-tenured management
team -- a signal that fires on the base rate is noise, not information.

That asymmetry drives two design choices here:

1. The verdict ladder needs much less evidence to call a buy "meaningful"
   than a sell: a single priced open-market buy already reads as
   ``"ALIM"``, while any amount of dollar-value selling alone is
   deliberately capped at the neutral ``"SATIŞ AĞIRLIKLI"`` -- selling only
   escalates to the bearish ``"YOĞUN SATIŞ"`` when it clears a *stake-size*
   bar (see below), not merely a dollar-value or headcount bar.
2. Dollar value sold is a bad materiality proxy on its own: a $24M sale by
   a director who still holds $2B of stock is trivial, while the same
   dollar amount from someone cashing out most of their position is not.
   :func:`sec_analyzer.fetch.insider._parse_transaction_row` now carries
   ``shares_owned_after`` (from Form 4's
   ``postTransactionAmounts/sharesOwnedFollowingTransaction``), which lets
   this module compute what fraction of a seller's remaining stake they
   actually disposed of. Only when the MEDIAN seller crossed
   :data:`_HEAVY_SELL_STAKE_PCT` of their own position does selling earn
   the strongest bearish label.

Transaction codes that are not an open-market buy/sell (awards, option
exercises, tax withholding, gifts, ...) are compensation mechanics, not a
market view -- they are still surfaced in ``recent`` for transparency, but
never drive the verdict.

**Derivative rows never enter the buy/sell aggregates.** SEC's transaction
codes ``P``/``S`` are valid on the derivative table too (an open-market
purchase/sale of an option or warrant), but a derivative row's ``shares`` is
a *contract* count, not a *common-share* count, priced in different units --
summing it into the common-share tallies (or into ``_stake_details``'s
per-person stake fractions) would silently mix incompatible quantities. All
of ``buy_count``/``sell_count``/``buy_shares``/``sell_shares``/``buy_value``/
``sell_value``/``buyers``/``sellers``/``cluster_buy``/``cluster_sell`` (and
therefore the verdict) are computed from non-derivative transactions only.
Derivative ``P``/``S`` activity is not discarded, though -- it is parsed,
still shown in ``recent`` (flagged ``derivative: True``), and its counts are
reported separately as ``derivative_buy_count``/``derivative_sell_count`` so
a reader can see it exists rather than have it silently vanish from the
totals.
"""

import logging
import statistics
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Date format used throughout SEC submissions/Form 4 data.
_DATE_FMT = "%Y-%m-%d"

#: Default lookback window mirrored from sec_analyzer.fetch.insider.
_DEFAULT_LOOKBACK_DAYS = 180

#: Median fraction of their own remaining stake that selling insiders must
#: dispose of before selling is treated as informational rather than
#: routine. Below this, selling is the base rate (diversification, taxes,
#: scheduled 10b5-1 plans) and must not read as bearish.
_HEAVY_SELL_STAKE_PCT = 25.0

#: SEC Form 3/4/5 transaction code -> (Turkish label, category).
#: Categories drive the signal: only `open_market_buy`/`open_market_sell`
#: carry conviction; awards, option exercises and tax withholding are
#: compensation mechanics, not a view on the price.
_CODE_MAP: Dict[str, Tuple[str, str]] = {
    "P": ("Açık piyasa alımı", "open_market_buy"),
    "S": ("Açık piyasa satışı", "open_market_sell"),
    "A": ("Hisse ödülü/tahsisi", "award"),
    "M": ("Opsiyon/türev kullanımı", "exercise"),
    "F": ("Vergi için hisse mahsubu", "tax"),
    "G": ("Bağış/devir", "gift"),
    "C": ("Türev dönüşümü", "exercise"),
    "X": ("Opsiyon kullanımı", "exercise"),
    "D": ("Şirkete geri devir", "other"),
    "J": ("Diğer edinim/elden çıkarma", "other"),
    "V": ("Gönüllü erken bildirim", "other"),
}


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse a ``YYYY-MM-DD`` date string, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _DATE_FMT).date()
    except (ValueError, TypeError):
        return None


def _classify_code(code: Optional[str]) -> Tuple[str, str]:
    """Return ``(label, category)`` for one transaction code, falling back to
    a generic label/category for an unrecognized code -- never silently
    dropped, same posture as ``events.py::_classify_item``."""
    if not code:
        return ("Bilinmeyen işlem", "other")
    mapped = _CODE_MAP.get(code)
    if mapped is not None:
        return mapped
    return (f"Form 4 kodu {code}", "other")


def _role_label(txn: dict) -> str:
    """Build a Turkish role label from a transaction's attribution flags.

    Precedence: officer with a title -> the title verbatim (an English SEC
    free-text field, kept as filed); officer without a title -> "Yönetici";
    director -> "Yönetim Kurulu Üyesi"; ten-percent owner -> "%10 Ortak";
    nothing set -> "İçeriden". Multiple applicable roles are joined with
    " · " in that order.
    """
    parts: List[str] = []

    officer_title = txn.get("officer_title")
    is_officer = bool(txn.get("is_officer"))
    if is_officer and officer_title:
        parts.append(str(officer_title))
    elif is_officer:
        parts.append("Yönetici")

    if txn.get("is_director"):
        parts.append("Yönetim Kurulu Üyesi")

    if txn.get("is_ten_percent"):
        parts.append("%10 Ortak")

    if not parts:
        return "İçeriden"
    return " · ".join(parts)


def _txn_value(txn: dict) -> Optional[float]:
    """``shares * price`` when both are known numbers, else ``None``."""
    shares = txn.get("shares")
    price = txn.get("price")
    if shares is None or price is None:
        return None
    try:
        return float(shares) * float(price)
    except (TypeError, ValueError):
        return None


def _fmt_money(value: float) -> str:
    """Compact Turkish-locale money string, e.g. ``"$2,5M"`` / ``"$840K"``.

    Uses a comma as the decimal separator (Turkish convention). Values under
    $1,000 render as a rounded whole-dollar amount.
    """
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{sign}${abs_value / 1_000_000:.1f}M".replace(".", ",")
    if abs_value >= 1_000:
        return f"{sign}${abs_value / 1_000:.0f}K"
    return f"{sign}${abs_value:.0f}"


def _fmt_shares(value: float) -> str:
    """Turkish-locale share count with ``.`` as the thousands separator,
    e.g. ``"12.000"``. Does not collide with :func:`_fmt_money`'s ``,``
    decimal separator, so the two can be embedded in the same sentence."""
    return f"{value:,.0f}".replace(",", ".")


def _fmt_pct(value: float) -> str:
    """Turkish-locale percentage, e.g. ``"%3,1"`` (comma decimal separator)."""
    return f"%{value:.1f}".replace(".", ",")


def _stake_details(nd_transactions: List[dict], target_code: str, is_buy: bool) -> List[dict]:
    """Per-person stake disposal/addition detail for one open-market code.

    Groups ``nd_transactions`` (non-derivative rows only -- derivative
    positions don't represent an outright change in beneficial ownership the
    same way) by ``person``, then for each person who has at least one
    ``target_code`` ("P" or "S") transaction computes:

    * the total shares transacted under that code in the window
    * ``shares_owned_after`` from that person's most recent transaction
      (any code, latest ``date``; ties broken by original list position) that
      carries a non-``None`` value -- the freshest available snapshot of
      their position
    * the resulting stake fraction:
        - sells: ``sold / (sold + remaining) * 100`` (``remaining`` already
          excludes the sold shares, so adding them back recovers the
          pre-sale stake as the denominator)
        - buys: ``bought / remaining * 100`` (``remaining`` already
          includes the just-bought shares)

    Returns a list of per-person dicts sorted by the stake percentage
    descending, unknown percentages last. Never raises: a person with no
    parseable ``shares_owned_after`` simply gets ``None`` for the percentage
    rather than being dropped -- the caller still wants to see the
    dollar/share activity, just without a materiality read.
    """
    by_person: Dict[str, List[dict]] = {}
    for txn in nd_transactions:
        person = txn.get("person")
        if not person:
            continue
        by_person.setdefault(person, []).append(txn)

    pct_key = "stake_added_pct" if is_buy else "stake_sold_pct"
    amount_key = "bought_shares" if is_buy else "sold_shares"
    details: List[dict] = []

    for person, txns in by_person.items():
        target_txns = [t for t in txns if t.get("code") == target_code]
        if not target_txns:
            continue

        amount = sum((t.get("shares") or 0.0) for t in target_txns)

        ordered = sorted(txns, key=lambda t: t.get("date") or "", reverse=True)
        remaining = next(
            (t.get("shares_owned_after") for t in ordered if t.get("shares_owned_after") is not None),
            None,
        )

        pct = None
        if is_buy:
            if remaining is not None and remaining > 0:
                pct = round(amount / remaining * 100, 1)
        else:
            if remaining is not None and (amount + remaining) > 0:
                pct = round(amount / (amount + remaining) * 100, 1)

        details.append(
            {
                "person": person,
                amount_key: amount,
                "shares_owned_after": remaining,
                pct_key: pct,
            }
        )

    details.sort(key=lambda d: (d[pct_key] is None, -(d[pct_key]) if d[pct_key] is not None else 0.0))
    return details


def _median_pct(details: List[dict], pct_key: str) -> Optional[float]:
    """Median of the known (non-``None``) percentages in ``details``."""
    known = [d[pct_key] for d in details if d.get(pct_key) is not None]
    if not known:
        return None
    return statistics.median(known)


def detect_insider_activity(
    fetched: Optional[dict],
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    today: Optional[date] = None,
    max_recent: int = 8,
) -> Optional[dict]:
    """Aggregate Form 4 transactions into a deterministic buy/sell signal.

    Args:
        fetched: The dict returned by
            :func:`sec_analyzer.fetch.insider.get_insider_transactions`
            (``{"transactions": [...], "truncated": bool, ...}``), or
            ``None``.
        lookback_days: Only transactions dated within this many days of
            ``today`` are considered.
        today: Reference date for the lookback window and point-in-time
            guard (a transaction dated after ``today`` is excluded --
            correctness for backtests, a no-op when ``today`` is the real
            current date). Defaults to :meth:`date.today`.
        max_recent: Maximum number of transactions kept in the returned
            ``"recent"`` list (most recent first).

    Returns:
        ``None`` when ``fetched`` is falsy or carries no transactions in the
        window. Otherwise a dict -- see module docstring for the verdict
        rationale::

            {
              "window_days": 180,
              "buy_count": 3, "sell_count": 1,
              "buy_shares": 12000.0, "sell_shares": 4000.0,
              "buy_value": 2520000.0, "sell_value": 840000.0,
              "net_value": 1680000.0,
              "buyers": ["Cook Timothy D", ...], "sellers": [...],
              "cluster_buy": True, "cluster_sell": False,
              "compensation_only": False,
              "buyers_detail": [{"person", "bought_shares",
                                  "shares_owned_after", "stake_added_pct"}, ...],
              "sellers_detail": [{"person", "sold_shares",
                                   "shares_owned_after", "stake_sold_pct"}, ...],
              "median_stake_sold_pct": 3.1,   # or None if unknown for everyone
              "derivative_buy_count": 0, "derivative_sell_count": 0,
              "verdict": "GÜÇLÜ ALIM", "severity": "positive",
              "note": "<one Turkish sentence>",
              "recent": [...],            # enriched transaction dicts
              "transaction_count": 12,
              "truncated": False,         # carried through from `fetched`
            }

        Never raises.
    """
    try:
        return _detect_insider_activity(
            fetched or {},
            lookback_days=lookback_days,
            today=today or date.today(),
            max_recent=max_recent,
        )
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("detect_insider_activity() failed unexpectedly; returning None.")
        return None


def _detect_insider_activity(
    fetched: dict,
    lookback_days: int,
    today: date,
    max_recent: int,
) -> Optional[dict]:
    raw_transactions = fetched.get("transactions") or []
    if not raw_transactions:
        return None

    in_window: List[dict] = []
    for txn in raw_transactions:
        txn_date = _parse_date(txn.get("date"))
        if txn_date is None:
            continue
        if txn_date > today:
            # Point-in-time guard: a transaction reported after the
            # reference date was not yet known as of `today`.
            continue
        if lookback_days > 0 and (today - txn_date).days > lookback_days:
            continue
        in_window.append(txn)

    if not in_window:
        return None

    # All headline tallies (and _stake_details) are computed from
    # non-derivative transactions only -- see module docstring. This must
    # happen BEFORE the tally loop so the loop, the buyer/seller sets, the
    # cluster flags, and _stake_details all draw on the same population.
    nd_transactions = [t for t in in_window if not t.get("derivative")]
    d_transactions = [t for t in in_window if t.get("derivative")]

    buy_count = 0
    sell_count = 0
    buy_shares = 0.0
    sell_shares = 0.0
    buy_value = 0.0
    sell_value = 0.0
    buyers = set()
    sellers = set()
    has_compensation_activity = False

    for txn in nd_transactions:
        code = txn.get("code")
        _, category = _classify_code(code)
        shares = txn.get("shares") or 0.0
        value = _txn_value(txn)
        person = txn.get("person")

        if category == "open_market_buy":
            buy_count += 1
            buy_shares += shares
            if value is not None:
                buy_value += value
            if person:
                buyers.add(person)
        elif category == "open_market_sell":
            sell_count += 1
            sell_shares += shares
            if value is not None:
                sell_value += value
            if person:
                sellers.add(person)
        else:
            has_compensation_activity = True

    # Derivative open-market rows are preserved as their own count rather
    # than discarded -- a reader must be able to see they exist, even though
    # contract counts can't be summed into common-share totals.
    derivative_buy_count = sum(1 for t in d_transactions if _classify_code(t.get("code"))[1] == "open_market_buy")
    derivative_sell_count = sum(1 for t in d_transactions if _classify_code(t.get("code"))[1] == "open_market_sell")

    net_value = buy_value - sell_value
    cluster_buy = len(buyers) >= 2
    cluster_sell = len(sellers) >= 2
    compensation_only = buy_count == 0 and sell_count == 0 and has_compensation_activity

    buyers_detail = _stake_details(nd_transactions, "P", is_buy=True)
    sellers_detail = _stake_details(nd_transactions, "S", is_buy=False)
    median_stake_sold_pct = _median_pct(sellers_detail, "stake_sold_pct")

    verdict, severity = _classify_verdict(
        buy_count, sell_count, buy_value, sell_value,
        cluster_buy, cluster_sell, median_stake_sold_pct,
    )

    note = _build_note(
        verdict, lookback_days, buy_count, sell_count, buy_shares, sell_shares,
        buy_value, sell_value, buyers, sellers, cluster_buy, compensation_only,
        median_stake_sold_pct, derivative_buy_count, derivative_sell_count,
    )

    # Enrich the most recent transactions for display.
    ordered = sorted(in_window, key=lambda t: t.get("date") or "", reverse=True)
    recent = []
    for txn in ordered[: max(max_recent, 0)]:
        label, category = _classify_code(txn.get("code"))
        recent.append(
            {
                **txn,
                "label": label,
                "category": category,
                "value": _txn_value(txn),
                "role": _role_label(txn),
            }
        )

    return {
        "window_days": lookback_days,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_shares": buy_shares,
        "sell_shares": sell_shares,
        "buy_value": buy_value,
        "sell_value": sell_value,
        "net_value": net_value,
        "buyers": sorted(buyers),
        "sellers": sorted(sellers),
        "cluster_buy": cluster_buy,
        "cluster_sell": cluster_sell,
        "compensation_only": compensation_only,
        "buyers_detail": buyers_detail,
        "sellers_detail": sellers_detail,
        "median_stake_sold_pct": median_stake_sold_pct,
        "derivative_buy_count": derivative_buy_count,
        "derivative_sell_count": derivative_sell_count,
        "verdict": verdict,
        "severity": severity,
        "note": note,
        "recent": recent,
        "transaction_count": len(in_window),
        "truncated": bool(fetched.get("truncated")),
    }


def _classify_verdict(
    buy_count: int,
    sell_count: int,
    buy_value: float,
    sell_value: float,
    cluster_buy: bool,
    cluster_sell: bool,
    median_stake_sold_pct: Optional[float],
) -> Tuple[str, str]:
    """Deterministic verdict rules, first match wins (see module docstring
    for the buy/sell asymmetry rationale):

    1. >= 2 distinct open-market buyers (cluster_buy) -> "GÜÇLÜ ALIM" / positive.
    2. >= 1 buy and buy_value >= sell_value -> "ALIM" / positive.
    3. >= 1 buy, but outsold by value -> "KARIŞIK" / neutral.
    4. cluster_sell AND a known median seller stake disposal >=
       :data:`_HEAVY_SELL_STAKE_PCT` -> "YOĞUN SATIŞ" / negative. Requires a
       KNOWN median: when no seller's post-transaction holding could be
       parsed, materiality cannot be established, so this must not fire --
       falls through to rule 5 instead of guessing.
    5. >= 1 sell (any amount, any stake fraction) -> "SATIŞ AĞIRLIKLI" /
       neutral, NOT negative -- routine selling is the base rate, not a
       bearish event on its own.
    6. otherwise -> "NÖTR" / neutral.
    """
    if cluster_buy:
        return "GÜÇLÜ ALIM", "positive"
    if buy_count >= 1 and buy_value >= sell_value:
        return "ALIM", "positive"
    if buy_count >= 1:
        return "KARIŞIK", "neutral"
    if cluster_sell and median_stake_sold_pct is not None and median_stake_sold_pct >= _HEAVY_SELL_STAKE_PCT:
        return "YOĞUN SATIŞ", "negative"
    if sell_count >= 1:
        return "SATIŞ AĞIRLIKLI", "neutral"
    return "NÖTR", "neutral"


def _derivative_clause(derivative_buy_count: int, derivative_sell_count: int) -> str:
    """Exclusion-transparency clause appended to ``note`` when derivative
    open-market rows exist. An exclusion the reader can't see is
    indistinguishable from data that was never there, so this is always
    spelled out rather than silently folded into (or dropped from) the
    common-share totals."""
    total = derivative_buy_count + derivative_sell_count
    if total <= 0:
        return ""
    # Turkish nouns don't pluralize after a numeral, so the phrasing is the
    # same for 1 or many.
    return (
        f" Ayrıca {total} türev (opsiyon/varant) açık piyasa işlemi var; "
        f"sözleşme adedi hisse adediyle toplanamayacağı için toplamlara "
        f"dahil edilmedi."
    )


def _build_note(
    verdict: str,
    lookback_days: int,
    buy_count: int,
    sell_count: int,
    buy_shares: float,
    sell_shares: float,
    buy_value: float,
    sell_value: float,
    buyers: set,
    sellers: set,
    cluster_buy: bool,
    compensation_only: bool,
    median_stake_sold_pct: Optional[float],
    derivative_buy_count: int,
    derivative_sell_count: int,
) -> str:
    """One Turkish sentence naming the actual counts (see module docstring).

    Branches on ``verdict`` (rather than re-deriving a condition) so the
    prose always agrees with the verdict :func:`_classify_verdict` chose.
    Appends :func:`_derivative_clause` when derivative open-market rows were
    excluded from the totals.
    """
    derivative_clause = _derivative_clause(derivative_buy_count, derivative_sell_count)

    if compensation_only:
        return "Açık piyasa işlemi yok; yalnızca hisse ödülü/opsiyon hareketleri var." + derivative_clause

    if verdict in ("GÜÇLÜ ALIM", "ALIM"):
        money_clause = f" (≈{_fmt_money(buy_value)})" if buy_value > 0 else ""
        cluster_clause = " — kümelenmiş alım." if cluster_buy else "."
        return (
            f"Son {lookback_days} günde {len(buyers)} farklı yönetici açık "
            f"piyasadan toplam {_fmt_shares(buy_shares)} hisse{money_clause} aldı"
            f"{cluster_clause}"
        ) + derivative_clause

    if verdict == "KARIŞIK":
        return "Hem alım hem satış var; satışlar değerce ağır basıyor." + derivative_clause

    if verdict == "YOĞUN SATIŞ":
        money_clause = f" (≈{_fmt_money(sell_value)})" if sell_value > 0 else ""
        # median_stake_sold_pct is guaranteed known here (rule 4's guard).
        return (
            f"Son {lookback_days} günde {len(sellers)} farklı yönetici açık "
            f"piyasada toplam {_fmt_shares(sell_shares)} hisse{money_clause} sattı; "
            f"satıcılar paylarının medyan {_fmt_pct(median_stake_sold_pct)}'ini "
            f"elden çıkardı — rutin çeşitlendirmenin ötesinde."
        ) + derivative_clause

    if verdict == "SATIŞ AĞIRLIKLI":
        money_clause = f" (≈{_fmt_money(sell_value)})" if sell_value > 0 else ""
        stake_clause = (
            f" medyan olarak paylarının {_fmt_pct(median_stake_sold_pct)}'ini elden çıkardılar;"
            if median_stake_sold_pct is not None
            else ""
        )
        return (
            f"Son {lookback_days} günde {len(sellers)} farklı yönetici açık "
            f"piyasada toplam {_fmt_shares(sell_shares)} hisse{money_clause} sattı;{stake_clause} "
            f"bu seviyedeki satış rutindir (10b5-1 planı / çeşitlendirme) ve tek "
            f"başına olumsuz sinyal sayılmaz."
        ) + derivative_clause

    return "İçeriden kayda değer bir işlem bulunamadı." + derivative_clause


def summarize_insider(activity: Optional[dict]) -> str:
    """Compact one-line Turkish summary for the CLI verdict card's
    "İçeriden:" line.

    Mirrors :func:`sec_analyzer.signals.events.summarize_events`'s shape and
    defensiveness: returns ``"yok"`` for ``None``/empty input, never raises,
    and never renders ``None``/``nan``. Appends ``" (kısmi)"`` when the
    underlying fetch was truncated by ``max_filings`` -- silently presenting
    a partial scan as complete would misstate the evidence.

    Example: ``"SATIŞ AĞIRLIKLI — 0 alım / 13 satış (180g) · net -$24,0M ·
    medyan pay satışı %3,1"``.
    """
    try:
        if not activity:
            return "yok"

        verdict = activity.get("verdict") or "NÖTR"
        buy_count = activity.get("buy_count") or 0
        sell_count = activity.get("sell_count") or 0
        window_days = activity.get("window_days")
        net_value = activity.get("net_value")
        median_stake_sold_pct = activity.get("median_stake_sold_pct")

        window_clause = f" ({window_days}g)" if window_days else ""
        summary = f"{verdict} — {buy_count} alım / {sell_count} satış{window_clause}"

        if isinstance(net_value, (int, float)) and net_value:
            sign = "+" if net_value >= 0 else "-"
            summary += f" · net {sign}{_fmt_money(abs(net_value))}"

        if isinstance(median_stake_sold_pct, (int, float)):
            summary += f" · medyan pay satışı {_fmt_pct(median_stake_sold_pct)}"

        if activity.get("truncated"):
            summary += " (kısmi)"

        return summary
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("summarize_insider() failed unexpectedly; returning 'yok'.")
        return "yok"
