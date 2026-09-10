"""Download and cache SEC Form 4 (insider ownership) transaction data.

Form 4 is the filing officers, directors, and 10%+ owners must submit within
two business days of a transaction in their company's securities. Unlike the
XBRL facts and submissions metadata this project already caches, Form 4 has
no structured JSON endpoint -- the machine-readable payload is a small
per-filing XML document (``ownershipDocument``) linked from the same
``submissions`` document this project fetches once per run for the earnings
catalyst (:mod:`sec_analyzer.fetch.filings`) and 8-K events
(:mod:`sec_analyzer.signals.events`), in the parallel
``filings.recent.{form,filingDate,accessionNumber,primaryDocument}`` arrays.

This module walks that history for ``4``/``4/A`` filings, downloads and
parses each filing's raw ownership XML, and returns a flat list of structured
transactions for :mod:`sec_analyzer.signals.insider` to aggregate into a
buy/sell signal. Each Form 4 document is immutable once filed, so the
per-filing cache below carries no TTL -- a cache hit never needs to be
revalidated against SEC.

Each transaction also carries ``shares_owned_after`` (parsed from
``<postTransactionAmounts><sharesOwnedFollowingTransaction><value>``), the
number of shares this person owned immediately after the transaction. That
is the denominator the signal layer needs to judge materiality: a handful of
dollars sold means nothing on its own, but "sold 60% of the position" does.

Like :mod:`sec_analyzer.fetch.earnings`, this module is never fatal: a
network failure, a malformed filing, or missing submissions data all result
in a logged warning and a smaller (or empty) result, never a raised
exception.
"""

import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Form types that carry insider ownership transactions (the base Form 4 and
#: its amendment).
INSIDER_FORMS = ("4", "4/A")

#: Default lookback window: only Form 4s filed within this many days of
#: ``today`` are considered "recent" enough to scan.
DEFAULT_LOOKBACK_DAYS = 180

#: Default cap on how many qualifying filings are fetched per call, bounding
#: the network cost of a single scan (SEC's fair-access throttle is already
#: handled by ``client``).
DEFAULT_MAX_FILINGS = 40

#: Raw ownership-XML document URL template. ``accession`` must have its
#: dashes stripped; ``cik`` is the unpadded integer form.
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

#: Date format used throughout SEC submissions data for filing dates.
_DATE_FMT = "%Y-%m-%d"


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse a ``YYYY-MM-DD`` date string, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _DATE_FMT).date()
    except (ValueError, TypeError):
        return None


def _cache_dir() -> str:
    """Return the per-filing Form 4 cache directory, creating it if missing."""
    path = os.path.join(Config.RAW_DIR, "form4")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(accession_stripped: str) -> str:
    """Return the on-disk cache path for one filing's parsed transactions."""
    return os.path.join(_cache_dir(), f"form4_{accession_stripped}.json")


def _load_cache(path: str) -> Optional[dict]:
    """Load a previously cached parsed-filing JSON file.

    Returns ``None`` (rather than raising) if the file is missing, unreadable,
    or not valid JSON -- a corrupt cache must not be fatal, it should just
    trigger a re-fetch.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a corrupt cache file must not be fatal
        logger.warning("Failed to load Form 4 cache at %s; will re-fetch.", path, exc_info=True)
        return None


def _write_cache(path: str, data: dict) -> None:
    """Serialize ``data`` to ``path`` as JSON. Failures are logged, not raised."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:  # noqa: BLE001 - a cache-write failure must not lose the fetched data
        logger.warning("Failed to write Form 4 cache at %s", path, exc_info=True)


def _strip_namespace(tag: str) -> str:
    """Strip an XML namespace prefix (``{uri}local``) down to the local tag name."""
    if not tag:
        return tag
    idx = tag.rfind("}")
    return tag[idx + 1 :] if idx != -1 else tag


def _find(elem: Optional[ET.Element], *path: str) -> Optional[ET.Element]:
    """Walk ``path`` (local tag names, namespace-agnostic) from ``elem``.

    Returns ``None`` at the first missing step instead of raising, so every
    caller can chain lookups defensively.
    """
    current = elem
    for step in path:
        if current is None:
            return None
        current = next(
            (child for child in current if _strip_namespace(child.tag) == step), None
        )
    return current


def _text(elem: Optional[ET.Element]) -> Optional[str]:
    """Return ``elem.text`` stripped, or ``None`` for a missing/blank element."""
    if elem is None or elem.text is None:
        return None
    stripped = elem.text.strip()
    return stripped or None


def _bool_text(elem: Optional[ET.Element]) -> bool:
    """Normalize a Form 4 boolean-ish text value (``"1"``/``"0"``/``"true"``/
    ``"false"``, case-insensitive) to a Python bool. Missing/unparseable is
    ``False``."""
    value = _text(elem)
    if value is None:
        return False
    return value.strip().lower() in ("1", "true")


def _coerce_float(value: Optional[str]) -> Optional[float]:
    """Best-effort ``float(value)``; ``None`` for anything unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_reporting_owner(root: ET.Element) -> Tuple[Optional[str], bool, bool, bool, Optional[str]]:
    """Extract attribution fields from the FIRST ``reportingOwner`` element.

    Returns ``(person, is_director, is_officer, is_ten_percent, officer_title)``.
    All-defaults (``None``/``False``) when no reporting owner is present.
    """
    owner = _find(root, "reportingOwner")
    if owner is None:
        return None, False, False, False, None

    person = _text(_find(owner, "reportingOwnerId", "rptOwnerName"))

    relationship = _find(owner, "reportingOwnerRelationship")
    is_director = _bool_text(_find(relationship, "isDirector"))
    is_officer = _bool_text(_find(relationship, "isOfficer"))
    is_ten_percent = _bool_text(_find(relationship, "isTenPercentOwner"))
    officer_title = _text(_find(relationship, "officerTitle"))

    return person, is_director, is_officer, is_ten_percent, officer_title


def _parse_transaction_row(
    row: ET.Element,
    derivative: bool,
    filed: str,
    accession: str,
    person: Optional[str],
    is_director: bool,
    is_officer: bool,
    is_ten_percent: bool,
    officer_title: Optional[str],
) -> Optional[dict]:
    """Parse one ``nonDerivativeTransaction``/``derivativeTransaction`` row.

    Returns ``None`` when the row carries no ``code`` or no ``date`` --
    nothing classifiable to emit.
    """
    date_text = _text(_find(row, "transactionDate", "value"))
    txn_date = date_text[:10] if date_text else None

    code_text = _text(_find(row, "transactionCoding", "transactionCode"))
    code = code_text.strip().upper() if code_text else None

    if not code or not txn_date:
        return None

    amounts = _find(row, "transactionAmounts")
    shares = _coerce_float(_text(_find(amounts, "transactionShares", "value")))
    price = _coerce_float(_text(_find(amounts, "transactionPricePerShare", "value")))
    direction = _text(_find(amounts, "transactionAcquiredDisposedCode", "value"))

    # The stake-size denominator: how many shares this person owned
    # immediately AFTER this transaction. This is what lets the signal layer
    # ask "what fraction of their position did they let go of?" rather than
    # only "how many dollars moved?" -- see sec_analyzer.signals.insider.
    post_amounts = _find(row, "postTransactionAmounts")
    shares_owned_after = _coerce_float(
        _text(_find(post_amounts, "sharesOwnedFollowingTransaction", "value"))
    )

    return {
        "date": txn_date,
        "filed": filed,
        "person": person,
        "is_director": is_director,
        "is_officer": is_officer,
        "is_ten_percent": is_ten_percent,
        "officer_title": officer_title,
        "code": code,
        "shares": shares,
        "price": price,
        "direction": direction,
        "derivative": derivative,
        "accession": accession,
        "shares_owned_after": shares_owned_after,
    }


def _parse_form4_xml(raw: bytes, filed: str, accession: str) -> List[dict]:
    """Parse one Form 4 ``ownershipDocument`` XML payload into transactions.

    Never raises: malformed XML or an unexpected structure simply yields no
    transactions.
    """
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        logger.debug("Form 4 accession %s: XML parse error.", accession, exc_info=True)
        return []
    except Exception:  # noqa: BLE001 - any other parse failure is non-fatal
        logger.debug("Form 4 accession %s: unexpected parse failure.", accession, exc_info=True)
        return []

    if _strip_namespace(root.tag) != "ownershipDocument":
        logger.debug("Form 4 accession %s: unexpected root tag %r.", accession, root.tag)
        return []

    person, is_director, is_officer, is_ten_percent, officer_title = _parse_reporting_owner(root)

    transactions: List[dict] = []

    non_derivative_table = _find(root, "nonDerivativeTable")
    if non_derivative_table is not None:
        for row in non_derivative_table:
            if _strip_namespace(row.tag) != "nonDerivativeTransaction":
                continue
            txn = _parse_transaction_row(
                row, False, filed, accession, person,
                is_director, is_officer, is_ten_percent, officer_title,
            )
            if txn is not None:
                transactions.append(txn)

    derivative_table = _find(root, "derivativeTable")
    if derivative_table is not None:
        for row in derivative_table:
            if _strip_namespace(row.tag) != "derivativeTransaction":
                continue
            txn = _parse_transaction_row(
                row, True, filed, accession, person,
                is_director, is_officer, is_ten_percent, officer_title,
            )
            if txn is not None:
                transactions.append(txn)

    return transactions


def _fetch_one_filing(
    cik_unpadded: str,
    accession: str,
    primary_document: str,
    filed: str,
    client,
    no_cache: bool,
) -> Optional[List[dict]]:
    """Fetch (or load from cache) and parse the transactions for one filing.

    Returns ``None`` on any failure (network, parse, non-XML primary
    document) -- the caller counts that as a failed filing and continues.
    """
    accession_stripped = accession.replace("-", "") if accession else None
    if not accession_stripped:
        return None

    cache_path = _cache_path(accession_stripped)

    if not no_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached.get("transactions") or []

    document = (primary_document or "").split("/")[-1]
    if not document.lower().endswith(".xml"):
        logger.debug(
            "Form 4 accession %s: primary document %r is not raw XML; skipping.",
            accession, primary_document,
        )
        return None

    url = _DOC_URL.format(cik=cik_unpadded, accession=accession_stripped, document=document)
    try:
        raw = client.get_bytes(url)
    except Exception:  # noqa: BLE001 - a single filing failure must not abort the scan
        logger.warning("Failed to fetch Form 4 document at %s", url, exc_info=True)
        return None

    transactions = _parse_form4_xml(raw, filed, accession)

    _write_cache(cache_path, {"accession": accession, "transactions": transactions})
    return transactions


def get_insider_transactions(
    cik,
    submissions: Optional[dict],
    client,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_filings: int = DEFAULT_MAX_FILINGS,
    no_cache: bool = False,
    today: Optional[date] = None,
) -> Optional[dict]:
    """Fetch and parse a filer's recent Form 4 insider transactions.

    Args:
        cik: The filer's CIK, zero-padded or not (e.g. ``"0000320193"`` or
            ``320193``).
        submissions: The dict returned by
            :func:`sec_analyzer.fetch.companyfacts.get_submissions` (or any
            dict with the same ``filings.recent.{form,filingDate,
            accessionNumber,primaryDocument}`` parallel-array shape). This is
            the same document the analyze pipeline already fetches once per
            run -- no extra submissions request is made here.
        client: :class:`sec_analyzer.http_client.SecHttpClient` used to fetch
            each filing's raw ownership XML.
        lookback_days: Only Form 4s filed within this many days of ``today``
            are scanned.
        max_filings: Cap on the number of qualifying filings fetched
            (newest first), bounding network cost.
        no_cache: When True, bypass the per-filing cache read (a fetch still
            writes/overwrites the cache).
        today: Reference date for the lookback window and point-in-time
            guard; defaults to :meth:`date.today`. Exposed for deterministic
            testing.

    Returns:
        ``None`` when ``submissions`` is missing/malformed or no Form 4
        filings qualify. Otherwise::

            {
              "transactions": [...],   # newest transaction date first
              "filings_scanned": 5,    # filings successfully fetched+parsed
              "filings_failed": 1,     # filings that errored or were unparseable
              "lookback_days": 180,
              "source": "SEC Form 4",
              "truncated": False,      # True when more qualifying filings existed
                                       # than max_filings allowed fetching
            }

        This function never raises.
    """
    try:
        return _get_insider_transactions(
            cik, submissions or {}, client,
            lookback_days=lookback_days,
            max_filings=max_filings,
            no_cache=no_cache,
            today=today or date.today(),
        )
    except Exception:  # noqa: BLE001 - this function must never raise
        logger.exception("get_insider_transactions() failed unexpectedly; returning None.")
        return None


def _get_insider_transactions(
    cik,
    submissions: dict,
    client,
    lookback_days: int,
    max_filings: int,
    no_cache: bool,
    today: date,
) -> Optional[dict]:
    if not submissions:
        return None

    recent = ((submissions.get("filings") or {}).get("recent")) or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []

    candidates = []
    for idx, form in enumerate(forms):
        if form not in INSIDER_FORMS:
            continue

        filed = _parse_date(filing_dates[idx] if idx < len(filing_dates) else None)
        if filed is None:
            continue
        if filed > today:
            # Point-in-time guard: a filing dated after the reference date
            # was not yet public. No-op when today == date.today().
            continue
        if lookback_days > 0 and (today - filed).days > lookback_days:
            continue

        accession = accessions[idx] if idx < len(accessions) else None
        primary_doc = primary_docs[idx] if idx < len(primary_docs) else None
        if not accession:
            continue

        candidates.append((filed, accession, primary_doc))

    if not candidates:
        return None

    # Newest-first, capped at max_filings to bound network cost.
    candidates.sort(key=lambda c: c[0], reverse=True)
    truncated = False
    if max_filings is not None and max_filings >= 0:
        truncated = len(candidates) > max_filings
        candidates = candidates[:max_filings]

    try:
        cik_unpadded = str(int(cik))
    except (TypeError, ValueError):
        logger.warning("get_insider_transactions: unparseable CIK %r", cik)
        return None

    all_transactions: List[dict] = []
    filings_scanned = 0
    filings_failed = 0

    for filed, accession, primary_doc in candidates:
        transactions = _fetch_one_filing(
            cik_unpadded, accession, primary_doc,
            filed.strftime(_DATE_FMT), client, no_cache,
        )
        if transactions is None:
            filings_failed += 1
            continue
        filings_scanned += 1
        all_transactions.extend(transactions)

    all_transactions.sort(key=lambda t: t.get("date") or "", reverse=True)

    return {
        "transactions": all_transactions,
        "filings_scanned": filings_scanned,
        "filings_failed": filings_failed,
        "lookback_days": lookback_days,
        "source": "SEC Form 4",
        "truncated": truncated,
    }
