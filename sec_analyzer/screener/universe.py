"""Constituent universes for the swing-trade screener.

Each universe is a bundled, manually-refreshed static CSV rather than a live
fetch: this feature is purely technical (price/volume only, see
SWING_SPEC.md Sec.1) and refreshing an index's membership list is an
out-of-band, infrequent operation (index reconstitutions happen a handful of
times a year), not something worth hitting a network endpoint for on every
scan. Replacing the bundled CSV (``sec_analyzer/data/sp500.csv`` or
``sec_analyzer/data/nasdaq100.csv``) with a freshly generated file is the
supported way to update membership.
"""

import csv
import logging
import os

logger = logging.getLogger(__name__)

#: Directory this module lives in, used to locate the bundled data files
#: regardless of the caller's working directory (mirrors
#: ``report.generator._PACKAGE_DIR``).
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Absolute path to the bundled S&P 500 constituent CSV
#: (``ticker,name,sector,cik``; 503 rows as of writing).
SP500_CSV_PATH = os.path.join(_PACKAGE_DIR, "..", "data", "sp500.csv")
SP500_CSV_PATH = os.path.normpath(SP500_CSV_PATH)

#: Absolute path to the bundled Nasdaq-100 constituent CSV
#: (``ticker,name,sector,cik``; 103 rows as of writing).
NASDAQ100_CSV_PATH = os.path.join(_PACKAGE_DIR, "..", "data", "nasdaq100.csv")
NASDAQ100_CSV_PATH = os.path.normpath(NASDAQ100_CSV_PATH)

#: Default universe code -- every existing call site (``load_universe()``,
#: ``scan_swing()``, ``load_latest_swing_scan()`` with no ``universe``) keeps
#: scanning/returning the S&P 500 unchanged (SWING_SPEC.md Sec.1).
DEFAULT_INDEX = "SP500"

#: code -> (display label, csv path). Ordered as shown in the UI selector.
UNIVERSES = {
    "SP500": ("S&P 500", SP500_CSV_PATH),
    "NDX": ("Nasdaq 100", NASDAQ100_CSV_PATH),
}

#: Aliases accepted by :func:`normalize_index`, case-insensitively, in
#: addition to the canonical keys of :data:`UNIVERSES` themselves.
_INDEX_ALIASES = {
    "NASDAQ100": "NDX",
    "NASDAQ-100": "NDX",
    "NDX100": "NDX",
}

#: Minimum number of comma-separated fields a data row must have to be kept
#: (``ticker,name,sector,cik``); anything shorter is a blank/malformed line
#: and is skipped rather than raising.
_MIN_FIELDS = 4

#: Dotted class-share tickers whose price-layer symbol uses a dash instead of
#: a dot (Stooq's ``brk-b.us``, yfinance's ``BRK-B``). Handled generically in
#: :func:`price_symbol` by replacing ``.`` with ``-``, so this constant only
#: documents the convention -- no lookup table is needed.


def universe_label(index: "str | None") -> str:
    """Display label for an index code.

    Args:
        index: A universe code, ideally already normalized via
            :func:`normalize_index`.

    Returns:
        The human-readable label (``"SP500" -> "S&P 500"``). Unknown codes
        are returned unchanged rather than raising, since this is
        display-only.
    """
    entry = UNIVERSES.get(index)
    return entry[0] if entry else (index or DEFAULT_INDEX)


def normalize_index(index: "str | None") -> str:
    """Case-insensitively resolve an index code to a key of :data:`UNIVERSES`.

    Args:
        index: A raw index code, possibly from a query param or CLI flag --
            may be ``None``, blank, aliased, or garbage.

    Returns:
        A key of :data:`UNIVERSES`. Falls back to :data:`DEFAULT_INDEX` for
        ``None``/blank/unrecognized input (so a bad query param degrades to
        the S&P 500 rather than erroring). Accepts the aliases
        ``"NASDAQ100"``/``"NASDAQ-100"``/``"NDX100"`` for ``"NDX"``.
    """
    code = (index or "").strip().upper()
    if not code:
        return DEFAULT_INDEX
    if code in UNIVERSES:
        return code
    return _INDEX_ALIASES.get(code, DEFAULT_INDEX)


def load_universe(index: str = DEFAULT_INDEX, path: "str | None" = None) -> "list[dict]":
    """Read a bundled constituent CSV.

    Args:
        index: Universe code, resolved via :func:`normalize_index`. Defaults
            to :data:`DEFAULT_INDEX` so existing callers keep scanning the
            S&P 500 unchanged.
        path: Override path to the CSV (mainly for tests). Takes precedence
            over ``index`` when given.

    Returns:
        Rows as ``{"ticker", "name", "sector", "cik"}``, de-duplicated by
        ticker (first occurrence wins), sorted ascending by ticker
        (deterministic order). Blank lines and lines with fewer than
        :data:`_MIN_FIELDS` fields are skipped.

    Raises:
        OSError: If the file is missing or unreadable -- this is a
            packaging error, not a runtime data issue, so it's allowed to
            propagate rather than being swallowed (mirrors
            ``report.generator._load_template``).
    """
    if path is not None:
        csv_path = path
    else:
        resolved = normalize_index(index)
        csv_path = UNIVERSES[resolved][1]

    seen: "dict[str, dict]" = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            if len([v for v in row.values() if v is not None]) < _MIN_FIELDS:
                continue
            if ticker in seen:
                continue
            seen[ticker] = {
                "ticker": ticker,
                "name": (row.get("name") or "").strip() or None,
                "sector": (row.get("sector") or "").strip() or None,
                "cik": (row.get("cik") or "").strip() or None,
            }
    return [seen[t] for t in sorted(seen)]


def price_symbol(ticker: "str | None") -> str:
    """Map a constituent ticker to the symbol the price layer understands.

    Dotted class shares (e.g. ``"BRK.B"``, ``"BF.B"``) become dash-separated
    (``"BRK-B"``, ``"BF-B"``), which is what both Stooq (``brk-b.us``) and
    yfinance expect. Everything else is returned upper-cased and stripped,
    unchanged.

    Args:
        ticker: A constituent ticker as stored in the universe CSV.

    Returns:
        The price-layer symbol. Empty string for ``None``/blank input
        (never raises).
    """
    if not ticker:
        return ""
    return ticker.strip().upper().replace(".", "-")
