"""Load Aswath Damodaran's public sector-multiple / ERP reference data.

Optional, purely local reference data (no network access): a small pair of
CSV files an operator drops into ``Config.DAMODARAN_DIR`` (see that folder's
own README for the expected format). Everything here is tolerant of a
missing directory, missing files, or missing/malformed columns -- it logs
what's unavailable and returns whatever subset *is* usable rather than
raising, since sector-median context is a nice-to-have enrichment, not a
requirement for the rest of the valuation engine to run.

Expected files, both with a header row (columns documented per-function
below):

* ``multiples.csv`` -- ``industry, pe, ps, pfcf`` (one row per industry,
  median multiples).
* ``erp.csv`` -- ``region, erp`` (only the ``region == "US"`` row is used).
"""

import csv
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MULTIPLES_FILENAME = "multiples.csv"
_ERP_FILENAME = "erp.csv"
_ERP_REGION = "US"


def _read_csv_rows(path: str) -> Optional[List[dict]]:
    """Read a CSV file into a list of ``{column: value}`` string dicts.

    Returns ``None`` (rather than raising) if the file doesn't exist or
    can't be parsed.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error, UnicodeDecodeError):
        logger.warning("damodaran: failed to read %s", path, exc_info=True)
        return None


def _to_float(row: dict, key: str) -> Optional[float]:
    """Parse ``row[key]`` as a float, returning ``None`` on any failure."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _parse_multiples_rows(rows: List[dict]) -> List[dict]:
    """Convert raw ``multiples.csv`` rows into ``{"industry","pe","ps","pfcf"}``.

    Rows without a usable ``industry`` value are skipped; missing/malformed
    ``pe``/``ps``/``pfcf`` columns become ``None`` on that row rather than
    dropping the whole row.
    """
    parsed = []
    for row in rows:
        industry = (row.get("industry") or "").strip()
        if not industry:
            continue
        parsed.append(
            {
                "industry": industry,
                "pe": _to_float(row, "pe"),
                "ps": _to_float(row, "ps"),
                "pfcf": _to_float(row, "pfcf"),
            }
        )
    return parsed


def _parse_erp(rows: List[dict]) -> Optional[float]:
    """Find the ``region == "US"`` row in raw ``erp.csv`` rows and return its ``erp``."""
    for row in rows:
        region = (row.get("region") or "").strip().upper()
        if region == _ERP_REGION:
            return _to_float(row, "erp")
    return None


def load_sector_data(dir_path: Optional[str]) -> Optional[dict]:
    """Load Damodaran multiples/ERP reference CSVs from ``dir_path``.

    Args:
        dir_path: Directory expected to contain ``multiples.csv`` and/or
            ``erp.csv`` (typically ``Config.DAMODARAN_DIR``).

    Returns:
        ``{"multiples": [{"industry","pe","ps","pfcf"}, ...] or None,
        "erp": float or None}``, or ``None`` if ``dir_path`` doesn't exist
        or neither file yielded anything usable. Never raises; every
        missing piece is logged.
    """
    if not dir_path or not os.path.isdir(dir_path):
        logger.info("damodaran: directory %r not found; sector medians unavailable.", dir_path)
        return None

    multiples_rows = _read_csv_rows(os.path.join(dir_path, _MULTIPLES_FILENAME))
    if multiples_rows is None:
        logger.info("damodaran: %s not found or unreadable in %s.", _MULTIPLES_FILENAME, dir_path)

    erp_rows = _read_csv_rows(os.path.join(dir_path, _ERP_FILENAME))
    if erp_rows is None:
        logger.info("damodaran: %s not found or unreadable in %s.", _ERP_FILENAME, dir_path)

    parsed_multiples = _parse_multiples_rows(multiples_rows) if multiples_rows is not None else None
    erp_value = _parse_erp(erp_rows) if erp_rows is not None else None

    if not parsed_multiples and erp_value is None:
        return None

    return {"multiples": parsed_multiples or None, "erp": erp_value}


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Non-distinctive connector/boilerplate words dropped from both the SIC
# description and industry names before scoring the fuzzy fallback. Kept
# tight on purpose -- distinctive words like "energy", "system", "software",
# "equipment" must NOT be added here, or real overlap signal is lost.
_STOPWORDS = frozenset(
    {
        "and", "the", "for", "with", "without", "related", "various",
        "general", "other", "misc", "miscellaneous", "services", "service",
        "svcs", "products", "product", "nec", "inc", "incorporated", "corp",
        "company", "holdings", "group",
    }
)

# Guard marker: skip any industry row whose name normalises to include this,
# in case a future CSV drop reintroduces Damodaran's "Total Market" rows.
_TOTAL_MARKET_MARKER = "total market"

# Explicit SIC-keyword -> exact Damodaran industry name aliases, checked
# BEFORE the fuzzy fallback (primary matching path). Each key is matched by
# substring containment against the *normalized* SIC description (see
# ``_normalize_text``): both sides are lowercased and every run of
# non-alphanumeric characters collapses to a single space, but word order is
# preserved, so adjacency still matters. That's what lets
# ``"radiotelephone communications"`` match SIC 4812 (`"RADIOTELEPHONE
# COMMUNICATIONS"`) without also matching SIC 4813 (`"TELEPHONE
# COMMUNICATIONS (NO RADIOTELEPHONE)"`) -- in the latter, "radiotelephone"
# and "communications" are not adjacent in that order, only "telephone
# communications" is.
#
# Ordered MOST-SPECIFIC-FIRST: the first alias whose key is contained in the
# normalized SIC description wins, so e.g. "semiconductor equipment" must be
# listed before the more generic "semiconductor", and "radiotelephone
# communications" before the more generic "telephone communications".
_ALIASES: List[Tuple[str, str]] = [
    ("semiconductor equipment", "Semiconductor Equip"),
    ("semiconductor", "Semiconductor"),
    ("prepackaged software", "Software (System & Application)"),
    ("computer programming", "Computer Services"),
    ("electronic computer", "Computers/Peripherals"),
    ("biological products", "Drugs (Biotechnology)"),
    ("pharmaceutical", "Drugs (Pharmaceutical)"),
    ("retail variety", "Retail (General)"),
    ("retail grocery", "Retail (Grocery and Food)"),
    ("eating places", "Restaurant/Dining"),
    ("commercial bank", "Banks (Regional)"),
    ("crude petroleum", "Oil/Gas (Production and Exploration)"),
    ("petroleum refining", "Oil/Gas (Integrated)"),
    ("motor vehicle", "Auto & Truck"),
    ("aircraft", "Aerospace/Defense"),
    ("radiotelephone communications", "Telecom (Wireless)"),
    ("telephone communications", "Telecom. Services"),
    ("real estate investment trust", "R.E.I.T."),
    ("advertising", "Advertising"),
]


def _normalize_text(text: Optional[str]) -> str:
    """Lowercase ``text`` and collapse every run of non-alphanumeric chars to a single space.

    E.g. ``"Oil/Gas (Integrated)"`` -> ``"oil gas integrated"``. Word order
    is preserved (this is not tokenization -- see :func:`_tokenize` for
    that), only punctuation/whitespace is normalized. Returns ``""`` for
    falsy input; never raises.
    """
    if not text:
        return ""
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def _tokenize(text: Optional[str]) -> List[str]:
    """Split ``text`` into distinctive, lowercase tokens for fuzzy scoring.

    Applies :func:`_normalize_text`, then drops tokens of length <= 2 and
    the ``_STOPWORDS`` set so that generic shared words (e.g. "related",
    "general", "services") can't manufacture a false match on their own.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return [tok for tok in normalized.split() if len(tok) > 2 and tok not in _STOPWORDS]


def _index_industries(multiples: List[dict]) -> Dict[str, dict]:
    """Build a ``{normalized industry name: row}`` lookup for alias resolution.

    Skips rows with a blank ``industry`` and any row whose normalized name
    contains ``_TOTAL_MARKET_MARKER`` (defensive guard; current data has no
    such rows). First occurrence of a given normalized name wins, matching
    the deterministic file-order tie-break used elsewhere in this module.
    """
    index: Dict[str, dict] = {}
    for row in multiples:
        industry = (row.get("industry") or "").strip()
        if not industry:
            continue
        normalized = _normalize_text(industry)
        if not normalized or _TOTAL_MARKET_MARKER in normalized:
            continue
        index.setdefault(normalized, row)
    return index


def _row_to_result(row: dict) -> dict:
    """Project a parsed ``multiples.csv`` row to the public result shape."""
    return {
        "industry": row.get("industry"),
        "pe": row.get("pe"),
        "ps": row.get("ps"),
        "pfcf": row.get("pfcf"),
    }


def sector_medians(sector_data: Optional[dict], sic_description: Optional[str]) -> Optional[dict]:
    """Match ``sic_description`` to a Damodaran industry row.

    Two-stage matching, in order:

    1. **Alias table** (``_ALIASES``): a curated list of distinctive SIC
       keywords/phrases mapped to an exact Damodaran industry name, checked
       most-specific-first. A key matches when it's a substring of the
       normalized SIC description (see :func:`_normalize_text` -- lowercased,
       punctuation collapsed to spaces, word order preserved). This is the
       primary path and resolves cases the old plain word-overlap scorer got
       wrong or missed (e.g. ``"(pharmaceutical)"`` vs ``"pharmaceutical"``,
       or ``"semiconductors & related devices"`` incorrectly matching "Coal
       & Related Energy" on the generic word "related").
    2. **Fuzzy fallback** (only if no alias matched): tokenizes (see
       :func:`_tokenize`) both the SIC description and each industry name,
       requires at least one shared *distinctive* (non-stopword) token --
       zero distinctive overlap always scores 0, so generic-word false hits
       are impossible -- and gives a bonus to full substring containment of
       one normalized string in the other. Deterministic tie-break: the
       first-encountered row with the best score wins.

    Any industry row whose name normalizes to contain "total market" is
    skipped defensively in both stages.

    Args:
        sector_data: The dict returned by :func:`load_sector_data`, or
            ``None``.
        sic_description: The company's ``sicDescription`` (from SEC
            ``submissions``), or ``None``.

    Returns:
        ``{"industry": str, "pe": float|None, "ps": float|None,
        "pfcf": float|None}`` for the best-matching row, or ``None`` if
        there's no sector data, no description, or no row scores a match.
        Never raises.
    """
    if not sector_data or not sector_data.get("multiples") or not sic_description:
        return None

    multiples = sector_data["multiples"]

    sic_norm = _normalize_text(sic_description)
    if not sic_norm:
        return None

    # 1) Explicit alias table, most-specific-first.
    industry_index = _index_industries(multiples)
    for alias_key, target_industry in _ALIASES:
        alias_norm = _normalize_text(alias_key)
        if alias_norm and alias_norm in sic_norm:
            row = industry_index.get(_normalize_text(target_industry))
            if row is not None:
                return _row_to_result(row)
            logger.warning(
                "damodaran: alias target %r not present in sector data; skipping alias.",
                target_industry,
            )

    # 2) Fuzzy token-overlap fallback.
    sic_tokens = set(_tokenize(sic_description))
    if not sic_tokens:
        return None

    best_row = None
    best_score = 0
    for row in multiples:
        industry = (row.get("industry") or "").strip()
        if not industry:
            continue
        industry_norm = _normalize_text(industry)
        if not industry_norm or _TOTAL_MARKET_MARKER in industry_norm:
            continue

        industry_tokens = set(_tokenize(industry))
        overlap = sic_tokens & industry_tokens
        if not overlap:
            # No shared distinctive token -- never a match, however similar
            # the raw strings might otherwise look.
            continue

        score = len(overlap)
        if industry_norm in sic_norm or sic_norm in industry_norm:
            score += 100  # Substring containment still ranks/ties highest.

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return None
    return _row_to_result(best_row)
