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
_ERP_HISTORY_FILENAME = "erp_history.csv"
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
    """Convert raw ``multiples.csv`` rows into
    ``{"industry","pe","ps","pfcf","growth","peg","beta"[,"capex_sales"]}``.

    Rows without a usable ``industry`` value are skipped; missing/malformed
    ``pe``/``ps``/``pfcf`` columns become ``None`` on that row rather than
    dropping the whole row. ``growth`` (expected multi-year growth, a decimal
    fraction e.g. ``0.15`` for 15%) and ``peg`` are OPTIONAL columns used
    only for the sector-median PEG comparison (VALUATION.md Sec.7); both
    default to ``None`` when absent, so older two-/four-column CSVs keep
    working unchanged. ``unlevered_beta`` (Damodaran's sector unlevered/asset
    beta, a plain number e.g. ``1.5``) is likewise OPTIONAL and feeds the CAPM
    cost-of-equity discount rate (:mod:`sec_analyzer.valuation.capm`); it
    parses into the ``"beta"`` key and defaults to ``None`` when the column is
    absent. ``capex_sales`` (Damodaran's sector Cap Ex/Sales, a plain decimal
    fraction e.g. ``0.045`` for 4.5% of revenue) is likewise OPTIONAL and
    feeds the sector-derived maintenance-CapEx floor
    (:func:`sec_analyzer.valuation.engine._maintenance_adjusted_margin`).
    Unlike the other optional columns, the ``"capex_sales"`` KEY ITSELF is
    only added to the row when the column parses to a usable value -- it's
    simply absent (rather than present-with-``None``) on rows/CSVs that
    lack it, so ``row.get("capex_sales")`` still degrades to ``None`` for any
    caller, while callers/tests that compare the parsed row against an exact
    dict literal predating this column are unaffected.
    """
    parsed = []
    for row in rows:
        industry = (row.get("industry") or "").strip()
        if not industry:
            continue
        parsed_row = {
            "industry": industry,
            "pe": _to_float(row, "pe"),
            "ps": _to_float(row, "ps"),
            "pfcf": _to_float(row, "pfcf"),
            "growth": _to_float(row, "growth"),
            "peg": _to_float(row, "peg"),
            "beta": _to_float(row, "unlevered_beta"),
        }
        capex_sales = _to_float(row, "capex_sales")
        if capex_sales is not None:
            parsed_row["capex_sales"] = capex_sales
        parsed.append(parsed_row)
    return parsed


def _find_us_row(rows: List[dict]) -> Optional[dict]:
    """Return the ``region == "US"`` row from raw ``erp.csv`` rows, or ``None``."""
    for row in rows:
        region = (row.get("region") or "").strip().upper()
        if region == _ERP_REGION:
            return row
    return None


def _parse_erp(rows: List[dict]) -> Optional[float]:
    """Find the ``region == "US"`` row in raw ``erp.csv`` rows and return its ``erp``."""
    us_row = _find_us_row(rows)
    return _to_float(us_row, "erp") if us_row is not None else None


def _parse_risk_free(rows: List[dict]) -> Optional[float]:
    """Return the US ``risk_free`` value from raw ``erp.csv`` rows, or ``None``.

    Optional column (older ``region,erp`` files lack it): the risk-free rate,
    stored as a percentage number consistent with ``erp`` (e.g. ``4.20`` for
    4.2%). Consumed by :mod:`sec_analyzer.valuation.capm` as the CAPM
    intercept.
    """
    us_row = _find_us_row(rows)
    return _to_float(us_row, "risk_free") if us_row is not None else None


def _load_erp_history(path: str) -> Optional[Dict[int, dict]]:
    """Read ``erp_history.csv`` into ``{year: {"erp": float|None, "risk_free": float|None}}``.

    Columns: ``year, erp[, risk_free]`` (percentage numbers, like ``erp.csv``;
    ``risk_free`` optional). Rows without a parseable integer ``year`` are
    skipped. First occurrence of a given year wins (deterministic file order).
    Returns ``None`` if the file is missing/unreadable/empty.
    """
    rows = _read_csv_rows(path)
    if not rows:
        return None
    history: Dict[int, dict] = {}
    for row in rows:
        try:
            year = int(str(row.get("year")).strip())
        except (TypeError, ValueError):
            continue
        if year in history:
            continue
        history[year] = {
            "erp": _to_float(row, "erp"),
            "risk_free": _to_float(row, "risk_free"),
        }
    return history or None


def _find_year_subdir(dir_path: str, as_of_year: int) -> Tuple[Optional[int], Optional[str]]:
    """Find the nearest past-year ``data/damodaran/{YEAR}/`` snapshot subfolder.

    Scans ``dir_path`` for subdirectories whose name is a 4-digit year and
    returns ``(year, path)`` for the largest such year ``<= as_of_year`` (the
    vintage that was current as of that date). Returns ``(None, None)`` when no
    qualifying subfolder exists. Never raises.
    """
    try:
        entries = os.listdir(dir_path)
    except OSError:
        return None, None
    best_year: Optional[int] = None
    for name in entries:
        if len(name) != 4 or not name.isdigit():
            continue
        if not os.path.isdir(os.path.join(dir_path, name)):
            continue
        year = int(name)
        if year <= as_of_year and (best_year is None or year > best_year):
            best_year = year
    if best_year is None:
        return None, None
    return best_year, os.path.join(dir_path, str(best_year))


def load_sector_data(
    dir_path: Optional[str],
    as_of=None,
    fred_rate: Optional[dict] = None,
) -> Optional[dict]:
    """Load Damodaran multiples/ERP reference CSVs from ``dir_path``.

    Args:
        dir_path: Directory expected to contain ``multiples.csv`` and/or
            ``erp.csv`` (typically ``Config.DAMODARAN_DIR``).
        as_of: Optional point-in-time date (``datetime.date`` or ISO string).
            When ``None`` (the default) this returns the exact current-value
            dict, bit-for-bit unchanged. When set, a ``"macro_asof"``
            provenance block is added and historical sources are preferred:
            * Multiples/betas: a ``data/damodaran/{YEAR}/multiples.csv`` snapshot
              subfolder (nearest year on/before ``as_of``) if present, else the
              current ``multiples.csv`` with an "anakronik çarpan/beta" warning.
            * ERP: per-year snapshot ``erp.csv`` -> ``erp_history.csv`` row for
              ``as_of.year`` -> current ``erp.csv`` (with an anachronism warning
              on the last fallback).
        fred_rate: Optional ``{"value_pct": float, ...}`` dict (from
            :func:`sec_analyzer.fetch.fred.get_risk_free_asof`) supplying the
            historical risk-free rate. Only consulted when ``as_of`` is set.
            Risk-free precedence: ``fred_rate`` -> per-year snapshot
            ``erp.csv`` -> ``erp_history.csv`` row's ``risk_free`` -> current
            ``erp.csv`` (with an anachronism warning on the last fallback).

    Returns:
        ``{"multiples": [...] or None, "erp": float or None, "risk_free":
        float or None}``, or ``None`` if ``dir_path`` doesn't exist or
        neither file yielded anything usable. ``erp``/``risk_free`` are
        percentage numbers (e.g. ``4.23`` for 4.23%). ``capex_sales`` is
        present per-row only when that row's CSV data carried a usable Cap
        Ex/Sales value (see :func:`_parse_multiples_rows`). When ``as_of`` is
        set the dict also carries ``"macro_asof": {"as_of","erp_source",
        "risk_free_source","multiples_source"[,"warnings"]}`` (Turkish source
        strings, surfaced in the report/notes; ``warnings`` lists anachronism
        notes when a current snapshot had to substitute for a missing
        historical one). Never raises; every missing piece is logged.
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
    current_erp = _parse_erp(erp_rows) if erp_rows is not None else None
    current_risk_free = _parse_risk_free(erp_rows) if erp_rows is not None else None

    if as_of is None:
        if not parsed_multiples and current_erp is None and current_risk_free is None:
            return None
        return {
            "multiples": parsed_multiples or None,
            "erp": current_erp,
            "risk_free": current_risk_free,
        }

    # --- Point-in-time (as-of) macro resolution ---
    as_of_year = as_of.year if hasattr(as_of, "year") else int(str(as_of)[:4])
    as_of_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
    warnings: List[str] = []

    # Per-year snapshot subfolder (data/damodaran/{YEAR}/): the nearest vintage
    # on/before as_of. When present it supplies the historical multiples/betas
    # AND (via its own erp.csv) an ERP/risk-free source, so sector multiples
    # can be point-in-time instead of the current static snapshot.
    hist_year, hist_dir = _find_year_subdir(dir_path, as_of_year)
    hist_multiples = None
    hist_erp = hist_rf = None
    if hist_dir is not None:
        hist_multiples_rows = _read_csv_rows(os.path.join(hist_dir, _MULTIPLES_FILENAME))
        hist_multiples = (
            _parse_multiples_rows(hist_multiples_rows) if hist_multiples_rows is not None else None
        )
        hist_erp_rows = _read_csv_rows(os.path.join(hist_dir, _ERP_FILENAME))
        if hist_erp_rows is not None:
            hist_erp = _parse_erp(hist_erp_rows)
            hist_rf = _parse_risk_free(hist_erp_rows)

    history = _load_erp_history(os.path.join(dir_path, _ERP_HISTORY_FILENAME))
    hist_row = history.get(as_of_year) if history else None

    # Multiples/betas: per-year snapshot -> current top-level (anachronistic).
    if hist_multiples:
        used_multiples = hist_multiples
        multiples_source = f"data/damodaran/{hist_year}/multiples.csv"
    else:
        used_multiples = parsed_multiples
        multiples_source = "multiples.csv (güncel snapshot — anakronik)"
        if parsed_multiples:
            warnings.append(
                f"Anakronik çarpan/beta: {as_of_year} için tarihsel Damodaran "
                "snapshot'ı yok; güncel multiples.csv kullanıldı."
            )
            logger.warning(
                "damodaran: no per-year snapshot for as_of year %s; using current "
                "multiples.csv (anachronistic sector multiples/betas).", as_of_year,
            )

    # ERP: per-year snapshot erp.csv -> erp_history.csv row -> current erp.csv.
    if hist_erp is not None:
        erp_value = hist_erp
        erp_source = f"data/damodaran/{hist_year}/erp.csv"
    elif hist_row is not None and hist_row.get("erp") is not None:
        erp_value = hist_row["erp"]
        erp_source = f"erp_history.csv ({as_of_year})"
    else:
        erp_value = current_erp
        erp_source = "erp.csv (güncel değer)"
        if current_erp is not None:
            warnings.append(
                f"Anakronik ERP: {as_of_year} için tarihsel ERP yok; güncel "
                "erp.csv değeri kullanıldı."
            )
            logger.warning(
                "damodaran: no historical ERP for as_of year %s; using current "
                "erp.csv value (anachronistic ERP).", as_of_year,
            )

    # Risk-free: FRED -> per-year snapshot erp.csv -> erp_history row -> current erp.csv.
    fred_value = fred_rate.get("value_pct") if isinstance(fred_rate, dict) else None
    if fred_value is not None:
        risk_free_value = fred_value
        rf_date = fred_rate.get("date")
        rf_series = fred_rate.get("series") or "FRED"
        risk_free_source = f"{rf_series} ({rf_date})" if rf_date else str(rf_series)
    elif hist_rf is not None:
        risk_free_value = hist_rf
        risk_free_source = f"data/damodaran/{hist_year}/erp.csv"
    elif hist_row is not None and hist_row.get("risk_free") is not None:
        risk_free_value = hist_row["risk_free"]
        risk_free_source = f"erp_history.csv ({as_of_year})"
    else:
        risk_free_value = current_risk_free
        risk_free_source = "erp.csv (güncel değer)"
        if current_risk_free is not None:
            warnings.append(
                "Anakronik risk-free: FRED DGS10 alınamadı ve tarihsel değer yok; "
                "güncel erp.csv risk-free değeri kullanıldı."
            )
            logger.warning(
                "damodaran: no historical/FRED risk-free for as_of %s; using current "
                "erp.csv value (anachronistic risk-free).", as_of_iso,
            )

    if not used_multiples and erp_value is None and risk_free_value is None:
        return None

    macro_asof = {
        "as_of": as_of_iso,
        "erp_source": erp_source,
        "risk_free_source": risk_free_source,
        "multiples_source": multiples_source,
    }
    if warnings:
        macro_asof["warnings"] = warnings

    return {
        "multiples": used_multiples or None,
        "erp": erp_value,
        "risk_free": risk_free_value,
        "macro_asof": macro_asof,
    }


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
    # Consumer-staples block (beverages/tobacco), most-specific-first: KO-style
    # "BOTTLED & CANNED SOFT DRINKS & CARBONATED WATERS" (SIC 2086) must
    # resolve to the soft-drink row before the generic "beverages" catch-all
    # below, and alcoholic-beverage keywords must not be swept into it either.
    ("bottled & canned soft drinks", "Beverage (Soft)"),
    ("malt beverages", "Beverage (Alcoholic)"),
    ("wines", "Beverage (Alcoholic)"),
    ("distilled", "Beverage (Alcoholic)"),
    ("brewer", "Beverage (Alcoholic)"),  # substring covers "brewery"/"breweries"/"brewers".
    ("soft drinks", "Beverage (Soft)"),
    ("beverages", "Beverage (Soft)"),  # generic catch-all: default ambiguous "beverages" to soft-drink.
    ("cigarettes", "Tobacco"),
    ("tobacco", "Tobacco"),
    ("soap", "Household Products"),  # SIC 2840 "SOAP, DETERGENT, CLEANING PREPARATIONS, PERFUME, COSMETICS".
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
    """Project a parsed ``multiples.csv`` row to the public result shape.

    ``growth``/``peg`` are included but are ``None`` unless the reference CSV
    carried those optional columns (VALUATION.md Sec.7 sector-PEG comparison).
    ``beta`` (the sector unlevered beta) is likewise ``None`` unless the CSV
    carried an ``unlevered_beta`` column (CAPM cost of equity, see
    :mod:`sec_analyzer.valuation.capm`). ``capex_sales`` (the sector Cap
    Ex/Sales ratio, sector-derived maintenance-CapEx floor, see
    :func:`sec_analyzer.valuation.engine._maintenance_adjusted_margin`) is
    always retrievable via ``.get("capex_sales")`` (``None`` when the CSV
    lacked the column), but -- unlike ``growth``/``peg``/``beta`` -- the key
    itself is only present in the returned dict when the row actually
    carried a usable value, so exact-dict comparisons against a result
    predating this column stay unaffected.
    """
    result = {
        "industry": row.get("industry"),
        "pe": row.get("pe"),
        "ps": row.get("ps"),
        "pfcf": row.get("pfcf"),
        "growth": row.get("growth"),
        "peg": row.get("peg"),
        "beta": row.get("beta"),
    }
    if row.get("capex_sales") is not None:
        result["capex_sales"] = row.get("capex_sales")
    return result


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
        "pfcf": float|None, "growth": float|None, "peg": float|None,
        "beta": float|None}`` for the best-matching row, plus a
        ``"capex_sales": float`` key when that row's CSV data carried a
        usable Cap Ex/Sales value (absent, not ``None``, otherwise -- use
        ``.get("capex_sales")`` to read it uniformly), or ``None`` if
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
