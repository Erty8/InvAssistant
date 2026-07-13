"""Normalize raw SEC companyfacts JSON into tidy, deduplicated time series.

This module is the core of the ``normalize`` layer. It takes the raw
``companyfacts`` document returned by
``sec_analyzer.fetch.companyfacts.get_company_facts`` -- a deeply nested
dict keyed by taxonomy (``us-gaap``, ``ifrs-full``, ...) and then by XBRL
tag -- and turns it into a small, predictable structure: one list of annual
records and one list of quarterly records per canonical concept (see
``sec_analyzer.normalize.concepts``).

Along the way it has to cope with the messiness of real-world XBRL data:

* The same concept can be reported under different tags across filers or
  even across fiscal years for the same filer (tag fallback, see
  ``concepts.CONCEPTS``).
* The same ``(concept, period_end)`` can appear multiple times because of
  restatements -- later filings correct earlier ones. We keep the value
  from the most recently *filed* row.
* 10-K filings sometimes carry a stray quarter-length fact alongside the
  annual one, and 10-Q filings often carry both a quarter figure and a
  year-to-date figure for flow concepts. We use period length (start/end
  span) heuristics to prefer the figure that actually matches the bucket
  it's being placed in.

This module never raises on missing or malformed data for an individual
concept -- it logs a warning and records the concept in the ``missing``
list instead, since a caller analyzing a real filer should be able to work
with whatever subset of concepts *is* available.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sec_analyzer.normalize.concepts import (
    CONCEPT_UNITS,
    CONCEPTS,
    FLOW_CONCEPTS,
    TAG_TAXONOMY,
)

logger = logging.getLogger(__name__)

#: Date format used throughout SEC XBRL facts for start/end/filed fields.
_DATE_FMT = "%Y-%m-%d"

#: A flow concept's annual period must span roughly a year to be accepted
#: into the annual bucket (guards against a stray quarter reported inside
#: a 10-K).
_ANNUAL_SPAN_DAYS = (350, 380)

#: A flow concept's quarterly period is preferred when its span falls in
#: this range (guards against picking a year-to-date figure that a 10-Q
#: sometimes reports alongside the quarter-only figure).
_QUARTER_SPAN_DAYS = (80, 100)


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse a ``YYYY-MM-DD`` date string, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _DATE_FMT)
    except (ValueError, TypeError):
        return None


def _span_days(record: dict) -> Optional[int]:
    """Return the number of days between a record's ``start`` and
    ``period_end``, or ``None`` if either is missing/unparseable."""
    start = _parse_date(record.get("start"))
    end = _parse_date(record.get("period_end"))
    if start is None or end is None:
        return None
    return (end - start).days


def _fiscal_year(period_end: Optional[str]) -> Optional[int]:
    """Derive the fiscal-year label from a period-end date.

    SEC companyfacts stamps every fact row with the *filing's* ``fy``/``fp``,
    not the period's own. A 10-K therefore reports its comparative
    prior-year columns with the same ``fy`` as the primary year -- e.g.
    Apple's FY2025 10-K tags all three years of Revenue as ``fy=2025``.
    Using that raw value as the fiscal-year label mislabels and duplicates
    years (multiple periods collapsing onto one ``fy``), so we instead
    derive the label from the period-end date.

    Convention: fiscal year == the calendar year of ``period_end``. This is
    exact for filers whose fiscal year ends in the second half of the
    calendar year (the common case, including Apple's late-September
    year-end). Filers with an early-in-the-year fiscal-year end would be
    off by one under this simple rule, but that trade-off is acceptable here
    and keeps the label unambiguous and self-consistent across a filing's
    comparative columns.

    Returns ``None`` if ``period_end`` is missing or malformed.
    """
    if not period_end or len(period_end) < 4:
        return None
    try:
        return int(period_end[:4])
    except (ValueError, TypeError):
        return None


def _extract_concept(
    facts: dict, tag_list: List[str], unit_keys: List[str]
) -> List[Tuple[int, str, str, dict]]:
    """Collect fact rows from ALL present fallback tags for a concept.

    Unlike a "first tag with data wins" strategy, this MERGES rows across
    every tag in ``tag_list`` that is present with usable data. This is
    necessary because a filer can report the same economic concept under
    different us-gaap tags in different fiscal years -- e.g. NVIDIA reports
    Revenue under ``RevenueFromContractWithCustomerExcludingAssessedTax`` for
    older years and switched to ``Revenues`` for recent years, so neither
    tag alone covers all periods.

    ``facts`` is the full ``facts_json["facts"]`` dict, keyed by taxonomy
    (``"us-gaap"``, ``"dei"``, ...). Most tags live under ``us-gaap``, but a
    handful (e.g. the dei cover-page tag ``EntityCommonStockSharesOutstanding``)
    live in a different taxonomy sub-dict -- see ``concepts.TAG_TAXONOMY``.
    Each tag in ``tag_list`` is looked up in its own taxonomy (defaulting to
    ``"us-gaap"`` for tags not listed in ``TAG_TAXONOMY``), so a single
    concept's fallback list can freely mix tags from different taxonomies.

    ``unit_keys`` is the ordered list of acceptable XBRL unit keys for this
    concept (see ``concepts.CONCEPT_UNITS`` -- most concepts are just
    ``["USD"]``, but per-share and share-count concepts use different unit
    keys). For each tag, the FIRST unit key in ``unit_keys`` that has any
    rows under that tag is used; units are not mixed within a single tag.

    Each collected row is tagged with its *priority index* -- the tag's
    position in ``tag_list`` (0 = most preferred) -- and the unit key its
    rows were found under. Downstream deduplication uses the priority index
    so that, when two tags both report the same period, the higher-priority
    tag's row wins. The specific matched tag name and unit travel with each
    row too.

    Returns:
        A list of ``(priority_index, tag, unit, raw_row)`` tuples, empty if
        none of the fallback tags are present with usable data under any of
        ``unit_keys``.
    """
    collected: List[Tuple[int, str, str, dict]] = []
    for priority, tag in enumerate(tag_list):
        taxonomy = TAG_TAXONOMY.get(tag, "us-gaap")
        taxonomy_data = facts.get(taxonomy) or {}
        tag_data = taxonomy_data.get(tag)
        if not tag_data:
            continue
        units = tag_data.get("units") or {}
        rows = None
        matched_unit = None
        for unit_key in unit_keys:
            candidate = units.get(unit_key)
            if candidate:
                rows = candidate
                matched_unit = unit_key
                break
        if not rows:
            continue
        for row in rows:
            collected.append((priority, tag, matched_unit, row))
    return collected


def _build_record(concept: str, tag: str, priority: int, unit: str, row: dict) -> dict:
    """Convert one raw XBRL fact row into a normalized record dict.

    The record's ``fy`` is derived from the period-end date via
    ``_fiscal_year`` -- NOT taken from the fact's raw ``fy`` field. See
    ``_fiscal_year`` for why: SEC stamps comparative prior-year columns in a
    filing with the *filing's* fy, which would mislabel them. The original
    SEC value is preserved as ``reported_fy`` so nothing is lost.

    ``priority`` is the fallback-priority index of the tag this row came from
    (0 = highest-priority). It is kept only for dedup winner selection and is
    not part of the public record contract.

    ``unit`` is the XBRL unit key the row was found under (e.g. ``"USD"``,
    ``"USD/shares"``, ``"shares"`` -- see ``concepts.CONCEPT_UNITS``) and is
    recorded on the record as ``"unit"``.
    """
    period_end = row.get("end")
    return {
        "concept": concept,
        "tag": tag,
        "_priority": priority,
        "period_end": period_end,
        "fy": _fiscal_year(period_end),
        "reported_fy": row.get("fy"),
        "fp": row.get("fp"),
        "form": row.get("form"),
        "value": row.get("val"),
        "filed": row.get("filed"),
        "start": row.get("start"),
        "unit": unit,
    }


def _is_annual_record(record: dict, concept: str) -> bool:
    """Whether ``record`` belongs in the annual bucket.

    Annual records must come from a 10-K with ``fp == "FY"``. For flow
    concepts (income statement / cash flow), we additionally require the
    reported period to span roughly a full year, to filter out a stray
    quarter that sometimes appears inside a 10-K's XBRL facts. If the span
    can't be determined (missing ``start``, or a parse error) we accept
    the record rather than discard good data over a formatting quirk.
    """
    if record.get("form") != "10-K" or record.get("fp") != "FY":
        return False

    if concept in FLOW_CONCEPTS:
        span = _span_days(record)
        if span is None:
            return True
        lo, hi = _ANNUAL_SPAN_DAYS
        return lo <= span <= hi

    return True


def _is_quarterly_record(record: dict) -> bool:
    """Whether ``record`` belongs in the quarterly bucket (any 10-Q row)."""
    return record.get("form") == "10-Q"


def _latest_filed(records: List[dict]) -> dict:
    """Return the record with the most recent ``filed`` date.

    Dates are parsed and compared as ``datetime`` objects when possible;
    records with a missing or unparseable ``filed`` value fall back to a
    plain string comparison (which, for the ISO ``YYYY-MM-DD`` dates SEC
    uses, still orders correctly).
    """

    def key(record: dict) -> Tuple[int, datetime, str]:
        filed = record.get("filed") or ""
        parsed = _parse_date(filed)
        if parsed is not None:
            return (1, parsed, filed)
        return (0, datetime.min, filed)

    return max(records, key=key)


def _select_winner(candidates: List[dict]) -> dict:
    """Pick the single record to keep from rows sharing a ``period_end``.

    Selection order:

    1. Prefer the row from the highest-priority tag (lowest ``_priority``).
       The fallback list in ``CONCEPTS`` is ordered by preference, so when
       two different tags both report the same period we take the preferred
       tag's value.
    2. Within the same tag priority, prefer the latest ``filed`` date
       (restatements supersede earlier filings).
    """
    best_priority = min(r.get("_priority", 0) for r in candidates)
    preferred = [r for r in candidates if r.get("_priority", 0) == best_priority]
    return _latest_filed(preferred)


def _dedup_latest_filed(records: List[dict], concept: str, bucket: str) -> List[dict]:
    """Collapse duplicate ``(concept, period_end)`` records within a bucket.

    Three kinds of duplication are handled:

    1. Cross-tag overlap: the same period reported under more than one
       fallback tag (see ``_extract_concept``). The higher-priority tag
       wins (see ``_select_winner``).
    2. Restatements: the same period reported in more than one filing. Among
       rows from the same tag, the row with the latest ``filed`` date wins.
    3. Ambiguous flow-concept spans within the quarterly bucket: a 10-Q can
       carry both a quarter-only figure and a year-to-date figure ending on
       the same date. Before picking a winner, we narrow the candidates to
       those whose span looks like a single quarter (``_QUARTER_SPAN_DAYS``);
       if none qualify, we fall back to the shortest available span as the
       best approximation of "one quarter".
    """
    groups: Dict[Optional[str], List[dict]] = {}
    for record in records:
        groups.setdefault(record.get("period_end"), []).append(record)

    result: List[dict] = []
    for period_end, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        candidates = group
        if bucket == "quarterly" and concept in FLOW_CONCEPTS:
            spans = [(_span_days(r), r) for r in group]
            lo, hi = _QUARTER_SPAN_DAYS
            quarter_like = [r for span, r in spans if span is not None and lo <= span <= hi]
            if quarter_like:
                candidates = quarter_like
            else:
                known = [(span, r) for span, r in spans if span is not None]
                if known:
                    shortest = min(span for span, _ in known)
                    candidates = [r for span, r in known if span == shortest]

        winner = _select_winner(candidates)
        if len(group) > 1:
            logger.debug(
                "Deduplicated %d rows for %s @ %s (bucket=%s): kept tag=%s filed=%s val=%s",
                len(group), concept, period_end, bucket,
                winner.get("tag"), winner.get("filed"), winner.get("value"),
            )
        result.append(winner)

    return result


def _limit_annual_years(records: List[dict], years: int) -> List[dict]:
    """Keep only the most recent ``years`` distinct fiscal years.

    ``records`` must already be sorted by ``period_end`` descending. Rows
    whose fiscal year has already been counted are kept regardless (this
    only happens if a filer has more than one annual record per fy, which
    dedup should normally prevent), but once ``years`` distinct fiscal
    years have been seen, older rows are dropped.
    """
    kept: List[dict] = []
    seen_fys: set = set()
    for record in records:
        fy = record.get("fy")
        if fy in seen_fys:
            kept.append(record)
            continue
        if len(seen_fys) >= years:
            break
        seen_fys.add(fy)
        kept.append(record)
    return kept


def normalize_facts(facts_json: dict, years: int = 5) -> dict:
    """Normalize a raw SEC companyfacts document into tidy time series.

    Args:
        facts_json: The parsed companyfacts JSON document (as returned by
            ``get_company_facts``), or any dict with the same shape.
        years: Number of most-recent distinct fiscal years to retain in the
            annual bucket. The quarterly bucket retains a comparable window
            of ``years * 4 + 1`` most recent periods.

    Returns:
        A dict of the form::

            {
              "cik": <int or str or None>,
              "entity_name": <str or None>,
              "currency": "USD",
              "annual": {"<concept>": [record, ...] or None, ...},
              "quarterly": {"<concept>": [record, ...] or None, ...},
              "missing": [<concept names with no contributing tag>, ...],
              "matched_tags": {"<concept>": [<tag>, ...] or None, ...},
            }

        ``matched_tags[concept]`` is the list of us-gaap tags that actually
        contributed at least one surviving record for that concept, ordered
        by fallback priority (most preferred first). It is a list because a
        filer can split one concept across several tags over time (see
        ``_extract_concept``); it is ``None`` for concepts that produced no
        usable records.

        where each ``record`` is::

            {
              "concept": str, "tag": str, "period_end": str,
              "fy": int,            # derived from period_end (see _fiscal_year)
              "reported_fy": int,   # the raw fy SEC stamped on the fact
              "fp": str, "form": str, "value": float,
              "filed": str, "start": str or None,
              "unit": str,   # XBRL unit key, e.g. "USD", "USD/shares", "shares"
            }

    This function never raises because a concept is missing or malformed;
    such concepts are simply reported via ``missing`` and set to ``None``.
    """
    entity_name = facts_json.get("entityName")
    cik = facts_json.get("cik")
    facts = facts_json.get("facts") or {}
    usgaap = facts.get("us-gaap") or {}

    if not usgaap:
        if facts.get("ifrs-full"):
            logger.warning(
                "CIK %s (%s) reports only the 'ifrs-full' taxonomy (typical of "
                "foreign private issuers filing Form 20-F). us-gaap concepts "
                "cannot be extracted for this filer; all concepts will be "
                "reported as missing.",
                cik, entity_name,
            )
        else:
            logger.warning(
                "CIK %s (%s) has no 'us-gaap' facts in its companyfacts "
                "document; all concepts will be reported as missing.",
                cik, entity_name,
            )

    annual: Dict[str, Optional[List[dict]]] = {}
    quarterly: Dict[str, Optional[List[dict]]] = {}
    matched_tags: Dict[str, Optional[List[str]]] = {}
    missing: List[str] = []

    for concept, tag_list in CONCEPTS.items():
        # Collect rows from ALL present fallback tags (merged across tags),
        # each carrying its fallback-priority index and matched unit key.
        unit_keys = CONCEPT_UNITS.get(concept, ["USD"])
        collected = _extract_concept(facts, tag_list, unit_keys)

        if not collected:
            annual[concept] = None
            quarterly[concept] = None
            matched_tags[concept] = None
            missing.append(concept)
            continue

        records = [
            _build_record(concept, tag, priority, unit, row)
            for priority, tag, unit, row in collected
        ]

        annual_records = [r for r in records if _is_annual_record(r, concept)]
        quarterly_records = [r for r in records if _is_quarterly_record(r)]

        # Dedup by period_end: higher-priority tag wins, then latest filed.
        annual_records = _dedup_latest_filed(annual_records, concept, bucket="annual")
        quarterly_records = _dedup_latest_filed(quarterly_records, concept, bucket="quarterly")

        annual_records.sort(key=lambda r: r["period_end"] or "", reverse=True)
        quarterly_records.sort(key=lambda r: r["period_end"] or "", reverse=True)

        annual_records = _limit_annual_years(annual_records, years)
        quarterly_records = quarterly_records[: years * 4 + 1]

        # Which tags actually contributed a surviving record, ordered by
        # fallback priority (most preferred first).
        contributing: Dict[str, int] = {}
        for record in annual_records + quarterly_records:
            tag = record.get("tag")
            if tag is not None and tag not in contributing:
                contributing[tag] = record.get("_priority", 0)
        contributing_tags = [
            tag for tag, _ in sorted(contributing.items(), key=lambda kv: kv[1])
        ]

        # The priority marker is internal-only; drop it from public records.
        for record in annual_records + quarterly_records:
            record.pop("_priority", None)

        if not contributing_tags:
            # Fallback tags were present, but nothing survived the
            # annual/quarterly filters (e.g. only non-USD or non-10-K/10-Q
            # rows). Treat the concept as missing rather than half-present.
            logger.info(
                "Concept %r had tag data for CIK %s but produced no usable "
                "annual or quarterly records; marking missing.", concept, cik,
            )
            annual[concept] = None
            quarterly[concept] = None
            matched_tags[concept] = None
            missing.append(concept)
            continue

        matched_tags[concept] = contributing_tags
        annual[concept] = annual_records or None
        quarterly[concept] = quarterly_records or None

    # Global fiscal-year window: a concept whose data simply STOPS years ago
    # (e.g. a filer switched to a different tag we don't know, or genuinely
    # stopped reporting the item) must not drag decade-old columns into the
    # output next to the entity's current years. Window every concept to the
    # entity-wide most recent `years` fiscal years; concepts with nothing
    # inside the window become missing.
    global_max_fy = max(
        (r["fy"] for recs in annual.values() if recs for r in recs
         if r.get("fy") is not None),
        default=None,
    )
    if global_max_fy is not None:
        min_fy = global_max_fy - years + 1
        for concept in list(annual.keys()):
            recs = annual[concept]
            if not recs:
                continue
            windowed = [r for r in recs if (r.get("fy") or 0) >= min_fy]
            if len(windowed) != len(recs):
                dropped = len(recs) - len(windowed)
                logger.info(
                    "Concept %r: dropped %d stale annual record(s) outside "
                    "the FY%d-FY%d window for CIK %s.",
                    concept, dropped, min_fy, global_max_fy, cik,
                )
            annual[concept] = windowed or None
            if not windowed and not quarterly.get(concept):
                matched_tags[concept] = None
                if concept not in missing:
                    missing.append(concept)

    if missing:
        logger.warning(
            "CIK %s (%s): no usable us-gaap data found for concepts: %s",
            cik, entity_name, ", ".join(missing),
        )

    return {
        "cik": cik,
        "entity_name": entity_name,
        "currency": "USD",
        "annual": annual,
        "quarterly": quarterly,
        "missing": missing,
        "matched_tags": matched_tags,
    }


def to_annual_series(normalized: dict, concept: str) -> Dict[int, float]:
    """Return ``{fiscal_year: value}`` for ``concept`` from the annual bucket.

    Returns an empty dict if the concept is missing, has no annual data, or
    individual records lack a usable ``fy``/``value`` pair.
    """
    records = (normalized.get("annual") or {}).get(concept)
    if not records:
        return {}

    series: Dict[int, float] = {}
    for record in records:
        fy = record.get("fy")
        value = record.get("value")
        if fy is not None and value is not None:
            series[fy] = value
    return series


def latest_annual_value(normalized: dict, concept: str) -> Optional[float]:
    """Return the most recent annual value for ``concept``, or ``None``.

    Relies on ``normalized["annual"][concept]`` already being sorted by
    ``period_end`` descending (as produced by ``normalize_facts``).
    """
    records = (normalized.get("annual") or {}).get(concept)
    if not records:
        return None
    for record in records:
        if record.get("value") is not None:
            return record.get("value")
    return None


def format_table(normalized: dict) -> str:
    """Render a compact, human-readable text table of the annual concepts.

    Fiscal years are columns (most recent first); concepts are rows. Each
    cell shows the raw value alongside a millions-scaled view for quick
    reading, e.g. ``96,995,000,000 (97,0.0M)``. All figures in this main
    table are in the ``normalized["currency"]`` unit (USD).

    Concepts reported under a non-USD unit (currently ``EPS`` and
    ``SharesOutstanding`` -- see ``concepts.CONCEPT_UNITS``) don't belong in
    a USD table, so they're rendered in a separate "Per-share / share
    counts" section instead: ``EPS`` as a plain number with 2 decimal
    places, ``SharesOutstanding`` scaled to millions.

    Intended for terminal/log output rather than machine consumption --
    callers that need structured numbers should use ``to_annual_series``
    or read ``normalized["annual"]`` directly.
    """
    entity_name = normalized.get("entity_name") or "Unknown entity"
    cik = normalized.get("cik")
    currency = normalized.get("currency", "USD")
    annual = normalized.get("annual") or {}

    fiscal_years: set = set()
    for records in annual.values():
        if records:
            for record in records:
                if record.get("fy") is not None:
                    fiscal_years.add(record["fy"])

    if not fiscal_years:
        logger.info("format_table: no annual data available for CIK %s (%s)", cik, entity_name)
        return f"{entity_name} (CIK {cik}): no annual data available."

    fiscal_years = sorted(fiscal_years, reverse=True)
    concepts = list(CONCEPTS.keys())
    monetary_concepts = [c for c in concepts if c not in CONCEPT_UNITS]
    per_share_concepts = [c for c in concepts if c in CONCEPT_UNITS]
    series_by_concept = {c: to_annual_series(normalized, c) for c in concepts}

    label_width = max(len(c) for c in concepts) + 2
    col_width = 22

    lines = [
        f"{entity_name} (CIK {cik}) -- annual figures in {currency}, raw and millions",
        " " * label_width + "".join(f"FY{fy}".rjust(col_width) for fy in fiscal_years),
    ]
    lines.append("-" * len(lines[-1]))

    for concept in monetary_concepts:
        series = series_by_concept[concept]
        cells = []
        for fy in fiscal_years:
            value = series.get(fy)
            if value is None:
                cell = "n/a"
            else:
                cell = f"{value:,.0f} ({value / 1_000_000:,.1f}M)"
            cells.append(cell.rjust(col_width))
        lines.append(concept.ljust(label_width) + "".join(cells))

    if per_share_concepts:
        lines.append("")
        lines.append(f"{entity_name} (CIK {cik}) -- per-share / share counts")
        lines.append(" " * label_width + "".join(f"FY{fy}".rjust(col_width) for fy in fiscal_years))
        lines.append("-" * len(lines[-1]))

        for concept in per_share_concepts:
            series = series_by_concept[concept]
            cells = []
            for fy in fiscal_years:
                value = series.get(fy)
                if value is None:
                    cell = "n/a"
                elif concept == "EPS":
                    cell = f"{value:,.2f}"
                else:
                    cell = f"{value / 1_000_000:,.1f}M"
                cells.append(cell.rjust(col_width))
            lines.append(concept.ljust(label_width) + "".join(cells))

    return "\n".join(lines)
