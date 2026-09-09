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
from collections import OrderedDict
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

#: Relative gap between the as-reported ``Revenue`` series and the
#: financial-filer ``NetRevenue`` series (measured against the NET figure)
#: above which the as-reported basis is rejected. Strictly above triggers.
_NET_REVENUE_DIVERGENCE_THRESHOLD = 0.05


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


def normalize_facts(facts_json: dict, years: int = 12, as_of=None) -> dict:
    """Normalize a raw SEC companyfacts document into tidy time series.

    Args:
        facts_json: The parsed companyfacts JSON document (as returned by
            ``get_company_facts``), or any dict with the same shape.
        years: Number of most-recent distinct fiscal years to retain in the
            annual bucket. The quarterly bucket retains a comparable window
            of ``years * 4 + 1`` most recent periods.
        as_of: Optional point-in-time cutoff (``datetime.date`` or ISO
            ``"YYYY-MM-DD"`` string). When set, only facts with
            ``filed <= as_of`` survive, so the dedup step yields the value
            as it was known on that date (restatements filed later are
            invisible). ``filed == as_of`` counts as knowable. ``None``
            leaves behavior unchanged.

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
    as_of_str = as_of.isoformat() if hasattr(as_of, "isoformat") else as_of

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

        # Point-in-time cutoff: drop any fact not yet filed as of the cutoff
        # date, so the dedup below picks the latest filing that existed then.
        if as_of_str:
            records = [
                r for r in records if r.get("filed") and r["filed"] <= as_of_str
            ]
            if not records:
                annual[concept] = None
                quarterly[concept] = None
                matched_tags[concept] = None
                missing.append(concept)
                continue

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

    # Financial-filer revenue basis (SPEC.md Sec.19). Runs LAST, on the same
    # windowed series every downstream consumer sees, so a swap cannot be
    # partially undone by the windowing above.
    revenue_basis = _apply_net_revenue_basis(annual, quarterly, matched_tags, missing)

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
        "revenue_basis": revenue_basis,
    }


def _records_to_fy_map(records: Optional[List[dict]]) -> Dict[int, float]:
    """``{fiscal_year: value}`` from a raw annual record list."""
    series: Dict[int, float] = {}
    for record in records or []:
        fy = record.get("fy")
        value = record.get("value")
        if fy is not None and value is not None:
            series[fy] = value
    return series


def _format_usd_tr(value: float) -> str:
    """Format a USD amount for a user-facing note (decimal-comma grouping is
    kept unchanged per this translation pass's rule to preserve existing
    number formatting)."""
    if abs(value) >= 1e9:
        scaled, suffix = value / 1e9, "B$"
    elif abs(value) >= 1e6:
        scaled, suffix = value / 1e6, "M$"
    else:
        scaled, suffix = value, "$"
    return f"{scaled:,.2f} {suffix}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _empty_revenue_basis() -> dict:
    """The no-swap ``revenue_basis`` block (SPEC.md Sec.19)."""
    return {
        "basis": "as_reported",
        "swapped": False,
        "max_divergence": None,
        "divergent_fys": [],
        "dropped_fys": [],
        "rejected_annual": None,
        "rejected_tags": None,
        "gross_annual": None,
        "note": None,
    }


def _apply_net_revenue_basis(
    annual: Dict[str, Optional[List[dict]]],
    quarterly: Dict[str, Optional[List[dict]]],
    matched_tags: Dict[str, Optional[List[str]]],
    missing: List[str],
) -> dict:
    """Prefer a financial filer's net-revenue top line over the ASC-606 slice.

    ``CONCEPTS["Revenue"]`` prefers
    ``RevenueFromContractWithCustomerExcludingAssessedTax``, which for a bank
    or lender carries only the contract-fee slice of revenue rather than the
    income statement's top line -- SoFi's FY2025 contract revenue is $0.62B
    against a $3.61B total net revenue. Every revenue-derived figure
    downstream (margins, growth, P/S, the multiples history) would then be
    computed off a number ~5.8x too small.

    When the ``NetRevenue`` concept (``RevenuesNetOfInterestExpense``, a
    financial-filer-only tag) contradicts the as-reported series by more than
    ``_NET_REVENUE_DIVERGENCE_THRESHOLD`` in any overlapping fiscal year, the
    ``Revenue`` series is replaced -- in both the annual and quarterly buckets
    -- by a copy of the ``NetRevenue`` series, and the rejection is recorded
    in the returned metadata rather than applied silently. Divergence is
    measured against the NET figure and in absolute value, so it catches an
    under-statement (the SoFi case) as well as the gross-revenue
    over-statement an aggregator "sales" field would produce.

    Mutates ``annual``/``quarterly``/``matched_tags``/``missing`` in place on
    a swap. Never raises: any unexpected shape degrades to the no-swap
    ``basis: "as_reported"`` block with the inputs left untouched. See
    ``sec_analyzer/valuation/SPEC.md`` Sec.19.

    Returns:
        The ``revenue_basis`` metadata block (SPEC.md Sec.19).
    """
    result = _empty_revenue_basis()
    try:
        net_records = annual.get("NetRevenue")
        net = _records_to_fy_map(net_records)

        # Gross wedge is informational only -- never a revenue basis, never a
        # ratio input. Reported so the reader can see how much interest
        # expense the net figure nets out.
        interest = _records_to_fy_map(annual.get("InterestIncome"))
        noninterest = _records_to_fy_map(annual.get("NoninterestIncome"))
        gross = {
            fy: interest[fy] + noninterest[fy]
            for fy in sorted(set(interest) & set(noninterest))
        }
        result["gross_annual"] = gross or None

        if not net:
            # Every non-financial filer takes this path: the tag does not
            # exist for them, so behavior is unchanged by construction.
            return result

        reported_records = annual.get("Revenue")
        reported = _records_to_fy_map(reported_records)

        # A zero net figure cannot serve as a divergence denominator.
        overlap = sorted(fy for fy in net if fy in reported and net[fy])
        divergence = {
            fy: abs(reported[fy] - net[fy]) / abs(net[fy]) for fy in overlap
        }
        max_divergence = max(divergence.values()) if divergence else None

        result["max_divergence"] = (
            round(max_divergence, 4) if max_divergence is not None else None
        )
        result["divergent_fys"] = [
            fy for fy in overlap
            if divergence[fy] > _NET_REVENUE_DIVERGENCE_THRESHOLD
        ]

        swap = (not overlap) or (
            max_divergence is not None
            and max_divergence > _NET_REVENUE_DIVERGENCE_THRESHOLD
        )
        if not swap:
            # The two tags agree within tolerance; the as-reported series
            # stands and nothing is recorded as rejected.
            return result

        # Years the as-reported series covered but NetRevenue does not are
        # dropped rather than spliced in: a shorter single-basis series beats
        # a longer one that silently mixes two revenue definitions.
        result["dropped_fys"] = sorted(fy for fy in reported if fy not in net)
        result["rejected_annual"] = reported or None
        result["rejected_tags"] = matched_tags.get("Revenue")

        annual["Revenue"] = [dict(record) for record in net_records or []] or None
        net_quarterly = quarterly.get("NetRevenue")
        # No mixed basis in the quarterly bucket either: with no quarterly
        # net-revenue rows the rejected series is cleared, not kept.
        quarterly["Revenue"] = [dict(record) for record in net_quarterly or []] or None
        matched_tags["Revenue"] = matched_tags.get("NetRevenue")
        if "Revenue" in missing:
            missing.remove("Revenue")

        worst_fy = max(divergence, key=lambda fy: divergence[fy]) if divergence else None
        result["basis"] = "net_revenue"
        result["swapped"] = True
        if worst_fy is not None:
            gap_pct = f"{divergence[worst_fy] * 100:.1f}".replace(".", ",")
            result["note"] = (
                f"Revenue basis corrected: for FY{worst_fy}, the reported revenue tag "
                f"shows {_format_usd_tr(reported[worst_fy])} while net revenue (interest "
                f"expense deducted) is {_format_usd_tr(net[worst_fy])}; the gap is "
                f"%{gap_pct}. For financial institutions, the correct basis is net "
                "revenue; all revenue-derived ratios were computed on net revenue."
            )
        else:
            result["note"] = (
                "Revenue basis corrected: since the reported revenue tag contained no "
                "usable data, net revenue (interest expense deducted) was used instead. "
                "For financial institutions, the correct basis is net revenue; all "
                "revenue-derived ratios were computed on net revenue."
            )
        return result
    except Exception:
        logger.warning(
            "Net-revenue basis reconciliation failed; keeping the as-reported "
            "Revenue series.", exc_info=True,
        )
        return _empty_revenue_basis()


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


#: A quarterly flow record whose span exceeds this (days) is a year-to-date
#: cumulative figure (H1/9M), not a single quarter, and must be differenced.
_QUARTER_MAX_SPAN_DAYS = 100


def _fiscal_year_end_month(normalized: dict) -> Optional[int]:
    """The month a filer's fiscal year ends in, from its ANNUAL records.

    Taken as the most common ``period_end`` month across every annual
    concept, ties broken toward the most recent record. ``None`` when the
    document carries no annual records at all. See
    :func:`_quarter_fiscal_year` for why this is needed (SPEC.md Sec.24a).
    """
    counts: Dict[int, int] = {}
    newest: Dict[int, str] = {}
    for records in (normalized.get("annual") or {}).values():
        for record in records or []:
            period_end = record.get("period_end")
            if not period_end or len(period_end) < 7:
                continue
            try:
                month = int(period_end[5:7])
            except (ValueError, TypeError):
                continue
            counts[month] = counts.get(month, 0) + 1
            if period_end > newest.get(month, ""):
                newest[month] = period_end
    if not counts:
        return None
    return max(counts, key=lambda m: (counts[m], newest.get(m, "")))


def _quarter_fiscal_year(period_end: Optional[str], fy_end_month: Optional[int]) -> Optional[int]:
    """Fiscal year for a QUARTER, anchored on the filer's fiscal-year end.

    :func:`_fiscal_year` labels a period by the calendar year of its end
    date. That is correct for an annual record (whose end month IS the
    fiscal-year end) but wrong for a quarter whenever the fiscal year does
    not end in December: Micron's year ends in late August, so its Q1 (ending
    in November) falls in the NEXT calendar year from its own fiscal year's
    label, and grouping by calendar year mixes it in with the previous
    fiscal year's Q2/Q3 -- which then corrupts the Q4 derivation in
    :func:`to_quarterly_series` (SPEC.md Sec.24a).

    With ``M = fy_end_month``, a quarter ending in month ``m`` of calendar
    year ``Y`` belongs to fiscal year ``Y`` when ``m <= M``, else ``Y + 1``.
    A December fiscal-year end (``M = 12``) leaves every quarter on its
    calendar year, so the overwhelmingly common case is unchanged.

    Known limitation: a 52/53-week filer whose year-end drifts across a month
    boundary can still be off by one bucket. This is a large improvement over
    the calendar rule, not a calendar-exact reconstruction.

    Falls back to :func:`_fiscal_year` when ``fy_end_month`` is unknown.
    """
    if fy_end_month is None:
        return _fiscal_year(period_end)
    if not period_end or len(period_end) < 7:
        return None
    try:
        year = int(period_end[:4])
        month = int(period_end[5:7])
    except (ValueError, TypeError):
        return None
    return year if month <= fy_end_month else year + 1


def to_quarterly_series(normalized: dict, concept: str) -> List[dict]:
    """Return a true *single-quarter* series for ``concept``, ascending.

    The quarterly bucket in ``normalized`` can mix genuine quarter-only rows
    (span ~90 days) with year-to-date cumulative rows (H1 ~180d, 9M ~270d),
    because a 10-Q sometimes reports only the YTD figure for a flow item
    (cash-flow lines especially). This reconstructs the per-quarter values:

    * **Instant/balance concepts** (not in ``FLOW_CONCEPTS``) are point-in-time
      and returned as-is.
    * **Flow concepts** are grouped by fiscal year (calendar year of
      ``period_end``, matching :func:`_fiscal_year`). Within each year, records
      are walked ascending while a running sum tracks the cumulative total from
      the fiscal-year start: a quarter-like row (span ``<=``
      :data:`_QUARTER_MAX_SPAN_DAYS`, or an unknown span) is taken directly; a
      YTD row is differenced against the running cumulative
      (``quarter = ytd - running_sum``). When three quarters of a fiscal year
      are present and an annual value exists, **Q4 is derived** as
      ``annual - (Q1+Q2+Q3)``.

    Each element is ``{"period_end": str, "value": float, "derived": bool}``
    (``derived`` marks a value obtained by differencing/subtraction rather than
    reported directly). Returns ``[]`` when the concept is missing or has no
    usable quarterly rows. Pure and never raises.
    """
    records = (normalized.get("quarterly") or {}).get(concept)
    if not records:
        return []

    fy_end_month = _fiscal_year_end_month(normalized)
    rows: List[dict] = []
    for r in records:
        pe = r.get("period_end")
        val = r.get("value")
        if pe is None or val is None:
            continue
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        # SPEC.md Sec.24a: re-derive the fiscal year from the filer's own
        # fiscal-year end rather than trusting the record's calendar-year
        # label, which mis-buckets quarters for any non-December year end.
        fy = _quarter_fiscal_year(pe, fy_end_month)
        if fy is None:
            fy = r.get("fy") if r.get("fy") is not None else _fiscal_year(pe)
        rows.append({"period_end": pe, "value": value, "span": _span_days(r), "fy": fy})

    if not rows:
        return []
    rows.sort(key=lambda x: x["period_end"])

    if concept not in FLOW_CONCEPTS:
        # Instant/balance concept: each row is already a point-in-time value.
        return [{"period_end": r["period_end"], "value": r["value"], "derived": False} for r in rows]

    # Annual values (keyed by fiscal year) for Q4 derivation.
    annual = to_annual_series(normalized, concept)
    annual_period_end: Dict[int, str] = {}
    for rec in (normalized.get("annual") or {}).get(concept) or []:
        fy = rec.get("fy")
        pe = rec.get("period_end")
        if fy is not None and pe is not None and fy not in annual_period_end:
            annual_period_end[fy] = pe

    groups: "OrderedDict[Optional[int], List[dict]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["fy"], []).append(r)

    out: List[dict] = []
    for fy, grp in groups.items():
        grp.sort(key=lambda x: x["period_end"])
        running_sum = 0.0
        emitted = 0
        for r in grp:
            span = r["span"]
            if span is None or span <= _QUARTER_MAX_SPAN_DAYS:
                quarter = r["value"]
                out.append({"period_end": r["period_end"], "value": quarter, "derived": False})
                running_sum += quarter
            else:
                quarter = r["value"] - running_sum
                out.append({"period_end": r["period_end"], "value": quarter, "derived": True})
                running_sum = r["value"]
            emitted += 1

        # Derive Q4 = annual - sum(first three quarters) when a fiscal year has
        # exactly its first three quarters reported and an annual value exists.
        if fy is not None and emitted == 3 and fy in annual:
            q4 = annual[fy] - running_sum
            q4_pe = annual_period_end.get(fy)
            if q4_pe is not None:
                out.append({"period_end": q4_pe, "value": q4, "derived": True})

    out.sort(key=lambda x: x["period_end"])
    return out


def quarterly_ratio_series(normalized: dict, numerator_concept: str, denominator_concept: str) -> List[dict]:
    """Quarterly ``numerator/denominator`` ratio (percent) series, aligned by
    quarter, ascending.

    Generalizes the numerator/``Revenue`` margin-ratio shape (originally
    hardcoded in ``sec_analyzer.signals.momentum._margin_series``) to an
    arbitrary denominator, so the same machinery covers margins (denominator
    ``Revenue``) and ROE (denominator ``StockholdersEquity``) alike. Both
    series come from :func:`to_quarterly_series`, so a flow numerator (e.g.
    ``NetIncome``) is reconstructed into true single-quarter values while an
    instant/balance denominator (e.g. ``StockholdersEquity``) is used as its
    point-in-time, end-of-quarter value -- matching the annual ratio
    convention in ``sec_analyzer.normalize.ratios`` (end-of-period equity,
    not an average).

    Args:
        normalized: The dict returned by :func:`normalize_facts`.
        numerator_concept: Canonical concept name for the ratio's numerator
            (e.g. ``"NetIncome"``, ``"GrossProfit"``).
        denominator_concept: Canonical concept name for the ratio's
            denominator (e.g. ``"Revenue"``, ``"StockholdersEquity"``).

    Returns:
        Ascending ``[{"period_end": str, "value": float}, ...]``, ``value``
        being the ratio as a percent (e.g. ``23.4`` for 23.4%), rounded to 2
        decimals. A quarter is only included when both series have a value
        for that ``period_end`` and the denominator is strictly positive (a
        zero/negative denominator makes the ratio undefined or meaningless,
        e.g. negative equity); ``[]`` if either series is missing entirely.
        Pure and never raises on well-formed input.
    """
    numerator = to_quarterly_series(normalized, numerator_concept)
    denominator = to_quarterly_series(normalized, denominator_concept)
    denom_by_pe = {q["period_end"]: q["value"] for q in denominator if q.get("period_end") is not None}
    out: List[dict] = []
    for q in numerator:
        pe = q.get("period_end")
        num_v = q.get("value")
        if pe is None or num_v is None or pe not in denom_by_pe:
            continue
        denom_v = denom_by_pe[pe]
        if denom_v is None or denom_v <= 0:
            continue
        out.append({"period_end": pe, "value": round(float(num_v) / float(denom_v) * 100.0, 2)})
    return out


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
