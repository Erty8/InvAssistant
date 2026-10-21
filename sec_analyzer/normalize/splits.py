"""Stock-split detection and share-basis normalization (SPEC.md Sec.29).

SEC companyfacts keeps every filing's own figures. After a stock split, new
filings restate their comparative periods onto the post-split basis, but
facts that were never re-filed keep the pre-split basis:

* the ``dei:EntityCommonStockSharesOutstanding`` cover-page count is never
  restated (each filing reports only its own cover), and
* per-share/share-count facts for periods older than the comparatives a
  post-split filing carries stay pre-split.

Downstream consumers combine these series with split-adjusted prices, so a
10:1 split shows up as a 10x jump in the share count (a false dilution flag)
and as 10x-too-low historical market caps and multiples.

Detection uses only restatements: the SAME ``(tag, start, end)`` weighted-
average share count reported by two filings whose values differ by a common
split ratio. Real share issuance (an offering, a stock-for-stock merger)
never rewrites past periods, so it cannot be mistaken for a split. Every
detected event needs at least :data:`_MIN_CORROBORATING_PERIODS` restated
periods.

Pure and deterministic: no network, no I/O, never raises.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Weighted-average share-count tags used for detection. Their restated
#: values are exact multiples of the originals, unlike EPS (rounded to the
#: cent), so they are the only detection source.
_DETECTION_TAGS = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)

#: Cover-page tag used, together with :data:`_DETECTION_TAGS`, to read one
#: share-count level per filing (see :func:`_filing_share_levels`).
_COVER_TAG = "EntityCommonStockSharesOutstanding"

#: Forward split ratios accepted as a split (new shares per old share).
#: Reverse splits are their reciprocals. Deliberately a short explicit list:
#: a dense set of fractions would let ordinary restatements snap onto one.
_FORWARD_RATIOS = (1.25, 4.0 / 3.0, 1.5, 2.0, 2.5) + tuple(float(n) for n in range(3, 51))

#: A restated/original ratio within this relative distance of an accepted
#: split ratio counts as that ratio.
_RATIO_TOLERANCE = 0.005

#: Minimum number of distinct restated ``(tag, start, end)`` periods that
#: must agree before an event is accepted.
_MIN_CORROBORATING_PERIODS = 2

#: Concept -> how a split factor ``f`` (new shares per old share) converts
#: an old-basis value onto the newest basis: share counts are multiplied by
#: ``f``, per-share amounts divided by it.
SPLIT_SENSITIVE_CONCEPTS = {"SharesOutstanding": "multiply", "EPS": "divide"}


def _snap_ratio(ratio: float) -> Optional[float]:
    """Return the accepted split ratio ``ratio`` is within tolerance of, else ``None``."""
    if not ratio or ratio <= 0 or not math.isfinite(ratio):
        return None
    for forward in _FORWARD_RATIOS:
        for candidate in (forward, 1.0 / forward):
            if abs(ratio - candidate) <= _RATIO_TOLERANCE * candidate:
                return candidate
    return None


def _rows(facts: dict, tag: str, taxonomy: str = "us-gaap") -> List[dict]:
    """Rows of ``tag`` in the ``shares`` unit, or ``[]``."""
    tag_data = (facts.get(taxonomy) or {}).get(tag) or {}
    return list((tag_data.get("units") or {}).get("shares") or [])


def _visible(rows: List[dict], as_of_str: Optional[str]) -> List[dict]:
    """Rows with a usable value and ``filed`` date, cut off at ``as_of_str``."""
    out = []
    for row in rows:
        filed = row.get("filed")
        value = row.get("val")
        if not filed or not isinstance(value, (int, float)):
            continue
        if as_of_str and filed > as_of_str:
            continue
        out.append(row)
    return out


def _candidates(facts: dict, as_of_str: Optional[str]) -> List[dict]:
    """Restatement pairs whose ratio is an accepted split ratio.

    For every ``(tag, start, end)`` period, rows are ordered by ``filed`` and
    each consecutive pair of filing dates is compared. A pair is a candidate
    when both values are positive and ``new / old`` snaps to an accepted
    ratio; the split then happened in the interval ``(old_filed, new_filed]``.
    """
    out: List[dict] = []
    for tag in _DETECTION_TAGS:
        periods: Dict[Tuple[Optional[str], Optional[str]], Dict[str, float]] = {}
        for row in _visible(_rows(facts, tag), as_of_str):
            by_filed = periods.setdefault((row.get("start"), row.get("end")), {})
            # Several rows from the same filing date: keep the first in a
            # deterministic order so the comparison never depends on input order.
            by_filed.setdefault(row["filed"], float(row["val"]))
        for (start, end), by_filed in sorted(periods.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            dates = sorted(by_filed)
            for old_filed, new_filed in zip(dates, dates[1:]):
                old, new = by_filed[old_filed], by_filed[new_filed]
                if old <= 0 or new <= 0:
                    continue
                factor = _snap_ratio(new / old)
                if factor is None or factor == 1.0:
                    continue
                out.append({
                    "factor": factor, "old_filed": old_filed, "new_filed": new_filed,
                    "period": (tag, start, end),
                })
    return out


def _cluster(candidates: List[dict]) -> List[dict]:
    """Merge candidates of one split into events.

    Candidates with the same factor whose ``(old_filed, new_filed]`` intervals
    still intersect after merging describe one split; the event's window is
    that intersection. Events with fewer than
    :data:`_MIN_CORROBORATING_PERIODS` distinct periods are dropped.
    """
    events: List[dict] = []
    for cand in sorted(candidates, key=lambda c: (c["new_filed"], c["old_filed"], c["factor"])):
        for event in events:
            if event["factor"] != cand["factor"]:
                continue
            lo = max(event["old_until"], cand["old_filed"])
            hi = min(event["new_from"], cand["new_filed"])
            if lo < hi:
                event["old_until"], event["new_from"] = lo, hi
                event["periods"].add(cand["period"])
                break
        else:
            events.append({
                "factor": cand["factor"], "old_until": cand["old_filed"],
                "new_from": cand["new_filed"], "periods": {cand["period"]},
            })
    kept = [e for e in events if len(e["periods"]) >= _MIN_CORROBORATING_PERIODS]
    kept = [e for e in kept if not _is_composite(e, kept)]
    kept.sort(key=lambda e: (e["new_from"], e["old_until"]))
    return kept


def _overlaps(a: dict, b: dict) -> bool:
    """Whether two events' ``(old_until, new_from)`` windows intersect."""
    return a["old_until"] < b["new_from"] and b["old_until"] < a["new_from"]


def _is_composite(event: dict, events: List[dict]) -> bool:
    """Whether ``event`` is two other splits seen through one restatement.

    A period last filed before split A and next re-filed only after split B
    jumps by ``f_A * f_B`` in a single step (e.g. Copart's 2:1 splits of 2022
    and 2023 show up as a 4x restatement of fiscal years nobody re-filed in
    between). Such an event's window overlaps both components', and keeping
    it would apply the same splits twice.
    """
    others = [e for e in events if e is not event and _overlaps(e, event)]
    for i, a in enumerate(others):
        for b in others[i + 1:]:
            product = a["factor"] * b["factor"]
            if abs(product - event["factor"]) <= _RATIO_TOLERANCE * event["factor"]:
                return True
    return False


def _filing_share_levels(facts: dict, as_of_str: Optional[str]) -> Dict[str, float]:
    """One share-count level per filing (``accn``).

    The cover-page count (summed across share classes) when the filing has
    one, else the weighted-average count of the latest period it reports.
    Used only to place a filing that falls inside an event's window on the
    right side of the split (see :class:`SplitBasis`).
    """
    levels: Dict[str, float] = {}
    for row in _visible(_rows(facts, _COVER_TAG, taxonomy="dei"), as_of_str):
        accn = row.get("accn")
        if accn:
            levels[accn] = levels.get(accn, 0.0) + float(row["val"])
    latest: Dict[str, Tuple[str, float]] = {}
    for tag in _DETECTION_TAGS:
        for row in _visible(_rows(facts, tag), as_of_str):
            accn, end = row.get("accn"), str(row.get("end") or "")
            if not accn or accn in levels:
                continue
            if accn not in latest or end > latest[accn][0]:
                latest[accn] = (end, float(row["val"]))
    for accn, (_, value) in latest.items():
        levels[accn] = value
    return levels


class SplitBasis:
    """The split events of one filer and the factor that maps a filing's
    figures onto the newest share basis."""

    def __init__(self, events: List[dict], levels: Dict[str, float], filed_by_accn: Dict[str, str]):
        self.events = events
        self._levels = levels
        for event in events:
            event["old_ref"] = self._level_on(event["old_until"], filed_by_accn)
            event["new_ref"] = self._level_on(event["new_from"], filed_by_accn)

    def _level_on(self, filed: str, filed_by_accn: Dict[str, str]) -> Optional[float]:
        """Share level of a filing filed on ``filed`` (first accn in sort order)."""
        for accn in sorted(a for a, f in filed_by_accn.items() if f == filed):
            if accn in self._levels:
                return self._levels[accn]
        return None

    def factor_for(self, filed: Optional[str], accn: Optional[str] = None) -> float:
        """Cumulative factor (new shares per old share) for a filing.

        A filing filed on/before an event's ``old_until`` is on the old basis
        and a filing filed on/after its ``new_from`` on the new one. A filing
        inside the window is placed by its share level: old basis when that
        level is closer (in log terms) to the old-basis reference than to the
        new-basis one. Without a usable level it is treated as new basis.
        """
        factor = 1.0
        if not filed:
            return factor
        for event in self.events:
            if filed <= event["old_until"]:
                factor *= event["factor"]
            elif filed < event["new_from"]:
                level = self._levels.get(accn) if accn else None
                old_ref, new_ref = event.get("old_ref"), event.get("new_ref")
                if level and old_ref and new_ref and level > 0:
                    if abs(math.log(level / old_ref)) < abs(math.log(level / new_ref)):
                        factor *= event["factor"]
        return factor

    def summary(self) -> List[dict]:
        """Public, JSON-safe description of the detected events."""
        return [
            {
                "factor": round(e["factor"], 4),
                "old_basis_until": e["old_until"],
                "new_basis_from": e["new_from"],
                "corroborating_periods": len(e["periods"]),
            }
            for e in self.events
        ]


def detect_share_splits(facts: dict, as_of_str: Optional[str] = None) -> Optional[SplitBasis]:
    """Detect stock splits from restated share counts.

    Args:
        facts: ``facts_json["facts"]`` (taxonomy -> tag -> data).
        as_of_str: Optional ISO cutoff; facts filed after it are invisible,
            matching ``normalize_facts``'s point-in-time mode.

    Returns:
        A :class:`SplitBasis`, or ``None`` when no split is detected (or on
        any unexpected error, which is logged).
    """
    try:
        events = _cluster(_candidates(facts, as_of_str))
        if not events:
            return None
        filed_by_accn: Dict[str, str] = {}
        for tag, taxonomy in [(t, "us-gaap") for t in _DETECTION_TAGS] + [(_COVER_TAG, "dei")]:
            for row in _visible(_rows(facts, tag, taxonomy), as_of_str):
                if row.get("accn"):
                    filed_by_accn.setdefault(row["accn"], row["filed"])
        basis = SplitBasis(events, _filing_share_levels(facts, as_of_str), filed_by_accn)
        logger.info("Detected stock split(s): %s", basis.summary())
        return basis
    except Exception:  # noqa: BLE001 - a detection failure must not break normalization
        logger.warning("detect_share_splits failed; share basis left as reported.", exc_info=True)
        return None


def apply_split_basis(records: List[dict], concept: str, basis: Optional[SplitBasis]) -> None:
    """Convert ``records`` of a split-sensitive concept onto the newest basis, in place.

    An adjusted record keeps its original value as ``reported_value`` and
    gains ``split_factor``. Records already on the newest basis, and every
    record of a concept outside :data:`SPLIT_SENSITIVE_CONCEPTS`, are left
    untouched.
    """
    mode = SPLIT_SENSITIVE_CONCEPTS.get(concept)
    if basis is None or mode is None:
        return
    for record in records:
        value = record.get("value")
        if not isinstance(value, (int, float)):
            continue
        factor = basis.factor_for(record.get("filed"), record.get("_accn"))
        if factor == 1.0:
            continue
        record["reported_value"] = value
        record["split_factor"] = round(factor, 6)
        # Per-share amounts are rounded so a division never leaves float
        # noise (0.9949999...) behind; share counts stay unrounded.
        record["value"] = value * factor if mode == "multiply" else round(value / factor, 6)
