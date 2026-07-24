"""Persistence for frozen phase-1 assumption sets (ASSUMPTIONS_CACHE_SPEC.md).

This module makes the two-phase flow's phase-1 output (bear/base/bull growth/
terminal-growth/discount-rate assumptions plus the sector-type guess) a
*persisted, reviewable, per-filing artifact* instead of something an LLM
re-proposes on every ``analyze`` run. The engine has always been
deterministic; caching the assumptions makes its INPUTS deterministic too, so
the same filer's fair value only moves when its fundamentals move (a new
filing / restatement) or when an analyst deliberately revises the set -- never
because of LLM sampling noise between two runs on the same data.

Lifecycle of one company's set: ``propose`` (the only LLM step, offline) ->
optional ``edit`` -> ``freeze`` -> runtime ``analyze`` reads the frozen set and
calls no LLM for phase 1. Every set is versioned and append-only: freezing a
new set supersedes (never deletes) the previously frozen one, preserving the
audit trail of what the model proposed and why at the time.

Built on nothing but the stdlib ``sqlite3``/``hashlib``/``json``, mirroring
``store.database``'s conventions (idempotent DDL, ``sqlite3.Row`` connections,
defensive/never-raise-into-CLI behavior). The single ``assumption_sets`` table
lives in the same database file as everything else; its DDL is registered with
``store.database.init_db`` via :func:`init_assumptions_table` so one entry
point still ensures the full schema.
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from sec_analyzer.store.database import get_connection, init_db

logger = logging.getLogger(__name__)

#: Status values a row can hold. Exactly one ``draft`` and at most one
#: ``frozen`` row may exist per cik at any time (enforced by
#: :func:`save_draft`/:func:`freeze_draft`); ``superseded`` rows are the
#: append-only history of previously-frozen sets.
STATUS_DRAFT = "draft"
STATUS_FROZEN = "frozen"
STATUS_SUPERSEDED = "superseded"

#: JSON-encoded columns, decoded back into Python objects by :func:`_row_to_dict`.
_JSON_COLUMNS = ("assumptions", "hyper_extras", "script_baseline", "sanity_notes")

#: All columns selected/returned by the read helpers, in table order.
_COLUMNS = (
    "id", "cik", "ticker", "status", "fundamental_fy", "facts_fingerprint",
    "source_provider", "source_model", "proposed_at", "frozen_at",
    "superseded_at", "sector_type", "assumptions_json", "hyper_extras_json",
    "script_baseline_json", "sanity_notes_json", "review_note",
)


def init_assumptions_table(conn: sqlite3.Connection) -> None:
    """Create the ``assumption_sets`` table on an already-open connection.

    Called by :func:`sec_analyzer.store.database.init_db` inside its own
    transaction so a single ``init_db`` call still ensures the full schema.
    Idempotent (``CREATE TABLE IF NOT EXISTS``); the caller owns the
    transaction/commit.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assumption_sets (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            cik                  TEXT NOT NULL,
            ticker               TEXT,
            status               TEXT NOT NULL,
            fundamental_fy       INTEGER,
            facts_fingerprint    TEXT,
            source_provider      TEXT,
            source_model         TEXT,
            proposed_at          TEXT,
            frozen_at            TEXT,
            superseded_at        TEXT,
            sector_type          TEXT,
            assumptions_json     TEXT NOT NULL,
            hyper_extras_json    TEXT,
            script_baseline_json TEXT,
            sanity_notes_json    TEXT,
            review_note          TEXT
        )
        """
    )


def fingerprint_annual(normalized: dict) -> str:
    """Return a sha256 hex digest of the canonical annual-series payload.

    The fingerprint decides whether a cached assumption set is still fresh:
    it changes when a new fiscal year lands in any annual series or a prior
    value is restated, but NOT when daily prices move or a new quarterly
    figure arrives (only annual series feed it). This is deliberately the
    same ``{concept: {fy: value}}`` shape ``interpret.analyzer._annual_by_
    concept`` sends to phase 1, so the fingerprint tracks exactly the data the
    assumptions were proposed from.

    Serialized with ``sort_keys=True`` so dict/iteration ordering can never
    change the digest for identical data.
    """
    # Local import keeps the store layer free of an import-time dependency on
    # the normalize package (and mirrors _annual_by_concept without importing
    # the heavier interpret.analyzer module just for one private helper).
    from sec_analyzer.normalize.normalizer import to_annual_series

    annual_by_concept: Dict[str, Dict[str, object]] = {}
    for concept in (normalized or {}).get("annual") or {}:
        series = to_annual_series(normalized, concept)
        if series:
            annual_by_concept[concept] = {str(fy): value for fy, value in series.items()}

    canonical = json.dumps(
        annual_by_concept, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_fresh(row: Optional[dict], normalized: dict, metrics: Optional[dict]) -> bool:
    """Whether a cached set ``row`` still matches the current fundamentals.

    Both checks (ASSUMPTIONS_CACHE_SPEC.md Sec.2) must pass: the stored
    ``fundamental_fy`` must equal ``resolve_fundamental_fy(metrics)`` of the
    current run, AND the stored ``facts_fingerprint`` must equal
    :func:`fingerprint_annual` of the current ``normalized`` facts. A ``None``
    row (no cached set) is never fresh.
    """
    if not row:
        return False
    from sec_analyzer.normalize.metrics import resolve_fundamental_fy

    current_fy = resolve_fundamental_fy(metrics or {})
    if row.get("fundamental_fy") != current_fy:
        return False
    return row.get("facts_fingerprint") == fingerprint_annual(normalized)


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    """Turn a raw ``sqlite3.Row`` into a plain dict, decoding JSON columns.

    Each ``<name>_json`` column is decoded into a ``<name>`` key holding the
    parsed Python object (``None`` if the column was NULL or unparseable), so
    callers get ``row["assumptions"]`` as a dict rather than a JSON string.
    The raw ``*_json`` strings are dropped from the returned dict.
    """
    if row is None:
        return None
    out = dict(row)
    for name in _JSON_COLUMNS:
        raw = out.pop(f"{name}_json", None)
        if raw is None:
            out[name] = None
            continue
        try:
            out[name] = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("assumption_sets: could not decode %s_json for id %s", name, out.get("id"))
            out[name] = None
    return out


def _load_by_status(conn: sqlite3.Connection, cik: str, status: str) -> Optional[dict]:
    """Return the newest row for ``cik`` with the given ``status``, or ``None``."""
    cursor = conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM assumption_sets "
        "WHERE cik = ? AND status = ? ORDER BY id DESC LIMIT 1",
        (str(cik), status),
    )
    return _row_to_dict(cursor.fetchone())


def save_draft(cik, ticker: Optional[str], payload: dict, db_path: Optional[str] = None) -> Optional[int]:
    """Insert a fresh ``draft`` set for ``cik``, replacing any existing draft.

    ``payload`` carries native Python objects (this function owns the JSON
    serialization):

    * ``fundamental_fy`` (int|None), ``facts_fingerprint`` (str)
    * ``source_provider`` (str), ``source_model`` (str|None)
    * ``proposed_at`` (ISO str; defaults to now if omitted)
    * ``sector_type`` (str)
    * ``assumptions`` (dict -- the CLAMPED bear/base/bull set)
    * ``hyper_extras`` (dict|None), ``script_baseline`` (dict|None)
    * ``sanity_notes`` (list[str])

    At most one draft exists per cik: any prior draft row for ``cik`` is
    deleted first (a re-``propose`` supersedes an unreviewed draft). Returns
    the new row id, or ``None`` on failure (never raises into the CLI).
    """
    cik_str = str(cik)
    try:
        init_db(db_path)
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001 - persistence must not crash the CLI
        logger.warning("save_draft: could not open DB", exc_info=True)
        return None
    try:
        proposed_at = payload.get("proposed_at") or datetime.now().isoformat(timespec="seconds")
        with conn:
            conn.execute(
                "DELETE FROM assumption_sets WHERE cik = ? AND status = ?",
                (cik_str, STATUS_DRAFT),
            )
            cursor = conn.execute(
                """
                INSERT INTO assumption_sets (
                    cik, ticker, status, fundamental_fy, facts_fingerprint,
                    source_provider, source_model, proposed_at, frozen_at,
                    superseded_at, sector_type, assumptions_json,
                    hyper_extras_json, script_baseline_json, sanity_notes_json,
                    review_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    cik_str, ticker, STATUS_DRAFT,
                    payload.get("fundamental_fy"), payload.get("facts_fingerprint"),
                    payload.get("source_provider"), payload.get("source_model"),
                    proposed_at, payload.get("sector_type"),
                    json.dumps(payload.get("assumptions"), ensure_ascii=False),
                    json.dumps(payload["hyper_extras"], ensure_ascii=False) if payload.get("hyper_extras") else None,
                    json.dumps(payload["script_baseline"], ensure_ascii=False) if payload.get("script_baseline") else None,
                    json.dumps(payload.get("sanity_notes") or [], ensure_ascii=False),
                ),
            )
            new_id = cursor.lastrowid
        logger.info("Saved draft assumption set id %s for CIK %s", new_id, cik_str)
        return new_id
    except Exception:  # noqa: BLE001
        logger.warning("save_draft failed for CIK %s", cik_str, exc_info=True)
        return None
    finally:
        conn.close()


def load_active(cik, status: str = STATUS_FROZEN, db_path: Optional[str] = None) -> Optional[dict]:
    """Return the current ``frozen`` (default) or ``draft`` set for ``cik``.

    JSON columns are decoded (see :func:`_row_to_dict`), so
    ``result["assumptions"]`` is a dict. Returns ``None`` if no such set
    exists. Never raises.
    """
    try:
        init_db(db_path)
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001
        logger.warning("load_active: could not open DB", exc_info=True)
        return None
    try:
        return _load_by_status(conn, str(cik), status)
    except Exception:  # noqa: BLE001
        logger.warning("load_active failed for CIK %s", cik, exc_info=True)
        return None
    finally:
        conn.close()


def freeze_draft(cik, review_note: Optional[str] = None, db_path: Optional[str] = None) -> Optional[dict]:
    """Promote ``cik``'s draft to ``frozen``, superseding any prior frozen set.

    Requires an existing draft (returns ``None`` if there is none). In one
    transaction: any current ``frozen`` row for ``cik`` is flipped to
    ``superseded`` (with ``superseded_at`` stamped), then the draft becomes
    ``frozen`` with ``frozen_at`` stamped and ``review_note`` updated when
    given. Returns the newly frozen row (decoded), or ``None`` on failure.
    """
    cik_str = str(cik)
    try:
        init_db(db_path)
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001
        logger.warning("freeze_draft: could not open DB", exc_info=True)
        return None
    try:
        now = datetime.now().isoformat(timespec="seconds")
        with conn:
            draft = _load_by_status(conn, cik_str, STATUS_DRAFT)
            if draft is None:
                return None
            conn.execute(
                "UPDATE assumption_sets SET status = ?, superseded_at = ? "
                "WHERE cik = ? AND status = ?",
                (STATUS_SUPERSEDED, now, cik_str, STATUS_FROZEN),
            )
            if review_note is not None:
                conn.execute(
                    "UPDATE assumption_sets SET status = ?, frozen_at = ?, review_note = ? WHERE id = ?",
                    (STATUS_FROZEN, now, review_note, draft["id"]),
                )
            else:
                conn.execute(
                    "UPDATE assumption_sets SET status = ?, frozen_at = ? WHERE id = ?",
                    (STATUS_FROZEN, now, draft["id"]),
                )
            frozen = _load_by_status(conn, cik_str, STATUS_FROZEN)
        logger.info("Froze assumption set id %s for CIK %s", draft["id"], cik_str)
        return frozen
    except Exception:  # noqa: BLE001
        logger.warning("freeze_draft failed for CIK %s", cik_str, exc_info=True)
        return None
    finally:
        conn.close()


def update_draft(
    cik,
    assumptions: dict,
    sector_type: Optional[str],
    review_note: Optional[str] = None,
    source_suffix: str = "+manual",
    db_path: Optional[str] = None,
) -> Optional[dict]:
    """Edit ``cik``'s draft in place: re-validate, re-clamp, mark as manual.

    Requires an existing draft (returns ``None`` if none). The supplied
    ``assumptions`` are run back through
    :func:`sec_analyzer.valuation.sanity.clamp_assumptions` (so what is stored
    is what every downstream computation would use), ``sanity_notes`` is
    refreshed with the resulting clamp notes, ``source_suffix`` (default
    ``"+manual"``) is appended to ``source_provider`` exactly once, and
    ``review_note`` is stored when given. ``sector_type`` overwrites the
    stored one when non-``None``. Returns the updated draft (decoded), or
    ``None`` on failure.
    """
    cik_str = str(cik)
    # Lazy import: keeps the store layer decoupled from the valuation package
    # at import time (only the edit path needs the clamp).
    from sec_analyzer.valuation.sanity import clamp_assumptions

    try:
        init_db(db_path)
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001
        logger.warning("update_draft: could not open DB", exc_info=True)
        return None
    try:
        with conn:
            draft = _load_by_status(conn, cik_str, STATUS_DRAFT)
            if draft is None:
                return None
            resolved_sector = sector_type or draft.get("sector_type")
            clamped, notes = clamp_assumptions(
                assumptions, is_unprofitable=(resolved_sector == "growth_unprofitable")
            )
            provider = draft.get("source_provider") or ""
            if source_suffix and not provider.endswith(source_suffix):
                provider = f"{provider}{source_suffix}"
            conn.execute(
                """
                UPDATE assumption_sets
                SET assumptions_json = ?, sector_type = ?, sanity_notes_json = ?,
                    source_provider = ?, review_note = COALESCE(?, review_note)
                WHERE id = ?
                """,
                (
                    json.dumps(clamped, ensure_ascii=False),
                    resolved_sector,
                    json.dumps(notes, ensure_ascii=False),
                    provider,
                    review_note,
                    draft["id"],
                ),
            )
            updated = _load_by_status(conn, cik_str, STATUS_DRAFT)
        logger.info("Updated draft assumption set id %s for CIK %s", draft["id"], cik_str)
        return updated
    except Exception:  # noqa: BLE001
        logger.warning("update_draft failed for CIK %s", cik_str, exc_info=True)
        return None
    finally:
        conn.close()


def load_history(cik, limit: int = 20, db_path: Optional[str] = None) -> List[dict]:
    """Return every stored set for ``cik`` (all statuses), newest first.

    Powers ``assumptions show``'s provenance/history panel. Ordered by ``id``
    descending, capped at ``limit``. Empty list if none (or on error).
    """
    try:
        init_db(db_path)
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001
        logger.warning("load_history: could not open DB", exc_info=True)
        return []
    try:
        cursor = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM assumption_sets "
            "WHERE cik = ? ORDER BY id DESC LIMIT ?",
            (str(cik), int(limit)),
        )
        return [_row_to_dict(row) for row in cursor.fetchall()]
    except Exception:  # noqa: BLE001
        logger.warning("load_history failed for CIK %s", cik, exc_info=True)
        return []
    finally:
        conn.close()
