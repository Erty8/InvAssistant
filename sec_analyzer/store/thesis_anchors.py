"""Persistence for the thesis-validation metric's day-1 anchor direction
(METODOLOJI.md Sec.7's quarterly-invalidation rule).

The thesis-validation metric picked by
:func:`sec_analyzer.interpret.planning.select_thesis_metric` is compared,
every run, against a fixed reference direction ("is this quarter's move
WITH or AGAINST the original thesis?") rather than against whatever its
trend happens to read as on the day of the comparison -- otherwise the
"direction" the metric is checked against would drift alongside the metric
itself, and the invalidation check would become tautological. This module
makes that reference direction a *persisted, per-filer fact*, established
once (the first time an anchor is set for a given ``(cik, metric_key)``
pair) and re-established only when the chosen anchor metric itself changes
(e.g. a sector reclassification swaps ``net_margin`` for ``gross_margin``).

Deliberately much simpler than :mod:`sec_analyzer.store.assumptions`'
``assumption_sets`` table: there is no draft/frozen/superseded lifecycle or
append-only audit trail here, just one current row per ``cik`` (keyed on
``cik`` alone -- a filer has exactly one active thesis anchor at a time,
matching ``select_thesis_metric``'s "single anchor metric" design). Built on
nothing but the stdlib ``sqlite3``, mirroring ``store.assumptions``'
conventions (idempotent ``CREATE TABLE IF NOT EXISTS``, ``sqlite3.Row``
connections via ``store.database.get_connection``, defensive/never-raise-
into-CLI behavior). The ``thesis_anchors`` table lives in the same database
file as everything else; its DDL is registered with
``store.database.init_db`` via :func:`init_thesis_anchors_table`, so one
entry point still ensures the full schema.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional

from sec_analyzer.store.database import get_connection, init_db

logger = logging.getLogger(__name__)

#: All columns selected/returned by the read helpers, in table order.
_COLUMNS = ("cik", "ticker", "metric_key", "direction", "established_fy", "established_at")


def init_thesis_anchors_table(conn: sqlite3.Connection) -> None:
    """Create the ``thesis_anchors`` table on an already-open connection.

    Called by :func:`sec_analyzer.store.database.init_db` inside its own
    transaction so a single ``init_db`` call still ensures the full schema.
    Idempotent (``CREATE TABLE IF NOT EXISTS``); the caller owns the
    transaction/commit.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thesis_anchors (
            cik             TEXT PRIMARY KEY,
            ticker          TEXT,
            metric_key      TEXT NOT NULL,
            direction       TEXT NOT NULL,
            established_fy  INTEGER,
            established_at  TEXT
        )
        """
    )


def get_anchor(cik, db_path: Optional[str] = None) -> Optional[dict]:
    """Return the current thesis anchor for ``cik``, or ``None`` if none is
    established yet (or on any DB failure -- never raises).

    Returns:
        ``{"cik": str, "ticker": str|None, "metric_key": str, "direction":
        "improving"|"deteriorating", "established_fy": int|None,
        "established_at": str|None}``, or ``None``.
    """
    try:
        init_db(db_path)
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001 - persistence must not crash the CLI
        logger.warning("get_anchor: could not open DB", exc_info=True)
        return None
    try:
        cursor = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM thesis_anchors WHERE cik = ?",
            (str(cik),),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None
    except Exception:  # noqa: BLE001
        logger.warning("get_anchor failed for CIK %s", cik, exc_info=True)
        return None
    finally:
        conn.close()


def set_anchor(
    cik,
    metric_key: str,
    direction: str,
    established_fy: Optional[int],
    ticker: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[dict]:
    """Establish (insert) or re-establish (overwrite) the thesis anchor for
    ``cik``.

    This unconditionally writes ``direction``/``metric_key``/
    ``established_fy`` -- the caller (:mod:`sec_analyzer.interpret.
    planning`'s ``_establish_or_load_anchor``) is responsible for only
    calling this when an anchor should actually be (re)established: no row
    exists yet for this ``cik``, or the stored row's ``metric_key`` no
    longer matches (e.g. a sector reclassification swapped the anchor
    metric). Once set, an anchor's ``direction`` is meant to stick across
    runs for the same ``(cik, metric_key)`` pair -- callers must not call
    this just because the metric's own trend has since moved; that
    "does the metric still agree with its own day-1 anchor" check is exactly
    what the quarterly comparison in ``planning.py`` is for.

    ``established_at`` is stamped with the current time as audit metadata
    only (mirrors ``store.assumptions``' ``proposed_at``/``frozen_at``) --
    it is never read back into any invalidation comparison, keeping the
    actual check deterministic.

    Args:
        cik: Filer CIK (stringified for storage).
        metric_key: The anchor metric's ratio-row key, e.g. ``"net_margin"``
            (one of :data:`sec_analyzer.interpret.planning.
            _SECTOR_METRIC_CANDIDATES`'s keys).
        direction: ``"improving"`` or ``"deteriorating"`` -- the metric's own
            annual trend label the first time this anchor is (re)established.
        established_fy: The fiscal year whose trend produced ``direction``,
            or ``None``.
        ticker: Optional display ticker (denormalized for convenience only,
            same spirit as other tables in this package); not used as a key.
        db_path: Optional override of the SQLite file path.

    Returns:
        The newly (re)established row (same shape as :func:`get_anchor`), or
        ``None`` on failure. Never raises.
    """
    cik_str = str(cik)
    try:
        init_db(db_path)
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001
        logger.warning("set_anchor: could not open DB", exc_info=True)
        return None
    try:
        established_at = datetime.now().isoformat(timespec="seconds")
        with conn:
            conn.execute(
                """
                INSERT INTO thesis_anchors
                    (cik, ticker, metric_key, direction, established_fy, established_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cik) DO UPDATE SET
                    ticker = excluded.ticker,
                    metric_key = excluded.metric_key,
                    direction = excluded.direction,
                    established_fy = excluded.established_fy,
                    established_at = excluded.established_at
                """,
                (cik_str, ticker, metric_key, direction, established_fy, established_at),
            )
            cursor = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM thesis_anchors WHERE cik = ?",
                (cik_str,),
            )
            row = cursor.fetchone()
        logger.info(
            "Established thesis anchor for CIK %s: metric=%s direction=%s fy=%s",
            cik_str, metric_key, direction, established_fy,
        )
        return dict(row) if row is not None else None
    except Exception:  # noqa: BLE001
        logger.warning("set_anchor failed for CIK %s", cik_str, exc_info=True)
        return None
    finally:
        conn.close()
