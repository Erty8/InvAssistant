"""SQLite persistence layer for normalized SEC financial data.

This module is the only place in ``sec_analyzer`` that talks to the on-disk
database. It is intentionally built on nothing but the stdlib ``sqlite3``
module -- no ORM, no third-party driver -- since the schema is small and
the access patterns are simple (upsert a company, upsert a batch of
financial facts, upsert a batch of ratios).

Every write operation here is an *upsert* (``INSERT ... ON CONFLICT ...
DO UPDATE``), so re-running a fetch/normalize/save pipeline against the
same filer is safe and idempotent: existing rows are refreshed in place
rather than duplicated, and restated figures simply overwrite the old
value for that ``(cik, concept, period_end, fp)`` (or ``(cik, fy)`` for
ratios).

Typical usage::

    from sec_analyzer.store.database import save_normalized

    save_normalized(
        ticker="AAPL",
        cik=320193,
        name="Apple Inc.",
        normalized=normalized,   # from normalize_facts()
        ratios=ratios,           # from compute_ratios()
    )
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Columns that must exist on the ``financials`` table beyond the ones
#: present when the table is first created below. New columns are appended
#: here as the normalize layer's record shape grows; ``init_db`` adds any
#: that are missing on an already-existing database file, since
#: ``CREATE TABLE IF NOT EXISTS`` has no effect on a table that already
#: exists with fewer columns.
_FINANCIALS_EXTRA_COLUMNS: List[Tuple[str, str]] = [
    ("unit", "TEXT"),
]

#: Same idea as ``_FINANCIALS_EXTRA_COLUMNS``, for the ``ratios`` table.
_RATIOS_EXTRA_COLUMNS: List[Tuple[str, str]] = [
    ("gross_margin", "REAL"),
    ("operating_margin", "REAL"),
    ("roa", "REAL"),
    ("debt_to_equity", "REAL"),
    ("fcf", "REAL"),
    ("fcf_margin", "REAL"),
]

#: Same idea as ``_FINANCIALS_EXTRA_COLUMNS``, for the ``verdicts`` table --
#: added once the deterministic valuation engine (SPEC.md Sec.11/14) started
#: producing a ``valuation`` dict alongside the LLM/rule-based commentary.
_VERDICTS_EXTRA_COLUMNS: List[Tuple[str, str]] = [
    ("confidence", "TEXT"),
    ("sector_type", "TEXT"),
    ("implied_growth", "REAL"),
    ("fair_value_json", "TEXT"),
    ("valuation_json", "TEXT"),
    # Optional manual annotation: flags a verdict to KEEP (never overwrite in
    # spirit -- the table is append-only anyway) as a deliberate forward test
    # case, e.g. a model↔market divergence whose thesis a few quarters of data
    # will referee. Defaults NULL; never written by save_verdict.
    ("watch_note", "TEXT"),
    # Point-in-time cutoff (ISO date) when the verdict was produced in as-of
    # mode; NULL for ordinary live runs. Lets the verdict-history screen and
    # backtests distinguish "analyzed today with today's data" from "analyzed
    # as of a past date".
    ("as_of", "TEXT"),
]


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: List[Tuple[str, str]]) -> None:
    """Add any of ``columns`` not already present on ``table``.

    ``CREATE TABLE IF NOT EXISTS`` only creates a table when it doesn't
    exist yet; it never alters an existing table's column set. This helper
    closes that gap for databases created by an older version of this
    module: it inspects ``table``'s current columns via
    ``PRAGMA table_info`` and issues an ``ALTER TABLE ... ADD COLUMN`` for
    each ``(name, sql_type)`` pair in ``columns`` that isn't already there.
    Idempotent -- safe to call every time ``init_db`` runs, whether the
    table was just created (nothing to add) or pre-existed with an older
    schema (missing columns get added).
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, sql_type in columns:
        if name in existing:
            continue
        logger.info("Migrating schema: adding column %s.%s (%s)", table, name, sql_type)
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a SQLite connection configured for this package's needs.

    Args:
        db_path: Path to the SQLite file. Defaults to ``Config.DB_PATH``.

    Returns:
        A ``sqlite3.Connection`` with ``row_factory`` set to
        ``sqlite3.Row`` (so result rows support both index and column-name
        access) and foreign-key enforcement turned on.

    Side effects:
        Calls ``Config.ensure_dirs()`` and, defensively, also creates the
        parent directory of ``db_path`` if it doesn't already exist (this
        matters when ``SEC_DB_PATH`` points somewhere outside the package's
        own directory tree, which ``Config.ensure_dirs()`` doesn't know
        about).
    """
    Config.ensure_dirs()
    resolved_path = db_path or Config.DB_PATH

    parent_dir = os.path.dirname(resolved_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    conn = sqlite3.connect(resolved_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Create the ``companies``, ``financials``, ``ratios``, ``prices``, and
    ``verdicts`` tables.

    Safe to call any number of times: every statement is
    ``CREATE TABLE IF NOT EXISTS``, and any columns added to ``financials``,
    ``ratios``, or ``verdicts`` since a database file was first created are backfilled
    via ``ALTER TABLE ... ADD COLUMN`` (see ``_ensure_columns``).

    Args:
        db_path: Path to the SQLite file. Defaults to ``Config.DB_PATH``.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS companies (
                    cik  TEXT PRIMARY KEY,
                    ticker TEXT,
                    name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financials (
                    cik        TEXT,
                    concept    TEXT,
                    period_end TEXT,
                    fy         INTEGER,
                    fp         TEXT,
                    form       TEXT,
                    value      REAL,
                    filed      TEXT,
                    unit       TEXT,
                    PRIMARY KEY (cik, concept, period_end, fp)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ratios (
                    cik                    TEXT,
                    fy                     INTEGER,
                    period_end             TEXT,
                    net_margin             REAL,
                    roe                    REAL,
                    current_ratio          REAL,
                    yoy_revenue_growth     REAL,
                    yoy_net_income_growth  REAL,
                    gross_margin           REAL,
                    operating_margin       REAL,
                    roa                    REAL,
                    debt_to_equity         REAL,
                    fcf                    REAL,
                    fcf_margin             REAL,
                    PRIMARY KEY (cik, fy)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    cik    TEXT,
                    date   TEXT,
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL,
                    volume REAL,
                    PRIMARY KEY (cik, date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verdicts (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    cik                 TEXT,
                    ticker              TEXT,
                    analyzed_at         TEXT,
                    horizon             TEXT,
                    provider            TEXT,
                    price               REAL,
                    fundamental_verdict TEXT,
                    technical_verdict   TEXT,
                    profile_fit         TEXT,
                    fv_bear_lo          REAL,
                    fv_bear_hi          REAL,
                    fv_base_lo          REAL,
                    fv_base_hi          REAL,
                    fv_bull_lo          REAL,
                    fv_bull_hi          REAL,
                    result_json         TEXT
                )
                """
            )

            # Migrate pre-existing database files (created by an older
            # version of this module) to the current column set. No-op on
            # a freshly created table above, since the columns are already
            # present there.
            _ensure_columns(conn, "financials", _FINANCIALS_EXTRA_COLUMNS)
            _ensure_columns(conn, "ratios", _RATIOS_EXTRA_COLUMNS)
            _ensure_columns(conn, "verdicts", _VERDICTS_EXTRA_COLUMNS)
        logger.debug("Schema ensured at %s", db_path or Config.DB_PATH)
    finally:
        conn.close()


def upsert_company(conn: sqlite3.Connection, cik: str, ticker: Optional[str], name: Optional[str]) -> None:
    """Insert or update a single row in ``companies``.

    Args:
        conn: An open connection (caller manages the transaction/commit).
        cik: Central Index Key, as a string (the table's primary key).
        ticker: Exchange ticker symbol, e.g. ``"AAPL"``. May be ``None``.
        name: Company/entity name. May be ``None``.
    """
    conn.execute(
        """
        INSERT INTO companies (cik, ticker, name)
        VALUES (?, ?, ?)
        ON CONFLICT(cik) DO UPDATE SET
            ticker = excluded.ticker,
            name   = excluded.name
        """,
        (cik, ticker, name),
    )


def upsert_financials(conn: sqlite3.Connection, cik: str, records: Iterable[dict]) -> int:
    """Insert or update a batch of normalized financial fact records.

    Each ``record`` is expected to have the shape produced by
    ``sec_analyzer.normalize.normalizer.normalize_facts`` --
    ``{"concept", "tag", "period_end", "fy", "reported_fy", "fp", "form",
    "value", "filed", "start", "unit"}`` -- though only the columns that
    exist in the ``financials`` table (``concept``, ``period_end``, ``fy``,
    ``fp``, ``form``, ``value``, ``filed``, ``unit``) are persisted.

    Records whose ``value`` is ``None`` are skipped: they carry no actual
    figure to store and would otherwise overwrite a previously-known good
    value on re-run.

    Args:
        conn: An open connection (caller manages the transaction/commit).
        cik: Central Index Key, as a string.
        records: An iterable of record dicts (annual and/or quarterly can
            be mixed together; they're distinguished by ``fp``/``form``).

    Returns:
        The number of rows actually written (i.e. after skipping records
        with a ``None`` value).
    """
    rows = [
        (
            cik,
            record.get("concept"),
            record.get("period_end"),
            record.get("fy"),
            record.get("fp"),
            record.get("form"),
            record.get("value"),
            record.get("filed"),
            record.get("unit"),
        )
        for record in records
        if record.get("value") is not None
    ]

    if not rows:
        logger.debug("upsert_financials: nothing to write for CIK %s", cik)
        return 0

    conn.executemany(
        """
        INSERT INTO financials (cik, concept, period_end, fy, fp, form, value, filed, unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cik, concept, period_end, fp) DO UPDATE SET
            fy    = excluded.fy,
            form  = excluded.form,
            value = excluded.value,
            filed = excluded.filed,
            unit  = excluded.unit
        """,
        rows,
    )
    return len(rows)


def upsert_ratios(conn: sqlite3.Connection, cik: str, ratios: Iterable[dict]) -> int:
    """Insert or update a batch of per-fiscal-year ratio records.

    Each ``ratio`` is expected to have the shape produced by
    ``sec_analyzer.normalize.ratios.compute_ratios`` -- ``{"fy",
    "period_end", "net_margin", "roe", "current_ratio",
    "yoy_revenue_growth", "yoy_net_income_growth", "gross_margin",
    "operating_margin", "roa", "debt_to_equity", "fcf", "fcf_margin"}``.

    Args:
        conn: An open connection (caller manages the transaction/commit).
        cik: Central Index Key, as a string.
        ratios: An iterable of per-fiscal-year ratio dicts.

    Returns:
        The number of rows written.
    """
    rows = [
        (
            cik,
            ratio.get("fy"),
            ratio.get("period_end"),
            ratio.get("net_margin"),
            ratio.get("roe"),
            ratio.get("current_ratio"),
            ratio.get("yoy_revenue_growth"),
            ratio.get("yoy_net_income_growth"),
            ratio.get("gross_margin"),
            ratio.get("operating_margin"),
            ratio.get("roa"),
            ratio.get("debt_to_equity"),
            ratio.get("fcf"),
            ratio.get("fcf_margin"),
        )
        for ratio in ratios
    ]

    if not rows:
        logger.debug("upsert_ratios: nothing to write for CIK %s", cik)
        return 0

    conn.executemany(
        """
        INSERT INTO ratios (
            cik, fy, period_end, net_margin, roe,
            current_ratio, yoy_revenue_growth, yoy_net_income_growth,
            gross_margin, operating_margin, roa, debt_to_equity, fcf, fcf_margin
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cik, fy) DO UPDATE SET
            period_end            = excluded.period_end,
            net_margin            = excluded.net_margin,
            roe                   = excluded.roe,
            current_ratio         = excluded.current_ratio,
            yoy_revenue_growth    = excluded.yoy_revenue_growth,
            yoy_net_income_growth = excluded.yoy_net_income_growth,
            gross_margin          = excluded.gross_margin,
            operating_margin      = excluded.operating_margin,
            roa                   = excluded.roa,
            debt_to_equity        = excluded.debt_to_equity,
            fcf                   = excluded.fcf,
            fcf_margin            = excluded.fcf_margin
        """,
        rows,
    )
    return len(rows)


def save_prices(cik, price_rows: Iterable[dict], db_path: Optional[str] = None) -> int:
    """Insert or update a batch of daily OHLCV price rows for a filer.

    This is the DB layer's own entry point (unlike ``upsert_financials``/
    ``upsert_ratios``, which take an already-open connection): callers such
    as the CLI hold price history in a pandas DataFrame, convert it to
    plain dicts, and call this directly. The DB layer itself stays
    pandas-free.

    Args:
        cik: Central Index Key. Accepted as ``int`` or ``str``; stored as
            ``str(cik)``.
        price_rows: An iterable of dicts, each shaped
            ``{"date": "YYYY-MM-DD", "open", "high", "low", "close",
            "volume"}``.
        db_path: Path to the SQLite file. Defaults to ``Config.DB_PATH``.

    Returns:
        The number of rows written (upserted).
    """
    cik_str = str(cik)
    rows = [
        (
            cik_str,
            row.get("date"),
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
            row.get("volume"),
        )
        for row in price_rows
        if row.get("date") is not None
    ]

    if not rows:
        logger.debug("save_prices: nothing to write for CIK %s", cik_str)
        return 0

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                """
                INSERT INTO prices (cik, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cik, date) DO UPDATE SET
                    open   = excluded.open,
                    high   = excluded.high,
                    low    = excluded.low,
                    close  = excluded.close,
                    volume = excluded.volume
                """,
                rows,
            )
        logger.info("Saved %d price row(s) for CIK %s to %s", len(rows), cik_str, db_path or Config.DB_PATH)
        return len(rows)
    finally:
        conn.close()


def save_verdict(
    ticker: str,
    cik,
    horizon: str,
    provider: str,
    price: Optional[float],
    result: dict,
    db_path: Optional[str] = None,
    analyzed_at: Optional[str] = None,
    valuation: Optional[dict] = None,
    as_of: Optional[str] = None,
) -> int:
    """Append one analysis result to the ``verdicts`` history table.

    Unlike the other ``save_*``/``upsert_*`` functions, this is a plain
    ``INSERT`` rather than an upsert: ``verdicts`` is an append-only history
    table by design, so every analysis run is preserved (rather than
    overwriting a prior run for the same filer/horizon) to support future
    backtesting of the verdicts against subsequent price action.

    Args:
        ticker: Exchange ticker symbol, e.g. ``"AAPL"``.
        cik: Central Index Key. Accepted as ``int`` or ``str``; stored as
            ``str(cik)``.
        horizon: Investment horizon used for this analysis, e.g. ``"1y"``.
        provider: Analyzer provider that produced ``result``, e.g.
            ``"script"``, ``"ollama"``, or ``"anthropic"``.
        price: The current market price used for the analysis, or ``None``.
        result: The dict returned by
            :func:`sec_analyzer.interpret.analyzer.interpret` (or
            :func:`sec_analyzer.interpret.rule_based.analyze`), matching the
            unified bear/base/bull schema. Band values are extracted
            ``None``-safely from ``result["fair_value_range"]``;
            ``profile_fit`` is stored as just its ``"verdict"`` string;
            ``confidence`` (``result["confidence"]``, per SPEC.md Sec.14) is
            stored in its own column. The full result is also stored
            verbatim (as JSON) in ``result_json``.
        db_path: Path to the SQLite file. Defaults to ``Config.DB_PATH``.
        analyzed_at: ISO-8601 timestamp to record. Defaults to
            ``datetime.now().isoformat(timespec="seconds")``.
        as_of: Point-in-time cutoff (ISO ``"YYYY-MM-DD"``) when the analysis
            was run in as-of mode, or ``None`` for an ordinary live run.
        valuation: The dict returned by
            :func:`sec_analyzer.valuation.engine.run_valuation` (typically
            ``result.get("valuation")``), or ``None``. When given, populates
            ``sector_type``, ``implied_growth`` (from
            ``valuation["reverse_dcf"]["implied_growth"]``),
            ``fair_value_json`` (``valuation["fair_value_range"]`` as JSON),
            and ``valuation_json`` (the full dict as JSON); all four stay
            ``None`` when omitted, e.g. for legacy callers or an error
            result with no valuation to attach.

    Returns:
        The id (``rowid``) of the newly inserted row.
    """
    cik_str = str(cik)
    result = result or {}
    fair_value_range = result.get("fair_value_range") or {}
    bear = fair_value_range.get("bear") or {}
    base = fair_value_range.get("base") or {}
    bull = fair_value_range.get("bull") or {}
    profile_fit = result.get("profile_fit") or {}
    profile_fit_verdict = profile_fit.get("verdict") if isinstance(profile_fit, dict) else profile_fit

    sector_type = (valuation or {}).get("sector_type")
    implied_growth = ((valuation or {}).get("reverse_dcf") or {}).get("implied_growth")
    fair_value_json = json.dumps(valuation.get("fair_value_range"), ensure_ascii=False) if valuation else None
    valuation_json = json.dumps(valuation, ensure_ascii=False) if valuation else None

    row = (
        cik_str,
        ticker,
        analyzed_at or datetime.now().isoformat(timespec="seconds"),
        horizon,
        provider,
        price,
        result.get("fundamental_verdict"),
        result.get("technical_verdict"),
        profile_fit_verdict,
        bear.get("lo"),
        bear.get("hi"),
        base.get("lo"),
        base.get("hi"),
        bull.get("lo"),
        bull.get("hi"),
        json.dumps(result, ensure_ascii=False),
        result.get("confidence"),
        sector_type,
        implied_growth,
        fair_value_json,
        valuation_json,
        as_of,
    )

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO verdicts (
                    cik, ticker, analyzed_at, horizon, provider, price,
                    fundamental_verdict, technical_verdict, profile_fit,
                    fv_bear_lo, fv_bear_hi, fv_base_lo, fv_base_hi, fv_bull_lo, fv_bull_hi,
                    result_json, confidence, sector_type, implied_growth,
                    fair_value_json, valuation_json, as_of
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            verdict_id = cursor.lastrowid
        logger.info(
            "Saved verdict for %s (CIK %s, horizon %s, provider %s) to %s",
            ticker, cik_str, horizon, provider, db_path or Config.DB_PATH,
        )
        return verdict_id
    finally:
        conn.close()


def _flatten_records(normalized: dict) -> List[dict]:
    """Flatten the ``annual`` and ``quarterly`` buckets of a normalized
    facts dict into a single flat list of records.

    ``normalized["annual"]`` and ``normalized["quarterly"]`` are each
    ``{concept: [record, ...] or None}``. Concepts with no data (``None``)
    are skipped; everything else is chained together, since
    ``upsert_financials`` distinguishes annual from quarterly rows via each
    record's own ``fp``/``form`` fields rather than needing them kept in
    separate buckets.
    """
    flattened: List[dict] = []
    for bucket_name in ("annual", "quarterly"):
        bucket: Dict[str, Optional[List[dict]]] = normalized.get(bucket_name) or {}
        for concept_records in bucket.values():
            if concept_records:
                flattened.extend(concept_records)
    return flattened


def save_normalized(
    ticker: str,
    cik,
    name: str,
    normalized: dict,
    ratios: List[dict],
    db_path: Optional[str] = None,
) -> None:
    """Persist a normalized filer to the database in one transaction.

    This is the high-level entry point most callers should use: it ensures
    the schema exists, upserts the company row, flattens and upserts all
    annual + quarterly financial records, and upserts the computed ratios
    -- all within a single transaction, so a partially-written filer never
    lands in the database if something goes wrong midway.

    Args:
        ticker: Exchange ticker symbol, e.g. ``"AAPL"``.
        cik: Central Index Key. Accepted as ``int`` or ``str``; stored as
            ``str(cik)`` for consistent lookups regardless of how the
            caller happened to have it.
        name: Company/entity name.
        normalized: The dict returned by
            ``sec_analyzer.normalize.normalizer.normalize_facts``.
        ratios: The list of per-fiscal-year ratio dicts returned by
            ``sec_analyzer.normalize.ratios.compute_ratios``.
        db_path: Path to the SQLite file. Defaults to ``Config.DB_PATH``.
    """
    cik_str = str(cik)

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            upsert_company(conn, cik_str, ticker, name)

            records = _flatten_records(normalized)
            financial_rows_written = upsert_financials(conn, cik_str, records)

            ratio_rows_written = upsert_ratios(conn, cik_str, ratios or [])

        logger.info(
            "Saved %s (CIK %s, %s): %d financial rows, %d ratio rows written to %s",
            ticker, cik_str, name,
            financial_rows_written, ratio_rows_written,
            db_path or Config.DB_PATH,
        )
    finally:
        conn.close()


def load_financials(cik: str, db_path: Optional[str] = None) -> List[dict]:
    """Read back all ``financials`` rows for a given CIK.

    Intended for tests and ad-hoc inspection rather than the main
    fetch/normalize/save pipeline.

    Args:
        cik: Central Index Key. Accepted as ``int`` or ``str``; looked up
            as ``str(cik)``.
        db_path: Path to the SQLite file. Defaults to ``Config.DB_PATH``.

    Returns:
        A list of plain ``dict`` rows (one per stored fact), each with
        keys ``cik``, ``concept``, ``period_end``, ``fy``, ``fp``,
        ``form``, ``value``, ``filed``, ``unit``. Empty list if the CIK has
        no rows (or the table doesn't exist yet).
    """
    cik_str = str(cik)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT cik, concept, period_end, fy, fp, form, value, filed, unit
            FROM financials
            WHERE cik = ?
            ORDER BY concept, period_end
            """,
            (cik_str,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


#: Scalar columns returned by :func:`load_verdicts` -- deliberately excludes
#: the large JSON blobs (``result_json``/``valuation_json``/``fair_value_json``)
#: so the verdict-history screen stays a light query.
_VERDICT_HISTORY_COLUMNS = (
    "id, cik, ticker, analyzed_at, as_of, horizon, provider, price, "
    "fundamental_verdict, technical_verdict, profile_fit, "
    "fv_bear_lo, fv_bear_hi, fv_base_lo, fv_base_hi, fv_bull_lo, fv_bull_hi, "
    "confidence, sector_type, implied_growth, watch_note"
)


def load_verdicts(ticker: str, db_path: Optional[str] = None, limit: int = 100) -> List[dict]:
    """Read back the stored verdict history for a ticker, newest first.

    Powers the web verdict-history screen. Runs :func:`init_db` first so the
    ``as_of`` migration is applied even on a legacy database, then reads only
    the scalar columns (no JSON blobs -- see :data:`_VERDICT_HISTORY_COLUMNS`).

    Args:
        ticker: Exchange ticker symbol; matched case-insensitively.
        db_path: Path to the SQLite file. Defaults to ``Config.DB_PATH``.
        limit: Maximum number of rows to return (most recent first).

    Returns:
        A list of plain ``dict`` rows ordered by ``analyzed_at`` descending.
        Empty list if the ticker has no stored verdicts.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            f"""
            SELECT {_VERDICT_HISTORY_COLUMNS}
            FROM verdicts
            WHERE ticker = ? COLLATE NOCASE
            ORDER BY analyzed_at DESC
            LIMIT ?
            """,
            (str(ticker), int(limit)),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def load_latest_stored_price(ticker: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Return the most recent stored ``prices`` row for a ticker, or ``None``.

    Best-effort, network-free helper for the verdict-history screen's
    "current price" delta column: it joins ``prices`` to ``companies`` by
    ticker (case-insensitive) and returns the latest bar. Returns ``None`` if
    the ticker is unknown or has no stored price rows. Never raises.

    Returns:
        ``{"date": "YYYY-MM-DD", "close": float}`` or ``None``.
    """
    try:
        conn = get_connection(db_path)
    except Exception:  # noqa: BLE001 - best-effort display helper
        logger.warning("load_latest_stored_price: could not open DB", exc_info=True)
        return None
    try:
        cursor = conn.execute(
            """
            SELECT p.date AS date, p.close AS close
            FROM prices p
            JOIN companies c ON c.cik = p.cik
            WHERE c.ticker = ? COLLATE NOCASE
            ORDER BY p.date DESC
            LIMIT 1
            """,
            (str(ticker),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {"date": row["date"], "close": row["close"]}
    except Exception:  # noqa: BLE001 - best-effort display helper
        logger.warning("load_latest_stored_price failed for %s", ticker, exc_info=True)
        return None
    finally:
        conn.close()
