"""Tests for the frozen-assumption-set cache (ASSUMPTIONS_CACHE_SPEC.md).

Covers the store roundtrip/freeze/supersede/edit lifecycle, the annual-facts
fingerprint + freshness check, the ``analyze`` phase-1 resolution matrix, the
``interpret(phase1_override=...)`` determinism guarantee, and verdict
provenance persistence. Everything here is fully offline (no network, no LLM):
the ``"script"`` provider and local Damodaran CSVs are deterministic.
"""

import argparse
import copy

import pytest

from sec_analyzer.cli import build_parser
from sec_analyzer.config import Config
from sec_analyzer.interpret import analyzer
from sec_analyzer.store import assumptions as A
from sec_analyzer.store.database import get_connection


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A throwaway SQLite path, also installed as Config.DB_PATH so code paths
    that read the default (the CLI resolution helpers) hit the same file."""
    p = str(tmp_path / "test.sqlite3")
    monkeypatch.setattr(Config, "DB_PATH", p)
    return p


def _record(fy, value, period_end=None):
    return {"fy": fy, "period_end": period_end or f"{fy}-12-31", "value": value}


def _normalized(revenue_2023=1000.0):
    return {
        "entity_name": "X",
        "currency": "USD",
        "annual": {
            "Revenue": [
                _record(2023, revenue_2023),
                _record(2022, 900.0),
                _record(2021, 800.0),
            ],
            "NetIncome": [_record(2023, 120.0)],
            "OperatingCashFlow": [_record(2023, 150.0)],
            "CapEx": [_record(2023, 30.0)],
            "SharesOutstanding": [_record(2023, 100.0)],
        },
        "quarterly": {},
        "missing": [],
        "matched_tags": {},
    }


def _metrics(fy=2023):
    return {
        "price": 50.0, "shares": 100.0, "eps": 1.2, "net_debt": 0.0,
        "pe": None, "ps": None, "pfcf": None,
        "revenue_cagr_3y": 0.1, "revenue_cagr_5y": None,
        "fcf": 120.0, "latest_fy": fy, "latest_fundamental_fy": fy,
    }


def _assumptions(tg=0.03, dr=0.11):
    def scen(g):
        return {"growth_5y": g, "terminal_growth": tg, "discount_rate": dr, "story": "x"}
    return {"bear": scen(0.06), "base": scen(0.10), "bull": scen(0.14)}


def _draft_payload(normalized, fy=2023):
    return {
        "fundamental_fy": fy,
        "facts_fingerprint": A.fingerprint_annual(normalized),
        "source_provider": "ollama",
        "source_model": "gemma4:latest",
        "sector_type": "mature",
        "assumptions": _assumptions(),
        "hyper_extras": None,
        "script_baseline": _assumptions(dr=0.12),
        "sanity_notes": [],
    }


# --------------------------------------------------------------------------
# Store lifecycle
# --------------------------------------------------------------------------

def test_save_draft_replaces_existing_draft(db_path):
    n = _normalized()
    A.save_draft("100", "AAA", _draft_payload(n), db_path=db_path)
    A.save_draft("100", "AAA", _draft_payload(n), db_path=db_path)
    history = A.load_history("100", db_path=db_path)
    assert len(history) == 1
    assert history[0]["status"] == A.STATUS_DRAFT


def test_freeze_promotes_and_supersedes_prior_frozen(db_path):
    n = _normalized()
    # First set, frozen.
    A.save_draft("100", "AAA", _draft_payload(n), db_path=db_path)
    first = A.freeze_draft("100", "first", db_path=db_path)
    assert first["status"] == A.STATUS_FROZEN
    assert first["review_note"] == "first"

    # Second set, frozen -> supersedes the first.
    A.save_draft("100", "AAA", _draft_payload(n), db_path=db_path)
    second = A.freeze_draft("100", "second", db_path=db_path)

    active = A.load_active("100", A.STATUS_FROZEN, db_path=db_path)
    assert active["id"] == second["id"]
    history = A.load_history("100", db_path=db_path)
    superseded = [r for r in history if r["status"] == A.STATUS_SUPERSEDED]
    assert len(superseded) == 1
    assert superseded[0]["id"] == first["id"]
    assert superseded[0]["superseded_at"] is not None


def test_freeze_without_draft_returns_none(db_path):
    assert A.freeze_draft("999", db_path=db_path) is None


def test_update_draft_reclamps_and_marks_manual_once(db_path):
    n = _normalized()
    A.save_draft("100", "AAA", _draft_payload(n), db_path=db_path)

    # terminal_growth 0.09 is out of range -> clamp_assumptions caps it at 0.04.
    edited = copy.deepcopy(_assumptions())
    for s in ("bear", "base", "bull"):
        edited[s]["terminal_growth"] = 0.09

    first = A.update_draft("100", edited, sector_type=None, db_path=db_path)
    assert first["assumptions"]["base"]["terminal_growth"] == pytest.approx(0.04)
    assert first["sanity_notes"], "clamp should have produced at least one note"
    assert first["source_provider"] == "ollama+manual"

    # Editing again must NOT append a second +manual.
    second = A.update_draft("100", edited, sector_type=None, db_path=db_path)
    assert second["source_provider"] == "ollama+manual"


def test_update_draft_without_draft_returns_none(db_path):
    assert A.update_draft("999", _assumptions(), sector_type=None, db_path=db_path) is None


def test_load_active_decodes_json_columns(db_path):
    n = _normalized()
    A.save_draft("100", "AAA", _draft_payload(n), db_path=db_path)
    draft = A.load_active("100", A.STATUS_DRAFT, db_path=db_path)
    assert isinstance(draft["assumptions"], dict)
    assert isinstance(draft["script_baseline"], dict)
    assert isinstance(draft["sanity_notes"], list)
    # Raw *_json keys are not leaked to callers.
    assert "assumptions_json" not in draft


# --------------------------------------------------------------------------
# Fingerprint + freshness
# --------------------------------------------------------------------------

def test_fingerprint_stable_regardless_of_dict_ordering():
    n1 = _normalized()
    n2 = _normalized()
    # Reverse the annual concept insertion order.
    n2["annual"] = dict(reversed(list(n2["annual"].items())))
    assert A.fingerprint_annual(n1) == A.fingerprint_annual(n2)


def test_fingerprint_unchanged_by_quarterly_and_price():
    n1 = _normalized()
    n2 = _normalized()
    n2["quarterly"] = {"Revenue": [_record(2024, 300.0, "2024-03-31")]}
    assert A.fingerprint_annual(n1) == A.fingerprint_annual(n2)


def test_fingerprint_changes_on_new_annual_fy():
    n1 = _normalized()
    n2 = _normalized()
    n2["annual"]["Revenue"] = [_record(2024, 1100.0)] + n2["annual"]["Revenue"]
    assert A.fingerprint_annual(n1) != A.fingerprint_annual(n2)


def test_fingerprint_changes_on_restated_value():
    assert A.fingerprint_annual(_normalized(1000.0)) != A.fingerprint_annual(_normalized(1050.0))


def test_is_fresh_requires_both_fy_and_fingerprint(db_path):
    n = _normalized()
    A.save_draft("100", "AAA", _draft_payload(n, fy=2023), db_path=db_path)
    A.freeze_draft("100", db_path=db_path)
    frozen = A.load_active("100", A.STATUS_FROZEN, db_path=db_path)

    assert A.is_fresh(frozen, n, _metrics(fy=2023)) is True
    # Newer fundamental fy -> stale.
    assert A.is_fresh(frozen, n, _metrics(fy=2024)) is False
    # Restated facts -> fingerprint mismatch -> stale.
    assert A.is_fresh(frozen, _normalized(1050.0), _metrics(fy=2023)) is False
    # No row is never fresh.
    assert A.is_fresh(None, n, _metrics()) is False


# --------------------------------------------------------------------------
# analyze phase-1 resolution matrix (_resolve_analyze_phase1)
# --------------------------------------------------------------------------

def _args(strategy, ticker="AAA"):
    return argparse.Namespace(
        assumptions=strategy, ticker=ticker, years=5, no_cache=False
    )


def _resolve(strategy, cik, normalized, metrics, as_of=None, db_path=None):
    import sec_analyzer.cli as cli
    return cli._resolve_analyze_phase1(
        _args(strategy), cik, normalized, [], metrics, {}, as_of, None
    )


def _freeze_a_set(cik, normalized, db_path, fy=2023):
    A.save_draft(cik, "AAA", _draft_payload(normalized, fy=fy), db_path=db_path)
    A.freeze_draft(cik, db_path=db_path)


def test_resolve_llm_returns_no_override(db_path):
    override, note, stop, vp = _resolve("llm", "100", _normalized(), _metrics())
    assert override is None and stop is False and vp is None


def test_resolve_script_builds_deterministic_override(db_path):
    override, note, stop, vp = _resolve("script", "100", _normalized(), _metrics())
    assert override is not None
    assert override["_provider"] == "script"
    assert "assumptions" in override
    assert stop is False and vp is None


def test_resolve_auto_uses_fresh_frozen_set(db_path):
    n = _normalized()
    _freeze_a_set("100", n, db_path)
    override, note, stop, vp = _resolve("auto", "100", n, _metrics())
    assert override is not None
    assert override["_provider"] == "cached:ollama"
    assert override["_assumption_set_id"] is not None
    assert vp == "cached:ollama"
    assert stop is False
    assert "kullanıldı" in (note or "")


def test_resolve_auto_falls_back_to_script_when_no_frozen(db_path):
    override, note, stop, vp = _resolve("auto", "100", _normalized(), _metrics())
    assert override is not None
    assert override["_provider"] == "script"
    assert vp is None
    assert stop is False
    assert "script" in (note or "").lower()


def test_resolve_auto_falls_back_when_frozen_is_stale(db_path):
    n = _normalized()
    _freeze_a_set("100", n, db_path, fy=2022)  # older fy than current metrics
    override, note, stop, vp = _resolve("auto", "100", n, _metrics(fy=2023))
    assert override is not None
    assert override["_provider"] == "script"
    assert "bayat" in (note or "")


def test_resolve_frozen_strict_stops_when_missing(db_path):
    override, note, stop, vp = _resolve("frozen", "100", _normalized(), _metrics())
    assert override is None
    assert stop is True
    assert "durduruldu" in (note or "")


def test_resolve_frozen_strict_stops_when_stale(db_path):
    n = _normalized()
    _freeze_a_set("100", n, db_path, fy=2022)
    override, note, stop, vp = _resolve("frozen", "100", n, _metrics(fy=2023))
    assert stop is True and override is None


def test_resolve_frozen_uses_fresh_set(db_path):
    n = _normalized()
    _freeze_a_set("100", n, db_path)
    override, note, stop, vp = _resolve("frozen", "100", n, _metrics())
    assert override is not None and stop is False
    assert vp == "cached:ollama"


def test_resolve_asof_never_consults_cache_even_when_fresh(db_path):
    from datetime import date
    n = _normalized()
    _freeze_a_set("100", n, db_path)  # a fresh set exists...
    override, note, stop, vp = _resolve(
        "auto", "100", n, _metrics(), as_of=date(2022, 6, 30)
    )
    assert override is None  # ...but as-of mode ignores it
    assert stop is False
    assert "hindsight" in (note or "")


# --------------------------------------------------------------------------
# interpret(phase1_override=...) — no LLM, deterministic
# --------------------------------------------------------------------------

def _override_from_script(normalized, metrics):
    """A concrete phase-1 override (built deterministically) with provenance."""
    p1 = analyzer.build_script_phase1(normalized, [], metrics, "mature", None)
    p1["_provider"] = "cached:ollama"
    p1["_assumption_set_id"] = 42
    return p1


def test_interpret_with_override_skips_phase1(monkeypatch):
    called = {"propose": False}

    def _boom(*a, **k):
        called["propose"] = True
        raise AssertionError("propose_assumptions must not be called with an override")

    monkeypatch.setattr(analyzer, "propose_assumptions", _boom)

    n, m = _normalized(), _metrics()
    override = {
        "assumptions": _assumptions(), "sector_type": "mature",
        "hyper_growth_extras": None, "_provider": "cached:ollama",
        "_assumption_set_id": 42,
    }
    result = analyzer.interpret(
        n, [], provider="script", metrics=m, phase1_override=override
    )
    assert called["propose"] is False
    assert result["_assumption_set_id"] == 42
    assert result["_phase1_provider"] == "cached:ollama"


def test_interpret_with_same_override_is_deterministic():
    n, m = _normalized(), _metrics()
    override = _override_from_script(n, m)
    r1 = analyzer.interpret(n, [], provider="script", metrics=m, phase1_override=override)
    r2 = analyzer.interpret(n, [], provider="script", metrics=m, phase1_override=copy.deepcopy(override))
    assert r1["fair_value_range"] == r2["fair_value_range"]


# --------------------------------------------------------------------------
# save_verdict persists assumption_set_id
# --------------------------------------------------------------------------

def test_save_verdict_persists_assumption_set_id(db_path):
    from sec_analyzer.store.database import save_verdict

    result = {
        "fundamental_verdict": "MAKUL",
        "fair_value_range": {"bear": {}, "base": {}, "bull": {}},
        "_assumption_set_id": 7,
    }
    vid = save_verdict("AAA", "100", "1y", "cached:ollama", 50.0, result, db_path=db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT assumption_set_id, provider FROM verdicts WHERE id = ?", (vid,)
        ).fetchone()
    finally:
        conn.close()
    assert row["assumption_set_id"] == 7
    assert row["provider"] == "cached:ollama"


def test_save_verdict_null_assumption_set_id_when_absent(db_path):
    from sec_analyzer.store.database import save_verdict

    result = {"fundamental_verdict": "MAKUL",
              "fair_value_range": {"bear": {}, "base": {}, "bull": {}}}
    vid = save_verdict("AAA", "100", "1y", "script", 50.0, result, db_path=db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT assumption_set_id FROM verdicts WHERE id = ?", (vid,)
        ).fetchone()
    finally:
        conn.close()
    assert row["assumption_set_id"] is None


# --------------------------------------------------------------------------
# cmd_analyze end-to-end wiring (boundaries stubbed, interpret captured)
# --------------------------------------------------------------------------

def _stub_analyze_boundaries(monkeypatch, normalized, captured):
    """Stub every network/LLM edge cmd_analyze touches; capture interpret's
    phase1_override. save_verdict stays real (writes the tmp DB)."""
    import sec_analyzer.cli as cli

    monkeypatch.setattr(cli, "_fetch_normalize_store", lambda args: ("100", "AAA", normalized, []))
    monkeypatch.setattr(cli, "_fetch_price_and_technical", lambda t, h, nc, ao: (50.0, ao, None, None))
    monkeypatch.setattr(cli, "_fetch_analyst_targets", lambda t, nc: None)
    monkeypatch.setattr(cli, "_fetch_risk_free_asof", lambda ao, nc: None)
    monkeypatch.setattr(cli, "_fetch_submissions", lambda cik, t, nc: {})
    monkeypatch.setattr(cli, "_fetch_catalyst", lambda s, t, as_of=None: None)
    monkeypatch.setattr(cli, "_detect_filing_events", lambda s, as_of=None: [])
    monkeypatch.setattr(cli, "_enrich_sector_momentum", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_attach_momentum", lambda *a, **k: None)

    def _fake_interpret(*args, **kwargs):
        captured["phase1_override"] = kwargs.get("phase1_override")
        captured["called"] = True
        return {
            "fundamental_verdict": "MAKUL",
            "fair_value_range": {"bear": {}, "base": {}, "bull": {}},
            "technical_verdict": "VERİ YOK",
            "profile_fit": {"verdict": "KISMEN", "reason": "t"},
            "summary": "t",
            "valuation": {"sector_type": "mature"},
        }

    monkeypatch.setattr(cli, "interpret", _fake_interpret)


def test_cmd_analyze_frozen_fresh_passes_override_and_saves_cached_provider(db_path, monkeypatch):
    import sec_analyzer.cli as cli
    from sec_analyzer.store.database import load_verdicts

    n = _normalized()
    _freeze_a_set("100", n, db_path)
    captured = {"called": False}
    _stub_analyze_boundaries(monkeypatch, n, captured)

    args = build_parser().parse_args(["analyze", "AAA", "--assumptions", "frozen"])
    args.func(args)

    assert captured["called"] is True
    override = captured["phase1_override"]
    assert override is not None and override["_provider"] == "cached:ollama"

    verdicts = load_verdicts("AAA", db_path=db_path)
    assert verdicts and verdicts[0]["provider"] == "cached:ollama"


def test_cmd_analyze_frozen_missing_stops_without_calling_interpret(db_path, monkeypatch):
    import sec_analyzer.cli as cli

    n = _normalized()
    captured = {"called": False}
    _stub_analyze_boundaries(monkeypatch, n, captured)

    args = build_parser().parse_args(["analyze", "AAA", "--assumptions", "frozen"])
    args.func(args)

    assert captured["called"] is False  # strict frozen mode short-circuited


def test_cmd_analyze_llm_passes_no_override(db_path, monkeypatch):
    n = _normalized()
    captured = {"called": False}
    _stub_analyze_boundaries(monkeypatch, n, captured)

    args = build_parser().parse_args(["analyze", "AAA", "--assumptions", "llm"])
    args.func(args)

    assert captured["called"] is True
    assert captured["phase1_override"] is None
