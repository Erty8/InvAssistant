"""Unit tests for ``sec_analyzer.backtest.runner``.

``read_tickers_file``/``parse_dates`` are pure parsing helpers. The critical
test is ``run_backtest`` never invoking an AI/LLM provider (SPEC.md's --no-ai
contract for backtests): every network/LLM boundary ``_run_one`` touches
(the ``cli`` fetch helpers and ``interpret.analyzer.interpret`` itself, both
imported lazily inside ``_run_one``) is monkeypatched to a cheap stub, and a
spy on ``interpret`` records the ``provider`` kwarg it actually received.
"""

from datetime import date, timedelta

import pytest

import sec_analyzer.cli as cli_module
import sec_analyzer.interpret.analyzer as analyzer_module
from sec_analyzer.backtest import runner
from sec_analyzer.store import database

# ---------------------------------------------------------------------------
# read_tickers_file
# ---------------------------------------------------------------------------


def test_read_tickers_file_parses_comments_commas_and_dedups(tmp_path):
    path = tmp_path / "watchlist.txt"
    path.write_text(
        "# a watchlist\n"
        "aapl, msft\n"
        "\n"
        "GOOGL  # trailing comment\n"
        "msft\n",
        encoding="utf-8",
    )

    result = runner.read_tickers_file(str(path))

    assert result == ["AAPL", "MSFT", "GOOGL"]


def test_read_tickers_file_blank_and_comment_only_file_returns_empty_list(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# nothing here\n\n   \n", encoding="utf-8")

    assert runner.read_tickers_file(str(path)) == []


def test_read_tickers_file_space_separated_line(tmp_path):
    path = tmp_path / "watchlist.txt"
    path.write_text("AAPL MSFT GOOGL\n", encoding="utf-8")

    assert runner.read_tickers_file(str(path)) == ["AAPL", "MSFT", "GOOGL"]


# ---------------------------------------------------------------------------
# parse_dates
# ---------------------------------------------------------------------------


def test_parse_dates_parses_comma_separated_iso_dates():
    result = runner.parse_dates("2020-06-30,2022-06-30")
    assert result == [date(2020, 6, 30), date(2022, 6, 30)]


def test_parse_dates_dedups_repeated_dates_preserving_order():
    result = runner.parse_dates("2020-06-30,2022-06-30,2020-06-30")
    assert result == [date(2020, 6, 30), date(2022, 6, 30)]


def test_parse_dates_rejects_a_future_date():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError):
        runner.parse_dates(tomorrow)


def test_parse_dates_ignores_blank_tokens():
    result = runner.parse_dates(" 2020-06-30 , , 2022-06-30 ")
    assert result == [date(2020, 6, 30), date(2022, 6, 30)]


# ---------------------------------------------------------------------------
# run_backtest -- the critical no-AI test (SPEC.md's --no-ai contract).
# ---------------------------------------------------------------------------


def _normalized_fixture():
    return {
        "entity_name": "Apple Inc.",
        "currency": "USD",
        "annual": {
            "Revenue": [
                {"fy": 2020, "period_end": "2020-12-31", "value": 100.0},
                {"fy": 2019, "period_end": "2019-12-31", "value": 90.0},
            ],
        },
        "quarterly": {},
        "missing": [],
        "matched_tags": {},
    }


def _wire_offline_backtest_stubs(monkeypatch, captured_providers):
    """Stub every network/LLM boundary ``runner._run_one`` touches (all
    lazily imported from ``sec_analyzer.cli`` and
    ``sec_analyzer.interpret.analyzer`` at call time), so ``run_backtest``
    exercises real orchestration logic (argument threading, save_verdict)
    without ever hitting SEC EDGAR, a price feed, or an LLM."""

    def _fake_fetch_normalize_store(args):
        ratios = [{"fy": 2020, "period_end": "2020-12-31"}]
        return "320193", "Apple Inc.", _normalized_fixture(), ratios

    def _fake_fetch_price_and_technical(ticker, horizon, no_cache, as_of):
        return 100.0, as_of, None, None

    def _fake_fetch_risk_free_asof(as_of, no_cache):
        return None

    def _fake_fetch_submissions(cik, ticker, no_cache):
        return None

    def _fake_fetch_catalyst(submissions, ticker, as_of=None):
        return None

    def _fake_detect_filing_events(submissions, as_of=None):
        return []

    def _fake_interpret(*args, **kwargs):
        captured_providers.append(kwargs.get("provider"))
        return {
            "fundamental_verdict": "MAKUL",
            "fair_value_range": {"bear": {}, "base": {}, "bull": {}},
            "valuation": {"sector_type": "mature"},
        }

    monkeypatch.setattr(cli_module, "_fetch_normalize_store", _fake_fetch_normalize_store)
    monkeypatch.setattr(cli_module, "_fetch_price_and_technical", _fake_fetch_price_and_technical)
    monkeypatch.setattr(cli_module, "_fetch_risk_free_asof", _fake_fetch_risk_free_asof)
    monkeypatch.setattr(cli_module, "_fetch_submissions", _fake_fetch_submissions)
    monkeypatch.setattr(cli_module, "_fetch_catalyst", _fake_fetch_catalyst)
    monkeypatch.setattr(cli_module, "_detect_filing_events", _fake_detect_filing_events)
    monkeypatch.setattr(analyzer_module, "interpret", _fake_interpret)


def test_run_backtest_never_calls_an_ai_provider(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite3")
    captured_providers = []
    _wire_offline_backtest_stubs(monkeypatch, captured_providers)

    tally = runner.run_backtest(["AAPL"], [date(2020, 6, 30)], db_path=db_path, evaluate=False)

    assert tally["cells"] == 1
    assert tally["ok"] == 1
    assert tally["error"] == 0
    # The single most important assertion: interpret() was called with
    # provider="script" -- never "ollama"/"anthropic" -- proving --no-ai.
    assert captured_providers == ["script"]


def test_run_backtest_saves_verdicts_with_as_of_set(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite3")
    _wire_offline_backtest_stubs(monkeypatch, [])

    runner.run_backtest(["AAPL"], [date(2020, 6, 30)], db_path=db_path, evaluate=False)

    rows = database.load_verdicts_for_outcomes(db_path)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["as_of"] == "2020-06-30"


def test_run_backtest_grid_covers_every_ticker_and_date_pair(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite3")
    captured_providers = []
    _wire_offline_backtest_stubs(monkeypatch, captured_providers)

    tally = runner.run_backtest(
        ["AAPL", "MSFT"], [date(2020, 6, 30), date(2021, 6, 30)],
        db_path=db_path, evaluate=False,
    )

    assert tally["cells"] == 4  # 2 tickers x 2 dates
    assert tally["ok"] == 4
    assert captured_providers == ["script"] * 4  # never AI, for every cell
