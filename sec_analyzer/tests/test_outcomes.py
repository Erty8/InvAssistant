"""Unit tests for ``sec_analyzer.backtest.outcomes``.

``evaluate_outcomes`` is the key end-to-end test: it reads verdicts from a
tmp SQLite DB, resolves forward/benchmark prices via a monkeypatched
``get_price_history``, and writes ``verdict_outcomes`` rows. Everything here
is offline -- no real network/price fetch, no LLM.
"""

from datetime import date

import pandas as pd
import pytest

from sec_analyzer.backtest import outcomes
from sec_analyzer.store import database


def _price_df(rows):
    """Build a minimal ``get_price_history``-shaped DataFrame: a DatetimeIndex
    (ascending) and a single "Close" column, from a list of (date_str, close)."""
    if not rows:
        return pd.DataFrame({"Close": []}, index=pd.to_datetime([]))
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"Close": [r[1] for r in rows]}, index=idx)


# ---------------------------------------------------------------------------
# classify_hit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict, rel_return, expected",
    [
        ("UCUZ", 0.1, True),
        ("UCUZ", -0.1, False),
        ("UCUZ", 0.0, False),  # rel_return > 0 is strict; exactly 0 is not a hit
        ("PAHALI", -0.1, True),
        ("PAHALI", 0.1, False),
        ("PAHALI", 0.0, False),  # rel_return < 0 is strict; exactly 0 is not a hit
        ("MAKUL", 0.5, None),  # neutral claim -- never a binary hit
        ("MAKUL", -0.5, None),
        ("YÜKSEK BEKLENTİ FİYATLANMIŞ", 5.0, None),  # referee label -- not binary
        ("MODEL-PİYASA AYRIŞMASI", -5.0, None),  # referee label -- not binary
        (None, 0.1, None),  # missing verdict
        ("", 0.1, None),  # falsy verdict
        ("SOME-UNKNOWN-LABEL", 0.1, None),  # unrecognized label
    ],
)
def test_classify_hit_branches(verdict, rel_return, expected):
    assert outcomes.classify_hit(verdict, rel_return) is expected


def test_classify_hit_none_rel_return_is_none_even_for_binary_verdicts():
    # A missing realized return means nothing to grade, regardless of verdict.
    assert outcomes.classify_hit("UCUZ", None) is None
    assert outcomes.classify_hit("PAHALI", None) is None
    assert outcomes.classify_hit(None, None) is None


# ---------------------------------------------------------------------------
# _add_years
# ---------------------------------------------------------------------------


def test_add_years_normal_date():
    assert outcomes._add_years(date(2020, 6, 30), 1) == date(2021, 6, 30)
    assert outcomes._add_years(date(2020, 6, 30), 3) == date(2023, 6, 30)


def test_add_years_feb29_clamps_to_feb28_in_non_leap_target_year():
    # 2020 is a leap year (Feb 29 exists); 2020 + 1 = 2021, NOT a leap year,
    # so date.replace(year=2021) would raise ValueError -- caught and
    # clamped to Feb 28 instead.
    assert outcomes._add_years(date(2020, 2, 29), 1) == date(2021, 2, 28)


def test_add_years_feb29_to_another_leap_year_stays_exact():
    # 2020 + 4 = 2024, also a leap year -> no clamping needed.
    assert outcomes._add_years(date(2020, 2, 29), 4) == date(2024, 2, 29)


# ---------------------------------------------------------------------------
# _close_on_or_before
# ---------------------------------------------------------------------------


def test_close_on_or_before_returns_last_close_at_or_before_target():
    df = _price_df([("2020-01-02", 100.0), ("2020-06-01", 110.0), ("2020-12-01", 120.0)])
    assert outcomes._close_on_or_before(df, date(2020, 7, 1)) == pytest.approx(110.0)


def test_close_on_or_before_exact_date_match():
    df = _price_df([("2020-01-02", 100.0), ("2020-06-30", 115.0)])
    assert outcomes._close_on_or_before(df, date(2020, 6, 30)) == pytest.approx(115.0)


def test_close_on_or_before_returns_none_when_target_before_first_row():
    df = _price_df([("2020-06-01", 110.0)])
    assert outcomes._close_on_or_before(df, date(2020, 1, 1)) is None


def test_close_on_or_before_returns_none_for_empty_df():
    assert outcomes._close_on_or_before(_price_df([]), date(2020, 1, 1)) is None


def test_close_on_or_before_returns_none_for_none_df():
    assert outcomes._close_on_or_before(None, date(2020, 1, 1)) is None


# ---------------------------------------------------------------------------
# is_referee_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("YÜKSEK BEKLENTİ FİYATLANMIŞ", True),
        ("MODEL-PİYASA AYRIŞMASI", True),
        ("UCUZ", False),
        ("PAHALI", False),
        ("MAKUL", False),
        (None, False),
        ("", False),
        ("SOME-UNKNOWN-LABEL", False),
    ],
)
def test_is_referee_label(label, expected):
    assert outcomes.is_referee_label(label) is expected


# ---------------------------------------------------------------------------
# evaluate_outcomes -- the key end-to-end test.
#
# Hand-verified arithmetic:
#   ref_price (AAPL, given explicitly to save_verdict)   = 100.0
#   AAPL forward close @ 2021-06-30 (ref + 1y)            = 130.0
#   SPY  close @ 2020-06-30 (ref date)                    = 300.0
#   SPY  close @ 2021-06-30 (ref + 1y)                    = 330.0
#
#   abs_return = fwd/ref - 1               = 130/100 - 1          = 0.30
#   spy_return =                           = 330/300 - 1          = 0.10
#   rel_return = abs_return - spy_return                          = 0.20
#   hit = classify_hit("UCUZ", 0.20) = (0.20 > 0)                 = True
#
# `today` = 2021-07-15, so the +1y window (fwd_date 2021-06-30) has
# matured, but the +3y window (fwd_date 2023-06-30) has not -- it must be
# counted under `skipped_immature` and NOT written to verdict_outcomes.
# ---------------------------------------------------------------------------


def _canned_price_history(monkeypatch):
    frames = {
        "AAPL": _price_df([("2020-06-30", 100.0), ("2021-06-30", 130.0)]),
        "SPY": _price_df([("2020-06-30", 300.0), ("2021-06-30", 330.0)]),
    }

    def _fake_get_price_history(ticker, no_cache=False):
        key = str(ticker).strip().upper()
        if key in frames:
            return frames[key], "stub"
        raise outcomes.PriceDataError(f"no canned price history for {key!r}")

    monkeypatch.setattr(outcomes, "get_price_history", _fake_get_price_history)


def _insert_ucuz_verdict(db_path):
    result = {
        "fundamental_verdict": "UCUZ",
        "fair_value_range": {"bear": {}, "base": {}, "bull": {}},
    }
    return database.save_verdict(
        "AAPL", "320193", "1y", "script", 100.0, result,
        db_path=db_path, as_of="2020-06-30",
    )


def test_evaluate_outcomes_computes_hand_verified_abs_rel_and_hit(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite3")
    verdict_id = _insert_ucuz_verdict(db_path)
    _canned_price_history(monkeypatch)

    counts = outcomes.evaluate_outcomes(db_path=db_path, today=date(2021, 7, 15))

    assert counts["verdicts_seen"] == 1
    assert counts["evaluated"] == 1
    assert counts["skipped_immature"] > 0  # the +3y horizon hasn't matured yet

    rows = database.load_outcomes(db_path)
    assert len(rows) == 1  # only the matured +1y horizon was written
    row = rows[0]
    assert row["verdict_id"] == verdict_id
    assert row["horizon"] == "1y"
    assert row["ref_date"] == "2020-06-30"
    assert row["ref_price"] == pytest.approx(100.0)
    assert row["fwd_date"] == "2021-06-30"
    assert row["fwd_price"] == pytest.approx(130.0)
    assert row["abs_return"] == pytest.approx(0.30)
    assert row["rel_return"] == pytest.approx(0.20)
    assert row["hit"] == 1  # True stored as 1
    assert row["evaluated_at"] == "2021-07-15"


def test_evaluate_outcomes_is_idempotent_across_repeated_calls(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite3")
    _insert_ucuz_verdict(db_path)
    _canned_price_history(monkeypatch)

    outcomes.evaluate_outcomes(db_path=db_path, today=date(2021, 7, 15))
    outcomes.evaluate_outcomes(db_path=db_path, today=date(2021, 7, 15))

    rows = database.load_outcomes(db_path)
    assert len(rows) == 1  # UPSERT keyed on (verdict_id, horizon) -- no duplicate row


def test_evaluate_outcomes_with_no_verdicts_returns_zero_counts(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite3")
    database.init_db(db_path)
    _canned_price_history(monkeypatch)

    counts = outcomes.evaluate_outcomes(db_path=db_path, today=date(2021, 7, 15))

    assert counts == {
        "verdicts_seen": 0, "evaluated": 0, "skipped_immature": 0, "skipped_no_data": 0,
    }
