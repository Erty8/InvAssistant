"""Unit tests for ``sec_analyzer.calibrate.save_calibration_snapshot``
(point-in-time / as-of persistence).

``run_calibration`` itself is a thin orchestration wrapper over
``cli._fetch_normalize_store``/``_fetch_price_and_technical``/``interpret``
(all network/LLM-touching); those are exercised indirectly via
``test_calibrate_method_slug.py``'s ``_method_slug`` unit tests and the CLI's
own as-of tests. This file focuses on the piece that's cleanly unit-testable
in isolation: snapshot persistence gaining an ``"as_of"`` field and an
``_asof-<date>`` filename segment.
"""

import json

from sec_analyzer.calibrate import save_calibration_snapshot
from sec_analyzer.config import Config


def _rows():
    return [
        {"ticker": "AAPL", "status": "ok", "price": 128.4, "fv_base_mid": 150.0, "ratio": 1.17, "method": "dcf"},
    ]


def _summary():
    return {
        "count": 1, "median": 1.17, "mean": 1.17, "p25": 1.17, "p75": 1.17,
        "bucket_under_0.8": 0, "bucket_0.8_1.2": 1, "bucket_over_1.2": 0,
    }


def test_save_calibration_snapshot_with_as_of_records_it_in_payload_and_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "REPORTS_DIR", str(tmp_path))

    path = save_calibration_snapshot("run", _rows(), _summary(), as_of="2022-06-30")

    assert path is not None
    filename = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    assert "_asof-2022-06-30" in filename

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["as_of"] == "2022-06-30"
    assert payload["label"] == "run"
    assert payload["rows"] == _rows()
    assert payload["summary"] == _summary()


def test_save_calibration_snapshot_without_as_of_omits_filename_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "REPORTS_DIR", str(tmp_path))

    path = save_calibration_snapshot("run", _rows(), _summary())

    assert path is not None
    filename = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    assert "_asof-" not in filename

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["as_of"] is None


def test_save_calibration_snapshot_creates_missing_reports_dir(tmp_path, monkeypatch):
    nested = tmp_path / "nested" / "reports"
    monkeypatch.setattr(Config, "REPORTS_DIR", str(nested))

    path = save_calibration_snapshot("run", _rows(), _summary(), as_of="2022-06-30")

    assert nested.exists()
    assert path.startswith(str(nested))


def test_save_calibration_snapshot_failure_returns_none_not_raise(tmp_path, monkeypatch):
    # Point REPORTS_DIR at a path that collides with an existing FILE (not a
    # directory), so os.makedirs(..., exist_ok=True) fails -- this must
    # degrade to None, never raise, per the documented defensive-persistence
    # contract.
    blocking_file = tmp_path / "blocked"
    blocking_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(Config, "REPORTS_DIR", str(blocking_file))

    result = save_calibration_snapshot("run", _rows(), _summary(), as_of="2022-06-30")
    assert result is None
