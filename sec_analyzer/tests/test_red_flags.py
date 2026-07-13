"""Unit tests for sec_analyzer.normalize.red_flags.

Like ``test_ratios.py``/``test_metrics.py``, fixtures build the normalized
``annual`` bucket shape directly rather than round-tripping through a full
companyfacts document. Since ``detect_red_flags`` takes the already-computed
``metrics`` dict (not raw facts) for several of its rules, those fixtures
build a minimal hand-crafted ``metrics`` dict with just the fields each rule
reads.
"""

from sec_analyzer.normalize.red_flags import detect_red_flags

_CONCEPTS = ["Receivables", "Revenue", "NetIncome", "OperatingCashFlow"]


def _record(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(annual_overrides):
    annual = {c: annual_overrides.get(c) for c in _CONCEPTS}
    return {
        "cik": 1, "entity_name": "Red Flag Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CONCEPTS},
        "missing": [c for c in _CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CONCEPTS},
    }


def _codes(flags):
    return {f["code"] for f in flags}


def test_receivables_outpace_fires_on_two_consecutive_years():
    normalized = _normalized(
        {
            "Receivables": [
                _record(2023, 170),
                _record(2022, 130),
                _record(2021, 100),
            ],
            "Revenue": [
                _record(2023, 900),
                _record(2022, 850),
                _record(2021, 800),
            ],
        }
    )
    flags = detect_red_flags(normalized, ratios=[], metrics={"latest_fy": 2023})

    assert "RECEIVABLES_OUTPACE" in _codes(flags)
    flag = next(f for f in flags if f["code"] == "RECEIVABLES_OUTPACE")
    assert "FY2023" in flag["detail"] and "FY2022" in flag["detail"]


def test_receivables_outpace_does_not_fire_for_single_year():
    """Only ONE year of outpacing (2023) -- 2022 growth doesn't outpace --
    must not be enough to fire (streak requirement is 2+ consecutive years)."""
    normalized = _normalized(
        {
            "Receivables": [
                _record(2023, 170),
                _record(2022, 100),
                _record(2021, 100),
            ],
            "Revenue": [
                _record(2023, 900),
                _record(2022, 850),
                _record(2021, 800),
            ],
        }
    )
    flags = detect_red_flags(normalized, ratios=[], metrics={"latest_fy": 2023})

    assert "RECEIVABLES_OUTPACE" not in _codes(flags)


def test_receivables_outpace_skipped_when_receivables_missing():
    normalized = _normalized({"Revenue": [_record(2023, 900), _record(2022, 800)]})
    flags = detect_red_flags(normalized, ratios=[], metrics={"latest_fy": 2023})

    assert "RECEIVABLES_OUTPACE" not in _codes(flags)


def test_ocf_negative_fires_when_profitable_but_cash_negative():
    normalized = _normalized(
        {
            "NetIncome": [_record(2023, 500)],
            "OperatingCashFlow": [_record(2023, -100)],
        }
    )
    flags = detect_red_flags(normalized, ratios=[], metrics={"latest_fy": 2023})

    assert "OCF_NEGATIVE" in _codes(flags)


def test_ocf_negative_does_not_fire_when_both_positive():
    normalized = _normalized(
        {
            "NetIncome": [_record(2023, 500)],
            "OperatingCashFlow": [_record(2023, 100)],
        }
    )
    flags = detect_red_flags(normalized, ratios=[], metrics={"latest_fy": 2023})

    assert "OCF_NEGATIVE" not in _codes(flags)


def test_dilution_fires_above_five_percent_threshold():
    flags = detect_red_flags(_normalized({}), ratios=[], metrics={"latest_fy": 2023, "shares_yoy": 0.10})
    assert "DILUTION" in _codes(flags)


def test_dilution_does_not_fire_at_or_below_threshold():
    flags = detect_red_flags(_normalized({}), ratios=[], metrics={"latest_fy": 2023, "shares_yoy": 0.05})
    assert "DILUTION" not in _codes(flags)

    flags = detect_red_flags(_normalized({}), ratios=[], metrics={"latest_fy": 2023, "shares_yoy": None})
    assert "DILUTION" not in _codes(flags)


def test_sbc_high_fires_above_ten_percent_threshold():
    flags = detect_red_flags(_normalized({}), ratios=[], metrics={"latest_fy": 2023, "sbc_revenue": 0.15})
    assert "SBC_HIGH" in _codes(flags)


def test_sbc_high_does_not_fire_below_threshold():
    flags = detect_red_flags(_normalized({}), ratios=[], metrics={"latest_fy": 2023, "sbc_revenue": 0.05})
    assert "SBC_HIGH" not in _codes(flags)


def test_cyclical_trap_fires_at_peak_margin_with_low_pe():
    ratios = [
        {"fy": 2023, "net_margin": 0.20},
        {"fy": 2022, "net_margin": 0.15},
        {"fy": 2021, "net_margin": 0.10},
        {"fy": 2020, "net_margin": 0.18},
    ]
    metrics = {"latest_fy": 2023, "pe": 10.0}
    flags = detect_red_flags(_normalized({}), ratios, metrics, horizon="5y")

    assert "CYCLICAL_TRAP" in _codes(flags)
    flag = next(f for f in flags if f["code"] == "CYCLICAL_TRAP")
    # The 5y-horizon wording should be used since horizon="5y" was passed.
    assert "5 yıllık" in flag["detail"]


def test_cyclical_trap_does_not_fire_with_fewer_than_four_years():
    ratios = [
        {"fy": 2023, "net_margin": 0.20},
        {"fy": 2022, "net_margin": 0.15},
        {"fy": 2021, "net_margin": 0.10},
    ]
    metrics = {"latest_fy": 2023, "pe": 10.0}
    flags = detect_red_flags(_normalized({}), ratios, metrics)

    assert "CYCLICAL_TRAP" not in _codes(flags)


def test_cyclical_trap_does_not_fire_when_pe_is_high():
    ratios = [
        {"fy": 2023, "net_margin": 0.20},
        {"fy": 2022, "net_margin": 0.15},
        {"fy": 2021, "net_margin": 0.10},
        {"fy": 2020, "net_margin": 0.18},
    ]
    metrics = {"latest_fy": 2023, "pe": 25.0}
    flags = detect_red_flags(_normalized({}), ratios, metrics)

    assert "CYCLICAL_TRAP" not in _codes(flags)


def test_cyclical_trap_does_not_fire_when_margin_not_near_peak():
    ratios = [
        {"fy": 2023, "net_margin": 0.05},
        {"fy": 2022, "net_margin": 0.15},
        {"fy": 2021, "net_margin": 0.10},
        {"fy": 2020, "net_margin": 0.20},
    ]
    metrics = {"latest_fy": 2023, "pe": 10.0}
    flags = detect_red_flags(_normalized({}), ratios, metrics)

    assert "CYCLICAL_TRAP" not in _codes(flags)


def test_clean_company_yields_no_flags():
    """A filer with receivables growing in line with revenue, positive OCF,
    negligible dilution/SBC, and margins well off their historical peak
    should trigger nothing."""
    normalized = _normalized(
        {
            "Receivables": [
                _record(2023, 108),
                _record(2022, 104),
                _record(2021, 100),
            ],
            "Revenue": [
                _record(2023, 1080),
                _record(2022, 1040),
                _record(2021, 1000),
            ],
            "NetIncome": [_record(2023, 50)],
            "OperatingCashFlow": [_record(2023, 120)],
        }
    )
    ratios = [
        {"fy": 2023, "net_margin": 0.05},
        {"fy": 2022, "net_margin": 0.15},
        {"fy": 2021, "net_margin": 0.10},
        {"fy": 2020, "net_margin": 0.20},
    ]
    metrics = {
        "latest_fy": 2023,
        "pe": 25.0,
        "shares_yoy": 0.01,
        "sbc_revenue": 0.02,
    }
    assert detect_red_flags(normalized, ratios, metrics) == []


def test_empty_inputs_yield_no_flags_without_raising():
    assert detect_red_flags({}, [], {}) == []
    assert detect_red_flags(None, None, None) == []
