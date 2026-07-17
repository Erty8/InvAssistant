"""Hand-verified numeric tests for ``valuation.revenue_dcf`` and the
``valuation.sector.detect_hyper_grower`` trigger (SPEC.md Sec.1-2).

Every numeric expectation in the ``revenue_first_dcf`` case is derived
independently by hand (shown in the comment above the assertions) and
checked against the implementation with ``pytest.approx``. See the module
docstring of ``test_valuation_dcf.py`` for the general methodology this
file follows.
"""

import pytest

from sec_analyzer.valuation.revenue_dcf import (
    implied_start_growth,
    implied_target_margin,
    revenue_first_dcf,
)
from sec_analyzer.valuation.sector import detect_hyper_grower

# ---------------------------------------------------------------------------
# 1. revenue_first_dcf hand-verified case (SPEC Sec.2.1)
# ---------------------------------------------------------------------------


def test_revenue_first_dcf_hand_verified_clean_case():
    # revenue0=1000, start_growth=0.40, terminal_growth=0.025, r=0.10,
    # current_margin=0.0, target_fcf_margin=0.20, steady_state_year=10,
    # shares0=100, annual_dilution=0.0, financing_shares=0.0.
    #
    # Growth path (fade from 0.40 to 0.025 over years 1..10, steady_state_
    # year=10 so the fraction denominator is steady_state_year-1=9):
    #   g_t = 0.40 + (0.025-0.40)*min(t-1,9)/9 = 0.40 - 0.375*min(t-1,9)/9
    #   g1  = 0.40 - 0.375*0/9 = 0.400000
    #   g2  = 0.40 - 0.375*1/9 = 0.358333
    #   g3  = 0.40 - 0.375*2/9 = 0.316667
    #   g4  = 0.40 - 0.375*3/9 = 0.275000
    #   g5  = 0.40 - 0.375*4/9 = 0.233333
    #   g6  = 0.40 - 0.375*5/9 = 0.191667
    #   g7  = 0.40 - 0.375*6/9 = 0.150000
    #   g8  = 0.40 - 0.375*7/9 = 0.108333
    #   g9  = 0.40 - 0.375*8/9 = 0.066667
    #   g10 = 0.40 - 0.375*9/9 = 0.025000  (== terminal_growth, as required)
    #
    # Margin path (converge from 0.0 to 0.20 over years 1..10, steady_state_
    # year=10): margin_t = 0.20 * min(t,10)/10 = 0.02*t.
    #   margin1..margin10 = 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16,
    #                        0.18, 0.20
    #
    # Revenue path (revenue_t = revenue_{t-1}*(1+g_t), revenue_0=1000):
    #   rev1  = 1000       *1.400000 = 1400.0000
    #   rev2  = 1400.0000  *1.358333 = 1901.6667
    #   rev3  = 1901.6667  *1.316667 = 2503.8611
    #   rev4  = 2503.8611  *1.275000 = 3192.4229
    #   rev5  = 3192.4229  *1.233333 = 3937.3216
    #   rev6  = 3937.3216  *1.191667 = 4691.9749
    #   rev7  = 4691.9749  *1.150000 = 5395.7711
    #   rev8  = 5395.7711  *1.108333 = 5980.3130
    #   rev9  = 5980.3130  *1.066667 = 6379.0005
    #   rev10 = 6379.0005  *1.025000 = 6538.4756
    #
    # FCF path (fcf_t = revenue_t * margin_t):
    #   fcf1  = 1400.0000*0.02 = 28.0000
    #   fcf2  = 1901.6667*0.04 = 76.0667
    #   fcf3  = 2503.8611*0.06 = 150.2317
    #   fcf4  = 3192.4229*0.08 = 255.3938
    #   fcf5  = 3937.3216*0.10 = 393.7322
    #   fcf6  = 4691.9749*0.12 = 563.0370
    #   fcf7  = 5395.7711*0.14 = 755.4080
    #   fcf8  = 5980.3130*0.16 = 956.8501
    #   fcf9  = 6379.0005*0.18 = 1148.2201
    #   fcf10 = 6538.4756*0.20 = 1307.6951
    #
    # Discounted at r=0.10, 1.10^t for t=1..10 = 1.1, 1.21, 1.331, 1.4641,
    # 1.61051, 1.771561, 1.9487171, 2.14358881, 2.357947691, 2.5937424601:
    #   pv1  = 28.0000  /1.1         = 25.4545
    #   pv2  = 76.0667  /1.21        = 62.8650
    #   pv3  = 150.2317 /1.331       = 112.8713
    #   pv4  = 255.3938 /1.4641      = 174.4374
    #   pv5  = 393.7322 /1.61051     = 244.4767
    #   pv6  = 563.0370 /1.771561    = 317.8197
    #   pv7  = 755.4080 /1.9487171   = 387.6437
    #   pv8  = 956.8501 /2.14358881  = 446.3776
    #   pv9  = 1148.2201/2.357947691 = 486.9574
    #   pv10 = 1307.6951/2.5937424601= 504.1731
    #   pv_sum = 25.4545+62.8650+112.8713+174.4374+244.4767+317.8197+387.6437
    #            +446.3776+486.9574+504.1731 = 2763.0764
    #
    # Terminal value: tv = fcf10*(1+terminal_growth)/(r-terminal_growth)
    #   = 1307.6951*1.025/0.075 = 1340.3874/0.075 = 17871.8320
    # pv(tv) = 17871.8320/2.5937424601 = 6890.3652
    #
    # ev = pv_sum + pv(tv) = 2763.0764 + 6890.3652 = 9653.4416
    # equity = ev (FCFE-direct, no net_debt subtraction) = 9653.4416
    # effective_shares = 100*(1+0)**10 + 0.0 = 100.0
    # per_share = 9653.4416/100 = 96.534416
    #
    # (Cross-checked against a full-precision scratch computation before
    # finalizing; the rounded figures above match to the displayed digits.)
    result = revenue_first_dcf(
        revenue0=1000.0, start_growth=0.40, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, target_fcf_margin=0.20, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0, financing_shares=0.0,
    )

    assert len(result["growth_path"]) == 10
    assert len(result["revenue_path"]) == 10
    assert len(result["margin_path"]) == 10
    assert len(result["fcf_path"]) == 10

    assert result["growth_path"][0] == pytest.approx(0.400000, rel=1e-4)
    assert result["growth_path"][4] == pytest.approx(0.233333, rel=1e-4)
    assert result["growth_path"][9] == pytest.approx(0.025000, rel=1e-4)

    assert result["margin_path"][0] == pytest.approx(0.02, rel=1e-4)
    assert result["margin_path"][9] == pytest.approx(0.20, rel=1e-4)

    assert result["revenue_path"][0] == pytest.approx(1400.0000, rel=1e-4)
    assert result["revenue_path"][4] == pytest.approx(3937.3216, rel=1e-4)
    assert result["revenue_path"][9] == pytest.approx(6538.4756, rel=1e-4)

    assert result["fcf_path"][0] == pytest.approx(28.0000, rel=1e-4)
    assert result["fcf_path"][9] == pytest.approx(1307.6951, rel=1e-4)

    assert result["tv"] == pytest.approx(17871.8320, rel=1e-4)
    assert result["ev"] == pytest.approx(9653.4416, rel=1e-4)
    assert result["equity"] == pytest.approx(9653.4416, rel=1e-4)
    assert result["effective_shares"] == pytest.approx(100.0)
    assert result["per_share"] == pytest.approx(96.534416, rel=1e-4)
    assert result["final_year_revenue"] == pytest.approx(6538.4756, rel=1e-4)
    assert result["revenue_multiple"] == pytest.approx(6.5384756, rel=1e-4)


def test_revenue_first_dcf_raises_on_programmer_errors():
    base_kwargs = dict(
        revenue0=1000.0, start_growth=0.40, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, target_fcf_margin=0.20, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0,
    )
    with pytest.raises(ValueError):
        revenue_first_dcf(**{**base_kwargs, "revenue0": 0.0})
    with pytest.raises(ValueError):
        revenue_first_dcf(**{**base_kwargs, "shares0": -1.0})
    with pytest.raises(ValueError):
        revenue_first_dcf(**{**base_kwargs, "discount_rate": 0.02, "terminal_growth": 0.025})
    with pytest.raises(ValueError):
        revenue_first_dcf(**{**base_kwargs, "steady_state_year": 0})
    # WP3: mature_discount_rate, when provided, must also clear the same
    # Gordon-defined bar as the flat discount_rate does.
    with pytest.raises(ValueError):
        revenue_first_dcf(**{**base_kwargs, "mature_discount_rate": 0.025})


# ---------------------------------------------------------------------------
# 1b. revenue_first_dcf -- mature_discount_rate fade (WP3, SPEC.md's
# "Damodaran fade": discount rate itself fades from a cohort rate toward a
# mature steady-state rate, cumulative-product discount factors instead of
# (1+r)**t, and the terminal value anchored at the MATURE rate). Both cases
# below hold growth (start_growth == terminal_growth == 0) and margin
# (current_margin == target_fcf_margin) FLAT on purpose, so revenue_path/
# fcf_path collapse to a trivial constant series and every number in the
# derivation below is purely about the discount-rate fade machinery, not
# entangled with the growth/margin fades already hand-verified above.
# ---------------------------------------------------------------------------


def test_revenue_first_dcf_mature_discount_rate_fade_hand_verified():
    # revenue0=1000, start_growth=terminal_growth=0.0 -> growth_path=[0,0,0]
    # -> revenue_path=[1000,1000,1000] (revenue_t=revenue_{t-1}*(1+0)).
    # current_margin=target_fcf_margin=0.10 -> margin_path=[0.10]*3 ->
    # fcf_path=[100,100,100]. steady_state_year=3, horizon=3 (short horizon
    # chosen purely so the discount-rate fade's 3 cells are easy to hand-
    # compute) shares0=100, annual_dilution=0.0, financing_shares=0.0.
    #
    # Cohort (year-1) discount_rate=0.20, mature_discount_rate=0.10.
    # _discount_path fades linearly, reaching mature_discount_rate exactly at
    # steady_state_year=3 (fraction denominator = steady_state_year-1 = 2):
    #   r_1 = 0.20 + (0.10-0.20)*min(0,2)/2 = 0.20
    #   r_2 = 0.20 + (0.10-0.20)*min(1,2)/2 = 0.20 - 0.05 = 0.15
    #   r_3 = 0.20 + (0.10-0.20)*min(2,2)/2 = 0.10  (== mature_discount_rate)
    #
    # Discounting is the CUMULATIVE product of (1+r_t), not (1+r)**t:
    #   cum_1 = 1.20
    #   cum_2 = 1.20*1.15 = 1.38
    #   cum_3 = 1.38*1.10 = 1.518
    #   pv1 = 100/1.20   = 83.333333
    #   pv2 = 100/1.38   = 72.463768
    #   pv3 = 100/1.518  = 65.876153
    #   pv_sum = 83.333333+72.463768+65.876153 = 221.673254
    #
    # Terminal value is a mature-firm perpetuity anchored at
    # mature_discount_rate (NOT the elevated cohort rate):
    #   tv = fcf_terminal*(1+terminal_growth)/(mature_discount_rate-terminal_growth)
    #      = 100*(1+0)/(0.10-0) = 1000
    #   pv(tv) = tv / cum_3 = 1000/1.518 = 658.761528
    #
    # ev = equity = 221.673254 + 658.761528 = 880.434783
    # effective_shares = 100*(1+0)**3 + 0 = 100
    # per_share = 880.434783/100 = 8.804348
    result = revenue_first_dcf(
        revenue0=1000.0, start_growth=0.0, terminal_growth=0.0, discount_rate=0.20,
        current_margin=0.10, target_fcf_margin=0.10, steady_state_year=3,
        shares0=100.0, annual_dilution=0.0, financing_shares=0.0, horizon=3,
        mature_discount_rate=0.10,
    )

    assert result["growth_path"] == pytest.approx([0.0, 0.0, 0.0])
    assert result["revenue_path"] == pytest.approx([1000.0, 1000.0, 1000.0])
    assert result["fcf_path"] == pytest.approx([100.0, 100.0, 100.0])

    # The additive discount_path key only appears when mature_discount_rate
    # is not None (per the docstring), and carries the fading rate series.
    assert "discount_path" in result
    assert result["discount_path"] == pytest.approx([0.20, 0.15, 0.10])

    assert result["tv"] == pytest.approx(1000.0)
    assert result["ev"] == pytest.approx(880.434783, rel=1e-5)
    assert result["equity"] == pytest.approx(880.434783, rel=1e-5)
    assert result["effective_shares"] == pytest.approx(100.0)
    assert result["per_share"] == pytest.approx(8.804348, rel=1e-5)


def test_revenue_first_dcf_mature_discount_rate_none_is_byte_for_byte_unchanged():
    # Identical inputs to the fade test above, EXCEPT mature_discount_rate is
    # omitted (None, the default) -> every year discounts at the flat
    # cohort discount_rate=0.20 with (1+r)**t (not the cumulative-product
    # fade path), and the terminal value also uses 0.20 (not a separate
    # mature rate) -- "byte-for-byte unchanged from before this parameter
    # existed", per the docstring.
    #   pv1 = 100/1.20    = 83.333333
    #   pv2 = 100/1.20**2 = 100/1.44   = 69.444444
    #   pv3 = 100/1.20**3 = 100/1.728  = 57.870370
    #   pv_sum = 83.333333+69.444444+57.870370 = 210.648148
    #   tv = 100*(1+0)/(0.20-0) = 500
    #   pv(tv) = 500/1.728 = 289.351852
    #   ev = 210.648148+289.351852 = 500.0 (exact)
    #   per_share = 500.0/100 = 5.0 (exact)
    #
    # 8.804348 (the fade case) > 5.0 (this flat case) is also the expected
    # DIRECTION: fading down to a lower mature rate values the (unchanged)
    # cash flows higher than discounting them at the permanently-elevated
    # cohort rate throughout -- confirms the fade isn't a no-op dressed up
    # as a new parameter.
    result = revenue_first_dcf(
        revenue0=1000.0, start_growth=0.0, terminal_growth=0.0, discount_rate=0.20,
        current_margin=0.10, target_fcf_margin=0.10, steady_state_year=3,
        shares0=100.0, annual_dilution=0.0, financing_shares=0.0, horizon=3,
        mature_discount_rate=None,
    )

    assert "discount_path" not in result
    assert result["tv"] == pytest.approx(500.0)
    assert result["ev"] == pytest.approx(500.0)
    assert result["per_share"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 2. detect_hyper_grower boundary tests (SPEC Sec.1)
# ---------------------------------------------------------------------------


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(revenue_2023):
    return {
        "cik": 1, "entity_name": "Hyper Test Co", "currency": "USD",
        "annual": {"Revenue": [_rec(2023, revenue_2023)]},
        "quarterly": {}, "missing": [], "matched_tags": {},
    }


def _metrics(revenue_cagr_5y, fcf, rnd_revenue=0.0, sbc_revenue=0.0, ps=None):
    return {
        "revenue_cagr_5y": revenue_cagr_5y, "revenue_cagr_3y": None,
        "latest_fy": 2023, "fcf": fcf,
        "rnd_revenue": rnd_revenue, "sbc_revenue": sbc_revenue,
        "ps": ps,
    }


def test_detect_hyper_grower_growth_exactly_threshold_not_triggered():
    # realized_cagr == 0.25 exactly -> no longer fails the strong tier's
    # strict ">" for its own sake (0.25 is now inside the gray zone,
    # (0.20, 0.25]); it fails because this call has no "ps" key, so
    # metrics.get("ps") is None -> high_ps is False -> the gray zone's
    # P/S gate blocks the trigger regardless of clause (a) firing.
    metrics = _metrics(revenue_cagr_5y=0.25, fcf=-10.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_growth_just_over_and_fcf_non_positive_triggers():
    # realized_cagr = 0.26 > 0.25, fcf = 0.0 <= 0 -> clause (a) fires.
    metrics = _metrics(revenue_cagr_5y=0.26, fcf=0.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is True
    assert "FCF negatif veya sıfır" in reasons
    assert reasons[0].startswith("Gelir CAGR %26.0")


def test_detect_hyper_grower_fcf_margin_exactly_threshold_clause_b_not_fired():
    # fcf_margin = 50/1000 = 0.05 exactly -> clause (b) does NOT fire.
    # fcf=50 > 0 so clause (a) also doesn't fire; opex_intensity=0 so (c)
    # doesn't fire either -> overall not triggered.
    metrics = _metrics(revenue_cagr_5y=0.30, fcf=50.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_fcf_margin_just_under_threshold_fires():
    # fcf_margin = 49/1000 = 0.049 < 0.05 -> clause (b) fires.
    metrics = _metrics(revenue_cagr_5y=0.30, fcf=49.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is True
    assert "FCF marjı %5'in altında (bastırılmış nakit akışı)" in reasons


def test_detect_hyper_grower_opex_intensity_exactly_threshold_clause_c_not_fired():
    # rnd_revenue + sbc_revenue = 0.25 + 0.15 = 0.40 exactly -> clause (c)
    # does NOT fire. fcf positive and margin high enough to keep (a)/(b)
    # from firing too -> overall not triggered.
    metrics = _metrics(revenue_cagr_5y=0.30, fcf=100.0, rnd_revenue=0.25, sbc_revenue=0.15)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_opex_intensity_just_over_threshold_fires():
    # rnd_revenue + sbc_revenue = 0.25 + 0.16 = 0.41 > 0.40 -> clause (c) fires.
    metrics = _metrics(revenue_cagr_5y=0.30, fcf=100.0, rnd_revenue=0.25, sbc_revenue=0.16)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is True
    assert "Ar-Ge + SBC / gelir %40'ı aşıyor (agresif büyüme yatırımı)" in reasons


def test_detect_hyper_grower_growth_condition_fails_returns_false_empty_regardless_of_clauses():
    # realized_cagr below threshold even though fcf<=0 would otherwise fire
    # clause (a) -> overall not triggered, reasons empty.
    metrics = _metrics(revenue_cagr_5y=0.10, fcf=-50.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_none_cagr_returns_false_empty():
    metrics = _metrics(revenue_cagr_5y=None, fcf=-50.0)
    metrics["revenue_cagr_3y"] = None
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_never_raises_on_malformed_metrics():
    triggered, reasons = detect_hyper_grower({}, [], {})
    assert triggered is False
    assert reasons == []
    # metrics=None entirely
    triggered, reasons = detect_hyper_grower(None, None, None)
    assert triggered is False
    assert reasons == []


# ---------------------------------------------------------------------------
# 2b. detect_hyper_grower gray-zone tier (SPEC Sec.8): CAGR in (0.20, 0.25]
#     AND a fired clause AND P/S > 8.0.
# ---------------------------------------------------------------------------


def test_detect_hyper_grower_gray_zone_fires_with_clause_a_and_high_ps():
    # CAGR=0.22 is in the gray zone, fcf=-10 <= 0 fires clause (a), and
    # ps=12.0 > 8.0 -> gray zone triggers.
    metrics = _metrics(revenue_cagr_5y=0.22, fcf=-10.0, ps=12.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is True
    assert "gri bölge" in reasons[0]
    assert "FCF negatif veya sıfır" in reasons
    # P/S renders as a plain multiple ("12.0x"), never "%12.0x" (percent AND
    # x is nonsensical).
    assert "(12.0x)" in reasons[0]
    assert "%12.0x" not in reasons[0]


def test_detect_hyper_grower_gray_zone_blocked_by_low_ps():
    # Same clause (a) as above, but ps=5.0 <= 8.0 -> P/S gate blocks it.
    metrics = _metrics(revenue_cagr_5y=0.22, fcf=-10.0, ps=5.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_gray_zone_blocked_by_no_clause():
    # High P/S but fcf=100 is positive, high margin, and no opex intensity
    # -> no clause fires -> gray zone doesn't trigger even with high P/S.
    metrics = _metrics(revenue_cagr_5y=0.22, fcf=100.0, ps=12.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_gray_zone_lower_boundary_not_qualified():
    # CAGR == 0.20 exactly is not strictly above the gray-zone min -> falls
    # below the gray zone entirely, regardless of clause/P/S.
    metrics = _metrics(revenue_cagr_5y=0.20, fcf=-10.0, ps=12.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is False
    assert reasons == []


def test_detect_hyper_grower_gray_zone_upper_boundary_at_25_triggers():
    # CAGR == 0.25 exactly is now inside the gray zone (<=0.25, not the
    # strong tier's strict >0.25); with clause (a) and high P/S it triggers.
    metrics = _metrics(revenue_cagr_5y=0.25, fcf=-10.0, ps=12.0)
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is True
    assert "gri bölge" in reasons[0]


def test_detect_hyper_grower_gray_zone_clause_c_via_opex_mrvl_like():
    # MRVL-like case: CAGR=0.225 in the gray zone, fcf positive (no clause
    # a/b) but rnd_revenue + sbc_revenue = 0.30 + 0.12 = 0.42 > 0.40 fires
    # clause (c), and ps=25.0 > 8.0 -> gray zone triggers.
    metrics = _metrics(
        revenue_cagr_5y=0.225, fcf=100.0, rnd_revenue=0.30, sbc_revenue=0.12, ps=25.0,
    )
    triggered, reasons = detect_hyper_grower(metrics, [], _normalized(1000.0))
    assert triggered is True
    assert "gri bölge" in reasons[0]
    assert "Ar-Ge + SBC / gelir %40'ı aşıyor (agresif büyüme yatırımı)" in reasons


# ---------------------------------------------------------------------------
# 3. implied_start_growth / implied_target_margin round-trips (SPEC Sec.2.2-2.3)
# ---------------------------------------------------------------------------


def test_implied_start_growth_recovers_known_growth():
    forward = revenue_first_dcf(
        revenue0=1000.0, start_growth=0.35, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, target_fcf_margin=0.20, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0, financing_shares=0.0,
    )
    price = forward["per_share"]

    implied = implied_start_growth(
        price=price, revenue0=1000.0, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, target_fcf_margin=0.20, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0, financing_shares=0.0,
    )

    assert implied == pytest.approx(0.35, abs=1e-3)


def test_implied_target_margin_recovers_known_margin():
    forward = revenue_first_dcf(
        revenue0=1000.0, start_growth=0.35, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, target_fcf_margin=0.22, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0, financing_shares=0.0,
    )
    price = forward["per_share"]

    implied = implied_target_margin(
        price=price, revenue0=1000.0, start_growth=0.35, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0, financing_shares=0.0,
    )

    assert implied == pytest.approx(0.22, abs=1e-3)


def test_implied_start_growth_returns_none_for_unusable_inputs():
    assert implied_start_growth(
        price=None, revenue0=1000.0, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, target_fcf_margin=0.20, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0,
    ) is None
    assert implied_start_growth(
        price=50.0, revenue0=1000.0, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, target_fcf_margin=0.20, steady_state_year=10,
        shares0=0.0, annual_dilution=0.0,
    ) is None


def test_implied_target_margin_returns_none_for_unusable_inputs():
    assert implied_target_margin(
        price=None, revenue0=1000.0, start_growth=0.35, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0,
    ) is None
    assert implied_target_margin(
        price=50.0, revenue0=0.0, start_growth=0.35, terminal_growth=0.025, discount_rate=0.10,
        current_margin=0.0, steady_state_year=10,
        shares0=100.0, annual_dilution=0.0,
    ) is None
