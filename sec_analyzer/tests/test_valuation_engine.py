"""Hand-verified numeric tests for ``valuation.triangulate``, ``valuation.sector``,
and the ``valuation.engine.run_valuation`` integration (SPEC.md Sec.8, 10, 11).

See the module docstring of ``test_valuation_dcf.py`` for the general
methodology (independent hand arithmetic in a comment above each assertion).
"""

import pytest

from sec_analyzer.valuation.engine import run_valuation
from sec_analyzer.valuation.sector import classify_sector
from sec_analyzer.valuation.triangulate import triangulate

# ---------------------------------------------------------------------------
# 8. triangulate (SPEC Sec.10)
# ---------------------------------------------------------------------------


def test_triangulate_all_three_agree_gives_high_confidence():
    # dcf: price=90 < band.lo=100 -> ucuz
    # reverse_dcf: implied=0.05 < ref(0.10) - 0.03 = 0.07 -> ucuz
    # multiples: pe_pct=20 < 30 -> ucuz
    # All three "ucuz" -> YÜKSEK confidence, direction "ucuz".
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["signals"] == {"dcf": "ucuz", "reverse_dcf": "ucuz", "multiples": "ucuz"}
    assert result["confidence"] == "YÜKSEK"
    assert result["direction"] == "ucuz"


def test_triangulate_two_of_three_agree_gives_medium_confidence():
    # dcf: price=90 < 100 -> ucuz
    # reverse_dcf: implied=0.05 < 0.07 -> ucuz
    # multiples: pe_pct=80 > 70 -> pahali
    # 2 agree (ucuz) / 1 disagrees -> ORTA, direction "ucuz".
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=80, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["signals"] == {"dcf": "ucuz", "reverse_dcf": "ucuz", "multiples": "pahali"}
    assert result["confidence"] == "ORTA"
    assert result["direction"] == "ucuz"


def test_triangulate_scattered_signals_give_low_confidence_and_unclear_direction():
    # dcf: price=110, inside [100,120] -> makul
    # reverse_dcf: implied=0.15 > ref(0.10)+0.03=0.13 -> pahali
    # multiples: pe_pct=20 < 30 -> ucuz
    # All three different -> no majority -> DÜŞÜK, "belirsiz".
    result = triangulate(
        price=110, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.15,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["signals"] == {"dcf": "makul", "reverse_dcf": "pahali", "multiples": "ucuz"}
    assert result["confidence"] == "DÜŞÜK"
    assert result["direction"] == "belirsiz"


def test_triangulate_two_or_more_no_data_signals_give_low_confidence():
    # price=None -> dcf signal "veri_yok"; implied_growth=None -> reverse_dcf
    # "veri_yok". Only multiples has data (pe_pct=50 -> makul). 2 "veri_yok"
    # -> DÜŞÜK regardless of the third signal.
    result = triangulate(
        price=None, dcf_base_band=None, implied_growth=None,
        realized_cagr=None, base_growth=0.10, pe_pct=50, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["signals"]["dcf"] == "veri_yok"
    assert result["signals"]["reverse_dcf"] == "veri_yok"
    assert result["confidence"] == "DÜŞÜK"
    assert result["direction"] == "belirsiz"


def test_triangulate_dcf_direction_boundaries_are_strict():
    # price exactly at band.lo/band.hi must be "makul" (strict < / > only).
    at_lo = triangulate(100, {"lo": 100, "hi": 120}, None, None, None, None, None, None, "mature")
    at_hi = triangulate(120, {"lo": 100, "hi": 120}, None, None, None, None, None, None, "mature")
    below_lo = triangulate(99.99, {"lo": 100, "hi": 120}, None, None, None, None, None, None, "mature")
    above_hi = triangulate(120.01, {"lo": 100, "hi": 120}, None, None, None, None, None, None, "mature")

    assert at_lo["signals"]["dcf"] == "makul"
    assert at_hi["signals"]["dcf"] == "makul"
    assert below_lo["signals"]["dcf"] == "ucuz"
    assert above_hi["signals"]["dcf"] == "pahali"


def test_triangulate_reverse_dcf_direction_boundaries_are_strict():
    # reference growth = base_growth = 0.10 (realized_cagr is None).
    # implied exactly at ref+0.03=0.13 or ref-0.03=0.07 -> "makul" (strict
    # inequalities only); just past either edge flips it.
    at_upper = triangulate(None, None, 0.13, None, 0.10, None, None, None, "mature")
    at_lower = triangulate(None, None, 0.07, None, 0.10, None, None, None, "mature")
    past_upper = triangulate(None, None, 0.1301, None, 0.10, None, None, None, "mature")
    past_lower = triangulate(None, None, 0.0699, None, 0.10, None, None, None, "mature")

    assert at_upper["signals"]["reverse_dcf"] == "makul"
    assert at_lower["signals"]["reverse_dcf"] == "makul"
    assert past_upper["signals"]["reverse_dcf"] == "pahali"
    assert past_lower["signals"]["reverse_dcf"] == "ucuz"


def test_triangulate_reverse_dcf_prefers_realized_cagr_over_base_growth():
    # When realized_cagr is provided it is the reference, NOT base_growth.
    # realized_cagr=0.20, implied=0.10: 0.10 < 0.20-0.03=0.17 -> ucuz (even
    # though base_growth=0.10 would have made it exactly "makul" against
    # itself).
    result = triangulate(None, None, 0.10, 0.20, 0.10, None, None, None, "mature")
    assert result["signals"]["reverse_dcf"] == "ucuz"


def test_triangulate_multiples_direction_boundaries_are_strict():
    # pct exactly 30/70 -> "makul"; just past either edge flips it.
    at_30 = triangulate(None, None, None, None, None, 30, None, None, "mature")
    at_70 = triangulate(None, None, None, None, None, 70, None, None, "mature")
    below_30 = triangulate(None, None, None, None, None, 29.9, None, None, "mature")
    above_70 = triangulate(None, None, None, None, None, 70.1, None, None, "mature")

    assert at_30["signals"]["multiples"] == "makul"
    assert at_70["signals"]["multiples"] == "makul"
    assert below_30["signals"]["multiples"] == "ucuz"
    assert above_70["signals"]["multiples"] == "pahali"


def test_triangulate_multiples_uses_ps_first_for_growth_unprofitable():
    # For growth_unprofitable, P/S is primary (fallback P/E then P/FCF).
    # pe_pct=90 (would be "pahali" if used) but ps_pct=10 is primary here
    # and must win: "ucuz".
    result = triangulate(None, None, None, None, None, 90, 10, None, "growth_unprofitable")
    assert result["signals"]["multiples"] == "ucuz"


# ---------------------------------------------------------------------------
# 8a. triangulate -- hyper-grower "yuksek_beklenti" DCF signal (HYPER_SPEC.md
# Sec.4): base band [100,120], bull band hi=150.
# ---------------------------------------------------------------------------


def test_triangulate_hyper_growth_price_between_base_and_bull_hi_is_yuksek_beklenti():
    # price=130: above base.hi=120 but at/below bull.hi=150 -> "yuksek_beklenti",
    # not "pahali". With only 1 substantive signal (reverse_dcf/multiples
    # both "veri_yok" here), the pre-existing "2+ no-data signals -> DÜŞÜK /
    # belirsiz" rule (unchanged by this milestone) applies exactly as it
    # would for any other single-signal reading -- "yuksek_beklenti" is not
    # special-cased into surfacing as the overall direction on its own.
    result = triangulate(
        price=130, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=None,
        realized_cagr=None, base_growth=None, pe_pct=None, ps_pct=None, pfcf_pct=None,
        sector_type="growth_unprofitable", hyper_growth=True, bull_band={"lo": 130, "hi": 150},
    )
    assert result["signals"]["dcf"] == "yuksek_beklenti"
    assert result["confidence"] == "DÜŞÜK"
    assert result["direction"] == "belirsiz"


def test_triangulate_hyper_growth_direction_surfaces_when_reverse_dcf_agrees():
    # direction only ever mirrors "signals.dcf" == "yuksek_beklenti" via the
    # ordinary >=2-signals-agree majority rule -- reverse_dcf/multiples never
    # emit "yuksek_beklenti" themselves, so it can only become the overall
    # direction by being the (unique) 2-of-3 majority value together with
    # itself... which is structurally impossible since only the DCF method
    # can ever produce this string. This test documents that: even with
    # reverse_dcf pointing the same *direction* (pahali, i.e. above the
    # reference growth), it's a different signal STRING ("pahali" != "
    # yuksek_beklenti"), so no majority forms and the result stays
    # "belirsiz" -- confirming HYPER_SPEC.md's "participates naturally in
    # the majority machinery" claim reduces, in practice, to "shows up in
    # signals.dcf", not "can become the top-level direction by itself".
    result = triangulate(
        price=130, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.30,
        realized_cagr=0.10, base_growth=None, pe_pct=None, ps_pct=None, pfcf_pct=None,
        sector_type="growth_unprofitable", hyper_growth=True, bull_band={"lo": 130, "hi": 150},
    )
    assert result["signals"]["dcf"] == "yuksek_beklenti"
    assert result["signals"]["reverse_dcf"] == "pahali"
    assert result["direction"] == "belirsiz"


def test_triangulate_hyper_growth_price_above_bull_hi_is_pahali():
    # price=150.01: past bull.hi=150 -> "pahali" (outright expensive), even
    # in hyper mode.
    result = triangulate(
        price=150.01, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=None,
        realized_cagr=None, base_growth=None, pe_pct=None, ps_pct=None, pfcf_pct=None,
        sector_type="growth_unprofitable", hyper_growth=True, bull_band={"lo": 130, "hi": 150},
    )
    assert result["signals"]["dcf"] == "pahali"


def test_triangulate_hyper_growth_boundaries_are_strict():
    # price exactly at base.hi=120 -> "makul" (inside base band); exactly at
    # bull.hi=150 -> "yuksek_beklenti" (at/below bull.hi still counts), not
    # "pahali" (only strictly above).
    at_base_hi = triangulate(
        120, {"lo": 100, "hi": 120}, None, None, None, None, None, None,
        "growth_unprofitable", hyper_growth=True, bull_band={"lo": 130, "hi": 150},
    )
    at_bull_hi = triangulate(
        150, {"lo": 100, "hi": 120}, None, None, None, None, None, None,
        "growth_unprofitable", hyper_growth=True, bull_band={"lo": 130, "hi": 150},
    )
    assert at_base_hi["signals"]["dcf"] == "makul"
    assert at_bull_hi["signals"]["dcf"] == "yuksek_beklenti"


def test_triangulate_non_hyper_ignores_bull_band_and_keeps_3way_signal():
    # hyper_growth=False (the default) must keep the plain 3-way signal even
    # if a bull_band happens to be passed -- regression guard for existing
    # (non-hyper) callers.
    result = triangulate(
        130, {"lo": 100, "hi": 120}, None, None, None, None, None, None,
        "mature", bull_band={"lo": 130, "hi": 150},
    )
    assert result["signals"]["dcf"] == "pahali"


def test_triangulate_hyper_growth_without_usable_bull_band_falls_back_to_3way():
    # hyper_growth=True but bull_band is None/unusable (missing "hi") ->
    # falls back to the plain 3-way logic exactly (price=130 > base.hi=120
    # -> "pahali", the pre-hyper behavior).
    no_bull = triangulate(
        130, {"lo": 100, "hi": 120}, None, None, None, None, None, None,
        "growth_unprofitable", hyper_growth=True, bull_band=None,
    )
    unusable_bull = triangulate(
        130, {"lo": 100, "hi": 120}, None, None, None, None, None, None,
        "growth_unprofitable", hyper_growth=True, bull_band={"lo": 130},
    )
    assert no_bull["signals"]["dcf"] == "pahali"
    assert unusable_bull["signals"]["dcf"] == "pahali"


# ---------------------------------------------------------------------------
# 9. classify_sector (SPEC Sec.8)
# ---------------------------------------------------------------------------


def _ni_record(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def test_classify_sector_reit_sic():
    assert classify_sector(6798, {}, {}) == "reit"


def test_classify_sector_financial_sic_range():
    assert classify_sector(6022, {}, {}) == "financial"  # state commercial bank
    # 6798 itself is excluded from the financial range (tested separately).
    assert classify_sector(6799, {}, {}) == "financial"


def test_classify_sector_cyclical_sic_singleton_semiconductors():
    assert classify_sector(3674, {}, {}) == "cyclical"


def test_classify_sector_cyclical_sic_ranges():
    assert classify_sector(1200, {}, {}) == "cyclical"  # mining/energy 1000-1499
    assert classify_sector(2911, {}, {}) == "cyclical"  # petroleum refining singleton
    assert classify_sector(3714, {}, {}) == "cyclical"  # autos 3711-3716
    assert classify_sector(4500, {}, {}) == "cyclical"  # shipping/air 4400-4599


def test_classify_sector_negative_net_income_gives_growth_unprofitable():
    # SIC 7372 (prepackaged software) is outside every special range, so the
    # NetIncome override applies: latest FY (2023) NetIncome = -50 (< 0).
    normalized = {"annual": {"NetIncome": [_ni_record(2023, -50.0)]}}
    metrics = {"latest_fy": 2023}
    assert classify_sector(7372, normalized, metrics) == "growth_unprofitable"


def test_classify_sector_positive_net_income_gives_mature():
    normalized = {"annual": {"NetIncome": [_ni_record(2023, 50.0)]}}
    metrics = {"latest_fy": 2023}
    assert classify_sector(7372, normalized, metrics) == "mature"


def test_classify_sector_missing_sic_falls_back_to_mature():
    assert classify_sector(None, {}, {}) == "mature"
    assert classify_sector("not-a-sic", {}, {}) == "mature"


# ---------------------------------------------------------------------------
# 10. engine.run_valuation integration
# ---------------------------------------------------------------------------

_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity",
]


def _rec(fy, value, end=None):
    return {
        "concept": None, "tag": None, "period_end": end or f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(overrides):
    annual = {c: overrides.get(c) for c in _CONCEPTS}
    return {
        "cik": 1, "entity_name": "Engine Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _CONCEPTS},
        "missing": [c for c in _CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _CONCEPTS},
    }


def _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10):
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": base_growth, "terminal_growth": base_terminal, "discount_rate": base_discount, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


def test_run_valuation_financial_sector_disables_dcf_and_computes_pb_roe():
    # roe=0.15 (latest FY), discount_rate_base=0.10, equity_latest=1000,
    # shares=100.
    #   fair_pb_base = clamp(0.15/0.10, 0.5, 4.0) = clamp(1.5, ...) = 1.5
    #   book_value_per_share = 1000/100 = 10.0
    #   bear: fair_pb = 1.5*0.8 = 1.2 -> per_share = 12.0
    #   base: fair_pb = 1.5*1.0 = 1.5 -> per_share = 15.0
    #   bull: fair_pb = 1.5*1.2 = 1.8 -> per_share = 18.0
    #
    # F3: the band is no longer a flat +/-10% -- it's the min/max of
    # re-clamping fair_pb at discount_rate +/- 1pp (0.09, 0.10, 0.11), scaled
    # by this scenario's own scale/book_value_per_share:
    #   fair_pb(dr=0.09) = clamp(0.15/0.09, 0.5, 4.0) = 1.666667
    #   fair_pb(dr=0.10) = 1.5 (center, matches per_share above)
    #   fair_pb(dr=0.11) = clamp(0.15/0.11, 0.5, 4.0) = 1.363636
    #
    #   bear (scale=0.8): cells = 1.666667*0.8*10=13.33, 12.0, 1.363636*0.8*10=10.91
    #     -> lo=10.91, hi=13.33
    #   base (scale=1.0): cells = 16.67, 15.0, 13.64 -> lo=13.64, hi=16.67
    #   bull (scale=1.2): cells = 20.00, 18.0, 16.36 -> lo=16.36, hi=20.00
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    assert result["dcf"]["enabled"] is False
    assert result["dcf"]["disabled_reason"]
    assert result["dcf"]["scenarios"] is None

    pb = result["pb_roe"]["scenarios"]
    assert pb["bear"]["per_share"] == pytest.approx(12.0)
    assert pb["bear"]["lo"] == pytest.approx(10.91)
    assert pb["bear"]["hi"] == pytest.approx(13.33)
    assert pb["base"]["per_share"] == pytest.approx(15.0)
    assert pb["base"]["lo"] == pytest.approx(13.64)
    assert pb["base"]["hi"] == pytest.approx(16.67)
    assert pb["bull"]["per_share"] == pytest.approx(18.0)
    assert pb["bull"]["lo"] == pytest.approx(16.36)
    assert pb["bull"]["hi"] == pytest.approx(20.00)

    # fair_value_range must be built FROM the pb_roe scenarios when DCF is
    # disabled.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(13.64)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(16.67)


def test_run_valuation_cyclical_sector_produces_normalized_variant():
    # Regular fcf0 comes from metrics["fcf"]=200 (ttm; 3y avg of ratios'
    # fcf=[200,110,100] is 136.667, deviation |200-136.667|/136.667=0.463
    # <=0.50 -> ttm stays valid, source="ttm").
    #
    # Cyclical normalized_fcf0 = median(fcf_margin over all FYs) * latest
    # revenue. Revenue/OCF/CapEx per FY all give margin exactly 0.10:
    #   FY2021: rev=1000, ocf=150, capex=50  -> fcf=100 -> margin=0.10
    #   FY2022: rev=1100, ocf=176, capex=66  -> fcf=110 -> margin=0.10
    #   FY2023: rev=1200, ocf=300, capex=180 -> fcf=120 -> margin=0.10
    #   median margin = 0.10; latest revenue = 1200 -> normalized_fcf0=120.
    #
    # base assumptions: growth_5y=0.10, terminal_growth=0.03,
    # discount_rate=0.10 -- IDENTICAL to the DCF "case A" hand-verified in
    # test_valuation_dcf.py (per_share for fcf0=100, net_debt=0, shares=10
    # was ~=216.7679). Since net_debt=0 here too, equity (and hence
    # per_share) scales LINEARLY with fcf0:
    #   regular dcf base (fcf0=200):        per_share ~= 216.7679*2   = 433.54
    #   normalized_variant base (fcf0=120): per_share ~= 216.7679*1.2 = 260.12
    normalized = _normalized({
        "Revenue": [_rec(2021, 1000.0), _rec(2022, 1100.0), _rec(2023, 1200.0)],
        "OperatingCashFlow": [_rec(2021, 150.0), _rec(2022, 176.0), _rec(2023, 300.0)],
        "CapEx": [_rec(2021, 50.0), _rec(2022, 66.0), _rec(2023, 180.0)],
    })
    ratios = [{"fy": 2023, "fcf": 200.0}, {"fy": 2022, "fcf": 110.0}, {"fy": 2021, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 200.0, "net_debt": 0.0}
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="cyclical",
    )

    assert result["fcf0"] == pytest.approx(200.0)
    assert result["fcf0_source"] == "ttm"
    assert result["dcf"]["scenarios"]["base"]["per_share"] == pytest.approx(433.54, rel=1e-3)
    assert result["dcf"]["normalized_variant"] is not None
    assert result["dcf"]["normalized_variant"]["base"]["per_share"] == pytest.approx(260.12, rel=1e-3)


def test_run_valuation_cyclical_normalized_variant_uses_upper_half_mean_not_median():
    # Cyclical company with divergent FCF margins across 4 FYs, including a
    # deep trough (negative) year, to prove the current rule -- mean of the
    # top ceil(N/2) margins -- differs from (and exceeds) what a plain
    # median would have given.
    #   FY2021: rev=1000, ocf=250,  capex=50  -> margin=(250-50)/1000  = 0.20
    #   FY2022: rev=1000, ocf=200,  capex=100 -> margin=(200-100)/1000 = 0.10
    #   FY2023: rev=1000, ocf=-300, capex=100 -> margin=(-300-100)/1000= -0.40 (trough)
    #   FY2024: rev=1000, ocf=140,  capex=60  -> margin=(140-60)/1000  = 0.08 (latest)
    #
    # N=4 margins = [0.20, 0.10, -0.40, 0.08]. k = ceil(4/2) = 2.
    # top-2 (sorted desc) = [0.20, 0.10] -> mean = 0.15.
    # latest_fy=2024, latest_revenue=1000 -> normalized_fcf0 = 0.15*1000=150.
    #
    # Contrast with the OLD median rule (no longer in effect): sorted
    # margins = [-0.40, 0.08, 0.10, 0.20], median of 4 = (0.08+0.10)/2=0.09
    # -> normalized_fcf0 would have been 90. 150 != 90, proving the
    # top-half-mean rule (not median) is what actually runs.
    #
    # Base assumptions (growth_5y=0.10, terminal_growth=0.03,
    # discount_rate=0.10) are IDENTICAL to DCF "case A" in
    # test_valuation_dcf.py (per_share for fcf0=100, net_debt=0, shares=10
    # was hand-derived there as ~=216.7679, computed precisely by
    # dcf_per_share as ~=216.7659). Since net_debt=0 here too, per_share
    # scales LINEARLY with fcf0 (equity=ev, and ev is a linear function of
    # fcf0; effective_shares is independent of fcf0):
    #   regular dcf base (fcf0=200 ttm):    per_share ~= 216.7659*2.0 = 433.5318
    #   normalized_variant base (fcf0=150): per_share ~= 216.7659*1.5 = 325.1488
    #   [old median fcf0=90 would give:     per_share ~= 216.7659*0.9 = 195.0893]
    normalized = _normalized({
        "Revenue": [_rec(2021, 1000.0), _rec(2022, 1000.0), _rec(2023, 1000.0), _rec(2024, 1000.0)],
        "OperatingCashFlow": [_rec(2021, 250.0), _rec(2022, 200.0), _rec(2023, -300.0), _rec(2024, 140.0)],
        "CapEx": [_rec(2021, 50.0), _rec(2022, 100.0), _rec(2023, 100.0), _rec(2024, 60.0)],
    })
    ratios = [{"fy": 2024, "fcf": 200.0}]
    metrics = {"shares": 10.0, "latest_fy": 2024, "fcf": 200.0, "net_debt": 0.0}
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="cyclical",
    )

    assert result["fcf0"] == pytest.approx(200.0)
    assert result["fcf0_source"] == "ttm"
    assert result["dcf"]["scenarios"]["base"]["per_share"] == pytest.approx(433.5318, rel=1e-3)

    normalized_variant = result["dcf"]["normalized_variant"]
    assert normalized_variant is not None
    assert normalized_variant["base"]["per_share"] == pytest.approx(325.1488, rel=1e-3)

    # Strictly greater than what the OLD median-based fcf0=90 would have
    # produced -- this is the concrete, observable proof the rule changed.
    old_median_per_share = 216.7659 * 0.9
    assert normalized_variant["base"]["per_share"] > old_median_per_share


def test_run_valuation_cyclical_normalized_variant_none_when_top_half_margin_not_positive():
    # Every FY has a non-positive FCF margin, so even the best (least bad)
    # ceil(N/2) margins average to <= 0 -- the "normalized" cyclical
    # variant is not meaningful and must be suppressed (returned as None)
    # with a Turkish note, rather than surfacing a nonsensical negative
    # "normalized" valuation.
    #   FY2022: rev=1000, ocf=50,  capex=100 -> margin=(50-100)/1000  = -0.05
    #   FY2023: rev=1000, ocf=80,  capex=150 -> margin=(80-150)/1000  = -0.07
    #   FY2024: rev=1000, ocf=100, capex=200 -> margin=(100-200)/1000 = -0.10 (latest)
    # N=3, k=ceil(3/2)=2. top-2 (sorted desc) = [-0.05, -0.07] -> mean=-0.06
    # <= 0 -> guard trips -> normalized_fcf0=None plus the exact Turkish
    # note.
    normalized = _normalized({
        "Revenue": [_rec(2022, 1000.0), _rec(2023, 1000.0), _rec(2024, 1000.0)],
        "OperatingCashFlow": [_rec(2022, 50.0), _rec(2023, 80.0), _rec(2024, 100.0)],
        "CapEx": [_rec(2022, 100.0), _rec(2023, 150.0), _rec(2024, 200.0)],
    })
    ratios = [{"fy": 2024, "fcf": 50.0}]
    metrics = {"shares": 10.0, "latest_fy": 2024, "fcf": 50.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="cyclical",
    )

    assert result["dcf"]["normalized_variant"] is None
    assert (
        "Döngüsel normalize edilmiş FCF anlamlı değil: üst yarı (mid/tepe döngü) ortalama FCF "
        "marjı pozitif değil."
    ) in result["notes"]


def test_run_valuation_cyclical_headline_uses_normalized_band_over_raw_dcf():
    # Reuses the exact fixture from
    # test_run_valuation_cyclical_sector_produces_normalized_variant: raw dcf
    # base per_share ~=433.54 (fcf0=200 ttm), normalized_variant base
    # per_share ~=260.12 (normalized_fcf0=120, from a constant 0.10 FCF
    # margin * latest revenue 1200) -- both hand-derived there, and clearly
    # different numbers (not a coincidental match).
    #
    # Per the new engine wiring (cyclical sector_type + normalized_variant
    # successfully computed): the HEADLINE result["fair_value_range"] must be
    # built from dcf.normalized_variant, NOT from the raw dcf.scenarios, and
    # a "Döngüsel sektör: manşet..." note must be appended. Both raw and
    # normalized bands remain reported side-by-side under result["dcf"].
    normalized = _normalized({
        "Revenue": [_rec(2021, 1000.0), _rec(2022, 1100.0), _rec(2023, 1200.0)],
        "OperatingCashFlow": [_rec(2021, 150.0), _rec(2022, 176.0), _rec(2023, 300.0)],
        "CapEx": [_rec(2021, 50.0), _rec(2022, 66.0), _rec(2023, 180.0)],
    })
    ratios = [{"fy": 2023, "fcf": 200.0}, {"fy": 2022, "fcf": 110.0}, {"fy": 2021, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 200.0, "net_debt": 0.0}
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="cyclical",
    )

    dcf = result["dcf"]
    assert dcf["scenarios"] is not None
    assert dcf["normalized_variant"] is not None

    # Fixture sanity check: the two bands must actually differ, or this test
    # would pass even if the engine always used the raw band.
    assert dcf["scenarios"]["base"]["per_share"] != dcf["normalized_variant"]["base"]["per_share"]
    assert dcf["scenarios"]["base"]["per_share"] == pytest.approx(433.54, rel=1e-3)
    assert dcf["normalized_variant"]["base"]["per_share"] == pytest.approx(260.12, rel=1e-3)

    # Headline band must equal the normalized_variant band...
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(dcf["normalized_variant"]["base"]["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(dcf["normalized_variant"]["base"]["hi"])
    # ...and NOT the raw scenarios band.
    assert result["fair_value_range"]["base"]["lo"] != dcf["scenarios"]["base"]["lo"]
    assert result["fair_value_range"]["base"]["hi"] != dcf["scenarios"]["base"]["hi"]

    assert any(n.startswith("Döngüsel sektör: manşet makul değer aralığı") for n in result["notes"])


def test_run_valuation_cyclical_headline_falls_back_to_raw_band_when_normalized_variant_none():
    # Reuses the exact fixture from
    # test_run_valuation_cyclical_normalized_variant_none_when_top_half_margin_not_positive:
    # every FY has a non-positive FCF margin, so the top-half-mean margin is
    # <=0 and _normalized_fcf0's guard returns None -- normalized_variant
    # stays None. The raw ttm fcf0=50.0 (metrics["fcf"]) is positive and
    # matches the 3y-average window exactly (ratios only has fy=2024 ->
    # avg=50.0, deviation=0), so the raw FCF-DCF still produces a full base
    # scenario band.
    #
    # Per the new engine wiring, when normalized_variant is None the
    # headline result["fair_value_range"] must fall back to the raw
    # dcf.scenarios band, and no "Döngüsel sektör: manşet..." note should be
    # appended (there is nothing normalized to prefer over the raw band).
    normalized = _normalized({
        "Revenue": [_rec(2022, 1000.0), _rec(2023, 1000.0), _rec(2024, 1000.0)],
        "OperatingCashFlow": [_rec(2022, 50.0), _rec(2023, 80.0), _rec(2024, 100.0)],
        "CapEx": [_rec(2022, 100.0), _rec(2023, 150.0), _rec(2024, 200.0)],
    })
    ratios = [{"fy": 2024, "fcf": 50.0}]
    metrics = {"shares": 10.0, "latest_fy": 2024, "fcf": 50.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="cyclical",
    )

    assert result["dcf"]["normalized_variant"] is None
    assert result["dcf"]["scenarios"] is not None
    assert result["dcf"]["scenarios"]["base"]["per_share"] is not None

    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(result["dcf"]["scenarios"]["base"]["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(result["dcf"]["scenarios"]["base"]["hi"])

    assert not any(n.startswith("Döngüsel sektör: manşet") for n in result["notes"])


def test_run_valuation_fcf0_falls_back_to_3y_average_when_latest_is_negative():
    # metrics["fcf"] (ttm) = -50 -- non-positive, so NOT usable directly.
    # ratios' 3y window fcf values = [-50 (2023), 100 (2022), 90 (2021)]
    # -> average = (-50+100+90)/3 = 140/3 = 46.6667, which IS positive
    # (usable) -> fcf0 = 46.6667, source="3y_avg", plus a Turkish note.
    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": -50.0}, {"fy": 2022, "fcf": 100.0}, {"fy": 2021, "fcf": 90.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": -50.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(140.0 / 3.0, rel=1e-6)
    assert result["fcf0_source"] == "3y_avg"
    assert any("3 yıllık ortalama" in n for n in result["notes"])


def test_run_valuation_fcf0_keeps_latest_when_ramp_is_monotonic_rising():
    # This is the NVDA-shaped case: a steady rising ramp whose latest year
    # deviates >50% from the 3y average (which, per _select_fcf0, includes
    # the latest year itself in the average).
    #   fcf by fy: 2021=20, 2022=50, 2023=90 (latest).
    #   avg = (90+50+20)/3 = 160/3 = 53.3333.
    #   deviation = |90-53.3333|/53.3333 = 36.6667/53.3333 = 0.6875 -> >0.50
    #   -> deviates=True.
    #   oldest->newest = [20, 50, 90] -> 20<=50<=90 -> non-decreasing ->
    #   monotonic=True.
    #   -> ttm kept: fcf0=90, source="ttm", plus a trend note (NOT the
    #   3y_avg fallback note).
    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 90.0}, {"fy": 2022, "fcf": 50.0}, {"fy": 2021, "fcf": 20.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 90.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(90.0)
    assert result["fcf0_source"] == "ttm"
    assert any("istikrarlı bir trend" in n for n in result["notes"])
    # Must NOT use the "average used instead of latest year" fallback note --
    # the ttm figure was kept, not discarded.
    assert not any("yerine 3 yıllık ortalama FCF kullanıldı" in n for n in result["notes"])


def test_run_valuation_fcf0_keeps_latest_when_ramp_is_monotonic_falling():
    # Mirror of the rising case: a steady declining ramp.
    #   fcf by fy: 2021=90, 2022=50, 2023=20 (latest).
    #   avg = (20+50+90)/3 = 160/3 = 53.3333.
    #   deviation = |20-53.3333|/53.3333 = 33.3333/53.3333 = 0.625 -> >0.50
    #   -> deviates=True.
    #   oldest->newest = [90, 50, 20] -> 90>=50>=20 -> non-increasing ->
    #   monotonic=True.
    #   -> ttm kept: fcf0=20, source="ttm", plus a trend note.
    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 20.0}, {"fy": 2022, "fcf": 50.0}, {"fy": 2021, "fcf": 90.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 20.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(20.0)
    assert result["fcf0_source"] == "ttm"
    assert any("istikrarlı bir trend" in n for n in result["notes"])


def test_run_valuation_fcf0_falls_back_to_3y_average_when_series_is_spiky():
    # Same three raw values as the monotonic cases (20, 100, 90) but
    # reordered into an oscillating (non-monotonic) sequence, to confirm
    # the old spike-normalization behavior is preserved when the trend
    # exception does NOT apply.
    #   fcf by fy: 2021=90, 2022=100, 2023=20 (latest).
    #   avg = (20+100+90)/3 = 210/3 = 70.0.
    #   deviation = |20-70|/70 = 50/70 = 0.7143 -> >0.50 -> deviates=True.
    #   oldest->newest = [90, 100, 20] -> 90<=100 but 100>20 (not
    #   non-decreasing); 90>=100 is False (not non-increasing either) ->
    #   monotonic=False.
    #   -> falls back to the 3y average: fcf0=70.0, source="3y_avg".
    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 20.0}, {"fy": 2022, "fcf": 100.0}, {"fy": 2021, "fcf": 90.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 20.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(70.0)
    assert result["fcf0_source"] == "3y_avg"
    assert any("3 yıllık ortalama" in n for n in result["notes"])


def test_run_valuation_fcf0_deviation_rule_applies_when_trend_unassessable():
    # Fewer than 3 consecutive fiscal years of fcf data -> the trend can't
    # be assessed (monotonic is forced False) -> falls through to the
    # plain deviation rule, exactly as before this change.
    #   fcf by fy: 2023=100 (latest), 2022=20, 2021=missing (None).
    #   avg window = [100, 20] (2021 dropped) -> avg = 120/2 = 60.0.
    #   deviation = |100-60|/60 = 40/60 = 0.6667 -> >0.50 -> deviates=True.
    #   window has a None entry -> monotonic can't be assessed -> False.
    #   -> falls back to the 3y average: fcf0=60.0, source="3y_avg".
    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 100.0}, {"fy": 2022, "fcf": 20.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(60.0)
    assert result["fcf0_source"] == "3y_avg"


def test_run_valuation_missing_price_df_degrades_multiples_with_a_note():
    normalized = _normalized({
        "Revenue": [_rec(2023, 1000.0)],
        "EPS": [_rec(2023, 1.0)],
        "SharesOutstanding": [_rec(2023, 100.0)],
    })
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0, "pe": 10.0, "ps": 1.0, "pfcf": 10.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=10.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["multiples"]["history"] == []
    assert result["multiples"]["pe_percentile"] is None
    # SPEC-per-the-general-contract expectation: a note should mention the
    # missing price history degrading the multiples analysis.
    assert any("fiyat" in n.lower() for n in result["notes"])


def test_run_valuation_never_raises_on_completely_empty_inputs():
    result = run_valuation({}, [], {}, price=None, price_df=None, assumptions={}, sector_type="mature")
    assert result["dcf"]["scenarios"] is None
    assert result["fair_value_range"]["base"]["lo"] is None
    assert isinstance(result["notes"], list) and result["notes"]


# ---------------------------------------------------------------------------
# 11. engine current-multiple / pb_roe fallbacks -- when metrics'
# single-"latest_fy" alignment leaves pe/ps/pfcf/pb_roe all None even though
# usable per-FY fundamentals exist (e.g. a filer whose dei cover-page share
# count is newer than its EPS/Revenue/FCF/equity, like JPM).
# ---------------------------------------------------------------------------


def test_run_valuation_derives_current_multiples_from_series_when_metrics_are_none():
    # metrics["latest_fy"]=2024 is newer than the fiscal year (2023) that
    # actually has EPS/Revenue/FCF data (mirrors JPM: a newer
    # SharesOutstanding fact than the underlying fundamentals) -- so
    # metrics' own pe/ps/pfcf are all None even though price=50 and 2023's
    # fundamentals are perfectly usable.
    #   pe   = price / eps          = 50 / 5.0   = 10.0
    #   ps   = price * shares / rev = 50 * 100 / 1000 = 5.0
    #   pfcf = price * shares / fcf = 50 * 100 / 200   = 25.0 (fcf from ratios)
    normalized = _normalized({
        "EPS": [_rec(2023, 5.0)],
        "Revenue": [_rec(2023, 1000.0)],
    })
    ratios = [{"fy": 2023, "fcf": 200.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2024, "net_debt": 0.0,
        "pe": None, "ps": None, "pfcf": None, "fcf": None,
    }
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    current = result["multiples"]["current"]
    assert current["pe"] == pytest.approx(10.0)
    assert current["ps"] == pytest.approx(5.0)
    assert current["pfcf"] == pytest.approx(25.0)
    assert any("2023 mali y" in n and "F/K" in n for n in result["notes"])
    assert any("2023 mali y" in n and "F/S" in n for n in result["notes"])
    assert any("2023 mali y" in n and "F/FCF" in n for n in result["notes"])


def test_run_valuation_derives_current_pfcf_from_ocf_minus_capex_when_ratios_fcf_missing():
    # ratios carries no "fcf" for FY2023 at all, so the pfcf fallback must
    # recompute it from OperatingCashFlow - CapEx (mirrors
    # normalize.metrics.compute_metrics' own fcf selection):
    #   fcf = 300 - 100 = 200; pfcf = price * shares / fcf = 40*50/200 = 10.0
    normalized = _normalized({
        "OperatingCashFlow": [_rec(2023, 300.0)],
        "CapEx": [_rec(2023, 100.0)],
    })
    ratios = [{"fy": 2023}]  # present but without a usable "fcf" key
    metrics = {"shares": 50.0, "latest_fy": 2024, "net_debt": 0.0, "pe": None, "ps": None, "pfcf": None, "fcf": None}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=40.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["multiples"]["current"]["pfcf"] == pytest.approx(10.0)


def test_run_valuation_current_multiples_left_none_without_price_or_shares():
    # No price at all -> nothing can be derived, current stays all-None.
    normalized = _normalized({"EPS": [_rec(2023, 5.0)], "Revenue": [_rec(2023, 1000.0)]})
    metrics = {"shares": 100.0, "latest_fy": 2024, "net_debt": 0.0, "pe": None, "ps": None, "pfcf": None, "fcf": None}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, [], metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )
    assert result["multiples"]["current"] == {"pe": None, "ps": None, "pfcf": None}

    # Price present but no shares at all -> ps/pfcf can't be derived
    # (no market-cap-style numerator to divide), but pe (price/eps, no
    # shares needed) still can be.
    metrics_no_shares = {"shares": None, "latest_fy": 2024, "net_debt": 0.0, "pe": None, "ps": None, "pfcf": None, "fcf": None}
    result_no_shares = run_valuation(
        normalized, [], metrics_no_shares, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )
    current_no_shares = result_no_shares["multiples"]["current"]
    assert current_no_shares["pe"] == pytest.approx(10.0)
    assert current_no_shares["ps"] is None
    assert current_no_shares["pfcf"] is None


def test_run_valuation_pb_roe_uses_latest_fy_where_equity_and_roe_are_aligned():
    # metrics["latest_fy"]=2024 has no StockholdersEquity/ROE data at all
    # (mirrors JPM: SharesOutstanding's fiscal year is newer than the
    # filer's latest reported equity/net income). The engine must not give
    # up -- it should fall back to FY2023, the newest fiscal year where
    # BOTH the equity series and a ROE figure are actually present.
    #   fair_pb_base = clamp(0.15/0.10, 0.5, 4.0) = 1.5
    #   book_value_per_share = 1000/100 = 10.0
    #   base: fair_pb = 1.5*1.0 -> per_share = 15.0
    #
    # F3: band = min/max of fair_pb re-clamped at discount_rate +/- 1pp (see
    # test_run_valuation_financial_sector_disables_dcf_and_computes_pb_roe's
    # comment for the full 0.09/0.10/0.11 cell derivation) -> lo=13.64, hi=16.67.
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2024, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    pb = result["pb_roe"]["scenarios"]
    assert pb["base"]["per_share"] == pytest.approx(15.0)
    assert pb["base"]["lo"] == pytest.approx(13.64)
    assert pb["base"]["hi"] == pytest.approx(16.67)
    assert any("2023 mali y" in n and "P/B" in n for n in result["notes"])


def test_run_valuation_pb_roe_none_when_no_fiscal_year_has_both_equity_and_roe():
    # Equity exists for 2023 but ROE data is entirely absent -> no fiscal
    # year has both, so pb_roe must stay None (not silently misalign
    # mismatched equity/ROE from different years).
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, [], metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    assert result["pb_roe"] is None
    assert any("özkaynak" in n.lower() for n in result["notes"])


# ---------------------------------------------------------------------------
# 12. engine.run_valuation -- hyper-grower revenue-first DCF wiring (SPEC
# Sec.1 detection + Sec.3 engine integration). All numeric expectations below
# were cross-checked by independently re-deriving the revenue-first DCF's
# growth-fade/margin-convergence paths (see ``revenue_first_dcf``'s own
# hand-verified unit test in ``test_valuation_dcf.py`` for the core-formula
# derivation this reuses):
#   revenue0=1000, start_growth=0.40 (min(realized_cagr=0.50, cap 0.40)),
#   terminal_growth=0.025, steady_state_year=10 -> the growth fade and the
#   resulting revenue path are IDENTICAL across bear/base/bull for this
#   fixture (bear/base/bull start_growth all clamp to the same 0.40 -- see
#   the per-scenario table below), so base/bull share the same
#   final_year_revenue/revenue_multiple (~6538.48 / ~6.538x); bear's
#   start_growth=0.24 is genuinely lower, giving a materially smaller
#   final_year_revenue (~3407.10 / ~3.407x). gross_margin=0.60 ->
#   target_base=min(0.60*0.5, 0.30)=0.30 -> bear target=0.30*0.7=0.21,
#   base target=0.30, bull target=min(0.30*1.2, 0.60)=0.36.
# ---------------------------------------------------------------------------

_HYPER_CONCEPTS_OVERRIDES = {"Revenue": [_rec(2023, 1000.0)]}
_HYPER_RATIOS = [
    {"fy": 2023, "gross_margin": 0.60, "fcf": -50.0},
    {"fy": 2022, "fcf": 100.0},
    {"fy": 2021, "fcf": 90.0},
]
_HYPER_METRICS = {
    "shares": 100.0, "latest_fy": 2023, "fcf": -50.0, "net_debt": 0.0,
    "revenue_cagr_5y": 0.50, "rnd_revenue": 0.0, "sbc_revenue": 0.0, "shares_yoy": None,
}


def test_run_valuation_hyper_grower_uses_revenue_first_dcf_as_headline():
    # Trigger: realized_cagr=0.50 (>0.25) AND fcf=-50<=0 (clause a) AND
    # fcf_margin=-50/1000=-0.05<0.05 (clause b) -- both fire.
    #
    # base scenario: start_growth=min(0.50,0.40)=0.40, target_fcf_margin=0.30,
    # discount_rate=0.10 (fixed), current_margin=-50/1000=-0.05,
    # annual_dilution=0.0 (shares_yoy None, sbc_revenue 0) -> revenue path
    # fades 0.40->0.025 over 10y, final_year_revenue~=6538.48 (multiple
    # ~=6.538x, comfortably <=8 -> arrival_flag "makul"). per_share~=139.73.
    #
    # F3: financing_shares is derived once from the base scenario's own
    # (financing_shares=0) preliminary fcf_path: cumulative negative-FCF
    # years sum to burn=-21.0 (undiscounted), financing_shares =
    # abs(-21.0)/price(50.0) = 0.42, reused for bear/base/bull alike. The
    # base band is then the min/max of a 3x3 grid over start_growth +/- 2pp
    # (0.38/0.40/0.42) x discount_rate +/- 1pp (0.09/0.10/0.11), everything
    # else (target_fcf_margin=0.30, current_margin=-0.05, steady_state=10,
    # annual_dilution=0.0, financing_shares=0.42) held fixed --
    # cross-checked against revenue_dcf.revenue_first_dcf directly (the
    # same function hand-verified in test_revenue_dcf.py):
    #   grid cells (rounded to 2dp) = [157.04, 129.43, 108.70,
    #                                  169.58, 139.73, 117.32,
    #                                  182.98, 150.73, 126.53]
    #   -> lo=min=108.70, hi=max=182.98 (center cell 139.73 matches per_share
    #      above, as expected).
    #
    # Standard (raw) FCF-DCF fcf0 falls back to the 3y average (ttm=-50
    # non-positive) = (-50+100+90)/3=46.6667 -> a MUCH smaller base per_share
    # (~10.12) -- deliberately a different number from the hyper band,
    # proving the headline is NOT silently reusing the raw DCF band.
    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail is not None

    # Headline fair_value_range.base must equal the hyper base band...
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(detail["scenarios"]["base"]["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(detail["scenarios"]["base"]["hi"])
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(108.70)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(182.98)

    # ...NOT the standard FCF-DCF band, which is still computed and reported
    # (secondary) under dcf.scenarios, and is clearly a different number.
    assert result["dcf"]["scenarios"] is not None
    assert result["dcf"]["scenarios"]["base"]["per_share"] is not None
    assert result["dcf"]["scenarios"]["base"]["per_share"] != detail["scenarios"]["base"]["per_share"]
    assert result["dcf"]["scenarios"]["base"]["lo"] != result["fair_value_range"]["base"]["lo"]

    # hyper_growth_detail is fully populated.
    assert detail["reasons"][0].startswith("Gelir CAGR %50.0")
    assert detail["scenarios"]["bear"]["start_growth"] == pytest.approx(0.24)
    assert detail["scenarios"]["base"]["start_growth"] == pytest.approx(0.40)
    assert detail["scenarios"]["bull"]["start_growth"] == pytest.approx(0.40)
    assert detail["scenarios"]["bear"]["target_fcf_margin"] == pytest.approx(0.21)
    assert detail["scenarios"]["base"]["target_fcf_margin"] == pytest.approx(0.30)
    assert detail["scenarios"]["bull"]["target_fcf_margin"] == pytest.approx(0.36)
    assert detail["probabilities"] == {"bear": 0.25, "base": 0.50, "bull": 0.25}
    # expected_value = 0.25*bear + 0.50*base + 0.25*bull (prob-weighted).
    expected_ev = round(
        0.25 * detail["scenarios"]["bear"]["per_share"]
        + 0.50 * detail["scenarios"]["base"]["per_share"]
        + 0.25 * detail["scenarios"]["bull"]["per_share"],
        2,
    )
    assert detail["expected_value"] == pytest.approx(expected_ev)
    assert detail["arrival_flag"] == "makul"
    assert detail["tam_usd"] is None
    assert detail["implied"]["growth"] is not None
    assert detail["implied"]["revenue_10y"] is not None
    assert detail["implied"]["revenue_multiple"] is not None
    assert detail["implied"]["steady_state_margin"] is not None
    assert detail["implied"]["tam_share"] is None  # no tam_usd supplied
    assert detail["target_margin_source"] == "brüt marj × 0.5 (tavan %30)"

    # Turkish hyper-mode note is present in the top-level notes.
    assert any(
        "Hiper-büyüme modu: manşet aralığı revenue-first DCF'ten" in n for n in result["notes"]
    )

    # Display consistency (Milestone F): fair_value_range's base scenario
    # growth/discount_rate/note must reflect the hyper revenue-first DCF's
    # OWN inputs (start_growth=0.40, discount=0.10, target margin=0.30) --
    # NOT the standard clamped assumptions (base_growth=0.10 in the fixture
    # above), which the headline band no longer actually uses once
    # hyper-grower mode takes over.
    base_fvr = result["fair_value_range"]["base"]
    assert "başlangıç" in base_fvr["growth"]
    assert "%40" in base_fvr["growth"]
    assert "%25 büyüme" not in base_fvr["growth"]
    assert base_fvr["discount_rate"] == "%10"
    assert "Hiper-büyüme" in base_fvr["note"]
    assert "%30" in base_fvr["note"]  # mature target FCF margin


def test_run_valuation_hyper_grower_extras_override_target_margin_and_tam():
    # Same fixture as above, but with hyper_growth_extras supplying a TAM
    # and an overridden base target_fcf_margin. The growth path (and hence
    # final_year_revenue~=6538.48) is UNCHANGED by a target-margin override
    # (revenue_first_dcf's growth path never depends on the margin path), so
    # tam_share = 6538.48 / 14000 ~= 0.467 -- strictly between the 0.40
    # ("agresif") and 0.60 ("gecersiz") TAM-share thresholds -> "agresif",
    # which OVERRIDES what the revenue-multiple-based flag would have been
    # (6.538x <= 8 -> "makul" without TAM).
    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)
    extras = {"tam_usd": 14000.0, "per_scenario": {"base": {"target_fcf_margin": 0.5}}}

    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable", hyper_growth_extras=extras,
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]

    assert detail["tam_usd"] == pytest.approx(14000.0)
    # Overridden target margin used verbatim (bypasses the gross-margin cap
    # and the deterministic 0.30 that clause (b) would otherwise have set).
    assert detail["scenarios"]["base"]["target_fcf_margin"] == pytest.approx(0.5)
    assert detail["scenarios"]["base"]["final_year_revenue"] == pytest.approx(6538.4756, rel=1e-3)

    tam_share = detail["scenarios"]["base"]["final_year_revenue"] / 14000.0
    assert 0.40 < tam_share < 0.60
    assert detail["arrival_flag"] == "agresif"

    assert "LLM/kullanıcı" in detail["target_margin_source"]
    assert any("TAM" in n for n in detail["notes"])

    # implied_tam_share is computed from the implied 10y revenue / tam_usd.
    if detail["implied"]["revenue_10y"] is not None:
        assert detail["implied"]["tam_share"] == pytest.approx(
            detail["implied"]["revenue_10y"] / 14000.0
        )


def test_run_valuation_hyper_grower_triangulation_uses_yuksek_beklenti_signal():
    # Same hyper fixture as above, but price=200 this time. F3 changed the
    # hyper band from a flat +/-10% to a grid-derived one (see the headline
    # test above), which also shifts financing_shares (it's derived from
    # burn/price, so a different price changes it too) -- at price=200,
    # financing_shares = 21.0/200 = 0.105 (vs. 0.42 at price=50 above), so
    # the base/bull bands themselves are numerically different from the
    # price=50 case (per_share moves slightly with financing_shares) but the
    # key inequality this test needs -- base.hi < price <= bull.hi -- still
    # holds comfortably: base.hi ~=183.55, bull.hi ~=275.11 (cross-checked
    # against revenue_dcf.revenue_first_dcf directly, same approach as the
    # headline test above), so the triangulation DCF signal must be
    # "yuksek_beklenti" (HYPER_SPEC.md Sec.4), proving base_band/bull_band
    # are actually threaded from hyper_growth_detail into
    # triangulate.triangulate(), not just computed and left unused.
    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=200.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    detail = result["hyper_growth_detail"]
    assert detail is not None
    base_hi = detail["scenarios"]["base"]["hi"]
    bull_hi = detail["scenarios"]["bull"]["hi"]
    assert base_hi < 200.0 <= bull_hi  # sanity-check the fixture actually lands in the intended zone
    assert base_hi == pytest.approx(183.55, abs=0.05)
    assert bull_hi == pytest.approx(275.11, abs=0.05)

    assert result["triangulation"]["signals"]["dcf"] == "yuksek_beklenti"


def test_run_valuation_non_hyper_mature_company_regression():
    # A mature, single-digit-growth company: realized_cagr=0.05 (well under
    # the 25% trigger threshold) -- detect_hyper_grower must NOT fire, so
    # hyper_growth stays False, hyper_growth_detail stays None, and the
    # headline fair_value_range comes from the ordinary FCF-DCF band exactly
    # as before this milestone (no behavior change for non-hyper filers).
    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 100.0}, {"fy": 2022, "fcf": 95.0}, {"fy": 2021, "fcf": 90.0}]
    metrics = {
        "shares": 100.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0,
        "revenue_cagr_5y": 0.05, "rnd_revenue": 0.0, "sbc_revenue": 0.0, "shares_yoy": None,
    }
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["hyper_growth"] is False
    assert result["hyper_growth_detail"] is None
    assert not any("Hiper-büyüme" in n for n in result["notes"])

    # Headline comes straight from the raw FCF-DCF band, unchanged.
    assert result["dcf"]["scenarios"] is not None
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(result["dcf"]["scenarios"]["base"]["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(result["dcf"]["scenarios"]["base"]["hi"])

    # Non-hyper filers keep the assumptions-derived growth/discount_rate/note
    # strings (scenario_meta=None path) -- no behavior change here.
    base_fvr = result["fair_value_range"]["base"]
    assert base_fvr["growth"] == "%10 büyüme"
    assert base_fvr["discount_rate"] == "%10"
    assert base_fvr["note"] == "Baz."


def test_run_valuation_hyper_grower_degrades_safely_when_revenue_missing():
    # detect_hyper_grower's growth/fcf-margin conditions still fire (realized
    # CAGR 0.50 with a negative fcf), but the normalized facts carry no
    # Revenue at all for the latest FY -- _build_hyper_growth can't derive
    # revenue0, so the whole hyper block must degrade to hyper_growth=False /
    # hyper_growth_detail=None (never crash, never half-build a detail dict)
    # while the rest of the valuation still returns a normal shaped result.
    normalized = _normalized({})  # no Revenue series at all
    metrics = dict(_HYPER_METRICS)
    assumptions = _assumptions()

    result = run_valuation(
        normalized, _HYPER_RATIOS, metrics, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    assert result["hyper_growth"] is False
    assert result["hyper_growth_detail"] is None
    assert any("Hiper-büyüme modu tetiklendi ancak" in n for n in result["notes"])
    # The rest of the valuation must still be a fully-shaped, usable result.
    assert isinstance(result["notes"], list)
    assert result["triangulation"] is not None


_HYPER_RDDT_CONCEPTS_OVERRIDES = {"Revenue": [_rec(2023, 1000.0)]}
_HYPER_RDDT_RATIOS = [
    {"fy": 2023, "fcf": 300.0},  # no "gross_margin" key -> gross margin data missing
]
_HYPER_RDDT_METRICS = {
    "shares": 100.0, "latest_fy": 2023, "fcf": 300.0, "net_debt": 0.0,
    "revenue_cagr_5y": 0.50, "rnd_revenue": 0.35, "sbc_revenue": 0.10, "shares_yoy": None,
}


def test_run_valuation_hyper_grower_gross_margin_missing_floors_target_at_current_fcf_margin():
    # Regression for a Reddit-shaped filer: no GrossProfit/CostOfRevenue in
    # the normalized facts (gross_margin missing) but the filer already runs
    # a healthy ~30% FCF margin today (fcf=300, revenue=1000). The OLD rule
    # derived target_base from a 15%-gross-margin fallback -> 0.15*0.5=0.075
    # (7.5%), modeling the margin as COLLAPSING from 30% to 7.5% at
    # maturity -- badly understating value. The fix: when gross margin is
    # missing, use a 20% default ceiling, then floor target_base at today's
    # positive FCF margin (0.30 > 0.20 ceiling) so a currently-profitable
    # hyper-grower is never modeled as getting *less* profitable at
    # maturity than it is today.
    #
    # Trigger: realized_cagr=0.50 (>0.25) AND rnd_revenue+sbc_revenue=
    # 0.35+0.10=0.45 (>0.40, clause c) -- fires even though fcf=300>0 and
    # fcf_margin=0.30 (>=0.05, so clause b does NOT fire on its own).
    #
    # target_base = max(current_margin=0.30, ceiling=0.20) = 0.30 (gm is
    # None, so no gross-margin cap applies) -> bear=0.30*0.7=0.21,
    # base=0.30, bull=0.30*1.2=0.36.
    normalized = _normalized(_HYPER_RDDT_CONCEPTS_OVERRIDES)
    assumptions = _assumptions()

    result = run_valuation(
        normalized, _HYPER_RDDT_RATIOS, _HYPER_RDDT_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail is not None

    # base target_fcf_margin is floored at current_margin (~0.30), NOT the
    # old 0.15*0.5=0.075 default-gross-margin-fallback ceiling.
    assert detail["scenarios"]["base"]["target_fcf_margin"] == pytest.approx(0.30)
    assert detail["scenarios"]["base"]["target_fcf_margin"] != pytest.approx(0.075)
    assert detail["scenarios"]["bear"]["target_fcf_margin"] == pytest.approx(0.21)
    assert detail["scenarios"]["bull"]["target_fcf_margin"] == pytest.approx(0.36)

    # target_margin_source explains the current-margin flooring (not the
    # old "brüt marj eksik: varsayılan %15 taban brüt marj × 0.5" wording).
    assert "brüt marj yok" in detail["target_margin_source"]
    assert "tabanlanmış" in detail["target_margin_source"]
    assert "%30" in detail["target_margin_source"]


# ---------------------------------------------------------------------------
# 13. F3 -- headline band derived from a 9-cell sensitivity grid, not a flat
# +/-10% (SPEC.md Sec.4). Small, round-number fixture so the 3x3 grid can be
# hand-computed cell by cell using dcf.dcf_per_share directly (the same
# function independently hand-verified in test_valuation_dcf.py's "case A"),
# rather than trusting the engine's own private grid helper.
# ---------------------------------------------------------------------------


def test_run_valuation_headline_band_matches_hand_computed_sensitivity_grid():
    # fcf0=100 (ttm == 3y average exactly, since ratios only has one FY --
    # no deviation, no fallback), shares=10, base assumptions growth_5y=
    # 0.10, terminal_growth=0.03, discount_rate=0.10 -- IDENTICAL numbers to
    # test_valuation_dcf.py's hand-verified DCF "case A"
    # (per_share ~=216.7679 there, ~=216.77 rounded, matching the center
    # cell below).
    #
    # F3's grid: growth_5y +/-2pp (0.08/0.10/0.12) x discount_rate +/-1pp
    # (0.09/0.10/0.11), terminal_growth held fixed at 0.03. Computing each of
    # the 9 cells independently via dcf_per_share (fcf0=100, shares=10,
    # dilution_rate=0.0), rounded to 2dp as the production code does:
    #
    #   r=0.09    r=0.10    r=0.11
    # g=0.08: 228.13    194.21    168.82
    # g=0.10: 255.26    216.77    187.99
    # g=0.12: 285.37    241.77    209.20
    #
    # (The g=0.10/r=0.10 center cell and the g=0.08/r=0.10 cell were already
    # independently hand-verified in test_valuation_multiples.py's
    # sensitivity_matrix tests -- 216.77 and 194.21 respectively -- this
    # table is consistent with those.)
    #
    # lo = min of all 9 = 168.82 (g=0.08, r=0.11)
    # hi = max of all 9 = 285.37 (g=0.12, r=0.09)
    from sec_analyzer.valuation.dcf import dcf_per_share

    fcf0, shares = 100.0, 10.0
    growth, terminal_growth, discount_rate = 0.10, 0.03, 0.10

    hand_computed_cells = []
    for g in (growth - 0.02, growth, growth + 0.02):
        for r in (discount_rate - 0.01, discount_rate, discount_rate + 0.01):
            hand_computed_cells.append(round(dcf_per_share(fcf0, g, terminal_growth, r, shares, 0.0)["per_share"], 2))
    hand_lo, hand_hi = min(hand_computed_cells), max(hand_computed_cells)
    assert hand_lo == pytest.approx(168.82)
    assert hand_hi == pytest.approx(285.37)

    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {"shares": shares, "latest_fy": 2023, "fcf": fcf0, "net_debt": 0.0}
    assumptions = _assumptions(base_growth=growth, base_terminal=terminal_growth, base_discount=discount_rate)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(fcf0)
    assert result["dcf"]["scenarios"]["base"]["per_share"] == pytest.approx(216.77, abs=0.01)
    # The engine's own grid-derived band must match the independently
    # hand-computed lo/hi above -- both from dcf.scenarios (the source) and
    # from fair_value_range (the headline, which for a "mature" non-cyclical
    # non-hyper filer is built straight from dcf.scenarios).
    assert result["dcf"]["scenarios"]["base"]["lo"] == pytest.approx(hand_lo)
    assert result["dcf"]["scenarios"]["base"]["hi"] == pytest.approx(hand_hi)
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(hand_lo)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(hand_hi)
