"""Hand-verified numeric tests for ``valuation.triangulate``, ``valuation.sector``,
and the ``valuation.engine.run_valuation`` integration (SPEC.md Sec.8, 10, 11).

See the module docstring of ``test_valuation_dcf.py`` for the general
methodology (independent hand arithmetic in a comment above each assertion).
"""

import pytest

from sec_analyzer.config import Config
from sec_analyzer.valuation.engine import (
    _build_pb_roe,
    _justified_pb,
    _non_sbc_dilution,
    _select_latest_ffo,
    run_valuation,
)
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


def test_triangulate_upside_divergence_floors_confidence_and_flags_verdict():
    # All three say "ucuz" (would normally be YÜKSEK), but the base band low
    # (250) is 2.5x the price (100) -> model-market divergence: three method
    # votes read off the SAME assumption set, so unanimity isn't independent
    # confirmation. Confidence is floored to DÜŞÜK and a verdict-action
    # divergence payload is attached. direction stays "ucuz" (the verdict
    # rename is a presentation-layer step, not done here).
    result = triangulate(
        price=100, dcf_base_band={"lo": 250, "hi": 300}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["signals"]["dcf"] == "ucuz"
    assert result["confidence"] == "DÜŞÜK"
    assert result["divergence"] is not None
    assert result["divergence"]["direction"] == "ucuz"
    assert result["divergence"]["action"] == "verdict"
    assert result["divergence"]["factor"] == pytest.approx(2.5)
    assert result["divergence"]["band_edge"] == 250
    assert "Model-piyasa ayrışması" in result["rationale"]["confidence"]


def test_triangulate_downside_divergence_is_log_only_and_leaves_confidence():
    # price 100, base band [30,40]: hi (40) is 0.4x price (< 0.5x) -> downside
    # divergence. All three say "pahali" -> confidence must STAY YÜKSEK
    # (down-side is log-only this pass); only the payload is recorded, with
    # action="log_only", so the coming low-side pass inherits the data.
    result = triangulate(
        price=100, dcf_base_band={"lo": 30, "hi": 40}, implied_growth=0.20,
        realized_cagr=None, base_growth=0.10, pe_pct=90, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["signals"]["dcf"] == "pahali"
    assert result["confidence"] == "YÜKSEK"
    assert result["direction"] == "pahali"
    assert result["divergence"]["action"] == "log_only"
    assert result["divergence"]["direction"] == "pahali"
    assert result["divergence"]["factor"] == pytest.approx(0.4)


def test_triangulate_no_divergence_within_normal_range():
    # base low (150) is 1.5x price (100): below the 2.0x up trigger; hi (180)
    # is 1.8x, above the 0.5x down trigger -> no divergence, unanimity stands.
    result = triangulate(
        price=100, dcf_base_band={"lo": 150, "hi": 180}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["divergence"] is None
    assert result["confidence"] == "YÜKSEK"


def test_triangulate_divergence_up_trigger_is_strict():
    # base low exactly 2.0x price -> NOT triggered (strict >); just past fires.
    at = triangulate(100, {"lo": 200, "hi": 260}, 0.05, None, 0.10, 20, None, None, "mature")
    past = triangulate(100, {"lo": 200.01, "hi": 260}, 0.05, None, 0.10, 20, None, None, "mature")
    assert at["divergence"] is None
    assert past["divergence"] is not None
    assert past["divergence"]["action"] == "verdict"


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


def test_triangulate_multiples_uses_pffo_first_for_reit_not_pe():
    # For reit (Package 2 / SPEC.md Sec.8/FFO), P/FFO is primary, NOT P/E:
    # pe_pct=90 (would be "pahali" if used) and ps_pct=50 (would be "makul")
    # are both present, but pffo_pct=10 must win -> "ucuz".
    result = triangulate(
        None, None, None, None, None, pe_pct=90, ps_pct=50, pfcf_pct=None,
        sector_type="reit", pffo_pct=10,
    )
    assert result["signals"]["multiples"] == "ucuz"
    assert "P/FFO" in result["rationale"]["multiples"]


def test_triangulate_multiples_reit_falls_back_to_ps_when_pffo_missing():
    # pffo_pct absent (e.g. no Depreciation data) -> reit falls back to
    # ps_pct as its next candidate, still never P/E.
    result = triangulate(
        None, None, None, None, None, pe_pct=90, ps_pct=10, pfcf_pct=None,
        sector_type="reit", pffo_pct=None,
    )
    assert result["signals"]["multiples"] == "ucuz"


# ---------------------------------------------------------------------------
# 8b. triangulate -- growth-adjusted (PEG / EV-Sales) divergence -> "karisik"
# multiples signal (VALUATION.md Sec.7).
# ---------------------------------------------------------------------------


def test_triangulate_multiples_karisik_when_raw_and_growth_adjusted_diverge():
    # Raw P/E percentile 88 (expensive) but growth-adjusted PEG percentile 45
    # (fair) -> the two components land in different buckets -> "karisik".
    result = triangulate(
        None, None, None, None, None, 88, None, None, "mature",
        raw_growth_pair_pct=88, growth_adj_pct=45,
    )
    assert result["signals"]["multiples"] == "karisik"
    assert "karışık" in result["rationale"]["multiples"]


def test_triangulate_multiples_not_karisik_when_both_components_agree():
    # Raw 88 (expensive) and growth-adjusted 82 (expensive) agree -> the
    # existing raw signal stands unchanged ("pahali"), not "karisik".
    result = triangulate(
        None, None, None, None, None, 88, None, None, "mature",
        raw_growth_pair_pct=88, growth_adj_pct=82,
    )
    assert result["signals"]["multiples"] == "pahali"


def test_triangulate_multiples_unchanged_when_growth_adjusted_missing():
    # No growth-adjusted percentile -> pre-PEG behavior preserved exactly.
    result = triangulate(
        None, None, None, None, None, 88, None, None, "mature",
        raw_growth_pair_pct=88, growth_adj_pct=None,
    )
    assert result["signals"]["multiples"] == "pahali"


def test_triangulate_multiples_karisik_lowers_confidence_vs_agreement():
    # DCF + reverse-DCF both say "pahali"; a "karisik" multiples signal can't
    # join that majority, so confidence is ORTA (2 agree), not YÜKSEK.
    result = triangulate(
        110, {"lo": 90, "hi": 100}, 0.20, 0.10, 0.10, 88, None, None, "mature",
        raw_growth_pair_pct=88, growth_adj_pct=45,
    )
    assert result["signals"]["multiples"] == "karisik"
    assert result["signals"]["dcf"] == "pahali"
    assert result["confidence"] == "ORTA"


# ---------------------------------------------------------------------------
# 8c. triangulate -- leverage gate: EV/EBITDA is the PRIMARY multiple for a
# leveraged filer, ahead of P/E (SPEC.md Sec.6/Sec.10, VALUATION.md Sec.7).
# ---------------------------------------------------------------------------


def test_triangulate_multiples_uses_ev_ebitda_first_when_leveraged():
    # net_debt/EBITDA = 2.0 >= 1.0 -> leveraged. pe_pct=90 (would be "pahali"),
    # but ev_ebitda_pct=15 (<30) must win -> "ucuz", and the rationale names
    # FD/FAVÖK rather than P/E.
    result = triangulate(
        None, None, None, None, None, pe_pct=90, ps_pct=50, pfcf_pct=None,
        sector_type="mature", ev_ebitda_pct=15, net_debt_to_ebitda=2.0,
    )
    assert result["signals"]["multiples"] == "ucuz"
    assert "FD/FAVÖK" in result["rationale"]["multiples"]


def test_triangulate_multiples_keeps_pe_when_not_leveraged():
    # net_debt/EBITDA = 0.4 < 1.0 -> NOT leveraged. Even though ev_ebitda_pct
    # is present, P/E (=90 -> "pahali") stays primary; EV/EBITDA is ignored.
    result = triangulate(
        None, None, None, None, None, pe_pct=90, ps_pct=50, pfcf_pct=None,
        sector_type="mature", ev_ebitda_pct=15, net_debt_to_ebitda=0.4,
    )
    assert result["signals"]["multiples"] == "pahali"


def test_triangulate_multiples_leveraged_falls_back_to_pe_when_no_ev_history():
    # Leveraged but ev_ebitda_pct is None (no usable EV history) -> the P/E ->
    # P/S -> P/FCF fallback order is used, so pe_pct=90 -> "pahali".
    result = triangulate(
        None, None, None, None, None, pe_pct=90, ps_pct=50, pfcf_pct=None,
        sector_type="mature", ev_ebitda_pct=None, net_debt_to_ebitda=3.0,
    )
    assert result["signals"]["multiples"] == "pahali"


def test_triangulate_leveraged_skips_pe_based_peg_divergence():
    # A P/E-vs-PEG divergence (raw 88 vs growth-adj 45) would normally flip the
    # signal to "karisik". But when EV/EBITDA is primary (leveraged), that
    # P/E-based axis is skipped, so the EV/EBITDA own-history read (15 -> ucuz)
    # stands instead of being masked as "karisik".
    result = triangulate(
        None, None, None, None, None, pe_pct=88, ps_pct=None, pfcf_pct=None,
        sector_type="mature", ev_ebitda_pct=15, net_debt_to_ebitda=1.5,
        raw_growth_pair_pct=88, growth_adj_pct=45,
    )
    assert result["signals"]["multiples"] == "ucuz"


def test_triangulate_leverage_gate_ignored_for_reit():
    # reit keeps P/FFO primary regardless of leverage: pffo_pct=10 -> "ucuz"
    # wins over a present ev_ebitda_pct=90.
    result = triangulate(
        None, None, None, None, None, pe_pct=50, ps_pct=50, pfcf_pct=None,
        sector_type="reit", pffo_pct=10, ev_ebitda_pct=90, net_debt_to_ebitda=5.0,
    )
    assert result["signals"]["multiples"] == "ucuz"
    assert "P/FFO" in result["rationale"]["multiples"]


# ---------------------------------------------------------------------------
# 7b. triangulate -- sector-relative multiples axis (VALUATION.md Sec.7
# axis-b): sector_ratio = current primary multiple / Damodaran sector median.
# Band: > 1.25 expensive-vs-sector, < 0.80 cheap-vs-sector, else in line.
# ---------------------------------------------------------------------------


def test_triangulate_multiples_karisik_when_own_history_and_sector_diverge():
    # Own-history P/E percentile 88 (expensive vs its own past) but the
    # current P/E is 0.5x the sector median (cheap vs peers) -> the two axes
    # disagree -> "karisik".
    result = triangulate(
        None, None, None, None, None, 88, None, None, "mature",
        sector_ratio=0.5,
    )
    assert result["signals"]["multiples"] == "karisik"
    assert "sektör medyanına göre" in result["rationale"]["multiples"]


def test_triangulate_multiples_not_karisik_when_own_history_and_sector_agree():
    # Own-history expensive (88) AND expensive vs sector (1.5x median) agree
    # -> raw "pahali" stands, with a sector-confirmation clause appended.
    result = triangulate(
        None, None, None, None, None, 88, None, None, "mature",
        sector_ratio=1.5,
    )
    assert result["signals"]["multiples"] == "pahali"
    assert "Sektör medyanına göre de pahalı" in result["rationale"]["multiples"]


def test_triangulate_multiples_sector_axis_disabled_when_ratio_none():
    # No sector_ratio -> pure own-history behavior preserved exactly.
    result = triangulate(
        None, None, None, None, None, 88, None, None, "mature",
        sector_ratio=None,
    )
    assert result["signals"]["multiples"] == "pahali"
    assert "sektör" not in result["rationale"]["multiples"].lower()


def test_triangulate_multiples_sector_band_boundaries_are_strict():
    # Own-history "makul" (pct 50) so any sector divergence is purely axis-b.
    # ratio exactly 1.25 / 0.80 stays in-line (makul); just past flips it.
    at_hi = triangulate(None, None, None, None, None, 50, None, None, "mature", sector_ratio=1.25)
    above_hi = triangulate(None, None, None, None, None, 50, None, None, "mature", sector_ratio=1.26)
    at_lo = triangulate(None, None, None, None, None, 50, None, None, "mature", sector_ratio=0.80)
    below_lo = triangulate(None, None, None, None, None, 50, None, None, "mature", sector_ratio=0.79)

    assert at_hi["signals"]["multiples"] == "makul"       # in-line, own-history makul
    assert above_hi["signals"]["multiples"] == "karisik"  # expensive vs sector, makul own -> mixed
    assert at_lo["signals"]["multiples"] == "makul"
    assert below_lo["signals"]["multiples"] == "karisik"  # cheap vs sector, makul own -> mixed


def test_triangulate_multiples_peg_divergence_takes_precedence_over_sector():
    # Both divergences present. PEG divergence (raw 88 vs growth-adj 45) is
    # higher precedence: the rationale shows the PEG sentence, not the sector
    # one, though both resolve the signal to "karisik".
    result = triangulate(
        None, None, None, None, None, 88, None, None, "mature",
        raw_growth_pair_pct=88, growth_adj_pct=45, sector_ratio=1.5,
    )
    assert result["signals"]["multiples"] == "karisik"
    assert "büyümeye göre normalize" in result["rationale"]["multiples"]
    assert "sektör medyanına göre" not in result["rationale"]["multiples"]


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


def test_classify_sector_reit_like_real_estate_operator_sic():
    # 6500 (real estate) and 6512 (operators of apartment buildings, inside
    # 6510-6519) get the same FFO treatment as 6798 REITs (SPEC Sec.8).
    assert classify_sector(6500, {}, {}) == "reit"
    assert classify_sector(6512, {}, {}) == "reit"


def test_classify_sector_real_estate_agents_and_developers_stay_financial():
    # 6531 (real estate agents/managers) and 6552 (land subdividers/
    # developers) are asset-light/inventory businesses, not depreciable-
    # property owners -- explicitly excluded from the reit widening.
    assert classify_sector(6531, {}, {}) == "financial"
    assert classify_sector(6552, {}, {}) == "financial"


def test_classify_sector_financial_sic_range():
    assert classify_sector(6022, {}, {}) == "financial"  # state commercial bank
    assert classify_sector(6021, {}, {}) == "financial"  # national commercial bank
    # 6798 itself is excluded from the financial range (tested separately).
    assert classify_sector(6799, {}, {}) == "financial"


def test_classify_sector_cyclical_sic_singleton_semiconductors():
    # Unknown-growth semi (empty metrics -> CAGR is None) -> stays on the
    # "cyclical" default path. A secular-growth semi (CAGR > 15%) instead
    # falls through to the profitability check -- see the dedicated
    # semiconductor tests below.
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


def test_classify_sector_secular_growth_semiconductor_profitable_gives_mature():
    # SIC 3674 with realized revenue CAGR (0.30) above the 15% secular-growth
    # threshold falls through to the profitability check instead of being
    # force-classified as cyclical; positive latest-FY NetIncome -> mature.
    normalized = {"annual": {"NetIncome": [_ni_record(2023, 50.0)]}}
    metrics = {"latest_fy": 2023, "revenue_cagr_5y": 0.30}
    assert classify_sector(3674, normalized, metrics) == "mature"


def test_classify_sector_secular_growth_semiconductor_unprofitable_gives_growth_unprofitable():
    # Same secular-growth semi, but negative latest-FY NetIncome ->
    # growth_unprofitable (independent of hyper-grower detection).
    normalized = {"annual": {"NetIncome": [_ni_record(2023, -50.0)]}}
    metrics = {"latest_fy": 2023, "revenue_cagr_5y": 0.30}
    assert classify_sector(3674, normalized, metrics) == "growth_unprofitable"


def test_classify_sector_low_growth_semiconductor_stays_cyclical():
    # Realized CAGR (0.10) at or below the 15% threshold -> stays cyclical
    # (commodity/memory-type semi), even with no normalized/NetIncome data.
    assert classify_sector(3674, {}, {"revenue_cagr_5y": 0.10}) == "cyclical"


# ---------------------------------------------------------------------------
# 9a. classify_sector one-off-loss guard (P3c)
# ---------------------------------------------------------------------------


def test_classify_sector_one_off_loss_with_profitable_history_gives_mature():
    # Latest FY (2023) is a loss, but >=2 prior years (2021, 2022) are
    # profitable, including the immediately prior year (2022) -> treated as a
    # one-off (writedown/litigation/tax charge), classifies "mature".
    normalized = {"annual": {"NetIncome": [
        _ni_record(2021, 40.0), _ni_record(2022, 60.0), _ni_record(2023, -50.0),
    ]}}
    metrics = {"latest_fy": 2023}
    assert classify_sector(7372, normalized, metrics) == "mature"


def test_classify_sector_structural_loss_stays_growth_unprofitable():
    # Latest FY loss AND the immediately prior year is also a loss (majority
    # negative) -> structural, not a one-off -> stays growth_unprofitable.
    normalized = {"annual": {"NetIncome": [
        _ni_record(2021, 40.0), _ni_record(2022, -20.0), _ni_record(2023, -50.0),
    ]}}
    metrics = {"latest_fy": 2023}
    assert classify_sector(7372, normalized, metrics) == "growth_unprofitable"


def test_classify_sector_loss_with_fewer_than_two_prior_years_stays_growth_unprofitable():
    # Only 1 prior fiscal year of data (2022) -- not enough history to call
    # the latest-FY loss a one-off -> stays growth_unprofitable.
    normalized = {"annual": {"NetIncome": [
        _ni_record(2022, 60.0), _ni_record(2023, -50.0),
    ]}}
    metrics = {"latest_fy": 2023}
    assert classify_sector(7372, normalized, metrics) == "growth_unprofitable"


def test_classify_sector_loss_with_prior_year_loss_but_older_years_profitable_stays_growth_unprofitable():
    # Profitable majority overall (2020, 2021 positive vs. 2022 negative),
    # but the immediately PRIOR year (2022) is itself a loss -> the
    # "immediately prior year profitable" requirement fails, so this is NOT
    # treated as a one-off -> stays growth_unprofitable.
    normalized = {"annual": {"NetIncome": [
        _ni_record(2020, 30.0), _ni_record(2021, 40.0), _ni_record(2022, -10.0),
        _ni_record(2023, -50.0),
    ]}}
    metrics = {"latest_fy": 2023}
    assert classify_sector(7372, normalized, metrics) == "growth_unprofitable"


# ---------------------------------------------------------------------------
# 10. engine.run_valuation integration
# ---------------------------------------------------------------------------

_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity", "Depreciation",
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
    # roe=0.15 (latest FY), discount_rate_base=0.10, terminal_growth_base=0.03
    # (the `_assumptions` helper's default), equity_latest=1000, shares=100.
    #
    # P3a: fair_pb is the growth-aware justified P/B (ROE - g) / (r - g), NOT
    # the no-growth ROE/r:
    #   fair_pb_base = clamp((0.15-0.03)/(0.10-0.03), 0.5, 4.0)
    #                = clamp(0.12/0.07, ...) = 1.714286
    #   book_value_per_share = 1000/100 = 10.0
    #   bear: fair_pb = 1.714286*0.8 -> per_share = 13.71
    #   base: fair_pb = 1.714286*1.0 -> per_share = 17.14
    #   bull: fair_pb = 1.714286*1.2 -> per_share = 20.57
    #
    # F3: the band is the min/max of re-clamping fair_pb at discount_rate +/-
    # 1pp (0.09, 0.10, 0.11), g=0.03 held FIXED across the band (mirroring how
    # the DCF band holds terminal_growth fixed), scaled by this scenario's own
    # scale/book_value_per_share:
    #   fair_pb(dr=0.09) = clamp((0.15-0.03)/(0.09-0.03), 0.5, 4.0) = 2.0
    #   fair_pb(dr=0.10) = 1.714286 (center, matches per_share above)
    #   fair_pb(dr=0.11) = clamp((0.15-0.03)/(0.11-0.03), 0.5, 4.0) = 1.5
    #
    #   bear (scale=0.8): cells = 2.0*0.8*10=16.0, 13.71, 1.5*0.8*10=12.0
    #     -> lo=12.0, hi=16.0
    #   base (scale=1.0): cells = 20.0, 17.14, 15.0 -> lo=15.0, hi=20.0
    #   bull (scale=1.2): cells = 24.0, 20.57, 18.0 -> lo=18.0, hi=24.0
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
    assert pb["bear"]["per_share"] == pytest.approx(13.71)
    assert pb["bear"]["lo"] == pytest.approx(12.0)
    assert pb["bear"]["hi"] == pytest.approx(16.0)
    assert pb["base"]["per_share"] == pytest.approx(17.14)
    assert pb["base"]["lo"] == pytest.approx(15.0)
    assert pb["base"]["hi"] == pytest.approx(20.0)
    assert pb["bull"]["per_share"] == pytest.approx(20.57)
    assert pb["bull"]["lo"] == pytest.approx(18.0)
    assert pb["bull"]["hi"] == pytest.approx(24.0)

    # fair_value_range must be built FROM the pb_roe scenarios when DCF is
    # disabled.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(15.0)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(20.0)


def test_run_valuation_leverage_flag_true_when_net_debt_to_ebitda_at_least_one():
    # metrics carry net_debt=200 and ebitda=100 -> ratio 2.0 >= 1.0 ->
    # leveraged. The multiples output surfaces both the ratio and the flag
    # (SPEC.md Sec.11), independent of price history.
    normalized = _normalized({"Revenue": [_rec(2023, 1000.0)], "EPS": [_rec(2023, 2.0)]})
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 200.0, "ebitda": 100.0}
    result = run_valuation(
        normalized, [{"fy": 2023, "fcf": 100.0}], metrics, price=20.0, price_df=None,
        assumptions=_assumptions(), sector_type="mature",
    )
    assert result["multiples"]["net_debt_to_ebitda"] == pytest.approx(2.0)
    assert result["multiples"]["leveraged"] is True


def test_run_valuation_leverage_flag_false_for_net_cash_or_missing_ebitda():
    # Net cash (negative net_debt) -> not leveraged, ratio None.
    normalized = _normalized({"Revenue": [_rec(2023, 1000.0)], "EPS": [_rec(2023, 2.0)]})
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": -50.0, "ebitda": 100.0}
    result = run_valuation(
        normalized, [{"fy": 2023, "fcf": 100.0}], metrics, price=20.0, price_df=None,
        assumptions=_assumptions(), sector_type="mature",
    )
    assert result["multiples"]["net_debt_to_ebitda"] is None
    assert result["multiples"]["leveraged"] is False

    # Positive net debt but EBITDA missing -> ratio can't be formed -> False.
    metrics_no_ebitda = {"shares": 100.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 200.0, "ebitda": None}
    result2 = run_valuation(
        normalized, [{"fy": 2023, "fcf": 100.0}], metrics_no_ebitda, price=20.0, price_df=None,
        assumptions=_assumptions(), sector_type="mature",
    )
    assert result2["multiples"]["net_debt_to_ebitda"] is None
    assert result2["multiples"]["leveraged"] is False


# ---------------------------------------------------------------------------
# 8c. REIT FFO-based Gordon-growth anchor (Package 2 / SPEC.md Sec.8c)
# ---------------------------------------------------------------------------


def test_select_latest_ffo_divides_by_fy_own_share_count():
    # NetIncome[2023]=800, Depreciation[2023]=200 -> FFO=1000.
    # SharesOutstanding[2023]=100 (the FY's OWN count) differs from
    # metrics["shares"]=150 (a larger, later/current count, as it would be
    # for a REIT that has issued more equity since FY2023).
    # ffo_per_share must use the FY's own count: 1000/100 = 10.0, NOT
    # 1000/150 = 6.666... -- proving the current point-in-time count is no
    # longer used when the FY's own share count is available.
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
        "SharesOutstanding": [_rec(2023, 100.0)],
    })
    metrics = {"shares": 150.0}

    ffo_per_share, selected_fy = _select_latest_ffo(normalized, metrics)

    assert selected_fy == 2023
    assert ffo_per_share == pytest.approx(10.0)


def test_select_latest_ffo_falls_back_to_metrics_shares_when_fy_shares_missing():
    # Same FFO=1000, but NO SharesOutstanding entry at all for FY2023 (or any
    # other FY) -> falls back to metrics["shares"]=100 -> 1000/100 = 10.0.
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
    })
    metrics = {"shares": 100.0}

    ffo_per_share, selected_fy = _select_latest_ffo(normalized, metrics)

    assert selected_fy == 2023
    assert ffo_per_share == pytest.approx(10.0)


def test_select_latest_ffo_falls_back_when_fy_shares_present_for_other_fy_only():
    # SharesOutstanding has an entry, but not for the selected FY (2023) --
    # only for 2022 -- so shares_series.get(2023) is None and the fallback
    # to metrics["shares"] kicks in, same as the fully-missing case above.
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
        "SharesOutstanding": [_rec(2022, 90.0)],
    })
    metrics = {"shares": 100.0}

    ffo_per_share, selected_fy = _select_latest_ffo(normalized, metrics)

    assert selected_fy == 2023
    assert ffo_per_share == pytest.approx(10.0)


def test_select_latest_ffo_returns_none_per_share_when_neither_shares_source_available():
    # Neither the FY's own SharesOutstanding NOR metrics["shares"] is
    # available (metrics["shares"] missing/0/None) -> per-share result is
    # None, but selected_fy is still populated since FFO itself computed
    # fine (NetIncome/Depreciation were both present).
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
    })
    metrics = {}

    ffo_per_share, selected_fy = _select_latest_ffo(normalized, metrics)

    assert ffo_per_share is None
    assert selected_fy == 2023


def test_run_valuation_reit_sector_computes_ffo_gordon_growth():
    # NetIncome_fy=800, Depreciation_fy=200 -> FFO=1000; shares=100 ->
    # ffo_per_share = 1000/100 = 10.0.
    #
    # Gordon multiple per scenario = (1+g)/(r-g); per_share = ffo_per_share *
    # multiple. Hand arithmetic (also the spec's own worked example for
    # base: ffo_per_share=10, r=0.09, g=0.025 -> multiple=1.025/0.065=
    # 15.76923... -> per_share=157.69):
    #   bear (r=0.13, g=0.02): multiple = 1.02/0.11 = 9.272727 -> 92.73
    #   base (r=0.09, g=0.025): multiple = 1.025/0.065 = 15.769231 -> 157.69
    #   bull (r=0.085, g=0.03): multiple = 1.03/0.055 = 18.727273 -> 187.27
    #
    # Bands: recompute the Gordon multiple at r +/- 1pp (g fixed at this
    # scenario's own value), take min/max, round 2dp.
    #   bear: r in {0.12, 0.13, 0.14}, g=0.02
    #     r=0.12: 1.02/0.10=10.2 -> 102.00
    #     r=0.13: 9.272727 -> 92.73 (center)
    #     r=0.14: 1.02/0.12=8.5 -> 85.00
    #     -> lo=85.00, hi=102.00
    #   base: r in {0.08, 0.09, 0.10}, g=0.025
    #     r=0.08: 1.025/0.055=18.636364 -> 186.36
    #     r=0.09: 15.769231 -> 157.69 (center)
    #     r=0.10: 1.025/0.075=13.666667 -> 136.67
    #     -> lo=136.67, hi=186.36
    #   bull: r in {0.075, 0.085, 0.095}, g=0.03
    #     r=0.075: 1.03/0.045=22.888889 -> 228.89
    #     r=0.085: 18.727273 -> 187.27 (center)
    #     r=0.095: 1.03/0.065=15.846154 -> 158.46
    #     -> lo=158.46, hi=228.89
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
    })
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.13, "story": "Ayı."},
        "base": {"growth_5y": 0.08, "terminal_growth": 0.025, "discount_rate": 0.09, "story": "Baz."},
        "bull": {"growth_5y": 0.12, "terminal_growth": 0.03, "discount_rate": 0.085, "story": "Boğa."},
    }

    result = run_valuation(
        normalized, [], metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )

    assert result["dcf"]["enabled"] is False
    assert "FFO" in result["dcf"]["disabled_reason"]
    assert result["pb_roe"] is None  # FFO succeeded -> no fallback needed.

    ffo = result["ffo"]
    assert ffo is not None
    assert ffo["ffo_per_share"] == pytest.approx(10.0)

    bear = ffo["scenarios"]["bear"]
    assert bear["per_share"] == pytest.approx(92.73)
    assert bear["lo"] == pytest.approx(85.00)
    assert bear["hi"] == pytest.approx(102.00)

    base = ffo["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(157.69)
    assert base["lo"] == pytest.approx(136.67)
    assert base["hi"] == pytest.approx(186.36)

    bull = ffo["scenarios"]["bull"]
    assert bull["per_share"] == pytest.approx(187.27)
    assert bull["lo"] == pytest.approx(158.46)
    assert bull["hi"] == pytest.approx(228.89)

    # Implied fair P/FFO multiple per scenario, rounded 1dp.
    assert ffo["implied_pffo"]["bear"] == pytest.approx(9.3)
    assert ffo["implied_pffo"]["base"] == pytest.approx(15.8)
    assert ffo["implied_pffo"]["bull"] == pytest.approx(18.7)

    # The headline fair_value_range and the triangulation base band must
    # come FROM the ffo block for reit, not pb_roe.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(136.67)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(186.36)


def test_run_valuation_reit_falls_back_to_pb_roe_when_depreciation_missing():
    # Same fixture as the financial P/B x ROE test above (equity + roe
    # present, NO Depreciation data at all) but sector_type="reit": FFO
    # can't be built (no fiscal year has both NetIncome and Depreciation),
    # so the engine must gracefully fall back to the same P/B x ROE anchor
    # `financial` uses, with a note explaining the fallback.
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )

    assert result["ffo"] is None
    assert result["pb_roe"] is not None
    pb = result["pb_roe"]["scenarios"]
    # Same numbers as test_run_valuation_financial_sector_disables_dcf_and_computes_pb_roe
    # (P3a: growth-aware justified P/B with the base scenario's terminal_growth=0.03).
    assert pb["base"]["per_share"] == pytest.approx(17.14)
    assert pb["base"]["lo"] == pytest.approx(15.0)
    assert pb["base"]["hi"] == pytest.approx(20.0)

    # Headline fair_value_range must fall back to pb_roe too.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(15.0)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(20.0)

    assert any("FFO" in n and "P/B x ROE" in n for n in result["notes"])


def test_run_valuation_reit_falls_back_to_pb_roe_when_ffo_non_positive():
    # Depreciation data IS present, but NetIncome_fy + Depreciation_fy <= 0
    # (a loss-making year whose D&A add-back still isn't enough to turn FFO
    # positive) -> _build_ffo must return None (not a negative/zero FFO
    # headline), and the engine falls back to P/B x ROE exactly as when
    # Depreciation was missing entirely.
    normalized = _normalized({
        "StockholdersEquity": [_rec(2023, 1000.0)],
        "NetIncome": [_rec(2023, -500.0)],
        "Depreciation": [_rec(2023, 200.0)],  # FFO = -500+200 = -300 <= 0.
    })
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )

    assert result["ffo"] is None
    assert result["pb_roe"] is not None
    # P3a: growth-aware justified P/B (base terminal_growth=0.03), same as
    # test_run_valuation_financial_sector_disables_dcf_and_computes_pb_roe.
    assert result["pb_roe"]["scenarios"]["base"]["per_share"] == pytest.approx(17.14)


def test_run_valuation_reit_ffo_reduced_by_gain_on_sale():
    # Same base as test_run_valuation_reit_sector_computes_ffo_gordon_growth
    # (NetIncome=800, Depreciation=200 -> base FFO=1000), but with a
    # GainOnSaleRealEstate=150 tagged for the SAME fiscal year: Nareit FFO
    # removes gains on real-estate sales, so FFO = 1000 - 150 = 850.
    # ffo_per_share = 850 / 100 = 8.5.
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
    })
    normalized["annual"]["GainOnSaleRealEstate"] = [_rec(2023, 150.0)]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, [], metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )
    assert result["ffo"] is not None
    assert result["ffo"]["ffo_per_share"] == pytest.approx(8.5)


def test_run_valuation_reit_ffo_increased_by_impairment():
    # Same base FFO=1000, but a RealEstateImpairment=150 for the same FY is
    # ADDED BACK (it's a non-cash expense that already reduced net income):
    # FFO = 1000 + 150 = 1150. ffo_per_share = 1150 / 100 = 11.5.
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
    })
    normalized["annual"]["RealEstateImpairment"] = [_rec(2023, 150.0)]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, [], metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )
    assert result["ffo"] is not None
    assert result["ffo"]["ffo_per_share"] == pytest.approx(11.5)


def test_run_valuation_reit_ffo_negative_gain_ie_loss_is_added_back():
    # A negative "GainOnSaleRealEstate" value represents a LOSS on a
    # real-estate sale (a us-gaap GainLoss element is negative for a loss).
    # "- gain" with gain=-300 becomes "+300", so it's added back exactly
    # like an impairment: FFO = 1000 - (-300) = 1300.
    # ffo_per_share = 1300 / 100 = 13.0.
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
    })
    normalized["annual"]["GainOnSaleRealEstate"] = [_rec(2023, -300.0)]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, [], metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )
    assert result["ffo"] is not None
    assert result["ffo"]["ffo_per_share"] == pytest.approx(13.0)


def test_run_valuation_reit_ffo_unchanged_when_no_re_adjustment_tags():
    # Backward compatibility: neither GainOnSaleRealEstate nor
    # RealEstateImpairment is present (same fixture as the base FFO test
    # above them) -> FFO must be IDENTICAL to before this change existed,
    # i.e. NetIncome_fy + Depreciation_fy with no adjustment applied:
    # ffo_per_share = 1000 / 100 = 10.0.
    normalized = _normalized({
        "NetIncome": [_rec(2023, 800.0)],
        "Depreciation": [_rec(2023, 200.0)],
    })
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, [], metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )
    assert result["ffo"] is not None
    assert result["ffo"]["ffo_per_share"] == pytest.approx(10.0)


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


def test_run_valuation_select_fcf0_uses_fundamental_fy_not_ghost_latest_fy():
    # Reproduces the "ghost fiscal year" anchor mismatch (see
    # test_metrics.py's ghost-year fixtures): metrics["latest_fy"]=2025 is a
    # cover-page (SharesOutstanding-only) fiscal year with NO ratios/fcf data
    # at all, while metrics["latest_fundamental_fy"]=2024 is the real
    # anchor, where ratios actually reports fcf.
    #
    # _select_fcf0 resolves its anchor via resolve_fundamental_fy(metrics),
    # so it must use 2024, NOT 2025:
    #   sbc_adjusted_fcf_by_fy = {2024: 100.0 - 0 (no SBC)} = {2024: 100.0}
    #   ttm_fcf = window.get(2024) = 100.0 (positive -> usable)
    #   avg window = [fcf.get(2024), fcf.get(2023), fcf.get(2022)]
    #             = [100.0, None, None] -> avg_window=[100.0] -> avg_fcf=100.0
    #   deviation = |100.0 - 100.0| / 100.0 = 0 -> NOT > 0.50 -> ttm kept as-is.
    #   -> fcf0=100.0, source="ttm".
    #
    # Contrast (the pre-fix bug): if latest_fy=2025 had been used instead,
    # sbc_adjusted_fcf_by_fy.get(2025) would be None (ratios has no fy=2025
    # entry at all) -> fcf0 would resolve to None -> DCF would be disabled
    # entirely (scenarios=None) for a company with perfectly good FCF data.
    normalized = _normalized({})
    ratios = [{"fy": 2024, "fcf": 100.0}]
    metrics = {
        "shares": 10.0, "latest_fy": 2025, "latest_fundamental_fy": 2024,
        "fcf": 100.0, "net_debt": 0.0,
    }
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["fcf0"] == pytest.approx(100.0)
    assert result["fcf0_source"] == "ttm"
    assert result["dcf"]["scenarios"] is not None
    assert result["dcf"]["scenarios"]["base"]["per_share"] is not None


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
    assert result["multiples"]["current"] == {
        "pe": None, "ps": None, "pfcf": None, "pffo": None, "ev_ebit": None, "ev_ebitda": None,
    }

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
    #
    # P3a: growth-aware justified P/B (base terminal_growth=0.03, the
    # `_assumptions` helper's default):
    #   fair_pb_base = clamp((0.15-0.03)/(0.10-0.03), 0.5, 4.0) = 1.714286
    #   book_value_per_share = 1000/100 = 10.0
    #   base: fair_pb = 1.714286*1.0 -> per_share = 17.14
    #
    # F3: band = min/max of fair_pb re-clamped at discount_rate +/- 1pp, g=0.03
    # held fixed (see
    # test_run_valuation_financial_sector_disables_dcf_and_computes_pb_roe's
    # comment for the full 0.09/0.10/0.11 cell derivation) -> lo=15.0, hi=20.0.
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2024, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    pb = result["pb_roe"]["scenarios"]
    assert pb["base"]["per_share"] == pytest.approx(17.14)
    assert pb["base"]["lo"] == pytest.approx(15.0)
    assert pb["base"]["hi"] == pytest.approx(20.0)
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
# 9b. _justified_pb / P/B x ROE growth-awareness (P3a)
# ---------------------------------------------------------------------------


def test_justified_pb_growth_aware_formula():
    # (ROE - g) / (r - g) = (0.15 - 0.03) / (0.10 - 0.03) = 0.12/0.07 = 1.714286.
    assert _justified_pb(0.15, 0.10, 0.03) == pytest.approx(1.714286, abs=1e-6)


def test_justified_pb_degrades_to_no_growth_form_when_g_missing():
    # g=None -> degrades to ROE/r = 0.15/0.10 = 1.5 (backward-compat).
    assert _justified_pb(0.15, 0.10, None) == pytest.approx(1.5)


def test_justified_pb_degrades_to_no_growth_form_when_g_negative():
    # A negative g is treated as degenerate for this multiple -> degrades to
    # ROE/r = 1.5, same as g missing.
    assert _justified_pb(0.15, 0.10, -0.02) == pytest.approx(1.5)


def test_justified_pb_degrades_to_no_growth_form_when_denominator_non_positive():
    # g >= r makes (r - g) non-positive/meaningless -> degrades to ROE/r.
    assert _justified_pb(0.15, 0.10, 0.10) == pytest.approx(1.5)
    assert _justified_pb(0.15, 0.10, 0.12) == pytest.approx(1.5)


def test_justified_pb_no_longer_clamps_above_reference_ceiling():
    # (0.30 - 0.03) / (0.05 - 0.03) = 0.27/0.02 = 13.5. WP5: the justified P/B
    # is no longer clamped to _PB_CLAMP_HI=4.0 -- a high-ROE compounder can
    # legitimately warrant a justified P/B above 4, so the raw ratio is now
    # returned as-is; _build_pb_roe flags (does not clamp) values outside
    # [_PB_CLAMP_LO, _PB_CLAMP_HI] instead.
    assert _justified_pb(0.30, 0.05, 0.03) == pytest.approx(13.5)


def test_build_pb_roe_flags_above_reference_without_clamping():
    # WP5: _build_pb_roe no longer clamps fair_pb to [_PB_CLAMP_LO=0.5,
    # _PB_CLAMP_HI=4.0] -- it flags an out-of-band value instead. Same
    # roe/dr/g as test_justified_pb_no_longer_clamps_above_reference_ceiling
    # above (0.30, 0.05, 0.03): fair_pb = (0.30-0.03)/(0.05-0.03) =
    # 0.27/0.02 = 13.5, strictly above 4.0 -> justified_pb_flag =
    # "above_reference". book_value_per_share = equity/shares = 1000/100 =
    # 10.0 -> scenarios (scale 0.8/1.0/1.2, unclamped):
    #   bear = 13.5*0.8*10 = 108.0, base = 13.5*10 = 135.0, bull = 13.5*1.2*10 = 162.0
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.30}]
    metrics = {"shares": 100.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.05, "terminal_growth": 0.03}}

    result, notes = _build_pb_roe(assumptions, normalized, metrics, ratios)

    assert result is not None
    assert result["fair_pb"] == pytest.approx(13.5)
    assert result["justified_pb_flag"] == "above_reference"
    assert result["scenarios"]["bear"]["per_share"] == pytest.approx(108.0)
    assert result["scenarios"]["base"]["per_share"] == pytest.approx(135.0)
    assert result["scenarios"]["bull"]["per_share"] == pytest.approx(162.0)
    assert any(
        "13.50x" in n and "referans aralığının dışında" in n and "kırpılmadı" in n for n in notes
    )


def test_build_pb_roe_flags_below_reference_without_clamping():
    # roe=0.05, discount_rate=0.20, terminal_growth=None -> _justified_pb
    # degrades g to 0.0 (missing) -> fair_pb = (0.05-0)/(0.20-0) = 0.25,
    # strictly below _PB_CLAMP_LO=0.5 -> justified_pb_flag = "below_reference".
    # book_value_per_share = 1000/100 = 10.0 -> scenarios:
    #   bear = 0.25*0.8*10 = 2.0, base = 0.25*10 = 2.5, bull = 0.25*1.2*10 = 3.0
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.05}]
    metrics = {"shares": 100.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.20, "terminal_growth": None}}

    result, notes = _build_pb_roe(assumptions, normalized, metrics, ratios)

    assert result is not None
    assert result["fair_pb"] == pytest.approx(0.25)
    assert result["justified_pb_flag"] == "below_reference"
    assert result["scenarios"]["bear"]["per_share"] == pytest.approx(2.0)
    assert result["scenarios"]["base"]["per_share"] == pytest.approx(2.5)
    assert result["scenarios"]["bull"]["per_share"] == pytest.approx(3.0)
    assert any("0.25x" in n and "referans aralığının dışında" in n for n in notes)


def test_build_pb_roe_flag_is_none_when_fair_pb_inside_reference_band():
    # roe=0.15, discount_rate=0.10, terminal_growth=0.03 -> fair_pb =
    # (0.15-0.03)/(0.10-0.03) = 0.12/0.07 = 1.714286, well inside
    # [0.5, 4.0] -> justified_pb_flag stays None, no out-of-band note.
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10, "terminal_growth": 0.03}}

    result, notes = _build_pb_roe(assumptions, normalized, metrics, ratios)

    assert result is not None
    assert result["justified_pb_flag"] is None
    assert not any("referans aralığının dışında" in n for n in notes)


def test_build_pb_roe_unavailable_when_roe_at_or_below_growth():
    # A loss-making (or sub-terminal-growth) filer drives the justified P/B
    # non-positive: roe=-0.09, discount_rate=0.10, terminal_growth=0.04 ->
    # fair_pb = (-0.09-0.04)/(0.10-0.04) = -0.13/0.06 = -2.17 (<= 0). A
    # negative price-to-book multiple is economically meaningless, so the
    # anchor is unavailable (None) with an explanatory note -- NOT a negative
    # per-share fair value (regression guard for the MSTR anomaly).
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": -0.09}]
    metrics = {"shares": 100.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10, "terminal_growth": 0.04}}

    result, notes = _build_pb_roe(assumptions, normalized, metrics, ratios)

    assert result is None
    assert any("pozitif değil" in n and "P/D x ROE çapası hesaplanamadı" in n for n in notes)


def test_build_pb_roe_unavailable_when_book_value_non_positive():
    # Negative book equity makes book value per share <= 0, so even a positive
    # justified P/B (roe=0.15, dr=0.10, g=0.03 -> fair_pb=1.71) would produce a
    # negative per-share value -- the anchor is unavailable instead.
    normalized = _normalized({"StockholdersEquity": [_rec(2023, -500.0)]})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10, "terminal_growth": 0.03}}

    result, notes = _build_pb_roe(assumptions, normalized, metrics, ratios)

    assert result is None
    assert any("defter değeri" in n and "pozitif değil" in n for n in notes)


def test_run_valuation_pb_roe_backward_compat_when_terminal_growth_missing():
    # Same fixture as test_run_valuation_financial_sector_disables_dcf_and_computes_pb_roe
    # but with base terminal_growth=None -> _justified_pb degrades to the old
    # ROE/r form, so fair_pb_base = 0.15/0.10 = 1.5 (unchanged from before P3a).
    normalized = _normalized({"StockholdersEquity": [_rec(2023, 1000.0)]})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _assumptions(base_discount=0.10, base_terminal=None)

    result = run_valuation(
        normalized, ratios, metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    pb = result["pb_roe"]["scenarios"]
    assert pb["base"]["per_share"] == pytest.approx(15.0)
    assert pb["base"]["lo"] == pytest.approx(13.64)
    assert pb["base"]["hi"] == pytest.approx(16.67)


# ---------------------------------------------------------------------------
# 11a. engine._non_sbc_dilution (WP1) -- shared by both the hyper-grower and
# mid-growth revenue-first DCF paths (SPEC.md's dilution rule): SBC is
# already expensed as a margin drag in both paths' FCF projections, so
# projecting future per-share dilution from the RAW `shares_yoy` (which
# itself embeds SBC-driven issuance) would double-count that same cost. When
# `market_cap` is usable, the SBC-implied share-issuance rate
# (`sbc_latest / market_cap`) is netted out of `shares_yoy` before clamping;
# without a usable `market_cap`, it falls back to the raw-`shares_yoy`-
# clamped behavior unchanged. Neither branch was previously exercised with a
# positive `shares_yoy` anywhere in the suite (every existing hyper/mid-growth
# fixture sets `shares_yoy: None`), so this is a genuine coverage gap, not a
# style choice -- these tests call the private helper directly, mirroring how
# `_justified_pb`/`_select_latest_ffo` are unit-tested above.
# ---------------------------------------------------------------------------


def test_non_sbc_dilution_returns_zero_when_shares_yoy_missing_or_non_positive():
    metrics_missing = {"shares_yoy": None, "market_cap": 1000.0}
    metrics_zero = {"shares_yoy": 0.0, "market_cap": 1000.0}
    metrics_negative = {"shares_yoy": -0.02, "market_cap": 1000.0}

    for metrics in (metrics_missing, metrics_zero, metrics_negative):
        rate, note, excluded = _non_sbc_dilution(metrics, _normalized({}), 2023)
        assert rate == pytest.approx(0.0)
        assert note is None
        assert excluded == pytest.approx(0.0)


def test_non_sbc_dilution_falls_back_to_raw_shares_yoy_clamp_when_market_cap_missing():
    # shares_yoy=0.03 (under the 5% cap), no market_cap key at all -> the
    # SBC-netting branch is skipped entirely (pre-WP1 behavior): rate is the
    # raw shares_yoy, clamped to [0.0, 0.05]; no note, no exclusion.
    normalized = _normalized({"SBC": [_rec(2023, 20.0)]})  # present but unused: no market_cap to divide by.
    metrics = {"shares_yoy": 0.03}

    rate, note, excluded = _non_sbc_dilution(metrics, normalized, 2023)

    assert rate == pytest.approx(0.03)
    assert note is None
    assert excluded == pytest.approx(0.0)


def test_non_sbc_dilution_raw_clamp_also_applies_when_market_cap_non_positive():
    # market_cap present but <= 0 (or non-numeric) is treated the same as
    # missing -- still the raw-clamp fallback, not a division by a bad value.
    normalized = _normalized({})
    metrics_zero_cap = {"shares_yoy": 0.03, "market_cap": 0.0}
    metrics_non_numeric = {"shares_yoy": 0.03, "market_cap": "n/a"}

    for metrics in (metrics_zero_cap, metrics_non_numeric):
        rate, note, excluded = _non_sbc_dilution(metrics, normalized, 2023)
        assert rate == pytest.approx(0.03)
        assert note is None
        assert excluded == pytest.approx(0.0)


def test_non_sbc_dilution_raw_shares_yoy_clamped_at_hyper_dilution_cap():
    # shares_yoy=0.10 is above the 5% cap (_HYPER_DILUTION_CAP) -> clamped
    # down to 0.05, even with no market_cap to trigger SBC netting at all.
    normalized = _normalized({})
    metrics = {"shares_yoy": 0.10}

    rate, note, excluded = _non_sbc_dilution(metrics, normalized, 2023)

    assert rate == pytest.approx(0.05)
    assert note is None
    assert excluded == pytest.approx(0.0)


def test_non_sbc_dilution_nets_sbc_out_when_market_cap_positive():
    # shares_yoy=0.03, market_cap=1000, SBC_fy2023=20 -> sbc_dilution =
    # 20/1000 = 0.02. non_sbc = max(0, 0.03-0.02) = 0.01 -> rate =
    # clamp(0.01, 0, 0.05) = 0.01 -- strictly LESS than the raw 0.03, proving
    # the SBC-implied issuance was actually netted out (not just clamped).
    normalized = _normalized({"SBC": [_rec(2023, 20.0)]})
    metrics = {"shares_yoy": 0.03, "market_cap": 1000.0}

    rate, note, excluded = _non_sbc_dilution(metrics, normalized, 2023)

    assert rate == pytest.approx(0.01)
    assert excluded == pytest.approx(0.02)
    assert note is not None
    assert "çift sayım önlendi" in note


def test_non_sbc_dilution_clamped_to_zero_when_sbc_exceeds_shares_yoy_but_note_still_fires():
    # shares_yoy=0.03, market_cap=1000, SBC_fy2023=50 -> sbc_dilution=0.05 >
    # shares_yoy=0.03 -> non_sbc=max(0, 0.03-0.05)=0.0 -> rate=0.0, but the
    # note still fires (sbc_dilution>0 and shares_yoy>0, per the function's
    # own condition) -- the netting note describes WHY the projected
    # dilution is (correctly) zero, rather than silently vanishing.
    normalized = _normalized({"SBC": [_rec(2023, 50.0)]})
    metrics = {"shares_yoy": 0.03, "market_cap": 1000.0}

    rate, note, excluded = _non_sbc_dilution(metrics, normalized, 2023)

    assert rate == pytest.approx(0.0)
    assert excluded == pytest.approx(0.05)
    assert note is not None


def test_non_sbc_dilution_market_cap_positive_but_sbc_missing_behaves_like_raw_clamp():
    # market_cap IS usable, but no SBC data exists for `fy` at all -> sbc_latest
    # degrades to 0.0 (per the docstring), so sbc_dilution=0.0 -> the "note
    # fires" condition (`sbc_dilution > 0.0`) never trips -- same numeric
    # rate as the raw-clamp fallback, but reached via the market_cap-positive
    # branch instead of skipping it.
    normalized = _normalized({})  # no SBC series at all
    metrics = {"shares_yoy": 0.03, "market_cap": 1000.0}

    rate, note, excluded = _non_sbc_dilution(metrics, normalized, 2023)

    assert rate == pytest.approx(0.03)
    assert note is None
    assert excluded == pytest.approx(0.0)


def test_non_sbc_dilution_fy_none_degrades_sbc_to_zero_even_with_market_cap_and_sbc_data():
    # fy=None (e.g. resolve_fundamental_fy couldn't resolve one) -> the
    # function never looks up SBC at all (`to_annual_series(...).get(fy)`
    # is skipped by the `fy is not None` guard) -- sbc_latest degrades to
    # 0.0 even though this normalized dict DOES have a 2023 SBC entry,
    # because fy itself is None, not 2023.
    normalized = _normalized({"SBC": [_rec(2023, 20.0)]})
    metrics = {"shares_yoy": 0.03, "market_cap": 1000.0}

    rate, note, excluded = _non_sbc_dilution(metrics, normalized, None)

    assert rate == pytest.approx(0.03)
    assert note is None
    assert excluded == pytest.approx(0.0)


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
#
# Discount rates: engine._HYPER_DISCOUNT_RATE_BY_SCENARIO is
# {"bear": 0.14, "base": 0.12, "bull": 0.10} (raised from an earlier
# 0.12/0.10/0.09 -- see valuation/sanity.py's ERP-spread-guard package). The
# revenue/margin paths (and hence final_year_revenue/revenue_multiple) never
# depend on discount_rate, only the discounted per_share/lo/hi numbers do --
# those were re-derived by calling revenue_dcf.revenue_first_dcf directly
# with the new rates (same function/approach as before).
#
# WP3 (Damodaran discount-rate fade): the fixture below uses
# _assumptions(base_discount=0.10), so `_run_valuation` derives
# mature_discount_rate = max(0.10, terminal_growth(0.025) + sanity.
# _MIN_ERP_SPREAD(0.045)=0.07) = 0.10. Every scenario's revenue-first DCF now
# fades from its OWN cohort rate (14/12/10) down to that shared 0.10 by
# steady_state_year=10 -- bull's cohort rate already equals 0.10, so the
# CENTER cell of its 3x3 band is unaffected, but bear/base's center cells and
# EVERY scenario's off-center grid cells shift because a fixed mature target
# combined with a moved row rate changes the average discount over the fade
# window. All numbers below were re-derived with a from-scratch
# reimplementation of the fade formula (growth-path/margin-path/discount-
# path/PV/terminal-value, NOT calling revenue_dcf.revenue_first_dcf) and then
# cross-checked against the real run_valuation() output before being pinned
# here -- both matched to the displayed digits.
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


def test_run_valuation_hyper_grower_uses_revenue_first_dcf_as_headline(tmp_path):
    # Trigger: realized_cagr=0.50 (>0.25) AND fcf=-50<=0 (clause a) AND
    # fcf_margin=-50/1000=-0.05<0.05 (clause b) -- both fire.
    #
    # WP5: _HYPER_START_GROWTH_CAP raised 0.40 -> 0.60, so growth_anchor=0.50
    # (realized_cagr; latest_yoy is None here since only FY2023 revenue is in
    # the fixture, so growth_anchor falls back to realized_cagr verbatim) is
    # now UNCAPPED for base: start_growth=min(0.50,0.60)=0.50 (was 0.40).
    # bull=min(0.50*1.2=0.60,0.60)=0.60 -- still exactly at the (now higher)
    # cap. bear=min(0.50,0.60)*0.6=0.30 (was 0.24).
    #
    # base scenario: start_growth=0.50, target_fcf_margin=0.30, discount_rate=
    # 0.12 (fixed hyper base rate, the FADE's year-1/cohort rate),
    # current_margin=-50/1000=-0.05, annual_dilution=0.0 (shares_yoy None,
    # sbc_revenue 0) -> revenue path fades 0.50->0.025 over 10y. Independently
    # re-derived (not calling revenue_dcf.py) via
    # g_t = 0.5 + (0.025-0.5)*min(t-1,9)/9, revenue_t = revenue_{t-1}*(1+g_t):
    #   growth_path = [0.5, 0.44722, 0.39444, 0.34167, 0.28889, 0.23611,
    #                  0.18333, 0.13056, 0.07778, 0.025]
    #   revenue_path = [1500.00, 2170.83, 3027.11, 4061.37, 5234.65, 6470.61,
    #                   7656.89, 8656.54, 9329.83, 9563.07]
    # final_year_revenue=9563.07 (multiple~=9.563x), matching the real
    # run_valuation() output exactly. 9.563 now sits in (8, 15] ->
    # arrival_flag "agresif" (was "makul" under the old, lower start_growth),
    # unaffected by the discount-rate fade -- the revenue/margin paths never
    # depend on discount_rate.
    #
    # This test is about hyper-grower MECHANICS (headline selection), not
    # about WP2's risk-free-linked terminal growth, so it pins
    # terminal_growth to the old fixed 0.025 by pointing damodaran_dir at a
    # directory that doesn't exist (load_sector_data returns None -> the
    # anchor falls back to engine._HYPER_TERMINAL_GROWTH=0.025). Without
    # this, run_valuation's default damodaran_dir picks up the repo's real
    # data/damodaran/erp.csv (risk_free=4.20) and the anchor becomes 0.04,
    # invalidating every hand-verified number below (see
    # test_run_valuation_hyper_grower_terminal_growth_linked_to_risk_free for
    # a dedicated test of that WP2 linkage).
    #
    # F3: financing_shares is derived once from the base scenario's own
    # (financing_shares=0) preliminary fcf_path: the margin path (independent
    # of start_growth) is margin_t=-0.05+0.35*t/10, negative only at t=1
    # (margin_1=-0.015); with the new revenue_1=1000*1.5=1500 (was 1000*1.4=
    # 1400), fcf_1=1500*(-0.015)=-22.5 (was -21.0) is the sole negative-FCF
    # year, so burn=-22.5 (undiscounted, discount_rate-independent since it
    # only affects the PV/TV step, not the fcf_path itself), financing_shares
    # = abs(-22.5)/price(50.0) = 0.45 (was 0.42), reused for bear/base/bull
    # alike.
    #
    # WP3 discount-rate fade: assumptions base_discount=0.10 ->
    # mature_discount_rate = max(0.10, 0.025+0.045) = 0.10. Each scenario's
    # revenue-first DCF now fades from its own cohort rate (bear 0.14, base
    # 0.12, bull 0.10) down to 0.10 by steady_state_year=10, and the terminal
    # value discounts at 0.10 instead of the cohort rate. The base band is
    # the min/max of a 3x3 grid over start_growth +/- 2pp (0.48/0.50/0.52) x
    # discount_rate +/- 1pp (0.11/0.12/0.13) -- each row's discount_rate is
    # itself the FADE's year-1 rate, fading to the same shared 0.10 -- with
    # everything else (target_fcf_margin=0.30, current_margin=-0.05,
    # steady_state=10, annual_dilution=0.0, financing_shares=0.45) held
    # fixed. Re-derived by calling the actual (fixed) revenue_dcf.revenue_
    # first_dcf directly for all 9 cells and cross-checked against the real
    # run_valuation() output (both matched exactly):
    #   grid cells (rounded to 2dp) = [180.14, 172.36, 164.97,
    #                                  193.75, 185.39, 177.43,
    #                                  208.25, 199.26, 190.70]
    #   -> lo=min=164.97, hi=max=208.25 (center cell 185.39 matches per_share
    #      below, as expected).
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
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail is not None

    # Headline fair_value_range.base must equal the hyper base band...
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(detail["scenarios"]["base"]["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(detail["scenarios"]["base"]["hi"])
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(164.97)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(208.25)

    # ...NOT the standard FCF-DCF band, which is still computed and reported
    # (secondary) under dcf.scenarios, and is clearly a different number.
    assert result["dcf"]["scenarios"] is not None
    assert result["dcf"]["scenarios"]["base"]["per_share"] is not None
    assert result["dcf"]["scenarios"]["base"]["per_share"] != detail["scenarios"]["base"]["per_share"]
    assert result["dcf"]["scenarios"]["base"]["lo"] != result["fair_value_range"]["base"]["lo"]

    # hyper_growth_detail is fully populated.
    assert detail["reasons"][0].startswith("Gelir CAGR %50.0")
    assert detail["scenarios"]["bear"]["start_growth"] == pytest.approx(0.30)
    assert detail["scenarios"]["base"]["start_growth"] == pytest.approx(0.50)
    assert detail["scenarios"]["bull"]["start_growth"] == pytest.approx(0.60)
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
    assert detail["arrival_flag"] == "agresif"
    assert detail["tam_usd"] is None
    assert detail["implied"]["growth"] is not None
    assert detail["implied"]["revenue_10y"] is not None
    assert detail["implied"]["revenue_multiple"] is not None
    assert detail["implied"]["steady_state_margin"] is not None
    assert detail["implied"]["tam_share"] is None  # no tam_usd supplied
    assert detail["target_margin_source"] == "brüt marj × 0.5"

    # Turkish hyper-mode note is present in the top-level notes.
    assert any(
        "Hiper-büyüme modu: manşet aralığı revenue-first DCF'ten" in n for n in result["notes"]
    )

    # WP3: the discount-rate fade is active (mature_discount_rate derived
    # from the base assumptions' own CAPM-aware discount rate) and its
    # Turkish note is surfaced.
    assert detail["mature_discount_rate"] == pytest.approx(0.10)
    assert any(
        "iskonto oranı sabit tutulmadı" in n and "Damodaran fade" in n for n in result["notes"]
    )

    # Display consistency (Milestone F): fair_value_range's base scenario
    # growth/discount_rate/note must reflect the hyper revenue-first DCF's
    # OWN inputs (start_growth=0.50, discount=0.12, target margin=0.30) --
    # NOT the standard clamped assumptions (base_growth=0.10 in the fixture
    # above), which the headline band no longer actually uses once
    # hyper-grower mode takes over.
    base_fvr = result["fair_value_range"]["base"]
    assert "başlangıç" in base_fvr["growth"]
    assert "%50" in base_fvr["growth"]
    assert "%25 büyüme" not in base_fvr["growth"]
    assert base_fvr["discount_rate"] == "%12"
    assert "Hiper-büyüme" in base_fvr["note"]
    assert "%30" in base_fvr["note"]  # mature target FCF margin


def test_run_valuation_hyper_grower_deceleration_guard_caps_start_growth(tmp_path):
    # Deceleration guard (A): with a prior-year revenue present, the latest
    # realized YoY is computable and BELOW the blended growth anchor, so
    # base/bull start growth are capped at the realized YoY -- a visibly
    # decelerating grower is not assumed to first re-accelerate before fading.
    #
    #   Revenue FY2023=1000, FY2022=800 -> latest_yoy = 1000/800 - 1 = 0.25.
    #   metrics revenue_cagr_5y=0.50 -> growth_anchor = 0.5*0.50 + 0.5*0.25
    #                                                 = 0.375.
    #   raw start growth: bear=min(0.375,0.60)*0.6=0.225, base=0.375,
    #                     bull=min(0.375*1.2=0.45,0.60)=0.45.
    #   decel_cap=0.25: bear 0.225<=0.25 kept; base 0.375->0.25; bull 0.45->0.25.
    normalized = _normalized({"Revenue": [_rec(2023, 1000.0), _rec(2022, 800.0)]})
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)
    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )
    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail["scenarios"]["bear"]["start_growth"] == pytest.approx(0.225)
    assert detail["scenarios"]["base"]["start_growth"] == pytest.approx(0.25)
    assert detail["scenarios"]["bull"]["start_growth"] == pytest.approx(0.25)
    # base and bull now share the same (capped) start growth, so bull's only
    # remaining lever is its margin uplift -- the "two 1.2x optimisms compound"
    # problem is gone in rule-based mode.
    assert any("Yavaşlama koruması" in n for n in result["notes"])


def test_run_valuation_hyper_grower_start_growth_override_allows_reacceleration(tmp_path):
    # AI-mode escape hatch: an explicit per-scenario start_growth override in
    # hyper_growth_extras is honored even above the deceleration cap, and a
    # deviation note is surfaced (re-acceleration is allowed, but never silent).
    normalized = _normalized({"Revenue": [_rec(2023, 1000.0), _rec(2022, 800.0)]})
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)
    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
        hyper_growth_extras={"per_scenario": {"base": {"start_growth": 0.45}}},
    )
    detail = result["hyper_growth_detail"]
    assert detail["scenarios"]["base"]["start_growth"] == pytest.approx(0.45)
    assert any("re-acceleration" in n.lower() for n in result["notes"])


def test_run_valuation_hyper_grower_terminal_growth_linked_to_risk_free(tmp_path):
    # WP2: the hyper-grower revenue-first DCF's terminal growth is no longer
    # the flat 2.5% fallback whenever real Damodaran reference data is
    # available -- it's linked to the risk-free rate instead
    # (min(risk_free, 4%), engine._run_valuation's `terminal_growth_anchor`).
    # This test is intentionally ROBUST (direction + note presence), not a
    # brittle pinned-magnitude test, because WP3's discount-rate fade (see
    # the headline hyper test above) also runs on top of this and would make
    # a fully hand-derived per_share fragile to re-derive independently here.
    #
    # Same hyper-grower fixture as the headline test above
    # (_HYPER_CONCEPTS_OVERRIDES/_HYPER_RATIOS/_HYPER_METRICS,
    # _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)):
    # realized_cagr=0.50 (>0.25) and fcf<=0 -> hyper-grower triggers.
    #
    # Real data/damodaran/erp.csv ships `region=US, erp=4.23, risk_free=4.20`
    # (repo file, read directly): risk_free_pct=4.20 -> terminal_growth_anchor
    # = min(4.20/100, sanity._TERMINAL_GROWTH_MAX=0.04) = 0.04. Passing
    # Config.DAMODARAN_DIR explicitly (rather than omitting the argument)
    # keeps this test independent of whatever the process cwd happens to be.
    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result_real_data = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=Config.DAMODARAN_DIR,
    )
    result_fallback = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    assert result_real_data["hyper_growth"] is True
    assert result_fallback["hyper_growth"] is True
    detail_real = result_real_data["hyper_growth_detail"]
    detail_fallback = result_fallback["hyper_growth_detail"]
    assert detail_real is not None
    assert detail_fallback is not None

    # Real-data anchor: the WP2 note fires and names the resolved 4.0% rate
    # (engine._build_hyper_growth's f-string uses `terminal_growth * 100:.1f`,
    # so 0.04 -> "%4.0").
    real_wp2_notes = [
        n for n in result_real_data["notes"] if "risksiz getiri oranına bağlandı" in n
    ]
    assert real_wp2_notes, "expected the WP2 terminal-growth note to fire with real Damodaran data"
    assert any("%4.0" in n for n in real_wp2_notes)

    # Nonexistent damodaran_dir: load_sector_data returns None -> risk_free is
    # unavailable -> terminal_growth stays the flat _HYPER_TERMINAL_GROWTH
    # fallback (0.025) and the WP2 note is ABSENT entirely (the note only
    # fires when `terminal_growth != _HYPER_TERMINAL_GROWTH`).
    fallback_wp2_notes = [
        n for n in result_fallback["notes"] if "risksiz getiri oranına bağlandı" in n
    ]
    assert not fallback_wp2_notes, "the WP2 note must not appear when no real risk-free rate was resolved"

    # Direction check (Gordon TV increases in g for r>g, and the growth-fade
    # path itself decelerates less over the projection window the higher the
    # terminal anchor is): the higher (0.04) real-data terminal-growth anchor
    # must yield a strictly higher base per_share than the lower (0.025)
    # fallback anchor, with every other input (start_growth, discount-rate
    # cohort, target margin, mature_discount_rate -- both resolve to 0.10
    # here since max(0.10, g + 0.045) <= 0.10 for both g=0.04 and g=0.025)
    # held identical.
    per_share_real = detail_real["scenarios"]["base"]["per_share"]
    per_share_fallback = detail_fallback["scenarios"]["base"]["per_share"]
    assert per_share_real is not None
    assert per_share_fallback is not None
    assert per_share_real > per_share_fallback

    # Both runs still land on the WP3 discount-rate fade (unaffected by which
    # terminal-growth anchor was used -- see the comment above).
    assert detail_real["mature_discount_rate"] == pytest.approx(0.10)
    assert detail_fallback["mature_discount_rate"] == pytest.approx(0.10)


def test_run_valuation_hyper_grower_nets_sbc_out_of_dilution_via_non_sbc_dilution(tmp_path):
    # WP1: same hyper-grower fixture as the headline test above, but with
    # shares_yoy=0.03 and market_cap=1000.0 added to metrics, plus a
    # SBC=20.0 tag for FY2023 -- engine._non_sbc_dilution (hand-verified
    # directly in the WP1 section above) nets the SBC-implied issuance rate
    # (20/1000=0.02) out of the raw shares_yoy (0.03) before the hyper
    # revenue-first DCF sees it: annual_dilution = clamp(0.03-0.02, 0, 0.05)
    # = 0.01 -- strictly LESS than the raw 0.03, proving the netting
    # actually ran for this call site too (the headline test above exercises
    # shares_yoy=None, i.e. the pre-WP1/no-op path).
    normalized = _normalized({**_HYPER_CONCEPTS_OVERRIDES, "SBC": [_rec(2023, 20.0)]})
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)
    metrics = {**_HYPER_METRICS, "shares_yoy": 0.03, "market_cap": 1000.0}

    result = run_valuation(
        normalized, _HYPER_RATIOS, metrics, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail is not None
    assert detail["annual_dilution"] == pytest.approx(0.01)
    assert detail["sbc_dilution_excluded"] == pytest.approx(0.02)
    assert any("çift sayım önlendi" in n for n in result["notes"])


def test_run_valuation_hyper_grower_target_margin_flag_above_30pct_reference(tmp_path):
    # WP4: same hyper-grower fixture as the headline test above, but with
    # gross_margin raised to 0.70 (was 0.60) -> _hyper_target_base: ceiling
    # = 0.70*0.5 = 0.35; current_margin = fcf/latest_revenue = -50/1000 =
    # -0.05 (unchanged, not > 0) -> base = ceiling = 0.35; min(0.35, gross_
    # margin=0.70) = 0.35. target_base=0.35 is strictly ABOVE the 30%
    # reference threshold (_HYPER_TARGET_BASE_CAP) -- WP4's point is that
    # this is now flagged/noted rather than silently clamped down to 0.30,
    # so target_margin_pct/scenario target margins actually reflect 0.35 (or
    # its per-scenario scaling), not a truncated 0.30.
    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)
    ratios = [
        {"fy": 2023, "gross_margin": 0.70, "fcf": -50.0},
        {"fy": 2022, "fcf": 100.0},
        {"fy": 2021, "fcf": 90.0},
    ]

    result = run_valuation(
        normalized, ratios, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]
    assert detail is not None
    assert detail["target_margin_flag"] == "above_reference"
    assert detail["target_margin_pct"] == pytest.approx(0.35)
    # Uncapped: the base scenario's own target_fcf_margin is 0.35 (scale
    # 1.0), not clamped down to the old 0.30 ceiling.
    assert detail["scenarios"]["base"]["target_fcf_margin"] == pytest.approx(0.35)
    assert any(
        "%35" in n and "%30 referans eşiğinin üzerinde" in n for n in result["notes"]
    )


def test_run_valuation_hyper_grower_extras_override_target_margin_and_tam(tmp_path):
    # Same fixture as above, but with hyper_growth_extras supplying a TAM
    # and an overridden base target_fcf_margin. The growth path (and hence
    # final_year_revenue) is UNCHANGED by a target-margin override
    # (revenue_first_dcf's growth path never depends on the margin path), so
    # this reuses the headline test's hand-verified/cross-checked
    # final_year_revenue=9563.0718 (WP5: base start_growth=min(0.50,0.60)=
    # 0.50, uncapped, vs. the old min(0.50,0.40)=0.40). tam_share =
    # 9563.0718 / 14000 ~= 0.6831 -- now ABOVE the 0.60 ("gecersiz")
    # TAM-share threshold (was ~0.467, "agresif", under the old, lower
    # start_growth) -> "gecersiz", which OVERRIDES what the revenue-
    # multiple-based flag would have been (9.563x -> "agresif" without TAM,
    # per the headline test above).
    #
    # Mechanics test (extras/TAM override), not WP2's macro linkage -- same
    # nonexistent-damodaran_dir trick as the headline test above pins
    # terminal_growth to the hand-verified 0.025 (final_year_revenue here
    # depends only on the growth path, which doesn't depend on terminal
    # growth at all in this fade formula up to steady_state_year=10, but the
    # per_share/band numbers implicitly checked via detail below do).
    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)
    extras = {"tam_usd": 14000.0, "per_scenario": {"base": {"target_fcf_margin": 0.5}}}

    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable", hyper_growth_extras=extras,
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    assert result["hyper_growth"] is True
    detail = result["hyper_growth_detail"]

    assert detail["tam_usd"] == pytest.approx(14000.0)
    # Overridden target margin used verbatim (bypasses the gross-margin cap
    # and the deterministic 0.30 that clause (b) would otherwise have set).
    assert detail["scenarios"]["base"]["target_fcf_margin"] == pytest.approx(0.5)
    assert detail["scenarios"]["base"]["final_year_revenue"] == pytest.approx(9563.0718, rel=1e-3)

    tam_share = detail["scenarios"]["base"]["final_year_revenue"] / 14000.0
    assert tam_share > 0.60
    assert detail["arrival_flag"] == "gecersiz"

    assert "LLM/kullanıcı" in detail["target_margin_source"]
    assert any("TAM" in n for n in detail["notes"])

    # implied_tam_share is computed from the implied 10y revenue / tam_usd.
    if detail["implied"]["revenue_10y"] is not None:
        assert detail["implied"]["tam_share"] == pytest.approx(
            detail["implied"]["revenue_10y"] / 14000.0
        )


def test_run_valuation_hyper_grower_triangulation_uses_yuksek_beklenti_signal(tmp_path):
    # Same hyper fixture as above, but a higher price this time so the DCF
    # signal reads "yuksek_beklenti" (price sits between the base and bull
    # bands' highs). F3 changed the hyper band from a flat +/-10% to a
    # grid-derived one (see the headline test above), which also shifts
    # financing_shares (it's derived from burn/price, so a different price
    # changes it too).
    #
    # WP5: _HYPER_START_GROWTH_CAP raised 0.40 -> 0.60 pushes both base
    # (start_growth=0.50, was 0.40) and bull (start_growth=min(0.6,0.6)=0.60,
    # was 0.40) revenue paths higher (see the headline test's comment for the
    # base derivation), which raises both bands' upper ends substantially --
    # the old price=170 fixture (chosen when base.hi~=144.62, bull.hi~=190.67)
    # no longer lands strictly below the new, much larger base.hi, so the
    # price is raised to 220 here, re-picked so the same base.hi < price <=
    # bull.hi inequality this test needs still holds. At price=220,
    # financing_shares = 22.5/220 ~= 0.1023 (burn=-22.5, same as the headline
    # test -- burn only depends on the base scenario's revenue/margin paths,
    # not on price): base.hi ~=208.98, bull.hi ~=391.05 (obtained by calling
    # the actual (fixed) run_valuation() directly at price=220 and cross-
    # checked against the headline test's independently-verified base-
    # scenario final_year_revenue=9563.07 -- the bull scenario's own
    # final_year_revenue=13733.69 was spot-checked the same way, by an
    # independent revenue-path re-derivation at start_growth=0.60), so the
    # triangulation DCF signal must be "yuksek_beklenti" (HYPER_SPEC.md
    # Sec.4), proving base_band/bull_band are actually threaded from
    # hyper_growth_detail into triangulate.triangulate(), not just computed
    # and left unused.
    #
    # Mechanics test (triangulation signal wiring) -- same nonexistent-
    # damodaran_dir trick pins terminal_growth to the hand-verified 0.025.
    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=220.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    detail = result["hyper_growth_detail"]
    assert detail is not None
    base_hi = detail["scenarios"]["base"]["hi"]
    bull_hi = detail["scenarios"]["bull"]["hi"]
    assert base_hi < 220.0 <= bull_hi  # sanity-check the fixture actually lands in the intended zone
    assert base_hi == pytest.approx(208.98, abs=0.05)
    assert bull_hi == pytest.approx(391.05, abs=0.05)

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


# ---------------------------------------------------------------------------
# 11. run_valuation -- point-in-time ("as-of") macro threading (Sec.11 of the
# as-of implementation): as_of=None must never carry "macro_asof"; as_of set
# must carry it (copied verbatim from damodaran.load_sector_data) plus two
# Turkish notes; the shared terminal-growth anchor must use the resolved
# as-of risk-free rate.
# ---------------------------------------------------------------------------


def test_run_valuation_without_as_of_never_carries_macro_asof_key(tmp_path):
    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0}
    assumptions = _assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
        damodaran_dir=str(tmp_path / "no_damodaran"),
    )

    assert "macro_asof" not in result


def test_run_valuation_as_of_copies_macro_asof_and_appends_notes(tmp_path):
    """as_of set + a resolvable macro (via fred_rate, even with an otherwise
    empty Damodaran directory) must copy `macro_asof` verbatim into the
    top-level result and append the documented Turkish provenance note naming
    the as-of macro sources ("Geçmiş tarih"). With an empty directory no
    anachronism warnings fire (nothing current to substitute), so only the
    single provenance note is present."""
    damodaran_dir = tmp_path / "damodaran"
    damodaran_dir.mkdir()  # exists but has no CSVs -- fred_rate alone must be enough

    normalized = _normalized({})
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0}
    assumptions = _assumptions()
    fred_rate = {"value_pct": 2.98, "date": "2022-06-29", "series": "DGS10"}

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
        damodaran_dir=str(damodaran_dir),
        as_of="2022-06-30", fred_rate=fred_rate,
    )

    assert result["macro_asof"] == {
        "as_of": "2022-06-30",
        "erp_source": "erp.csv (güncel değer)",
        "risk_free_source": "DGS10 (2022-06-29)",
        "multiples_source": "multiples.csv (güncel snapshot — anakronik)",
    }
    # Single provenance note naming all three macro sources; no warnings key
    # (empty dir -> nothing current substituted for a missing historical file).
    assert any(
        "Geçmiş tarih" in n and "2022-06-30" in n and "çarpan/beta kaynağı" in n
        for n in result["notes"]
    )
    assert "warnings" not in result["macro_asof"]


def test_run_valuation_hyper_grower_terminal_growth_anchor_uses_as_of_fred_rate(tmp_path):
    # Hand-verified: fred_rate value_pct=2.98 -> risk_free_pct=2.98 ->
    # terminal_growth_anchor = min(2.98/100, sanity._TERMINAL_GROWTH_MAX=0.04)
    #   = min(0.0298, 0.04) = 0.0298.
    # This mirrors test_run_valuation_hyper_grower_terminal_growth_linked_to_risk_free
    # above (which uses the REAL data/damodaran/erp.csv, resolving to 4.20%
    # and clamping at the 4% ceiling) but drives the same code path off an
    # as-of fred_rate instead, landing BELOW the ceiling so the resolved
    # value itself (2.98%, not the 4% cap) is what's under test. The engine's
    # own note f-string rounds to 1dp: 2.98 -> "3.0" (engine._build_hyper_growth
    # uses `terminal_growth * 100:.1f}`).
    damodaran_dir = tmp_path / "damodaran"
    damodaran_dir.mkdir()
    fred_rate = {"value_pct": 2.98, "date": "2022-06-29", "series": "DGS10"}

    normalized = _normalized(_HYPER_CONCEPTS_OVERRIDES)
    assumptions = _assumptions(base_growth=0.10, base_terminal=0.03, base_discount=0.10)

    result = run_valuation(
        normalized, _HYPER_RATIOS, _HYPER_METRICS, price=50.0, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
        damodaran_dir=str(damodaran_dir), as_of="2022-06-30", fred_rate=fred_rate,
    )

    assert result["hyper_growth"] is True
    wp2_notes = [n for n in result["notes"] if "risksiz getiri oranına bağlandı" in n]
    assert wp2_notes, "expected the WP2 terminal-growth note to fire with an as-of fred_rate"
    assert any("%3.0" in n for n in wp2_notes)
