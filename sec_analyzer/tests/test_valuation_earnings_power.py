"""Hand-verified numeric tests for the earnings-power-value (EPV) anchor and
the FCF-DCF reliability gate (SPEC.md Sec.8a): ``valuation.engine``'s
``_build_earnings_power``/``_fcf_dcf_unreliable`` and their ``run_valuation``
wiring, plus ``triangulate``'s confidence cap when the EPV anchor becomes the
headline.

See ``test_valuation_dcf.py``'s module docstring for the general methodology
(independent hand arithmetic in a comment above each assertion, checked with
``pytest.approx``). Unlike ``test_valuation_engine.py`` (which only exercises
``valuation.engine`` through the public ``run_valuation`` entry point), Parts
A and B below unit-test the two new private helpers directly -- their branch
logic (the margin-median sanity guard, the three-way suppressed/cash-backed/
investment-driven gate) is intricate enough that isolating it is clearer and
more thorough than only reaching it indirectly through ``run_valuation``.
"""

import pytest

from sec_analyzer.valuation.engine import _build_earnings_power, _fcf_dcf_unreliable, run_valuation
from sec_analyzer.valuation.triangulate import triangulate

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _annual(**concepts) -> dict:
    """Minimal ``normalized``-shaped dict for direct unit calls into
    ``_build_earnings_power``/``_fcf_dcf_unreliable``: ``_annual(NetIncome=
    {2023: 100.0}, Revenue={2023: 1000.0})`` -> ``{"annual": {"NetIncome":
    [{"fy": 2023, "value": 100.0}], "Revenue": [...]}}``. ``to_annual_series``
    only reads ``fy``/``value`` off each record, so this is sufficient
    without the fuller record shape ``test_valuation_engine.py`` uses for
    full ``run_valuation`` integration fixtures."""
    return {
        "annual": {
            concept: [{"fy": fy, "value": value} for fy, value in by_fy.items()]
            for concept, by_fy in concepts.items()
        }
    }


_EPV_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity",
]


def _rec_epv(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized_epv(overrides: "dict[str, dict[int, float]]") -> dict:
    """Fuller ``normalized``-shaped fixture (mirrors ``test_valuation_engine
    .py``'s ``_normalized``/``_rec``) for full ``run_valuation`` integration
    tests: ``overrides`` is ``{concept: {fy: value}}``."""
    annual = {
        concept: [_rec_epv(fy, value) for fy, value in (overrides.get(concept) or {}).items()] or None
        for concept in _EPV_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "EPV Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _EPV_CONCEPTS},
        "missing": [c for c in _EPV_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _EPV_CONCEPTS},
    }


def _epv_assumptions(base_discount=0.10):
    """Assumptions whose base discount rate is the only field
    ``_build_earnings_power`` actually reads; bear/bull growth/terminal
    values are irrelevant to EPV but kept distinct from base to mirror a
    realistic phase-1 assumption set."""
    return {
        "bear": {"growth_5y": 0.03, "terminal_growth": 0.01, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": base_discount, "story": "Baz."},
        "bull": {"growth_5y": 0.08, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }


# ---------------------------------------------------------------------------
# A. _build_earnings_power (unit)
# ---------------------------------------------------------------------------


def test_build_earnings_power_base_per_share_and_scenario_scale_hand_verified():
    # Single fiscal year only (2023): NI=100, Revenue=1000 -> margin=0.10.
    # With only one positive-margin year, the sanity guard's reference
    # margin (median of a 1-element list) trivially equals this year's own
    # margin, so ref_ni == latest_ni exactly -> sanity can never trigger
    # here, independent of the deviation threshold.
    #
    # dr_base (cost of equity) = 0.10, shares = 10.
    # base_value_per_share = NI / dr / shares = 100 / 0.10 / 10 = 100.0.
    #
    # Scenario point estimates (_PB_SCENARIO_SCALE bear=0.8/base=1.0/bull=1.2,
    # reused as-is per spec B3):
    #   bear = round(100.0*0.8, 2) = 80.0
    #   base = round(100.0*1.0, 2) = 100.0
    #   bull = round(100.0*1.2, 2) = 120.0
    #
    # Band (_epv_scenario_band): dr in {0.09, 0.10, 0.11} (dr_base +/-
    # sensitivity._DISCOUNT_RATE_STEP=0.01), cell = round(NI/dr*scale/shares, 2):
    #   base: dr=0.09 -> 100/0.09/10  = 111.1111... -> 111.11
    #         dr=0.10 -> 100.0
    #         dr=0.11 -> 100/0.11/10  =  90.9091...  ->  90.91
    #         -> lo=90.91, hi=111.11
    #   bear: dr=0.09 -> 111.1111*0.8 =  88.8889...  ->  88.89
    #         dr=0.10 -> 80.0
    #         dr=0.11 ->  90.9091*0.8 =  72.7273...  ->  72.73
    #         -> lo=72.73, hi=88.89
    #   bull: dr=0.09 -> 111.1111*1.2 = 133.3333...  -> 133.33
    #         dr=0.10 -> 120.0
    #         dr=0.11 ->  90.9091*1.2 = 109.0909...  -> 109.09
    #         -> lo=109.09, hi=133.33
    normalized = _annual(NetIncome={2023: 100.0}, Revenue={2023: 1000.0})
    metrics = {"shares": 10.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10}}

    detail, notes = _build_earnings_power(assumptions, normalized, metrics, ratios=[])

    assert detail is not None
    assert detail["sanity_applied"] is False
    assert detail["normalized_net_income"] == pytest.approx(100.0)
    assert detail["cost_of_equity"] == pytest.approx(0.10)
    assert detail["per_share"] == pytest.approx(100.0)

    base = detail["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(100.0)
    assert base["lo"] == pytest.approx(90.91)
    assert base["hi"] == pytest.approx(111.11)

    bear = detail["scenarios"]["bear"]
    assert bear["per_share"] == pytest.approx(80.0)
    assert bear["lo"] == pytest.approx(72.73)
    assert bear["hi"] == pytest.approx(88.89)

    bull = detail["scenarios"]["bull"]
    assert bull["per_share"] == pytest.approx(120.0)
    assert bull["lo"] == pytest.approx(109.09)
    assert bull["hi"] == pytest.approx(133.33)


def test_build_earnings_power_sanity_guard_triggers_when_latest_margin_deviates_over_50pct():
    # 3 prior years at a steady 10% margin, latest year spikes to a 20%
    # margin (e.g. a one-off mark-to-market gain, the Rivian-shaped case the
    # spec calls out):
    #   FY2020: rev=1000, ni=100 -> margin=0.10
    #   FY2021: rev=1000, ni=100 -> margin=0.10
    #   FY2022: rev=1000, ni=100 -> margin=0.10
    #   FY2023: rev=1000, ni=200 -> margin=0.20 (latest)
    # margins (all positive-NI/positive-revenue years, latest included) =
    # [0.10, 0.10, 0.10, 0.20]; median of 4 = avg(2nd, 3rd sorted) =
    # (0.10+0.10)/2 = 0.10.
    # ref_ni = 0.10 * latest_rev(1000) = 100.0.
    # deviation = |latest_ni/ref_ni - 1| = |200/100 - 1| = 1.0 > 0.5
    # (_EPV_SANITY_DEVIATION) -> sanity_applied=True, normalized_ni=
    # ref_ni=100.0 (NOT the raw 200.0).
    normalized = _annual(
        NetIncome={2020: 100.0, 2021: 100.0, 2022: 100.0, 2023: 200.0},
        Revenue={2020: 1000.0, 2021: 1000.0, 2022: 1000.0, 2023: 1000.0},
    )
    metrics = {"shares": 10.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10}}

    detail, notes = _build_earnings_power(assumptions, normalized, metrics, ratios=[])

    assert detail is not None
    assert detail["sanity_applied"] is True
    assert detail["normalized_net_income"] == pytest.approx(100.0)
    # per_share must be built off the normalized figure (100.0), not the
    # raw, distrusted one (200.0): 100/0.10/10 = 100.0, not 200.0.
    assert detail["scenarios"]["base"]["per_share"] == pytest.approx(100.0)
    assert any("marj medyanından" in n for n in notes)
    assert any("200" in n for n in notes)  # cites the distrusted raw figure


def test_build_earnings_power_sanity_guard_does_not_trigger_under_50pct_deviation():
    # Same 3 prior years at a 10% margin; latest year at a 14% margin this
    # time -- a real bump, but below the 50% guard (mirrors the spec's own
    # AMZN worked example, where a 16.5% deviation does not trigger):
    #   median margin (of [0.10,0.10,0.10,0.14]) = (0.10+0.10)/2 = 0.10
    #   ref_ni = 0.10*1000 = 100.0
    #   deviation = |140/100 - 1| = 0.40 (NOT > 0.5) -> guard does NOT fire.
    #   -> normalized_ni = latest_ni = 140.0, used as-is.
    normalized = _annual(
        NetIncome={2020: 100.0, 2021: 100.0, 2022: 100.0, 2023: 140.0},
        Revenue={2020: 1000.0, 2021: 1000.0, 2022: 1000.0, 2023: 1000.0},
    )
    metrics = {"shares": 10.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10}}

    detail, notes = _build_earnings_power(assumptions, normalized, metrics, ratios=[])

    assert detail is not None
    assert detail["sanity_applied"] is False
    assert detail["normalized_net_income"] == pytest.approx(140.0)
    assert detail["scenarios"]["base"]["per_share"] == pytest.approx(140.0 / 0.10 / 10.0)
    assert not any("marj medyanından" in n for n in notes)


def test_build_earnings_power_adds_advisory_note_for_high_implied_roe_but_does_not_clamp_value():
    # eq=10, normalized_ni=100 (single-year fixture, sanity never triggers,
    # same as the first test) -> implied roe = normalized_ni/eq = 100/10 =
    # 10.0; roe/dr_base = 10.0/0.10 = 100.0, far above the 4.0 pb_roe-style
    # ceiling (_PB_CLAMP_HI) -> an advisory note is appended, but the EPV
    # value itself is NOT clamped (base_value_per_share stays NI/dr/shares =
    # 100/0.10/10 = 100.0 -- EPV never touches equity/ROE in its valuation
    # math, only in this advisory check).
    normalized = _annual(
        NetIncome={2023: 100.0}, Revenue={2023: 1000.0}, StockholdersEquity={2023: 10.0},
    )
    metrics = {"shares": 10.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10}}

    detail, notes = _build_earnings_power(assumptions, normalized, metrics, ratios=[])

    assert detail is not None
    assert detail["scenarios"]["base"]["per_share"] == pytest.approx(100.0)
    assert any("yukarı-yanlı okumayın" in n for n in notes)


def test_build_earnings_power_none_without_valid_shares():
    normalized = _annual(NetIncome={2023: 100.0}, Revenue={2023: 1000.0})
    assumptions = {"base": {"discount_rate": 0.10}}

    detail, notes = _build_earnings_power(assumptions, normalized, {"shares": None, "latest_fy": 2023}, ratios=[])
    assert detail is None
    assert any("hisse sayısı" in n for n in notes)

    detail2, _ = _build_earnings_power(assumptions, normalized, {"shares": 0, "latest_fy": 2023}, ratios=[])
    assert detail2 is None

    detail3, _ = _build_earnings_power(assumptions, normalized, {"shares": -5.0, "latest_fy": 2023}, ratios=[])
    assert detail3 is None


def test_build_earnings_power_none_without_valid_discount_rate():
    normalized = _annual(NetIncome={2023: 100.0}, Revenue={2023: 1000.0})
    metrics = {"shares": 10.0, "latest_fy": 2023}

    detail, notes = _build_earnings_power({"base": {}}, normalized, metrics, ratios=[])
    assert detail is None
    assert any("iskonto oranı" in n for n in notes)

    detail2, _ = _build_earnings_power({"base": {"discount_rate": -0.05}}, normalized, metrics, ratios=[])
    assert detail2 is None

    detail3, _ = _build_earnings_power({"base": {"discount_rate": 0.0}}, normalized, metrics, ratios=[])
    assert detail3 is None

    detail4, _ = _build_earnings_power({}, normalized, metrics, ratios=[])
    assert detail4 is None


def test_build_earnings_power_none_when_latest_net_income_not_positive():
    metrics = {"shares": 10.0, "latest_fy": 2023}
    assumptions = {"base": {"discount_rate": 0.10}}

    # Missing NI entirely for the resolved fiscal year.
    normalized_missing = _annual(Revenue={2023: 1000.0})
    detail, notes = _build_earnings_power(assumptions, normalized_missing, metrics, ratios=[])
    assert detail is None
    assert any(("negatif" in n) or ("eksik" in n) for n in notes)

    # Negative NI.
    normalized_negative = _annual(NetIncome={2023: -50.0}, Revenue={2023: 1000.0})
    detail2, _ = _build_earnings_power(assumptions, normalized_negative, metrics, ratios=[])
    assert detail2 is None

    # Exactly zero NI (the "<= 0" branch, not just "< 0").
    normalized_zero = _annual(NetIncome={2023: 0.0}, Revenue={2023: 1000.0})
    detail3, _ = _build_earnings_power(assumptions, normalized_zero, metrics, ratios=[])
    assert detail3 is None


# ---------------------------------------------------------------------------
# B. _fcf_dcf_unreliable (unit) -- the cash-conversion-guarded gate. This is
# the CRITICAL branch per the spec: FCF suppression alone must never be
# sufficient to switch to the EPV headline.
# ---------------------------------------------------------------------------


def test_fcf_dcf_unreliable_gate_fires_when_suppressed_cash_backed_and_investment_driven():
    # epv_base=100 -> suppression threshold = 0.5*100=50; dcf_hi=40 < 50 ->
    # fcf_suppressed=True.
    # cash_backed: ocf=90 >= 0.8*ni(100)=80 -> True.
    # investment_driven: capex/ocf = 50/90 = 0.5556 >= 0.5 -> True.
    # All three hold -> gate fires: (True, None).
    earnings_power = {"scenarios": {"base": {"per_share": 100.0}}}
    dcf_scenarios = {"base": {"hi": 40.0}}
    normalized = _annual(OperatingCashFlow={2023: 90.0}, NetIncome={2023: 100.0}, CapEx={2023: 50.0})
    metrics = {"latest_fy": 2023}

    unreliable, note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics)
    assert unreliable is True
    assert note is None


def test_fcf_dcf_unreliable_gate_refuses_and_flags_quality_when_not_cash_backed():
    # CRITICAL regression guard: FCF suppression alone must NOT be enough to
    # switch to the EPV headline -- if net income isn't actually converting
    # to cash (a genuine earnings-quality problem, not growth investment),
    # the gate must refuse to fire and instead surface a red-flag note,
    # leaving the (correctly) suppressed FCF-DCF as the headline.
    # epv_base=100 -> threshold=50; dcf_hi=40 < 50 -> fcf_suppressed=True.
    # cash_backed: ocf=70 < 0.8*ni(100)=80 -> False.
    # -> gate must NOT fire: (False, <quality note>), never (True, None).
    earnings_power = {"scenarios": {"base": {"per_share": 100.0}}}
    dcf_scenarios = {"base": {"hi": 40.0}}
    normalized = _annual(OperatingCashFlow={2023: 70.0}, NetIncome={2023: 100.0}, CapEx={2023: 50.0})
    metrics = {"latest_fy": 2023}

    unreliable, note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics)
    assert unreliable is False
    assert note is not None
    assert "kazanç-kalitesi/nakde-çevirme uyarısıdır" in note
    assert "OCF < 0.8" in note


def test_fcf_dcf_unreliable_gate_does_not_fire_when_fcf_not_suppressed():
    # epv_base=100 -> threshold=50; dcf_hi=60 >= 50 -> fcf_suppressed=False,
    # so the gate never fires regardless of the cash-conversion/investment
    # checks (which would otherwise both pass -- same OCF/NI/CapEx as the
    # first test) -- FCF-DCF stays the (correct) headline, no note at all.
    earnings_power = {"scenarios": {"base": {"per_share": 100.0}}}
    dcf_scenarios = {"base": {"hi": 60.0}}
    normalized = _annual(OperatingCashFlow={2023: 90.0}, NetIncome={2023: 100.0}, CapEx={2023: 50.0})
    metrics = {"latest_fy": 2023}

    unreliable, note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics)
    assert unreliable is False
    assert note is None


def test_fcf_dcf_unreliable_gate_does_not_fire_when_not_investment_driven():
    # Suppressed and cash-backed, but CapEx is small relative to OCF (the
    # suppression can't be attributed to growth investment): ocf=90, ni=100
    # -> cash_backed=True (90>=80); capex=10 -> capex/ocf=10/90=0.111 < 0.5
    # -> investment_driven=False. Gate must not fire -- and since
    # cash_backed IS true here, this also must NOT produce the "not cash
    # backed" quality note (that branch is specifically gated on
    # cash_backed being False).
    earnings_power = {"scenarios": {"base": {"per_share": 100.0}}}
    dcf_scenarios = {"base": {"hi": 40.0}}
    normalized = _annual(OperatingCashFlow={2023: 90.0}, NetIncome={2023: 100.0}, CapEx={2023: 10.0})
    metrics = {"latest_fy": 2023}

    unreliable, note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, normalized, metrics)
    assert unreliable is False
    assert note is None


def test_fcf_dcf_unreliable_returns_false_none_when_epv_base_missing():
    dcf_scenarios = {"base": {"hi": 10.0}}
    normalized = _annual()
    metrics = {"latest_fy": 2023}

    # earnings_power present but its base per_share is None.
    unreliable, note = _fcf_dcf_unreliable(
        dcf_scenarios, {"scenarios": {"base": {"per_share": None}}}, normalized, metrics
    )
    assert (unreliable, note) == (False, None)

    # earnings_power itself is None.
    unreliable2, note2 = _fcf_dcf_unreliable(dcf_scenarios, None, normalized, metrics)
    assert (unreliable2, note2) == (False, None)


def test_fcf_dcf_unreliable_treats_missing_dcf_scenarios_as_suppressed():
    # dcf_scenarios=None (DCF wasn't computable at all) counts as
    # "suppressed" by definition (per spec: "dcf_scenarios is None ... ->
    # fcf_suppressed"); combined with cash-backed + investment-driven, the
    # gate still fires exactly as if a real (low) dcf_hi had been suppressed.
    earnings_power = {"scenarios": {"base": {"per_share": 100.0}}}
    normalized = _annual(OperatingCashFlow={2023: 90.0}, NetIncome={2023: 100.0}, CapEx={2023: 50.0})
    metrics = {"latest_fy": 2023}

    unreliable, note = _fcf_dcf_unreliable(None, earnings_power, normalized, metrics)
    assert unreliable is True
    assert note is None


def test_fcf_dcf_unreliable_missing_ocf_or_ni_is_not_cash_backed():
    # Missing OCF or NI must degrade to "not cash-backed" (None comparisons
    # are never silently treated as passing the guard) -> gate refuses,
    # quality note surfaces.
    earnings_power = {"scenarios": {"base": {"per_share": 100.0}}}
    dcf_scenarios = {"base": {"hi": 40.0}}
    metrics = {"latest_fy": 2023}

    missing_ocf = _annual(NetIncome={2023: 100.0}, CapEx={2023: 50.0})
    unreliable, note = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, missing_ocf, metrics)
    assert unreliable is False
    assert note is not None

    missing_ni = _annual(OperatingCashFlow={2023: 90.0}, CapEx={2023: 50.0})
    unreliable2, note2 = _fcf_dcf_unreliable(dcf_scenarios, earnings_power, missing_ni, metrics)
    assert unreliable2 is False
    assert note2 is not None


# ---------------------------------------------------------------------------
# C. run_valuation integration (SPEC.md Sec.8a/11)
# ---------------------------------------------------------------------------


def test_run_valuation_mature_fcf_suppressed_and_earnings_backed_triggers_epv_headline():
    # Mature, profitable, single-fiscal-year fixture (fy=2023): NI=100,
    # Revenue=1000 (margin 0.10; single year -> the EPV sanity guard never
    # triggers, identical derivation to the _build_earnings_power unit test
    # above):
    #   EPV base_value_per_share = NI/dr/shares = 100/0.10/10 = 100.0
    #   EPV base scenario: per_share=100.0, lo=90.91, hi=111.11 (same grid
    #   derivation as the unit test above).
    #
    # DCF leg deliberately starved: fcf0=1.0 (100x smaller than NI=100),
    # using a lower growth/terminal pair than the DCF's own "case A" (see
    # test_valuation_dcf.py) so the resulting per-share value is even
    # smaller still. The exact 10-year two-stage DCF value isn't hand-
    # derived here (see test_valuation_dcf.py for that derivation
    # methodology) -- what's asserted below is only the fixture
    # precondition that it lands far under 50 (0.5*epv_base=100), which is
    # what the gate itself actually keys off.
    #
    # Cash-conversion guard: OCF=90 >= 0.8*NI(100)=80 -> cash_backed=True.
    # Investment-driven: CapEx=50, capex/ocf=50/90=0.556>=0.5 -> True.
    # -> _fcf_dcf_unreliable's gate fires -> earnings_power_headline=True,
    # and fair_value_range must equal the EPV base band, not the DCF band.
    normalized = _normalized_epv({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 50.0},
    })
    ratios = [{"fy": 2023, "fcf": 1.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 1.0, "net_debt": 0.0}
    assumptions = _epv_assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    # Fixture precondition: DCF really is suppressed relative to EPV.
    assert result["dcf"]["scenarios"] is not None
    assert result["dcf"]["scenarios"]["base"]["hi"] < 50.0

    assert result["earnings_power"] is not None
    assert result["earnings_power_headline"] is True
    ep_base = result["earnings_power"]["scenarios"]["base"]
    assert ep_base["per_share"] == pytest.approx(100.0)
    assert ep_base["lo"] == pytest.approx(90.91)
    assert ep_base["hi"] == pytest.approx(111.11)

    # Headline fair_value_range must reflect the EPV band, not the DCF one.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(90.91)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(111.11)
    assert result["fair_value_range"]["base"]["growth"] == "sıfır büyüme (kazanç gücü çapası)"
    assert result["fair_value_range"]["base"]["discount_rate"] == "%10"

    assert any("kazanç-gücü (EPV) çapasına" in n for n in result["notes"])


def test_run_valuation_mature_fcf_suppressed_but_not_cash_backed_stays_fcf_dcf_headline():
    # CRITICAL regression guard at the integration level: same FCF-
    # suppressed fixture as the EPV-headline test above (fcf0=1.0, EPV
    # base=100.0), but OCF=70 this time -> cash_backed check fails
    # (70 < 0.8*100=80). The gate must refuse to fire even though FCF still
    # looks "suppressed" relative to EPV -- headline stays FCF-DCF, and the
    # earnings-quality warning note must be surfaced (not silently dropped).
    normalized = _normalized_epv({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 70.0}, "CapEx": {2023: 50.0},
    })
    ratios = [{"fy": 2023, "fcf": 1.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 1.0, "net_debt": 0.0}
    assumptions = _epv_assumptions(base_discount=0.10)

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["dcf"]["scenarios"]["base"]["hi"] < 50.0  # still "suppressed" vs EPV base=100
    assert result["earnings_power"] is not None  # still built (mature, not hyper)
    assert result["earnings_power_headline"] is False  # gate refused to fire

    # Headline stays the (raw, suppressed) FCF-DCF band, NOT the EPV band.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(result["dcf"]["scenarios"]["base"]["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(result["dcf"]["scenarios"]["base"]["hi"])

    assert any("kazanç-kalitesi/nakde-çevirme uyarısıdır" in n for n in result["notes"])
    assert not any("kazanç-gücü (EPV) çapasına" in n for n in result["notes"])


def test_run_valuation_mature_healthy_fcf_keeps_fcf_dcf_headline_regression():
    # REGRESSION guard: earnings_power must still be built for a mature,
    # profitable filer (same NI/Revenue single-year fixture as the EPV-
    # headline test above -> EPV base=100.0), but because FCF is genuinely
    # healthy here -- fcf0=100, matching test_valuation_dcf.py's "case A"
    # exactly (growth_5y=0.10, terminal_growth=0.03, discount_rate=0.10,
    # shares=10, net_debt=0 -> base per_share ~=216.7679, hand-derived
    # there) -- the DCF band's high end is nowhere near suppressed relative
    # to EPV (216+ >> 0.5*100=50). The gate must NOT fire, so the headline
    # fair_value_range must stay the raw FCF-DCF band, unchanged from
    # pre-EPV behavior (this is the AAPL-shaped acceptance case from the
    # spec).
    normalized = _normalized_epv({
        "NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0},
        "OperatingCashFlow": {2023: 90.0}, "CapEx": {2023: 20.0},
    })
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0}
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="mature",
    )

    assert result["earnings_power"] is not None
    assert result["earnings_power"]["scenarios"]["base"]["per_share"] == pytest.approx(100.0)
    assert result["earnings_power_headline"] is False

    dcf_base = result["dcf"]["scenarios"]["base"]
    assert dcf_base["per_share"] == pytest.approx(216.7679, rel=1e-3)

    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(dcf_base["lo"])
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(dcf_base["hi"])
    assert result["fair_value_range"]["base"]["growth"] == "%10 büyüme"
    # Must NOT be (or collide with) the EPV band.
    assert result["fair_value_range"]["base"]["lo"] != pytest.approx(90.91)

    assert not any("kazanç-gücü (EPV) çapasına" in n for n in result["notes"])


def test_run_valuation_earnings_power_not_applicable_outside_mature_sector():
    # earnings_power is only ever attempted for sector_type == "mature" (and
    # only when NOT in hyper-grower mode, per SPEC A1) -- financial/reit and
    # growth_unprofitable filers must get earnings_power=None and
    # earnings_power_headline=False even when the underlying NI/shares/
    # discount-rate inputs would otherwise be perfectly sufficient to build
    # one (reuses the same NI/Revenue fixture as the mature tests above).
    normalized = _normalized_epv({"NetIncome": {2023: 100.0}, "Revenue": {2023: 1000.0}})
    ratios = [{"fy": 2023, "fcf": 100.0}]
    metrics = {"shares": 10.0, "latest_fy": 2023, "fcf": 100.0, "net_debt": 0.0}
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result_fin = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )
    assert result_fin["earnings_power"] is None
    assert result_fin["earnings_power_headline"] is False

    result_reit = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )
    assert result_reit["earnings_power"] is None
    assert result_reit["earnings_power_headline"] is False

    result_unprofitable = run_valuation(
        normalized, ratios, metrics, price=None, price_df=None,
        assumptions=assumptions, sector_type="growth_unprofitable",
    )
    assert result_unprofitable["earnings_power"] is None
    assert result_unprofitable["earnings_power_headline"] is False


# ---------------------------------------------------------------------------
# D. triangulate -- earnings_power_headline confidence cap (SPEC.md S5.3)
# ---------------------------------------------------------------------------


def test_triangulate_earnings_power_headline_caps_high_confidence_to_medium():
    # Same three-way "ucuz" agreement as test_valuation_engine.py's
    # test_triangulate_all_three_agree_gives_high_confidence (dcf: price=90
    # < band.lo=100; reverse_dcf: implied=0.05 < ref(0.10)-0.03=0.07;
    # multiples: pe_pct=20 < 30) -- would normally be CONFIDENCE_HIGH,
    # direction "ucuz". With earnings_power_headline=True, confidence must
    # be capped to ORTA (not YÜKSEK): DCF and multiples both ultimately
    # derive from the same underlying earnings signal in EPV-headline mode,
    # so three-way agreement is weaker evidence than usual.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature", earnings_power_headline=True,
    )
    assert result["signals"] == {"dcf": "ucuz", "reverse_dcf": "ucuz", "multiples": "ucuz"}
    assert result["confidence"] == "ORTA"
    assert result["direction"] == "ucuz"
    assert "ORTA ile sınırlandı" in result["rationale"]["confidence"]


def test_triangulate_earnings_power_headline_false_preserves_high_confidence():
    # Regression: earnings_power_headline defaults to False, so the exact
    # same three-way agreement stays CONFIDENCE_HIGH, unchanged from before
    # this feature existed.
    result = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature",
    )
    assert result["confidence"] == "YÜKSEK"
    assert "ORTA ile sınırlandı" not in result["rationale"]["confidence"]


def test_triangulate_earnings_power_headline_does_not_alter_already_medium_or_low_confidence():
    # The cap only ever applies when confidence would otherwise have been
    # HIGH; a 2-of-3 (ORTA) or scattered (DÜŞÜK) result must come out
    # exactly as it would without earnings_power_headline -- no further
    # downgrade, and no rationale suffix appended (the cap sentence is only
    # ever appended on the HIGH-being-capped-to-MEDIUM path).
    two_of_three = triangulate(
        price=90, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.05,
        realized_cagr=None, base_growth=0.10, pe_pct=80, ps_pct=None, pfcf_pct=None,
        sector_type="mature", earnings_power_headline=True,
    )
    assert two_of_three["confidence"] == "ORTA"
    assert "ORTA ile sınırlandı" not in two_of_three["rationale"]["confidence"]

    scattered = triangulate(
        price=110, dcf_base_band={"lo": 100, "hi": 120}, implied_growth=0.15,
        realized_cagr=None, base_growth=0.10, pe_pct=20, ps_pct=None, pfcf_pct=None,
        sector_type="mature", earnings_power_headline=True,
    )
    assert scattered["confidence"] == "DÜŞÜK"
    assert "ORTA ile sınırlandı" not in scattered["rationale"]["confidence"]
