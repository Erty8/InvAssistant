"""Hand-verified numeric tests for the residual income model (RIM) anchor
(SPEC.md Sec.8f): ``valuation.dcf.rim_per_share`` and ``valuation.engine``'s
``_build_rim``/``_rim_scenario_band`` and their ``run_valuation`` wiring
(RIM as financial's PRIMARY anchor, with a fallback to the pre-existing
``_build_pb_roe`` when RIM can't be built).

Every numeric expectation below was cross-checked with an independent
from-scratch scratch script (reimplementing the documented formulas -- year-
by-year growth via ``_year_growth_rate``, the ``g_eff = min(g, roe)``
reinvestment cap, clean-surplus book-value roll-forward, Gordon-growth
terminal value -- NOT calling ``dcf.py``) before finalizing, following
``test_valuation_cyclical_fcfe.py``'s own methodology for this style of
10-year, two-stage projection.
"""

import pytest

from sec_analyzer.valuation.dcf import rim_per_share
from sec_analyzer.valuation.engine import _build_rim, run_valuation

# ---------------------------------------------------------------------------
# 1. rim_per_share (pure function, SPEC.md Sec.8f)
# ---------------------------------------------------------------------------


def test_rim_per_share_normal_case_growth_below_roe_hand_verified():
    # bve0=1000, ni0=120, roe=0.12, growth_5y=0.08, terminal_growth=0.03,
    # r=0.09, shares=50, terminal_roe=0.09 (scenario's own cost of equity,
    # the convention _build_rim always uses).
    #
    # growth_5y(0.08) < roe(0.12) for the ENTIRE horizon (years 1-10 fade
    # from 0.08 down to 0.03, always < 0.12) -> g_eff=min(g,roe) never binds;
    # g_eff == the raw _year_growth_rate value every year.
    #
    # Years 1-5 (g=0.08 flat): b = 0.08/0.12 = 0.666667 every year.
    #   ni1 = 120*1.08     = 129.6        retained = 129.6*0.666667=86.4      bve1=1086.4   ri1=129.6-0.09*1000=39.6
    #   ni2 = 129.6*1.08   = 139.968      retained = 93.312                  bve2=1179.712 ri2=139.968-0.09*1086.4=42.192
    #   ni3 = 139.968*1.08 = 151.16544    retained = 100.77696               bve3=1280.48896 ri3=151.16544-0.09*1179.712=44.99136
    #   ni4 = 151.16544*1.08=163.258675   retained=108.839117                bve4=1389.328077 ri4=48.014669
    #   ni5 = 163.258675*1.08=176.319369  retained=117.546246                bve5=1506.874323 ri5=51.279842
    # Years 6-10 fade g linearly from 0.08 to 0.03 (g_y = 0.08-0.05*(y-5)/5):
    #   y6 g=0.07  ni6=176.319369*1.07=188.661725 b=0.07/0.12=0.583333 retained=110.052673 bve6=1616.926996 ri6=53.043036
    #   y7 g=0.06  ni7=188.661725*1.06=199.981429 b=0.5               retained=99.990714  bve7=1716.91771  ri7=54.457999
    #   y8 g=0.05  ni8=199.981429*1.05=209.9805   b=0.416667          retained=87.491875  bve8=1804.409585 ri8=55.457906
    #   y9 g=0.04  ni9=209.9805*1.04=218.37972    b=0.333333          retained=72.79324   bve9=1877.202825 ri9=55.982857
    #   y10 g=0.03 ni10=218.37972*1.03=224.931112 b=0.25              retained=56.232778  bve10=1933.435603 ri10=55.982857
    # (g at y10 == terminal_growth, as the two-stage fade formula requires.)
    #
    # pv_sum (discount each ri_y at 1.09^y, y=1..10) = 312.601703
    #
    # Terminal (F1 genuine fade: terminal_roe=0.09 == discount_rate):
    #   The terminal-year net income is what the ending book earns at the
    #   FADED terminal ROE, so terminal residual income is
    #   ri_terminal = (terminal_roe - discount_rate) * bve10
    #               = (0.09 - 0.09) * 1933.435603 = 0.0
    #   -> tv = 0 / (0.09 - 0.03) = 0.0, pv_tv = 0.0. The bank's year-10
    #   excess return (ROE 12% > r 9%) is assumed to compete away in steady
    #   state, so the terminal adds nothing beyond book value (standard RIM
    #   "excess returns fade to zero" terminal).
    #
    # equity = bve0 + pv_sum + pv_tv = 1000 + 312.601703 + 0 = 1312.601703
    # effective_shares = 50*(1+0)^5 = 50
    # per_share = 1312.601703/50 = 26.252034
    result = rim_per_share(
        bve0=1000.0, ni0=120.0, roe=0.12, growth_5y=0.08, terminal_growth=0.03,
        discount_rate=0.09, shares=50.0, terminal_roe=0.09,
    )

    assert len(result["ri_path"]) == 10
    assert len(result["bve_path"]) == 10
    assert result["ri_path"][0] == pytest.approx(39.6, rel=1e-5)
    assert result["bve_path"][0] == pytest.approx(1086.4, rel=1e-6)
    assert result["ri_path"][9] == pytest.approx(55.982857, rel=1e-5)
    assert result["bve_path"][9] == pytest.approx(1933.435603, rel=1e-6)

    # F1: terminal residual income (and TV) are exactly zero when terminal_roe
    # is faded to the cost of equity.
    assert result["tv"] == pytest.approx(0.0, abs=1e-9)
    assert result["equity"] == pytest.approx(1312.601703, rel=1e-6)
    assert result["bve0"] == 1000.0
    assert result["effective_shares"] == pytest.approx(50.0)
    assert result["per_share"] == pytest.approx(26.252034, rel=1e-6)


def test_rim_per_share_growth_destroys_value_when_roe_below_discount_rate():
    # bve0=500, ni0=25 (roe=0.05, consistent), growth_5y=0.08 (EXCEEDS roe),
    # terminal_growth=0.02, r=0.10, shares=20, terminal_roe=None (falls back
    # to roe=0.05). roe(0.05) < discount_rate(0.10) -- the textbook "growth
    # destroys value" case: retaining a dollar earns less than investors
    # require, so every year's residual income is NEGATIVE, and even the
    # terminal value comes out negative.
    #
    # g_eff = min(g_year, 0.05): years 1-5 (g=0.08) -> capped to 0.05 (b=1.0,
    # full retention). Year 6 (g=0.08-0.06*(1)/5=0.068) -> capped to 0.05.
    # Year 7 (g=0.056) -> capped to 0.05. Year 8 (g=0.044) -> UNCAPPED
    # (0.044 < 0.05). Year 9 (g=0.032), year 10 (g=0.02) -> both uncapped.
    #
    # Year 1: ni1=25*1.05=26.25, b=0.05/0.05=1.0, retained=26.25,
    #   bve1=500+26.25=526.25, ri1=26.25 - 0.10*500 = 26.25-50 = -23.75.
    #
    # terminal_roe=None -> falls back to roe=0.05 (a permanent, NON-faded
    # terminal for this direct-call case, since no caller-supplied fade):
    #   ri_terminal = (0.05 - 0.10) * bve10 = -0.05 * 785.765761 = -39.288288
    #   tv = -39.288288 / (0.10 - 0.02) = -491.103601 (negative -- a firm
    #   earning below its cost of equity forever DESTROYS value in perpetuity).
    #
    # equity = bve0 + pv_sum + pv_tv = 500 + (negative) + (negative)
    #   = 130.390128 (still POSITIVE overall -- book value dominates -- but
    #   per_share = 130.390128/20 = 6.519506, far BELOW the no-growth
    #   book-value-per-share baseline of 500/20 = 25.0: attempting this growth
    #   genuinely destroys value, exactly as the roe < discount rate case
    #   should).
    result = rim_per_share(
        bve0=500.0, ni0=25.0, roe=0.05, growth_5y=0.08, terminal_growth=0.02,
        discount_rate=0.10, shares=20.0, terminal_roe=None,
    )

    assert result["ri_path"][0] == pytest.approx(-23.75, rel=1e-6)
    assert result["bve_path"][0] == pytest.approx(526.25, rel=1e-6)
    # Every year's residual income is negative (roe < discount_rate).
    assert all(ri < 0 for ri in result["ri_path"])
    assert result["tv"] < 0
    assert result["equity"] == pytest.approx(130.390128, rel=1e-5)
    book_value_per_share = 500.0 / 20.0
    assert result["per_share"] == pytest.approx(6.519506, rel=1e-5)
    assert result["per_share"] < book_value_per_share  # growth destroys value here


def test_rim_terminal_roe_fade_actually_bites_regression_guard():
    # Regression guard for F1: terminal_roe MUST change the result (an
    # earlier construction left it inert via min(terminal_growth,
    # terminal_roe), which always collapsed to terminal_growth). Same base
    # inputs, three terminal_roe values that straddle/exceed the cost of
    # equity -- the per-share values must be DISTINCT and monotonically
    # increasing in terminal_roe (a higher perpetual ROE spread is worth
    # more).
    kw = dict(bve0=100.0, ni0=15.0, roe=0.15, growth_5y=0.08, terminal_growth=0.03,
              discount_rate=0.10, shares=1.0)
    faded = rim_per_share(**kw, terminal_roe=0.10)      # == cost of equity -> terminal RI 0
    mild = rim_per_share(**kw, terminal_roe=0.13)       # modest durable spread
    strong = rim_per_share(**kw, terminal_roe=0.20)     # large durable spread

    # Faded terminal contributes exactly zero (excess return competes away).
    assert faded["tv"] == pytest.approx(0.0, abs=1e-9)
    # A durable positive spread produces a positive, larger terminal.
    assert mild["tv"] > 0
    assert strong["tv"] > mild["tv"]
    # Monotone and strictly distinct -- terminal_roe genuinely bites.
    assert faded["per_share"] < mild["per_share"] < strong["per_share"]


def test_rim_per_share_raises_on_missing_ni0():
    with pytest.raises(ValueError):
        rim_per_share(1000.0, None, 0.12, 0.08, 0.03, 0.09, 50.0)


def test_rim_per_share_raises_on_non_positive_bve0():
    with pytest.raises(ValueError):
        rim_per_share(0.0, 120.0, 0.12, 0.08, 0.03, 0.09, 50.0)
    with pytest.raises(ValueError):
        rim_per_share(-100.0, 120.0, 0.12, 0.08, 0.03, 0.09, 50.0)


def test_rim_per_share_raises_on_bad_shares():
    with pytest.raises(ValueError):
        rim_per_share(1000.0, 120.0, 0.12, 0.08, 0.03, 0.09, 0.0)


def test_rim_per_share_raises_on_non_positive_roe():
    with pytest.raises(ValueError):
        rim_per_share(1000.0, 120.0, 0.0, 0.08, 0.03, 0.09, 50.0)


def test_rim_per_share_raises_on_bad_rate_relationship():
    with pytest.raises(ValueError):
        rim_per_share(1000.0, 120.0, 0.12, 0.08, terminal_growth=0.09, discount_rate=0.05, shares=50.0)


# ---------------------------------------------------------------------------
# 2. _build_rim (engine.py, SPEC.md Sec.8f)
# ---------------------------------------------------------------------------

_RIM_CONCEPTS = [
    "Revenue", "NetIncome", "OperatingCashFlow", "CapEx", "Cash",
    "LongTermDebt", "LongTermDebtCurrent", "SharesOutstanding", "EPS",
    "SBC", "StockholdersEquity",
]


def _rec(fy, value):
    return {
        "concept": None, "tag": None, "period_end": f"{fy}-12-31",
        "fy": fy, "fp": "FY", "form": "10-K", "value": value,
        "filed": None, "start": None, "unit": "USD",
    }


def _normalized(overrides: "dict[str, dict[int, float]]") -> dict:
    annual = {
        concept: [_rec(fy, value) for fy, value in (overrides.get(concept) or {}).items()] or None
        for concept in _RIM_CONCEPTS
    }
    return {
        "cik": 1, "entity_name": "RIM Test Co", "currency": "USD",
        "annual": annual, "quarterly": {c: None for c in _RIM_CONCEPTS},
        "missing": [c for c in _RIM_CONCEPTS if annual[c] is None],
        "matched_tags": {c: None for c in _RIM_CONCEPTS},
    }


def _rim_assumptions():
    """bear/base/bull assumptions for the ``_build_rim`` scenario-band test
    below (bve0=1000, ni0=120, roe=0.12, shares=50 -- same base-scenario
    inputs as ``test_rim_per_share_normal_case_growth_below_roe_hand_verified``,
    per_share=26.25 there)."""
    return {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.11, "story": "Ayı."},
        "base": {"growth_5y": 0.08, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Baz."},
        "bull": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.075, "story": "Boğa."},
    }


def test_build_rim_scenario_band_hand_verified():
    # bve0=1000 (StockholdersEquity FY2023), ni0=120 (NetIncome FY2023),
    # roe=0.12 (ratios FY2023), shares=50.
    #
    # Per-scenario point estimates (rim_per_share with terminal_roe=that
    # scenario's own discount_rate -> F1 genuine fade -> terminal RI 0),
    # rounded to 2dp, cross-checked via scratch script:
    #   bear (g=0.05, gt=0.02, r=0.11): per_share = 22.01
    #   base (g=0.08, gt=0.03, r=0.09): per_share = 26.25 (matches the pure-
    #     function test above)
    #   bull (g=0.10, gt=0.03, r=0.075): per_share = 30.50
    #
    # Bands from _rim_scenario_band (recompute at dr +/- 0.01, terminal_roe
    # = that same nearby rate -> terminal RI stays 0 at each band point,
    # growth_5y/terminal_growth held fixed):
    #   bear: dr in (0.10, 0.11, 0.12) -> per_share cells [23.58, 22.01, 20.57]
    #     -> lo=20.57, hi=23.58
    #   base: dr in (0.08, 0.09, 0.10) -> cells [28.41, 26.25, 24.29]
    #     -> lo=24.29, hi=28.41
    #   bull: dr in (0.065, 0.075, 0.085) -> cells [33.23, 30.50, 28.02]
    #     -> lo=28.02, hi=33.23
    normalized = _normalized({
        "StockholdersEquity": {2023: 1000.0},
        "NetIncome": {2023: 120.0},
    })
    ratios = [{"fy": 2023, "roe": 0.12}]
    metrics = {"shares": 50.0, "latest_fy": 2023}
    assumptions = _rim_assumptions()

    result, notes = _build_rim(assumptions, normalized, metrics, ratios)

    assert result is not None
    assert result["bve0"] == 1000.0
    assert result["book_value_per_share"] == pytest.approx(20.0)
    assert result["normalized_net_income"] == 120.0
    assert result["roe"] == pytest.approx(0.12)

    bear = result["scenarios"]["bear"]
    assert bear["per_share"] == pytest.approx(22.01)
    assert bear["lo"] == pytest.approx(20.57)
    assert bear["hi"] == pytest.approx(23.58)

    base = result["scenarios"]["base"]
    assert base["per_share"] == pytest.approx(26.25)
    assert base["lo"] == pytest.approx(24.29)
    assert base["hi"] == pytest.approx(28.41)

    bull = result["scenarios"]["bull"]
    assert bull["per_share"] == pytest.approx(30.50)
    assert bull["lo"] == pytest.approx(28.02)
    assert bull["hi"] == pytest.approx(33.23)

    assert result["per_share"] == base["per_share"]


def test_build_rim_uses_latest_fy_with_equity_ni_and_roe_all_present():
    # JPM-style edge case (mirrors _build_pb_roe's own FY-selection test):
    # metrics["latest_fy"]=2024, but 2024 has no StockholdersEquity/NetIncome
    # at all -- _build_rim must walk down to 2023, the newest year with ALL
    # THREE of equity/NI/roe present, rather than reporting unavailable.
    normalized = _normalized({
        "StockholdersEquity": {2023: 1000.0},
        "NetIncome": {2023: 120.0},
    })
    ratios = [{"fy": 2023, "roe": 0.12}]
    metrics = {"shares": 50.0, "latest_fy": 2024}
    assumptions = _rim_assumptions()

    result, notes = _build_rim(assumptions, normalized, metrics, ratios)

    assert result is not None
    assert result["bve0"] == 1000.0
    assert any("2023" in n and "kullanıldı" in n for n in notes)


def test_build_rim_unavailable_when_net_income_missing():
    # Equity + ROE present, but NetIncome missing for every fiscal year ->
    # no fiscal year clears the "all three present" bar -> anchor unavailable.
    normalized = _normalized({"StockholdersEquity": {2023: 1000.0}})
    ratios = [{"fy": 2023, "roe": 0.12}]
    metrics = {"shares": 50.0, "latest_fy": 2023}
    assumptions = _rim_assumptions()

    result, notes = _build_rim(assumptions, normalized, metrics, ratios)

    assert result is None
    assert any("özkaynak, net kâr veya ROE verisi eksik" in n for n in notes)


def test_build_rim_unavailable_when_shares_missing():
    normalized = _normalized({
        "StockholdersEquity": {2023: 1000.0}, "NetIncome": {2023: 120.0},
    })
    ratios = [{"fy": 2023, "roe": 0.12}]
    metrics = {"shares": None, "latest_fy": 2023}
    assumptions = _rim_assumptions()

    result, notes = _build_rim(assumptions, normalized, metrics, ratios)

    assert result is None
    assert any("hisse sayısı" in n for n in notes)


def test_build_rim_unavailable_when_roe_non_positive():
    normalized = _normalized({
        "StockholdersEquity": {2023: 1000.0}, "NetIncome": {2023: -50.0},
    })
    ratios = [{"fy": 2023, "roe": -0.05}]
    metrics = {"shares": 50.0, "latest_fy": 2023}
    assumptions = _rim_assumptions()

    result, notes = _build_rim(assumptions, normalized, metrics, ratios)

    assert result is None
    assert any("pozitif değil" in n for n in notes)


# ---------------------------------------------------------------------------
# 3. run_valuation wiring: RIM primary, pb_roe fallback (SPEC.md Sec.8f)
# ---------------------------------------------------------------------------


def test_run_valuation_financial_sector_headlines_rim_when_available():
    normalized = _normalized({
        "StockholdersEquity": {2023: 1000.0}, "NetIncome": {2023: 120.0},
    })
    ratios = [{"fy": 2023, "roe": 0.12}]
    metrics = {"shares": 50.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _rim_assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=30.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    assert result["dcf"]["enabled"] is False
    assert result["rim"] is not None
    assert result["rim"]["scenarios"]["base"]["per_share"] == pytest.approx(26.25)
    assert result["pb_roe"] is None

    # fair_value_range must be built FROM the rim scenarios when DCF is
    # disabled and RIM is available.
    assert result["fair_value_range"]["base"]["lo"] == pytest.approx(24.29)
    assert result["fair_value_range"]["base"]["hi"] == pytest.approx(28.41)


def test_run_valuation_financial_sector_falls_back_to_pb_roe_when_rim_unavailable():
    # No NetIncome at all -> _build_rim returns None -> engine falls back to
    # _build_pb_roe (the pre-existing anchor), exactly as it did before RIM
    # existed. StockholdersEquity/roe alone are enough for pb_roe.
    normalized = _normalized({"StockholdersEquity": {2023: 1000.0}})
    ratios = [{"fy": 2023, "roe": 0.15}]
    metrics = {"shares": 100.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = {
        "bear": {"growth_5y": 0.05, "terminal_growth": 0.02, "discount_rate": 0.12, "story": "Ayı."},
        "base": {"growth_5y": 0.10, "terminal_growth": 0.03, "discount_rate": 0.10, "story": "Baz."},
        "bull": {"growth_5y": 0.15, "terminal_growth": 0.03, "discount_rate": 0.09, "story": "Boğa."},
    }

    result = run_valuation(
        normalized, ratios, metrics, price=15.0, price_df=None,
        assumptions=assumptions, sector_type="financial",
    )

    assert result["dcf"]["enabled"] is False
    assert result["rim"] is None
    assert result["pb_roe"] is not None
    assert result["pb_roe"]["scenarios"]["base"]["per_share"] is not None
    assert any("RIM" in n and "geri dönüldü" in n for n in result["notes"])

    # fair_value_range falls back to pb_roe's band.
    assert result["fair_value_range"]["base"]["lo"] == result["pb_roe"]["scenarios"]["base"]["lo"]
    assert result["fair_value_range"]["base"]["hi"] == result["pb_roe"]["scenarios"]["base"]["hi"]


def test_run_valuation_reit_sector_unaffected_by_rim():
    # reit stays on its own FFO/pb_roe path -- rim must be None regardless of
    # whether equity/NI/roe data is present, and this fixture (no
    # Depreciation) also can't build FFO, so it falls back to pb_roe exactly
    # as before RIM existed.
    normalized = _normalized({
        "StockholdersEquity": {2023: 1000.0}, "NetIncome": {2023: 120.0},
    })
    ratios = [{"fy": 2023, "roe": 0.12}]
    metrics = {"shares": 50.0, "latest_fy": 2023, "fcf": None, "net_debt": 0.0}
    assumptions = _rim_assumptions()

    result = run_valuation(
        normalized, ratios, metrics, price=30.0, price_df=None,
        assumptions=assumptions, sector_type="reit",
    )

    assert result["rim"] is None
    assert result["ffo"] is None
    assert result["pb_roe"] is not None
