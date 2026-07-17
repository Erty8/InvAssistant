"""Tests for :func:`sec_analyzer.calibrate._method_slug`'s hyper-growth
precedence (LEVER 4).

Background: ``_method_slug`` mirrors ``cli.py``'s
``_valuation_method_label`` precedence order, but originally never checked
the hyper-growth headline. When a hyper-grower's revenue-first DCF headlines
the fair-value range (``valuation["hyper_growth"] = True`` with a populated,
non-suppressed ``hyper_growth_detail``), ``run_valuation`` still leaves the
standard two-stage DCF enabled and populated as a secondary/comparison
scenario (``valuation["dcf"]["enabled"] = True``), so without an explicit
hyper check ``_method_slug`` would fall through to the ``"dcf"`` branch and
mislabel a hyper-headlined filer (e.g. RDDT) as plain "dcf".

These are plain-dict unit tests -- no need to run the full
``run_valuation``/``interpret`` pipeline to exercise the precedence logic.
"""

from sec_analyzer.calibrate import _method_slug


def test_hyper_headline_takes_precedence_over_dcf_enabled():
    """A hyper-headlined filer (RDDT-shaped) must slug as "hyper", even
    though the standard DCF is still enabled and computed as a secondary
    scenario alongside it."""
    valuation = {
        "hyper_growth": True,
        "hyper_growth_detail": {"suppressed": False, "scenarios": {"base": {"per_share": 50.0}}},
        "dcf": {"enabled": True, "scenarios": {"base": {"per_share": 40.0}}},
    }
    assert _method_slug(valuation) == "hyper"


def test_plain_dcf_valuation_without_hyper_key_still_slugs_dcf():
    """A valuation dict with no ``hyper_growth`` key at all (the common,
    non-hyper case) must fall through unaffected to the "dcf" slug."""
    valuation = {
        "dcf": {"enabled": True, "scenarios": {"base": {"per_share": 40.0}}},
    }
    assert _method_slug(valuation) == "dcf"


def test_suppressed_hyper_detail_does_not_slug_hyper():
    """When hyper-growth detection fired but was suppressed (non-credible
    negative valuation guard -- see test_valuation_hyper_suppression.py),
    the headline fair-value range is emptied and the ticker is skipped
    upstream by the calibration harness; ``_method_slug`` must NOT return
    "hyper" in this case, falling through to whatever the remaining
    precedence chain gives (here, "dcf")."""
    valuation = {
        "hyper_growth": True,
        "hyper_growth_detail": {"suppressed": True, "suppressed_reason": "negatif değer"},
        "dcf": {"enabled": True, "scenarios": {"base": {"per_share": 40.0}}},
    }
    assert _method_slug(valuation) != "hyper"
    assert _method_slug(valuation) == "dcf"
