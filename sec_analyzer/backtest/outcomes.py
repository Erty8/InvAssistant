"""Backtest outcome evaluation: did each stored verdict prove right?

For every verdict in the ``verdicts`` table (live or as-of), this measures the
realized forward return at +1y and +3y from the verdict's reference date,
relative to SPY, and records a "hit" flag where the verdict makes a binary
claim:

* ``UCUZ`` (cheap)     -> hit when the SPY-relative return is positive.
* ``PAHALI`` (expensive) -> hit when the SPY-relative return is negative.
* ``MAKUL`` (fair)     -> no hit evaluation (a neutral claim).
* ``YÜKSEK BEKLENTİ FİYATLANMIŞ`` / ``MODEL-PİYASA AYRIŞMASI`` -> not a binary
  hit; the realized return is still recorded and a manual ``referee_note`` field
  is left for the analyst to judge whether the market-priced assumption played
  out.

Results are written to the ``verdict_outcomes`` table (idempotent per
verdict+horizon). Verdicts whose forward window hasn't matured yet are skipped.

This module is deterministic given its price inputs; the only wall-clock use is
deciding which verdicts have matured (``today``), which is inherent to
evaluating the past and is injectable for testing.
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

from sec_analyzer.fetch.prices import PriceDataError, get_price_history
from sec_analyzer.store.database import (
    load_verdicts_for_outcomes,
    save_outcome,
)

logger = logging.getLogger(__name__)

#: Benchmark for relative return (same one the technical layer uses).
_BENCHMARK = "SPY"

#: Horizons evaluated, as (label, years) pairs.
_HORIZONS: Tuple[Tuple[str, int], ...] = (("1y", 1), ("3y", 3))

#: Fundamental-verdict label buckets.
_CHEAP = "UCUZ"
_EXPENSIVE = "PAHALI"
_NEUTRAL = "MAKUL"
#: Verdicts whose correctness is NOT a binary hit -- recorded, refereed manually.
_REFEREE_LABELS = frozenset({"YÜKSEK BEKLENTİ FİYATLANMIŞ", "MODEL-PİYASA AYRIŞMASI"})


def _add_years(d: date, years: int) -> date:
    """Return ``d`` plus ``years`` calendar years (Feb-29 -> Feb-28 safe)."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        # d is Feb 29 and the target year isn't a leap year.
        return d.replace(year=d.year + years, day=28)


def _parse_iso(value: Optional[str]) -> Optional[date]:
    """Parse an ISO date/datetime string's date part, or return ``None``."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _close_on_or_before(df, target: date) -> Optional[float]:
    """Return the last close in ``df`` dated on/before ``target``, or ``None``."""
    if df is None or df.empty:
        return None
    import pandas as pd

    sliced = df.loc[df.index <= pd.Timestamp(target)]
    if sliced.empty:
        return None
    return float(sliced.iloc[-1]["Close"])


def classify_hit(fundamental_verdict: Optional[str], rel_return: Optional[float]) -> Optional[bool]:
    """Map a fundamental verdict + realized SPY-relative return to a hit flag.

    Returns ``True``/``False`` only for the binary-claim verdicts (``UCUZ`` ->
    rel>0, ``PAHALI`` -> rel<0); ``None`` for neutral (``MAKUL``), referee
    (high-expectation / divergence), unknown labels, or a missing return.
    """
    if rel_return is None or not fundamental_verdict:
        return None
    if fundamental_verdict == _CHEAP:
        return rel_return > 0
    if fundamental_verdict == _EXPENSIVE:
        return rel_return < 0
    return None


class _PriceCache:
    """Fetch each ticker's full Stooq history at most once per run."""

    def __init__(self, no_cache: bool = False):
        self._no_cache = no_cache
        self._cache: Dict[str, object] = {}

    def get(self, ticker: str):
        key = str(ticker).strip().upper()
        if key in self._cache:
            return self._cache[key]
        try:
            df, _source = get_price_history(key, no_cache=self._no_cache)
        except PriceDataError as exc:
            logger.warning("outcomes: no price history for %s: %s", key, exc)
            df = None
        except Exception:  # noqa: BLE001 - one bad ticker must not abort the batch
            logger.warning("outcomes: price fetch failed for %s", key, exc_info=True)
            df = None
        self._cache[key] = df
        return df


def evaluate_outcomes(
    db_path: Optional[str] = None,
    no_cache: bool = False,
    today: Optional[date] = None,
    horizons: Tuple[Tuple[str, int], ...] = _HORIZONS,
) -> dict:
    """Evaluate forward outcomes for every stored verdict (idempotent).

    Args:
        db_path: SQLite path. Defaults to ``Config.DB_PATH``.
        no_cache: Bypass the price-history disk cache and re-fetch.
        today: Maturity reference date (defaults to :meth:`date.today`);
            verdicts whose ``horizon`` window ends after this are skipped.
        horizons: ``(label, years)`` pairs to evaluate.

    Returns:
        A summary dict ``{"evaluated": int, "skipped_immature": int,
        "skipped_no_data": int, "verdicts_seen": int}``. Never raises on a
        single verdict/ticker failure -- it logs and moves on.
    """
    today = today or date.today()
    verdicts = load_verdicts_for_outcomes(db_path)
    prices = _PriceCache(no_cache=no_cache)
    spy_df = prices.get(_BENCHMARK)
    evaluated_at = today.isoformat()

    counts = {
        "verdicts_seen": len(verdicts),
        "evaluated": 0,
        "skipped_immature": 0,
        "skipped_no_data": 0,
    }

    for v in verdicts:
        verdict_id = v.get("id")
        ticker = v.get("ticker")
        # Reference date: the as-of cutoff for backtest verdicts, else the
        # (live) analysis date. This is the point the forward window starts.
        ref_date = _parse_iso(v.get("as_of")) or _parse_iso(v.get("analyzed_at"))
        if ref_date is None or verdict_id is None or not ticker:
            counts["skipped_no_data"] += 1
            continue

        ref_price = v.get("price")
        ticker_df = prices.get(ticker)
        if ref_price is None:
            ref_price = _close_on_or_before(ticker_df, ref_date)

        fundamental_verdict = v.get("fundamental_verdict")

        for label, years in horizons:
            fwd_date = _add_years(ref_date, years)
            if fwd_date > today:
                counts["skipped_immature"] += 1
                continue

            fwd_price = _close_on_or_before(ticker_df, fwd_date)
            spy_ref = _close_on_or_before(spy_df, ref_date)
            spy_fwd = _close_on_or_before(spy_df, fwd_date)

            abs_return = None
            rel_return = None
            if ref_price and fwd_price:
                abs_return = fwd_price / ref_price - 1.0
                if spy_ref and spy_fwd:
                    spy_return = spy_fwd / spy_ref - 1.0
                    rel_return = abs_return - spy_return

            if abs_return is None:
                counts["skipped_no_data"] += 1
                # Still record the (ref) row so the report can show a gap? No --
                # a missing forward price means nothing to evaluate; skip it.
                continue

            hit = classify_hit(fundamental_verdict, rel_return)
            try:
                save_outcome(
                    verdict_id=verdict_id,
                    horizon=label,
                    ref_date=ref_date.isoformat(),
                    ref_price=ref_price,
                    fwd_date=fwd_date.isoformat(),
                    fwd_price=fwd_price,
                    abs_return=abs_return,
                    rel_return=rel_return,
                    hit=hit,
                    evaluated_at=evaluated_at,
                    db_path=db_path,
                )
                counts["evaluated"] += 1
            except Exception:  # noqa: BLE001 - persistence failure must not abort the batch
                logger.warning(
                    "outcomes: failed to save outcome for verdict %s (%s)",
                    verdict_id, label, exc_info=True,
                )

    logger.info(
        "outcomes: %d evaluated, %d immature, %d no-data (of %d verdicts).",
        counts["evaluated"], counts["skipped_immature"], counts["skipped_no_data"],
        counts["verdicts_seen"],
    )
    return counts


def is_referee_label(fundamental_verdict: Optional[str]) -> bool:
    """True for verdicts whose correctness is judged manually (referee_note)."""
    return fundamental_verdict in _REFEREE_LABELS
