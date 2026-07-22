"""Backtest reporting: hit-rate, calibration time series, divergence cases.

Reads the ``verdicts`` + ``verdict_outcomes`` tables and produces:

* **Hit-rate table** -- isabet oranı by verdict type x horizon x route (method),
  with the sample size ``n``. Cells with ``n < 10`` are flagged
  "yetersiz örneklem".
* **Calibration time series** -- per as-of/run date, the basket's median
  fair-value/price ratio (the regime-independent form of the "0.925 median"
  finding: separates engine conservatism from how expensive the period was).
* **Divergence cases** -- verdicts whose correctness isn't binary
  (``MODEL-PİYASA AYRIŞMASI`` / ``YÜKSEK BEKLENTİ FİYATLANMIŞ``), with their
  realized returns and the manual ``referee_note``.

Every rendered report carries :data:`sec_analyzer.backtest.BACKTEST_DISCLAIMER`.
"""

import html
import json
import logging
import os
import statistics
from collections import defaultdict
from typing import List, Optional

from sec_analyzer.backtest import BACKTEST_DISCLAIMER
from sec_analyzer.backtest.outcomes import is_referee_label
from sec_analyzer.calibrate import _method_slug
from sec_analyzer.config import Config
from sec_analyzer.store.database import load_outcomes, load_verdicts_for_outcomes

logger = logging.getLogger(__name__)

#: Below this many samples a hit-rate cell is not trustworthy.
_MIN_SAMPLE = 10


def _route_of(valuation_json: Optional[str]) -> str:
    """Method/route slug for a verdict from its stored ``valuation_json``."""
    if not valuation_json:
        return "—"
    try:
        valuation = json.loads(valuation_json)
    except (TypeError, ValueError):
        return "—"
    try:
        return _method_slug(valuation) or "—"
    except Exception:  # noqa: BLE001 - route is display-only, never fatal
        return "—"


def _ref_date_of(verdict: dict) -> Optional[str]:
    """The reference (run) date of a verdict: its as-of cutoff, else analysis date."""
    return verdict.get("as_of") or (str(verdict.get("analyzed_at"))[:10] if verdict.get("analyzed_at") else None)


def build_report_data(db_path: Optional[str] = None) -> dict:
    """Assemble the backtest report payload from the stored tables.

    Returns a dict with ``hit_rate`` (list of row dicts), ``calibration``
    (list of ``{date, median_ratio, n}``), ``divergence`` (list of case dicts),
    and ``disclaimer``.
    """
    outcomes = load_outcomes(db_path)
    verdicts = load_verdicts_for_outcomes(db_path)

    # --- Hit-rate: group binary-hit outcomes by (verdict_type, horizon, route) ---
    buckets = defaultdict(lambda: {"hits": 0, "n": 0})
    for o in outcomes:
        if o.get("hit") is None:
            continue  # neutral / referee verdicts carry no binary hit
        key = (o.get("fundamental_verdict") or "—", o.get("horizon") or "—", _route_of(o.get("valuation_json")))
        buckets[key]["n"] += 1
        buckets[key]["hits"] += 1 if o.get("hit") else 0

    hit_rate = []
    for (verdict_type, horizon, route), agg in sorted(buckets.items()):
        n = agg["n"]
        hit_rate.append({
            "verdict_type": verdict_type,
            "horizon": horizon,
            "route": route,
            "n": n,
            "hits": agg["hits"],
            "rate": (agg["hits"] / n) if n else None,
            "insufficient": n < _MIN_SAMPLE,
        })

    # --- Momentum hit-rate: does the momentum label sharpen the value verdict? ---
    # Groups binary-hit outcomes by (fundamental_verdict, momentum_verdict,
    # horizon). Answers e.g. "is UCUZ + NEGATİF momentum really a falling knife
    # (worse hit-rate), and UCUZ + POZİTİF the strongest combination?".
    mom_buckets = defaultdict(lambda: {"hits": 0, "n": 0})
    for o in outcomes:
        if o.get("hit") is None:
            continue
        key = (
            o.get("fundamental_verdict") or "—",
            o.get("momentum_verdict") or "—",
            o.get("horizon") or "—",
        )
        mom_buckets[key]["n"] += 1
        mom_buckets[key]["hits"] += 1 if o.get("hit") else 0
    hit_rate_momentum = []
    for (verdict_type, momentum_verdict, horizon), agg in sorted(mom_buckets.items()):
        n = agg["n"]
        hit_rate_momentum.append({
            "verdict_type": verdict_type,
            "momentum_verdict": momentum_verdict,
            "horizon": horizon,
            "n": n,
            "hits": agg["hits"],
            "rate": (agg["hits"] / n) if n else None,
            "insufficient": n < _MIN_SAMPLE,
        })

    # --- Calibration time series: median FV_base_mid / ref_price per run date ---
    by_date = defaultdict(list)
    for v in verdicts:
        ref_date = _ref_date_of(v)
        lo, hi, price = v.get("fv_base_lo"), v.get("fv_base_hi"), v.get("price")
        if ref_date is None or lo is None or hi is None or not price:
            continue
        ratio = ((lo + hi) / 2.0) / price
        by_date[ref_date].append(ratio)
    calibration = [
        {"date": d, "median_ratio": statistics.median(ratios), "n": len(ratios)}
        for d, ratios in sorted(by_date.items())
    ]

    # --- Per-ticker verdict-momentum: FV_base_mid/price trajectory across runs ---
    # The per-ticker counterpart of the cross-sectional calibration series: does
    # the model find a given name progressively cheaper (ratio rising) or is fair
    # value eroding toward price (ratio falling) as more data arrives?
    by_ticker = defaultdict(list)
    for v in verdicts:
        ref_date = _ref_date_of(v)
        lo, hi, price = v.get("fv_base_lo"), v.get("fv_base_hi"), v.get("price")
        ticker = v.get("ticker")
        if not ticker or ref_date is None or lo is None or hi is None or not price:
            continue
        by_ticker[ticker].append({"date": ref_date, "ratio": round(((lo + hi) / 2.0) / price, 3)})
    verdict_momentum = []
    for ticker, pts in sorted(by_ticker.items()):
        pts = sorted(pts, key=lambda p: p["date"])
        if len(pts) < 2:
            continue
        first, last = pts[0]["ratio"], pts[-1]["ratio"]
        change = (last / first - 1.0) if first else None
        verdict_momentum.append({
            "ticker": ticker,
            "n": len(pts),
            "first_date": pts[0]["date"],
            "last_date": pts[-1]["date"],
            "first_ratio": first,
            "last_ratio": last,
            "change": change,
        })

    # --- Divergence / referee cases ---
    divergence = []
    outcomes_by_vid = defaultdict(list)
    for o in outcomes:
        outcomes_by_vid[o.get("verdict_id")].append(o)
    for v in verdicts:
        if not is_referee_label(v.get("fundamental_verdict")):
            continue
        vid = v.get("id")
        rows = outcomes_by_vid.get(vid) or [{}]
        for o in rows:
            divergence.append({
                "ticker": v.get("ticker"),
                "ref_date": _ref_date_of(v),
                "fundamental_verdict": v.get("fundamental_verdict"),
                "horizon": o.get("horizon"),
                "rel_return": o.get("rel_return"),
                "referee_note": o.get("referee_note"),
            })

    return {
        "hit_rate": hit_rate,
        "hit_rate_momentum": hit_rate_momentum,
        "calibration": calibration,
        "verdict_momentum": verdict_momentum,
        "divergence": divergence,
        "disclaimer": BACKTEST_DISCLAIMER,
    }


def _fmt_pct(value: Optional[float]) -> str:
    return "—" if value is None else f"%{value * 100:.1f}"


def _fmt_ratio(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_terminal(data: dict) -> str:
    """Render the backtest report as plain text for the terminal."""
    lines: List[str] = []
    lines.append("=== Backtest raporu ===")

    lines.append("\n[Hit-rate] verdict türü × vade × route")
    lines.append(f"{'Tür':<24}{'Vade':<6}{'Route':<16}{'n':>5}{'İsabet':>9}")
    lines.append("-" * 60)
    if not data["hit_rate"]:
        lines.append("(değerlendirilmiş ikili-iddia verdict'i yok)")
    for row in data["hit_rate"]:
        flag = "  ⚠ yetersiz örneklem" if row["insufficient"] else ""
        lines.append(
            f"{row['verdict_type']:<24}{row['horizon']:<6}{row['route']:<16}"
            f"{row['n']:>5}{_fmt_pct(row['rate']):>9}{flag}"
        )

    lines.append("\n[Momentum hit-rate] verdict × momentum × vade")
    lines.append(f"{'Tür':<20}{'Momentum':<12}{'Vade':<6}{'n':>5}{'İsabet':>9}")
    lines.append("-" * 60)
    mom_rows = data.get("hit_rate_momentum") or []
    if not mom_rows:
        lines.append("(momentum etiketli değerlendirilmiş verdict yok)")
    for row in mom_rows:
        flag = "  ⚠ yetersiz örneklem" if row["insufficient"] else ""
        lines.append(
            f"{row['verdict_type']:<20}{row['momentum_verdict']:<12}{row['horizon']:<6}"
            f"{row['n']:>5}{_fmt_pct(row['rate']):>9}{flag}"
        )

    lines.append("\n[Kalibrasyon] tarih başına medyan makul-değer/fiyat")
    lines.append(f"{'Tarih':<14}{'Medyan FV/Fiyat':>16}{'n':>5}")
    lines.append("-" * 36)
    if not data["calibration"]:
        lines.append("(veri yok)")
    for row in data["calibration"]:
        lines.append(f"{row['date']:<14}{_fmt_ratio(row['median_ratio']):>16}{row['n']:>5}")

    lines.append("\n[Verdict momentum] hisse başına FV/fiyat oranının seyri")
    lines.append(f"{'Hisse':<8}{'n':>4}{'İlk':>9}{'Son':>9}{'Değişim':>10}")
    lines.append("-" * 40)
    vm_rows = data.get("verdict_momentum") or []
    if not vm_rows:
        lines.append("(≥2 çalıştırması olan hisse yok)")
    for row in vm_rows:
        lines.append(
            f"{row['ticker']:<8}{row['n']:>4}{_fmt_ratio(row['first_ratio']):>9}"
            f"{_fmt_ratio(row['last_ratio']):>9}{_fmt_pct(row['change']):>10}"
        )

    lines.append("\n[Ayrışma vakaları] (MODEL-PİYASA AYRIŞMASI / YÜKSEK BEKLENTİ)")
    if not data["divergence"]:
        lines.append("(ayrışma verdict'i yok)")
    for row in data["divergence"]:
        note = row.get("referee_note") or "—"
        lines.append(
            f"  {row['ticker']} @ {row['ref_date']} [{row['horizon'] or '—'}] "
            f"{row['fundamental_verdict']} · rel {_fmt_pct(row['rel_return'])} · hakem: {note}"
        )

    lines.append(f"\n{data['disclaimer']}")
    return "\n".join(lines)


def _esc(value) -> str:
    return html.escape("—" if value is None else str(value))


def render_html(data: dict, generated_on: str) -> str:
    """Render the backtest report as a self-contained dark-themed HTML page."""
    def hit_rows() -> str:
        if not data["hit_rate"]:
            return '<tr><td colspan="5" class="empty">Değerlendirilmiş ikili-iddia verdict\'i yok.</td></tr>'
        out = []
        for r in data["hit_rate"]:
            cls = ' class="insufficient"' if r["insufficient"] else ""
            flag = ' <span class="warn">yetersiz örneklem</span>' if r["insufficient"] else ""
            out.append(
                f"<tr{cls}><td>{_esc(r['verdict_type'])}</td><td>{_esc(r['horizon'])}</td>"
                f"<td>{_esc(r['route'])}</td><td class='num'>{r['n']}{flag}</td>"
                f"<td class='num'>{_esc(_fmt_pct(r['rate']))}</td></tr>"
            )
        return "".join(out)

    def mom_rows() -> str:
        rows = data.get("hit_rate_momentum") or []
        if not rows:
            return '<tr><td colspan="5" class="empty">Momentum etiketli değerlendirilmiş verdict yok.</td></tr>'
        out = []
        for r in rows:
            cls = ' class="insufficient"' if r["insufficient"] else ""
            flag = ' <span class="warn">yetersiz örneklem</span>' if r["insufficient"] else ""
            out.append(
                f"<tr{cls}><td>{_esc(r['verdict_type'])}</td><td>{_esc(r['momentum_verdict'])}</td>"
                f"<td>{_esc(r['horizon'])}</td><td class='num'>{r['n']}{flag}</td>"
                f"<td class='num'>{_esc(_fmt_pct(r['rate']))}</td></tr>"
            )
        return "".join(out)

    def vm_rows() -> str:
        rows = data.get("verdict_momentum") or []
        if not rows:
            return '<tr><td colspan="5" class="empty">≥2 çalıştırması olan hisse yok.</td></tr>'
        return "".join(
            f"<tr><td>{_esc(r['ticker'])}</td><td class='num'>{r['n']}</td>"
            f"<td class='num'>{_esc(_fmt_ratio(r['first_ratio']))}</td>"
            f"<td class='num'>{_esc(_fmt_ratio(r['last_ratio']))}</td>"
            f"<td class='num'>{_esc(_fmt_pct(r['change']))}</td></tr>"
            for r in rows
        )

    def calib_rows() -> str:
        if not data["calibration"]:
            return '<tr><td colspan="3" class="empty">Veri yok.</td></tr>'
        return "".join(
            f"<tr><td>{_esc(r['date'])}</td><td class='num'>{_esc(_fmt_ratio(r['median_ratio']))}</td>"
            f"<td class='num'>{r['n']}</td></tr>"
            for r in data["calibration"]
        )

    def div_rows() -> str:
        if not data["divergence"]:
            return '<tr><td colspan="5" class="empty">Ayrışma verdict\'i yok.</td></tr>'
        return "".join(
            f"<tr><td>{_esc(r['ticker'])}</td><td>{_esc(r['ref_date'])}</td>"
            f"<td>{_esc(r['fundamental_verdict'])}</td>"
            f"<td class='num'>{_esc(_fmt_pct(r['rel_return']))}</td>"
            f"<td>{_esc(r.get('referee_note'))}</td></tr>"
            for r in data["divergence"]
        )

    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Backtest Raporu · {_esc(generated_on)}</title>
<style>
  body {{ margin:0; background:#0d1420; color:#e7ecf5;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  .page {{ max-width:960px; margin:0 auto; padding:32px 18px; }}
  h1 {{ font-size:1.15rem; margin:0 0 4px; }}
  h2 {{ font-size:0.95rem; color:#9fb3c8; margin:28px 0 10px; }}
  .sub {{ color:#6b7f96; font-size:0.8rem; margin-bottom:18px; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #223349; }}
  th {{ color:#9fb3c8; font-weight:600; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr.insufficient {{ opacity:0.6; }}
  .warn {{ color:#f0b429; font-size:0.72rem; margin-left:6px; }}
  .empty {{ color:#6b7f96; font-style:italic; }}
  .disclaimer {{ margin-top:30px; padding:12px 14px; background:#111b2b;
    border:1px solid #223349; border-radius:10px; color:#9fb3c8; font-size:0.8rem; }}
</style></head><body><div class="page">
  <h1>Backtest Raporu</h1>
  <div class="sub">Oluşturulma: {_esc(generated_on)}</div>

  <h2>Hit-rate — verdict türü × vade × route</h2>
  <table><thead><tr><th>Tür</th><th>Vade</th><th>Route</th><th class="num">n</th>
    <th class="num">İsabet</th></tr></thead><tbody>{hit_rows()}</tbody></table>

  <h2>Momentum hit-rate — verdict × momentum × vade</h2>
  <table><thead><tr><th>Tür</th><th>Momentum</th><th>Vade</th><th class="num">n</th>
    <th class="num">İsabet</th></tr></thead><tbody>{mom_rows()}</tbody></table>

  <h2>Kalibrasyon — tarih başına medyan makul-değer/fiyat</h2>
  <table><thead><tr><th>Tarih</th><th class="num">Medyan FV/Fiyat</th>
    <th class="num">n</th></tr></thead><tbody>{calib_rows()}</tbody></table>

  <h2>Verdict momentum — hisse başına FV/fiyat oranının seyri</h2>
  <table><thead><tr><th>Hisse</th><th class="num">n</th><th class="num">İlk oran</th>
    <th class="num">Son oran</th><th class="num">Değişim</th></tr></thead>
    <tbody>{vm_rows()}</tbody></table>

  <h2>Ayrışma vakaları</h2>
  <table><thead><tr><th>Hisse</th><th>Tarih</th><th>Verdict</th>
    <th class="num">Rel. getiri</th><th>Hakem notu</th></tr></thead>
    <tbody>{div_rows()}</tbody></table>

  <div class="disclaimer">{_esc(data['disclaimer'])}</div>
</div></body></html>"""


def write_html_report(data: dict, generated_on: str, out_dir: Optional[str] = None) -> str:
    """Render and write the HTML backtest report; return the path."""
    target_dir = out_dir or Config.REPORTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"backtest_{generated_on}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(data, generated_on))
    return path
