# SWING_SPEC.md — S&P 500 Swing-Trade Screener (binding contract)

Status: **binding**. Code that contradicts this file is wrong. Same standing as
`sec_analyzer/valuation/SPEC.md` for the valuation engine.

Scope: a deterministic, purely technical, **long-only** swing-setup score for
every S&P 500 constituent, plus the scan orchestration, persistence, HTTP API
and UI tab that expose it.

Non-goals (explicitly out of scope):

- No fundamentals, no valuation, no red flags, no SEC EDGAR access anywhere in
  this feature. Price/volume history is the only input.
- No LLM anywhere. No randomness. No wall-clock dependence in any *score*.
- No new third-party dependencies. `pandas` + `requests` + stdlib only.
- No short/bearish ranking. The score is one-directional (see §3).

---

## 1. Universes

Two bundled static CSVs, both already generated and validated:

| code | label | file | rows |
|---|---|---|---|
| `SP500` | `S&P 500` | `sec_analyzer/data/sp500.csv` | 503 |
| `NDX` | `Nasdaq 100` | `sec_analyzer/data/nasdaq100.csv` | 103 |

Both share the header `ticker,name,sector,cik`. `sector` is a broad sector
label (GICS Sector for SP500, ICB Industry for NDX — comparable granularity,
deliberately not harmonized further since it is display-only). `cik` is a
zero-padded 10-digit CIK (carried for future use; this feature does not read
it). The two files overlap on 88 tickers with zero CIK disagreement.

`SP500` is the default everywhere, so every existing call site keeps working
unchanged.

Module: **`sec_analyzer/screener/universe.py`**

```python
SP500_CSV_PATH: str          # absolute path to the bundled S&P 500 CSV
NASDAQ100_CSV_PATH: str      # absolute path to the bundled Nasdaq-100 CSV

DEFAULT_INDEX = "SP500"

#: code -> (display label, csv path). Ordered as shown in the UI selector.
UNIVERSES: dict[str, tuple[str, str]]

def universe_label(index: str) -> str:
    """Display label for an index code ("SP500" -> "S&P 500"). Unknown codes
    return the code itself rather than raising."""

def normalize_index(index: str | None) -> str:
    """Case-insensitively resolve an index code to a key of UNIVERSES,
    falling back to DEFAULT_INDEX for None/blank/unrecognized input (so a bad
    query param degrades to the S&P 500 rather than erroring). Accepts the
    aliases "NASDAQ100"/"NASDAQ-100"/"NDX100" for "NDX"."""

def load_universe(index: str = DEFAULT_INDEX, path: str | None = None) -> list[dict]:
    """Read a bundled constituent CSV.

    Returns rows as {"ticker","name","sector","cik"}, de-duplicated by ticker,
    sorted ascending by ticker (deterministic order). Blank/short lines are
    skipped. An explicit `path` overrides the index lookup (used by tests).
    Raises OSError only if the file is missing/unreadable (a packaging error,
    mirroring report/generator._load_template)."""

def price_symbol(ticker: str) -> str:
    """Map a constituent ticker to the symbol the price layer understands.

    Dotted class shares (BRK.B, BF.B) become dash-separated (BRK-B, BF-B),
    which is what yfinance expects. Everything
    else is returned upper-cased and stripped, unchanged."""
```

`load_universe(index=...)` must keep working when called with no arguments —
existing callers and tests rely on the S&P 500 default.

Refreshing either list is a manual, out-of-band operation (replace the CSV).
The runtime never fetches a constituent list over the network.

---

## 2. Score inputs

The scorer takes exactly one argument: the flat indicators dict produced by
`sec_analyzer.technical.indicators.compute_indicators(price_df)`, optionally
merged with a `relative_strength` dict (the SPY cross-check from
`technical.indicators.relative_strength`). No other input. No sector-relative
strength (it would require SIC lookups → SEC access → out of scope).

Every field may be `None`; the scorer must never raise and must never assume a
field is present.

---

## 3. Swing score — `sec_analyzer/technical/swing.py`

```python
def compute_swing_score(indicators: dict | None) -> dict | None
```

Pure, `None`-safe, deterministic. Returns `None` only when **not a single**
component sub-score can be computed (or the input is not a dict). Never raises.

Design mirrors `sec_analyzer/technical/momentum.py`: component sub-scores in
`[-1, 1]`, fixed module-constant weights, renormalized over whichever
components were actually computable, folded into one `s ∈ [-1, 1]`, displayed
as `score = clamp(round(50 + 50*s), 0, 100)` (50 = neutral).

Reuse `momentum.py`'s helper conventions (`_clamp`, `_is_num`) — re-implement
them locally in `swing.py` rather than importing privates across modules.

### 3.0 Coverage floor (ranking integrity)

Renormalizing over a tiny surviving component set produces extreme,
non-comparable scores — a ticker whose only computable component is a
Bollinger squeeze would score ~97 and top the ranking on almost no evidence.
Because this feature's entire purpose is a **cross-sectional ranking**, a row
is only scoreable when it rests on enough evidence:

- `price` must be numeric and > 0, **and**
- the `trend` and `setup` sub-scores must both be computable, **and**
- the summed raw weight of the computable components must be ≥ `0.60`.

If any of those fail, `compute_swing_score` returns `None` (the scan then
records the ticker in `skipped` with a Turkish "yetersiz veri" reason). This
check runs *after* the sub-scores are built and *before* renormalization.

### 3.1 Component weights

| key            | weight | meaning                                          |
|----------------|--------|--------------------------------------------------|
| `trend`        | 0.25   | is the swing-timeframe trend up?                  |
| `setup`        | 0.25   | is price in a *buyable* position (not extended)?  |
| `trigger`      | 0.20   | is there a fresh actionable turn *now*?           |
| `rel_strength` | 0.15   | is it beating SPY?                                |
| `volume`       | 0.10   | do buyers show up?                                |
| `risk_reward`  | 0.05   | is the nearest R:R favourable?                    |

Weights sum to 1.00. A component whose sub-score is `None` is dropped and the
remaining weights renormalized (identical to `momentum.py`).

### 3.2 `trend` sub-score

Average of whichever of these are computable:

- `clamp(sma50_slope_pct / 5.0)`
- `clamp(sma200_slope_pct / 3.0)`
- `+1.0` if `sma50_above_sma200` is True, `-1.0` if False
- `clamp(dist_sma50_pct / 10.0)`

`None` if none available.

### 3.3 `setup` sub-score — the swing-specific part

Average of whichever are computable:

**(a) Extension / pullback quality**, from `ext = dist_sma50_pct` — a swing
entry wants price *at or slightly below* the 50-day, not 25% above it:

```
ext in [-6, +2]        -> +1.0                       (the buy zone)
ext in (+2, +30]       -> linear +1.0 @ +2  -> -1.0 @ +30   (extended)
ext in [-30, -6)       -> linear +1.0 @ -6  -> -1.0 @ -30   (broken down)
|ext| beyond those     -> clamped to -1.0
```

**(b) Base tightness**, from `bb_squeeze`:

```
bb_squeeze.active is True                 -> +1.0
else, percentile p available              -> clamp((50 - p) / 50)
```

**(c) 52-week-high proximity**, from `d = dist_52w_high_pct` (≤ 0):

```
d = 0    -> +1.0 ;  d = -15 -> 0.0 ;  d <= -35 -> -1.0   (piecewise linear)
```

### 3.4 `trigger` sub-score

Average of whichever are computable:

- `rsi_reclaim`: `"bullish"` → `+1.0`, `"bearish"` → `-1.0`, `None` → skip
- MACD: `macd_cross == "bullish"` → `+1.0`; `"bearish"` → `-1.0`; else
  `macd_hist > 0` → `+0.4`, `< 0` → `-0.4`, `== 0` → `0.0`; missing → skip
- `rsi_divergence`: `"bullish"` → `+1.0`, `"bearish"` → `-1.0`, `None` → skip

`volume_climax` deliberately does **not** score — it is a badge only (§3.9).

### 3.5 `rel_strength` sub-score

From `indicators["relative_strength"]` (dict, may be absent/`None`), average of:

- `clamp(rs_3m_pct / 20.0)`
- `clamp(rs_1m_pct / 10.0)`

`None` if the dict is missing or neither field is numeric.

### 3.6 `volume` sub-score

Average of whichever are computable:

- `updown_volume_ratio > 0` → `clamp(log(uvr) / log(2.0))`
- `obv_trend`: `"up"` → `+1.0`, `"flat"` → `0.0`, `"down"` → `-1.0`

### 3.7 `risk_reward` sub-score

Needs `price`, `nearest_support` (`< price`) and `nearest_resistance`
(`> price`); otherwise `None`.

```
reward = nearest_resistance - price
risk   = price - nearest_support
rr     = reward / risk                       (risk > 0 required)
sub    = clamp(log(rr) / log(3.0))           (rr = 3 -> +1, rr = 1 -> 0, rr = 1/3 -> -1)
```

### 3.8 Setup classification (`setup` label, Turkish)

Deterministic, **first match wins**, evaluated against the raw indicators and
the already-computed `trend` sub-score `T` (treat a `None` `T` as `0.0`):

1. `"BREAKOUT"` — `dist_52w_high_pct >= -3` and `T >= 0.2`
2. `"TRENDDE GERİ ÇEKİLME"` — `T >= 0.2` and `-12 <= dist_sma50_pct <= 2`
3. `"SIKIŞMA"` — `bb_squeeze.active is True` and `T >= 0`
4. `"AŞIRI SATIM TEPKİSİ"` — `rsi_reclaim == "bullish"`, or
   (`rsi14 < 40` and `rsi_divergence == "bullish"`)
5. `"MOMENTUM DEVAM"` — `T >= 0.4`
6. `"KURULUM YOK"` — otherwise

A comparison against a `None` field simply fails that rule (no crash).

### 3.9 Badges (`badges`, list of short Turkish strings)

Emitted in this fixed order, only when the condition holds:

- `"HACİM"` — `rel_volume >= 1.5`
- `"KAPİTÜLASYON"` — `volume_climax.detected` and `direction == "down"` and `bars_ago <= 5`
- `"SIKIŞMA"` — `bb_squeeze.active is True`
- `"GOLDEN CROSS"` — `golden_cross is True`
- `"RSI UYUMSUZLUK"` — `rsi_divergence == "bullish"`

### 3.10 Grade label (`label`) — 5 bands on `score`

| score   | label            |
|---------|------------------|
| ≥ 75    | `GÜÇLÜ FIRSAT`   |
| 60–74   | `FIRSAT`         |
| 45–59   | `NÖTR`           |
| 30–44   | `ZAYIF`          |
| < 30    | `KAÇIN`          |

### 3.11 Trade levels

Computed only when `price` is available; each field independently `None` when
its inputs are missing.

- `entry` = `price` (market entry; the screener does not invent limit levels)
- `stop`:
  - base = `price - 2 * atr14` when `atr14` is numeric and > 0, else `None`
  - if `nearest_support` is numeric and `< price`: `stop = max(base, nearest_support * 0.99)`
    (when `base` is `None`, `stop = nearest_support * 0.99`)
  - **Volatility floor.** A structural stop is only usable if the stock's own
    daily range can't reach it by accident: after the step above, if `atr14` is
    available and `price - stop < 1.0 * atr14`, discard the structural stop and
    use `base` instead. Rationale — the `max()` above picks the support-based
    stop whenever support sits above the 2×ATR level, and when support is very
    close to price that yields a sub-ATR stop that ordinary daily noise will
    sweep, while inflating `rr` into fantasy territory (observed on a live
    501-row scan: `UHS` stop 1.6% away against a 3.1% ATR → `rr` 17.33; `AES`
    0.8% against a 0.4% ATR → `rr` 13.92; 12 of 501 rows had a stop under 2%).
    Since `rr` is a sortable column, those artifacts would otherwise top it.
    With `atr14` unavailable there is nothing to floor against, so the
    structural stop stands.
  - `stop` must end up `> 0` and `< price`, else `None`
- `target`:
  - `nearest_resistance` when numeric and `>= price * 1.02`
  - else `price + 3 * atr14` when `atr14` numeric and > 0
  - else `None`
- `rr` = `(target - price) / (price - stop)` when both are valid, rounded 2dp
- `stop_pct` = `(stop / price - 1) * 100`, 1dp; `target_pct` likewise

All prices rounded to 2dp.

### 3.12 Return shape

```python
{
  "score": int,            # 0-100
  "s": float,              # raw [-1, 1], 3dp
  "label": str,            # §3.10
  "setup": str,            # §3.8
  "badges": list[str],     # §3.9, possibly empty
  "components": [          # only the computed ones, in _WEIGHTS declaration order
    {"key": str, "label": str, "sub": float, "weight": float, "points": float}
  ],
  "entry": float | None,
  "stop": float | None,
  "stop_pct": float | None,
  "target": float | None,
  "target_pct": float | None,
  "rr": float | None,
  "summary": str,          # one-line Turkish readout
}
```

`points = weight * sub * 50` (3dp→1dp as in momentum.py); the points sum to
`score - 50` up to rounding. `components[].label` uses these Turkish labels:
`trend` → `"Trend"`, `setup` → `"Kurulum kalitesi"`, `trigger` → `"Tetikleyici"`,
`rel_strength` → `"Relatif güç"`, `volume` → `"Hacim"`, `risk_reward` → `"Risk/Ödül"`.

`summary` format: `"{score}/100 {label.lower()} — {setup.lower()}"`, plus
`"; en güçlü: {label}"` for the highest-`points` component when positive, and
`", en zayıf: {label}"` for the lowest when negative. All JSON-native scalars
(no numpy types — they break `json.dumps`).

---

## 4. Scan orchestration — `sec_analyzer/screener/swing_scan.py`

```python
BENCHMARK = "SPY"
DEFAULT_MAX_WORKERS = os.cpu_count() or 8

def scan_swing(
    tickers: list[str] | None = None,
    no_cache: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress_cb=None,
    index: str = "SP500",
    use_processes: bool = True,
) -> dict
```

**Concurrency.** The per-ticker work is CPU-bound pure Python (`compute_indicators`
does pivot/S-R/divergence loops, ~1s per ticker), so the fan-out uses
`ProcessPoolExecutor` — a thread pool cannot parallelize it under the GIL.
Measured on a 12-core box over a warm 103-name Nasdaq-100 cache: **17.8s with
processes vs. 71.8s with threads (4.0x)**, with byte-identical rows and skips.
`DEFAULT_MAX_WORKERS` is therefore core-count-based, not the network-oriented
`8` this spec originally specified. See `backtest/SWING_STUDY_SPEC.md` §6 for
the full rationale and the worker rules (module-level picklable workers,
per-process benchmark cache, parent-side progress, thread fallback).

`use_processes=False` forces the thread path; both paths must produce identical
results. Fan-outs below a small ticker threshold stay on threads regardless,
since process-pool startup dominates for a handful of names (and monkeypatched
in-process test fixtures are not visible to spawned workers).

`index` is resolved through `normalize_index` and selects which bundled
universe to scan; it defaults to `SP500` so existing calls are unaffected.

Behaviour:

1. Load the universe for `index` (§1). When `tickers` is given, filter that
   universe to it (unknown tickers are still scanned, with `name`/`sector` =
   `None`).
2. Fetch the SPY benchmark frame **once** via
   `fetch.prices.get_price_history(BENCHMARK, no_cache=no_cache)`. On failure,
   log a warning and continue with `relative_strength = None` for every row
   (the `rel_strength` component then simply drops out).
3. Scan constituents through a `concurrent.futures.ThreadPoolExecutor`
   (`max_workers`, stdlib). Per ticker:
   `get_price_history(price_symbol(t))` → `compute_indicators(df)` →
   attach `relative_strength(df["Close"], spy["Close"], benchmark="SPY")` →
   `compute_swing_score(indicators)`.
4. Any exception for a ticker (including `PriceDataError`) is caught, logged at
   warning level, and recorded in `skipped` as
   `{"ticker": str, "reason": str}` — **one bad ticker never fails a scan.**
5. `progress_cb(done: int, total: int, ticker: str)` is called after each
   ticker completes (success or skip), if provided. It must be wrapped in its
   own try/except — a caller's broken callback cannot break the scan.

Per-row output:

```python
{
  "ticker": str, "name": str|None, "sector": str|None,
  "price": float|None, "as_of": str|None,          # last bar date
  "change_1d_pct": float|None,
  "rsi14": float|None, "dist_sma50_pct": float|None,
  "dist_52w_high_pct": float|None,
  "rs_3m_pct": float|None, "atr_pct": float|None,   # atr14/price*100, 1dp
  "score": int, "s": float, "label": str, "setup": str, "badges": list[str],
  "entry": float|None, "stop": float|None, "stop_pct": float|None,
  "target": float|None, "target_pct": float|None, "rr": float|None,
  "summary": str, "components": list[dict],
}
```

Return value:

```python
{
  "generated_at": str,       # ISO-8601 UTC seconds, e.g. "2026-07-27T09:12:03Z"
  "price_as_of": str|None,   # most common (mode) last-bar date across rows
  "universe": str,           # the resolved index code, "SP500" or "NDX"
  "universe_label": str,     # its display label, "S&P 500" or "Nasdaq 100"
  "count": int,              # len(rows)
  "total": int,              # tickers attempted
  "skipped": list[dict],
  "rows": list[dict],        # sorted: score DESC, then ticker ASC
}
```

`generated_at` is the **only** wall-clock value in the feature; no score
depends on it (determinism rule satisfied: same price inputs → same scores and
same row order).

---

## 5. Persistence — `sec_analyzer/store/database.py`

New table, created inside the existing `init_db()` transaction alongside the
others:

```sql
CREATE TABLE IF NOT EXISTS swing_scans (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    price_as_of  TEXT,
    universe     TEXT NOT NULL,
    count        INTEGER NOT NULL,
    payload      TEXT NOT NULL          -- the full §4 dict as JSON
)
```

A single JSON payload column is deliberate: this is a display artifact
(recomputable from prices at any time), not a fundamental fact, and 500 rows
render fine client-side. Do not shred it into per-ticker columns.

```python
def save_swing_scan(result: dict, db_path: str | None = None) -> int:
    """Persist a §4 scan result; returns the new row id. The `universe`
    column is taken from result["universe"]. Never raises — logs and returns
    0 on failure (mirrors save_prices' non-fatal posture)."""

def load_latest_swing_scan(
    db_path: str | None = None, universe: str | None = None
) -> dict | None:
    """Most recent scan's payload as a dict, or None when none stored /
    unreadable. With `universe` given, returns the most recent scan for THAT
    index only (so switching indexes in the UI never shows another index's
    rows); with `universe=None` returns the most recent scan of any index,
    preserving the original single-index behaviour. Never raises."""
```

Each index keeps its own independent scan history — persisting an `NDX` scan
must never make the stored `SP500` scan unreachable.

---

## 6. HTTP API — `sec_analyzer/web/app.py`

Scans take minutes on a cold cache, so the scan runs in a background thread
with a module-level state guard. Exactly one scan may run at a time.

```python
GET  /swing?index=NDX          -> HTML page (mode "swing", §7)
POST /api/swing/scan           -> start a scan
GET  /api/swing/status         -> progress poll
GET  /api/swing/results?index= -> latest stored scan for an index
```

Every index parameter is passed through `normalize_index`, so a missing or
bogus value degrades to `SP500` instead of erroring.

- **`POST /api/swing/scan`** — JSON body (all optional):
  `{"no_cache": bool, "limit": int|None, "index": str}` (`limit` = first N
  universe tickers, for a quick smoke run). Starts a
  `threading.Thread(daemon=True)` and returns
  `202 {"ok": true, "status": "running", "total": N, "index": <code>}`. If a
  scan is already running, returns
  `409 {"ok": false, "error": "<label> için bir tarama zaten çalışıyor."}`
  naming the index actually running — still **one scan at a time globally**,
  regardless of index, since the bottleneck is CPU (see §10). On completion the
  worker calls `save_swing_scan(...)`.
- **`GET /api/swing/status`** →
  `{"ok": true, "running": bool, "done": int, "total": int, "index": str|None,
    "index_label": str|None, "started_at": str|None, "finished_at": str|None,
    "error": str|None}`. `index` is the index of the currently-running (or
  last-completed) scan, so the UI can tell whether the progress it sees
  belongs to the index it is displaying.
- **`GET /api/swing/results`** — query param `index` (default `SP500`) →
  `{"ok": true, "index": str, "scan": <§4 dict or null>}`.
- **`GET /swing`** — query param `index` (default `SP500`) →
  `render_swing_page(scan=load_latest_swing_scan(universe=<code>), index=<code>)`
  (§7). Wrapped in try/except → `_error_page(...)` + 500, exactly like
  `/history`.

All state (`running`, `done`, `total`, `index`, `started_at`, `finished_at`,
`error`) lives in one module-level dict guarded by a `threading.Lock`.

---

## 7. Rendering — `sec_analyzer/report/generator.py` + `template.html`

New generator entry point, same shell/placeholder mechanism as the others:

```python
def render_swing_page(
    scan: dict | None, index: str = "SP500", indexes: list[tuple[str, str]] | None = None
) -> str:
    payload = {
        "mode": "swing",
        "scan": scan,                    # §4 dict or None
        "index": index,                  # currently displayed index code
        "indexes": [list(p) for p in (indexes or [])],   # [[code, label], ...]
        "generated_on": date.today().isoformat(),
    }
    return _inject_payload(payload)
```

`indexes` comes from `universe.UNIVERSES` (the route builds it), so adding a
third index later needs no template change.

`template.html` changes (no external resources — the self-containment rule
still holds):

1. `render()` dispatch gains `else if (data.mode === "swing") renderSwingMode();`
2. `renderSearchMode()`'s nav gains a `Swing` button next to `Geçmiş`
   (`.history-nav-btn` class reused) that navigates to `/swing`.
3. `renderSwingMode()` renders:
   - Topnav: brand + a `Ara` button back to `/`.
   - Header card: title `Swing Trade Fırsatları — <index label>`, an **index
     selector**, scan meta (`generated_at`, `price_as_of`, `count`, skipped
     count), a `Taramayı Başlat` button, and a progress line while running.
   - **Index selector.** Rendered from `data.indexes` as a segmented pill
     control (reuse the `.horizon-pill` / `nav-horizon-toggle` pattern already
     in the template) with the current `data.index` active. Selecting an index
     navigates to `/swing?index=<code>` — a plain navigation, not an in-page
     fetch, so the server serves that index's stored scan and the URL stays
     shareable/bookmarkable. Each index shows its OWN last scan: switching to
     an index never scanned yet shows the empty state, not the other index's
     rows.
   - The scan button scans the **currently selected index** and sends its code
     as `index` in the POST body. While a scan for a *different* index is
     running, the button is disabled with a note naming that index — one scan
     at a time globally.
   - Filter bar: minimum-score number input, setup-type `<select>` (populated
     from the setups actually present), and a ticker/name text filter. All
     filtering is client-side over the already-loaded rows.
   - Ranked table (reusing `.sensitivity-table` styling):
     `# · Hisse · Sektör · Fiyat · Skor · Etiket · Kurulum · RSI · SMA50 % ·
     RS 3a · Giriş · Stop · Hedef · R:R · Rozetler`.
     `Skor` renders as a number plus a thin proportional bar. Column headers
     for `Skor`, `RSI`, `RS 3a`, `R:R` are click-to-sort (default: score DESC).
   - Ticker cell links to `/report?ticker=<T>&horizon=3m` (new tab).
   - **Pagination: 50 rows per page.** The `#` rank column shows the row's
     absolute rank within the current filter+sort (page 2 starts at 51), not
     its index on the page. Pager controls sit below the table: `Önceki` /
     `Sonraki`, the page numbers, and a `X–Y / N` counter. Changing a filter or
     the sort column resets to page 1. Pagination is client-side over the
     already-loaded rows — no refetch on page change.
   - Empty state when `scan` is null: `Henüz tarama yapılmadı.`
   - `globalFooterHtml()` at the end, like the other modes.
4. The `Taramayı Başlat` button scans the **full selected universe** — it
   POSTs `{"no_cache": false, "index": <selected code>}` with **no** `limit`.
   (`limit` stays in the API for CLI/smoke use, but the UI never sends it.) A
   full scan takes minutes, so the progress readout is not optional.
5. Scan button flow: `POST /api/swing/scan` → poll `GET /api/swing/status`
   every 2000 ms → on `running: false` fetch `GET /api/swing/results` and
   re-render the table in place. Show `done/total` while running and disable
   the button. A `409` shows the returned error and starts polling anyway.
6. Missing numbers render as `—`, never `None`/`NaN`/`undefined`.

CSS: add only what's needed (score bar, badge chips, filter bar) in the same
dark palette / variable set already in the file.

---

## 8. CLI — `sec_analyzer/cli.py`

New subcommand, so the cache can be warmed offline before using the web tab:

```
python -m sec_analyzer.cli swing [--index sp500|ndx] [--limit N] [--no-cache]
                                 [--workers 8] [--top 25] [--no-save]
```

`--index` defaults to `sp500`, is case-insensitive, and goes through
`normalize_index`. The printed header names the index label.

Prints a ranked plain-text table (rank, ticker, score, label, setup, price,
entry/stop/target, R:R) of the top `--top` rows, plus a one-line summary of
skipped tickers, and persists the scan via `save_swing_scan` unless
`--no-save`. Progress goes to the log, one line per ~25 tickers. Never raises
out to the user: a failed scan prints an error line and returns exit code 1.

---

## 9. Invariants (test targets)

1. `compute_swing_score({})` → `None`. `compute_swing_score(None)` → `None`.
   Neither raises.
2. Score is monotone in each component: raising one sub-score's input, all else
   equal, never lowers `score`.
3. A hand-built "textbook pullback" indicator dict scores ≥ 70; a hand-built
   "extended, deteriorating downtrend" dict scores ≤ 30.
4. Determinism: calling `compute_swing_score` twice on the same dict returns
   equal dicts.
5. Every value in the returned dict is JSON-serializable (`json.dumps` round
   trip).
6. `price_symbol("BRK.B") == "BRK-B"`; `price_symbol(" aapl ") == "AAPL"`.
7. `load_universe()` returns ≥ 490 rows, unique tickers, ascending order.
8. `scan_swing` with a stubbed price layer where one ticker raises → that
   ticker lands in `skipped`, all others still appear in `rows`.
9. `scan_swing` rows are sorted score DESC, ticker ASC.
10. `save_swing_scan` / `load_latest_swing_scan` round-trip a payload through a
    temp DB, and `load_latest_swing_scan` on an empty DB returns `None`.
11. Coverage floor (§3.0): an indicators dict carrying only `bb_squeeze`
    (no price, no trend inputs) returns `None`, not a high score. A dict with
    `trend` + `setup` computable but `price` missing also returns `None`.
12. Stop volatility floor (§3.11): a support-based stop closer than 1×ATR
    falls back to the 2×ATR stop; one exactly at 1×ATR is kept; with `atr14`
    unavailable the structural stop stands.
13. Multi-index (§1): `load_universe()` with no argument still returns the 503
    S&P 500 rows; `load_universe("NDX")` returns 103 Nasdaq-100 rows, unique
    and ascending. `normalize_index` maps `None`/`""`/`"garbage"` →
    `"SP500"`, and `"ndx"`/`"NASDAQ100"`/`"Nasdaq-100"` → `"NDX"`.
14. Per-index persistence (§5): after saving an `SP500` scan and then an `NDX`
    scan, `load_latest_swing_scan(universe="SP500")` still returns the S&P 500
    payload, `load_latest_swing_scan(universe="NDX")` returns the Nasdaq one,
    and `load_latest_swing_scan()` returns the most recent of the two.
    `load_latest_swing_scan(universe="NDX")` on a DB holding only an `SP500`
    scan returns `None`.
15. `scan_swing(index="NDX")` scans the Nasdaq-100 universe and its result
    carries `universe == "NDX"` / `universe_label == "Nasdaq 100"`.
