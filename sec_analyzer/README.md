# sec_analyzer

`sec_analyzer` fetches a public company's financial statements straight from
the **official SEC EDGAR API**, normalizes the raw XBRL facts into a clean,
per-fiscal-year time series, computes a handful of standard fundamental
ratios, stores everything locally in SQLite, and (optionally) produces a
conservative, data-grounded fundamental read -- either from Claude, from a
local Ollama/Gemma model, or from a fully deterministic, no-AI script
analyzer (see "Script-based analysis" below).

**The only data source is SEC EDGAR.** There is no `yfinance`, no
`openai`, and no other third-party market-data or LLM library anywhere in
this package -- just `requests` against `data.sec.gov` / `www.sec.gov`, and
the official `anthropic` SDK for the optional Claude-backed analysis step
(not required for the `ollama` or `script` providers).

## What it does

1. **Resolve** a ticker symbol to its SEC CIK via `company_tickers.json`.
2. **Fetch** the filer's full XBRL "company facts" document
   (`data.sec.gov/api/xbrl/companyfacts/...`).
3. **Normalize** the raw, messy XBRL facts (tag fallbacks, restatements,
   annual vs. quarterly period selection) into a tidy per-concept,
   per-fiscal-year series.
4. **Compute ratios**: net margin, ROE, current ratio, and YoY revenue /
   net-income growth.
5. **Store** the ticker, CIK, normalized series, and ratios in a local
   SQLite database.
6. **Interpret** (optional, `analyze` command): run the normalized figures
   and ratios through a selectable backend -- Claude, a local Ollama/Gemma
   model, or the deterministic script analyzer -- and get back a structured
   JSON verdict: a conservative fair-value range, a fundamental-quality
   verdict, cyclicality commentary, and a plain-language summary.

## Install

```powershell
pip install -r sec_analyzer/requirements.txt
```

## ⚠️ Set your SEC User-Agent first

SEC EDGAR **blocks or rate-limits requests with a generic or missing
User-Agent header.** Every request this tool makes must identify a real
requester. Before running anything, create a `.env` file (in the directory
you run the CLI from, or anywhere `python-dotenv` will discover it) with:

```env
SEC_USER_AGENT="Your Name your.email@example.com"
ANTHROPIC_API_KEY="sk-ant-..."
```

`SEC_USER_AGENT` is required for every command (`fetch` and `analyze`).
`ANTHROPIC_API_KEY` is only required for `analyze`. There is no default for
`SEC_USER_AGENT` on purpose -- the tool will raise a clear configuration
error rather than silently sending an anonymous-looking request that SEC
might block.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SEC_USER_AGENT` | Yes (all commands) | *(none -- must be set)* | Identifies the requester to SEC EDGAR, e.g. `"Jane Doe jane.doe@example.com"`. |
| `ANTHROPIC_API_KEY` | Yes (`analyze` only) | *(none -- must be set)* | Anthropic API key used by the `analyze` command. |
| `SEC_ANTHROPIC_MODEL` | No | `claude-opus-4-8` | Anthropic model ID used for the Claude analysis. Deliberately *not* the generic `ANTHROPIC_MODEL`, so a shared `.env` can't silently downgrade the model. |
| `SEC_DB_PATH` | No | `sec_analyzer/sec_data.sqlite3` | Path to the local SQLite database. |
| `METODOLOJI_PATH` | No | `sec_analyzer/METODOLOJI.md` | Optional path to a custom analysis methodology (see below). |
| `PROFIL_PATH` | No | `./PROFIL.md` | Optional path to your personal risk/style profile (see "Your investor profile (`PROFIL.md`)" below). |
| `REPORTS_DIR` | No | `./reports` | Directory the `--html` verdict-report files are written into (see "HTML verdict report (`--html`)" below). |
| `SEC_MAX_RPS` | No | `8` | Maximum requests per second sent to SEC EDGAR (SEC's fair-access limit is 10/s). |

## Usage

```powershell
# Fetch, normalize, and store 5 years of financials for Apple (no Claude call)
python -m sec_analyzer.cli fetch AAPL --years 5

# Same, plus a Claude fundamental + technical analysis (default horizon: 1y)
python -m sec_analyzer.cli analyze AAPL

# Analyze for a specific investment horizon
python -m sec_analyzer.cli analyze AAPL --horizon 5y

# Same, but with the deterministic, no-AI script analyzer instead
python -m sec_analyzer.cli analyze AAPL --provider script

# Also save a standalone HTML verdict-card report
python -m sec_analyzer.cli analyze AAPL --html

# Bypass the on-disk cache and re-fetch fresh data from SEC
python -m sec_analyzer.cli fetch AAPL --no-cache

# Verbose / quiet logging
python -m sec_analyzer.cli analyze AAPL --verbose
python -m sec_analyzer.cli fetch AAPL --quiet
```

Both subcommands accept `--years N` (default 5) and `--no-cache`. `analyze`
additionally accepts `--horizon {3m,1y,5y}` (default `1y`) and `--html`.

**`--years` note:** the default of 5 fiscal years is fine for a quick read,
but the valuation engine's cyclical-earnings DCF variant (median FCF margin
across all available years) and the historical multiples percentiles (P/E,
P/S, P/FCF) both get meaningfully more reliable with more history --
**10-15 years is recommended** for cyclical names (SIC-classified
`cyclical`, e.g. miners, chemicals, autos, semis, shipping) or whenever you
want a trustworthy multiples percentile (the percentile calculation itself
requires at least 5 non-null historical values to return anything at all).
For a fast look at a mature, non-cyclical name, the default of 5 is usually
enough.

If the analysis backend fails for any reason (missing API key, network error,
invalid API response, or a non-JSON model reply), it prints a clear warning
to stderr but still shows the financials that were fetched, normalized, and
stored, plus a verdict card with whatever it could still compute -- it never
crashes on an analysis failure.

`analyze`'s default output is a compact, Turkish-language terminal **verdict
card** (see "Investment horizon (`--horizon`)" below) rather than a raw JSON
dump; pass `--verbose` if you want the full JSON result printed as well.

## Investment horizon (`--horizon`)

`analyze` takes a `--horizon` flag (`3m`, `1y`, or `5y`; default `1y`) that
controls how much weight the final verdict puts on fundamentals vs.
technicals, and how the technical/fundamental commentary is framed (e.g. a
3-month horizon foregrounds momentum and the next earnings catalyst; a
5-year horizon foregrounds fundamentals and explicitly discounts short-term
indicators like RSI). The weighting is fixed and defined in
`Config.HORIZON_WEIGHTS`:

| Horizon | Fundamental weight | Technical weight | Framing |
|---|---|---|---|
| `3m` | 30% | 70% | Momentum/technicals lead; the next earnings catalyst matters most. |
| `1y` (default) | 50% | 50% | Balanced: fundamentals and technicals are weighed evenly. |
| `5y` | 80% | 20% | Fundamentals lead; short-term indicators (e.g. RSI) are explicitly de-emphasized. |

The same flag is available in the web UI as a **Horizon** dropdown next to
the provider selector, and is sent as `horizon` in the `/api/analyze` JSON
body.

## HTML verdict report (`--html`)

Passing `--html` to `analyze` additionally renders a standalone,
self-contained HTML "verdict card" report -- a dark-themed page with a
fair-value fan chart (bear/base/bull bands vs. the current price), three
color-coded verdict chips (fundamental / technical / profile fit), a red-flags
panel, and the catalyst + summary -- and saves it to disk. No CDNs, fonts, or
other external resources are used, so the file opens correctly straight from
disk (`file://...`) or as an email attachment.

Reports are written to `Config.REPORTS_DIR` (default `./reports`, overridable
via the `REPORTS_DIR` environment variable) as
`<TICKER>_<YYYY-MM-DD>_<horizon>.html`, e.g. `reports/NVDA_2026-07-12_1y.html`.
The CLI prints the saved path after generating the report.

## Değerleme motoru (valuation engine)

`analyze`'ın fair-value bandı, artık **deterministik bir Python hesap
motoru** (`sec_analyzer/valuation/`) tarafından üretilir; LLM (Claude/Ollama)
sayıların kendisini hesaplamaz. Mimari, iki aşamalı bir LLM akışı üzerine
kurulu:

1. **Faz 1 — Varsayım seti:** LLM'e sadece bear/base/bull için `growth_5y`,
   `terminal_growth`, `discount_rate` ve tek cümlelik bir `story` önerisi
   sorulur (ayrıca bir `sector_type` tahmini). Öneri `VALUATION.md`'deki
   sınırları (örn. terminal büyüme ≤ %4, iskonto oranı ≥ %7/%10) ihlal
   ederse bir kez revize edilmesi istenir; hâlâ geçersizse veya LLM
   kullanılamıyorsa (`--provider script`), deterministik varsayılan
   varsayımlara (gerçekleşen gelir CAGR'ından türetilen) düşülür.
2. **Deterministik hesap:** Bu varsayımlar, `valuation/engine.py`'daki
   `run_valuation()` içinden DCF, reverse DCF, sektöre göre P/B×ROE veya
   normalize-kazanç DCF varyantı, tarihsel multiples yüzdelikleri,
   duyarlılık matrisi ve üç yöntemli üçgenlemeyi (DCF/reverse-DCF/multiples
   → GÜVEN: YÜKSEK/ORTA/DÜŞÜK) hesaplar. Aynı girdiler her zaman aynı
   sayıları üretir.
3. **Faz 2 — Yorum:** LLM'e hesaplanmış `valuation` sözlüğünün tamamı
   verilir; LLM sadece yorum alanlarını (`fundamental_verdict`,
   `profile_fit`, `reverse_dcf_comment`, `cyclical_risk`, `summary`, ...)
   doldurur ve rakamların kendisini değiştiremez. `script` sağlayıcısı bu
   akışı tamamen çevrimdışı, şablon tabanlı yorumla (LLM'siz) yürütür.

Değerleme kurallarının kendisi (sektör → yöntem eşlemesi, varsayım
sınırları, senaryo kurgusu, duyarlılık/belirsizlik eşiği, reverse DCF
yorum kuralı, üçgenleme güven eşikleri) `sec_analyzer/VALUATION.md`'de
(`Config.VALUATION_PATH`, varsayılan `sec_analyzer/VALUATION.md`)
belgelenmiştir ve `METODOLOJI.md`'den sonra, `PROFIL.md`'den önce sistem
promptuna enjekte edilir.

### Planlama alanları: `scenario_returns`, `entry_plan`, `stop_adding`, `thesis_metric`

`analyze`'ın sonuç sözlüğü, Faz 2 yorumunun yanında dört ek alan daha içerir.
Bunlar `fair_value_range`/`confidence` gibi **her zaman kod tarafından**
(`sec_analyzer/interpret/planning.py`) hesaplanır ve her sağlayıcı için
(`ollama`, `anthropic`, `script`) aynı şekilde enjekte edilir — hiçbir
sağlayıcı (LLM'ler dahil) bu alanları kendisi hesaplamaz. METODOLOJI.md
§4-§7'yi uygular:

- **`scenario_returns`** — her bear/base/bull senaryosu için güncel fiyattan
  bant kenarına % getiri: `{"bear": {"ret_lo_pct": sayı|null, "ret_hi_pct":
  sayı|null}, "base": {...}, "bull": {...}}` (1 ondalık; fiyat veya bant
  eksikse `null`).
- **`entry_plan`** — 0-5 elemanlı, fiyata göre azalan sıralı, İKİ YÖNLÜ tek
  plan (toplam boyut ~%100): `{"n": int, "trigger": str (Türkçe, sadece
  günlük kapanış), "price_zone": {"lo": sayı, "hi": sayı}, "size_pct": sayı,
  "invalidation": sayı, "target": sayı|null, "rr": sayı|null, "note":
  str|null (ör. tetik seviyesi model bull.hi hedefinin üzerindeyse "Model
  üstü" işareti — bkz. `METODOLOJI.md` §1.5), "kind": "dip"|"breakout"}`.
  **`kind="dip"`** tranche'ları
  (seviye ≤ güncel fiyat) fair-value bandının bear.lo/base.lo/base.hi/
  bull.hi'ından ve teknik low_52w/sma50/sma200'den gelir; hepsi TEK
  paylaşılan yapısal invalidation'ı paylaşır, bu yüzden R:R fiyat düştükçe
  hiç azalmaz. **`kind="breakout"`** tranche'ları (seviye > güncel fiyat)
  teknik sma50/sma200 geri alımından, `resistance_levels` kırılımından ve
  high_52w kırılımından gelir; her biri KENDİ başarısız-kırılım
  invalidation'ını taşır (paylaşılan invalidation'a dahil değildir), bu
  yüzden dip R:R'leriyle aynı ölçekte karşılaştırılmaz. Her iki yönde de
  aday varsa en az birer tranche garanti edilir, kalan slotlar fiyata en
  yakın seviyelerden doldurulur; sadece tek yönde aday varsa o yönden en
  fazla 5 alınır. Fiyat eksik/negatifse veya hiçbir yönde aday seviye yoksa
  `[]`.
- **`stop_adding`** — `[{"code": str, "message": str}, ...]` biçiminde,
  sabit sırayla kontrol edilen sinyaller: `BELOW_BEAR_FLOOR`,
  `NEAR_INVALIDATION`, `HIGH_UNCERTAINTY`, `ACTIVE_RED_FLAG`,
  `BINARY_CATALYST_NEAR`. Konsantrasyon limiti sinyalleri kapsam dışıdır
  (henüz bir `POZISYONLAR.md` pozisyon şeması yok — bkz. `ROADMAP.md`).
- **`thesis_metric`** — tek çapa metrik: `{"name": str, "latest_value":
  str|null, "trend": "iyileşiyor"|"bozuluyor"|"yatay"|null, "rationale":
  str, "cycle": dict|null}`. Metrik, sektör tipine göre seçilir (olgun →
  net kâr marjı/ROE, kâr etmeyen büyüme → yıllık gelir büyümesi, finansal →
  ROE, GYO → FCF marjı, döngüsel → brüt/net kâr marjı) ve
  `ratios`/`metrics`'ten okunur, asla uydurulmaz; `rationale` her zaman
  METODOLOJI §7'nin "iki ardışık çeyrek tezin aksini gösterirse tez
  geçersizdir" kuralını içerir. `cycle`, çapa metriğin güncel değerini kendi
  çok yıllık dip→zirve aralığı içinde konumlandırır (raporda "döngü konumu"
  çubuğu olarak çizilir): `{"low", "high", "current", "position" (0..1),
  "low_fy", "high_fy", "current_fy", "n_years", "is_cyclical", "series"}`.
  `series`, mali yıla göre artan sıralı tam yıllık seridir (`[{"fy",
  "value"}, ...]`) ve konum çubuğunun yanında metriğin seyrini gösteren bir
  sparkline çizgisi çizmek için kullanılır. En az iki farklı yıllık değer
  yoksa, seri tamamen düzse veya metrik tek-noktalı `metrics` yedeğinden
  geldiyse `null` olur.

### Damodaran sektör verisi kurulumu

Sektör bazlı çoklu karşılaştırması (P/E, P/S, P/FCF medyanları), CAPM
iskonto oranı (sektör kaldıraçsız betası + ERP + risksiz getiri, bkz.
VALUATION.md §4) ve sektör ERP tabanı, isteğe bağlı olarak
`data/damodaran/multiples.csv` ve
`data/damodaran/erp.csv` dosyalarından okunur (yol: `Config.DAMODARAN_DIR`,
varsayılan `<çalışma dizini>/data/damodaran`, `DAMODARAN_DIR` ortam
değişkeniyle override edilebilir). Dosya formatı, kaynak (Damodaran'ın NYU
Stern sayfası), Excel→CSV dönüştürme adımları ve dosyalar eksikken neyin
devre dışı kaldığı için bkz. **[`data/damodaran/README.md`](../data/damodaran/README.md)**.
Kısacası: dosyalar yoksa araç normal çalışmaya devam eder, sadece sektör
karşılaştırması ve sektör ERP tabanı devre dışı kalır (bu durum loglanır).

### Terminal çıktısı örneği (değerleme eklentileriyle)

`analyze`'ın verdict kartı, eski satırların ardından şu ek satırları
gösterir (her biri eksik veriyse `—` ile):

```
AAPL — Vade: 1y — 2026-07-13
─────────────────────────────
Fiyat: $212.40
Fair Value (base): $180–$220
  bear $150–190 (%8 büyüme, %12 dr) | bull $230–270 (%20 büyüme, %9 dr)
Fair Value (base, DCF): $95–$115   Güven: ORTA
Reverse DCF: fiyat 10y %19 CAGR ima ediyor (gerçekleşen 5y: %14)
Multiples:   P/E kendi 9y medyanının 88. yüzdeliğinde
Üçgenleme:   DCF pahalı · rDCF pahalı · multiples pahalı → yön net
Duyarlılık:  base $87–$131 (g±2pp, r±1pp) — yüksek belirsizlik
Fundamental: MAKUL
Teknik:      GÜÇLÜ AL (RSI 45, SMA50 üstü)
Profil:      UYUMLU — Growth+senaryo stiliyle uyumlu
Red flags:   yok
Katalizör:   ~15 Ağu 2026 (tahmini)
Özet: ...
```

`Fair Value (base, DCF)` satırındaki yöntem etiketi sektöre göre değişir
(finansal/REIT için `P/B×ROE` yazar). `Duyarlılık` satırındaki "yüksek
belirsizlik" uyarısı, sadece 3×3 duyarlılık matrisinin bant genişliği base
hücrenin %60'ını aştığında eklenir.

## Price data source

The technical-analysis layer (RSI, moving averages, 52-week range,
volatility, and the technical verdict) needs a daily OHLCV price history,
which is **not** available from SEC EDGAR. `sec_analyzer` fetches it from
[Stooq](https://stooq.com)'s free, no-key CSV endpoint first; if Stooq is
unavailable or returns something unusable (Stooq occasionally serves an
HTML/JS-walled page instead of the CSV on some networks), it automatically
falls back to the optional [`yfinance`](https://pypi.org/project/yfinance/)
package if it's installed. Price history is cached on disk for 24 hours.

If both sources fail (or `yfinance` isn't installed and Stooq is
unreachable), the technical layer is skipped gracefully: `analyze` still
runs the full fundamental analysis, with the technical verdict reported as
unavailable rather than the command failing outright.

## SEC rate limits

SEC EDGAR's fair-access policy caps requests at **10 per second per IP**.
`SecHttpClient` throttles itself to `SEC_MAX_RPS` (default 8, safely under
the limit) and automatically retries with exponential backoff (capped at
30 seconds per attempt, honoring any `Retry-After` header) on `429` (rate
limited), `403` (transient block), and `5xx` responses. A `404` is raised
immediately without retrying, since it usually means the resource genuinely
doesn't exist.

## Caching

Raw JSON responses from SEC (the ticker index and each filer's
`companyfacts` document) are cached on disk under `sec_analyzer/raw/`, keyed
by CIK. Subsequent runs against the same ticker reuse the cache instead of
re-fetching. Pass `--no-cache` to bypass the cache and force a fresh fetch
(the cache file is overwritten with the new response).

## Custom analysis methodology (`METODOLOJI.md`)

The `analyze` command's system prompt comes from a methodology document. If
a file exists at `METODOLOJI_PATH` (default: `sec_analyzer/METODOLOJI.md`)
and is non-empty, its contents are used verbatim as the system prompt's
analysis framework. **`sec_analyzer/METODOLOJI.md` ships checked into the
repository** as the default, muhafazakâr fundamental-analysis framework
(conservative stance, "no invented numbers" rule, margin/cash-flow quality,
cyclical-trap check, horizon weighting -- see that file for the full text);
edit it in place, or point `METODOLOJI_PATH` elsewhere, to use your own
methodology instead. If the file is missing, empty, or unreadable for any
reason, a minimal built-in fallback framework is used instead (see
`sec_analyzer/interpret/analyzer.py`, `DEFAULT_METHODOLOGY`) so the system
prompt is never empty.

The system prompt is assembled in a fixed order: `METODOLOJI.md` (general
methodology) -> `sec_analyzer/VALUATION.md` (the valuation-engine rulebook
-- sector-to-method mapping, assumption bounds, scenario construction,
reverse-DCF and triangulation interpretation rules; see "Değerleme motoru
(valuation engine)" below) -> `PROFIL.md` (your investor profile, see
next section) -> the horizon instruction -> the fixed JSON output contract.

## Your investor profile (`PROFIL.md`)

`analyze` can weigh its verdict against your own stated risk tolerance,
investing style, position limits, and known behavioral quirks. If a file
named `PROFIL.md` exists at `PROFIL_PATH` (default: `./PROFIL.md`, i.e. the
current working directory), its contents feed the `profile_fit` verdict
(`UYUMLU`/`KISMEN`/`UYUMSUZ` -- fits / partially fits / doesn't fit -- plus a
reason) in every analysis. It's entirely optional: if the file is missing,
a neutral default profile is assumed and this is called out in the result.

Use exactly this template (any of the four sections may be filled in as much
or as little as you like):

```markdown
## Risk toleransı
Yüksek — %30 drawdown kaldırabilirim, binary-outcome pozisyonlara açığım (küçük boyutla)

## Stil
Growth + senaryo bazlı value. Momentum'a girmem (bilinen zaafım: giriyorum — uyar)

## Limitler
Tek pozisyon max %15, tek sektör max %40

## Davranışsal notlar
Tetik seviyesi beklemeden haber üzerine erken girme eğilimi var — verdict'te hatırlat
```

Two important caveats:

* **How it's used differs by provider.** The LLM providers (`ollama`,
  `anthropic`) receive the full text of `PROFIL.md` verbatim in the system
  prompt and can actually reason about it (e.g. flagging a position that
  would breach your stated sector limit). The deterministic `script`
  provider **cannot** interpret free text -- it only detects whether
  `PROFIL.md` exists at all, and always returns `KISMEN` with a note
  pointing you at an LLM provider for a real profile-fit read.
* This file is **not** checked into the repository -- unlike
  `METODOLOJI.md`/`VALUATION.md` (which ship as the default,
  repository-tracked methodology/valuation rulebook), `PROFIL.md` is
  personal, optional, user-supplied configuration.

## Script-based analysis (no AI)

Alongside the two LLM providers, `analyze` supports a third backend:
`--provider script` (also selectable in the web UI as **Script (no AI ·
deterministic)**).

* **What it is.** A deterministic fundamental screen computed entirely with
  plain arithmetic over the same normalized SEC figures, ratios, valuation
  metrics, technical indicators, and red flags the LLM providers see -- no
  network call, no model, no randomness. It runs a fixed ten-point checklist
  (profitability, margin level and trend, revenue/earnings growth,
  liquidity, leverage, cash-flow quality, and shareholder returns), derives
  a cyclicality read from the volatility of year-over-year revenue growth,
  and estimates a per-share fair-value range with a two-stage discounted
  cash-flow model (FCF/share, or EPS as a fallback anchor, projected at a
  historical growth rate and discounted back) run three times -- bear, base,
  and bull -- at different growth/discount-rate assumptions, matching the
  same `fair_value_range` schema the LLM providers return. The result
  includes a `score` field listing every checklist item by name with a
  pass/fail/n/a and the exact figures behind it, so the verdict is fully
  auditable -- see `sec_analyzer/interpret/rule_based.py`.
* **Who it's for.** Anyone who wants a quick, reproducible fundamentals
  read without an API key, without running Ollama, and fully offline
  (beyond the SEC EDGAR fetch itself).
* **How to use it.** CLI: `python -m sec_analyzer.cli analyze AAPL --provider script`.
  Web UI: select **Script (no AI · deterministic)** from the provider dropdown.
* **What it isn't.** The fair-value band is a heuristic screen based only on
  the figures provided, not a price target or a personalized investment
  recommendation.

## Known edge cases

* **ADR / 20-F filers (IFRS reporters):** Foreign private issuers filing
  Form 20-F typically report under the `ifrs-full` XBRL taxonomy instead of
  `us-gaap`. Since this tool only extracts `us-gaap` concepts, such filers
  will show every concept as `missing` rather than raising an error --
  `analyze` will still run and will call out the data gap in its summary.
* **Restatements:** When the same fiscal period appears in more than one
  filing (a correction/restatement), the row with the most recently *filed*
  date wins.
* **Fiscal year labeling:** The fiscal year for each record is derived from
  its `period_end` date (calendar year of the period end), not from the raw
  `fy` field SEC stamps on the fact -- SEC stamps a filing's *comparative*
  prior-year columns with the filing's own `fy`, which would otherwise
  mislabel and collapse distinct years onto one label.
* **Insufficient data:** If a filer has too little usable annual data (e.g.
  a recent IPO with only one or two 10-Ks on file), `compute_ratios` may
  return an empty or partial list, and `analyze` is instructed to report
  that uncertainty and return `null` fair-value bounds rather than
  guessing.

## Not investment advice

Output from the `analyze` command is educational commentary on public SEC
filings, generated from a limited set of historical figures. It is **not**
personalized investment advice, and should not be treated as a
recommendation to buy, hold, or sell any security.

## Web UI

A small Flask front end wraps the same fetch/normalize/store/interpret
pipeline as the CLI: type a ticker, click **Get Earnings** for financials
straight from SEC EDGAR, or **Analyze with LLM** to also get a full
fundamental + technical read (fair-value bands, verdict chips, red flags,
catalyst, and summary -- the same information the CLI's terminal verdict
card and `--html` report show).

```powershell
python -m sec_analyzer.web.app
```

This serves the UI at **http://127.0.0.1:5050**. As with the CLI,
`SEC_USER_AGENT` must be set (e.g. in `.env`) before fetching anything.

The page has a **Horizon** dropdown (3m / 1y / 5y, default 1y -- see
"Investment horizon (`--horizon`)" above) and a provider selector for the
analysis step:

* **Script (no AI · deterministic)** -- the deterministic rule-based
  screen described above; no API key or local model required.
* **Gemma (local · Ollama)** -- the default. Requires
  [Ollama](https://ollama.com) running locally with the model pulled:
  ```powershell
  ollama pull gemma4:latest
  ```
* **Claude (Anthropic)** -- requires `ANTHROPIC_API_KEY` to be set.

Note the web UI does not currently offer the CLI's `--html` standalone
report; use the CLI for that.

## Tests

```powershell
pytest sec_analyzer/tests
```
