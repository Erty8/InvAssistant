# Casual + Educational Redesign — Report Card & Web UI

Goal: someone who knows almost nothing about investing can open the page, type
a ticker, and immediately understand "is this stock cheap, fair, or expensive
right now, and why — in plain words." Anyone who wants the full glass-box
detail can expand it. Every piece of jargon is explained. Tone: warm, plain
Turkish, reassuring, never condescending. This is educational, not advice.

This changes the SHARED verdict card (`report/template.html` + `generator.py`),
so both the web `/report` view and the CLI `--html` file benefit. The terminal
(text) card stays as-is. The web landing page (`index.html`) gets a lighter,
friendlier framing too.

## Principle: progressive disclosure (three layers)

**Layer 1 — "Özet" (default, always visible, zero jargon).** A friendly hero
block at the very top:
- One big, color-coded plain verdict in everyday words. Map the fundamental
  verdict: UCUZ → "Şu an ucuz görünüyor" (green), MAKUL → "Fiyatı makul
  görünüyor" (amber), PAHALI → "Şu an pahalı görünüyor" (red). Never show the
  raw word DCF/multiples here.
- One plain sentence built deterministically from the numbers, e.g.:
  "Hissenin fiyatı bugün **$315**. Şirketin işine bakarak tahmin ettiğimiz
  makul değer aralığı **$110–$135**. Yani fiyat, tahmini değerin belirgin
  şekilde üstünde." (adjust the last clause for üstünde/altında/civarında).
- A tiny plain-language "peki bu ne demek?" line explaining that expensive
  doesn't automatically mean "satılmalı" and cheap doesn't mean "alınmalı" —
  it's one input, and the estimate depends on assumptions you can inspect
  below. Keep it short and non-advice.
- A soft confidence chip in plain words: YÜKSEK → "Farklı yöntemler aynı yöne
  işaret ediyor (güçlü sinyal)", ORTA → "Yöntemler kısmen hemfikir", DÜŞÜK →
  "Yöntemler dağınık, temkinli ol".
- A one-line, gentle disclaimer: "Bu bir eğitim aracı; yatırım tavsiyesi
  değildir."
All of Layer 1 must be derivable from existing result fields (fundamental
verdict, price, fair_value_range.base, confidence) — no new LLM field, fully
deterministic and None-safe. If fair value is missing, say so plainly
("Yeterli veri olmadığı için değer tahmini yapılamadı") instead of blanks.

**Layer 2 — the three verdicts, plain-labeled (visible by default).** Keep the
existing Fundamental / Teknik / Profil gauge boxes, but relabel for beginners:
- Fundamental → "Değer (şirketin işine göre)"
- Teknik → "Momentum (fiyatın son hareketi)"
- Profil → "Sana uygunluk"
Keep the original term in small muted text under the plain label (so learners
connect the two), each with a "?" tooltip (see educational layer).

**Layer 3 — "Teknik detaylar" (collapsed by default, one click to expand).**
Everything that is glass-box-but-jargon-heavy goes inside a collapsible
`<details>`-style section, collapsed on load: the fan chart with its scenario
assumptions, reverse-DCF line, multiples percentile, triangulation row, and the
sensitivity table. A clear toggle: "🔍 Teknik detayları göster / gizle".
Nothing is removed — the current detailed card lives here intact.

## Educational layer (tooltips + glossary)

- Every jargon term anywhere on the card gets an inline info affordance: a
  superscript "?" or ⓘ that on hover (desktop) AND tap/click (mobile) reveals a
  1–2 sentence plain-Turkish definition. Implement with a small self-contained
  JS tooltip (or accessible `<details>`/popover) — no external libs. Terms to
  cover at minimum: adil değer (fair value), ucuz/makul/pahalı, iskonto oranı
  (discount rate), büyüme varsayımı, terminal büyüme, DCF, ters/reverse DCF,
  ima edilen büyüme, çarpan/multiples, P/E, P/S, P/FCF, yüzdelik (percentile),
  P/B×ROE, üçgenleme (triangulation), güven, duyarlılık (sensitivity), RSI,
  SMA50/SMA200, momentum, katalizör, red flag (uyarı işareti), bear/base/bull
  senaryo.
- Add a collapsed "📚 Kavramlar sözlüğü" section at the bottom listing the same
  terms with their plain definitions, so a learner can read them all in one
  place. Definitions written for a total beginner, with a tiny concrete example
  where it helps (e.g. P/E: "Şirketin fiyatının, yıllık kârının kaç katı
  olduğunu gösterir. Örn. P/E 20 ise, şirketi 20 yıllık kârı kadar bir fiyata
  alıyorsun demektir.").
- Definitions are static Turkish content embedded in the template — a single
  JS object mapping term→definition, reused by both tooltips and the glossary.

## Tone & visual

- Keep the existing dark navy theme, colors, and self-contained rule (inline
  CSS/JS, no external requests, works from file://, mobile single-column).
- Layer 1 hero should feel calm and friendly: big readable type, a soft
  colored status pill, generous spacing. Avoid walls of numbers up top.
- Use plain Turkish throughout Layer 1/2 labels; keep technical terms only in
  Layer 3 and in the muted sub-labels (each with a "?").
- Everything None-safe; missing data degrades to a plain sentence, never
  "None"/"NaN"/blank gauges.

## index.html (landing page) — lighter framing

- Add a short, friendly one-line intro under the title, e.g. "Bir hisse kodu
  yaz, şirketin ucuz mu pahalı mı göründüğünü sade bir dille anlatalım."
- Make "View Full Report" the clear PRIMARY, most prominent button (this is the
  beginner path). Keep the existing "Get Earnings"/"Analyze with LLM" buttons
  but visually secondary (they're the power-user paths).
- Default the report to a beginner-safe setup (provider script is fine as
  default so it works with no API key). A short "?" next to the horizon
  selector explaining 3m/1y/5y in one plain line each.
- Don't break the existing financials-explorer functionality.

## Constraints

- Keep `generate_report(...)` and `render_report_html(...)` signatures
  unchanged; all new content is derived inside the template from the data
  already passed. No interpret/valuation contract changes, no new deps.
- Keep the existing structural markers the tests grep for (fan-band,
  triangulation-row, sensitivity-table) present (they now live inside Layer 3).
- Full suite stays green; add tests asserting the new Layer-1 summary text and
  the glossary render, and that Layer 3 markers still exist.
