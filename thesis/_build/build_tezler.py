"""Build thesis/tezler.xlsx from the four 2026-10-26 theses."""
import sys

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

OUT = sys.argv[1]
FONT = "Arial"
BLUE = Font(name=FONT, color="0000FF")
BLACK = Font(name=FONT, color="000000")
GREEN = Font(name=FONT, color="008000")
HEAD = Font(name=FONT, bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
TITLE = Font(name=FONT, bold=True, size=14)
WRAP = Alignment(wrap_text=True, vertical="top")
TOP = Alignment(vertical="top")
THIN = Border(bottom=Side(style="thin", color="D9D9D9"))
USD = '$#,##0.00;($#,##0.00);"-"'
PCT = '0.0%;(0.0%);"-"'
MULT = '0.0"x"'

SRC = "Kaynak: sec_analyzer analyze <TICKER> --no-ai --horizon 1y, 2026-10-26 (SEC EDGAR + fiyat verisi)."

# ---------------------------------------------------------------------------
# Data (inputs, blue)
# ---------------------------------------------------------------------------
THESES = [
    dict(t="MU", name="Micron Technology", date="2026-10-26", price=877.55, eps=74.33,
         eps_note="FY2026 EPS (mali yıl 2026-09-03'te bitti; son 12 ay = mali yıl).",
         growth=3.793, margin=0.638, stance="Temkinli",
         line="Yapısal hikâye gerçek ama $877 fiyatı döngü boyunca %50+ net marjı varsayıyor; iyi fiyatlanmış, doğrulama bekliyor.",
         verdict="FAIR", lo=756.80, hi=1275.89, ref=598.0,
         ref_note="Döngü analizi, yapısal kırılma rejimi orta değeri ($133B gelir kalıcı, %38 net marj, 13.5x F/K).",
         analyst=1561.91, catalyst="Q1 FY2027 sonuçları ~30 Ara 2026",
         risk="Bellek döngüsünün dönmesi; capex bir yılda 2 katına çıktı ($30.7B)."),
    dict(t="NVDA", name="NVIDIA", date="2026-10-26", price=210.96, eps=7.87,
         eps_note="Son 12 ay EPS (2026-07-26'ya kadar).",
         growth=1.059, margin=0.637, stance="Koşullu olumlu",
         line="Hızlanarak büyüyor ve son 12 aya göre 27x F/K pahalı değil; ama marj zirvede ve nakit dönüşümü zayıflıyor.",
         verdict="FAIR", lo=164.48, hi=277.36, ref=235.0,
         ref_note="Tezdeki 'büyüme yavaşlar' senaryosu: 12 ay EPS $10.2 x 23 F/K.",
         analyst=328.95, catalyst="Q3 FY2027 sonuçları ~25 Kas 2026",
         risk="Hiperscaler yatırımlarının yavaşlaması; alacaklar 2 çeyrekte +%64."),
    dict(t="NFLX", name="Netflix", date="2026-10-26", price=65.85, eps=2.70,
         eps_note="Son 12 ay EPS, $2.8B WBD fesih ücreti hariç (motorun gösterdiği $3.23 ücret dahil).",
         growth=0.134, margin=0.282, stance="Nötr, hafif olumlu",
         line="12 yılın en düşük çarpanlarında kaliteli şirket; ama büyüme yavaşlıyor ve motorun FAIR kararı tek seferlik ödemeyle şişmiş.",
         verdict="FAIR", lo=51.07, hi=91.04, ref=52.0,
         ref_note="Motor DCF'si, $2.8B fesih ücreti son 12 ay FCF'sinden çıkarılarak ölçeklendi (~$7.8B FCF).",
         analyst=84.83, catalyst="FY2026 sonuçları ~19 Oca 2027",
         dcf_note="Motor DCF'si $2.8B fesih ücretini içerir; ücret hariç ~$37–$67 (orta ~$52).",
         risk="Gelir büyümesinin %17'den %13'e yavaşlaması; brüt marj kötüleşiyor."),
    dict(t="APP", name="AppLovin", date="2026-10-26", price=260.25, eps=12.90,
         eps_note="Son 12 ay EPS (2026-06-30'a kadar).",
         growth=0.528, margin=0.646, stance="Olumlu, belirsizlik yüksek",
         line="Yılda %50 büyüyen, son 12 aya göre 20x F/K'lı çok kârlı şirket; ama 6 ayda %42 düşüşün nedeni veride görünmüyor.",
         verdict="CHEAP", lo=460.31, hi=867.10, ref=350.0,
         ref_note="Tezdeki 'büyüme ~%25'e yavaşlar' senaryosu: 12 ay EPS $17.5 x 20 F/K.",
         analyst=471.69, catalyst="Q3 2026 sonuçları ~4 Kas 2026",
         risk="Düşüşün açıklanmamış nedeni (düzenleyici/hukuki olabilir); içeriden ~$197M satış."),
    dict(t="UBER", name="Uber Technologies", date="2026-10-26", price=66.70, eps=2.45,
         eps_note="Temiz son 12 ay EPS tahmini: faaliyet kârı $6.69B x (1 - %21 vergi) ≈ $5B / 2.04B hisse. "
                  "Raporlanan $4.65B, Q3 2025'teki $4.05B tek seferlik vergi gelirini ve yatırım değerlemelerini içerir.",
         growth=0.122, margin=0.091, stance="Olumlu, satın alma riski",
         line="Piyasa neredeyse sıfır büyüme fiyatlıyor, FCF getirisi %6 ve faaliyet kârı +%30; ama Delivery Hero alımı borcu büyütecek ve büyüme yavaşlıyor.",
         verdict="MODEL-PRICE DIVERGENCE", lo=186.67, hi=355.52,
         dcf_note="Motor DCF'si FCF'yi yıllık %20–30 büyütüyor; gelir %12 büyürken iyimser. Ters DCF: fiyat 5 yılda %1.2 büyüme ima ediyor.",
         analyst=100.86, catalyst="Q3 2026 sonuçları ~4 Kas 2026; Delivery Hero teklifi",
         risk="Delivery Hero alımı (€14.2B finansman, €700M ceza riski); büyüme %20'den %12'ye; otonom araç riski."),
]

SCENARIOS = {
    "MU": [("Yapısal kırılma, düşük", 34.96, 12.0), ("Yapısal kırılma, orta", 44.28, 13.5),
           ("Yapısal kırılma, yüksek", 52.44, 15.0)],
    "NVDA": [("Büyüme durur", 8.90, 20.0), ("Büyüme yavaşlar", 10.20, 23.0), ("Büyüme sürer", 11.50, 25.0)],
    "NFLX": [("Büyüme %8'e iner", 3.30, 18.0), ("Büyüme ~%13 sürer", 3.55, 22.0),
             ("Büyüme yeniden hızlanır", 3.75, 26.0)],
    "APP": [("Büyüme durur / düzenleyici darbe", 15.0, 15.0), ("Büyüme ~%25'e yavaşlar", 17.5, 20.0),
            ("Büyüme ~%40 sürer", 20.0, 25.0)],
    "UBER": [("Büyüme yavaşlar, satın alma sorunlu", 2.60, 20.0), ("Marj genişlemesi sürer", 3.00, 25.0),
             ("Marj ve büyüme birlikte güçlenir", 3.30, 30.0)],
}
SCENARIO_NOTE = {
    "MU": "MU senaryoları motorun döngü analizinden (rejim B: $133B gelir x %30/38/45 net marj, 12/13.5/15x F/K).",
}

INDICATORS = {
    "MU": [("Net marj (çeyreklik)", "%69.5", "%65'in üzerinde kalır", "2 çeyrek üst üste düşer"),
           ("Gelir büyümesi (çeyreklik, yıllık)", "%379", "Ardışık büyür", "Ardışık daralır"),
           ("Capex / gelir", "%23", "Sabit kalır veya düşer", "Daha da artar"),
           ("İçeriden işlemler", "0 alım / 261 satış", "Açık piyasadan alım", "Satışlar hızlanır"),
           ("Fiyat", "$877.55", "$1,012 üstünde kalıcı kapanış", "$736 desteği kırılır")],
    "NVDA": [("Gelir büyümesi (çeyreklik, yıllık)", "%106", "%50'nin üzerinde kalır", "Ardışık iki çeyrek sert yavaşlar"),
             ("Net marj (son 12 ay)", "%63.7", "%60'ın üzerinde kalır", "2 çeyrek üst üste düşer"),
             ("Alacaklar / gelir", "Tahsil süresi ~60 gün", "Normale döner", "Artmaya devam eder"),
             ("FCF / net kâr", "%62", "Yükselir", "Düşmeye devam eder"),
             ("Fiyat", "$210.96", "$244 üstünde kalıcı kapanış", "$189 desteği kırılır")],
    "NFLX": [("Gelir büyümesi (çeyreklik, yıllık)", "%13.4", "%13'te kalır veya yükselir", "%10'un altına iner"),
             ("Faaliyet marjı (yıllık bazda)", "%33.4 (Q2 2026)", "Artar", "Düz kalır veya düşer"),
             ("Faaliyet nakit akışı (tek seferlik hariç)", "$1.74B (Q2 2026)", "Çeyreklik $2.5B+", "$1.7B seviyesi sürer"),
             ("Fiyat", "$65.85", "$73–75 direnci aşılır", "$64 desteği kırılır")],
    "APP": [("Gelir büyümesi (çeyreklik, yıllık)", "%53", "%40'ın üzerinde kalır", "%30'un altına iner"),
            ("Faaliyet marjı", "%77", "%70'in üzerinde", "Düşüş başlar"),
            ("Düzenleyici veya hukuki 8-K", "Yok", "Yok", "Soruşturma veya dava bildirimi"),
            ("İçeriden işlemler", "0 alım / 211 satış", "Alım gelir", "Satışlar hızlanır"),
            ("Fiyat", "$260.25", "$300–308 direnci aşılır", "$245 desteği kırılır")],
    "UBER": [("Gelir büyümesi (çeyreklik, yıllık)", "%12.2", "%12 ve üstünde kalır", "%10'un altına iner"),
             ("Faaliyet marjı", "%13.3 (Q2 2026)", "Artmaya devam eder", "Düşer"),
             ("FCF (SBC sonrası, son 12 ay)", "$8.2B", "Artar", "Düşer"),
             ("Delivery Hero teklifi", "Teklif süreci açık", "Yüksek kabul, onay, net borç planı",
              "Koşullu onay, €700M ceza, kaldıraç artışı"),
             ("Yönetici işlemleri", "CEO ve A. Macdonald aldı", "Alımlar sürer", "Satışlar başlar"),
             ("Fiyat", "$66.70", "$69.5 direnci aşılır", "$59 desteği kırılır")],
}

# Further theses arrive as JSON (one object per ticker, the subagent schema).
if len(sys.argv) > 2:
    import json

    for d in json.load(open(sys.argv[2], encoding="utf-8")):
        THESES.append({
            "t": d["t"], "name": d["name"], "date": d.get("date") or "2026-10-26", "price": d["price"],
            "eps": d.get("eps"), "eps_note": d.get("eps_note"), "growth": d.get("growth"),
            "margin": d.get("margin"), "stance": d["stance"], "line": d["line"], "verdict": d["verdict"],
            "lo": d.get("lo"), "hi": d.get("hi"), "dcf_note": d.get("dcf_note"), "analyst": d.get("analyst"),
            "catalyst": d.get("catalyst"), "risk": d.get("risk"),
        })
        SCENARIOS[d["t"]] = [tuple(x) for x in d["scenarios"]]
        if d.get("scenario_note"):
            SCENARIO_NOTE[d["t"]] = d["scenario_note"]
        INDICATORS[d["t"]] = [tuple(x) for x in d["indicators"]]


def header(ws, row, labels, widths):
    for col, (label, width) in enumerate(zip(labels, widths), start=1):
        c = ws.cell(row=row, column=col, value=label)
        c.font, c.fill, c.alignment = HEAD, HEAD_FILL, Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[row].height = 32


def add_table(ws, name, ref):
    tab = Table(displayName=name, ref=ref)
    tab.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=True)
    ws.add_table(tab)


wb = Workbook()

# ---------------------------------------------------------------------------
# Özet
# ---------------------------------------------------------------------------
ws = wb.active
ws.title = "Özet"
ws["A1"] = "Yatırım Tezleri — Özet"
ws["A1"].font = TITLE
ws["A2"] = SRC + " Mavi = girdi, siyah = formül. Kişisel yatırım tavsiyesi değildir."
ws["A2"].font = Font(name=FONT, italic=True, size=9)

HR = 4
TIPS = ("Ayı", "Baz", "Boğa")
S_FIRST = HR + 1
S_LAST = HR + sum(len(v) for v in SCENARIOS.values())
last = HR + len(THESES)

# Column layout: the three targets sit right after the price, so one glance
# shows bear / base / bull against today's price.
cols = [
    ("Ticker", 9), ("Şirket", 18), ("Duruş", 16), ("Fiyat ($)", 11),
    ("Ayı hedefi ($)", 11), ("Ayı getirisi", 10), ("Baz hedefi ($)", 11), ("Baz getirisi", 10),
    ("Boğa hedefi ($)", 11), ("Boğa getirisi", 10), ("Analist ort. hedef ($)", 12), ("Analist potansiyeli", 11),
    ("Tek cümlelik tez", 60), ("Motor kararı (güncel)", 11), ("Motor DCF baz alt ($)", 12),
    ("Motor DCF baz üst ($)", 12), ("Motor DCF orta ($)", 12), ("Motor ortasına göre potansiyel", 12),
    ("EPS son 12 ay ($)", 11), ("F/K (son 12 ay)", 10), ("Son çeyrek gelir büyümesi", 12),
    ("Net marj (son 12 ay)", 11), ("Tez tarihi", 11), ("Ana katalizör", 26), ("Ana risk", 44), ("Tez dosyası", 22),
]
header(ws, HR, [c[0] for c in cols], [c[1] for c in cols])
for col, color in {5: "C00000", 6: "C00000", 7: "2F5597", 8: "2F5597", 9: "548235", 10: "548235"}.items():
    ws.cell(row=HR, column=col).fill = PatternFill("solid", fgColor=color)


def target(r, tip):
    return (f"=SUMIFS('Senaryolar'!$F${S_FIRST}:$F${S_LAST},'Senaryolar'!$A${S_FIRST}:$A${S_LAST},$A{r},"
            f"'Senaryolar'!$B${S_FIRST}:$B${S_LAST},\"{tip}\")")


def upside(r, col):
    return f"=IFERROR(IF({col}{r}=0,\"-\",{col}{r}/$D{r}-1),\"-\")"


for i, d in enumerate(THESES):
    r = HR + 1 + i
    vals = {
        1: (d["t"], BLUE), 2: (d["name"], BLUE), 3: (d["stance"], BLUE), 4: (d["price"], BLUE),
        5: (target(r, "Ayı"), GREEN), 6: (upside(r, "E"), BLACK),
        7: (target(r, "Baz"), GREEN), 8: (upside(r, "G"), BLACK),
        9: (target(r, "Boğa"), GREEN), 10: (upside(r, "I"), BLACK),
        11: (d["analyst"], BLUE), 12: (upside(r, "K"), BLACK),
        13: (d["line"], BLUE), 14: (d["verdict"], BLUE), 15: (d["lo"], BLUE), 16: (d["hi"], BLUE),
        17: (f"=IFERROR(AVERAGE(O{r}:P{r}),\"-\")", BLACK), 18: (upside(r, "Q"), BLACK),
        19: (d["eps"], BLUE), 20: (f"=IF(N(S{r})<=0,\"-\",D{r}/S{r})", BLACK), 21: (d["growth"], BLUE),
        22: (d["margin"], BLUE), 23: (d["date"], BLUE), 24: (d["catalyst"], BLUE), 25: (d["risk"], BLUE),
        26: (f"{d['t']}_{d['date']}.md", BLUE),
    }
    for col, (v, font) in vals.items():
        c = ws.cell(row=r, column=col, value=v)
        c.font, c.alignment, c.border = font, (WRAP if col in (13, 25) else TOP), THIN
    for col in (4, 5, 7, 9, 11, 15, 16, 17, 19):
        ws.cell(row=r, column=col).number_format = USD
    for col in (6, 8, 10, 12, 18, 21, 22):
        ws.cell(row=r, column=col).number_format = PCT
    ws.cell(row=r, column=20).number_format = MULT
    for col in (5, 7, 9):
        ws.cell(row=r, column=col).font = Font(name=FONT, color="008000", bold=True)
    link = ws.cell(row=r, column=26)
    link.hyperlink = f"{d['t']}_{d['date']}.md"
    link.font = Font(name=FONT, color="0000FF", underline="single")
    if d.get("eps_note"):
        ws.cell(row=r, column=19).comment = Comment(d["eps_note"], "Claude")
    if d.get("dcf_note"):
        ws.cell(row=r, column=15).comment = Comment(d["dcf_note"], "Claude")
    ws.row_dimensions[r].height = 62

# Green for upside, red for downside, on every return column.
for col in ("F", "H", "J", "L", "R"):
    rng = f"{col}{HR + 1}:{col}{last}"
    ws.conditional_formatting.add(rng, CellIsRule(
        operator="greaterThan", formula=["0"],
        font=Font(name=FONT, color="006100", bold=True), fill=PatternFill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add(rng, CellIsRule(
        operator="lessThan", formula=["0"],
        font=Font(name=FONT, color="9C0006", bold=True), fill=PatternFill("solid", fgColor="FFC7CE")))

add_table(ws, "Tezler", f"A{HR}:{get_column_letter(len(cols))}{last}")
ws.freeze_panes = f"B{HR + 1}"
ws.cell(row=last + 2, column=1, value=(
    "Notlar: Ayı / Baz / Boğa hedefleri Senaryolar sayfasından gelir (yeşil) ve 12 aylık EPS x F/K varsayımlarıdır. "
    "MU'nun üç hedefi de döngü analizinin 'yapısal kırılma' bandıdır; tez, fiyatın bu bandın bile üstünde olduğunu söyler. "
    "Motor DCF sütunları 26 Ekim 2026'daki güncel motor çıktısıdır; NFLX'inki $2.8B tek seferlik fesih ücretini hâlâ içerir."
)).font = Font(name=FONT, italic=True, size=9)

# ---------------------------------------------------------------------------
# Senaryolar
# ---------------------------------------------------------------------------
ss = wb.create_sheet("Senaryolar")
ss["A1"] = "12 Aylık Senaryolar"
ss["A1"].font = TITLE
ss["A2"] = "Hedef fiyat = Metrik x Çarpan (çoğunlukla EPS x F/K; farklıysa Senaryo hücresindeki yoruma bakın). Bugünkü fiyat Özet sayfasından gelir (yeşil). Senaryolar Claude'un varsayımlarıdır."
ss["A2"].font = Font(name=FONT, italic=True, size=9)
s_cols = [("Ticker", 9), ("Tip", 8), ("Senaryo", 34), ("Metrik, 12 ay ($)", 14), ("Çarpan", 9),
          ("Hedef fiyat ($)", 13), ("Bugünkü fiyat ($)", 13), ("Bugüne göre getiri", 12)]
header(ss, HR, [c[0] for c in s_cols], [c[1] for c in s_cols])
r = HR
for d in THESES:
    for tip, (label, eps, pe) in zip(TIPS, SCENARIOS[d["t"]]):
        r += 1
        for col, v in ((1, d["t"]), (2, tip), (3, label), (4, eps), (5, pe)):
            ss.cell(row=r, column=col, value=v).font = BLUE
        ss.cell(row=r, column=6, value=f"=D{r}*E{r}").font = BLACK
        ss.cell(row=r, column=7, value=(
            f"=INDEX('Özet'!$D${HR + 1}:$D${last},MATCH(A{r},'Özet'!$A${HR + 1}:$A${last},0))"
        )).font = GREEN
        ss.cell(row=r, column=8, value=f"=IFERROR(F{r}/G{r}-1,\"-\")").font = BLACK
        for col in (4, 6, 7):
            ss.cell(row=r, column=col).number_format = USD
        ss.cell(row=r, column=5).number_format = MULT
        ss.cell(row=r, column=8).number_format = PCT
        for col in range(1, 9):
            ss.cell(row=r, column=col).border = THIN
        if d["t"] in SCENARIO_NOTE and tip == "Ayı":  # metric definition
            ss.cell(row=r, column=3).comment = Comment(SCENARIO_NOTE[d["t"]], "Claude")
assert r == S_LAST
rng = f"H{HR + 1}:H{r}"
ss.conditional_formatting.add(rng, CellIsRule(
    operator="greaterThan", formula=["0"], font=Font(name=FONT, color="006100"),
    fill=PatternFill("solid", fgColor="C6EFCE")))
ss.conditional_formatting.add(rng, CellIsRule(
    operator="lessThan", formula=["0"], font=Font(name=FONT, color="9C0006"),
    fill=PatternFill("solid", fgColor="FFC7CE")))
add_table(ss, "Senaryolar", f"A{HR}:H{r}")
ss.freeze_panes = f"A{HR + 1}"

# ---------------------------------------------------------------------------
# Göstergeler
# ---------------------------------------------------------------------------
gs = wb.create_sheet("Göstergeler")
gs["A1"] = "Tezi Doğrulayacak / Çürütecek Göstergeler"
gs["A1"].font = TITLE
gs["A2"] = "Son değer tez tarihindeki okumadır. Yeni çeyrek geldikçe 'Son değer' sütununu güncelleyin."
gs["A2"].font = Font(name=FONT, italic=True, size=9)
g_cols = [("Ticker", 9), ("Gösterge", 36), ("Son değer", 20), ("Boğayı destekler", 32), ("Ayıyı destekler", 34)]
header(gs, HR, [c[0] for c in g_cols], [c[1] for c in g_cols])
r = HR
for d in THESES:
    for row in INDICATORS[d["t"]]:
        r += 1
        for col, v in enumerate((d["t"],) + row, start=1):
            c = gs.cell(row=r, column=col, value=v)
            c.font, c.alignment, c.border = BLUE, WRAP, THIN
add_table(gs, "Gostergeler", f"A{HR}:E{r}")
gs.freeze_panes = f"A{HR + 1}"

# ---------------------------------------------------------------------------
# Portföy (optional: argv[3] = portfolio_stats.json)
# ---------------------------------------------------------------------------
if len(sys.argv) > 3:
    import json as _json

    stats = _json.load(open(sys.argv[3], encoding="utf-8"))
    # Weights only: position amounts never reach the committed workbook.
    WEIGHTS, SECTOR = stats["weights"], stats["sector_by_ticker"]

    ps = wb.create_sheet("Portföy", 0)
    ps["A1"] = "Portföy — Pozisyonlar, Sektörler ve Risk"
    ps["A1"].font = TITLE
    ps["A2"] = ("Ağırlıklar 26 Ekim 2026 ekran görüntülerinden (mavi = girdi). Oynaklık ve beta son 252 işlem "
                "gününün günlük kapanışlarından (SPY'a göre). Senaryo getirileri Özet sayfasından gelir (yeşil).")
    ps["A2"].font = Font(name=FONT, italic=True, size=9)
    p_cols = [("Ticker", 9), ("Sektör", 34), ("Ağırlık (girdi)", 11), ("Normalize ağırlık", 10), ("Yıllık oynaklık", 10),
              ("Beta", 8), ("Ayı getirisi", 10), ("Baz getirisi", 10), ("Boğa getirisi", 10), ("Tez duruşu", 24)]
    header(ps, HR, [c[0] for c in p_cols], [c[1] for c in p_cols])
    tickers = sorted(WEIGHTS, key=lambda t: -WEIGHTS[t])
    first, plast = HR + 1, HR + len(tickers)
    oz = lambda col, r: f"=INDEX('Özet'!${col}${HR + 1}:${col}${last},MATCH($A{r},'Özet'!$A${HR + 1}:$A${last},0))"
    for i, t in enumerate(tickers):
        r = first + i
        cells = {
            1: (t, BLUE), 2: (SECTOR[t], BLUE), 3: (WEIGHTS[t], BLUE),
            4: (f"=C{r}/SUM($C${first}:$C${plast})", BLACK),
            5: (round(stats["stock"][t]["vol"], 4), BLUE), 6: (round(stats["stock"][t]["beta"], 2), BLUE),
            7: (oz("F", r), GREEN), 8: (oz("H", r), GREEN), 9: (oz("J", r), GREEN), 10: (oz("C", r), GREEN),
        }
        for col, (v, font) in cells.items():
            c = ps.cell(row=r, column=col, value=v)
            c.font, c.border, c.alignment = font, THIN, TOP
        for col in (3, 4, 5, 7, 8, 9):
            ps.cell(row=r, column=col).number_format = PCT
        ps.cell(row=r, column=6).number_format = "0.00"
    add_table(ps, "Pozisyonlar", f"A{HR}:J{plast}")
    for col in ("G", "H", "I"):
        rng = f"{col}{first}:{col}{plast}"
        ps.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0"],
                                      font=Font(name=FONT, color="006100"), fill=PatternFill("solid", fgColor="C6EFCE")))
        ps.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0"],
                                      font=Font(name=FONT, color="9C0006"), fill=PatternFill("solid", fgColor="FFC7CE")))

    # Totals / weighted figures
    tr = plast + 1
    ps.cell(row=tr, column=1, value="Toplam / ağırlıklı").font = Font(name=FONT, bold=True)
    ps.cell(row=tr, column=3, value=f"=SUM(C{first}:C{plast})").number_format = PCT
    ps.cell(row=tr, column=4, value=f"=SUM(D{first}:D{plast})").number_format = PCT
    ps.cell(row=tr, column=5, value=f"=SUMPRODUCT(D{first}:D{plast},E{first}:E{plast})").number_format = PCT
    ps.cell(row=tr, column=6, value=f"=SUMPRODUCT(D{first}:D{plast},F{first}:F{plast})").number_format = "0.00"
    for col in "GHI":
        c = ps[f"{col}{tr}"]
        c.value = f"=SUMPRODUCT($D${first}:$D${plast},{col}{first}:{col}{plast})"
        c.number_format = PCT
    for col in range(1, 11):
        ps.cell(row=tr, column=col).font = Font(name=FONT, bold=True)
    ps.cell(row=tr, column=5).comment = Comment(
        "Ağırlıklı ortalama oynaklık; çeşitlendirme etkisini içermez. Gerçek portföy oynaklığı aşağıdaki "
        "risk tablosunda (%37.5).", "Claude")

    # Sector table (SUMIF on the position table)
    sr = tr + 3
    ps.cell(row=sr - 1, column=1, value="Sektör dağılımı").font = Font(name=FONT, bold=True, size=12)
    header(ps, sr, ["Sektör", "", "Ağırlık", "Pay"], [9, 34, 11, 10])
    ps.cell(row=sr, column=1).value = "Sektör"
    sectors = sorted(set(SECTOR.values()), key=lambda s: -sum(WEIGHTS[t] for t in WEIGHTS if SECTOR[t] == s))
    for j, sec in enumerate(sectors):
        r = sr + 1 + j
        ps.cell(row=r, column=1, value=sec).font = BLUE
        ps.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        ps.cell(row=r, column=3, value=f"=SUMIF($B${first}:$B${plast},A{r},$C${first}:$C${plast})").number_format = PCT
        ps.cell(row=r, column=4, value=f"=C{r}/$C${tr}").number_format = PCT
        for col in (1, 3, 4):
            ps.cell(row=r, column=col).border = THIN
    ar = sr + 1 + len(sectors)
    ps.cell(row=ar, column=1, value="Yapay zekâ teması (NVDA, MU, AVGO, ORCL, APP, CRWV, CEG, OKLO, EOSE)").font = Font(name=FONT, italic=True)
    ps.merge_cells(start_row=ar, start_column=1, end_row=ar, end_column=2)
    ai = ["NVDA", "MU", "AVGO", "ORCL", "APP", "CRWV", "CEG", "OKLO", "EOSE"]
    ps.cell(row=ar, column=3, value="=" + "+".join(
        f"SUMIF($A${first}:$A${plast},\"{t}\",$C${first}:$C${plast})" for t in ai)).number_format = PCT
    ps.cell(row=ar, column=4, value=f"=C{ar}/$C${tr}").number_format = PCT

    # Risk table (inputs computed from daily prices)
    rr = ar + 3
    ps.cell(row=rr - 1, column=1, value="Risk (son 252 işlem günü, bugünkü ağırlıklarla)").font = Font(name=FONT, bold=True, size=12)
    header(ps, rr, ["Ölçü", "", "Portföy", "SPY"], [9, 34, 11, 9])
    risk_rows = [
        ("Yıllık oynaklık", stats["port_vol"], stats["spy_vol"], PCT),
        ("Beta (SPY'a göre, regresyon)", stats["port_beta"], 1.0, "0.00"),
        ("En büyük düşüş", stats["port_max_dd"], stats["spy_max_dd"], PCT),
        ("1 yıl getiri (varsayımsal, sabit ağırlık)", stats["port_ret_1y"], stats["spy_ret_1y"], PCT),
        ("Ortalama ikili korelasyon", stats["avg_pair_corr"], None, "0.00"),
    ]
    for k, (label, pv, sv, fmt) in enumerate(risk_rows):
        r = rr + 1 + k
        ps.cell(row=r, column=1, value=label).font = BLACK
        ps.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        for col, v in ((3, pv), (4, sv)):
            if v is not None:
                c = ps.cell(row=r, column=col, value=round(v, 4))
                c.font, c.number_format = BLUE, fmt
    note_r = rr + len(risk_rows) + 2
    ps.cell(row=note_r, column=1, value=(
        "Not: Ağırlıklar ekranda görünen 18 hisse içindir; uygulamadaki toplamın ekranda görünmeyen kısmı "
        "(muhtemelen nakit) dahil değil. Kişisel yatırım tavsiyesi değildir. Ayrıntılı yorum: PORTFOY_2026-10-26.md"
    )).font = Font(name=FONT, italic=True, size=9)
    ps.freeze_panes = f"B{HR + 1}"

# ---------------------------------------------------------------------------
# Teknik (optional: argv[4] = technical.json from technical_stats.py)
# ---------------------------------------------------------------------------
if len(sys.argv) > 4:
    import json as _json

    tech = _json.load(open(sys.argv[4], encoding="utf-8"))
    LABEL_TR = {
        "PULLBACK — NO BOUNCE YET": "Yükseliş içinde geri çekilme — henüz dönüş yok",
        "DOWNTREND — TURN CONFIRMING": "Düşüş trendi — dönüş teyit ediliyor",
        "DOWNTREND — NO TURN": "Düşüş trendi — dönüş yok",
        "DOWNTREND — EARLY TURN SIGNAL": "Düşüş trendi — erken dönüş sinyali",
        "UPTREND — MOMENTUM FADING": "Yükseliş trendi — ivme zayıflıyor",
        "UPTREND — HEALTHY": "Yükseliş trendi — sağlıklı",
    }
    VERDICT_TR = {"NEUTRAL": "Nötr", "OVERSOLD": "Aşırı satım", "OVERBOUGHT": "Aşırı alım",
                  "BULLISH": "Olumlu", "BEARISH": "Olumsuz"}
    ts = wb.create_sheet("Teknik", 2)
    ts["A1"] = "Teknik Görünüm"
    ts["A1"].font = TITLE
    ts["A2"] = ("Motorun teknik katmanı (sec_analyzer.technical), önbellekteki günlük kapanışlarla. Geriye dönük testte "
                "bu durumların 10–63 günlük getirileri tahmin etme gücü bulunmadı: trend ve risk seviyelerini okumak "
                "içindir, al/sat sinyali değildir.")
    ts["A2"].font = Font(name=FONT, italic=True, size=9)
    t_cols = [("Ticker", 9), ("Tarih", 11), ("Fiyat ($)", 10), ("Teknik karar", 12), ("Aksiyon", 9),
              ("Durum", 40), ("Momentum (0-100)", 10), ("Swing (0-100)", 9), ("RSI 14", 7),
              ("SMA50'ye uzaklık", 10), ("SMA200'e uzaklık", 10), ("52h aralıkta konum", 10),
              ("52h zirveye uzaklık", 10), ("3 ay getiri", 9), ("12 ay getiri", 9), ("3 ay SPY'a göre", 9),
              ("En yakın destek ($)", 11), ("En yakın direnç ($)", 11), ("Tetikleyici ($)", 11), ("Stop ($)", 10)]
    header(ts, HR, [c[0] for c in t_cols], [c[1] for c in t_cols])
    order = [d["t"] for d in THESES if d["t"] in tech]
    for i, t in enumerate(order):
        r = HR + 1 + i
        d = tech[t]
        pct = lambda v: None if v is None else v / 100.0
        vals = [t, d["as_of"], d["price"], VERDICT_TR.get(d["verdict"], d["verdict"]), d["action"],
                LABEL_TR.get(d["action_label"], d["action_label"]), d["momentum"], d["swing"], d["rsi14"],
                pct(d["dist_sma50_pct"]), pct(d["dist_sma200_pct"]), pct(d["range_position_pct"]),
                pct(d["dist_52w_high_pct"]), pct(d["return_3m_pct"]), pct(d["return_12m_pct"]),
                pct(d["rs_3m_pct"]), d["nearest_support"], d["nearest_resistance"], d["trigger"], d["stop"]]
        for col, v in enumerate(vals, start=1):
            c = ts.cell(row=r, column=col, value=v)
            c.font, c.border, c.alignment = BLUE, THIN, (WRAP if col == 6 else TOP)
        for col in (3, 17, 18, 19, 20):
            ts.cell(row=r, column=col).number_format = USD
        for col in range(10, 17):
            ts.cell(row=r, column=col).number_format = PCT
        ts.row_dimensions[r].height = 30
    tlast = HR + len(order)
    add_table(ts, "TeknikTablo", f"A{HR}:T{tlast}")
    action_fill = {"BUY": ("C6EFCE", "006100"), "HOLD": ("DDEBF7", "1F4E79"),
                   "WAIT": ("FFEB9C", "9C5700"), "AVOID": ("FFC7CE", "9C0006")}
    for code, (bg, fg) in action_fill.items():
        ts.conditional_formatting.add(f"E{HR + 1}:E{tlast}", CellIsRule(
            operator="equal", formula=[f'"{code}"'], font=Font(name=FONT, color=fg, bold=True),
            fill=PatternFill("solid", fgColor=bg)))
    ts.conditional_formatting.add(f"I{HR + 1}:I{tlast}", CellIsRule(
        operator="lessThan", formula=["30"], font=Font(name=FONT, color="9C0006", bold=True)))
    ts.conditional_formatting.add(f"I{HR + 1}:I{tlast}", CellIsRule(
        operator="greaterThan", formula=["70"], font=Font(name=FONT, color="006100", bold=True)))
    for col in ("J", "K", "N", "O", "P"):
        rng = f"{col}{HR + 1}:{col}{tlast}"
        ts.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0"],
                                      font=Font(name=FONT, color="006100")))
        ts.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0"],
                                      font=Font(name=FONT, color="9C0006")))
    ts.cell(row=tlast + 2, column=1, value=(
        "Aksiyon: BUY = düşüş trendinde dönüş teyit ediliyor · HOLD = yükseliş sürüyor · WAIT = sinyal yok, bekle · "
        "AVOID = düşüş trendi sürüyor, dönüş yok. Tetikleyici: motorun trend onarımı/dönüş için izlediği seviye. "
        "RSI < 30 aşırı satım (kırmızı), > 70 aşırı alım (yeşil). Kişisel yatırım tavsiyesi değildir."
    )).font = Font(name=FONT, italic=True, size=9)
    ts.freeze_panes = f"B{HR + 1}"

for sheet in wb.worksheets:
    for row in sheet.iter_rows():
        for c in row:
            if c.font is not None and c.font.name != FONT:
                c.font = Font(name=FONT, bold=c.font.bold, italic=c.font.italic, size=c.font.size,
                              color=c.font.color, underline=c.font.underline)

wb.calculation.fullCalcOnLoad = True
wb.save(OUT)
print("saved", OUT)
