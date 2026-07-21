# Damodaran sektör verisi (`data/damodaran/`)

`sec_analyzer`'ın değerleme motoru (`valuation/damodaran.py`), sektör
karşılaştırması ve sektör bazlı özkaynak risk primi (ERP) için isteğe bağlı
olarak bu klasördeki iki CSV dosyasını okur. Klasörün yolu
`Config.DAMODARAN_DIR` ile yapılandırılır (varsayılan: `<çalışma
dizini>/data/damodaran`, yani projeyi kökünden çalıştırıyorsanız tam olarak
bu klasör — `DAMODARAN_DIR` ortam değişkeniyle override edilebilir).

## Dosyalar

### `multiples.csv`

Sektör bazında **medyan** fiyat/değer çoklularını içerir. Kolonlar (bu
sıra ve bu ad, başlık satırı dahil):

```
industry,pe,ps,pfcf
Semiconductor,28.4,6.1,24.7
Software (Internet),35.2,7.8,30.1
Retail (General),18.6,1.1,17.2
```

- `industry` — sektör/endüstri adı (Damodaran'ın kendi endüstri
  etiketiyle aynı olması gerekmez; şirketin SEC `sicDescription`'ı ile
  büyük/küçük harf duyarsız alt-dize/anahtar kelime örtüşmesiyle
  eşleştirilir — bkz. "Eşleştirme nasıl çalışır" aşağıda).
- `pe`, `ps`, `pfcf` — o sektörün medyan Fiyat/Kazanç, Fiyat/Satış,
  Fiyat/Serbest Nakit Akışı çoklusu (ondalık sayı, örn. `28.4`, `%` işareti
  veya para birimi simgesi olmadan).
- `growth` ve/veya `peg` — **isteğe bağlı** kolonlar; yalnızca sektör medyan
  PEG karşılaştırması için kullanılır (VALUATION.md §7). `growth`, sektörün
  beklenen çok-yıllı büyümesidir ve **oran** olarak yazılır (örn. `%15` büyüme
  → `0.15`); motor sektör PEG'ini `pe / (growth × 100)` ile türetir. Doğrudan
  `peg` kolonu verilirse o kullanılır. Her ikisi de yoksa (mevcut dört-kolonlu
  format) sektör PEG'i boş (`None`) geçilir, başka hiçbir şey etkilenmez. Örnek
  başlık: `industry,pe,ps,pfcf,growth`.
- `unlevered_beta` — **isteğe bağlı** kolon; sektörün **kaldıraçsız (asset)
  betası**, düz bir sayı olarak (örn. `1.5`, `%` veya oran değil). CAPM
  özkaynak maliyeti iskonto oranını besler (`valuation/capm.py`,
  VALUATION.md §4): motor bunu şirketin kendi piyasa D/E'si ve marjinal
  vergiyle relever edip `cost_of_equity = risk_free + β_L × ERP`'i türetir.
  Yoksa CAPM devre dışı kalır ve iskonto oranı eski düz sektör-bağımsız
  varsayılana döner (hata vermez). Örnek başlık:
  `industry,pe,ps,pfcf,unlevered_beta`.
- `capex_sales` — **isteğe bağlı** kolon; sektörün bakım (maintenance)
  yatırım harcamasının satışlara oranı, **ondalık oran** olarak (örn.
  `0.045` = satışların %4.5'i; `%` işareti veya yüzde formatı değil).
  Kavramsal olarak Damodaran'ın **"Capital Expenditures by Sector (US)"**
  (Cap Ex/Sales) tablosuna dayanır. Değerleme motoru (`valuation/engine.py`)
  bakım capex'i hesaplarken bu oranı, motorun düz `%5` bakım-capex tabanı
  yerine kullanır (`_MAINTENANCE_CAPEX_MIN_PCT_REVENUE = 0.05`); kolon
  boş/yoksa bu düz `%5` tabanına geri döner (hata vermez). Banka/sigorta/
  finansal hizmetler gibi sektörlerde capex/satış anlamlı bir metrik
  olmadığından bu satırlarda kolon **bilerek boş** bırakılabilir. Örnek
  başlık: `industry,pe,ps,pfcf,unlevered_beta,capex_sales`.

### `erp.csv`

Bölge bazında özkaynak risk primini (Equity Risk Premium) içerir. Kolonlar:

```
region,erp,risk_free
US,4.23,4.20
Europe,5.1,
Emerging Markets,6.8,
```

Motor **sadece `region == "US"` satırını** kullanır; diğer bölge satırları
şimdilik yalnızca referans/gelecekteki kullanım içindir, göz ardı edilir.
`erp` ondalık bir yüzde sayısıdır (örn. `4.6` = %4.6), oran (`0.046`)
değil.

- `risk_free` — **isteğe bağlı** kolon; risksiz getiri oranı, `erp` ile aynı
  yüzde formatında (örn. `4.20` = %4.2, oran değil). CAPM özkaynak maliyetinin
  sabit terimidir (`valuation/capm.py`). Yalnızca US satırı okunur. Yoksa (eski
  `region,erp` formatı) CAPM devre dışı kalır ve iskonto oranı düz varsayılana
  döner.

### `erp_history.csv` (geçmiş tarih / as-of modu için)

`analyze TICKER --as-of YYYY-MM-DD` (SPEC.md §18) motoru geçmiş bir tarihte
bilinebilecek verilerle çalıştırır; bunun makro (ERP/risksiz getiri) ayağını
besleyen dosya budur — `erp.csv`'nin tek satırlık "güncel" değerinin aksine,
**yıl bazında** bir geçmiş serisi tutar. Kolonlar:

```
year,erp,risk_free
2008,4.37,4.02
2009,6.43,2.21
...
2025,4.33,4.58
```

- `year` — tam sayı yıl (dört haneli).
- `erp` — o yılın **implied equity risk premium**'u, `erp.csv` ile aynı yüzde
  formatında (örn. `4.37` = %4.37, oran değil).
- `risk_free` — **isteğe bağlı** kolon; o yılın risksiz getiri oranı, aynı
  yüzde formatında. Boş bırakılabilir — motor bu durumda FRED'in geçmiş
  DGS10 serisine (`fetch/fred.py`, gerçek işlem günü bazında) veya `erp.csv`'nin
  güncel değerine geri döner (öncelik sırası: FRED -> bu dosyanın `risk_free`
  hücresi -> `erp.csv`'nin güncel değeri; SPEC.md §18).

**Motor bunu nasıl kullanır:** `--as-of` verildiğinde, motor `as_of` tarihinin
YILINA denk gelen satırı okur; o yıl bu dosyada yoksa (veya `erp` hücresi
boşsa) `erp.csv`'nin güncel değerine geri döner — hiçbir zaman hata vermez.
Hangi kaynağın kullanıldığı raporun `valuation["macro_asof"]` alanında ve bir
Türkçe notta şeffaf biçimde belirtilir.

**Elle doldurma yöntemi (operatör sorumluluğu):** Bu dosya Damodaran'ın
`histimpl.xls` ("Implied Equity Risk Premiums (by year)", S&P 500) dosyasından,
her yıl için **yıl BAŞI (start-of-year) implied ERP** değeri elle satır satır
okunup girilerek doldurulur (dosyadaki "Implied ERP (FCFE)" kolonu — `erp.csv`
için kullanılan kolonla aynı tanım, farklı yıl). `risk_free` kolonu da aynı
sayfadaki ilgili yılın risksiz getiri satırından elle doldurulabilir; boş
bırakılırsa yukarıdaki FRED geri dönüşü devreye girer.

**Hazır bir başlangıç dosyası zaten mevcut:** Bu klasörde 2008-2025 yıllarını
kapsayan dolu bir `erp_history.csv` bulunuyor. Bu bir "tohum" (seed) veridir —
`unlevered_beta`/`capex_sales` gibi, `histimpl.xls`'ten satır satır birebir
alınmış olsa da, yılda bir Damodaran'ın güncellemesiyle **doğrulanmalı ve
tazelenmelidir** (özellikle en güncel yıl için, `histimpl.xls`'in kendi
yıllık güncellemesi tamamlandıktan sonra). Yenilenene kadar as-of hesaplamaları
yön olarak doğru ama en yeni yıl için yaklaşık kalabilir.

## Şu an bu klasörde bulunan verinin kökeni

Bu klasördeki `multiples.csv` (94 sektör satırı) ve `erp.csv`, Aswath
Damodaran'ın NYU Stern veri setlerinden ([pages.stern.nyu.edu/~adamodar/pc/datasets/](https://pages.stern.nyu.edu/~adamodar/pc/datasets/))
indirilip dönüştürüldü:

- `pe` kolonu **`pedata.xls`** ("Price Earnings multiples", "Industry
  Averages" sayfası) dosyasındaki **"Aggregate Mkt Cap/ Trailing Net
  Income (only money making firms)"** kolonundan alındı. Bu kolon, basit
  "Current PE"/"Trailing PE" yerine tercih edildi çünkü zarar eden
  şirketleri dışarıda bırakıyor — onlar dahil edilseydi sektörü temsil eden
  PE çarpıtılırdı; dolayısıyla en sağlam sektör-PE temsilcisi bu kolon.
- `ps` kolonu **`psdata.xls`** ("Revenue multiples", "Industry Averages"
  sayfası) dosyasındaki **"Price/Sales"** kolonundan alındı.
- `erp.csv`'deki değer **`histimpl.xls`** ("Implied Equity Risk Premiums
  (by year)", S&P 500) dosyasındaki en güncel yıllık satırın **"Implied
  ERP (FCFE)"** kolonundan geliyor: 2025 sonu için `0.0423`, bu dosyanın
  beklediği yüzde formatına çevrilerek `4.23` olarak yazıldı (oran değil).
- `pfcf` kolonu her satırda **bilerek boş** bırakıldı: Damodaran sektör
  bazında bir Fiyat/Serbest Nakit Akışı tablosu yayınlamıyor; veri
  uydurmak yerine kolon boş bırakıldı ve yükleyici (`valuation/damodaran.py`)
  bunu `None` olarak ele alıyor — tek etkisi P/FCF sektör karşılaştırmasının
  o şirket için mevcut olmaması, başka hiçbir şeyi etkilemiyor.
- `unlevered_beta` kolonu ve `erp.csv`'deki `risk_free` değeri, CAPM iskonto
  oranı için eklendi. **ÖNEMLİ — bu iki alan yaklaşık "tohum" (seed)
  değerlerdir**, `pe`/`ps` gibi belirli bir Damodaran anlık görüntüsünden
  birebir alınmış değildir: sektör betaları temsili sektör aralıklarına göre
  elle atandı, `risk_free` yaklaşık güncel US 10-yıllık tahvil getirisine göre
  konuldu. Gerçek değerler için Damodaran'ın **"Betas by Sector (US)"**
  (`betas.xls`, kaldıraçsız/unlevered beta kolonu) ve ERP sayfasındaki
  risksiz getiri satırından, `pe`/`ps` ile aynı yıllık güncellemede
  yenilenmelidir. Yenilenene kadar CAPM çıktısı yön olarak doğru ama
  kalibrasyonu yaklaşıktır.
- `capex_sales` kolonu da, `unlevered_beta` gibi, **yaklaşık "tohum" (seed)
  veridir** — Damodaran'ın "Capital Expenditures by Sector (US)" tablosundan
  birebir, satır satır alınmamıştır; temsili sektör aralıklarına göre elle
  yaklaşık atanmıştır. Gerçek değerler için Damodaran'ın ilgili Cap Ex/Sales
  tablosundan, `pe`/`ps` ile aynı yıllık güncellemede yenilenmelidir. Banka,
  sigorta ve finansal hizmetler gibi sektör satırlarında bu kolon **bilerek
  boş** bırakılmıştır, çünkü capex/satış oranı finansal şirketler için
  anlamlı bir metrik değildir; bu satırlarda motor düz `%5` bakım-capex
  tabanına geri döner.
- Damodaran dosyalarının veri tarihi: `pedata.xls`/`psdata.xls`'te
  belirtilen **"Date updated": 2026-01-05**; `histimpl.xls`'teki en son
  yıllık satır 2025 sonuna ait.
- `pedata.xls`/`psdata.xls`'teki **"Total Market" / "Total Market (without
  financials)"** toplu satırları alınmadı — bunlar sektör değil, piyasa
  geneli agregeleridir.

Ayrıca: `valuation/damodaran.py`'deki SIC açıklaması -> Damodaran endüstri
adı eşleştirmesi, açık bir alias tablosu ve noktalama-normalize eden bir
fuzzy fallback ile güçlendirildi; örn. "SEMICONDUCTORS & RELATED DEVICES"
artık doğru şekilde "Semiconductor"a eşleniyor. Ayrıntılar için kod yorumları
kaynak kabul edilmeli.

Aşağıdaki bölümler, bu dosyaları **gelecekte** güncellerken izlenecek genel
format/dönüştürme kurallarını anlatır.

## Kaynak ve güncelleme

Kaynak: [Aswath Damodaran'ın NYU Stern sayfası](https://pages.stern.nyu.edu/~adamodar/),
**"Data"** bölümü:

- `multiples.csv` için: **"Price and Value to Book Ratios"** / **"PE
  Ratios"** / **"Price/Sales Ratios"** vb. sektör bazlı Excel tablolarından
  (industry medyanları) `pe`, `ps`, `pfcf` medyan kolonlarını derleyin.
  `unlevered_beta` için **"Betas by Sector (US)"** (`betas.xls`)
  dosyasındaki kaldıraçsız/unlevered beta kolonunu kullanın.
- `erp.csv` için: **"Equity Risk Premiums"** sayfasındaki güncel ülke/bölge
  risk primi tablosu (`erp`) ve aynı sayfadaki risksiz getiri oranı
  (`risk_free`).

Bu veriler Damodaran tarafından **yılda bir** (genelde yıl başında)
güncellenir; bu klasördeki CSV'leri de yılda bir güncellemeniz yeterlidir
— her `analyze` çalıştırmasında yeniden indirmeye gerek yoktur.

### Excel → CSV dönüştürme

Damodaran'ın sayfası verileri `.xls`/`.xlsx` olarak sunar. Adımlar:

1. İlgili Excel dosyasını indirin (örn. sektör bazlı P/E tablosu için
   "pedata.xls" benzeri bir dosya, ERP için "histimpl.xls"/ülke risk primi
   tablosu).
2. Excel/LibreOffice Calc/Google Sheets'te açın, ilgili sektör medyan
   satırlarını (`multiples.csv` için) veya ülke/bölge risk primi satırlarını
   (`erp.csv` için) seçin.
3. Yalnızca yukarıdaki kolon adlarını (`industry,pe,ps,pfcf` veya
   `region,erp`) başlık satırı yaparak, virgülle ayrılmış düz metin olarak
   **"CSV olarak kaydet / Dosyayı Farklı Kaydet → CSV (UTF-8)"** ile bu
   klasöre `multiples.csv` / `erp.csv` adıyla kaydedin.
4. Sayısal hücrelerde `%` işareti, para birimi simgesi veya binlik ayırıcı
   (`,`) bırakmayın — sadece düz ondalık sayı (`28.4`), aksi halde
   yükleyici o satırı okunamaz kabul eder.

## Eşleştirme nasıl çalışır

`sector_medians(sector_data, sic_description)`, şirketin SEC
`sicDescription`'ını `multiples.csv`'deki `industry` sütunuyla büyük/küçük
harf duyarsız alt-dize/anahtar kelime örtüşmesiyle eşleştirir. Eşleşme
bulunamazsa sektör karşılaştırması o şirket için `None` döner (hata
vermez).

## Dosyalar eksikse ne olur

Bu klasör veya içindeki dosyalar **isteğe bağlıdır**. `multiples.csv`
ve/veya `erp.csv` yoksa, bozuksa, ya da beklenen kolonlardan biri eksikse:

- Araç **çalışmaya devam eder** — hiçbir komut hata vermez.
- Sadece şu iki parça devre dışı kalır: sektör çoklu karşılaştırması
  (`valuation.multiples.sector.available = false`) ve sektör bazlı ERP
  tabanı. Bu durum çalışma zamanında **loglanır** (hangi dosyanın/kolonun
  eksik olduğu belirtilerek), sessizce yutulmaz.
- Şirkete özgü tarihsel çoklu yüzdelikleri (P/E, P/S, P/FCF'nin kendi 5-10
  yıllık geçmişine göre yüzdelik konumu) ve DCF/reverse DCF/triangülasyon
  etkilenmez — bunlar Damodaran verisine bağlı değildir.
