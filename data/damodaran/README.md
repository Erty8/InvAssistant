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

### `erp.csv`

Bölge bazında özkaynak risk primini (Equity Risk Premium) içerir. Kolonlar:

```
region,erp
US,4.6
Europe,5.1
Emerging Markets,6.8
```

Motor **sadece `region == "US"` satırını** kullanır; diğer bölge satırları
şimdilik yalnızca referans/gelecekteki kullanım içindir, göz ardı edilir.
`erp` ondalık bir yüzde sayısıdır (örn. `4.6` = %4.6), oran (`0.046`)
değil.

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
- `erp.csv` için: **"Equity Risk Premiums"** sayfasındaki güncel ülke/bölge
  risk primi tablosu.

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
