# VALUATION.md — Değerleme Kuralları ve Damıtılmış İlkeler

> Bu dosya interpret katmanının system prompt'una METODOLOJI.md'nin ardından eklenir.
> Kaynak damıtması: Damodaran (Investment Valuation, Narrative & Numbers), McKinsey Valuation,
> Rappaport & Mauboussin (Expectations Investing). Sayısal hesaplar kodda yapılır (bkz. prompt'taki
> valuation engine bölümü); bu dosya yöntem SEÇİMİNİ, varsayım SINIRLARINI ve yorum KURALLARINI belirler.

---

## 1. Temel ilke

Değer bir nokta değil, varsayımların fonksiyonudur. Her fair value çıktısı:
- hangi yöntemden geldiğini,
- hangi 2-3 kritik varsayıma dayandığını,
- bu varsayımlardan hangisinin en kırılgan olduğunu ve nedenini
açıkça söylemek ZORUNDADIR. Varsayımsız sayı üretme.

Hikaye → sayı zinciri: önce şirketin hikayesi (ne satıyor, neden büyüyecek/büyümeyecek),
sonra bu hikayenin sayısal karşılığı (growth, margin, reinvestment), en son değer.
Hikayesi anlatılamayan bir growth varsayımı geçersizdir.

## 2. Sektör → yöntem haritası

Motor SIC kodundan sektörü belirler; yöntem seçimi buna göre:

| Şirket tipi | Birincil yöntem | İkincil | KULLANMA |
|---|---|---|---|
| Olgun, FCF-pozitif (çoğu şirket) | DCF | Multiples (P/E, EV/EBITDA, P/FCF) | — |
| Olgun, kârlı ama FCF büyüme-CapEx/SBC ile bastırılmış (§3b) | Kazanç-gücü (EPV) çapası | DCF (ikincil, "neden bastırılmış" kanıtı) | Bastırılmış FCF'i büyütmek |
| ...yukarıdakiyle AYNI ama gerçekleşen ciro CAGR'ı ≥ %10 (§3c) | Olgun revenue-first DCF (EPV tabanını geçerse), aksi halde EPV | EPV taban (her zaman ikincil çapraz-kontrol) | Bastırılmış FCF'i büyütmek |
| Banka / sigorta / finansal | P/B + ROE ilişkisi, Dividend Discount | P/E | Standart DCF (FCF tanımı bozuk) |
| Zarar eden / hiper-growth | Revenue-first DCF (§4a) + Reverse DCF (birincil) | P/S, brüt marj tavan kontrolü | P/E (anlamsız), FCF-growth ekstrapolasyonu |
| Cyclical (semi, emtia, enerji) | Normalize (mid-cycle) earnings üzerinden DCF/PE | P/B (dip dönemlerde) | Peak-earnings P/E |
| REIT / yüksek temettü | FFO bazlı multiples, DDM | — | Net income bazlı P/E |
| Pre-revenue / binary outcome | Senaryo ağacı (başarı olasılığı × sonuç değeri) | — | DCF (sahte hassasiyet üretir) |

## 3. Cyclical kuralları (zorunlu, özellikle 5y vadede)

- Margin'ler kendi 10-15 yıllık aralığının üst %25'indeyse: bugünkü earnings'i KALICI SAYMA.
  Normalize earnings = son tam döngünün (tepe + dip dahil) ortalama margin'i × güncel revenue.
- "Düşük P/E + tepe margin" kombinasyonu ucuzluk değil, cyclical trap sinyalidir. Verdict'te açıkça uyar.
- Tersi de geçerli: dip margin'de yüksek P/E, pahalılık kanıtı değildir.
- Döngü konumu belirsizse fair value bandını GENİŞLET ve güveni düşür — daraltma.

### Döngüsel şirketlerde normalize edilmiş FCF

- Motor, `cyclical` şirketler için ham (trailing) FCF DCF'in yanında ayrı bir "normalize edilmiş" DCF varyantı üretir. Bu varyantın tabanı: `normalized_fcf0 = (en iyi ceil(N/2) yılın FCF marjı ortalaması) × son yılın geliri`, N = FCF marjı hesaplanabilen mali yıl sayısı, FCF marjı = (İşletme Nakit Akışı − CapEx) / Gelir.
- Neden medyan değil üst-yarı ortalaması: ~5 yıllık bir pencerede tek bir felaket-dip yıl varsa medyan cari (dibe yakın) yıla denk gelir ve "normalize edilmiş" değer ham dip-FCF DCF'iyle birebir aynı çıkar (no-op) — üstteki "Düşük P/E + tepe margin" cyclical trap uyarısının simetriğini kaçırmış oluruz. Üst yarının (en iyi ~yarısı) ortalaması alınınca değer tek bir dip yılın çarpıtmasından arınır ve döngü-ortası/tepesi kazanç gücünü yansıtır.
- Üst-yarı ortalama marj yine de pozitif değilse normalize varyant üretilmez (anlamlı olmadığı için); yalnızca ham DCF raporlanır.
- Döngüsel şirketlerde manşet makul değer aralığı (`fair_value_range`) ve üçgenlemedeki DCF sinyali — hesaplanabildiği sürece — bu normalize edilmiş varyanttan alınır; ham tek-yıl (dip-FCF) DCF senaryoları yalnızca detay olarak `dcf.scenarios` altında yan yana raporlanır. Normalize varyant üretilemezse (ör. üst-yarı marj pozitif değil) manşet, eskiden olduğu gibi ham FCF-DCF bandına düşer. Neden: böylece döngüsel bir hissenin verdict'i, dibe yakın tek bir yılın nakit akışıyla yapay biçimde "aşırı ucuz/pahalı" görünmek yerine döngü-ortası kazanç gücünü yansıtır.

## 3b. Kazanç-gücü (EPV) çapası — olgun, FCF'i bastırılmış şirketler

Amazon-tipi problem: olgun ve kârlı, ama serbest nakit akışı büyük büyüme
CapEx'i ve/veya hisse bazlı ücretlendirme (SBC) yüzünden yapay biçimde
bastırılmış bir şirkette standart FCF-DCF anlamsız-düşük (hatta sıfıra yakın)
bir manşet üretir. Bu durumda motor, Bruce Greenwald'ın sıfır-büyüme
"kazanç-gücü" (earnings power value, EPV) kavramına geçer: `normalize edilmiş
net kâr / özkaynak maliyeti (iskonto oranı) / hisse sayısı`. EPV kasıtlı
olarak büyüme İÇERMEZ (muhafazakâr bir taban, büyüme değerlemesi değil) ve
net-borç köprüsü kullanmaz (net kâr zaten kaldıraçlı/özkaynak seviyesinde bir
figürdür).

Bu geçiş yalnızca ÜÇ koşul birden sağlandığında olur: (a) FCF-DCF'in üst bandı
EPV'ye göre belirgin biçimde bastırılmış görünüyor, (b) işletme nakit akışı
net kârı yeterince destekliyor (nakde-çevirme sağlıklı — bu koruyucu
KRİTİKTİR: aksi halde bir kazanç-kalitesi kırmızı bayrağı EPV'nin arkasına
gizlenebilir), (c) CapEx işletme nakit akışının büyük bir kısmını tüketiyor
(bastırmanın büyüme yatırımından kaynaklandığının kanıtı). Nakde-çevirme
koşulu sağlanmadan FCF düşük görünüyorsa, motor EPV'ye GEÇMEZ; bunun yerine
bunu açık bir kazanç-kalitesi uyarısı olarak raporlar.

EPV manşet olduğunda duyarlılık tablosu ve ters-DCF, manşetin kendisini değil,
ikincil (bastırılmış) FCF-DCF tabanını göstermeye devam eder — bu, bastırmanın
KANITI olarak bilinçli tutulur, §5'teki üçgenleme tutarlılık ilkesinin
belgelenmiş bir istisnasıdır. EPV manşetken üçgenleme güveni en fazla ORTA ile
sınırlanır (DCF ve multiples bacakları artık aynı kazanç sinyalinden türetilir,
üçünün de "aynı yönde" olması bağımsız kanıt sayılmaz). Tam mekanik: SPEC.md
§8a.

## 3c. Olgun revenue-first / olgun-marj DCF — büyüme-dahil EPV alternatifi

§3b'deki EPV çapası KASITLI olarak sıfır-büyümedir — muhafazakâr bir taban,
şirketin hâlâ sürmekte olan gerçek ciro büyümesini yansıtmaz. Amazon-tipi
şirket hem (a) olgun ve kârlı hem de (b) FCF'i büyüme-CapEx/SBC ile bastırılmış
OLUP AYNI ZAMANDA (c) gerçekten yüksek bir ciro büyüme oranını sürdürüyorsa,
salt EPV'ye dayanmak büyümeyi tamamen masaya bırakır. Bu durumda motor, §4a'nın
hiper-grower revenue-first DCF'iyle AYNI deseni — ama olgun bir filer için
kalibre edilmiş biçimde — ikinci bir alternatif olarak dener.

**Tetik (kapı):** §3b'nin FCF-DCF güvenilmezlik testi zaten sağlanmış olmalı
(EPV de bu yüzden deneniyor) VE gerçekleşen ciro CAGR'ı (5y, yoksa 3y) en az
%10 VE bu oran senaryonun terminal büyüme oranının üzerinde (fade edecek
gerçek bir büyüme var). Bu eşiğin altında (durgun/yavaş büyüyen "olgun"
şirket) yöntem hiç denenmez; manşet EPV'de kalır.

**Yöntem — revenue-first, ince olgun marj:** Değerleme yine gelirden başlar
(§4a'daki gibi FCF geriye eklenmez: `FCF_yıl = gelir_yıl × marj_yıl`).
Başlangıç büyümesi gerçekleşen CAGR ile en son mali yılın YoY büyümesinin
harmanıdır (`0.5×CAGR + 0.5×YoY`); bu tek oran, hiper-grower'ın aksine,
bear/base/bull arasında ÖLÇEKLENMEZ — sadece iskonto oranı ve hedef marj
senaryoya göre değişir (0.7×/1.0×/1.2×). Marj, bugünkü (bastırılmış, son 3
yılın medyanı) seviyeden 7 yıl içinde (hiper-grower'ın 10 yılından daha kısa
bir ufuk — olgun bir şirketin büyüme hikâyesi zaten daha ileri bir aşamada)
veri-türevli bir olgun hedefe yakınsar: hedef marj = min(NOPAT-vekili
[medyan operasyon marjı × (1−%25 vergi) × %85 yeniden-yatırım tutma payı],
tarihsel-tepe FCF marjı × 1.5, mutlak tavan %15), bugünkü marj tabanına
oturtulmuş. Bunun §4a'nın 30%'a varan hiper-grower tavanından çok daha ince
olmasının nedeni: bu artık büyük ve zaten olgun bir şirket, steady-state
ekonomisini hâlâ arayan bir hiper-grower değil.

**Neden çift-saymıyor (reddedilen owner-earnings alternatifinden farkı):**
`FCF_yıl = gelir_yıl × marj_yıl` — hiçbir şey FCF'e geri eklenmez. Erken
yıllarda yüksek büyüme + düşük (bugünkü) marj, geç yıllarda faded büyüme +
olgun marj birleşir; yeniden-yatırım drenajı marj-fade'in içinde örtük olarak
yaşar. §4'teki "büyüme bedavaya gelmez" ilkesiyle bu şekilde tutarlıdır. Bu,
FCF'i doğrudan büyütüp sonra bir CapEx tahminini geri ekleyen bir
owner-earnings varyantından KASITLI olarak farklıdır — o yaklaşım
değerlendirilip reddedildi, çünkü marj-fade yönteminin yapısal olarak zaten
fiyatladığı yeniden-yatırımı ikinci kez (ve keyfi biçimde) saymak riski
taşır.

**EPV-taban guardrail:** Büyüme-dahil bu değer, sıfır-büyüme EPV tabanının
ALTINDA çıkarsa (base senaryo, hisse başına), manşet EPV'de KALIR ve
revenue-first band `mature_revenue_detail` altında ikincil bir çapraz-kontrol
olarak raporlanır — manşete geçmez. Neden: EPV net-kâr temelli, daha kalın
bir tabandır; revenue-first yöntem daha ince bir FCF marjı kullanır. Büyümeyi
dahil etmek değeri EPV'nin zaten kapitalize ettiği kazancın da altına
düşürüyorsa, büyüme hikâyesi henüz gerçek bir değer eklemiyor demektir — bu
durumda muhafazakâr EPV tabanını korumak, savunulabilir olmayan bir "büyüme"
rakamı sunmaktan daha güvenlidir. Bu ancak revenue-first değer EPV tabanını
GEÇERSE manşete geçer. **Ampirik not:** mevcut kalibrasyonla test edilen tüm
örneklerde (AMZN, ORCL dahil) revenue-first değer EPV tabanının altında
kaldı — yani yöntem şu an pratikte hep ikincil çapraz-kontrol rolünde kaldı,
hiç manşete geçmedi; bu, kalibrasyon veya test edilen şirket seti
genişledikçe değişebilir.

Manşet olduğunda (guardrail'i geçtiğinde) duyarlılık tablosu ve ters-DCF, §3b
ile aynı şekilde, manşetin kendisini değil ikincil (bastırılmış) FCF-DCF
tabanını göstermeye devam eder (§5 istisnası); ters-DCF'in referans büyümesi
de FCF değil GELİR CAGR'ıdır (yöntemin kendisi gelir-temelli olduğu için).
Üçgenleme güveni burada da en fazla ORTA ile sınırlanır — bu moddaki DCF ve
ters-DCF bacakları AYNI revenue-first modelden türer, bağımsız kanıt
sayılmaz. Tam mekanik: SPEC.md §8b.

## 3d. REIT'lerde FFO bazlı Gordon büyüme değerlemesi

§2'deki "Banka / sigorta / finansal" satırı P/B×ROE'de KALIR (değişmedi), ama
"REIT / yüksek temettü" satırının kod tarafı artık tablodaki yöntemi
(FFO bazlı multiples, DDM) gerçekten uyguluyor — önceden REIT'ler de yanlışlıkla
banka yolundan (P/B×ROE, net kâr bazlı P/E) değerleniyordu; bu bir çelişkiydi ve
düzeltildi. Sebep: GAAP gayrimenkul amortismanı hem net kârı hem özkaynağı büyük
ölçüde bastıran nakit-dışı bir kalemdir — bu yüzden P/B×ROE (ya da P/E) bir
REIT'in gerçek değerini sistematik olarak düşük gösterir.

**Yöntem:** FFO (funds from operations) = net kâr + amortisman (D&A) − gayrimenkul
satış kârı + gayrimenkul değer düşüklüğü (impairment), en son mali yılın
verisiyle (net kâr VE amortisman aynı mali yılda birlikte mevcut olmalı; satış
kârı/impairment etiketleri o mali yılda varsa eklenir, yoksa 0 kabul edilir),
güncel hisse sayısına bölünerek hisse başı FFO elde edilir. Ardından
her senaryo (bear/base/bull) KENDİ iskonto oranı (r, özkaynak maliyeti) ve
uçtaki büyüme oranıyla (g) klasik Gordon büyüme formülünü uygular:

```
hisse_başı_değer = ffo_hisse_başı × (1 + g) / (r − g)
```

`(1 + g) / (r − g)` çarpanının kendisi, o senaryonun ima ettiği makul (fair)
P/FFO çarpanıdır — P/B×ROE'nin kırpılmış (clamp) `fair_pb` sabitinin aksine,
keyfi bir hedef-çarpan sabiti gerekmez. `r`/`g` eksikse ya da `r ≤ g` ise (Paket
1'in asgari risk-primi koruması bunun normalde olmamasını sağlar, ama yine de
kontrol edilir) o senaryo atlanır, uydurulmaz.

**Gayrimenkul satış kârı / değer düşüklüğü düzeltmesi (Paket 2/P2a):** Motor artık
gayrimenkul-satış kârlarını FFO'dan çıkarıyor ve gayrimenkul değer düşüklüğü
(impairment) kalemlerini geri ekliyor — bunun için iki yeni, GAYRİMENKULE ÖZGÜ
(genel `AssetImpairmentCharges` gibi geniş etiketler KULLANILMADI, çünkü bunlar
gayrimenkul-dışı kalemleri de yanlışlıkla düzeltir) opsiyonel kavram eklendi
(`normalize/concepts.py`): `GainOnSaleRealEstate` ve `RealEstateImpairment`. Bu
etiketlerden biri o mali yılda yoksa katkısı 0 kabul edilir — yani bu etiketleri
hiç raporlamayan bir şirket için hesap ÖNCEKİYLE BİREBİR AYNI sonucu verir
(geriye dönük uyumlu). Etiket kapsamı kaçınılmaz olarak kısmi (bir şirket
listede olmayan bir etiket kullanıyorsa o düzeltme sessizce 0 kalır).

**D&A-vekili sınırlaması (dürüstçe belirtilmeli, hâlâ geçerli):** Gerçek Nareit
FFO yalnızca gayrimenkul amortismanını geri ekler; bu motorun normalize
verisinden gayrimenkul-özel amortisman ayrıştırılamıyor, bu yüzden toplam D&A
(nakit akışı tablosundaki amortisman kalemi) topluca geri eklenir. Bu,
gayrimenkul-dışı amortismanı (ör. bir satın almadan gelen maddi olmayan varlık
amortismanı) belirgin olan bir şirket için FFO'yu hafifçe ŞİŞİRİR; saf bir
REIT'te (D&A'sı ezici çoğunlukla bina/gayrimenkul amortismanı olan) bu yakın
bir yaklaşıklıktır.

**Çarpan sinyali:** Multiples tarihçesi artık `pffo` (P/FFO = fiyat × hisse /
FFO, aynı düzeltilmiş FFO tanımıyla) da içeriyor; REIT'lerde çarpan sinyali
P/E YERİNE öncelikle P/FFO persentilini kullanır (yedek: P/S). P/E, D&A
yüzünden REIT'lerde anlamsızdır — tıpkı FFO'nun var olma sebebi gibi.

**Zarif geri düşüş (fallback):** FFO hesaplanamazsa (hiçbir mali yılda net kâr
VE amortisman birlikte yoksa, ya da sonuçtaki FFO sıfır/negatifse), motor
manşet/üçgenleme çapası olarak `financial` sektörünün kullandığı P/B×ROE
çapasına geri döner ve bunu bir notla açıkça belirtir — hiçbir zaman uydurma
bir FFO değeri üretmez. Tam mekanik: SPEC.md §8c.

## 4. Varsayım sınırları (sanity check — kod da ayrıca doğrular)

- **Terminal growth ≤ uzun vadeli nominal GDP büyümesi (~%3-4).** Üstü otomatik geçersiz.
- **Discount rate tabanı:** risk-free + sektör ERP (Damodaran verisi). %7'nin altı hiçbir hisse için kabul edilmez; spekülatif/zararda şirketlerde %11-15 bandı.
- **Büyüme kısıtı ORAN değil VARIŞ NOKTASIDIR:** "%20+ growth en fazla N yıl" gibi keyfi bir tavanla kırpma YAPILMAZ. Kısıt, gerçekleşen oranın terminale doğru fade (kademeli, mean-reversion) etmesinden gelir — sonsuza dek sabit yüksek growth diye bir şey yok, ama fade zorunlu olduğu sürece keyfi bir yıl sınırı gerekmez. Varış noktasının (10 yıl sonraki gelir seviyesinin) makullüğü şöyle denetlenir: TAM (toplam adreslenebilir pazar) biliniyorsa, son yıl geliri/TAM oranı %40'ı aşıyorsa "agresif", %60-70'i aşıyorsa "geçersiz" sayılır. TAM bilinmiyorsa, gelir-katı (son yıl geliri / bugünkü gelir) 8×'in üstü "agresif", 15×'in üstü "aşırı agresif" sayılır. Detay: §4a.
- **Büyüme bedavaya gelmez:** yüksek growth varsayımı, yüksek reinvestment (CapEx/R&D) varsayımı gerektirir. FCF margin'i büyürken CapEx'i sabit tutan model tutarsızdır.
- **Dilution:** SBC/revenue > %5 ise per-share değer hesabında hisse sayısı artışını projeksiyona dahil et.

## 4a. Hiper-grower kuralları

Bu bölüm §2'deki "Zarar eden / hiper-growth" satırının ve §4'teki "varış noktası" ilkesinin
uygulama detayıdır. Reddit-tipi problem: hiper-büyüyen, henüz (tam) kâr etmeyen bir şirketin
bugünkü FCF'i büyüme harcamasıyla BASTIRILMIŞTIR; standart bir FCF-DCF bu şirketi
sistematik olarak ıskalar (undervalue eder).

**Tetik koşulu:** gerçekleşen gelir CAGR'ı %25'in üstünde VE aşağıdakilerden en az biri:
- FCF sıfır veya negatif, VEYA
- FCF marjı %5'in altında (bastırılmış nakit akışı), VEYA
- (Ar-Ge + SBC) / gelir %40'ı aşıyor (agresif büyüme yatırımı).

**Yasak:** FCF-growth ekstrapolasyonu. Bugünkü FCF bastırılmış olduğundan onu sabit bir
oranla büyütmek (standart FCF-DCF'in yaptığı şey) yapısal olarak yanlıştır ve bu modda
BAŞVURULMAZ.

**Revenue-first DCF:** Değerleme, FCF'den değil gelirden başlar. Gelir yolu gerçekleşen
büyüme oranından terminal büyümeye doğru fade (kademeli yakınsama) ile ilerler (bkz. §4).
Her yıl için bir FCF marjı projekte edilir; bu marj bugünkü (bastırılmış) seviyeden
olgunluk (mature) durumundaki bir hedef marja doğru lineer olarak yakınsar. Hedef olgun
marj brüt marjı AŞAMAZ (brüt marj tavan kontrolü) ve steady-state yılına kadar tam
yakınsamış olmalıdır. Projekte edilen gelir × marj = FCF; bu FCF yolu iskonto edilerek
bugüne indirgenir.

**Dilution ZORUNLUDUR:** SBC (hisse bazlı ücretlendirme) ve negatif-FCF döneminin
finansmanı (nakit yakan yılların sermaye ihtiyacı) hisse başı değeri seyreltir. Per-share
değer BUGÜNKÜ hisse sayısıyla HESAPLANMAZ; projeksiyonlu (SBC kaynaklı yıllık seyreltme +
finansman amaçlı ek hisse ihracı dahil edilmiş) hisse sayısı kullanılır.

**Reverse DCF BİRİNCİLDİR:** Bu modda üçgenlemenin ağırlık merkezi fiyatın ima ettiği
değerlerdir — ima edilen 10 yıllık gelir, ima edilen olgun FCF marjı ve (TAM biliniyorsa)
ima edilen TAM payı. "Fiyat şunu varsayıyor" cümlesi, ileri projeksiyon yapmaktan daha
güvenilir bir kanıttır çünkü tahmin gerektirmez.

**Verdict dili:** Fiyat bull senaryosunun üst bandı İÇİNDEYSE (base bandın üstünde ama
bull bandını aşmıyorsa) "PAHALI" DENMEZ; bunun yerine "YÜKSEK BEKLENTİ FİYATLANMIŞ" denir —
bu bir uyarı tonudur, "PAHALI"nın kırmızı alarmından ayrıdır. "PAHALI" ancak fiyat bull
bandının da üstüne çıkıp o iması da aşıldığında kullanılır.

**Geniş bant normaldir:** Hiper-grower değerlemesinde dar bant üretmek sahte hassasiyettir.
Bear/base/bull senaryoları olasılık ağırlıklandırılır (ör. %25/%50/%25) ve olasılık-ağırlıklı
beklenen değer (expected value) ayrıca raporlanır; bant daraltılmaya ÇALIŞILMAZ.

**Tez kill-switch:** Gerçekleşen gelir büyümesi iki ardışık çeyrek boyunca base senaryonun
öngördüğü yolun ALTINA inerse, tez (ve dolayısıyla mevcut fair value bandı) GEÇERSİZ sayılır
ve yeniden değerlendirme gerekir.

**Deterministik uygulama notu:** LLM/kullanıcı TAM (toplam adreslenebilir pazar) sağlamadığında,
büyüme-fade kısıtı (§4) ve brüt marjdan türetilen hedef olgun marj TAM'ın YERİNE GEÇER —
yani TAM olmadan da varış noktası makullük kontrolü yapılabilir (gelir-katı 8×/15× eşiği).
TAM sonradan sağlanırsa bu deterministik varsayılanları RAFİNE eder (ör. gelir-katı yerine
doğrudan TAM payı ile agresiflik değerlendirilir); TAM asla ZORUNLU değildir.

**Negatif özkaynak değeri guardrail'i (bastırma):** CapEx-ağır hiper-grower'larda (ör. veri
merkezi yatırımcıları) büyüme CapEx'i gelirin kat kat üzerinde olabilir; bu durumda bugünkü
FCF marjı derinden negatiftir ve marj yalnızca lineer olarak olgun (pozitif) bir hedefe fade
edildiğinden, erken yılların iskonto edilmiş nakit yakımı terminal değeri aşıp baz senaryoda
negatif özkaynak değeri (hisse başı ≤ $0) üretebilir. **Tetik koşulu: baz senaryonun hisse
başı değeri ≤ $0.** Bu durumda motor sonucu BASTIRIR: manşet `fair_value_range` boşaltılır ve
üçgenlemedeki DCF bacağı `veri_yok` olur. Hiper-grower modu yine de TESPİT EDİLMİŞ sayılır ve
(negatif) senaryolar şeffaflık için `hyper_growth_detail` altında görünür kalır — sadece
manşet olarak yayınlanmazlar. Gerekçe: faal ve sermaye toplayabilen bir şirketi $0'ın altında
değerlemek kullanılabilir bir sayı değildir; negatif bir bant basmak veya keyfi biçimde
sıfıra kırpmak yerine sonucu bastırıp nedenini açıklamak daha dürüsttür.

## 5. Üçgenleme ve güven kuralı

Üç yöntem (DCF, reverse DCF iması, multiples konumu) birlikte değerlendirilir:
- Üçü aynı yönde → güven YÜKSEK.
- İkisi aynı, biri ayrık → güven ORTA, ayrık yöntemin neden ayrıştığını açıkla.
- Üçü dağınık → güven DÜŞÜK; verdict "belirsiz" tonunda verilir, dar bant ÜRETME.
Yöntem çelişkisini gizlemek en büyük hatadır; çelişki bilgidir.

Hiper-grower'larda (§4a) bu üçgenleme reverse DCF merkezli yorumlanır: fiyatın ima ettiği
gelir/marj/TAM payı birincil kanıttır; revenue-first DCF bandı ve multiples (P/S, brüt marj
konumu) ikincil kanıt olarak üçgenlemeyi destekler/çürütür.

## 6. Reverse DCF yorumu

Koddan gelen "fiyatın ima ettiği growth" değeri şöyle yorumlanır:
- İma edilen growth, şirketin son 5y gerçekleşen CAGR'ının belirgin ÜSTÜNDEyse ve
  hikaye bunu desteklemiyorsa → PAHALI yönünde güçlü kanıt.
- İma edilen growth, muhafazakar base senaryonun bile ALTINDAysa → UCUZ yönünde güçlü kanıt.
- Bu yöntemin gücü: tahmin gerektirmez, sadece fiyatın varsayımını ifşa eder. Verdict özetinde
  mümkünse tek cümleyle kullan: "Bugünkü fiyat X yıl boyunca %Y büyüme ima ediyor."

## 7. Multiples kuralları

- Çarpanı iki eksende konumlandır: (a) şirketin KENDİ 10-15y medyanına göre, (b) sektör medyanına göre.
- Kendi tarihine göre pahalı + sektöre göre pahalı → net sinyal. Karışık → nedenini açıkla
  (ör. iş modeli değişti, margin yapısı kalıcı iyileşti — "bu sefer farklı" iddiası ancak kanıtla).
- EV bazlı çarpanları (EV/EBITDA) borçlu şirketlerde P/E'ye tercih et.

### Growth-ayarlı çarpanlar (PEG katmanı)

Motor, ham çarpanın yanında bir **growth-ayarlı çarpan** üretir: standart modda **PEG = güncel P/E ÷ base büyüme** (assumptions pipeline'ın base senaryosundan, yüzde puan olarak — ör. %15 → 15); hiper-grower modunda P/E anlamsız olduğundan yerini **growth-ayarlı EV/Satış = güncel EV/Satış ÷ base büyüme** alır. Kurallar:

- **PEG bağımsız verdict ÜRETMEZ; yalnızca multiples sinyalini rafine eder.** Üçgenlemeye ayrı bir yöntem olarak girmez.
- **Mutlak eşik kuralı YOK.** "PEG < 1 = ucuz" gibi doğrusal eşikler KULLANILMAZ (PEG'in doğrusallık kusuru); yalnızca (a) şirketin KENDİ tarihsel PEG serisine göre GÖRELİ konum (percentile) ve (b) — veri varsa — sektör medyan PEG'i kullanılır. Tarihsel seri: her geçmiş yılın yıl-sonu çarpanı, o yılı takip eden 3 yılda GERÇEKLEŞEN gelir CAGR'ına bölünür (verisi tam olan yıllar için).
- **Payda her zaman assumptions pipeline'ın base büyümesidir** ve çıktıda `base_growth_pct` olarak GÖRÜNÜR. Şeffaflık için pay/payda gizlenmez.
- **Uygulanabilirlik:** PEG yalnızca TTM kâr pozitifken (P/E > 0) VE base büyüme ≥ %5 iken hesaplanır. Aksi halde "uygulanamaz" olarak raporlanır — asla negatif ya da patlamış (payda → 0) bir PEG gösterilmez. Aynı %5 tabanı growth-ayarlı EV/Satış için de geçerlidir.
- **İki bileşen ayrışırsa sinyal "karışık"tır.** Ham çarpanın percentile'ı ile growth-ayarlı percentile FARKLI yönlere düşerse (ör. ham %88 → pahalı, PEG %45 → ortada) multiples sinyali `karisik` olur ve verdict'te tek cümleyle açıklanır ("çarpan yüksek ama büyümeye göre normalize edildiğinde tarihsel ortalamasında"). İkisi aynı yöndeyse ham çarpan sinyali korunur. Bu, §5'teki "yöntem çelişkisini gizleme" ilkesinin multiples içi uygulamasıdır.
- **Sektör medyan PEG** yalnızca Damodaran referans verisinde beklenen büyüme (veya doğrudan PEG) sütunu VARSA hesaplanır (nice-to-have); yoksa boş geçilir.

## 8. Yasak davranışlar

- Sahte hassasiyet: "$103.47" gibi tek sayı verme; her zaman bant + varsayım.
- Multiples'ı DCF'e "yakınsın" diye seçmek (yöntem bağımsızlığını bozar).
- Duyarlılık tablosu geniş dağılıyorsa dar bant sunmak.
- Veri eksikse (ADR, kısa geçmiş) tahmin uydurmak — eksikliği raporla, güveni düşür.
- Kullanıcının pozisyonu olan hissede iyimserliğe kaymak: PROFIL/POZISYONLAR bağlamı
  yorumu kişiselleştirir, MATEMATİĞİ değiştirmez.
