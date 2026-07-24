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
| ...yukarıdakiyle AYNI ama gerçekleşen ciro CAGR'ı %12-20 arası (hiper eşiğinin altında) (§4b) | Orta-büyüme revenue-first DCF (büyüme kapısını geçerse), aksi halde multiples | Reverse DCF (ikincil, gelir-temelli) | Standart FCF-DCF (bastırılmış FCF'i büyütmek) |
| Cyclical (semi, emtia, enerji) | Normalize (mid-cycle) earnings üzerinden DCF/PE | P/B (dip dönemlerde) | Peak-earnings P/E |
| ...yukarıdakiyle AYNI ama FCF her yıl büyüme-CapEx'iyle yapısal olarak bastırılmış (§3e) | Sürdürülebilir-büyüme FCFE çapası (kazanç + reinvestment=g/ROE; EPV tabanını geçerse), aksi halde EPV | EPV taban + döngü-ortası normalize FCF-DCF (ikisi de ikincil çapraz-kontrol) | Bastırılmış FCF'i büyütmek |
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

## 3e. Sermaye-yoğun döngüsel şirketlerde sürdürülebilir-büyüme FCFE çapası

§3'teki "döngü-ortası normalize FCF-DCF" çözümü, tek bir yılın dip nakit
akışını düzeltir — ama bazı sermaye-yoğun döngüsel şirketlerde (Micron/MU
tipik örnek) sorun tek bir dip yılı değil: serbest nakit akışı HER YIL, büyüme
CapEx'i (fab genişletme) tarafından yapısal olarak bastırılıyor. Bu durumda
döngü-ortası normalize edilmiş FCF marjı bile gerçek kazanç gücünü ciddi
biçimde ıskalar — çünkü FCF-DCF, büyümeyi finanse eden CapEx'in TÜMÜNÜ kalıcı
bir nakit tüketimi gibi fiyatlarken karşılığında sadece mütevazı bir ciro
büyümesi kaydeder: MU'da CapEx gelirin %36-49'u kadardır (her yıl), ham
dip-FCF DCF baz değeri ~$10'da, döngü-ortası normalize FCF-DCF ise ~$31'de
kalır — halbuki net kâr marjı %23 ile sağlıklıdır.

**Neden earnings-tabanlı bir çapa (revenue-first değil):** Bu şirketin net
kârı nakde iyi çevriliyor (nakit akışı net kârı destekliyor) ve büyümesi
gerçek — sorun kazanç kalitesinde değil, CapEx muhasebesinde. Doğru
büyüme-dahil değer, normalize edilmiş KAZANCI sürdürülebilir-büyüme
(sustainable growth) mantığıyla büyütür: bir şirket kazancını `g` oranında
büyütmek istiyorsa, ROE'sini sabit tutarak bunu yapabilmesi için kârının
`b = g / ROE` kadarını yeniden yatırması (reinvest) gerekir; kalan kısım
(`net kâr × (1 − b)`) hissedara dağıtılabilir serbest nakit akışıdır (FCFE).
Bu, büyümenin neden yalnızca ROE, özkaynak maliyetinden (r) YÜKSEK olduğunda
değer eklediğinin ders-kitabı gerekçesidir: `ROE == r` ise yeniden yatırılan
her dolar tam da yatırımcının talep ettiği getiriyi kazanır (dağıtmak yerine
yeniden yatırmak değer-nötrdür — bu çapa o noktada EPV'ye yakınsar); `ROE >
r` ise yeniden yatırılan dolarlar fazla getiri kazanır ve büyüme gerçekten
değer eklemeye başlar; `ROE < r` ise büyüme değer YOK EDER — bu son durumda
çapa, aşağıdaki EPV-taban güvencesiyle otomatik olarak sıfır-büyüme EPV'ye
geri döner. MU'da ROE ~%15.8, özkaynak maliyeti ~%10.6 — büyüme gerçekten
değer ekliyor.

**Büyüme ROE'yi aşamaz:** Her projeksiyon yılında (ve uçtaki/terminal yılda)
fiilen kaydedilen büyüme `g_efektif = min(g, ROE)`'dir — hem kazanç
büyütmesi hem de reinvestment oranı (`b = g_efektif / ROE`) bu sınırlı oranı
kullanır. `g >= ROE` olduğunda şirket bu büyümeyi salt kendi kazancından
finanse edemez (dış özkaynak gerekir); model bunu ya (a) finanse edilemeyen
büyümeyi kaydederek ya da (b) reinvestment oranını %100'ün üzerine çıkarıp
nakit icat ederek çözmez — büyümenin kendisini ROE'nin fonlayabileceği
seviyeyle sınırlar. Bu sayede reinvestment oranı asla keyfi bir tavana (ör.
sabit %90) çarpmaz; `g_efektif`, `ROE`'ye yaklaştıkça 1.0'a (ama hiç aşmadan)
yükselir ve dağıtılabilir FCFE o yıl için 0'a (ama hiç altına inmeden) düşer.

**Uçtaki (terminal) ROE, özkaynak maliyetine söner:** 1-10. yıllar kendi
güncel ROE'sini kullanırken, terminal/perpetuite yılın reinvestment oranı
şirketin ÖZKAYNAK MALİYETİNE (o senaryonun `r`'sine) sönen bir ROE kullanır —
Damodaran'ın istikrarlı-faz konvansiyonu: rekabet avantajları eridikçe bir
şirketin ROE'si sonsuza dek özkaynak maliyetinin üzerinde kalamaz, yakın
vadede daha yüksek olsa bile. Bu, tek başına MU'nun baz değerini ~$89'dan
~$84.77'ye düşüren muhafazakâr bir düzeltmedir.

**EPV-taban güvencesi (§3b'yle aynı desen):** Bu FCFE çapası, sıfır-büyüme
EPV tabanının (MU'da ~$71.59) ALTINDA çıkarsa, manşet EPV'de KALIR — tıpkı
§3c'nin revenue-first-vs-EPV guardrail'i gibi. Yalnızca FCFE değeri EPV
tabanını GEÇERSE manşete geçer. Her iki durumda da döngü-ortası normalize
FCF-DCF (§3) ve ham dip-FCF DCF ikincil çapraz-kontrol olarak yan yana
raporlanır.

**Hangi döngüsel şirket bu yola girer:** Motor önce §3b'nin FCF-DCF
güvenilmezlik testini (aynı üç koşul: bastırılmış görünüyor, nakde-çevirme
sağlıklı, CapEx işletme nakit akışının büyük kısmını tüketiyor) bu döngüsel
şirkete de uygular. Test tetiklenmezse (sıradan dip-yılı bastırması) hiçbir
şey değişmez — manşet §3'teki döngü-ortası normalize FCF-DCF'te kalır. Test
tetiklenirse FCFE çapası (veya EPV tabanı) devreye girer.

**Yapısal re-rating / dip-dışlama varsayımı (açıkça belirtilir):** Bu çapanın
kazanç tabanı, EPV'nin normalize net kârıdır — son temsili (kârlı) yıllardan
gelir ve şiddetli bir döngü dibini (MU'nun 2023 bellek-glut zarar yılı gibi)
döngünün tekrar edecek kalıcı bir parçası saymaz; onu istisnai, tekrar
etmeyecek bir olay olarak DIŞLAR. Bu, kullanıcının bilinçli tercih ettiği bir
yapısal-re-rating varsayımıdır — dipleri kalıcı sayan tam-döngü ortalaması
kullanılsaydı değer belirgin biçimde daha düşük çıkardı. Motor bu varsayımı
her seferinde bir notla açıkça bildirir; sessizce varsaymaz.

Duyarlılık tablosu ve ters-DCF, §3b/§3c'yle aynı şekilde, FCFE manşetinin
kendisini değil ikincil tabanları gösterir (§5 istisnası): duyarlılık tablosu
döngü-ortası normalize FCF-DCF'i, ters-DCF ise ham (baskılanmış) FCF'i
yansıtır — ikisi de FCFE çapasından farklıdır. Üçgenleme güveni burada da en
fazla ORTA ile sınırlanır (DCF bacağı zaten bu kazanç-tabanlı çapadan gelir).
Tam mekanik: SPEC.md §8e.

**Kapıdan rapora:** §3b/§3e'deki "büyüme-dahil değer EPV tabanının altında
kalırsa manşet EPV'de kalır" kuralı artık yalnızca sessiz bir kapı değil,
her iki sayıyı da adlandıran bir rapor — motor `growth_vs_floor` alanına
büyüme-dahil değer EPV tabanının altındaysa `"destroys"` (büyüme değer
SİLİYOR — ROE/re-rating özkaynak maliyetinin altında kaldığı anlamına
gelir), üstündeyse/eşitse `"adds"` (büyüme değer EKLİYOR) yazar ve bu
sınıflandırma hangi taraf manşet olursa olsun (mature revenue-first DCF
için §3c, döngüsel FCFE çapası için burada) hesaplanabildiğinde raporlanır
— rapor katmanı iki sayıyı ve aralarındaki ilişkiyi her zaman gösterebilir.

**MU (Micron) örneği (script provider, gerçek veriyle):** normalize net kâr
$8.54mlyr, ROE %15.76 (özkaynak $54.2mlyr, spot son-mali-yıl bakiyesi —
döngü-ortalaması DEĞİL), özkaynak maliyeti ~%10.6, baz senaryo büyümesi %6.7
→ reinvestment oranı ~%42.6. Baz FCFE çapası: **$84.77** (band $75.39-96.30).
Sıfır-büyüme EPV taban: **$71.59**. İkincil çapraz-kontroller: döngü-ortası
normalize FCF-DCF baz **~$31**, ham dip-FCF DCF baz **~$10**. FCFE çapası EPV
tabanını geçtiği için manşet oldu; üçgenleme güveni ORTA'ya sınırlandı.

## 4. Varsayım sınırları (sanity check — kod da ayrıca doğrular)

- **Terminal growth = min(risksiz getiri oranı, %4).** Üstü otomatik geçersiz (sabit üst sınır: `sanity._TERMINAL_GROWTH_MAX`). Kohort farkı YOK — hiper-grower da dahil HER yol (standart/olgun/orta-büyüme/hiper) aynı kurala bağlıdır: steady-state'e ulaşan bir hiper-grower tanım gereği artık olgun bir şirkettir, risk cezası zaten iskonto oranında ve senaryo olasılıklarında fiyatlanmıştır; uçtaki büyümede üçüncü kez kesmek katmanlı (üst üste binen) bir ceza olurdu. Risksiz getiri kaynağı: önce şirketin SIC'ine eşleşen sektörün CAPM'inden gelen `risk_free` (varsa), yoksa Damodaran'ın GLOBAL (`erp.csv`) risksiz getirisi — SIC'i hiçbir Damodaran sektörüyle eşleşmeyen bir şirketin (ör. düz iskonto oranına düşen bir isim) terminal büyümesi de aynı düz eski sabite (%2.5) düşüp iskonto oranı + terminal büyüme ile ÇİFTE cezalandırılmasın diye. Her ikisi de yoksa düz %2.5 sabiti kullanılır.
- **Discount rate tabanı (CAPM):** iskonto oranı bir özkaynak maliyetidir (WACC değil) ve `cost_of_equity = risk_free + β_levered × ERP` ile hesaplanır (kod: `valuation/capm.py`). `risk_free` ve `ERP` Damodaran'ın US verisinden gelir (`data/damodaran/erp.csv`); `β` ise şirketin SIC'ine eşleşen sektörün **kaldıraçsız (asset) betasıdır** (`multiples.csv`'deki `unlevered_beta` sütunu), şirketin kendi piyasa borç/özkaynak oranı ve marjinal vergi ile **relever** edilir (Hamada: `β_L = β_U × (1 + (1−vergi)×D/E)`). Böylece iskonto oranı düz bir sabit değil, şirketin riskine göre gerekçelendirilmiş bir sayıdır. Damodaran verisi (beta/ERP/risk-free) yoksa eski düz sektör-bağımsız varsayılana (%10, zararda %12) güvenli şekilde geri döner. Her durumda: %7'nin (zararda %10) altı hiçbir hisse için kabul edilmez — bear/bull senaryoları CAPM tabanının ±deltasıdır ve `sanity.clamp_assumptions` her senaryo oranını taban + Gordon ERP-spread kuralıyla ayrıca denetler.
- **Büyüme kısıtı ORAN değil VARIŞ NOKTASIDIR:** "%20+ growth en fazla N yıl" gibi keyfi bir tavanla kırpma YAPILMAZ. Kısıt, gerçekleşen oranın terminale doğru fade (kademeli, mean-reversion) etmesinden gelir — sonsuza dek sabit yüksek growth diye bir şey yok, ama fade zorunlu olduğu sürece keyfi bir yıl sınırı gerekmez. Varış noktasının (10 yıl sonraki gelir seviyesinin) makullüğü şöyle denetlenir: TAM (toplam adreslenebilir pazar) biliniyorsa, son yıl geliri/TAM oranı %40'ı aşıyorsa "agresif", %60-70'i aşıyorsa "geçersiz" sayılır. TAM bilinmiyorsa, gelir-katı (son yıl geliri / bugünkü gelir) 8×'in üstü "agresif", 15×'in üstü "aşırı agresif" sayılır. Detay: §4a. (Mutlak sağlık-kontrolü tavanı — `growth_5y` için — %60'tır, eskiden %40'tı; bu ARIŞ NOKTASI kontrollerinin yedeği, birincil mekanizması değil.)
- **Büyüme bedavaya gelmez:** yüksek growth varsayımı, yüksek reinvestment (CapEx/R&D) varsayımı gerektirir. FCF margin'i büyürken CapEx'i sabit tutan model tutarsızdır.
- **Dilution:** SBC/revenue > %5 ise per-share değer hesabında hisse sayısı artışını projeksiyona dahil et. Hiper-grower ve orta-büyüme revenue-first yollarında bu dilüsyon artık SBC'nin marjda zaten gider yazılmış payını hariç tutuyor (aşağıda §4a) — aksi halde aynı SBC maliyeti hem marj düşüşü hem hisse seyreltmesi olarak iki kez sayılırdı.
- **Marj/çarpan tavanları artık çoğunlukla BAYRAK, sabit KIRPMA değil:** hiper-grower olgun hedef marjı (%30 referans), olgun/orta-büyüme revenue-first hedef marjları (%15/%20 referans) ve finansal sektörün justified P/B'si (`[0.5, 4.0]` referans aralığı) artık bu eşiklerin üstüne/altına çıktığında hesaplanan değeri sabit bir tavana/tabana kırpmıyor; bunun yerine değeri OLDUĞU GİBİ kullanıp bir not + bayrak ekliyor. Gerekçe: yüksek brüt marjlı/yüksek ROE'li bir şirket için bu eşiklerin üstü meşru olabilir — sabit kırpma gerçek ekonomiyi gizlerdi.

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
değer BUGÜNKÜ hisse sayısıyla HESAPLANMAZ; projeksiyonlu (yıllık seyreltme + finansman
amaçlı ek hisse ihracı dahil edilmiş) hisse sayısı kullanılır.

**SBC-dilüsyon çift sayımı çıkarıldı:** Yukarıdaki yıllık seyreltme oranı, gerçekleşen
hisse-sayısı büyümesinden (`shares_yoy`) gelir — ki bu SBC ihraçlarını da içerir. Ama SBC
zaten FCF marjında (yukarıda) doğrudan gider olarak fiyatlanıyor; aynı SBC maliyetini bir
de dilüsyon olarak projekte etmek çift sayım olurdu. Motor artık `sbc_dilution = SBC /
piyasa değeri` oranını hesaplayıp yıllık seyreltmeden çıkarıyor
(`non_sbc = max(0, shares_yoy - sbc_dilution)`, hâlâ %5 tavanlı); yalnızca SBC-DIŞI
ihraçlar (ör. M&A finansmanı, opsiyon dışı hisse satışı) dilüsyona giriyor. Nakit-yakımı
finansmanı için ek hisse ihracı (yukarıdaki "finansman" kalemi) bu düzeltmeden bağımsız,
aynen kalıyor. Orta-büyüme revenue-first yolu (§4b) da aynı düzeltmeyi kullanıyor.

**Reverse DCF BİRİNCİLDİR:** Bu modda üçgenlemenin ağırlık merkezi fiyatın ima ettiği
değerlerdir — ima edilen 10 yıllık gelir, ima edilen olgun FCF marjı ve (TAM biliniyorsa)
ima edilen TAM payı. "Fiyat şunu varsayıyor" cümlesi, ileri projeksiyon yapmaktan daha
güvenilir bir kanıttır çünkü tahmin gerektirmez.

**Verdict dili:** Fiyat bull senaryosunun üst bandı İÇİNDEYSE (base bandın üstünde ama
bull bandını aşmıyorsa) "PAHALI" DENMEZ; bunun yerine "YÜKSEK BEKLENTİ FİYATLANMIŞ" denir —
bu bir uyarı tonudur, "PAHALI"nın kırmızı alarmından ayrıdır. "PAHALI" ancak fiyat bull
bandının da üstüne çıkıp o iması da aşıldığında kullanılır.

**İskonto oranı da olgunlaşarak fade eder:** Yukarıdaki gelir/marj fade'i, şirketin
steady-state'e ulaştığını varsayıyor — ama şirket steady-state'e ulaşırken riskini SABİT
yüksek bir oranla (bear/base/bull için %14/%12/%10) sonsuza dek iskontolamak tutarsızdır;
değerin büyük kısmı zaten uzak yıllarda ve uçtaki (terminal) değerdedir, sabit yüksek oran
tam da o kısmı eziyor. Motor artık her senaryonun kendi başlangıç (kohort) oranından, ortak
bir "olgun" orana (base senaryonun kendi CAPM-farkındalı iskonto oranı, ERP-spread
korumasıyla tabanlı) steady-state yılına kadar lineer olarak iner; iskonto faktörleri
kümülatif çarpım olarak hesaplanır ve uçtaki (terminal) değer de olgun orana göre
iskontolanır. Bu yalnızca hiper-grower yoluna uygulanır — olgun ve orta-büyüme revenue-first
yolları zaten CAPM-farkındalı, kırpılmış varsayım hattından iskonto oranı aldığı için
fade'e ihtiyaç duymaz.

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

**Bakım/büyüme CapEx ayrımı (CapEx-yoğun hiper-grower'lar):** Yukarıdaki bastırma
guardrail'i doğru ama muhafazakâr bir önlemdir — negatif bir bandı basmayı reddeder, ama
veri merkezi yatırımcısı APLD gibi CapEx-ağır, fiilen finanse edilebilir bir büyüyücüyü
hiçbir DCF manşeti olmadan bırakır. Motor artık semptomu korumakla birlikte kök nedeni de
ele alıyor: `_build_hyper_growth`'ın başlangıç FCF marjı `(İşletme Nakit Akışı − TOPLAM
CapEx − SBC) / Gelir` olarak hesaplanır. CapEx'i gelirin kat kat üzerinde olan bir
şirkette bu marj derinden negatif çıkar — ama bu CapEx'in büyük kısmı **büyüme**
CapEx'idir: revenue-first projeksiyonun büyüme yolu üzerinden ZATEN yakaladığı geleceğin
gelirini inşa eden yatırımdır. Bunu başlangıç marjından düşmek aynı büyümeyi ÇİFT
cezalandırır — bir kere bugünün nakit çıkışı olarak, bir kere de projeksiyonun kendisinin
zaten fiyatladığı kaybedilmiş terminal nakit akışı olarak.

Motor bu fazlalığı, amortisman (D&A) — Damodaran'ın standart bakım-CapEx vekili, ama
**sektörün kendi Cap Ex/Satış oranı (varsa) ya da yoksa gelirin en az %5'i tabanıyla** —
kullanarak ayırır: `bakım_capex = max(d&a, bakım_oranı·gelir)`, `büyüme_capex = capex −
bakım_capex`, ve düzeltilmiş "işletme" marjı = başlangıç marjı + `büyüme_capex / gelir`.
(Taban gerekli: bir veri merkezi büyürken bugünkü D&A, olgun filonun bakım yükünü
olduğundan az gösterir; taban, büyüme-CapEx'inin fazla iyimser hesaplanmasını engeller.)
`bakım_oranı`, Damodaran'ın sektörel Cap Ex/Sales verisinden gelen `capex_sales`
kolonuyla (`data/damodaran/multiples.csv`, isteğe bağlı) eşleşen sektör için doldurulur —
veri merkezi/telekom/enerji gibi bakım-yoğun sektörler için düz %5'ten daha isabetli bir
taban sağlar; sektör verisi yoksa (ya da eşleşmezse) düz %5'e geri döner. Bu ayrım yalnızca
İKİ koşul birden sağlandığında yapılır: CapEx/gelir %30'u aşıyor VE CapEx bu tabanlı bakım
seviyesini aşıyor.

**Ama bu düzeltme MANŞET DEĞİLDİR.** Finansal bir inceleme şunu ortaya koydu: büyüme
CapEx'ini başlangıç marjından silerken projeksiyonun geliri büyütmeye devam etmesi, ciro
rampasını hanesine yazıp o rampayı finanse eden CapEx'i HİÇBİR yere yazmamak demektir —
tek yönlü bir aşırı-değerleme (§3c'nin bilinçle reddettiği "owner-earnings geri-ekleme"
çift-sayımının aynısı). "Doğru" düzeltme (büyümeye bağlı bir reinvestment gideri) ise bu
isimler için güvenilir değildir; lumpy/ileriye dönük CapEx'te tek yıllık sales-to-capital
oranı aşırı oynaktır. Bu yüzden:

- Manşet senaryolar bugünkü GERÇEK (düzeltilmemiş) marjı kullanmaya devam eder; dolayısıyla
  CapEx-ağır isimler yukarıdaki negatif-değer bastırma guardrail'ine takılıp manşetten
  düşer — dürüst, muhafazakâr davranış.
- Düzeltilmiş marj YALNIZCA ayrı, açıkça "AGRESİF ÜST-SENARYO (manşet değil)" olarak
  etiketlenen bir baz-senaryo değeri üretmek için kullanılır; kullanıcı capex normalleşirse
  ima edilen iyimser değeri görür ama bu asla verdict'in temeli olmaz.
- Olgun hedef marj floor'u da her zaman GERÇEK (düzeltilmemiş) marjla hesaplanır, böylece
  düzeltilmiş bir marj terminal marja sızamaz.

Mid-growth yolu (§4b) bu düzeltmeyi bilinçli olarak KULLANMAZ — orası savunulabilir (agresif
değil) bir değer hedefler; CapEx-ağır bir orta-büyüyücü bastırılırsa doğrudan çarpanlara
düşer. Tam mekanik: SPEC.md §3.6.

## 4b. Orta-büyüme (%12-20 gerçekleşen CAGR), zarar eden şirketlerde revenue-first DCF

§2'deki "Zarar eden / hiper-growth" satırı, §4a'nın hiper-grower tetikleyicisini
(gerçekleşen CAGR > %20 VE FCF/marj/Ar-Ge-SBC koşullarından biri) geçen şirketleri
kapsar. Ama gerçekleşen ciro CAGR'ı %20 eşiğinin altında kalan (kabaca %12-20 aralığında),
yine de zarar eden bir `growth_unprofitable` şirket için motor eskiden doğrudan
multiples-only (yalnızca P/S) bir manşete düşüyordu — bilinçli bir tercihti, spekülatif bir
DCF değeri üretmek yerine. Ama hiper eşiğini geçmeyen bu şirketler yine de gerçek, anlamlı
bir büyüme hikâyesi taşıyor; standart FCF-DCF (bastırılmış FCF'i sabit oranla büyütmek)
onlar için de yasaktır, ama salt çarpanlara dayanmak da hikâyenin büyüme kısmını hiç
sayısallaştırmadan masada bırakır.

Bu boşluk, §3c'nin olgun revenue-first DCF'i ile §4a'nın hiper-grower revenue-first
DCF'i ARASINA oturan üçüncü bir revenue-first varyantla dolduruluyor: 8 yıllık bir fade
ufku (olgun yoldan uzun, hiper yoldan kısa), %20 tavanlı brüt-marj-türevli bir olgun hedef
marj (olgun yolun %15 tavanından yüksek, hiper yolun %30 tavanından düşük — bu şirketler
olgun bir filer'dan daha genç ama en agresif hiper-grower'dan daha az spekülatif), hiper
yoldaki gibi ZORUNLU seyreltme/finansman hisseleri (bir orta-büyüme zarar eden şirket yine
nakit yakımını sermaye ihracıyla finanse eder — olgun yolun sıfır seyreltme
varsayımından farklı olarak) ve kırpılmış (clamped) varsayımlar hattından gelen iskonto
oranları (zarar eden şirket tabanı: en az %10). Başlangıç marjı, olgun yoldaki gibi son 3
yılın SBC-düzeltilmiş FCF marjı medyanıdır; §4a'nın bakım/büyüme CapEx ayrımı burada
BİLİNÇLİ OLARAK uygulanmaz (o düzeltme yukarı-yanlı bir üst-senaryodur, bu yol ise
savunulabilir bir değer hedefler).

Büyüme kapısı: gerçekleşen CAGR %12'nin altındaysa veya terminal büyümenin altındaysa
(fade edecek gerçek bir büyüme yoksa) yöntem hiç denenmez, manşet multiples'ta kalır.
Bastırma guardrail'i burada da geçerlidir — baz senaryo hisse başı ≤ $0 çıkarsa yöntem geri
çekilir ve mevcut multiples-only davranış korunur. Duyarlılık
tablosu ve ters-DCF, §3b/§3c ile aynı mantıkla ikincil (bastırılmış) FCF-DCF tabanını
göstermeye devam eder; ters-DCF'in referans büyümesi gelir CAGR'ıdır (yöntem gelir-temelli
olduğu için). Üçgenleme güveni burada da en fazla ORTA ile sınırlanır (DCF ve ters-DCF
bacakları aynı revenue-first modelden türer, bağımsız kanıt sayılmaz). Tam mekanik:
SPEC.md §8d.

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
  **Kodda:** motor `net borç / FAVÖK ≥ 1.0` olan (`financial`/`reit`/
  `growth_unprofitable` hariç her sektörde) bir filer'ı "borçlu" sayar ve
  triangülasyonun birincil kendi-tarihi çarpanını P/E yerine **EV/EBITDA**
  (FD/FAVÖK) yapar — EV/EBIT(DA) sermaye-yapısı-nötrdür (pay net borcu ekler,
  payda faiz öncesidir), dolayısıyla kaldıraçlı bir şirketi P/E'nin taşıdığı
  kaldıraç çarpıtması olmadan sıralar. EV/EBITDA tarihçesi yoksa eski
  P/E→P/S→P/FCF sırasına düşer. Borçlu filer'da P/E-tabanlı PEG ekseni
  atlanır (P/E'nin güvenilmez sayıldığı yerde bir P/E-vs-PEG ayrışması
  EV/EBITDA okumasını bastırmamalı); Damodaran EV/EBITDA medyanı olmadığı
  için sektör-göreli eksen-b de devre dışı kalır (REIT'in P/FFO'su gibi).
  EV/EBIT her zaman yalnızca bilgi amaçlıdır (hiçbir zaman sinyal değil).

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

## 9. Kalibrasyon metodolojisi (`calibrate` CLI, normalizasyon çabası)

Bu bölümdeki kurallar (§1-§8) tek tek savunulabilir, ama katmanlı uygulandığında
(her biri kendi payına "muhafazakâr" bir kırpma/tavan eklerse) sistematik bir
düşük-değerleme yaratabilirler — her kısıt kendi başına makul görünür, ama üst
üste binince motor piyasa fiyatının çok altında bir manşet üretmeye eğilimli hale
gelir. 2026-07'de yürütülen bir normalizasyon çabası bunu ÖLÇÜLEBİLİR hale
getirdi: `sec_analyzer/calibrate.py` + `python -m sec_analyzer.cli calibrate`
komutu.

**Ne yapar:** ~28 hisseden oluşan sektör-çeşitli bir sepet (AAPL MSFT GOOGL AMZN
META NVDA JPM BAC O PLD XOM CVX JNJ PFE PG KO CAT DE MU CRM ADBE RDDT PLTR UBER
SHOP WMT COST VZ) üzerinde, `provider="script"` (LLM'siz, tam deterministik)
ile `analyze` komutunun attığı adımları (fetch/normalize → fiyat/teknik →
metrikler → kırmızı bayraklar → SEC submissions/SIC → `interpret`) baştan sona
tekrarlar — yani ölçülen yol, üretimde gerçekten çalışan yolun (CAPM dahil)
birebir aynısıdır. Her hisse için `ratio = fair_value_range.base'in (lo+hi)/2'si
/ güncel fiyat` hesaplanır; medyan/ortalama/p25/p75 ve üç kova (<0.8 ucuz /
0.8-1.2 makul / >1.2 pahalı) özetlenir; sonuç `reports/calibration_<etiket>_
<zaman>.json` olarak kaydedilir.

**Nasıl çalıştırılır:**
```
python -m sec_analyzer.cli calibrate [--tickers AAPL,MSFT,...] [--label baseline] [--years 5] [--no-cache]
```
`--tickers` verilmezse varsayılan ~28 hisselik sepet kullanılır; `--label`
yalnızca kaydedilen JSON dosyasının adını etkiler (varsayılan `"run"`).

**Oran/medyan/kova ne anlama gelir:** `ratio` 1.0'a yakınsa motorun manşet
makul değeri piyasa fiyatına yakın demektir — bu tek başına "doğru" ya da
"yanlış" anlamına gelmez (aradaki fark gerçek bir yatırım fırsatı da olabilir),
ama BASKET medyanının sistematik olarak 1.0'ın belirgin altında (ör. ~0.4)
takılı kalması, tek tek hisselerin değil motorun KENDİSİNİN sistematik bir
düşük-değerleme yanlılığı taşıdığına işaret eder — bu tam olarak bu çalışmanın
tespit ettiği durumdu. Sağlıklı bir kalibrasyon hedefi: medyan ~0.9-1.1
aralığında VE geniş dağılım (kovalar arasında dağılmış) — medyanın 1.0'a çok
sıkı kenetlenmesi de kuşkulu olurdu (motorun fiyata ÇAPALANDIĞI, bağımsız
hesaplamadığı anlamına gelebilir).

**Ölçülen yörünge (bu sepet, script provider, 2026-07-16/17):**

| Aşama | Medyan | Ortalama | p25 | p75 | <0.8 / 0.8-1.2 / >1.2 |
|---|---|---|---|---|---|
| WP2 sonrası (kalibrasyon aracında bir hata vardı: `submissions` geçilmiyordu → CAPM hiç devrede değildi) | 0.390 | 0.567 | 0.284 | 0.559 | 18/1/3 (n=22) |
| WP2b hata düzeltmesi + temsili yeniden ölçüm (CAPM+sektör medyanları devrede) | 0.768 | 0.891 | 0.443 | 1.030 | 13/9/4 (n=26) — **en büyük tek kazanç** |
| WP3 (hiper iskonto fade) | 0.768 | 0.899 | 0.476 | 1.030 | 13/9/4 (n=26) — sepette etkisi yok (sepette hiper-grower isim az) |
| WP4 (marj tavanı → bayrak) | 0.768 | 0.899 | 0.476 | 1.030 | 13/9/4 (n=26) — sepette etkisi yok |
| WP5 (büyüme cap %40→%60, justified P/B clamp→bayrak) + LEVER'lar sonrası (final) | **0.925** | 0.940 | 0.476 | 1.065 | 11/11/4 (n=26) |

**En büyük tek kazanç WP2b'ydi** — bir float-sınır tutarsızlığı (bkz. SPEC.md
§3'ün `_ERP_SPREAD_EPS` notu) `clamp_assumptions`'ın az önce geçerli kıldığı
bir CAPM tabanlı iskonto oranını `validate_assumptions`'ın hemen ardından
spuriously reddetmesine yol açıyordu — bu, düşük-beta birçok şirket için
CAPM'i sessizce devre dışı bırakıp düz %10 varsayılana düşürüyordu. Bu TEK
hata, ölçülen düşük-değerlemenin BASKIN nedeniydi.

**Kalan aykırı değerler (dispersion, muhafazakârlık değil — ayrı kök nedenler):**
- **KO (Coca-Cola), oran ~0.35:** iki bağımsız neden birleşiyor — (1) `fcf0`
  gerçekten baskılı (bir seferlik IRS vergi depozitosu; motorun tek-yıl-sıçrama
  koruması bunu YAKALAMIYOR çünkü düşüş "yapısal" görünen monoton bir örüntü
  oluşturuyor — bu, normalize edilmiş bir `fcf0` ihtiyacını işaret ediyor,
  aşağıya bakın), (2) SIC eşleştirmesi (LEVER 2 ile "Beverage (Soft)" alias'ı
  eklendi, artık eşleşiyor) düzeldi ama CAPM+terminal-büyüme birlikte hâlâ
  tam telafi etmiyor.
- **MU (Micron), oran ~0.10:** döngüsel + sermaye-yoğun sürdürülebilir-büyüme
  FCFE çapası (§3e) matematiksel olarak doğru hesaplanıyor (ROE özkaynak
  maliyetinin üzerinde, büyüme değer ekliyor) — ama motorun ürettiği manşet
  ile MU'nun güncel piyasa fiyatı (bu yazının tarihinde ~$876) arasındaki
  makul-değer/fiyat oranı çok düşük kalıyor. Bu, modelin bir hatası mı yoksa
  fiyatın kendisinin bir bellek-süper-döngüsü fiyatlaması mı olduğu açık bir
  soru — kullanıcı onayı/girdisi bekliyor.
- **XOM, VZ:** kalibrasyon sepetinde "skipped" (makul-değer bandı üretilemedi
  — XOM için yıllık veri eksikliği, VZ için negatif `fcf0`), ayrı bir veri
  boşluğu, kalibrasyon mantığının bir hatası değil.

**Açık kalan risk (LEVER 4, finansal inceleme tarafından tespit edildi):**
standart iki-aşamalı DCF'in artık `high_growth_flag`'i var (`growth_5y >
%40`), ama bu bayrak yalnızca RAPORLAMA amaçlıdır — hiper-grower/orta-büyüme
revenue-first yollarındaki "varış noktası" (TAM payı/gelir çarpanı) güvenlik
ağına sahip DEĞİLDİR. `script` sağlayıcısında bu şu an etkisiz kalıyor
(`rule_based._default_growth_anchor` kendi büyüme varsayımını zaten %25'e
kırpıyor), ama bir LLM sağlayıcısının (kendi kırpması olmadan) genç,
yüksek-büyümeli bir şirketi standart DCF'e yönlendirmesi hâlâ güvenlik ağı
olmayan bir senaryodur. Bkz. ROADMAP.md.

## 10. Geçmiş tarih (as-of) modu

`analyze TICKER --as-of YYYY-MM-DD` yukarıdaki tüm kuralları (DCF, EPV,
hiper-grower, çarpanlar, üçgenleme) DEĞİŞTİRMEDEN, yalnızca girdileri o
tarihte bilinebilecek olanlarla sınırlayarak çalıştırır: SEC faktları
(dosyalanma tarihine göre), fiyat geçmişi (o tarihe kadar) ve makro (ERP/
risksiz getiri) `data/damodaran/erp_history.csv` arşivinden ve FRED'in geçmiş
DGS10 serisinden gelir — sektör çarpanları ve betalar ise (§4'teki CAPM
girdisi) tarihe göre arşivlenmemiştir, güncel Damodaran anlık görüntüsü
kullanılır (bilinen bir yaklaşıklık, gizlenmez). Bu, §9'daki kalibrasyon
aracının geçmiş piyasa rejimlerinde (ör. 2021 zirve vs. 2022 dip) de
çalıştırılabilmesini sağlar. Tam sözleşme, fonksiyon imzaları ve sınırlamalar:
SPEC.md §18; pratik kullanım: METODOLOJI.md §7.
