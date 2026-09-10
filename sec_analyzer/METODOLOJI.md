# METODOLOJI.md — Hisse Analiz Çerçevesi

> Hisse-agnostik ve kullanıcı-agnostik analiz sistemi. Interpret katmanının system prompt'una
> İLK sırada eklenir; ardından VALUATION.md (sayısal yöntem kuralları) ve PROFIL.md
> (kullanıcıya özgü profil, limitler ve davranışsal notlar) gelir.
> İş bölümü: Bu dosya ÇIKTININ YAPISINI ve KARAR DİSİPLİNİNİ tanımlar.
> Fair value sayılarının NASIL hesaplandığı VALUATION.md + valuation engine'in işidir;
> kullanıcıya özgü her şey PROFIL.md'nin işidir.

---

## 1. Analiz çıktısının zorunlu bölümleri (sıralı)

1. **Durum özeti** — şirket ne yapıyor, güncel fiyat, son çeyreğin tek cümlelik özeti, aktif katalizörler (tarihli).
2. **Fair value bandı** — valuation engine'den gelen bear/base/bull; her senaryonun varsayımı görünür. Cyclical trap kontrolü sonucu burada raporlanır (VALUATION.md §3 kuralları).
3. **İkili ucuzluk verdict'i** — HER ZAMAN ayrı ayrı:
   - **Fundamental:** UCUZ / MAKUL / PAHALI (fiyat vs base bandı + üçgenleme + güven seviyesi)
   - **Teknik:** AŞIRI SATIM / NÖTR / AŞIRI ALIM (kural bazlı modülden; vade ağırlığına göre öne çıkar veya not seviyesine iner)
   İkisi çelişebilir ve çelişki GİZLENMEZ ("fundamental ucuz ama teknik olarak bıçak düşüyor" geçerli bir sonuçtur).
4. **Senaryo tablosu** — bear/base/bull (+ gerekirse tail): her satırda hedef fiyat, güncel fiyattan % getiri, senaryonun tetikleyicisi.
5. **Kademeli giriş planı** — tek plan, en fazla 5 tranche'lık tablo, toplam boyut ~%100. İki yönlüdür:
   - **Birikim (dip) tranche'ları** — seviye güncel fiyatın altında/eşiğinde. Tetik koşulu fiyat DEĞİL, koşuldur: "günlük kapanış X'in ALTINA inerse". Tetik cümlesi seviyenin kaynağını da adlandırır (ör. "baz senaryo alt bandı", "SMA200 desteği", "52 hafta dibi"). Tüm dip tranche'ları TEK ortak yapısal invalidation paylaşır (bear.lo / 52 hafta dip'in altında bir tampon). Teknik okumadaki destek bölgeleri (`support_levels`) bilinçli olarak dip adayı DEĞİLDİR — dip seviyeleri değerleme-çapalı kalır; bir dip seviyesi teknik destek bölgesine denk gelirse (bölgenin low/high bandı, %2 dedupe toleransıyla genişletilmiş) tranche'a yalnızca bilgilendirici bir not düşülür: "Teknik destek bölgesiyle örtüşüyor (lo-hi USD)." — seçim, boyutlandırma, invalidation ve R:R bu nottan etkilenmez.
   - **Yükseliş teyidi (breakout) tranche'ları** — seviye güncel fiyatın üzerinde (SMA50/SMA200 geri alımı, direnç/önceki zirve kırılımı, 52 hafta zirve kırılımı gibi kaynaklardan). Tetik koşulu: "günlük kapanış X'in ÜZERİNE çıkarsa (yükseliş teyidi)" — örn. "X seviyesinin günlük kapanışla geri alınması". Her breakout tranche'ı KENDİ "başarısız kırılım" invalidation'ını taşır (kırılan seviyenin hemen altı); bir breakout'un iptali diğer tranche'ları geçersiz kılmaz. Bir breakout tetik seviyesi modelin kendi bull.hi (yoksa base.hi) hedefinin ÜZERİNDE olabilir — bu "model üstü" tranche'lar yine de KORUNUR (trend-takip eklemesidirler) ama işaretlenir: `rr = None` raporlanır ve `note` alanına "Model üstü: tetik seviyesi model bull hedefinin üzerinde; değer-çapalı R:R tanımsız -- yalnızca trend-takip girişi" notu düşülür, çünkü raporlanacak değer-çapalı bir ödül yoktur.

   Seçim: her iki yönde de aday varsa en az birer tranche garanti edilir, kalan slotlar fiyata en yakın seviyelerden doldurulur (sadece tek yönde aday varsa o yönden en fazla 5 alınır). Boyutlandırma: en ucuz (en düşük fiyatlı) tranche en büyük payı alır. Her tranche için ortak hedef ve **per-tranche R:R** (kendi invalidation'ına göre) raporlanır. "Düşük fiyatlı tranche'lar daha yüksek R:R sunmalı" kuralı SADECE birikim (dip) merdiveninin ardışık tranche'ları arasında geçerlidir — bunlar ortak invalidation'ı paylaştığı için R:R fiyat düştükçe monoton artar. Breakout tranche'ları kendi dar/kendine-özgü invalidation'larıyla bu ölçekte karşılaştırılamaz; dolayısıyla mekanik "R:R ters" uyarısı yalnızca ardışık dip tranche'ları arasında uygulanır, breakout tranche'larına veya dip/breakout çiftlerine uygulanmaz.

   Kabul edilen tasarım tradeoff'u (Finding 1): boyutlandırma dip ve breakout tranche'larını ayrı ayrı değil, TEK ~%100'lük fiyata-göre-azalan sırada birleştirir (en ucuz tranche, yönü ne olursa olsun, en büyük payı alır). Bilinçli olarak kabul edilen iki sonucu var: (a) en derin dip tranche'ları TEK paylaşılan (uzak) yapısal stop'u taşıdığından, dar kendi-stop'lu breakout tranche'larından daha düşük R:R sunabilirler — bu yüzden "düşük fiyat → yüksek R:R" monotonluğu yukarıda belirtildiği gibi SADECE dip merdiveni içinde geçerlidir, dip/breakout karşılaştırması için değil; (b) gerçek bir fiyat hareketi tek yönlüdür (ya düşüş ya yükseliş), dolayısıyla gerçekleşen tek yol ~%100'ün yalnızca kendi tarafına düşen kısmını devreye sokar, tamamını değil.

   Momentum bağlam katmanı (bkz. §8) dip tranche'larını doğrudan etkiler: fundamental UCUZ verdict'i negatif fiyat momentumuyla çakışırsa ("düşen bıçak"), dip merdivenine paylaşılan bir stabilizasyon ön-koşulu eklenir — tranche boyutlandırması ve invalidation mantığı yukarıdaki gibi kalır, yalnızca tetiğe bir ek koşul iliştirilir.
6. **Stop-adding sinyalleri** — hangi koşullar gerçekleşirse yeni tranche AÇILMAZ (tez metriği bozulması, invalidation'a yaklaşma, konsantrasyon limiti).
7. **Tez doğrulama metriği** — hisse başına TEK çapa metrik, ilk analizde tanımlanır, her çeyrek kontrol edilir. Örnekler: bellek üreticisi → gross margin; SaaS → NRR; banka → NIM; pre-profit story hissesi → revenue re-acceleration. Metrik iki ardışık çeyrek tezin aksini gösterirse tez GEÇERSİZ sayılır ve bu açıkça söylenir.
8. **Özet** — 2-3 cümle, eyleme dönük, %'li ve R:R'lı.

## 2. Sunum kuralları

- Tablolar tercih edilir; verdict'ler net etiketli ve birbirinden görsel olarak ayrık.
- Tüm risk ve getiriler HEM fiyat HEM yüzde olarak yazılır.
- Kısa, eyleme dönük düzyazı; süsleme yok.
- Intraday fiyat hareketi ASLA tetik sayılmaz — sadece günlük kapanış geçerlidir. (Fiyatın gün içinde tetik seviyesine dokunup altında kapatması, tetiğin ÇALIŞMADIĞI anlamına gelir.)
- İşlem maliyeti (komisyon, iki bacak) R:R hesaplarına dahil edilir.

## 3. Karar disiplini kuralları

- Tetik dolmadan yapılmış bir giriş analiz edilirken: pozisyon reddedilmez, ama daha sıkı stop + hedefe bağlı zorunlu çıkış tarihi ile yeniden çerçevelenir.
- Binary katalizör (earnings, FDA kararı, launch) öncesi tetiksiz pozisyon büyütme önerilmez; katalizör tarihi her analizde hatırlatılır.
- Story/momentum hisselerinde (negatif margin, negatif book value) fundamental verdict'in "NOT cheap" çıkması normaldir; tez tamamen katalizöre bağlıysa "spekülatif" olarak etiketlenir, value diliyle aklanmaz.
- Kullanıcıya özgü davranışsal kalıplar ve uyarı kuralları PROFIL.md'de tanımlanır; oradaki notlar profil uyumu verdict'inde uygulanır.

## 4. Vade entegrasyonu

Verdict ağırlıkları analizde belirtilen vadeye göre değişir (3m: teknik %70 · 1y: %50/50 · 5y: fundamental %80). Aynı hisse farklı vadelerde farklı verdict alabilir ve bu tutarsızlık değil, sistemin özelliğidir. Cyclical trap kontrolü 5y vadede zorunludur.

## 5. Pozisyon bağlamı (POZISYONLAR.md varsa)

- Açık pozisyon varsa: tez metriğinin son durumu, ortalama maliyet, hangi tranche'ların dolu olduğu ve bir sonraki tetik raporlanır.
- Pozisyon bağlamı YORUMU kişiselleştirir, SAYILARI asla değiştirmez (VALUATION.md §8 ile aynı ilke).
- Konsantrasyon: PROFIL.md limitlerine göre yeni pozisyonun portföy/sektör payına etkisi belirtilir.

## 6. Dürüstlük kuralları

- Veri eksikse (yeni halka arz, ADR, kısa geçmiş) bant genişletilir ve güven düşürülür; kesinlik taklidi yapılmaz.
- Model bir önceki analizde yanıldıysa (invalidation çalıştı, tetik hatalıydı) bu saklanmaz, bir sonraki analizde açıkça not edilir.
- Hiçbir çıktı yatırım tavsiyesi değildir; mekanik referans çerçevesidir. Nihai karar kullanıcınındır.

## 7. Geçmiş tarih (backtest / otopsi) modu

`--as-of YYYY-MM-DD` bayrağı, motoru o tarihte BİLİNEBİLECEK verilerle
çalıştırır: yalnızca o tarihten önce dosyalanmış SEC faktları, o tarihe kadarki
fiyat geçmişi ve o tarihin ERP/risksiz getiri arşivi kullanılır — sonradan
gelen düzeltmeler (restatement), sonraki fiyat hareketi ve bugünün makro
verisi modele hiç girmez. Amaç: "model o gün ne derdi, sonradan ne oldu"
sorusunu cevaplayabilmek — bir tez otopsisi ya da geriye dönük kalibrasyon
aracı, canlı analizin yerine geçmez.

**Örnek kullanım:**
```
python -m sec_analyzer analyze BYND --as-of 2020-06-30
python -m sec_analyzer analyze AMZN --as-of 2015-12-31
```
Bu, "BYND'yi 2020 verisiyle, AMZN'i 2015 verisiyle analiz et" senaryosudur —
o tarihte elde olan bilgiyle bugünkü motorun ne söylerdi'sini görmek,
ardından gerçekleşen fiyat hareketiyle karşılaştırmak için kullanılır.

**Kalibrasyon karşılaştırması (2021-zirve vs 2022-dip):**
```
python -m sec_analyzer calibrate --as-of 2021-11-19 --label peak2021
python -m sec_analyzer calibrate --as-of 2022-10-14 --label trough2022
```
Aynı sepeti iki farklı piyasa rejiminde (zirve ve dip) çalıştırıp
`reports/calibration_peak2021_*.json` / `reports/calibration_trough2022_*.json`
medyan oranlarını karşılaştırmak, motorun KENDİ muhafazakarlığı ile o anki
piyasa rejiminin etkisini birbirinden ayırmayı sağlar (§9'daki kalibrasyon
metodolojisiyle aynı araç, farklı zaman noktalarında).

**Bilinmesi gerekenler:**
- Analist konsensüs hedefleri (yfinance) tarihsiz veridir; `--as-of` modunda
  hiç çekilmez, raporda bir Türkçe notla belirtilir.
- Finansallar veritabanına YAZILMAZ (`financials`/`ratios` tabloları güncel
  görünümdür); yalnızca `verdicts` tablosuna, o kaydın hangi tarihte
  ("otopsi" mi "canlı" mı) üretildiğini ayırt eden bir `as_of` sütunuyla
  kaydedilir.
- En riskli bilinen sınırlama: fiyat verisi (yfinance) bugüne göre
  split-ayarlıdır — `as_of`'tan SONRA gerçekleşen bir hisse bölünmesi (ör.
  NVDA'nın 2024'teki 10:1 bölünmesi), o bölünmeden önceki bir tarih analiz
  edildiğinde piyasa değeri/çarpanları bölünme oranı kadar çarpıtır.
- Sektör çarpanları ve betalar (`multiples.csv`) tarihe göre arşivlenmemiştir;
  yalnızca ERP ve risksiz getiri geçmişe göre alınır — bu SPEC.md §18'de tam
  olarak belgelenmiştir.

Tam teknik sözleşme (fonksiyon imzaları, veri kaynağı önceliği, sınırlamalar):
SPEC.md §18.

## 8. Momentum bağlamı (context layer)

> **Değişmez kural:** Momentum HİÇBİR ZAMAN fair value / valuation hesaplamasına
> girmez. "Ne kadar eder" (fair value, VALUATION.md'nin işi) ve "ne zaman ve
> nasıl girilir" (kademeli giriş, §1.5) ayrı tezlerdir; momentum yalnızca
> İKİNCİ soruyu besler — value yakınsaması ve trend devamı farklı iddialardır
> ve biri diğerini geçersiz kılmaz. Tamamen deterministiktir (LLM yok, wall-
> clock bağımlılığı yok); üç alt katman da saf ve savunmalıdır (hata fırlatmaz,
> eksik veride `None`'a düşer, kartta "—" görünür).

Üç bağımsız alt katman, terminale/HTML rapora tek bir **MOMENTUM** satırı ve
`verdicts` tablosuna bir `momentum_verdict` kolonu olarak yansır:

1. **Fiyat momentumu** (`technical/momentum.py::compute_price_momentum`) — 0-100
   bileşik skor: çok-ufuklu getiriler (3a/6a + klasik 12-1 momentum,
   volatiliteye göre normalize), SPY'a ve sektör ETF'ine (SIC → ETF eşlemesi
   aynı modülde) göre relatif güç, trend kalitesi (SMA50/200 eğimi, 52 haftalık
   zirveye uzaklık) ve yukarı/aşağı hacim oranı; RSI/MACD yalnızca hafif bir
   dürtme olarak katılır. 3 aylık ufuk anlatısı bu skorla açılır (§4); 5 yıllık
   ufukta ise yalnızca bir zamanlama dipnotuna iner ve tez için belirleyici
   sayılmaz (`technical/verdict.py`).
2. **Fundamental momentum** (`signals/momentum.py::compute_fundamental_momentum`)
   — işin *ikinci türevi*: çeyreklik YoY gelir büyümesi hızlanıyor mu/sabit mi/
   yavaşlıyor mu, brüt ve FCF marjı trendi, ve bir "model bazlı sürpriz"
   (gerçekleşen gelirin, önceki saklı analizin baz senaryosunun o ana kadar
   öngördüğü seviyeye göre sapması). Çeyreklik + YTD-kümülatif XBRL satırları
   karışık geldiğinde gerçek tek-çeyrek değerleri `normalize/normalizer.py::
   to_quarterly_series` türetir (YTD farklama + üç çeyrek + yıllık varsa
   Q4 = yıllık − Q1..Q3). Model-sürpriz sinyali hiper-grower senaryolarında
   artık kalıcı hale getirilen `revenue_path`/`base_revenue`/
   `steady_state_year` alanlarına dayanır (`valuation/engine.py`
   `scenarios_detail`); bu alanları içermeyen eski saklı verdict'lerde
   sessizce `None`'a düşer.
3. **Verdict momentumu** (`signals/momentum.py::compute_verdict_momentum`) —
   ardışık saklı LIVE analizlerde modelin KENDİ fair-value/fiyat oranının
   yörüngesi: oran yükselirken fiyat düşüyorsa "yakınsama fırsatı" (model
   ismi giderek daha ucuz buluyor); oran düşerken adil değer fiyata doğru
   eriyorsa "zayıflayan tez". Ekstra veri toplamaya gerek yok — `verdicts`
   tablosu doldukça kendiliğinden gelir.

Üçü `signals/momentum.py::synthesize_momentum` içinde tek bir
`result["momentum"]` verdict'inde birleşir (GÜÇLÜ+ / POZİTİF / NÖTR / NEGATİF)
ve şu **çapraz sinyalleri** üretir:

- **UCUZ + negatif fiyat momentumu → "düşen bıçak" uyarısı.** Kademeli giriş
  planının dip tranche'larına ortak bir stabilizasyon ön-koşulu eklenir
  (bkz. §1.5).
- **PAHALI + GÜÇLÜ+ momentum → profil güvenlik uyarısı.** "Momentum cazibesi
  yüksek, değerleme tetiği yok, plan dışı alım riski" olarak işaretlenir —
  profil uyumu tarafına bir uyarı olarak akar (bkz. §3, PROFIL.md).
- **UCUZ + pozitif fundamental momentum (özellikle model-beat ile) → en güçlü
  kombinasyon.** Tranche planını öne çekmek için gerekçe sayılabilir.

Katman `cli.py` / `web/app.py` içinde interpret adımından SONRA, olaylar
(events) katmanıyla aynı desende bağlanır (bkz. `signals/events.py`) ve
`--as-of` (backtest/otopsi) modunda verdict-momentumu ile model-sürpriz alt
sinyalleri devre dışı kalır — yalnızca o ana kadarki fiyat momentumu
kullanılır, böylece geçmişe dönük bir analiz gelecekteki saklı verdict'lere
veya sonraki fiyat hareketine bakmaz (bkz. §7'deki point-in-time ilkesi).
Backtest raporu ayrıca bir (verdict × momentum × vade) isabet-oranı tablosu
ve hisse başına bir verdict-momentum bölümü taşır.

## 9. İçeriden işlem sinyali (SEC Form 4)

> **Kapsam sınırı:** Momentum gibi (§8), bu katman da HİÇBİR ZAMAN fair value
> hesaplamasına girmez. Yalnızca giriş zamanlaması ve tez güveni tarafını
> besler; "ne kadar eder" sorusuna dokunmaz.

**Veri ve önbellek.** Kaynak, SEC Form 4 (ve düzeltmesi 4/A) mülkiyet
dosyalamalarıdır — yönetici/yönetim kurulu üyesi/%10 ortakların, kendi
şirketlerindeki hisse hareketlerini dosyalamak zorunda oldukları belgeler.
`fetch/insider.py`, `analyze` işleminin zaten çektiği `submissions` belgesinin
(`filings.recent.{form,filingDate,accessionNumber,primaryDocument}` paralel
dizileri — aynı belge `fetch/filings.py`'nin kazanç katalizörü ve
`signals/events.py`'nin 8-K taraması için de kullandığı belgedir) içinde
`"4"`/`"4/A"` formlarını arar, her bir dosyalamanın ham `ownershipDocument`
XML'ini indirir ve ayrıştırır. LLM yok, üçüncü taraf veri sağlayıcı yok,
tamamen deterministik — yalnızca SEC EDGAR'a karşı `requests`. Dosyalama
başına önbellek (`sec_analyzer/raw/form4/form4_<accession>.json`, tire'leri
kırpılmış accession numarasına göre içerik-adresli) TTL TAŞIMAZ, çünkü
dosyalanmış bir belge değişmezdir — bir önbellek isabeti asla SEC'e karşı
yeniden doğrulanmaz.

**Hangi işlemler sinyal taşır.** `signals/insider.py::_CODE_MAP`, SEC'in
işlem kodlarını Türkçe etiket + kategoriye eşler. Yalnızca açık piyasa alımı
(`P`, `open_market_buy`) ve açık piyasa satışı (`S`, `open_market_sell`)
BİLGİLENDİRİCİ kodlardır — fiyat hakkında bir görüş ifade ederler. Hisse
ödülü/tahsisi (`A`), opsiyon/türev kullanımı (`M`/`X`/`C`), vergi için hisse
mahsubu (`F`) ve bağış/devir (`G`) tazminat mekaniğidir; kimse bunları
yaparak fiyat hakkında bir görüş bildirmez, dolayısıyla verdict'i hiç
etkilemezler (yine de şeffaflık için `recent` listesinde gösterilirler).
Türev tablosundaki satırlar da ayrıştırılır ve `derivative: True` ile
işaretlenir, ama alım/satış TOPLAMLARINA HİÇ GİRMEZLER. Gerekçe birim
uyuşmazlığıdır: SEC'in `P`/`S` kodları türev tablosunda da geçerlidir (bir
opsiyonun/varantın açık piyasada alınıp satılması), fakat orada `shares`
alanı SÖZLEŞME adedidir, hisse adedi değil — ve fiyatı da sözleşme başınadır.
Bu iki büyüklüğü aynı toplamda birleştirmek, farklı birimleri sessizce
toplamak olurdu. Bu yüzden `buy_count`/`sell_count`, pay/değer toplamları,
alıcı-satıcı kümeleri, kümelenme bayrakları ve pay-oranı (materyalite)
hesabı — dolayısıyla verdict'in tamamı — yalnızca `nonDerivativeTable`
satırlarından hesaplanır. Türev işlemleri yok sayılmaz: `recent` listesinde
gösterilir ve `derivative_buy_count`/`derivative_sell_count` olarak ayrıca
raporlanır; sıfırdan farklı olduklarında notta toplamlara dahil
edilmedikleri açıkça belirtilir — okurun göremediği bir dışlama, hiç var
olmamış veriden ayırt edilemez.

**Alım/satış asimetrisi — bu bölümün merkezi noktası.** Bir içeriden kişi
açık piyasadan hisse ALIRSA bunun tek bir makul açıklaması vardır: kendi
parasıyla, hissenin ucuz olduğuna inanmaktadır. SATMASININ ise fiyatla hiç
ilgisi olmayan pek çok nedeni olabilir — portföy çeşitlendirmesi, vergi
yükümlülüğü, likidite ihtiyacı veya önceden planlanmış bir 10b5-1 satış
planı. Bu yüzden alım bir KANAAT sinyali olarak, satış ise — materyalite
testini geçmedikçe — BAZ ORAN (routine) olarak ele alınır. Bu asimetri iki
tasarım kararına yansır: (1) verdict merdiveni alımı "anlamlı" saymak için
çok daha az kanıt ister — tek bir fiyatlı açık piyasa alımı bile doğrudan
`"ALIM"`e taşır — ama tek başına dolar bazlı satış, ne kadar büyük olursa
olsun, nötr `"SATIŞ AĞIRLIKLI"`nın ötesine geçemez; (2) dolar tutarı tek
başına kötü bir materyalite ölçüsüdür — hâlâ 2 milyar dolarlık payı olan bir
yöneticinin 24 milyon dolarlık satışı önemsizken, aynı dolar tutarı payının
çoğunu elden çıkaran biri için önemlidir.

**Materyalite testi.** `_stake_details`, her satıcı için
`sharesOwnedFollowingTransaction` (işlemden hemen sonra elde kalan hisse
sayısı, Form 4'ün `postTransactionAmounts` alanından) kullanarak
`stake_sold_pct = satılan / (satılan + kalan) * 100` hesaplar — pay ORANI,
dolar değeri değil. Kişi başına oranların MEDYANI `_HEAVY_SELL_STAKE_PCT`
(**%25**) eşiğini geçmeden satış olumsuz etiketi kazanamaz. Neden dolar değeri
değil de kendi payının kesri doğru payda: büyük bir pozisyonu olan birinin
büyük dolarlık satışı rutindir; ama küçük bir pozisyonu olan birinin payının
yarısını elden çıkarması rutin değildir — payda kişinin KENDİ pozisyonu
olduğunda bu ayrım otomatik olarak ortaya çıkar, dolar tutarında çıkmaz.

**Verdict merdiveni** (`_classify_verdict`, ilk eşleşen kural kazanır):

1. En az 2 farklı açık-piyasa alıcısı (`cluster_buy`) → **`GÜÇLÜ ALIM`**
   (positive).
2. En az 1 alım VE alım değeri satış değerinden büyük/eşit → **`ALIM`**
   (positive).
3. En az 1 alım var ama değerce satışlarca geride bırakılmış →
   **`KARIŞIK`** (neutral).
4. `cluster_sell` (≥2 farklı satıcı) VE satıcıların medyan pay-satış oranı
   BİLİNİYOR ve `_HEAVY_SELL_STAKE_PCT` (%25) eşiğine ulaşıyor →
   **`YOĞUN SATIŞ`** (negative). Bilinçli kural: medyan hiçbir satıcı için
   hesaplanamadıysa (kimsenin işlem-sonrası pay bilgisi ayrıştırılamadıysa)
   materyalite KANITLANAMAZ; bu kural ateşlemez ve bir sonraki kurala düşer
   — tahmin yürütülmez.
5. En az 1 satış (miktarı/oranı ne olursa olsun) → **`SATIŞ AĞIRLIKLI`**
   (neutral, DEĞİL negative) — rutin satış çoğu büyük/uzun-kıdemli yönetim
   ekibi için baz orandır, tek başına olumsuz bir olay sayılmaz.
6. Aksi halde → **`NÖTR`** (neutral).

**Nerede görünür.** `verdicts` tablosundaki `insider_verdict` kolonu,
terminal kartındaki `İçeriden:` satırı, HTML raporun "Zamanlama" sekmesindeki
kart ve portföy genel bakışındaki `İçeriden` sütunu — hepsi salt görüntüleme
amaçlıdır (yukarıdaki kapsam sınırı notuna bakınız).

**Zaman-noktası (point-in-time) davranışı.** `--as-of` modunda referans
tarih hem dosyalamaları (`fetch/insider.py`, filing tarihi referans tarihten
sonraysa atlanır) hem de işlemleri (`signals/insider.py`, işlem tarihi
referans tarihten sonraysa atlanır) filtreler — geçmişe dönük bir koşu,
henüz gerçekleşmemiş bir dosyalamayı asla görmez. Pratik sınırlama: pencere
`lookback_days` (varsayılan 180 gün) ile ve indirilen dosyalama sayısı
`max_filings` (varsayılan 40) ile sınırlıdır; çekilebilecek nitelikli
dosyalama sayısı bu tavanı aşarsa sonuç `truncated: True` işaretlenir ve
`İçeriden:` satırına `" (kısmi)"` eklenir — kısmi bir taramayı tam gibi
sunmak yerine bu açıkça belirtilir.
