# Haftalık Trend Kuralı (H2) — devir notu

Son güncelleme: 2026-11-09 (§7 ve §7b sonuçları eklendi). Bu not, "haftalık SMA200 + EMA8/21" stratejisinin araştırmasını başka bir oturumda sürdürmek için hazırlandı. Hiçbir şey commit edilmedi.

## 1. Kullanıcı ve çalışma şekli
- Portföyünü **haftada bir** (cuma kapanışından sonra) kontrol ediyor, stopları haftada bir güncelliyor, **satışı yalnızca stop emriyle** yapıyor. Stopları "bir ortalamanın / desteğin altına" koymayı seviyor.
- Testlerde ilke: **kurallar çalıştırmadan önce yazılır** ve sonuca bakıp değiştirilmez. Her sonuç "işlemlerin kâr/zarar toplamı = son bakiye − başlangıç" kontrolüyle (`chk ≈ 0`) doğrulanır.
- Sonuçlar yatırım tavsiyesi olarak değil, "test edilmiş kural" olarak sunulur; örneklemin küçük olduğu hatırlatılır.

## 2. Mevcut strateji: H2
Günlük grafik, sadece uzun pozisyon, her hisseye eşit pay.

**Alım**
1. Cuma kapanışı **SMA200 üstünde** → sonraki haftanın ilk açılışında (pazartesi) al.
2. Cuma kapanışı **SMA200 altında** → bu cuma EMA8 > EMA21 ve kapanış ikisinin de üstünde **ve** bir önceki cuma da bir alım koşulu sağlanmış (SMA200 üstü ya da EMA koşulu) → **bu cumanın kapanışında** al ("2 cuma teyidi").
- Pazartesi açılışı stopun altındaysa alım yapılmaz; stopla satılan hisse aynı gün yeniden alınmaz.

**Stop** (cuma değerleriyle haftada bir hesaplanır, yalnızca yukarı çekilir; gün içinde dokununca satış, aşağı boşlukta açılıştan)
- Kapanış > SMA50 > SMA200 → **SMA50 × 0,95**
- SMA200 üstünde ama yukarıdaki değil → **SMA200 × 0,95**
- Kural 2 ile alınmış ve fiyat hâlâ SMA200 altında → **EMA21 × 0,95**

**İsteğe bağlı X2 ("Early exit on SMA50 rejection", varsayılan KAPALI)**
Pozisyondayken: 1. gün kapanış SMA50 altında ve SMA200 × 1,10 üstünde, gün içi en yüksek ≥ SMA50 × 0,98; 2. gün kapanış yine SMA50 altında → 2. gün kapanışında sat. Günlük takip gerekir.

## 3. Dosyalar
| Dosya | Ne |
|---|---|
| `tradingview/weekly_ma_strategy.pine` | **Güncel gösterge.** Pine v6, İngilizce, "Weekly Trend Rule (SMA200 + EMA8/21)". H2 + X2 ayarı (kapalı), stop tamponu varsayılan **Volatility** (V10: 1 × 252 günlük ortalama ATR14/kapanış, %2–12; 252 bar yoksa %5; "Fixed %" modu eski kural), kompakt/tam tablo, "Your stop" satırı (kullanıcının gerçek pozisyonu için aday stop), alarmlar. Claude tarafından TradingView'da derlenmedi. |
| `tradingview/daily_ma_strategy.pine` | Eski, H2 öncesi deneme sürümü. Kullanılmıyor. |
| `research/weekly_trend/weekly_trend_backtest.py` | **Tek parça backtest script'i** (aşağıda). |
| `sec_analyzer/technical/weekly_strategy.py` + `WEEKLY_STRATEGY_SPEC.md` + `tests/test_weekly_strategy.py` | Başka bir oturumda (fork) eklendi: motor H2'yi hesaplayıp CLI kartında ve HTML raporda gösteriyor. Orada X2 varsayılan **açık** — göstergeyle tutarlı olması için kapatılması konuşuldu, yapılmadı. Commit edilmedi. |
| Hafıza: `manual-weekly-ma-strategy.md` | Tüm test geçmişinin kısa kaydı. |

## 4. Backtest script'i
```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 python research/weekly_trend/weekly_trend_backtest.py sonuc.json [MOD1,MOD2,...]
```
- Yorumlayıcı: miniconda3'te pandas yok; `Python313/python` (PATH'teki `python`) kullanılmalı.
- 2. argüman virgülle ayrılmış mod listesi (varsayılan `H2,H2_X2_approach`). `sectors` sepeti otomatik olarak `crisis` döneminde de çalışır.
- Fiyatlar `sec_analyzer.fetch.prices.get_price_history` (yfinance, temettü düzeltmeli, önbellekli).
- Dönemler (`PERIODS`): `test` = 2025-11-03 … 2026-11-03, `train` = 2020-11-02 … 2025-10-31, `crisis` = 2007-01-03 … 2012-12-31 (son döngü `for per in ("test", "train")` — crisis için ekle).
- Sepetler (`BASKETS`) ve kurallar (`for name, MODE in (...)` döngüsü) dosyanın sonunda düzenlenir.
- Simülasyon: hisse başına ayrı hesap (sermaye / hisse sayısı), alım payın tamamıyla, %0,1 işlem maliyeti, açık pozisyonlar dönem sonu kapanışından değerlenir. Karşılaştırma: aynı paylarla "al ve tut".
- `ROW ... ret / dd / n / wins | hold ... | chk ...` satırları: getiri %, en büyük düşüş %, işlem sayısı, kazanan, al-tut getirisi/düşüşü, kâr/zarar mutabakatı.
- Mod adlarıyla açılan özellikler (MODE metninde geçerse): `_X2` (X2 çıkışı), `_E1` (SMA200 üstü alımda EMA8>EMA21 şartı), `RB50` / `RBEMA` (satış sonrası yeniden alım yasağı), `_Q1` / `_Q2` (yasağı yalnızca zayıf trendlere uygula), `RCAP` (%15 risk sınırı), `RSIZE` (risk bazlı büyüklük), `_RC` (stop geri alma, §7), `_Bnn` (stop tamponu %nn, ör. `H2_B02`; yoksa %5), `_Vnn` (oynaklık tamponu, k = nn/10, §7c). Hepsi `H2` ile başlamalı.
- Doğrulandı (2026-11-09): script olduğu gibi çalıştırıldığında aşağıdaki §5 tablosundaki 20 sonucun hepsini birebir üretiyor.
- Not: script, geçici klasörlerdeki yamalı bir zincirden üretildi; içinde kullanılmayan kalıntılar (ör. `kalman_trend` fonksiyonu, `ONE_PER_WEEK`, `MONDAY_ONLY`) var. Temizlemek iyi olur ama sonuçları değiştirmemeli — temizledikten sonra aşağıdaki H2 rakamlarını tekrar üretip doğrula.

## 5. Güncel sepetler ve temel sonuçlar
Kullanıcı **MU ve APP'yi testlerden çıkardı** (tek başlarına sonuçları belirliyorlardı).

| Sepet | Hisseler |
|---|---|
| semi_noMU | NVDA AVGO MRVL NBIS ORCL |
| first_noAPP | UBER NFLX RDDT |
| b3_etf | GLD SLV USO BTC-USD |
| sectors | XLK XLF XLE XLV XLY XLP XLU XLI |
| b5 | HIMS EOSE RKLB SOFI HOOD |

**H2 ve H2 + X2** (getiri, parantezde en büyük düşüş)

| Sepet | Son yıl H2 | Son yıl X2 | Al-tut | 2020–25 H2 | 2020–25 X2 | Al-tut |
|---|---|---|---|---|---|---|
| semi_noMU | +14,0 (−34,5) | +8,5 (−30,4) | +7,0 | +527 (−32,6) | +435 (−33,2) | +786 |
| first_noAPP | −28,2 (−28,3) | −26,0 (−26,1) | −43,5 | +143 (−26,1) | +135 (−27,3) | +220 |
| b3_etf | +28,6 (−24,5) | +33,1 (−21,7) | +21,5 | +93 (−34,4) | +106 (−31,6) | +272 |
| sectors | +9,5 (−4,5) | +8,7 (−4,4) | +17,8 | +57 (−13,6) | +57 (−14,5) | +116 |
| b5 | −46,9 (−46,5) | −39,7 (−43,9) | −46,6 | +250 (−55,1) | +299 (−47,2) | +274 |

Sektör ETF'leri 2007–2012: H2 +28,3 % (düşüş −19,6 %), al-tut +23,5 % (−50,8 %), SPY +13,6 % (−55,2 %).

**Karakter:** çöküşlerde ve uzun düşüşlerde güçlü koruma; güçlü boğa piyasalarında al-tutun gerisinde; çok oynak spekülatif hisselerde (b5, son yıl) koruma yok — stoplar %20–30 uzakta kalıyor ve SMA200 üstünde tekrar tekrar yeniden alım yapılıyor.

## 6. Denenip reddedilenler (kısa)
- **Alım:** sistem swing skoru (≥60), direnç kırılımı, Kalman trendi, SMA50 geri alımı, Fibonacci filtresi, EMA50/200, E1 (SMA200 üstünde EMA8>EMA21 şartı — düşüşü azaltıyor, güçlü trendde getiriyi kesiyor).
- **Zamanlama:** günlük karar, günlük alım + haftalık stop, haftada 1 alım herhangi bir gün, 2 gün teyit, pazartesi teyidi, cuma kapanışında alım (döneme bağlı), her durumda 2 cuma teyidi (H1/F2). H2 = "SMA200 üstü 1 cuma + pazartesi açılış, altı 2 cuma" en dengelisi.
- **Stop:** 2–6×ATR, destek/Donchian, Fibonacci, EMA21 kademesi, ATR tamponu (1×ATR 2020–25'te iyi, son yıl kötü), min(%5, 2×ATR) (=%5), günlük stop güncelleme, %15 risk sınırı.
- **Çıkış:** X1, oynaklığa göre (ATR>%4) çıkış, haftalık X2 (etkisiz).
- **Pozisyon büyüklüğü:** risk bazlı (riski ve getiriyi birlikte düşürüyor).
- **Yeniden alım yasağı (RB50):** oynak/düşen sepetlerde iyi, güçlü trendde (MU) çok kötü; fiyat bazlı trend kalitesi sınıflandırması (Q1/Q2) bunu güvenilir şekilde ayıramadı. Temel veriye dayalı sınıflandırma (kârlılık, sulandırma) denenmedi.

## 7. Stop geri alma (RC) — test edildi 2026-11-09, REDDEDİLDİ
Kurallar önceden sabitlendi: yalnızca stop çıkışından sonra; stop gününden sonra seviyenin üstünde 3 ardışık kapanış (altında kapanış sayacı sıfırlar); 3. günden *sonraki* ilk cuma kapanışı seviyenin üstündeyse o kapanışta al, değilse kural iptal; ilk stop seviye × 0,95, sonra normal H2 (SMA200 altında EMA21 kademesi); H2 normal alımı kuralı sıfırlar.

| Sepet | Son yıl H2 → RC | 2020–25 H2 → RC |
|---|---|---|
| semi_noMU | +14,0 → +13,0 | +527 → +525 |
| first_noAPP | −28,2 → −29,1 | +143 → +127 (düşüş −26,1 → −29,1) |
| b3_etf | +28,6 → +25,4 | +93 → +108 (düşüş −34,4 → −37,5) |
| sectors | +9,5 → +8,5 | +57 → +59 |
| b5 | −46,9 → −41,4 | +250 → +231 |

Sektörler 2007–12: +28,3 → +28,8, düşüş −19,6 → −20,2. 10 karşılaştırmanın 6'sında getiri düşüyor, en büyük düşüş çoğunlukla artıyor; RECLAIM işlemlerinin isabeti düşük (son yıl 30 işlemde 6 kazanan). Sonuç: H2'ye eklenmesin.

## 7b. Stop tamponu ızgarası — test edildi 2026-11-09
Üç kademede 0,95 yerine (1 − tampon); ızgara %0/2/3/4/5/6/7/8/10/12/15. Önceden ilan edilen seçim ölçütü: 2020–25, 5 sepetin ortalama getiri/|düşüş|.

| Tampon | 2020–25 ort. getiri / düşüş / oran | Son yıl ort. getiri / düşüş / oran | Sektör 2007–12 getiri / düşüş |
|---|---|---|---|
| %0 | 185 / −28,4 / **7,76** | −5,1 / −25,4 / 0,21 | 11,3 / −16,2 (421 işlem) |
| %2 | 194 / −30,5 / 7,39 | −2,7 / −25,3 / 0,43 | 29,4 / −17,7 |
| %3 | 202 / −32,3 / 6,79 | −5,6 / −26,4 / 0,36 | 27,2 / −19,3 |
| %4 | 215 / −32,5 / 6,78 | −5,3 / −27,6 / 0,34 | 32,2 / −19,7 |
| **%5 (H2)** | 214 / −32,3 / 6,62 | −4,6 / −27,7 / 0,33 | 28,3 / −19,6 |
| %6 | 228 / −33,4 / 6,72 | −3,7 / −26,7 / 0,52 | 26,3 / −21,7 |
| %7 | 232 / −34,2 / 6,51 | −3,0 / −27,7 / 0,60 | 29,7 / −22,3 |
| %8 | 240 / −34,1 / 6,77 | −4,0 / −28,8 / 0,58 | 31,1 / −22,5 |
| %10 | 248 / −35,0 / 7,19 | −7,9 / −29,8 / 0,38 | 28,3 / −25,4 |
| %12 | 255 / −35,5 / 6,99 | −6,0 / −29,8 / 0,46 | 27,0 / −30,0 |
| %15 | 256 / −35,0 / 7,00 | −7,8 / −32,0 / 0,51 | 28,4 / −33,6 |

- İlan edilen ölçütün kazananı %0, ama dışarıda tutmuyor: son yıl en kötü oran, 2007–12'de getiri 28 → 11, işlem sayısı 4 kat. Kabul edilmedi.
- Net bir optimum yok; komşu tamponlar arasında sıçramalar büyük (gürültü). Genel eğilim: tampon büyüdükçe 2020–25 getirisi artıyor (çoğu b5'ten), düşüş derinleşiyor.
- Sepet bazında (getiri/|düşüş| en iyisi, 2020–25 / son yıl): semi_noMU %0 / %5; first_noAPP %10 / %0–2; b3_etf %4 / %7; sectors %2 / %15 (2007–12: %2); b5 %15 / hepsi ≈ −47…−55. Sepet başına iki dönem aynı tamponu seçmiyor; yalnızca sectors'ta %2 iki dönemde öne çıkıyor. b5'te tampon getiriyi değiştiriyor (+151 → +505) ama düşüş hep ≈ −55 (boşluklar ve çöküş belirliyor).
- Sonradan gözlem (seçim ölçütüyle bulunmadı): %2, üç dönemin üçünde de %5'ten daha iyi risk/getiri veriyor, ama işlem sayısı ~1,5 kat ve %3/%4 bu üstünlüğü göstermiyor. Değiştirmek için yeterli kanıt değil; yeni bir sepet/dönemde önceden ilan edilerek doğrulanmalı.

## 7c. Oynaklığa göre ölçeklenen tampon (`_Vnn`) — test edildi 2026-11-09
Kural (önceden sabitlendi): tampon = k × (son 252 günün ortalama ATR14/kapanış), %2–%12 arasına sınırlı, her cuma o günün değeriyle; veri yetersizse %5. Mod `H2_V10` = k 1,0. Izgara k = 1 / 1,5 / 2 / 2,5 / 3. Seçim ölçütü: 2020–25 ortalama getiri/|düşüş| + komşu k'larda çökmeme.

| Mod | 2020–25 ort. getiri / düşüş / oran | Son yıl ort. getiri / düşüş / oran | Sektör 2007–12 |
|---|---|---|---|
| H2 (sabit %5) | 214 / −32,3 / 6,62 | −4,6 / −27,7 / 0,33 | +28,3 / −19,6 |
| **V10** | 243 / −30,0 / **8,19** | −5,2 / −27,1 / 0,37 | **+32,2 / −17,9** |
| V15 | 242 / −30,9 / 7,76 | −6,5 / −28,9 / 0,39 | +35,8 / −19,2 |
| V20 | 256 / −32,8 / 7,76 | −7,0 / −29,0 / 0,50 | +32,6 / −20,4 |
| V25 | 280 / −34,2 / 7,83 | −7,1 / −29,8 / 0,42 | +26,6 / −21,3 |
| V30 | 265 / −34,0 / 7,53 | −6,4 / −30,1 / 0,48 | +29,9 / −20,4 |

- Kazanan V10; tüm k bölgesi (7,5–8,2) sabit %5'in (6,62) ve ızgaradaki her sabit tamponun üstünde → düz/sağlam bölge, tek nokta değil.
- V10 pratikte "tampon ≈ ATR%, taban %2": sektörler ve GLD %2'de, NVDA/MRVL ~%4, ORCL ~%2,4, b5 %6–11.
- 2020–25 sepet bazında V10 vs H2: semi 542/−27 vs 527/−33, sectors 57/−10,5 vs 57/−13,6, b5 379/−54 vs 250/−55, b3 123/−36 vs 93/−34; first_noAPP daha kötü (114/−23 vs 143/−26).
- Son yıl karışık: oran biraz iyi (0,37 vs 0,33) ama semi +3,8 vs +14,0, b5 −50 vs −47; first_noAPP (−22 vs −28) ve b3 (+33,5 vs +28,6) daha iyi.
- Durum: önceden ilan edilen testi geçen ilk tampon varyantı; doğrulama dönemlerinde risk/getiri eşit ya da daha iyi, son yıl getirisi biraz düşük. Göstergeye/motora henüz uygulanmadı.


**Sepet değişikliği (2026-11-09):** yarı iletken sepetinde NBIS yerine AMD konuldu (`semi_AMD`: NVDA AVGO MRVL AMD ORCL). Yukarıdaki `semi_noMU` rakamları NBIS'li eski sepete ait; o sepeti yeniden üretmek için `BASKETS`'te geri değiştirmek gerekir. Script'e 3. argüman olarak sepet filtresi eklendi (ör. `... sonuc.json H2,H2_V10 semi_AMD`).

| semi_AMD | Son yıl getiri / düşüş | 2020–25 getiri / düşüş |
|---|---|---|
| Al-tut | +21,6 / −32,3 | +671 / −51,6 |
| H2 (sabit %5) | +18,2 / −27,6 | +538 / −40,7 |
| V10 | +16,0 / −28,3 | +550 / −34,8 |
| V15 | +18,3 / −26,8 | +537 / −38,7 |
| V20 | +9,1 / −28,3 | +538 / −41,2 |
| V25 | +7,9 / −31,6 | +619 / −41,0 |
| V30 | +12,0 / −30,2 | +548 / −41,0 |

NBIS çıkınca V10'un son yıl farkı −10 puandan −2 puana iniyor (kalan fark AVGO −1.155 $ ve MRVL −2.030 $; AMD +279 $). AMD'nin tipik tamponu ~%4,3–4,7, yani %5'e yakın. 2020–25'te V10 düşüşü −40,7'den −34,8'e indiriyor.

## 7d. V10 doğrulama, 2015-01-02 – 2020-10-30 (hiç kullanılmamış dönem) — GEÇTİ
Önceden ilan edilen şart: ana 3 sepette (semi_AMD, b3_etf, sectors) ortalama getiri/|düşüş| V10 ≥ H2 ve V10 en az 2 sepette H2'den kötü değil. first_noAPP ikincil (yalnız NFLX + 2019'dan UBER), b5 verisi yok. Çalıştırma: `... val.json H2,H2_V10,... semi_AMD,b3_etf,sectors,first_noAPP val` (4. argüman = dönem listesi).

| Sepet | Al-tut | H2 getiri / düşüş / oran | V10 getiri / düşüş / oran |
|---|---|---|---|
| semi_AMD | +1145 / −45,6 | +765 / −32,3 / 23,7 | +791 / −31,4 / **25,2** |
| b3_etf | +1057 / −79,8 | +2102 / −67,4 / 31,2 | +2536 / −70,6 / **35,9** |
| sectors | +63 / −35,1 | +31 / −12,9 / 2,4 | +41 / −8,3 / **4,9** |
| first_noAPP (ikincil) | +282 / −35,8 | +107 / −24,0 / **4,5** | +68 / −24,4 / 2,8 |

- Ana 3 sepetin üçünde de V10 daha iyi; ortalama oran 22,0 / 19,1 → **şart sağlandı**. first_noAPP yine H2 lehine (2020–25'teki gibi).
- Sağlamlık zayıf: V15 bu dönemde semi (673/−31) ve b3'te (1444/−78) H2'nin gerisinde; V20–V30 sectors'ta iyi, b3'te kötü. Yani k=1 iyi, ama k bölgesi 2020–25'teki kadar düz değil → k=1 sabit kalsın, ayarlanmasın.
- b3_etf rakamları BTC'nin 2015–17 yükselişi ve 2018 çöküşüyle belirleniyor.
- sectors'ta V10 işlem sayısını ikiye katlıyor (192 / 95); %0,1 maliyet dahil.

## 8. Bilinen veri uyarıları
- TradingView'da CFD'ler (ör. IG "US Tech 100 Cash") ve sürekli vadeli seriler (CL1!, NQ1!) farklı günlük mumlar / kontrat değişim boşlukları içerir; sinyaller endeks/ETF'ten farklı günlere kayar. Test ve takip için NDX/QQQ, ETF'ler ya da hissenin kendisi kullanılmalı.
- BTC hafta sonu da işlem görür; backtest stopu yalnızca hafta içi günlerde kontrol ediyor.
- Göstergedeki simüle pozisyon grafikte yüklü ilk bardan başlar; kullanıcının gerçek pozisyonu için tablodaki "Your stop" satırı kullanılmalı.

## 9. Giriş filtreleri (rule 1, pazartesi alımı) — test edildi 2026-11-09, HEPSİ REDDEDİLDİ
Taban: H2 + V10 (göstergedeki güncel kural). Filtre yalnızca pazartesi alımını engeller; önceki-cuma koşulu ve 2-cuma EMA alımı değişmez. Modlar: `_EA<k>` (cuma kapanışı > SMA200 × (1 + clip(k × 252g ATR%, %2–12))), `_EB` (SMA200 > 21 gün önceki SMA200), `_EC` (21/63/252 günlük getirilerden ≥2'si pozitif). Önceden ilan: seçim 2020–25 ort. getiri/|düşüş|; geçen aday son yıl, 2015–20 ve 2007–12'de de tabanın gerisinde kalmamalı. Önceden ilan edilen A k=1; k=0,5 ve 2 yalnız bilgi.

| Mod | 2020–25 oran | Son yıl ort. getiri / düşüş / oran | 2015–20 oran (semi, b3, sectors) | Sektör 2007–12 |
|---|---|---|---|---|
| Taban (V10) | **7,35** | −2,7 / −25,9 / 0,46 | **22,0** | +32,2 / −17,9 |
| A k=1 | 6,01 | −1,7 / −23,6 / 0,44 | 24,5 | +32,3 / −16,3 |
| A k=0,5 (bilgi) | 7,08 | −1,7 / −25,1 / 0,49 | 24,6 | +33,8 / −16,6 |
| A k=2 (bilgi) | 6,01 | −0,8 / −22,6 / 0,47 | 16,9 | +26,7 / −15,1 |
| B eğim | 5,85 | +0,7 / −22,9 / 0,50 | 21,3 | +29,3 / −17,6 |
| C momentum | 7,51 | −0,6 / −23,5 / 0,60 | 20,1 | +34,3 / −16,7 |

- A ve B seçim ölçütünde tabanın gerisinde → elendi.
- C ölçütü kıl payı geçti (7,51 / 7,35) ama kazanç tek sepetten (b3_etf +228 / +123); diğer 4 sepette 2020–25 daha kötü. 2015–20'de üç ana sepetin üçünde de geride (semi 689 / 791, sectors 34 / 41) → şart sağlanmadı, elendi.
- Ortak desen (literatürle uyumlu): filtreler işlem sayısını azaltıyor; dalgalı/düşen son yılda hepsi yardımcı (b5 −40…−44 / −50), güçlü trendlerde (2020–25) getiriyi kesiyor.
- Sonradan gözlem: A k=0,5 (çoğu hissede pratikte %2 bant) 2020–25'te tabana yakın, son yıl/2015–20/2007–12'de biraz daha iyi. Önceden seçilmediği için kural değil; yeni veride önceden ilan edilerek denenebilir.

## 10. Düşen trend kırılımı girişleri (SMA200 altı) — test edildi 2026-11-09
Önceden sabitlenen kurallar: tepe/dip = ±5 günün en yükseği/en düşüğü (5 gün sonra kesinleşir). Yalnız fiyat SMA200 altındayken, cuma kapanışında karar ve alım; ilk stop EMA21 × (1 − V10 tamponu), sonrası normal H2. Taze kırılım: bu cuma kapanışı seviyenin üstünde, önceki cuma değil.
- **D1 trend çizgisi:** son 1 yıldaki son iki tepe alçalıyorsa aralarından çizgi, bugüne uzatılır; kapanış çizginin üstünde.
- **D2 yapı değişimi:** son tepe öncekinden düşük, o tepeden sonraki dip öncekinden yüksek; kapanış son tepenin üstünde.
- `_D1`/`_D2` = EMA8/21 alımına ek; `_D1R`/`_D2R` = onun yerine. Taban H2 + V10; şart §9 ile aynı.

| Mod | 2020–25 oran | Son yıl oran | 2015–20 oran | Sektör 2007–12 | Sonuç |
|---|---|---|---|---|---|
| Taban | 7,35 | 0,46 | 22,0 | +32,2 / −17,9 | — |
| D1 ek | 7,41 | 0,56 | 30,2 | +15,2 / −24,3 | ❌ kriz |
| **D2 ek** | **7,92** | 0,46 | 22,8 | +32,7 / −18,1 | ✅ geçti (küçük etki) |
| D1 yerine | 8,00 | 0,56 | 29,4 | +14,3 / −23,3 | ❌ kriz |
| D2 yerine | 8,12 | 0,40 | 29,3 | +21,8 / −17,4 | ❌ son yıl, kriz |

- D1 (trend çizgisi) 2015–20'de çok iyi ama 2007–12'de getiriyi yarıya indirip düşüşü derinleştiriyor: ayı piyasası tepkilerinde sürekli sahte kırılım. Çok sık sinyal (tüm koşularda 373 işlem, %34 kazanan).
- D2 tüm şartları geçti ama etkisi küçük: tüm koşularda yalnız 51 D2 işlemi (%43 kazanan); 2020–25 kazancı çoğunlukla b5 (+433 / +379) ve sectors düşüşü (−8,9 / −10,5). Son yıl ve 2007–12'de tabanla neredeyse aynı. Göstergeye eklenmedi.

## 11. Swing modu — test edildi 2026-11-09, REDDEDİLDİ (göstergeye eklenmedi)
Ayrı script: `research/weekly_trend/swing_mode_backtest.py` (kurallar docstring'de; `python ... out.json [sepetler] [dönemler]`). Günlük karar, kapanışta alım, %0,1 maliyet, hisse başına hesap.
- **S1 RSI(2):** RSI(2) < 10 → kapanışta al; kapanış > SMA5 ya da 10 gün → kapanışta sat; stop giriş − 2×ATR.
- **S2 EMA21 dönüşü:** dip ≤ EMA21 < kapanış ve kapanış > önceki gün en yükseği → al; stop gün dibi − 0,5×ATR, hedef 2R, 10 gün.
- **F** = göstergenin alım durumu filtresi (kapanış > SMA200 ya da EMA dönüşü).
Önceden ilan: (1) filtre 2020–25 oranında filtresizi geçmeli ve 3 doğrulamanın ≥2'sinde geride kalmamalı; (2) göstergeye eklemek için filtreli sürüm 4 dönemde de kârlı ve 2020–25 oranı ≥ H2+V10 (7,35).

| Mod | 2020–25 ort. getiri / düşüş / oran | Son yıl | 2015–20 (semi, b3, sectors) | Sektör 2007–12 | Piyasada kalma |
|---|---|---|---|---|---|
| H2+V10 (haftalık) | +245 / −31,6 / 7,35 | −2,7 / −25,9 / 0,46 | +1123 / −36,8 / 22,0 | +32,2 / −17,9 | — |
| S1 | +52 / −19,1 / 2,44 | +0,8 / −12,6 / 1,40 | −7,7 / −23,6 / −0,09 | +9,7 / −17,6 | %14–26 |
| S1F | +25 / −11,4 / 2,33 | +2,2 / −7,9 / 0,94 | −1,4 / −14,9 / −0,09 | +9,1 / −3,8 | %2–13 |
| S2 | +12 / −27,9 / 0,47 | −1,9 / −13,3 / 0,41 | +25,5 / −12,7 / 1,76 | −16,4 / −30,6 | %15–26 |
| S2F | +18 / −17,9 / 0,92 | −3,1 / −11,6 / 0,13 | +9,8 / −11,1 / 0,77 | −12,4 / −21,4 | %9–23 |

- Şart 1: S1'de filtre seçimde kaybediyor (2,33 / 2,44); S2'de seçimde kazanıyor ama 3 doğrulamanın üçünde de geride. Şart 2: S1F 2015–20'de zararda, S2F son yıl zararda; ikisinin oranı da haftalık kuralın çok altında.
- Filtrenin etkisi: piyasada kalma süresini ve düşüşü yaklaşık yarıya indiriyor, getiriyi de aynı oranda; risk/getiri oranı değişmiyor. Tek istisna kriz: S1F 2007–12'de düşüşü −17,6'dan −3,8'e indiriyor.
- S1 (RSI2) yüksek kazanma oranı (%60–75) ve düşük düşüşle en makul swing kurulumu, ama 2015–20'de para kaybetti. S2 (EMA21 dönüşü) belirgin bir avantaj göstermedi (%40–49 kazanan, 2R hedefe rağmen).
