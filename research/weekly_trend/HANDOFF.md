# Haftalık Trend Kuralı (H2) — devir notu

Son güncelleme: 2026-11-09. Bu not, "haftalık SMA200 + EMA8/21" stratejisinin araştırmasını başka bir oturumda sürdürmek için hazırlandı. Hiçbir şey commit edilmedi.

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
| `tradingview/weekly_ma_strategy.pine` | **Güncel gösterge.** Pine v6, İngilizce, "Weekly Trend Rule (SMA200 + EMA8/21)". H2 + X2 ayarı (kapalı), kompakt/tam tablo, "Your stop" satırı (kullanıcının gerçek pozisyonu için aday stop), alarmlar. Claude tarafından TradingView'da derlenmedi. |
| `tradingview/daily_ma_strategy.pine` | Eski, H2 öncesi deneme sürümü. Kullanılmıyor. |
| `research/weekly_trend/weekly_trend_backtest.py` | **Tek parça backtest script'i** (aşağıda). |
| `sec_analyzer/technical/weekly_strategy.py` + `WEEKLY_STRATEGY_SPEC.md` + `tests/test_weekly_strategy.py` | Başka bir oturumda (fork) eklendi: motor H2'yi hesaplayıp CLI kartında ve HTML raporda gösteriyor. Orada X2 varsayılan **açık** — göstergeyle tutarlı olması için kapatılması konuşuldu, yapılmadı. Commit edilmedi. |
| Hafıza: `manual-weekly-ma-strategy.md` | Tüm test geçmişinin kısa kaydı. |

## 4. Backtest script'i
```bash
PYTHONPATH=. PYTHONIOENCODING=utf-8 /c/Users/ertug/miniconda3/python.exe research/weekly_trend/weekly_trend_backtest.py sonuc.json
```
- Fiyatlar `sec_analyzer.fetch.prices.get_price_history` (yfinance, temettü düzeltmeli, önbellekli).
- Dönemler (`PERIODS`): `test` = 2025-11-03 … 2026-11-03, `train` = 2020-11-02 … 2025-10-31, `crisis` = 2007-01-03 … 2012-12-31 (son döngü `for per in ("test", "train")` — crisis için ekle).
- Sepetler (`BASKETS`) ve kurallar (`for name, MODE in (...)` döngüsü) dosyanın sonunda düzenlenir.
- Simülasyon: hisse başına ayrı hesap (sermaye / hisse sayısı), alım payın tamamıyla, %0,1 işlem maliyeti, açık pozisyonlar dönem sonu kapanışından değerlenir. Karşılaştırma: aynı paylarla "al ve tut".
- `ROW ... ret / dd / n / wins | hold ... | chk ...` satırları: getiri %, en büyük düşüş %, işlem sayısı, kazanan, al-tut getirisi/düşüşü, kâr/zarar mutabakatı.
- Mod adlarıyla açılan özellikler (MODE metninde geçerse): `_X2` (X2 çıkışı), `_E1` (SMA200 üstü alımda EMA8>EMA21 şartı), `RB50` / `RBEMA` (satış sonrası yeniden alım yasağı), `_Q1` / `_Q2` (yasağı yalnızca zayıf trendlere uygula), `RCAP` (%15 risk sınırı), `RSIZE` (risk bazlı büyüklük). Hepsi `H2` ile başlamalı.
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

## 7. Sıradaki test (kurallar sabitlendi, henüz çalıştırılmadı)
**"Stop geri alma" ile yeniden alım** (kullanıcının fikri, "başarısız kırılım"):
- Yalnızca **stopla** kapanan pozisyonlardan sonra devreye girer; referans = çalışan stop fiyatı.
- Stoptan sonra fiyat bu seviyenin üstünde **3 ardışık gün** kapanmalı.
- Bundan sonraki ilk **cuma kapanışı** da seviyenin üstündeyse → **o cuma kapanışında al**.
- İlk stop: geri alınan seviyenin %5 altı; sonrası normal H2 (SMA200 altında EMA21 × 0,95, üstünde kademeli), yalnızca yukarı.
- H2'nin normal alımı tetiklenirse ya da geri alma oluşmazsa kural sıfırlanır.
- Kapsam: yukarıdaki beş sepet, son yıl ve 2020–2025; H2 ile karşılaştır.
- Beklenti: asıl etkisi stoptan sonra fiyatın SMA200 altına indiği durumlarda (H2 o zaman SMA200'ü ya da 2 cuma EMA teyidini bekliyor). V dönüşlerinde yardımcı olabilir, düşüş trendlerinde sahte sinyal riski var.
- Uygulama ipucu: script'teki `simulate()` içinde stop çıkışında seviyeyi ve tarihi sakla; günlük döngüde 3 ardışık kapanışı say; cuma kapanışı alım bloğu (`CLOSE_ENTRY` bloğu) içinde yeni giriş türü ekle (kind adı örn. `"RECLAIM"`, iz sürme `trail_kind`/`stop_level` ile EMA kuralı gibi).

## 8. Bilinen veri uyarıları
- TradingView'da CFD'ler (ör. IG "US Tech 100 Cash") ve sürekli vadeli seriler (CL1!, NQ1!) farklı günlük mumlar / kontrat değişim boşlukları içerir; sinyaller endeks/ETF'ten farklı günlere kayar. Test ve takip için NDX/QQQ, ETF'ler ya da hissenin kendisi kullanılmalı.
- BTC hafta sonu da işlem görür; backtest stopu yalnızca hafta içi günlerde kontrol ediyor.
- Göstergedeki simüle pozisyon grafikte yüklü ilk bardan başlar; kullanıcının gerçek pozisyonu için tablodaki "Your stop" satırı kullanılmalı.
