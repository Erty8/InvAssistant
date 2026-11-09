# Swing Planı: GOOG

**Tarih:** 30 Ekim 2026, ABD seansı açılmadan önce (son kapanış 29 Ekim) · **Vade:** 2–8 hafta
**Kaynak:** Motorun teknik katmanı (`sec_analyzer.technical`: göstergeler, swing skoru, aksiyon) ve 30 Ekim'deki `analyze GOOG` çıktısı. Plan kuralları Claude'a ait.

> **Önce okuyun:** Motorun geriye dönük testinde (126 S&P 500 hissesi, 2015–2026) bu teknik durumların 10–63 günlük getirileri tahmin etme gücü bulunmadı. Aşağıdakiler bir **risk yönetimi çerçevesi**dir: nerede yanıldığınızı önceden tanımlar. Kazanç vaat eden bir sinyal değildir. Kişisel yatırım tavsiyesi değildir.

## 1. Durum

28 Ekim akşamı açıklanan Q3 sonuçlarından sonra GOOG 29 Ekim'de **-%12 boşlukla** açıldı ($321.89 → $283.84). Gün **$282.32**'den kapandı. Hacim 92.6M oldu; 50 günlük ortalama 16.8M, yani yaklaşık 5.5 katı.

Düşüşün sebebi temel. Son 10-Q'da:
- Çeyreklik serbest nakit akışı **-$9.1B**'a döndü. Son 12 ayın capex'i $160B.
- Borç 9 ayda $49B'tan $127.5B'a çıktı.
- Hisse geri alımları durdu.

Faaliyet tarafı güçlü: gelir +%25, faaliyet karı +%30. Motorun temel kararı **EXPENSIVE**. Bu yüzden bu işlem bir **tepki yükselişi işlemi**dir, uzun vadeli bir tez değildir.

| Gösterge | Değer | Okuma |
|---|---|---|
| Motor aksiyonu | BUY: "Düşüş trendi, dönüş teyit ediliyor" | Dönüş kontrollerinin 5'te 2'si gerçekleşti (hacim, RSI uyumsuzluğu). Yine de giriş trende karşı |
| Swing kurulumu | Aşırı satım dönüşü, skor 30/100 (zayıf) | En zayıf bileşen göreli güç (SPY'a karşı) |
| Momentum | 27/100, aşağı, hızlanıyor | |
| RSI (14) | 22.1 | Aşırı satım. 30'un üzerine geri dönüş henüz yok |
| MACD | Histogram -4.75, kesişim yok | Kısa vadeli ivme negatif |
| ATR (14 gün) | $10.47 (~%3.7) | Boşluk günü (gerçek aralık $43.6) şişiriyor. Boşluk öncesi ~$7.9 (~%2.8) |
| 50 / 200 günlük ortalama | $336.88 / $338.68 (~+%19 / +%20) | 60 gün içinde ölüm kesişimi oluştu |
| 52 hafta | En yüksek $403.96 (18 Mayıs), fiyat aralığın %8'inde | |

**Kilit seviyeler**

| Seviye | Fiyat | Neden |
|---|---|---|
| Direnç | **$326.7–342.0** | SMA50/SMA200 kümesi ve %23.6 Fibonacci. Motorun dönüş tetikleyicisi $334.35 |
| Boşluk bölgesi | **$313.5–321.9** | Boşluk öncesi dip ve kapanış. Doldurulması ilk doğal hedef |
| Direnç | **$303.9** | %38.2 Fibonacci. Nisan 2026'daki $303.5 tepesi |
| Boşluk günü tepesi | **$286.43** | Kısa vadeli teyit seviyesi |
| **Destek bölgesi** | **$271.2–278.3** | Mart 2026 dibi (gün içi $271.20, kapanış $272.80), %50 Fibonacci ($272.98) ve boşluk günü dibi ($278.29) |
| Destek | **$242–254** | %61.8 Fibonacci ($242.06) ve Ekim 2025 dipleri ($244–254) |
| Ana destek | $197.5–207.5 | 6 kez test edildi, %78.6 Fibonacci |

## 2. Kurallar

1. **Boşluğun ilk gününde alma.** Büyük hacimli, temel kaynaklı bilanço boşlukları birkaç hafta aynı yönde sürebilir. Buna "bilanço sonrası sürüklenme" (post-earnings drift) denir. Önce fiyatın **en az 2–3 seans** destek bölgesinde taban yapması beklenir.
2. **Riski pozisyon büyüklüğüyle sınırla.**

   **Pozisyon payı = (portföyün % riski) ÷ (girişten stopa % mesafe)**

   Volatilite yüksek (20 günlük yıllık %48) ve temel arka plan olumsuz, bu yüzden işlem başına **%0.5 risk** öneriliyor. Pozisyon portföyün **%10'unu aşmamalı**. Örneğin %0.5 risk ve %4.3 stop mesafesiyle pozisyon payı %11.6 çıkar; tavan nedeniyle %10'a indirilir.
3. **Stop, emir verilmeden önce belirlenir** ve sonradan aşağı çekilmez. Tetikleyiciler **günlük kapanışla** değerlendirilir, gün içi dokunuşla değil.
4. **Kârı kademeli al.**
   - Birinci hedefte pozisyonun yarısı kapatılır ve stop başa başa çekilir.
   - Kalan yarı ikinci hedefe ya da iz süren stopa kadar taşınır.
5. **Zaman stopu.** Girişten 4 hafta sonra birinci hedefe ulaşılmadıysa işlem kapatılır.
6. **Bilanço kuralı.** Bir sonraki sonuç açıklaması **~30 Ocak 2027**'de. 8 haftalık pencereyle çakışmıyor, ama pozisyon açıklamaya taşınmaz.
7. **Temel tezle karıştırma.** Bu işlem kısa vadeli aşırı satıma dayanıyor. Motorun değerleme okuması (EXPENSIVE; baz DCF $65–122) bu işlemle değişmez.

## 3. Kurulumlar

### Kurulum A: Destek bölgesinde dönüş (ana kurulum)

- **Ön koşul:** Boşluktan sonra en az 2–3 seans geçmiş olması ve **$271.2'nin altında günlük kapanış olmaması**.
- **Tetikleyici:** Fiyatın $271.2–278.3 bölgesini test etmesi ve aşağıdakilerden en az birinin gerçekleşmesi:
  - RSI'ın 30'un üzerine geri dönmesi. Bu, motorun dönüş kontrollerinden biri.
  - Önceki günün tepesinin üzerinde, günlük aralığın üst üçte birinde bir kapanış.
  - MACD'nin yukarı kesişimi.
- **Giriş bölgesi:** **$274–280.** Referans giriş $278.
- **Stop:** **$266.** Mart dibinin ~0.65 ATR altı. $278 girişte ~-%4.3.
- **Hedef 1:** **$303.9** (%38.2 Fibonacci), ~+%9.3. Burada yarısı kapatılır, stop başa başa çekilir.
- **Hedef 2:** **$313.5–321.9** (boşluk doldurma), orta nokta ~+%14.
- **Risk/ödül:** H1'e ~2.2, H2'ye ~3.3.
- **Geçersizlik:** Günlük kapanışın $271.2'nin altında olması. Bu durumda Kurulum C'ye geçilir.

### Kurulum B: Boşluk günü tepesinin kırılması (teyitli, daha düşük R/R)

- **Tetikleyici:** Günlük kapanışın **$286.43'ün üzerinde** olması. Fiyat destek bölgesine inmeden toparlanırsa kullanılır.
- **Giriş:** $286.5–289. Referans $288.
- **Stop:** **$270.5.** Destek bölgesinin altı, ~-%6.1.
- **Hedef 1 / 2:** $303.9 (+%5.5) / ~$317.7 (+%10.3).
- **Risk/ödül:** H1'e ~0.9, H2'ye ~1.7. Zayıf bir kurulum; Kural 2'ye göre pozisyon daha küçük olur.

Motorun bugünkü fiyattan hesapladığı ham kurulum şöyle: giriş $282.32, stop $261.38 (-%7.4, 2 ATR), hedef $334.35 (+%18.4), R/R 2.48. Ancak hedef SMA kümesinde, 8 haftalık bir swing için uzak. Giriş de teyit gelmeden yapılmış oluyor.

### Kurulum C: Destek kırılırsa daha derin destekte dönüş

- **Senaryo:** Günlük kapanış $271.2'nin altında olursa bir sonraki destek **$242–254**.
- **Tetikleyici:** Bu bölgede Kurulum A'dakine benzer dönüş işaretleri ya da yüksek hacimli bir teslimiyet barı.
- **Giriş:** ~$245–250. Referans $248.
- **Stop:** **$234**, %61.8 Fibonacci'nin ~1 ATR altı, ~-%5.6.
- **Hedef 1 / 2:** **$271.2** (kırılan destek artık direnç, +%9.4) / **$286.4** (+%15.5).
- **Risk/ödül:** ~1.7 / ~2.7. Trende karşı işlem; küçük pozisyon.

### Kurulum D: Trend onarımı (boşluk doldurma)

- **Tetikleyici:** Günlük kapanışın **$303.9'un üzerinde** olması.
- **Giriş:** $304–306. Referans $305.
- **Stop:** **$293**, ~-%3.9.
- **Hedef 1 / 2:** **$321.9** (boşluk tepesi, +%5.5) / **$334.4** (SMA50/200 kümesi, motorun dönüş tetikleyicisi, +%9.6).
- **Risk/ödül:** ~1.4 / ~2.5.

## 4. Özet tablo

| | A: Destekte dönüş | B: Boşluk tepesi kırılımı | C: Derin destek | D: Trend onarımı |
|---|---|---|---|---|
| Tetikleyici | $271–278'de dönüş teyidi; kapanış ≥ $271.2 | Kapanış > $286.43 | $242–254'te dönüş teyidi | Kapanış > $303.9 |
| Giriş (ref.) | $278 | $288 | $248 | $305 |
| Stop | $266 | $270.5 | $234 | $293 |
| Hedef 1 / 2 | $303.9 / $313.5–321.9 | $303.9 / ~$317.7 | $271.2 / $286.4 | $321.9 / $334.4 |
| Yaklaşık R/R | 2.2 / 3.3 | 0.9 / 1.7 | 1.7 / 2.7 | 1.4 / 2.5 |
| Trende göre | Karşı | Karşı | Karşı | Kısmen birlikte |

## 5. Takvim ve olay riskleri

- **3 Kasım:** ABD ara seçimleri. Piyasa genelinde volatilite beklenebilir.
- **Kasım sonu (~):** NVDA sonuçları. Yapay zekâ capex'ine bakışı etkileyebilir; GOOG'un satış sebebiyle doğrudan ilgili.
- **8–9 Aralık (~):** FOMC toplantısı. 10 yıllık faiz %4.98, 10 yıllık aralığının tepesinde.
- **~30 Ocak 2027:** GOOG Q4 sonuçları. Pencerenin dışında.

## 6. Uyarılar

- **Plan seans öncesi verilerle hazırlandı.** 30 Ekim seansı, boşluğun ikinci günü. İlk 2–3 seansın kapanışları Kurulum A'nın ön koşulunu belirler.
- **ATR şişkin.** 14 günlük ATR, boşluk gününün $43.6'lık gerçek aralığını içeriyor. Stoplar boşluk öncesi ATR'ye (~$7.9) göre ayarlandı. Günlük oynaklık birkaç hafta yüksek kalabilir; stop mesafeleri buna göre sıkı sayılır.
- **Düşüş temel kaynaklı.** Negatif FCF, artan borç ve durdurulan geri alımlar teknik bir kaza değil. Aşağı yönlü sürüklenme riski, aşırı satım okumasından daha ağır basabilir.
- **Motorun aksiyon etiketleri tahmin değildir.** Geriye dönük testte öngörü gücü bulunmadı.
- **GOOG ve GOOGL.** Seviyeler GOOG (C sınıfı) içindir. GOOGL ile fark küçüktür ama birebir aynı değildir.

*Bu belge eğitim amaçlı bir çerçevedir, kişisel yatırım tavsiyesi değildir; lisanslı bir danışman değilim. Swing işlemleri yüksek risk içerir ve sermaye kaybına yol açabilir.*
