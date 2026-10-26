# Swing Planı: NFLX ve UBER

**Tarih:** 26 Ekim 2026, gün içi (ABD seansının ilk saati) · **Vade:** 2–8 hafta
**Kaynak:** Motorun teknik katmanı (`sec_analyzer.technical`: göstergeler, swing skoru, aksiyon), canlı fiyatlarla. Plan kuralları Claude'a ait.

> **Önce okuyun:** Motorun geriye dönük testinde (126 S&P 500 hissesi, 2015–2026) bu teknik durumların 10–63 günlük getirileri tahmin etme gücü bulunmadı. Aşağıdakiler bir **risk yönetimi çerçevesi**dir: nerede yanıldığınızı önceden tanımlar. Kazanç vaat eden bir sinyal değildir. Kişisel yatırım tavsiyesi değildir.

## 1. Ortak kurallar

1. **Tahminle değil, teyitle gir.** İki hisse de düşüş trendinde: fiyat 50 ve 200 günlük ortalamaların altında. Trende karşı işlemde giriş, yalnızca tanımlı tetikleyici gerçekleştiğinde yapılır.
2. **Riski pozisyon büyüklüğüyle sınırla.** İşlem başına kabul edilen kayıp, portföyün sabit bir yüzdesiyle sınırlanır (yaygın aralık %0.5–1).

   **Pozisyon payı = (portföyün % riski) ÷ (girişten stopa % mesafe)**

   Örneğin %0.5 risk ve %6.3 stop mesafesiyle pozisyon, portföyün ~%8'i olur.
3. **Stop, emir verilmeden önce belirlenir.** Destek seviyesinin ~1–1.5 ATR altına konur ve sonradan aşağı çekilmez.
4. **Kârı kademeli al.**
   - Birinci hedefte pozisyonun yarısı kapatılır ve stop giriş fiyatına (başa baş) çekilir.
   - Kalan yarı ikinci hedefe ya da iz süren stopa kadar taşınır.
5. **Zaman stopu.** Girişten 4 hafta sonra birinci hedefe ulaşılmadıysa işlem kapatılır; sermaye ve dikkat başka yere aktarılır.
6. **Bilanço kuralı.** Swing pozisyonu şirket sonuç açıklamasına taşınmaz. Açıklama günü fiyat, stop seviyesinin ötesine boşluk bırakarak açılabilir.
7. **Uzun vadeli pozisyondan ayrı tut.** Portföyde zaten NFLX ve UBER var. Swing işlemi ayrı bir dilim olarak takip edilir; uzun vadeli tez ile kısa vadeli işlem karıştırılmaz.

## 2. NFLX

**Durum:** Fiyat $65.21. 52 haftanın en düşüğünde ($65.04) ve $64.10–65.08 desteğinde.

| Gösterge | Değer | Okuma |
|---|---|---|
| Motor aksiyonu | BUY: "Düşüş trendi, dönüş teyit ediliyor" | Dönüş için 6 kontrolün 2'si gerçekleşti |
| Swing kurulumu | Aşırı satım dönüşü, skor 33/100 (zayıf) | Risk/ödül güçlü, trend zayıf |
| Teslimiyet barı | 21 Ekim, normalin 2.5 katı hacimle düşüş | Satıcıların tükendiğine işaret olabilir |
| RSI | 35.8; yükseliş yönlü uyumsuzluk (2 Ekim) | Satış baskısı zayıflıyor |
| MACD | Aşağı kesişim | Kısa vadeli ivme hâlâ negatif |
| ATR (14 gün) | $2.04 (~%3.1) | Günlük tipik hareket |
| 50 / 200 günlük ortalama | $74.01 (+%13.5) / $82.15 (+%26) | |
| Dirençler | $80.62–82.21 (SMA200 + %23.6 Fibonacci), $88.66–91.33 | |

### Kurulum A: Aşırı satım dönüşü (trende karşı, yüksek risk/ödül)

- **Tetikleyici:** $64.10 desteği korunurken günlük kapanışın **$67.60'ın üzerinde** olması. Bu, 2 Ekim'deki uyumsuzluk tepesi. Ek teyit olarak MACD'nin yukarı kesişimi aranabilir.
- **Giriş bölgesi:** $67.60–68.50, tetikleyici kapanışından sonra.
- **Stop:** **$61.13** (motorun seviyesi; desteğin ~1.5 ATR altı). $68 girişte ~-%10.
- **Hedef 1:** **$74.0** (SMA50), girişe göre ~+%9. Burada yarısı kapatılır, stop başa başa çekilir.
- **Hedef 2:** **$80.6–81.4** (SMA200 ve %23.6 Fibonacci), ~+%19.
- **Risk/ödül:** H1'e ~0.9, H2'ye ~1.9. Tetikleyiciyi beklemek R/R'ı düşürür ama başarı olasılığını artırır.
- **Geçersizlik:** Günlük kapanışın $64.10'un altında olması kurulumu bozar; stop beklenmeden çıkılabilir.

Motorun bugünkü fiyattan hesapladığı ham kurulum daha iyi bir R/R veriyor: giriş $65.21, stop $61.13 (-%6.3), hedef $81.42 (+%24.9), R/R 3.97. Ama bu, henüz teyit gelmeden giriş anlamına geliyor.

### Kurulum B: Trend onarımı (trendle birlikte, daha düşük R/R)

- **Tetikleyici:** Günlük kapanışın **$74.01'in (SMA50) üzerinde** olması.
- **Giriş:** $74–75.
- **Stop:** ~$70.9 (SMA50'nin 1.5 ATR altı), ~-%4.5.
- **Hedef:** **$80.6–82.2** (SMA200), ~+%9–10. R/R ~2.

**Zamanlama:** NFLX'in bir sonraki sonuç açıklaması ~19 Ocak 2027. 8 haftalık swing penceresi buna çakışmıyor.

## 3. UBER

**Durum:** Fiyat $66.42. $65.41–67.24 desteğinde (5 kez test edildi), 52 haftanın en düşüğünün ($65.94) hemen üstünde.

| Gösterge | Değer | Okuma |
|---|---|---|
| Motor aksiyonu | AVOID: "Düşüş trendi, dönüş yok" | Dönüş kontrollerinin hiçbiri gerçekleşmedi (0/4) |
| Swing kurulumu | Kurulum yok, skor 29/100 | Bugünden R/R 0.82 |
| RSI / MACD | 37.8 / aşağı kesişim | Uyumsuzluk yok |
| ATR (14 gün) | $1.78 (~%2.7) | |
| Dirençler | **$69.16–69.51** (10 kez test edildi, %78.6 Fibonacci), $72.10 (SMA50), $73.73 (SMA200), $75.63 (%61.8 Fibonacci) | |
| Destekler | $65.41–67.24, ardından **$59.33–60.63** (-%9.7) | |

**Bugün geçerli bir swing kurulumu yok.** Ayrıca **Q3 sonuçları ~4 Kasım'da**, yani 9 gün sonra. Kural 6 gereği açıklamadan önce swing pozisyonu açılmaz. Delivery Hero teklif süreci de ayrı bir olay riski.

### Açıklama sonrası Kurulum A: Direnç kırılımı

- **Tetikleyici:** Sonuçlardan sonra günlük kapanışın **$69.51'in üzerinde** olması. Bu direnç 10 kez test edildi; kırılması anlamlı olur.
- **Giriş:** $69.5–70.5.
- **Stop:** ~$66.3 (destek merkezi), ~-%5.
- **Hedef 1:** $72.1–73.7 (SMA50 ve SMA200), ~+%4–6.
- **Hedef 2:** $75.6 (%61.8 Fibonacci), ~+%8.
- **Risk/ödül:** H1'e ~1, H2'ye ~1.6. Ortalamalar çok yakın olduğu için hareket alanı dar; bu zayıf bir kurulum.

### Açıklama sonrası Kurulum B: Daha derin destekte dönüş

- **Senaryo:** Fiyat $65.41'in altında kapanırsa, bir sonraki destek **$59.33–60.63**.
- **Tetikleyici:** Bu bölgede NFLX'tekine benzer işaretler: RSI'ın 30'un altından geri dönmesi, yükseliş yönlü uyumsuzluk veya yüksek hacimli bir teslimiyet barı.
- **Stop:** $57.5 civarı (~1 ATR altı).
- **Hedef:** $65.4–67.2 (eski destek, artık direnç), ~+%9–12.

## 4. Özet tablo

| | NFLX: A (dönüş) | NFLX: B (onarım) | UBER: A (kırılım) | UBER: B (derin destek) |
|---|---|---|---|---|
| Tetikleyici | Kapanış > $67.60, $64.10 korunurken | Kapanış > $74.01 | Sonuçlardan sonra kapanış > $69.51 | $59.3–60.6'da dönüş teyidi |
| Stop | $61.13 | ~$70.9 | ~$66.3 | ~$57.5 |
| Hedef 1 / 2 | $74.0 / $80.6–81.4 | $80.6–82.2 | $72.1–73.7 / $75.6 | $65.4–67.2 |
| Yaklaşık R/R | 0.9 / 1.9 | ~2 | ~1 / 1.6 | ~1.5–2 |
| Olay riski | Yok (sonuçlar Ocak'ta) | Yok | Sonuçlar 4 Kasım, Delivery Hero | Sonuçlar 4 Kasım |

## 5. Uyarılar

- **Canlı veriler gün içi.** Seansın ilk saatinde hacim göstergeleri (NFLX normalin 0.16 katı, UBER 0.12 katı) eksik günü yansıtır; yanıltıcıdır. Tetikleyiciler **günlük kapanışla** değerlendirilmelidir, gün içi dokunuşla değil.
- **Motorun aksiyon etiketleri tahmin değildir.** Geriye dönük testte öngörü gücü bulunmadı.
- **Trende karşı işlemler daha sık stop olur.** NFLX Kurulum A ve UBER Kurulum B trende karşı. Pozisyon büyüklüğü bu yüzden küçük tutulur.
- **Temel tezler ayrı.** NFLX tezi "nötr, hafif olumlu" (büyüme yavaşlıyor). UBER tezi "olumlu, satın alma riski". Swing planı bu tezleri değiştirmez.

*Bu belge eğitim amaçlı bir çerçevedir, kişisel yatırım tavsiyesi değildir; lisanslı bir danışman değilim. Swing işlemleri yüksek risk içerir ve sermaye kaybına yol açabilir.*
