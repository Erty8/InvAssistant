# Swing Planı: IREN

**Tarih:** 4 Kasım 2026, seans içi (son kapanış 3 Kasım $24.46; anlık ~$21.54) · **Vade:** 2–8 hafta
**Kaynak:** Motorun teknik katmanı (`sec_analyzer.technical`: göstergeler, swing skoru, aksiyon), 4 Kasım'daki `analyze IREN --no-ai` çıktısı. Plan kuralları Claude'a ait.

> **Önce okuyun:** Motorun geriye dönük testinde (126 S&P 500 hissesi, 2015–2026) bu teknik durumların 10–63 günlük getirileri tahmin etme gücü bulunmadı. Aşağıdakiler bir **risk yönetimi çerçevesi**dir. Kişisel yatırım tavsiyesi değildir.

## 1. Durum

- Hisse 5 Kasım 2025'teki $76.41 zirvesinden **-%72** aşağıda. Son 1 ayda -%40 (3 Kasım kapanışına göre), Ekim ortasında $41'dan $27'a sert bir kırılım yaşandı (14–15 Ekim, hacim 2–3 kat).
- 3 Kasım'da +%11 tepki yükselişi geldi ($22.02 → $24.46). **Bugün bu yükselişin tamamı geri verildi:** anlık $21.54, gün içi dip $21.50. Bu, 2 Kasım'daki 52 haftalık dibin ($22.02) ve motorun destek bölgesinin ($21.54–21.86) **altı**.
- **Q1 (FY2027) bilançosu ~5 Kasım'da**, yani yarın. Son dört çeyrekte EPS sürprizleri: +%44, +%38, +%88, -%94. Bilanço günü hareketi ATR'nin birkaç katı olabilir.

## 2. Teknik görünüm

Göstergeler 3 Kasım kapanışına ($24.46) göre; bugünkü düşüş bazı sinyalleri geçersiz kılıyor (aşağıda belirtildi).

| Gösterge | Değer | Okuma |
|---|---|---|
| Motor aksiyonu | "Düşüş trendi, dönüş teyit ediliyor" (amber) | Tek dayanak RSI'ın 30'u geri alması. **Bugün $21.5'a inişle bu sinyal büyük ölçüde boşa çıktı** |
| Swing skoru | **22/100, AVOID** · kurulum: aşırı satım tepkisi | En zayıf bileşen trend (-1.0), en güçlü tetikleyici |
| Momentum | 14/100, güçlü aşağı, hızlanıyor | |
| RSI (14) | 35.2 (3 Kasım) → ~31 (anlık) | Tekrar 30'a dayanıyor |
| MACD | -4.02 / sinyal -3.91, histogram negatif | Yükseliş kesişimi yok |
| OBV | Aşağı | Para çıkışı sürüyor; yükseliş/düşüş hacim oranı 0.75 |
| ATR (14 gün) | **$2.64 (~%11–12)** | Çok yüksek oynaklık; 20 günlük oynaklık yıllıklandırılmış ~%104 |
| SMA50 / SMA200 | $37.09 / $44.04 | Fiyat SMA50'nin -%34, SMA200'ün -%45 altında. SMA50 eğimi -%10.6, SMA200 -%3.4: iki ortalama da aşağı |
| 52 hafta | Zirve $76.41 (5 Kas 2025), dip ~$21.50 (bugün) | Aralığın en altında |

**Kilit seviyeler**

| Seviye | Fiyat | Neden |
|---|---|---|
| Direnç | **$36.5–37.4** | SMA50, 3 kez test edilmiş |
| Direnç | **$32.5** | Uzun vadeli %61.8 Fibonacci ($5.12 → $76.87) |
| Direnç / dönüş tetiği | **$28.9** | 28 Ekim tepkisinin tepesi ($28.84–28.88), Temmuz direnci. Motorun dönüş tetiği: bunun üzerinde günlük kapanış |
| Kırılan destek | **$21.5–22.0** | 52 haftalık dip, 2 Kasım dibi. Bugün test ediliyor |
| Destek | **$20.5** | %78.6 Fibonacci |
| Destek | **$13.6–13.7** | Bir sonraki belirgin fiyat kümelenmesi (2 dokunuş) |

## 3. Değerleme notu

Motorun değerleme çıktısı bu hisse için **kullanılamaz**: IREN "Financials" sektörüne sınıflandırıldı (P/B×ROE yolu), oysa iş modeli AI bulut/veri merkezi + bitcoin madenciliği. Adil değer hesaplanamadı. Temel tablo yine de bilgi veriyor (FY2026, Haziran sonu):
- Gelir $707M (+%41), net zarar -$703M, faaliyet zararı -$1.05B.
- Capex $3.0B; uzun vadeli borç $0.96B → **$7.42B**; hisse sayısı 223M → 316M (+%42 seyrelme).
- Faaliyet nakit akışı $2.1B (büyük ölçüde müşteri ön ödemeleri olmalı; doğrulanmadı), nakit $5.9B.
- Merton temerrüt olasılığı %21.6 (yüksek). Analist ortalama hedefi $75 (18 analist, aralık $37–139); fiyatla arasındaki uçurum, piyasanın bu hedeflere inanmadığını gösteriyor.

## 4. Plan

**Ana kural: bilanço öncesi yeni pozisyon açılmaz.** Trend her zaman diliminde aşağı, destek kırılıyor ve yarın ikili (binary) bir olay var. Bir stop'un ($21.48) bilanço boşluğunda işe yaramama ihtimali yüksek.

**Kurulum A — Bilanço sonrası dönüş (tercih edilen)**
- Koşul: bilanço sonrası **$24.5 üzerinde iki günlük kapanış** ve hacim ≥1.3× ortalama (bugünkü kırılımın geri alınması).
- Giriş: $24.5–25.5. Stop: bilanço günü dibinin altı veya en fazla 1.5×ATR (~$4) aşağısı.
- Hedef 1: $28.9 (dönüş tetiği). Hedef 2: $32.5 (Fib %61.8). $28.9 üzeri kapanışta stop başabaşa.

**Kurulum B — Kapitülasyon dibi (agresif, küçük boyut)**
- Koşul: bilanço sonrası $20.5 altına boşluk, ardından **aynı gün veya ertesi gün büyük hacimle (≥2×) kapanışın günün üst yarısında** olması.
- Giriş: teyit günü kapanışı. Stop: o günün dibinin altı. Hedef: $24.5, sonra $28.9.
- Boyut: normalin en fazla yarısı (ATR %11+).

**Kurulum C — Kaçınma**
- Bilanço sonrası $20.5 altında hacimsiz kapanışlar sürerse: kurulum yok. Bir sonraki referans $13.6–13.7.

**Mevcut pozisyon sahipleri için**
- Motorun stop'u ($21.48) bugün tetiklenmek üzere. Bilançoya pozisyonla girmek isteniyorsa, bu karar boşluk riskini (±%20–25 olası) göze almak demektir; boyut buna göre ayarlanmalı.
- Dönüşün teyidi: $28.9 üzerinde günlük kapanış. O olmadan trend aşağı.

## 5. Ne değiştirir?

- Bilançoda yeni bir büyük AI bulut sözleşmesi veya gelir rehberliği artışı + $24.5 geri alımı → Kurulum A aktif.
- Yeni hisse/konvertibl ihracı ya da capex finansman kaygısı → Kurulum C. (Ekim kırılımının sebebini doğrulamadım; 8-K'lar kontrol edilmeli.)
- Bitcoin fiyatındaki sert hareketler hâlâ hisseyi etkiliyor.
