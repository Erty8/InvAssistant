# Swing Planı: NBIS

**Tarih:** 4 Kasım 2026, seans içi (son kapanış 3 Kasım $151.64; anlık ~$143.41) · **Vade:** 2–8 hafta
**Kaynak:** Motorun teknik katmanı (`sec_analyzer.technical`: göstergeler, swing skoru, aksiyon), 4 Kasım'daki `analyze NBIS --no-ai` çıktısı, yfinance takvimi. Plan kuralları Claude'a ait.

> **Önce okuyun:** Motorun geriye dönük testinde (126 S&P 500 hissesi, 2015–2026) bu teknik durumların 10–63 günlük getirileri tahmin etme gücü bulunmadı. Aşağıdakiler bir **risk yönetimi çerçevesi**dir. Kişisel yatırım tavsiyesi değildir.

## 1. Durum

- Hisse 18 Haziran'daki $286.69 kapanış zirvesinden **-%50** aşağıda. Son 1 ayda -%35 (3 Kasım kapanışına göre). 52 haftalık dip $73.87 (5 Şubat); yıllık getiri hâlâ +%16.
- Ekim'deki düşüş IREN ile aynı günlerde hızlandı (14–15 Ekim, hacim 2–3 kat). AI altyapı / neocloud grubunun ortak satışı gibi görünüyor.
- 28 Ekim'de $182'ye tepki geldi, sonra iki günde $141'a geri düşüldü. 3 Kasım'da +%7 ($151.64); **bugün yine $143 civarı**, gün içi dip $140.66, 2 Kasım dibine ($140.60) çok yakın.
- **Q3 bilançosu: 12 Kasım (yfinance; şirketten teyit edilmedi).** Beklenti: gelir ~$906M (aralık $877–943M), EPS -$0.61.

## 2. Teknik görünüm

Göstergeler 3 Kasım kapanışına ($151.64) göre.

| Gösterge | Değer | Okuma |
|---|---|---|
| Motor aksiyonu | "Düşüş trendi, dönüş teyit ediliyor" (amber) | Tek dayanak RSI'ın 30'u geri alması (1/4). Bugünkü düşüşle zayıfladı |
| Swing skoru | **36/100, ZAYIF** · kurulum: aşırı satım tepkisi | En zayıf bileşen kurulum kalitesi; risk/getiri 0.47 (kötü) |
| RSI (14) | 37.3 (3 Kasım) → ~33 (anlık) | |
| MACD | -17.3 / sinyal -15.5, histogram negatif | Yükseliş kesişimi yok |
| OBV | Aşağı | Yükseliş/düşüş hacim oranı 0.79; göreli hacim 0.7× |
| ATR (14 gün) | **$15.78 (~%10–11)** | 20 günlük oynaklık yıllıklandırılmış ~%100 |
| SMA50 / SMA200 | $206.96 / $177.89 | Fiyat ikisinin de altında (-%27 / -%15). SMA50 düşüyor (-%5.8), SMA200 hâlâ yükseliyor (+%5.6). **Ölüm kesişimi yaklaşıyor** (fark ~$29) |
| 12-1 ay momentum | +%77.8 | Uzun vadeli trend IREN'e göre çok daha sağlam |

**Kilit seviyeler**

| Seviye | Fiyat | Neden |
|---|---|---|
| Direnç | **$200–204** | SMA50, Eylül direnci |
| Direnç | **$178–182** | SMA200, 28 Ekim tepki tepesi |
| Direnç / dönüş tetiği | **$164–169** (motor tetiği $166.51) | Temmuz direnci, 16–21 Ekim konsolidasyonu. Bunun üzerinde günlük kapanış dönüşü tamamlar |
| %50 Fibonacci | $158.6 | ($17 → $300 yükselişi) |
| Kırılan / test edilen destek | **$140.6–146.2** | Temmuz dibi, 2 Kasım dibi. Bugün test ediliyor |
| Destek | **$125.3** | %61.8 Fibonacci; motor stop'u $120.08 bunun altında |
| Destek | $73.5–75.6 | 52 haftalık dip bölgesi (5 dokunuş) |

## 3. Değerleme notu

Motorun değerleme çıktısı **kullanılamaz**: EPV $3.16–3.90 çıkıyor ve SEC serisi eski Yandex N.V. yıllarıyla karışık (FY2021–2023 rakamları Yandex dönemine ait). Gösterilebilecek olanlar (FY2025):
- Gelir $530M (FY2024: $92M), faaliyet zararı -$612M; net kâr $82.5M (operasyon dışı kalemlerden).
- Capex **$4.07B**; uzun vadeli borç $4.10B; nakit $3.68B.
- Altman Z 3.28 (güvenli), Merton temerrüt olasılığı %3.5. IREN'den (%21.6) belirgin şekilde daha sağlam bilanço.
- İçeriden satış: 180 günde 110 satış / 11 alış, net -$199M.

## 4. Plan

**Ana kural:** Trend günlük ölçekte aşağı ve fiyat $140–146 desteğini test ediyor. Bilançoya 6 işlem günü var. Mevcut durumda risk/getiri kötü (giriş $151.6 → stop $120 = -%21, hedef $166.5 = +%10).

**Kurulum A — Destek tutunursa, bilanço öncesi küçük pozisyon (agresif)**
- Koşul: $140.6 üzerinde kapanışlar ve **günün üst yarısında, ≥1.3× hacimli bir yükseliş günü**.
- Giriş: ~$145–150. Stop: $134 (2 Kasım dibi - ~0.4×ATR). Risk ~%8–10.
- Hedef: $158.6, sonra $166.5. **Bilanço öncesi en az yarısını kapat**, ya da tamamını kapat.

**Kurulum B — Bilanço sonrası dönüş (tercih edilen)**
- Koşul: bilanço sonrası **$166.5 üzerinde günlük kapanış** (motorun dönüş tetiği) ve hacim ≥1.3×.
- Giriş: $167–172. Stop: bilanço günü dibinin altı veya 1.5×ATR (~$24).
- Hedef 1: $178–182 (SMA200). Hedef 2: $200–204 (SMA50).

**Kurulum C — Destek kırılırsa**
- $140 altında kapanış → kurulum yok. Bir sonraki referans $125 (Fib %61.8). Orada hacimli bir dönüş mumu Kurulum A'yı yeni seviyelerle tekrar açar.

**Mevcut pozisyon sahipleri için**
- Motor stop'u $120.08 (-%16 anlık fiyattan); daha sıkı bir alternatif $140 altı kapanış.
- Dönüş teyidi: $166.5 üzerinde günlük kapanış.

## 5. IREN ile karşılaştırma

Aynı sektör satışı, ama NBIS daha sağlam: SMA200 hâlâ yükseliyor, 12 aylık momentum pozitif, bilanço riski daha düşük, swing skoru 36 vs 22. NBIS'te bilançoya bir haftalık süre var; IREN'de bilanço yarın.
