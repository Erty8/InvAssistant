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

## 8. Yasak davranışlar

- Sahte hassasiyet: "$103.47" gibi tek sayı verme; her zaman bant + varsayım.
- Multiples'ı DCF'e "yakınsın" diye seçmek (yöntem bağımsızlığını bozar).
- Duyarlılık tablosu geniş dağılıyorsa dar bant sunmak.
- Veri eksikse (ADR, kısa geçmiş) tahmin uydurmak — eksikliği raporla, güveni düşür.
- Kullanıcının pozisyonu olan hissede iyimserliğe kaymak: PROFIL/POZISYONLAR bağlamı
  yorumu kişiselleştirir, MATEMATİĞİ değiştirmez.
