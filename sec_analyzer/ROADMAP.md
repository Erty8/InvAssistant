# Roadmap

## Faz 2

- Watchlist + verdict değişim bildirimi
- 8-K takibi + AI özet
- Portfolio review (POZISYONLAR.md entegrasyonu)
- Peer comparison
- Risk faktörü diff'i (ardışık 10-K/10-Q Risk Factors karşılaştırması)
- Verdict backtest raporu (verdicts tablosu şimdiden bunun için kuruldu)
- [TAMAMLANDI] CapEx-yoğun hiper-grower'lar (ör. APLD gibi veri merkezi yatırımcıları) için
  revenue-first DCF'te büyüme CapEx'ini bakım CapEx'inden ayırma. Motor artık amortismanı
  (D&A) standart bakım-CapEx vekili olarak kullanıp başlangıç FCF marjını büyüme CapEx'i
  için rölöve ediyor (CapEx/gelir > %30 VE CapEx > D&A olduğunda); olgun hedef marj brüt-marj
  vekilinden ayrı türetildiği için steady-state'te bakım-CapEx yine fiyatlanıyor. Eski
  bastırma (suppression) guardrail'i, düzeltilmiş marjla bile baz senaryo ≤ $0 çıkarsa
  devreye giren bir backstop olarak kalıyor. Bkz. SPEC.md §3.6, VALUATION.md §4a.

## Değerleme motoru — katmanlı-ceza normalizasyonu (2026-07-16/17)

Sorun: SPEC.md/VALUATION.md'deki kurallar tek tek savunulabilir, ama üst üste
binince (her biri kendi "muhafazakâr" tavanını/kırpmasını eklediğinde)
sistematik bir %40-50 düşük-değerleme üretiyordu. Yeni `sec_analyzer/
calibrate.py` + `python -m sec_analyzer.cli calibrate` aracıyla (VALUATION.md
§9) ölçülebilir hale getirildi: bir ~28-hisselik sepette medyan makul-değer/
fiyat oranı, düzeltmeler öncesi bir hatalı ölçümde 0.39'a kadar düşük çıkıyordu
(kalibrasyon aracının kendisindeki bir hata yüzünden CAPM devre dışı kalmıştı);
düzeltme sonrası final medyan **0.925** (n=26, kova dağılımı 11/11/4). Detaylı
metodoloji ve ölçülen tam yörünge: VALUATION.md §9.

**Landed (WP1-7 + WP2b + LEVER 1/2/4, SPEC.md §3/§3.6/§8/§8b/§8d bölümlerinde
belgelenmiştir):**
- **WP1 — SBC-dilüsyon çift sayımı giderildi** (hiper-grower + orta-büyüme
  revenue-first yolları): SBC ihraçları hem marjda gider hem de
  `shares_yoy` üzerinden dilüsyon olarak iki kez cezalandırılıyordu; artık
  `engine._non_sbc_dilution` SBC'nin payını dilüsyondan çıkarıyor
  (`annual_dilution`/`sbc_dilution_excluded` yeni alanlar).
- **WP2 + LEVER 1 — terminal büyüme tek kural: `min(risk_free, %4)`**, hem
  hiper hem standart/olgun/orta-büyüme yollarında paylaşılıyor; kohort-özel
  sabit terminal büyüme (hiper için eskiden %2.5) kalktı. LEVER 1, Opus
  incelemesinin bulduğu bir hatayı düzeltti: paylaşım aslında SIC-eşleşmeyen
  isimler için çalışmıyordu (standart yol yalnızca CAPM'in kendi
  `risk_free`'ini okuyordu, CAPM yoksa terminal büyüme de düz %2.5'e
  düşüyordu — çifte ceza); artık GLOBAL `erp.csv` risk-free'sine düşüyor.
- **WP2b — ERP-spread float-sınır tutarlılık hatası düzeltildi**
  (`sanity._ERP_SPREAD_EPS`): `clamp_assumptions` ve `validate_assumptions`
  artık sınırda anlaşıyor; bu, ölçülen düşük-değerlemenin BASKIN nedeniydi
  (bkz. VALUATION.md §9).
- **WP3 — hiper-grower iskonto oranı fade eder**: her senaryo kendi
  14/12/10 kohort oranından, base senaryonun CAPM-farkındalı iskonto oranına
  (ERP-spread korumalı) steady-state yılına kadar lineer olarak iner;
  kümülatif iskonto faktörleri, uçtaki değer olgun oranda.
- **WP4 — marj tavanları (hiper %30, olgun %15, orta-büyüme %20) artık
  BAYRAK, sabit KIRPMA değil**: brüt-marj×0.5 (hiper/orta-büyüme) ve
  NOPAT×(1-%25)×%85 / min(nopat,hist) (olgun) mekanizmaları hedefi
  belirliyor; referans eşiği aşan bir sonuç not + `target_margin_flag`
  üretiyor, sabit tavana kırpılmıyor.
- **WP5 — büyüme sağlık-kontrolü tavanı %40→%60** (`sanity.
  _GROWTH_5Y_HARD_MAX`, hiper başlangıç tavanı, ters-DCF bracket); justified
  P/B `[0.5, 4.0]` kırpması → bayrak (`fair_pb` + `justified_pb_flag`).
- **LEVER 4** — standart iki-aşamalı DCF artık `growth_5y > %40` için
  `high_growth_flag` üretiyor (yalnızca rapor, hiper/orta-büyüme
  yollarındaki varış-noktası güvenlik ağına sahip değil — bkz. açık kalan
  riskler).
- **WP6 — bakım-CapEx tabanı sektörleştirildi**: `multiples.csv`'ye
  isteğe bağlı `capex_sales` kolonu eklendi; hiper-grower'ın bakım/büyüme
  CapEx ayrımı artık düz %5 yerine (varsa) sektörün Cap Ex/Sales oranını
  taban alıyor.
- **WP7 — "beats-floor" davranışı kapıdan rapora çevrildi**: döngüsel FCFE
  çapası ve olgun revenue-first DCF'in EPV tabanına karşı testi artık
  sessiz bir geçiş/kalış kararı değil, her iki sayıyı da (`growth_vs_floor`:
  `"adds"`/`"destroys"`) adlandıran bir rapor; bu davranış artık döngüsel
  yolda da olgun yoldakiyle simetrik.

**LEVER 2 (SIC eşleşme kapsamı genişletme):** KO'nun SIC açıklaması artık
Damodaran'ın "Beverage (Soft)" satırına eşleşiyor (`damodaran.py`'deki alias
tablosuna eklendi); XOM ("no annual data" — ayrı bir veri boşluğu) ve VZ
(negatif `fcf0` — ayrı) kapsam dışı kaldı.

**Bilinçli olarak dokunulmadı** (doğruluk gerekçesiyle, muhafazakârlık için
değil): `r ≤ g` iptali; `g/ROE` retention mantığı; ROE < r'de büyümenin değer
silmesi; `fcf0` tek-yıl-sıçrama koruması; EPV kazanç-kalitesi kapıları (OCF ≥
0.8×NI, CapEx ≥ 0.5×OCF); ERP-spread guard'ın kendisi (yalnızca sınır
tutarlılığı düzeltildi); iskonto tabanları (%7/%10).

**Açık kalan (kullanıcı onayı/daha fazla veri bekliyor):**
- **MU (Micron) döngü-tepe sorusu:** cyclical sürdürülebilir-büyüme FCFE
  çapası matematiksel olarak tutarlı (ROE %15.8 > özkaynak maliyeti %10.6 →
  büyüme değer ekliyor), ama motorun makul-değer/fiyat oranı MU'nun güncel
  fiyatına (~$876) göre çok düşük çıkıyor. Bu fiyatın bir bellek
  süper-döngüsünü mü fiyatladığı yoksa veri/model tarafında bir sorun mu
  olduğu netleşmedi — kullanıcı girdisi bekliyor.
- **XOM/VZ kalibrasyon-sepeti atlamaları:** XOM için yıllık veri eksikliği,
  VZ için negatif `fcf0` — SIC eşleşmesinden bağımsız, ayrı veri boşlukları.
- **KO (ve benzeri tek-seferlik-vergi isimleri) için normalize edilmiş
  `fcf0`:** KO'nun `fcf0`'ı bir seferlik bir IRS vergi depozitosu yüzünden
  baskılı, ama motorun mevcut tek-yıl-sıçrama koruması (monoton düşüş
  örüntüsünü "yapısal" sayan istisna) bunu YAKALAMIYOR. Vergi/tek-seferlik
  kalemlere duyarlı bir normalize `fcf0` (KO gibi isimler için) düşünülmeli.
- **LEVER 4'ün standart-DCF yüksek-büyüme bayrağı (`high_growth_flag`):**
  `script` sağlayıcısında `rule_based._default_growth_anchor`'ın kendi %25
  kırpması yüzünden etkisiz; bir LLM sağlayıcısının kırpmasız yüksek-büyüme
  bir varsayım kümesini standart DCF'e (hiper/orta-büyüme yerine)
  yönlendirmesi hâlâ güvenlik ağı olmayan bir gizli (latent) risktir.
- **`engine._hyper_scenario_meta`'nın görüntü metni:** hiper-grower
  `fair_value_range` notundaki "%2.5 terminale fade" ifadesi hâlâ SABİT bir
  metin — gerçek `terminal_growth_anchor` artık %2.5'ten farklı olabilir
  (WP2/LEVER 1 sonrası). Hesaplanan sayılar doğru, yalnızca bu görüntü
  metni güncel değil (bkz. SPEC.md §3'ün "Known display inconsistency"
  notu). `_mature_scenario_meta`/`_midgrowth_scenario_meta` zaten dinamik
  (`terminal_str`) — `_hyper_scenario_meta`'nın da aynı düzeltmeyi alması
  gerekiyor.

## Değerleme motoru — Damodaran/Nareit denetimi takibi (2026-07)

Denetimde tamamlananlar: FFO'yu Nareit tanımına yaklaştırma (gayrimenkul satış
kazançlarını çıkar + değer düşüklüklerini geri ekle), FFO/hisse'yi FFO mali yılının
hisse sayısıyla hesaplama, growth-aware justified P/B `(ROE−g)/(r−g)`, tek-yıl zarar
yanlış-sınıflandırma guard'ı, REIT sınıflandırmasını gayrimenkul-operatörü SIC
kodlarına genişletme, REIT kartlarında P/E-tabanlı PEG yerine P/FFO. Kalanlar:

- REIT için ters-DCF sinyali (ima edilen P/FFO): FFO-Gordon modelinin ters-DCF ayağı
  yok, bu yüzden REIT üçgenlemesi 2 ayakla sınırlı ve güven düşük kalıyor. İma edilen
  P/FFO (veya ima edilen FFO büyümesi) hesabı eklenirse 3. ayak kazanılır. (P2c)
- FFO doğruluğu: toplam D&A yerine yalnızca gayrimenkul amortismanı ayrıştırılabilirse
  ve/veya forward/run-rate FFO (yıllıklandırılmış son çeyrek) kullanılabilirse proxy
  daha da yakınsar; şu an trailing-FFO seri-hisse-ihraççılarında hâlâ küçük sapma verir.
- Döngüsel SIC kapsamı boşlukları: konut inşaatçıları (1500-1799), kağıt (2600'ler),
  makine (çoğu 3500'ler) döngüsel sette değil — normalize-earnings varyantını almıyorlar.
  (düşük öncelik)
- Greenwald EPV yöntemini bağımsız bir kaynağa karşı doğrula (kod değil, doğrulama).
  Finansal P/B×ROE'nin justified-P/B tarafı P3a'da ele alındı.
- [TAMAMLANDI] Orta-büyümeli (%12-20 gerçekleşen CAGR) zarar eden şirketler için
  revenue-first DCF: `sector.detect_hyper_grower`'ın yakalamadığı (CAGR ≤ %20) ama gerçek
  bir büyüme hikâyesi olan `growth_unprofitable` filer'lar artık, büyüme kapısını (CAGR ≥
  %12) ve bastırma guardrail'ini geçtiğinde, salt multiples yerine §3c/§4a arası kalibre
  edilmiş (8 yıllık fade, %20 hedef-marj tavanı) bir revenue-first band manşet oluyor. Bkz.
  SPEC.md §8d, VALUATION.md §4b.
