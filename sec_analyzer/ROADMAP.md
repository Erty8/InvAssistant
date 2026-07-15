# Roadmap

## Faz 2

- Watchlist + verdict değişim bildirimi
- 8-K takibi + AI özet
- Portfolio review (POZISYONLAR.md entegrasyonu)
- Peer comparison
- Risk faktörü diff'i (ardışık 10-K/10-Q Risk Factors karşılaştırması)
- Verdict backtest raporu (verdicts tablosu şimdiden bunun için kuruldu)
- CapEx-yoğun hiper-grower'lar (ör. APLD gibi veri merkezi yatırımcıları) için revenue-first
  DCF'te büyüme CapEx'ini bakım CapEx'inden ayırma — veya bu şirketleri asset-bazlı/kapasite
  EV-çarpanı yöntemiyle değerleme. Mevcut bastırma (suppression) guardrail'i doğru ama
  muhafazakâr bir önlemdir, bu şirketlerin gerçek bir değerlemesi değildir; büyüme yatırımı
  yakımını gelire-ölçeklenmiş bir operasyon marjı gibi ele almak yerine kalıcı bir çözüm
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
- Bilinçli olarak DEĞİŞTİRİLMEDİ: orta-büyümeli (%12-20) zarar eden şirketler için
  revenue-first modeli (spekülatif bir değer üretmek yerine multiples'a düşmek tercih
  edildi). Gerçek şirketlerin eksik değerlendiği kanıtlanırsa yeniden ele alınabilir.
