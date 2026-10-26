# tezler.xlsx üretimi

Repo kökünden çalıştırın:

```bash
PYTHONPATH=. python thesis/_build/technical_stats.py [--live]   # önce teknik: --live portföy ağırlıkları için gerekir
PYTHONPATH=thesis/_build python thesis/_build/portfolio_stats.py [--live]
PYTHONPATH=thesis/_build python thesis/_build/build_tezler.py thesis/tezler.xlsx thesis/_build/new_theses.json thesis/_build/portfolio_stats.json thesis/_build/technical.json
```

- `build_tezler.py`: MU, NVDA, NFLX, APP, UBER verisi betiğin içinde; diğer tezler `new_theses.json`'dan.
- `portfolio_stats.py`: pozisyon tutarlarını git dışı `thesis/_private/portfolio_positions.json`'dan okur
  (dosya yoksa commit'li ağırlıkları kullanır), risk metriklerini `sec_analyzer/raw/prices_<TICKER>_full.csv`
  önbelleğinden hesaplar ve `portfolio_stats.json`'a **yalnızca ağırlık ve risk** yazar; tutar asla yazılmaz.
- `thesis/_private/` `.gitignore`'da: pozisyon tutarları commit edilmez.
- Yeni tez eklemek: `new_theses.json`'a aynı şemada bir nesne ekleyip ikinci komutu çalıştırın.
- `technical_stats.py`: motorun teknik katmanını her tez hissesine önbellekteki fiyatlarla uygular ve `technical.json`'a yazar; "Teknik" sayfası buradan üretilir.
