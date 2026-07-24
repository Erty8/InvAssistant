@echo off
chcp 65001 >nul
title SEC Analyzer - Web Arayuzu
cd /d "%~dp0"

echo ============================================
echo    SEC Analyzer - Web Arayuzu
echo ============================================
echo.

rem === Projeyi git "main" branch'inden guncelle ===========================
rem Baslatmadan once en guncel kodu cek. Git kurulu degilse ya da internet
rem yoksa bu adim atlanir (guncelleme hatasi uygulamayi durdurmaz).
where git >nul 2>&1
if errorlevel 1 (
    echo Git bulunamadi; guncelleme atlaniyor.
    echo.
    goto :git_done
)
echo Proje guncelleniyor ^(git main^)...
git -C "%~dp0." fetch origin main
if errorlevel 1 (
    echo   Uyari: guncelleme yapilamadi ^(internet/git sorunu^); mevcut kodla devam.
    echo.
    goto :git_done
)
git -C "%~dp0." pull --ff-only origin main
if errorlevel 1 (
    echo   Uyari: "main" birlestirilmesi hizli-ileri degil; mevcut kodla devam.
    echo   ^(Yerel degisiklikleriniz olabilir; elle "git pull origin main" deneyin.^)
)
echo.
:git_done

rem === Calisan bir Python yorumlayicisi bul ===============================
rem PATH'teki "python" cogu Windows'ta Microsoft Store kisayoludur ve gercek
rem yorumlayici degildir; bu yuzden adaylari tek tek deneyip GERCEKTEN calisan
rem ilkini seciyoruz (once PATH, sonra py launcher, sonra tipik conda/Python
rem kurulum yollari).
set "PY="
call :try_py python
if not defined PY call :try_py py
if not defined PY call :try_py "%USERPROFILE%\miniconda3\python.exe"
if not defined PY call :try_py "%USERPROFILE%\anaconda3\python.exe"
if not defined PY call :try_py "%LOCALAPPDATA%\miniconda3\python.exe"
if not defined PY call :try_py "%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY call :try_py "%ProgramData%\miniconda3\python.exe"
if not defined PY call :try_py "%ProgramData%\anaconda3\python.exe"
if not defined PY (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if not defined PY call :try_py "%%D\python.exe"
    )
)

if not defined PY (
    echo HATA: Calisan bir Python bulunamadi.
    echo.
    echo   - Python kurulu degilse: https://www.python.org/downloads/ ^(kurulumda
    echo     "Add python.exe to PATH" secenegini isaretleyin^) veya Miniconda kurun.
    echo   - Kuruluysa ama PATH'te "python" Microsoft Store kisayoluna gidiyorsa:
    echo     Ayarlar ^> Uygulamalar ^> Uygulama takma adlari'ndan python takma
    echo     adlarini kapatin, ya da yorumlayicinizi PATH'e ekleyin.
    echo.
    pause
    exit /b 1
)

echo Python bulundu: %PY%
echo.

rem pip yoksa devreye almayi dene (nadir; bazi minimal kurulumlarda gerekir).
"%PY%" -m pip --version >nul 2>&1
if errorlevel 1 "%PY%" -m ensurepip --upgrade >nul 2>&1

rem === Gerekli paketler kurulu mu? Degilse requirements.txt'ten kur =========
rem Web arayuzunun ihtiyac duydugu ucuncu-parti paketleri hizlica yokla; biri
rem bile eksikse tum bagimliliklari kur (ilk calistirmada birkac dakika
rem surebilir, sonrakilerde bu adim aninda gecilir). "dotenv" (python-dotenv)
rem ozellikle onemli: kurulu degilse .env dosyasi SESSIZCE yok sayilir ve
rem gecerli bir .env'e ragmen "SEC_USER_AGENT is not set" hatasi alinir.
"%PY%" -c "import flask, pandas, requests, dotenv" >nul 2>&1
if errorlevel 1 (
    echo Gerekli paketler eksik. Kuruluyor...
    echo   "%PY%" -m pip install -r requirements.txt
    echo.
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo HATA: Paket kurulumu basarisiz oldu. Yukaridaki mesaji kontrol edin.
        pause
        exit /b 1
    )
    echo.
    echo Paketler kuruldu.
    echo.
)

rem === SEC_USER_AGENT ayarli mi? Degilse sor ve .env'e yaz =================
rem SEC EDGAR her istegin gercek bir istek sahibini tanimlamasini zorunlu tutar.
rem Kontrolu uygulamanin kendi mantigiyla yapiyoruz (ortam degiskeni + .env'i
rem python-dotenv ile okur), boylece bat ile uygulama ayni sonuca varir.
"%PY%" -c "from sec_analyzer.config import Config; Config.get_user_agent()" >nul 2>&1
if not errorlevel 1 goto :ua_ok

echo SEC EDGAR, her istekte sizi tanimlayan bir "User-Agent" bilgisi ister.
echo Bu bilgi yalnizca SEC'e giden isteklerde kullanilir ve proje kokundeki
echo .env dosyasina kaydedilir (bir daha sorulmaz).
echo Ornek bicim:  Ad Soyad email@ornek.com
echo.
set "UA_TRIES=0"
:ua_ask
set "UA="
set /p "UA=SEC_USER_AGENT girin (Ad Soyad e-posta): "
if defined UA goto :ua_have
set /a UA_TRIES+=1
if %UA_TRIES% geq 5 (
    echo   Cok fazla bos giris; cikiliyor. SEC_USER_AGENT satirini .env dosyasina
    echo   elle ekleyip ^(ornek: SEC_USER_AGENT=Ad Soyad email@ornek.com^) tekrar deneyin.
    pause
    exit /b 1
)
echo   Bos birakilamaz. Tekrar deneyin ^(cikmak icin pencereyi kapatin^).
goto :ua_ask
:ua_have
rem Proje kokundeki .env'e ekle (yoksa olusturur) ve bu oturum icin de ayarla.
>>".env" echo SEC_USER_AGENT="%UA%"
set "SEC_USER_AGENT=%UA%"
echo.
echo Kaydedildi: .env dosyasina SEC_USER_AGENT yazildi.
echo.
:ua_ok

rem === Yapay zeka arka ucu: Claude Code kullanilsin mi? ====================
rem SEC_USER_AGENT gibi: SORU yalnizca daha once secilmediyse sorulur ve secim
rem .env'e kaydedilir. ANCAK secim "claude_code" ise, soru sorulmasa bile
rem kurulum/oturum dogrulamasi (npm install vb.) HER acilista calisir -- boylece
rem "claude_code seciliydi ama claude kurulu degildi" durumu kendiliginden
rem duzeltilir. "none" ise her sey atlanir. Secim web arayuzunun varsayilan
rem saglayicisini da belirler.
if defined LLM_BACKEND goto :llm_env_set
findstr /i /c:"LLM_BACKEND=none" ".env" >nul 2>&1
if not errorlevel 1 goto :llm_ok
findstr /i /c:"LLM_BACKEND=claude_code" ".env" >nul 2>&1
if not errorlevel 1 goto :cc_from_saved
goto :llm_ask

:llm_env_set
rem Ortamda LLM_BACKEND tanimli: claude_code ise kurulumu dogrula, degilse atla.
if /i not "%LLM_BACKEND%"=="claude_code" goto :llm_ok
echo Ortam tercihi: Claude Code. Kurulum/oturum dogrulaniyor...
goto :cc_setup

:cc_from_saved
echo Kayitli tercih (.env): Claude Code. Kurulum/oturum dogrulaniyor...
goto :cc_setup

:llm_ask
echo ============================================
echo    Yapay zeka yorumu (istege bagli)
echo ============================================
echo.
echo Analiz yorumlarinda Claude Code kullanabilirsiniz: yerel "claude" komutu,
echo Claude ABONELIGINIZLE faturalanir (API ucreti DEGIL). Alternatif: yapay
echo zeka yok - deterministik, kural bazli, ucretsiz analiz.
echo.
echo Not: Degerleme SAYILARI her iki durumda da deterministik kod uretir;
echo Claude Code yalnizca yorum metnini ve varsayim onerisini etkiler.
echo.
set "USE_CC="
set /p "USE_CC=Claude Code kullanilsin mi? (E/H) [H]: "
if /i "%USE_CC%"=="E" goto :cc_setup
if /i "%USE_CC%"=="Y" goto :cc_setup

rem --- Hayir / bos: yapay zeka kapali (kural bazli) ---
>>".env" echo LLM_BACKEND=none
set "LLM_BACKEND=none"
echo.
echo Secildi: yapay zeka KAPALI (kural bazli). Istediginizde .env icindeki
echo LLM_BACKEND satirini claude_code yaparak acabilirsiniz.
echo.
goto :llm_ok

:cc_setup
set "JUST_INSTALLED="
echo.
echo Claude Code kontrol ediliyor...
rem 1) "claude" komutu PATH'te mi? Yoksa npm ile otomatik kurmayi dene.
where claude >nul 2>&1
if not errorlevel 1 goto :cc_have_bin

where npm >nul 2>&1
if errorlevel 1 goto :cc_no_npm
echo   "claude" bulunamadi -^> Claude Code CLI kuruluyor:
echo     npm install -g @anthropic-ai/claude-code
echo   ^(Ilk kurulum birkac dakika surebilir, lutfen bekleyin...^)
echo.
call npm install -g @anthropic-ai/claude-code
set "JUST_INSTALLED=1"
echo.
where claude >nul 2>&1
if not errorlevel 1 goto :cc_installed
rem PATH'te hemen gorunmediyse npm global klasorunu dogrudan hedefle.
set "NPM_PREFIX="
for /f "delims=" %%P in ('npm prefix -g 2^>nul') do set "NPM_PREFIX=%%P"
if not defined NPM_PREFIX goto :cc_install_unsure
if not exist "%NPM_PREFIX%\claude.cmd" goto :cc_install_unsure
set "CLAUDE_CODE_BIN=%NPM_PREFIX%\claude.cmd"
echo   [OK] Kuruldu; CLAUDE_CODE_BIN = %NPM_PREFIX%\claude.cmd
goto :cc_key
:cc_installed
echo   [OK] Claude Code CLI kuruldu.
goto :cc_key
:cc_install_unsure
echo   [!] Kurulum tamamlandi ama "claude" hemen gorunmuyor. Yeni bir terminal
echo       acip "claude --version" ile dogrulayin; gerekirse .env'e
echo       CLAUDE_CODE_BIN=^<claude.cmd tam yolu^> ekleyin.
goto :cc_key
:cc_no_npm
echo   [!] "claude" yok ve "npm" de bulunamadi; otomatik kurulum yapilamadi.
echo       Once Node.js kurun ^(https://nodejs.org^), sonra sunu calistirin:
echo         npm install -g @anthropic-ai/claude-code
echo       Bu arada analiz kural bazli ^(yapay zeka olmadan^) calisir.
goto :cc_key
:cc_have_bin
echo   [OK] "claude" bulundu.
:cc_key
rem 2) ANTHROPIC_API_KEY dolu mu? Dolu ise "claude -p" aboneligi DEGIL API'yi
rem    faturalar; bu yuzden Claude Code arka ucu calismayi reddeder.
set "KEY_FOUND="
if defined ANTHROPIC_API_KEY set "KEY_FOUND=1"
findstr /b /i "ANTHROPIC_API_KEY" ".env" >nul 2>&1
if not errorlevel 1 set "KEY_FOUND=1"
if defined KEY_FOUND goto :cc_key_warn
echo   [OK] ANTHROPIC_API_KEY bos ^(abonelik faturalamasi icin gerekli^).
goto :cc_login
:cc_key_warn
echo   [!] ANTHROPIC_API_KEY DOLU gorunuyor. Bu durumda "claude -p"
echo       aboneliginizi DEGIL API hesabinizi faturalar; Claude Code arka ucu
echo       bunu onlemek icin calismayi reddeder ve kural bazli moda duser.
echo       Cozum: ANTHROPIC_API_KEY'i ortamdan ve .env dosyasindan kaldirin.
:cc_login
rem 3) CLI yeni kurulduysa oturum acmasi gerekir; opsiyonel olarak simdi sor.
rem    (Zaten kuruluysa oturumu vardir varsayilir, sorulmaz.)
if not defined JUST_INSTALLED goto :cc_save
where claude >nul 2>&1
if errorlevel 1 goto :cc_save
echo.
set "DO_LOGIN="
set /p "DO_LOGIN=Claude'a simdi oturum acilsin mi? ('claude login' calisir) (E/H) [E]: "
if /i "%DO_LOGIN%"=="H" goto :cc_save
if /i "%DO_LOGIN%"=="N" goto :cc_save
echo   Tarayici acilacak; oturum acmayi tamamladiktan sonra bu pencereye donun...
call claude login
:cc_save
echo.
echo   Aboneligi dogrulamak icin bir terminalde "claude" yazip ic komut
echo   "/status" ile plani gorun ^(API kredisi degil, abonelik gorunmeli^).
rem .env'e yalnizca satir yoksa yaz (tekrar acilislarda cift satir olusmasin).
findstr /i /c:"LLM_BACKEND=claude_code" ".env" >nul 2>&1
if not errorlevel 1 goto :cc_saved
>>".env" echo LLM_BACKEND=claude_code
echo Kaydedildi: .env'e LLM_BACKEND=claude_code yazildi.
:cc_saved
set "LLM_BACKEND=claude_code"
echo.
echo Aktif: Claude Code ^(abonelik faturalamasi^).
echo.
:llm_ok

echo Sunucu baslatiliyor... tarayici birkac saniye icinde
echo   http://127.0.0.1:5050  adresinde acilacak.
echo.
echo Ticker'i acilan sayfaya yazacaksin.
echo Kapatmak icin BU pencereyi kapat (sunucu durur).
echo.

rem Sunucu bu pencerede on planda calisir; pencere kapaninca sunucu da durur.
rem Kucuk bir yardimci, sunucu ayaga kalktiktan ~3 sn sonra tarayiciyi acar.
start "" /min cmd /c "timeout /t 3 >nul & start http://127.0.0.1:5050"

"%PY%" -m sec_analyzer.web.app

echo.
echo Sunucu durdu.
pause
exit /b 0

rem ==========================================================================
rem :try_py <aday>  --  aday gercekten calisan bir Python ise %PY%'ye yazar.
rem "import sys" basariyla calisirsa aday gecerli; Store kisayolu/eksik yorumlayici
rem sifirdan farkli hata kodu doner ve secilmez.
:try_py
%1 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=%~1"
goto :eof
