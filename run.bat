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
