@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title SEC Analyzer

rem Bu dosya bulundugu dizini proje koku kabul eder.
cd /d "%~dp0"

echo ============================================
echo    SEC Analyzer - Sirket Analizi
echo ============================================
echo.

rem Ticker: 1. arguman olarak verilebilir, yoksa sorulur.
set "TICKER=%~1"
if "!TICKER!"=="" set /p "TICKER=Ticker (orn. AAPL): "
if "!TICKER!"=="" (
    echo Ticker girmediniz. Cikiliyor.
    pause
    exit /b 1
)

rem Vade: 2. arguman olarak verilebilir (3m/1y/5y), yoksa sorulur.
set "HORIZON=%~2"
if not "!HORIZON!"=="" goto :have_horizon

echo.
echo Vade seciniz:
echo   1^) 3m  - kisa vade (teknik agirlikli)
echo   2^) 1y  - dengeli (varsayilan)
echo   3^) 5y  - uzun vade (fundamental agirlikli)
echo.
set "HSEL="
set /p "HSEL=Secim [1/2/3] (bos=1y): "
if "!HSEL!"=="1" set "HORIZON=3m"
if "!HSEL!"=="2" set "HORIZON=1y"
if "!HSEL!"=="3" set "HORIZON=5y"
if "!HSEL!"=="" set "HORIZON=1y"
:have_horizon
if not defined HORIZON set "HORIZON=1y"

rem --- Analiz motoru ---
rem script    = tamamen offline, API/Ollama gerektirmez (varsayilan, guvenli)
rem anthropic = Claude API ile iki asamali yorum (ANTHROPIC_API_KEY gerekir)
rem ollama    = yerel Ollama modeli
set "PROVIDER=script"

echo.
echo !TICKER! icin analiz calisiyor (vade=!HORIZON!, motor=!PROVIDER!)...
echo.

python -m sec_analyzer.cli analyze "!TICKER!" --horizon !HORIZON! --provider !PROVIDER! --html --years 12
if errorlevel 1 (
    echo.
    echo HATA: Analiz basarisiz oldu. Yukaridaki mesaji kontrol edin.
    pause
    exit /b 1
)

rem En son uretilen HTML raporu bul ve tarayicida ac.
set "LATEST="
for /f "delims=" %%f in ('dir /b /a-d /o-d "reports\*.html" 2^>nul') do (
    set "LATEST=%%f"
    goto :open
)

:open
if defined LATEST (
    echo.
    echo Rapor aciliyor: reports\!LATEST!
    start "" "reports\!LATEST!"
) else (
    echo.
    echo UYARI: reports klasorunde HTML rapor bulunamadi.
)

echo.
pause
