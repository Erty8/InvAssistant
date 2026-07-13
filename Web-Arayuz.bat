@echo off
chcp 65001 >nul
title SEC Analyzer - Web Arayuzu
cd /d "%~dp0"

echo ============================================
echo    SEC Analyzer - Web Arayuzu
echo ============================================
echo.
echo Sunucu baslatiliyor... tarayici birkac saniye icinde
echo   http://127.0.0.1:5000  adresinde acilacak.
echo.
echo Ticker'i acilan sayfaya yazacaksin.
echo Kapatmak icin BU pencereyi kapat (sunucu durur).
echo.

rem Sunucu bu pencerede on planda calisir; pencere kapaninca sunucu da durur.
rem Kucuk bir yardimci, sunucu ayaga kalktiktan ~3 sn sonra tarayiciyi acar.
start "" /min cmd /c "timeout /t 3 >nul & start http://127.0.0.1:5000"

python -m sec_analyzer.web.app

echo.
echo Sunucu durdu.
pause
