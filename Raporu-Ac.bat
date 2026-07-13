@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem reports klasorundeki en son (en yeni) HTML raporu dogrudan acar.
for /f "delims=" %%f in ('dir /b /a-d /o-d "reports\*.html" 2^>nul') do (
    start "" "reports\%%f"
    exit /b 0
)

echo reports klasorunde HTML rapor bulunamadi.
pause
