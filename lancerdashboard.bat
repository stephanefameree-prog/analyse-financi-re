@echo off
chcp 65001 >nul
title Analyse financiere - Dashboard V6.3

set "ROOT=C:\Users\steph\OneDrive\Documenten\analyse financiere"
set "PORT=8501"

cd /d "%ROOT%"
if errorlevel 1 (
    echo ERREUR: dossier introuvable
    echo %ROOT%
    pause
    exit /b 1
)

set "PY=C:\Program Files\Python314\python.exe"
if not exist "%PY%" (
    where py >nul 2>&1
    if errorlevel 1 (
        echo ERREUR: Python introuvable.
        pause
        exit /b 1
    )
    set "PY=py"
    set "PYFLAG=-3"
) else (
    set "PYFLAG="
)

echo.
echo === Dashboard financier V6.3 ===
echo Dossier : %ROOT%
echo URL     : http://localhost:%PORT%
echo.

echo Liberation du port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

"%PY%" %PYFLAG% -m streamlit cache clear 2>nul
"%PY%" %PYFLAG% -m streamlit run dashboardV6.3.py --server.port %PORT% --browser.gatherUsageStats false

echo.
echo L'application s'est arretee.
pause
