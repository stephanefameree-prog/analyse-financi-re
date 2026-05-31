@echo off
cd /d "C:\Users\steph\OneDrive\Documenten\analyse financiere"
echo Construction de dividendes_universe.json (reprise automatique)
python build_dividend_universe.py %*
pause
