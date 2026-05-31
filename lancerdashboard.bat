@echo off
cd /d "C:\Users\steph\OneDrive\Documenten\analyse financiere"
python -m streamlit cache clear
python -m streamlit run dashboardV6.3.py
pause
