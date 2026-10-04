@echo off
rem Launch the Streamlit dashboard detached from this shell.
rem The virtualenv lives one level up: ..\.venv
cd /d "C:\Users\venka\OneDrive\Documents\Default Project\student-emotion-detection"
"..\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false > artifacts\streamlit.log 2>&1