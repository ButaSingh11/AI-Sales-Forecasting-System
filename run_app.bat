@echo off
setlocal

cd /d "%~dp0app"

if exist "..\.venv\Scripts\python.exe" (
    call "..\.venv\Scripts\activate.bat"
)

python -m streamlit run app.py --server.port 8501

endlocal
