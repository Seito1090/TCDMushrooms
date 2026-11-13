@echo off
setlocal

cd /d "%~dp0"

python -m venv .venv
if errorlevel 1 exit /b %errorlevel%

".venv\Scripts\python.exe" -m pip install seaborn numpy matplotlib scipy pandas ipykernel
if errorlevel 1 exit /b %errorlevel%

echo Setup complete. Activate the environment with:
echo   .venv\Scripts\activate
