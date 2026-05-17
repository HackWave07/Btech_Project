@echo off
echo ==========================================
echo SPC-GAN Environment Setup
echo ==========================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from python.org and ensure "Add to PATH" is checked.
    exit /b 1
)

echo [INFO] Python found. Setting up environment...

if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing required packages...
pip install -r requirements.txt

echo ==========================================
echo Setup Complete!
echo Run 'venv\Scripts\activate' to enter the environment.
echo ==========================================
