@echo off
echo ==========================================
echo SPC-GAN Environment Verification
echo ==========================================

call venv\Scripts\activate 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Virtual environment not found. Using system Python...
)

python -c "import torch; print('PyTorch:', torch.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch is not installed.
    exit /b 1
) else (
    python -c "import torch; print('PyTorch:', torch.__version__)"
)

python -c "import torchvision; print('Torchvision:', torchvision.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Torchvision is not installed.
    exit /b 1
) else (
    python -c "import torchvision; print('Torchvision:', torchvision.__version__)"
)

python -c "import streamlit; print('Streamlit:', streamlit.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Streamlit is not installed.
    exit /b 1
) else (
    python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
)

echo [INFO] Environment verified successfully!
