@echo off
echo ==========================================
echo SPC-GAN Full Execution Pipeline
echo ==========================================

call venv\Scripts\activate 2>nul

echo [1/7] Preparing Dataset...
python main.py --mode prepare
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/7] Training Baseline Classifier...
python main.py --mode train_classifier
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/7] Training SPC-GAN...
python main.py --mode train_gan
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/7] Generating Synthetic Images...
python main.py --mode generate --num_samples 50
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/7] Evaluating Models...
python main.py --mode evaluate
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/7] Running Quick Demo Mode...
python main.py --mode demo
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/7] Launching Streamlit App...
streamlit run app/app.py
