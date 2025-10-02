@echo off
echo ===========================================
echo    COGNITIVE ASSESSMENT DUAL MODEL SYSTEM
echo ===========================================
echo.
echo This script runs both models:
echo 1. Original Cognitive Assessment ML
echo 2. Enhanced MMSE Assessment v2.0
echo.
echo Press any key to continue...
pause > nul

echo.
echo [STEP 1] Setting up backend environment...
cd backend

echo.
echo [STEP 2] Activating virtual environment...
if exist ".ok\Scripts\activate.bat" (
    echo Using existing .ok environment...
    call .ok\Scripts\activate.bat
) else (
    echo Creating new virtual environment...
    python -m venv dual_env
    call dual_env\Scripts\activate.bat

    echo Installing dependencies...
    pip install -r requirements.txt
    pip install python-Levenshtein sentence-transformers fpdf2 cryptography
)

echo.
echo [STEP 3] Setting up MMSE v2.0 files...
if not exist "questions.json" (
    echo Copying questions.json...
    copy ..\release_v1\questions.json . > nul
)

if not exist "services\mmse_assessment_service.py" (
    echo Copying MMSE services...
    copy ..\release_v1\scoring_engine.py services\ > nul
    copy ..\release_v1\feature_extraction.py services\ > nul
    copy ..\release_v1\encryption.py services\ > nul
    copy ..\release_v1\mmse_assessment_service.py services\ > nul
)

echo.
echo [STEP 4] Testing imports...
python -c "
try:
    import cognitive_assessment_ml
    print('✓ Old model available')
except ImportError as e:
    print('⚠ Old model import issue:', e)

try:
    from services.mmse_assessment_service import get_mmse_service
    service = get_mmse_service()
    print('✓ MMSE v2.0 service available')
    print(f'  - Scorer: {service.get_model_info()[\"scorer_available\"]}')
    print(f'  - Questions: {len(service.get_questions())}')
except ImportError as e:
    print('⚠ MMSE v2.0 import issue:', e)
"

echo.
echo [STEP 5] Starting backend server...
start "Cognitive Assessment Backend" cmd /k "python start_dual_model.py"

echo.
echo [STEP 6] Starting frontend...
cd ../frontend
start "Cognitive Assessment Frontend" cmd /k "npm run dev"

echo.
echo ===========================================
echo          SYSTEM STARTUP COMPLETE!
echo ===========================================
echo.
echo 🌐 Available Services:
echo    Backend API: http://localhost:5001
echo    Frontend:    http://localhost:3000
echo    MMSE v2:     http://localhost:3000/mmse-v2
echo.
echo 📊 API Endpoints:
echo.
echo OLD MODEL (Original):
echo    POST /api/assess          - Full assessment
echo    POST /api/transcribe      - Audio transcription
echo    POST /api/features        - Feature extraction
echo    GET  /api/health          - Health check
echo.
echo MMSE v2.0 (Enhanced):
echo    POST /api/mmse/assess     - MMSE assessment
echo    GET  /api/mmse/questions  - Get questions
echo    GET  /api/mmse/model-info - Model status
echo    POST /api/mmse/transcribe - Audio transcription
echo.
echo 🧪 Test commands:
echo    curl http://localhost:5001/api/health
echo    curl http://localhost:5001/api/mmse/model-info
echo.
echo Press any key to exit setup...
pause > nul
