@echo off
rem MMSE Assessment Pipeline Runner for Windows
rem Installs dependencies and runs training pipeline to reproduce results

echo ===================================
echo MMSE Assessment Pipeline v1.0
echo ===================================

rem Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✓ Python found

rem Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

rem Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

rem Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

rem Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

rem Check if dataset exists
if not exist "raw_dataset.csv" (
    echo Error: raw_dataset.csv not found in current directory
    echo Please provide dataset with columns: session_id,audio_path,mmse_true,age,gender,education_years
    pause
    exit /b 1
)

echo ✓ Dataset found: raw_dataset.csv

rem Check if audio directory exists
if not exist "audio" (
    echo Warning: audio/ directory not found. Audio files should be accessible via paths in raw_dataset.csv
)

rem Create output directory
if not exist "release_v1_output" mkdir release_v1_output

rem Set environment variables for reproducibility
set PYTHONHASHSEED=42
set CUDA_VISIBLE_DEVICES=0

rem Run the pipeline
echo Starting MMSE training pipeline...
echo This may take 30-60 minutes depending on dataset size and hardware...

python train_pipeline.py --dataset raw_dataset.csv --output_dir release_v1_output --seed 42

if errorlevel 1 (
    echo ===================================
    echo ✗ Pipeline failed!
    echo ===================================
    echo.
    echo Check logs for error details:
    echo   - release_v1_output\pipeline.log
    echo   - release_v1_output\error.json (if present)
    echo.
    echo Common issues:
    echo   - Insufficient memory for Whisper model
    echo   - Missing or corrupted audio files
    echo   - Invalid dataset format
    echo   - Missing dependencies
    pause
    exit /b 1
)

echo ===================================
echo ✓ Pipeline completed successfully!
echo ===================================
echo.
echo Generated files in release_v1_output\:
echo   - evaluation_report.pdf     (Comprehensive evaluation report)
echo   - model_MMSE_v1.pkl        (Trained model)
echo   - model_metadata.json      (Model configuration and weights)
echo   - test_predictions.csv     (Predictions on test set)
echo   - questions.json           (Assessment schema)
echo   - run_log.json             (Execution log)
echo.
echo Intermediate files:
echo   - intermediate\transcripts.csv    (ASR transcriptions)
echo   - intermediate\*_features.csv     (Extracted features)
echo   - intermediate\*_scores.csv       (Item-level scores)
echo.
echo Analysis results:
echo   - evaluation_analysis.json        (Performance metrics)
echo   - ablation_results.json          (Feature ablation study)
echo   - robustness_results.json        (ASR robustness testing)
echo   - item_analysis.json             (Individual item analysis)
echo.
echo Visualizations:
echo   - plots\shap_summary.png         (SHAP feature importance)
echo   - plots\model_evaluation.png     (Model comparison plots)
echo.
echo Next steps:
echo   1. Review evaluation_report.pdf for detailed analysis
echo   2. Check test_predictions.csv for individual predictions
echo   3. Examine plots\ directory for visualizations
echo   4. Use model_MMSE_v1.pkl for inference on new data
echo.
echo For deployment:
echo   - Encrypt audio files using encryption.py
echo   - Store model artifacts securely
echo   - Validate performance on independent test set

pause
