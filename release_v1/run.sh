#!/bin/bash

# MMSE Assessment Pipeline Runner
# Installs dependencies and runs training pipeline to reproduce results

set -e  # Exit on error

echo "==================================="
echo "MMSE Assessment Pipeline v1.0"
echo "==================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if dataset exists
if [ ! -f "raw_dataset.csv" ]; then
    echo "Error: raw_dataset.csv not found in current directory"
    echo "Please provide dataset with columns: session_id,audio_path,mmse_true,age,gender,education_years"
    exit 1
fi

echo "✓ Dataset found: raw_dataset.csv"

# Check if audio directory exists
if [ ! -d "audio" ]; then
    echo "Warning: audio/ directory not found. Audio files should be accessible via paths in raw_dataset.csv"
fi

# Create output directory
mkdir -p release_v1_output

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available

# Run the pipeline
echo "Starting MMSE training pipeline..."
echo "This may take 30-60 minutes depending on dataset size and hardware..."

python train_pipeline.py \
    --dataset raw_dataset.csv \
    --output_dir release_v1_output \
    --seed 42

# Check if pipeline completed successfully
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "✓ Pipeline completed successfully!"
    echo "==================================="
    echo ""
    echo "Generated files in release_v1_output/:"
    echo "  - evaluation_report.pdf     (Comprehensive evaluation report)"
    echo "  - model_MMSE_v1.pkl        (Trained model)"
    echo "  - model_metadata.json      (Model configuration and weights)"
    echo "  - test_predictions.csv     (Predictions on test set)"
    echo "  - questions.json           (Assessment schema)"
    echo "  - run_log.json             (Execution log)"
    echo ""
    echo "Intermediate files:"
    echo "  - intermediate/transcripts.csv    (ASR transcriptions)"
    echo "  - intermediate/*_features.csv     (Extracted features)"
    echo "  - intermediate/*_scores.csv       (Item-level scores)"
    echo ""
    echo "Analysis results:"
    echo "  - evaluation_analysis.json        (Performance metrics)"
    echo "  - ablation_results.json          (Feature ablation study)"
    echo "  - robustness_results.json        (ASR robustness testing)"
    echo "  - item_analysis.json             (Individual item analysis)"
    echo ""
    echo "Visualizations:"
    echo "  - plots/shap_summary.png         (SHAP feature importance)"
    echo "  - plots/model_evaluation.png     (Model comparison plots)"
    echo ""
    
    # Show model performance summary
    if [ -f "release_v1_output/model_results.json" ]; then
        echo "Model Performance Summary:"
        python3 -c "
import json
with open('release_v1_output/model_results.json', 'r') as f:
    results = json.load(f)
for model, metrics in results.items():
    if 'test_rmse' in metrics:
        print(f'  {model}: RMSE={metrics[\"test_rmse\"]:.3f}, R²={metrics[\"test_r2\"]:.3f}')
        "
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Review evaluation_report.pdf for detailed analysis"
    echo "  2. Check test_predictions.csv for individual predictions" 
    echo "  3. Examine plots/ directory for visualizations"
    echo "  4. Use model_MMSE_v1.pkl for inference on new data"
    echo ""
    echo "For deployment:"
    echo "  - Encrypt audio files using encryption.py"
    echo "  - Store model artifacts securely"
    echo "  - Validate performance on independent test set"
    
else
    echo "==================================="
    echo "✗ Pipeline failed!"
    echo "==================================="
    echo ""
    echo "Check logs for error details:"
    echo "  - release_v1_output/pipeline.log"
    echo "  - release_v1_output/error.json (if present)"
    echo ""
    echo "Common issues:"
    echo "  - Insufficient memory for Whisper model"
    echo "  - Missing or corrupted audio files"
    echo "  - Invalid dataset format"
    echo "  - Missing dependencies"
    exit 1
fi
