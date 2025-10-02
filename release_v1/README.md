# MMSE-like Assessment System v1.0

Automated Vietnamese cognitive assessment system using audio + transcript analysis for MMSE-like scoring.

## Overview

This system implements a comprehensive pipeline for automated cognitive assessment that:

- **Automatically scores** 30-point MMSE-like assessments from audio recordings
- **Combines linguistic and acoustic features** for enhanced prediction accuracy  
- **Uses machine learning** (XGBoost, LightGBM, Ridge) with NNLS weight optimization
- **Provides interpretability** through SHAP analysis and ablation studies
- **Ensures privacy** with AES-256 audio encryption
- **Supports Vietnamese language** with fuzzy matching and semantic similarity

## Quick Start

### Prerequisites

- Python 3.8+
- 16GB+ RAM (for Whisper ASR model)
- CUDA-capable GPU (optional, for faster processing)

### Installation

```bash
# Clone or download the release_v1/ directory
cd release_v1/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional ASR models
# pip install underthesea  # Vietnamese NLP
```

### Usage

1. **Prepare your dataset** as `raw_dataset.csv` with required columns:
   ```csv
   session_id,audio_path,mmse_true,age,gender,education_years
   session_001,audio/recording_001.wav,25,72,female,12
   session_002,audio/recording_002.wav,18,68,male,8
   ```

2. **Run the training pipeline**:
   ```bash
   python train_pipeline.py --dataset raw_dataset.csv --output_dir release_v1 --seed 42
   ```

3. **Check results** in the output directory:
   - `evaluation_report.pdf` - Comprehensive evaluation report
   - `model_MMSE_v1.pkl` - Trained model
   - `test_predictions.csv` - Predictions on test set
   - `model_metadata.json` - Model configuration and weights

## System Architecture

### Pipeline Components

1. **Audio Processing**
   - Whisper ASR for Vietnamese transcription
   - Acoustic feature extraction (MFCC, spectral, prosodic)
   - Speech timing analysis (pauses, rate)

2. **Linguistic Analysis**
   - Fuzzy string matching (Levenshtein ≥ 0.8)
   - Semantic similarity (multilingual SBERT ≥ 0.7)
   - Vietnamese-specific linguistic features

3. **Automated Scoring**
   - 11 MMSE items + 1 auxiliary fluency task
   - Domain-specific scoring rules
   - Confidence-based error flagging

4. **Feature Engineering**
   - **L_scalar**: Linguistic composite (F_flu: 40%, TTR: 30%, ID: 20%, Semantic: 10%)
   - **A_scalar**: Acoustic composite (Speech rate: 50%, Pauses: 30%, F0: 20%)
   - **M_raw**: Raw automated MMSE score

5. **Model Training**
   - NNLS weight optimization: `Score = w_M*M + w_L*(L*30) + w_A*(A*30)`
   - Multiple ML models with hyperparameter tuning
   - 5-fold cross-validation with early stopping

### Assessment Domains

| Domain | Items | Max Points | Description |
|--------|--------|------------|-------------|
| Time Orientation | T1 | 5 | Date, weekday, time of day |
| Place Orientation | P1 | 5 | Country, province, city, district, specific place |
| Registration | R1 | 3 | Immediate recall of 3 words |
| Attention | A1 | 5 | Serial 7s subtraction |
| Recall | D1 | 3 | Delayed recall of 3 words |
| Naming | L1, L5 | 3 | Object naming and description |
| Language | L2, L3, L4 | 5 | Repetition, commands, sentence construction |
| Visuospatial | V1 | 1 | Clock reading (verbal) |
| **Total** | | **30** | |
| Auxiliary | F1 | - | Fluency (not counted in total) |

## File Structure

```
release_v1/
├── train_pipeline.py          # Main training pipeline
├── scoring_engine.py          # MMSE item scoring logic
├── feature_extraction.py      # Linguistic/acoustic features
├── evaluation_analysis.py     # SHAP, ablation, robustness
├── report_generator.py        # PDF report generation
├── encryption.py              # AES-256 audio encryption
├── questions.json             # 30-point assessment schema
├── requirements.txt           # Python dependencies
├── tests/
│   └── test_mmse_pipeline.py  # Unit tests
├── intermediate/              # Generated during training
│   ├── transcripts.csv
│   ├── train_features.csv
│   └── ...
├── plots/                     # Generated visualizations
│   ├── shap_summary.png
│   └── model_evaluation.png
├── model_MMSE_v1.pkl         # Trained model
├── model_metadata.json       # Model configuration
├── test_predictions.csv      # Final predictions
├── evaluation_report.pdf     # Comprehensive report
└── run_log.json              # Execution log
```

## Configuration

### Scoring Parameters

- **Fuzzy matching threshold**: 0.80 (Levenshtein ratio)
- **Semantic similarity threshold**: 0.70 (SBERT cosine)
- **Serial 7s tolerance**: ±1 per step
- **ASR confidence threshold**: 0.60 (error flagging)

### Model Parameters

```python
# XGBoost (default best model)
n_estimators=500
max_depth=6
learning_rate=0.01
subsample=0.8
early_stopping_rounds=50
```

### Audio Requirements

- **Format**: WAV, 16 kHz, 16-bit PCM, mono
- **Duration**: 5-20 minutes typical
- **Quality**: Clear speech, minimal background noise
- **Language**: Vietnamese

## Evaluation Metrics

### Regression Performance
- **RMSE**: Root mean squared error (target ≤3.5)
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **95% CI**: Bootstrap confidence intervals

### Classification Performance (≤23 cutoff)
- **Sensitivity**: Recall for cognitive impairment
- **Specificity**: True negative rate for normal cognition
- **Precision**: Positive predictive value
- **F1-score**: Harmonic mean of precision/recall

### Interpretability
- **SHAP values**: Feature importance and explanations
- **Ablation study**: Feature group contributions
- **Item analysis**: Individual question psychometrics

### Robustness
- **ASR error simulation**: Word deletion/substitution
- **Confidence degradation**: Performance vs ASR quality
- **Cross-validation**: 5-fold stratified validation

## Privacy and Security

### Audio Encryption (AES-256-GCM)

```python
from encryption import AudioEncryption

# Encrypt dataset
encryptor = AudioEncryption()
encryptor.encrypt_dataset_audio(
    dataset_csv="raw_dataset.csv",
    audio_dir="audio/",
    encrypted_dir="encrypted_audio/", 
    key_file="encryption_key.bin"
)
```

### Key Management

- **Key length**: 256-bit AES-GCM
- **Key derivation**: PBKDF2-HMAC-SHA256 (100,000 iterations)
- **File permissions**: Owner read/write only (0o600)
- **Key rotation**: Regenerate keys periodically
- **Storage**: Store keys separately from encrypted data

### Data Protection

- Audio files encrypted at rest
- No raw audio stored in model artifacts
- Transcript data anonymized
- Compliance with healthcare data regulations

## Troubleshooting

### Common Issues

1. **Out of memory during ASR**
   ```bash
   # Reduce batch size or use smaller Whisper model
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

2. **Audio format errors**
   ```bash
   # Convert audio to required format
   ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
   ```

3. **Vietnamese text encoding**
   ```python
   # Ensure UTF-8 encoding
   df = pd.read_csv('dataset.csv', encoding='utf-8')
   ```

4. **Missing dependencies**
   ```bash
   # Install specific versions if conflicts
   pip install torch==1.9.0 transformers==4.15.0
   ```

### Performance Optimization

- **GPU acceleration**: Install CUDA-enabled PyTorch
- **Parallel processing**: Use multiple workers for feature extraction
- **Memory optimization**: Process audio files in batches
- **Model compression**: Use quantized models for deployment

## Validation and Testing

### Unit Tests

```bash
# Run test suite
cd tests/
python test_mmse_pipeline.py

# Or with pytest
pytest test_mmse_pipeline.py -v
```

### Manual Validation

1. **Check questions total**: Should sum to 30 points (excluding F1)
2. **Verify feature ranges**: L_scalar, A_scalar ∈ [0,1]
3. **Test encryption**: Encrypt/decrypt cycle preserves data
4. **Validate scoring**: Known responses produce expected scores

### Cross-validation

The pipeline automatically performs:
- 70/15/15 train/val/test split (stratified by MMSE score)
- 5-fold cross-validation on training set
- Bootstrap confidence intervals (1000 resamples)

## API Reference

### Core Classes

```python
from scoring_engine import MMSEScorer
from feature_extraction import FeatureExtractor
from train_pipeline import MMSEAssessmentPipeline

# Initialize components
scorer = MMSEScorer(sbert_model=None)
extractor = FeatureExtractor()
pipeline = MMSEAssessmentPipeline("output_dir")

# Score single session
session_data = {
    'session_id': 'test_001',
    'transcript': 'Patient responses...',
    'asr_confidence': 0.85
}
result = scorer.score_session(session_data)
```

### Feature Extraction

```python
# Extract features
linguistic_features = extractor.extract_linguistic_features(
    transcript, per_item_scores
)
acoustic_features = extractor.extract_acoustic_features(audio_path)

# Compute scalars
L_scalar, A_scalar = extractor.compute_scalars(
    linguistic_features, acoustic_features
)
```

### Encryption

```python
from encryption import AudioEncryption

# Create encryptor
encryptor = AudioEncryption()

# Encrypt single file
encryptor.encrypt_file("input.wav", "output.wav.enc")

# Decrypt
encryptor.decrypt_file("output.wav.enc", "decrypted.wav")
```

## Contributing

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write unit tests for new features

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Implement with tests
4. Update documentation
5. Submit pull request

### Reporting Issues
- Use GitHub issues for bug reports
- Include system information and error logs
- Provide minimal reproduction example
- Check existing issues first

## Citation

If you use this system in your research, please cite:

```bibtex
@software{mmse_assessment_v1,
  title={MMSE-like Assessment System v1.0},
  author={Cognitive Assessment Team},
  year={2025},
  url={https://github.com/your-repo/mmse-assessment},
  version={1.0}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: This README and inline code comments
- **Issues**: GitHub issue tracker
- **Email**: support@yourorganization.com
- **Demo**: See `tests/` directory for usage examples

## Changelog

### Version 1.0.0 (2025-01-XX)
- Initial release
- 30-point MMSE assessment automation
- Vietnamese language support
- XGBoost/LightGBM models with NNLS weights
- SHAP interpretability analysis
- AES-256 audio encryption
- Comprehensive evaluation pipeline
- PDF report generation

---

**Note**: This system is for research purposes and should be validated in clinical settings before diagnostic use.
