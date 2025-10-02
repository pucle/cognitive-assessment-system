# Training Data Schema

## Input Data Format

### raw_dataset.csv

Required columns:
- `session_id` (string): Unique identifier for each assessment session
- `audio_path` (string): Path to audio file (relative to dataset root or absolute)
- `mmse_true` (int): Ground truth MMSE score (0-30)
- `age` (int): Participant age in years
- `gender` (string): "male" or "female" 
- `education_years` (int): Years of formal education

Optional columns:
- `notes` (string): Additional clinical notes
- `diagnosis` (string): Clinical diagnosis if available
- `date_recorded` (string): Recording date (YYYY-MM-DD)

### Audio Files
- Format: WAV, 16 kHz, 16-bit PCM, mono
- Location: `audio/` directory relative to CSV
- Naming: Should match `audio_path` column in CSV
- Duration: Variable (typically 5-20 minutes)

## Intermediate Data Formats

### intermediate/transcripts.csv
Generated after ASR processing:
- `session_id`: Session identifier
- `transcript`: Full ASR transcript text
- `asr_confidence`: Average word confidence (0-1)
- `asr_engine`: ASR engine used (e.g., "whisper-large-v2")
- `word_timestamps`: JSON array of word-level timestamps
- `processing_time_seconds`: Time taken for transcription

### intermediate/features.csv
Generated after feature extraction:
- `session_id`: Session identifier
- `M_raw`: Raw MMSE score from automated scoring
- `L_scalar`: Linguistic scalar (0-1)
- `A_scalar`: Acoustic scalar (0-1)
- `F_flu`: Fluency score from auxiliary F1 task
- `TTR`: Type-token ratio
- `idea_density`: Idea density score
- `semantic_similarity_avg`: Average semantic similarity
- `speech_rate_wpm`: Words per minute
- `pause_rate`: Pause rate (pause_seconds/total_seconds)
- `f0_variability`: F0 variability measure
- `mfcc_1_mean` through `mfcc_13_mean`: MFCC features
- `spectral_centroid_mean`: Spectral features
- `zero_crossing_rate_mean`: Zero crossing rate
- Additional acoustic features...

### intermediate/per_item_scores.csv
Generated after item-level scoring:
- `session_id`: Session identifier
- `T1` through `L5`: Individual item scores
- `F1`: Auxiliary fluency score (not counted in total)
- `T1_confidence` through `L5_confidence`: ASR confidence per item
- `T1_error_flag` through `L5_error_flag`: Low confidence flags

## Output Data Formats

### test_predictions.csv
Final predictions on test set:
- `session_id`: Session identifier
- `mmse_true`: Ground truth score
- `mmse_pred`: Predicted score
- `M_raw`: Raw automated MMSE score
- `L_scalar`: Linguistic scalar
- `A_scalar`: Acoustic scalar
- `T1` through `L5`: Per-item scores
- `feature_sr`: Speech rate feature
- `feature_pause_ratio`: Pause ratio feature
- Additional features used in final model...

## Data Validation Rules

1. **Required files existence check**:
   - `raw_dataset.csv` must exist
   - All audio files referenced in `audio_path` must exist
   
2. **Column validation**:
   - All required columns must be present
   - No missing values in required columns
   - `mmse_true` must be integers 0-30
   - `age` must be positive integers
   - `education_years` must be non-negative integers

3. **Audio file validation**:
   - Files must be readable by librosa
   - Sample rate validation (convert to 16kHz if needed)
   - Duration > 30 seconds (minimum viable)

4. **Minimum dataset size**:
   - At least 50 labeled examples required for training
   - At least 10 examples in test set

## Error Handling

If validation fails, the pipeline will output structured error JSON:

```json
{
  "status": "error",
  "code": "missing_columns",
  "message": "raw_dataset.csv missing required columns: mmse_true, age",
  "details": {
    "missing_columns": ["mmse_true", "age"],
    "found_columns": ["session_id", "audio_path", "gender"]
  }
}
```

Common error codes:
- `missing_columns`: Required columns not found
- `missing_audio_file`: Audio files referenced but not found
- `insufficient_data`: Less than 50 training examples
- `asr_failed`: ASR processing failed
- `invalid_audio_format`: Audio files in unsupported format
