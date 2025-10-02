# MMSE v2.0 Integration Guide

## Tổng quan

Hướng dẫn này giúp bạn tích hợp và test hệ thống MMSE v2.0 mới vào ứng dụng Cognitive Assessment hiện tại.

## Thay đổi chính

### 🧠 **MMSE Model v2.0**
- **30-point assessment** với 11 items + 1 auxiliary task
- **Fuzzy matching** (Levenshtein ≥ 0.8) + **semantic similarity** (≥ 0.7)
- **L_scalar**: Linguistic features (TTR, idea density, fluency, semantic)
- **A_scalar**: Acoustic features (speech rate, pauses, F0 variability)
- **NNLS weight optimization**: Optimal combination of M_raw, L_scalar, A_scalar
- **Vietnamese-specific** scoring rules và language processing

### 🚀 **New Backend API Endpoints**
- `POST /api/mmse/assess` - Complete MMSE assessment
- `GET /api/mmse/questions` - Get questions schema
- `GET /api/mmse/model-info` - Model status and metadata
- `POST /api/mmse/transcribe` - Audio transcription only

### 💻 **New Frontend Component**
- **MMSEv2Assessment** component với tabs interface
- **Real-time recording** và file upload
- **Detailed results** với item scores, features, recommendations
- **Patient information** form
- **Progressive UI** với loading states

## Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Activate environment
source .ok/bin/activate  # Linux/Mac
# or
.ok\Scripts\activate     # Windows

# Install additional dependencies for MMSE v2
pip install sentence-transformers
pip install python-Levenshtein
pip install fpdf2

# Start backend
python app.py
```

### 2. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies (if not already done)
npm install

# Start development server
npm run dev
```

### 3. Model Setup

Hệ thống sẽ tự động tìm model trong thư mục `release_v1/`. Nếu chưa có model trained:

```bash
# Train new model (optional)
cd release_v1/
python train_pipeline.py --dataset raw_dataset.csv --output_dir . --seed 42
```

## Testing Integration

### Automated Testing

Chạy script test tích hợp:

```bash
python test_mmse_v2_integration.py
```

Script này sẽ test:
- ✅ Backend health và API endpoints
- ✅ MMSE model availability
- ✅ Questions loading
- ✅ Audio transcription
- ✅ Complete assessment pipeline
- ✅ Frontend accessibility
- ✅ MMSE v2 page functionality

### Manual Testing

1. **Truy cập ứng dụng**: http://localhost:3000
2. **Vào menu chính**: Chọn "MMSE Assessment v2.0"
3. **Test workflow**:
   - Check "Questions" tab để xem 30-point structure
   - Nhập "Patient Info" (optional)
   - "Record/Upload" audio file
   - "Run MMSE Assessment"
   - Xem "Results" với detailed analysis

### Expected Results

**Successful Assessment Should Include:**
- **MMSE Scores**: M_raw, L_scalar, A_scalar, final_score
- **Item Scores**: T1-L5 individual scores
- **Features**: Linguistic và acoustic features
- **Transcription**: Text với confidence score
- **Cognitive Status**: Normal/mild/moderate/severe impairment
- **Recommendations**: Clinical suggestions

## API Documentation

### POST /api/mmse/assess

**Request:**
```http
POST /api/mmse/assess
Content-Type: multipart/form-data

audio: <audio file>
session_id: optional_session_id
patient_info: {"name":"John","age":65,"gender":"male","education_years":12}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "session_12345",
    "status": "success",
    "timestamp": "2025-01-07T10:30:00",
    "mmse_scores": {
      "M_raw": 25,
      "L_scalar": 0.75,
      "A_scalar": 0.68,
      "final_score": 26,
      "ml_prediction": 25.8
    },
    "item_scores": {
      "T1": 5, "P1": 4, "R1": 3, "A1": 5, "D1": 2, "L1": 2, "L2": 1, "L3": 2, "L4": 1, "V1": 1, "L5": 1
    },
    "features": {
      "linguistic": {"TTR": 0.65, "idea_density": 0.45, "F_flu": 0.8, "word_count": 150},
      "acoustic": {"speech_rate_wpm": 120, "pause_rate": 0.15, "f0_variability": 25.5, "f0_mean": 180}
    },
    "cognitive_status": {
      "status": "normal",
      "risk_level": "low",
      "description": "Nhận thức bình thường",
      "confidence": 0.85,
      "recommendations": ["Duy trì lối sống lành mạnh", "Tập thể dục thường xuyên"]
    },
    "transcription": {
      "text": "Patient's spoken responses...",
      "confidence": 0.92,
      "language": "vi"
    }
  }
}
```

### GET /api/mmse/questions

**Response:**
```json
{
  "success": true,
  "data": {
    "questions": [
      {
        "id": "T1",
        "domain": "orientation_time",
        "question_text": "Hôm nay là ngày mấy (ngày/tháng/năm)?",
        "answer_type": "date_numeric",
        "max_points": 5,
        "scoring_rule": "day(1)+month(1)+year(1)+weekday(1)+time_of_day(1)"
      }
    ],
    "total_points": 30
  }
}
```

## Troubleshooting

### Common Issues

1. **"MMSE v2.0 service not available"**
   ```bash
   # Check if release_v1 directory exists
   ls release_v1/
   
   # Check Python path
   export PYTHONPATH="$PWD/release_v1:$PYTHONPATH"
   ```

2. **Import errors for sentence-transformers**
   ```bash
   pip install sentence-transformers==2.2.0
   pip install torch>=1.9.0
   ```

3. **Audio transcription fails**
   ```bash
   # Install Whisper
   pip install openai-whisper
   
   # Check audio format (should be WAV, 16kHz)
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

4. **Model not found**
   ```bash
   # Train model first
   cd release_v1/
   python train_pipeline.py --dataset ../backend/dx-mmse.csv --output_dir . --seed 42
   ```

5. **Frontend build errors**
   ```bash
   cd frontend/
   rm -rf .next/
   npm run build
   ```

### Performance Optimization

- **GPU Acceleration**: Install CUDA-enabled PyTorch for faster Whisper processing
- **Memory Management**: Increase Node.js memory limit if needed
- **Caching**: MMSE service caches models for faster subsequent assessments

## Model Performance

**Expected Performance Metrics:**
- **RMSE**: ≤ 3.5 points
- **Classification Accuracy**: ≥ 85% (cognitive impairment detection)
- **Sensitivity**: ≥ 80% (detecting impairment)
- **Specificity**: ≥ 90% (avoiding false positives)

## Deployment Notes

### Production Setup

1. **Environment Variables**:
   ```bash
   export MMSE_MODEL_PATH="/path/to/release_v1"
   export WHISPER_MODEL_SIZE="large-v2"
   export ASR_CONFIDENCE_THRESHOLD="0.6"
   ```

2. **Model Security**:
   - Use AES-256 encryption for audio files
   - Store models in secure location
   - Regular model updates and validation

3. **Monitoring**:
   - Log assessment requests and results
   - Monitor model performance metrics
   - Alert on high error rates

## Support

- **Backend Issues**: Check `backend/cognitive_assessment.log`
- **Frontend Issues**: Check browser developer console
- **Model Issues**: Check `release_v1/pipeline.log`
- **Integration Issues**: Run `test_mmse_v2_integration.py` for diagnostics

---

**Developed by**: Cognitive Assessment Team  
**Version**: 2.0  
**Last Updated**: January 2025
