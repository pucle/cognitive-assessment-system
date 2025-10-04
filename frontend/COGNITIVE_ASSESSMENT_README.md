# 🧠 Cognitive Assessment Frontend - Phiên bản mới

## ✨ Tính năng mới

### 🎵 Audio Feature Extraction Tự động
- **librosa**: Trích xuất 9+ đặc trưng âm học tự động
- **soundfile**: Xử lý audio files (WAV, MP3, WEBM)
- **Features**: duration, pitch, energy, tempo, silence, MFCC, spectral features

### 🤖 Model MMSE Mới
- **Algorithm**: Ridge Regression với 15 features
- **Training**: Sử dụng dataset dx-mmse.csv và progression.csv
- **Features**: age, gender, dx_encoded + audio features
- **Output**: Điểm MMSE (0-30) + mức độ suy giảm

### 🧠 GPT-3.5 Evaluation
- **Model**: gpt-3.5-turbo thay vì O4-mini
- **Metrics**: repetition_rate, vocabulary_score, context_relevance
- **Language**: Vietnamese với prompt tối ưu

## 🔧 Cách sử dụng

### 1. Khởi động Backend
```bash
cd backend
python app.py
```

### 2. Khởi động Frontend
```bash
cd frontend
npm run dev
```

### 3. Quy trình đánh giá
1. **Nhập thông tin cá nhân**: Tên, tuổi, giới tính
2. **Ghi âm câu trả lời**: Nhấn nút ghi âm và nói
3. **Nhập transcript**: Gõ lại nội dung đã nói
4. **Gửi đánh giá**: Backend xử lý tự động
5. **Xem kết quả**: MMSE + GPT + Audio features

## 📊 Kết quả đánh giá

### MMSE Prediction (12-item Audio-First Configuration)
- **≥24 điểm**: Nhận thức bình thường (Primary cutoff for cognitive impairment)
- **18-23 điểm**: Suy giảm nhận thức nhẹ (MCI)
- **10-17 điểm**: Suy giảm nhận thức trung bình
- **0-9 điểm**: Suy giảm nhận thức nặng

### Clinical Validation Targets
- **Primary**: RMSE ≤2.5 điểm so với gold-standard MMSE
- **Secondary**: Sensitivity ≥0.80, Specificity ≥0.75 tại cutoff ≤24
- **Stability**: ICC ≥0.85 cho test-retest reliability

### Audio Features
- **Duration**: Thời lượng ghi âm
- **Pitch**: Cao độ trung bình và độ lệch
- **Energy**: Năng lượng âm thanh
- **Tempo**: Nhịp độ nói
- **Silence**: Khoảng nghỉ giữa các từ
- **Spectral**: Tần số trung tâm và rolloff
- **MFCC**: Mel-frequency cepstral coefficients

### GPT Evaluation
- **Repetition Rate**: Tỷ lệ lặp từ (0-1)
- **Vocabulary Score**: Độ phong phú từ vựng (0-10)
- **Context Relevance**: Sự phù hợp ngữ cảnh (0-10)
- **Analysis**: Phân tích chi tiết bằng tiếng Việt

## 🎯 API Endpoints

### POST /assess-cognitive
```json
{
  "audio": "audio_file.webm",
  "transcript": "nội dung nói",
  "question": "câu hỏi",
  "user_id": "email@example.com",
  "age": "25",
  "gender": "male"
}
```

### Response
```json
{
  "success": true,
  "transcript": "nội dung",
  "mmse_prediction": {
    "predicted_mmse": 28.5,
    "severity": "Bình thường",
    "description": "Không có suy giảm",
    "confidence": 0.8
  },
  "audio_features": {
    "duration": 15.2,
    "pitch_mean": 180.5,
    "energy_mean": 0.15
  },
  "gpt_evaluation": {
    "repetition_rate": 0.1,
    "vocabulary_score": 8.5,
    "context_relevance": 9.0,
    "analysis": "Câu trả lời rõ ràng..."
  }
}
```

## 🚀 Tính năng nâng cao

### Auto-dependency Installation
- Backend tự động cài đặt packages thiếu
- Hỗ trợ librosa, scikit-learn, matplotlib, seaborn
- Fallback cho các packages không thể cài đặt

### Error Handling
- Graceful fallback khi audio extraction thất bại
- Default audio features khi có lỗi
- Detailed logging cho debugging

### Performance Optimization
- Audio processing với librosa
- Feature scaling với StandardScaler
- Model caching và reuse

## 🔍 Debugging

### Console Logs
- Audio blob status
- API request/response
- Feature extraction progress
- Model prediction details

### Health Check
- `/health`: Kiểm tra trạng thái server
- `/system-status`: Thông tin chi tiết components
- `/api-info`: Thông tin API và capabilities

## 📝 Ghi chú

- **Audio Format**: Hỗ trợ WEBM, WAV, MP3
- **Language**: Vietnamese (tiếng Việt)
- **Model**: Ridge Regression với 15 features
- **Features**: 9+ audio features + 6 demographic features
- **Backend**: Flask API với auto-dependency management
- **Frontend**: Next.js với TypeScript và Tailwind CSS

## 🆘 Troubleshooting

### Audio không được xử lý
- Kiểm tra backend có chạy không
- Xem console logs cho lỗi
- Kiểm tra audio file format

### Model không dự đoán
- Kiểm tra dx-mmse.csv có tồn tại không
- Xem backend logs cho lỗi model
- Kiểm tra feature names matching

### GPT evaluation thất bại
- Kiểm tra OPENAI_API_KEY
- Xem network tab cho API calls
- Kiểm tra prompt format
