# 🚀 Hướng Dẫn Khởi Động Dual Model System

Hướng dẫn khởi động cả **Cognitive Assessment ML (cũ)** và **MMSE Assessment v2.0 (mới)** cùng lúc.

## 📋 Tổng Quan

Hệ thống dual model cho phép bạn:
- ✅ Chạy **2 models song song** mà không xung đột
- ✅ Sử dụng **cả API cũ và mới** cùng lúc
- ✅ **Backward compatible** với code cũ
- ✅ **Easy switching** giữa các models

## 🔧 Chuẩn Bị Environment

### Option 1: Sử dụng .ok Environment (Khuyến nghị)

```bash
# 1. Activate environment cũ
cd backend
.ok\Scripts\activate

# 2. Cài thêm dependencies cho MMSE v2.0
pip install python-Levenshtein sentence-transformers fpdf2 cryptography

# 3. Copy files cần thiết
copy ..\release_v1\questions.json .
copy ..\release_v1\scoring_engine.py services\
copy ..\release_v1\feature_extraction.py services\
copy ..\release_v1\encryption.py services\
copy ..\release_v1\mmse_assessment_service.py services\
```

### Option 2: Tạo Environment Mới

```bash
# 1. Tạo environment mới
python -m venv dual_env
dual_env\Scripts\activate

# 2. Install tất cả dependencies
pip install -r backend/requirements.txt
pip install python-Levenshtein sentence-transformers fpdf2 cryptography

# 3. Copy files
copy release_v1\questions.json backend\
copy release_v1\*.py backend\services\
```

## 🚀 Khởi Động Hệ Thống

### Cách 1: Script Tự Động (Easiest)

```bash
# Chạy script setup tự động
run_dual_model_system.bat
```

Script sẽ tự động:
- ✅ Setup environment
- ✅ Install dependencies
- ✅ Copy files cần thiết
- ✅ Test imports
- ✅ Khởi động backend + frontend

### Cách 2: Manual Setup

```bash
# Terminal 1: Backend
cd backend
.ok\Scripts\activate
python start_dual_model.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Cách 3: Sử dụng Python Script

```bash
# Chạy test script để khởi động
python test_dual_model.py
```

## 🧪 Test Hệ Thống

Sau khi khởi động, test các endpoints:

### Test Backend Health
```bash
curl http://localhost:5001/api/health
```

### Test Old Model
```bash
# Health check
curl http://localhost:5001/api/health

# Test với audio file
curl -X POST http://localhost:5001/api/assess \
  -F "audio=@test_audio.wav" \
  -F "sessionId=test_001"
```

### Test MMSE v2.0
```bash
# Model info
curl http://localhost:5001/api/mmse/model-info

# Questions
curl http://localhost:5001/api/mmse/questions

# Test assessment
curl -X POST http://localhost:5001/api/mmse/assess \
  -F "audio=@test_audio.wav" \
  -F "session_id=test_mmse_001" \
  -F "patient_info={\"name\":\"Test Patient\",\"age\":65,\"gender\":\"female\"}"
```

### Test Frontend
- 🌐 **Main App**: http://localhost:3000
- 🧠 **MMSE v2.0**: http://localhost:3000/mmse-v2

## 📊 API Endpoints So Sánh

| Feature | Old Model | MMSE v2.0 |
|---------|-----------|-----------|
| **Assessment** | `POST /api/assess` | `POST /api/mmse/assess` |
| **Transcription** | `POST /api/transcribe` | `POST /api/mmse/transcribe` |
| **Model Info** | N/A | `GET /api/mmse/model-info` |
| **Questions** | N/A | `GET /api/mmse/questions` |
| **Features** | `POST /api/features` | Built-in |
| **Health** | `GET /api/health` | `GET /api/health` |

## 🔄 Switching Between Models

### Frontend Switching
```typescript
// Old model
const response = await fetch('/api/assess', {
  method: 'POST',
  body: formData
});

// MMSE v2.0
const response = await fetch('/api/mmse/assess', {
  method: 'POST',
  body: formData
});
```

### Backend Switching
```python
# Old model
from cognitive_assessment_ml import EnhancedMultimodalCognitiveModel
old_model = EnhancedMultimodalCognitiveModel()

# MMSE v2.0
from services.mmse_assessment_service import get_mmse_service
mmse_service = get_mmse_service()
```

## 🐛 Troubleshooting

### 1. Import Errors
```bash
# Nếu lỗi sentence-transformers
pip install sentence-transformers

# Nếu lỗi cryptography
pip install cryptography

# Nếu lỗi Levenshtein
pip install python-Levenshtein
```

### 2. Path Errors
```bash
# Nếu không tìm thấy questions.json
copy release_v1\questions.json backend\

# Nếu không tìm thấy services
copy release_v1\*.py backend\services\
```

### 3. Port Conflicts
```bash
# Nếu port 5001 bị chiếm
# Thay đổi port trong app.py
app.run(port=5002)

# Hoặc kill process
netstat -ano | findstr :5001
taskkill /PID <PID> /F
```

### 4. Environment Issues
```bash
# Nếu .ok environment không hoạt động
# Tạo environment mới
python -m venv new_env
new_env\Scripts\activate
pip install -r requirements.txt
```

## 📈 Performance & Resources

### Memory Usage
- **Old Model**: ~2-3GB RAM
- **MMSE v2.0**: ~3-4GB RAM
- **Combined**: ~4-5GB RAM

### Disk Space
- **Models**: ~5-10GB (Whisper + ML models)
- **Dependencies**: ~2GB (Python packages)

### GPU Support
```bash
# Enable GPU (nếu có)
export CUDA_VISIBLE_DEVICES=0
# hoặc
set CUDA_VISIBLE_DEVICES=0
```

## 🔒 Security & Privacy

### Data Encryption
```python
# MMSE v2.0 tự động encrypt audio
from encryption import AudioEncryption
encryptor = AudioEncryption()
encryptor.encrypt_file('input.wav', 'output.enc')
```

### API Security
```python
# CORS settings trong app.py
from flask_cors import CORS
CORS(app, origins=['http://localhost:3000'])
```

## 📝 Logs & Monitoring

### Backend Logs
```bash
# Xem logs real-time
tail -f backend/backend.log

# Windows
Get-Content backend\backend.log -Wait
```

### Frontend Logs
```javascript
// Browser console
console.log('API Response:', response);

// Network tab để xem API calls
```

## 🎯 Best Practices

### Development
1. ✅ **Use virtual environment** để tránh conflicts
2. ✅ **Test both models** trước khi deploy
3. ✅ **Monitor resource usage** (RAM, CPU)
4. ✅ **Backup models** trước khi update

### Production
1. ✅ **Use production server** (gunicorn/uwsgi)
2. ✅ **Enable HTTPS** cho security
3. ✅ **Monitor logs** thường xuyên
4. ✅ **Backup data** và models

### Performance
1. ✅ **Use GPU** nếu có thể
2. ✅ **Batch processing** cho multiple files
3. ✅ **Cache models** trong memory
4. ✅ **Optimize audio** trước khi process

## 🚨 Emergency Stop

```bash
# Stop backend
Ctrl+C (trong terminal backend)

# Stop frontend
Ctrl+C (trong terminal frontend)

# Kill processes
taskkill /IM python.exe /F
taskkill /IM node.exe /F
```

## 📞 Support

Nếu gặp vấn đề:
1. 📋 **Check logs**: `backend/backend.log`
2. 🧪 **Run tests**: `python test_dual_model.py`
3. 🔄 **Restart services** theo hướng dẫn
4. 📧 **Contact support** nếu cần

---

**🎉 Chúc mừng! Bạn đã có hệ thống dual model hoàn chỉnh!**

Hệ thống này cho phép bạn sử dụng cả model cũ và mới một cách linh hoạt và hiệu quả. 🚀
