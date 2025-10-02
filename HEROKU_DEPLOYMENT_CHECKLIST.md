# 🚀 Heroku Deployment Checklist cho Vietnamese Cognitive Assessment

## ✅ Chuẩn bị Files (Đã hoàn thành)

### 1. **Environment Variables**
- [x] `.env.example` - Template cho environment variables
- [x] `backend/config.env` - Development configuration
- [x] Đã xác định tất cả biến environment cần thiết

### 2. **Dependencies & Requirements**
- [x] `requirements.txt` - Python dependencies với pinned versions
- [x] `runtime.txt` - Python 3.11.6
- [x] Đã scan toàn bộ import statements trong project

### 3. **Heroku Configuration**
- [x] `Procfile` - Web dyno với gunicorn
- [x] `backend/run.py` - Production-ready với gunicorn support
- [x] `.gitignore` - Updated cho Heroku deployment

### 4. **Health Check**
- [x] `/api/health` endpoint có sẵn
- [x] `/api/status` endpoint có sẵn
- [x] `/api/config` endpoint có sẵn

## 🔧 Heroku Setup Steps

### Bước 1: Tạo Heroku App
```bash
# Cài đặt Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login vào Heroku
heroku login

# Tạo app mới
heroku create your-cognitive-assessment-app

# Hoặc để Heroku tự tạo tên
heroku create

# Set remote
git remote add heroku https://git.heroku.com/your-app-name.git
```

### Bước 2: Configure Environment Variables
```bash
# Database (Heroku sẽ tự set DATABASE_URL khi add PostgreSQL)
heroku addons:create heroku-postgresql:mini

# Flask Configuration
heroku config:set FLASK_ENV=production
heroku config:set FLASK_DEBUG=false
heroku config:set DEBUG=false
heroku config:set HOST=0.0.0.0

# API Keys
heroku config:set OPENAI_API_KEY=sk-proj-your_openai_api_key_here
heroku config:set GEMINI_API_KEY=your_gemini_api_key_here
heroku config:set MINIMAX_API_KEY=your_minimax_api_key_here

# Auth (Clerk)
heroku config:set NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_key
heroku config:set CLERK_SECRET_KEY=sk_test_your_key

# Email
heroku config:set EMAIL_USER=your_email@example.com
heroku config:set EMAIL_PASS=your_app_password

# NextAuth
heroku config:set NEXTAUTH_URL=https://your-app-name.herokuapp.com
heroku config:set NEXT_PUBLIC_APP_URL=https://your-app-name.herokuapp.com

# Model Configuration
heroku config:set MODEL_PATH=./models
heroku config:set UPLOAD_PATH=./uploads
heroku config:set STORAGE_PATH=./storage

# Transcription Settings
heroku config:set ENABLE_PAID_TRANSCRIPTION=true
heroku config:set TRANSCRIPTION_BUDGET_LIMIT=5.00

# Vietnamese ASR Model
heroku config:set VI_ASR_MODEL=nguyenvulebinh/wav2vec2-large-vietnamese-250h

# Node.js Environment
heroku config:set NODE_ENV=production

# Logging
heroku config:set LOG_LEVEL=INFO
heroku config:set LOG_FILE=backend.log
```

### Bước 3: Database Setup (Optional)
```bash
# Nếu cần database
heroku addons:create heroku-postgresql:mini

# Kiểm tra DATABASE_URL
heroku config:get DATABASE_URL

# Connect to database nếu cần init tables
heroku pg:psql
```

### Bước 4: Deploy Application
```bash
# Add tất cả files vào git
git add .
git commit -m "Prepare for Heroku deployment"

# Deploy to Heroku
git push heroku master

# Hoặc nếu branch khác
git push heroku main:master
```

### Bước 5: Scale Web Dynos
```bash
# Scale web dyno
heroku ps:scale web=1

# Kiểm tra status
heroku ps
```

### Bước 6: Kiểm tra Logs
```bash
# Xem logs
heroku logs --tail

# Xem logs của web dyno
heroku logs --source app --tail
```

## 🔍 Testing & Verification

### Endpoints cần test sau khi deploy:
```bash
# Health check
curl https://your-app-name.herokuapp.com/api/health

# Status
curl https://your-app-name.herokuapp.com/api/status

# Config
curl https://your-app-name.herokuapp.com/api/config
```

### Expected responses:
- `/api/health` - Should return `{"status": "healthy"}`
- `/api/status` - Should return model loading status
- `/api/config` - Should return configuration info

## ⚡ Performance Optimization

### Buildpack Configuration
```bash
# Set Python buildpack
heroku buildpacks:set heroku/python

# Hoặc multiple buildpacks nếu cần
heroku buildpacks:add heroku/nodejs
heroku buildpacks:add heroku/python
```

### Memory & CPU Tuning
```bash
# Scale to hobby dyno (nếu cần)
heroku ps:scale web=1:hobby

# Professional dyno (có nhiều memory hơn)
heroku ps:scale web=1:standard-1x
```

## 🚨 Troubleshooting

### Lỗi thường gặp:

1. **Module import errors**
   - Kiểm tra `requirements.txt`
   - Đảm bảo tất cả packages có version

2. **Port binding errors**
   - Đảm bảo app sử dụng `os.environ.get('PORT')`
   - Procfile đúng format

3. **Memory errors**
   - Giảm model size
   - Optimize memory usage
   - Scale lên dyno lớn hơn

4. **Timeout errors**
   - Tăng timeout trong Procfile
   - Optimize initialization time

### Debug commands:
```bash
# Check config
heroku config

# Check dyno status
heroku ps

# Run one-off commands
heroku run python backend/run.py

# Restart app
heroku restart

# Check buildpack
heroku buildpacks
```

## 📋 Final Verification Checklist

### Pre-deployment:
- [ ] Tất cả environment variables đã set
- [ ] Database addon đã add (nếu cần)
- [ ] Code đã commit và push lên git
- [ ] Dependencies trong requirements.txt đã test

### Post-deployment:
- [ ] App startup thành công (`heroku logs`)
- [ ] Health endpoint response OK
- [ ] Status endpoint response OK
- [ ] Config endpoint response OK
- [ ] Core functionality hoạt động

### Security:
- [ ] Không có API keys trong code
- [ ] Environment variables secure
- [ ] Debug mode = false
- [ ] CORS configuration đúng

## 🎯 Next Steps sau khi deploy thành công:

1. **Monitor Performance**
   ```bash
   heroku logs --tail
   heroku ps
   ```

2. **Set up custom domain** (Optional)
   ```bash
   heroku domains:add your-domain.com
   ```

3. **Set up monitoring** (Optional)
   ```bash
   heroku addons:create newrelic:wayne
   ```

4. **Backup strategy** (Database)
   ```bash
   heroku pg:backups:capture
   heroku pg:backups:schedules
   ```

## 📞 Support Resources

- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Heroku Config Vars](https://devcenter.heroku.com/articles/config-vars)
- [Heroku Logs](https://devcenter.heroku.com/articles/logging)
- [Heroku PostgreSQL](https://devcenter.heroku.com/articles/heroku-postgresql)

---

**🎉 Deployment Summary:**
- ✅ Flask app ready for production
- ✅ Gunicorn configured
- ✅ Environment variables templated
- ✅ Dependencies pinned
- ✅ Health checks implemented
- ✅ .gitignore optimized
- ✅ Runtime specified

**🚀 Ready to deploy!**
