# 🚀 COGNITIVE ASSESSMENT SYSTEM - DEPLOYMENT GUIDE

**For Beginners - Step-by-Step Deployment to Production**

*Generated for Cognitive Assessment System (cavang) - Next.js + Flask + PostgreSQL*

---

## 📋 QUICK OVERVIEW

**Your App:** Vietnamese Cognitive Assessment platform using MMSE with AI transcription
**Tech Stack:** Next.js 15 + Flask + PostgreSQL + Vercel Blob
**Estimated Time:** 3-4 hours
**Monthly Cost:** $5
**Services:** Vercel + Railway + Neon + Clerk

---

## 🎯 PHẦN 1: CHUẨN BỊ (Prerequisites)

### 1.1. Kiểm tra môi trường local

Trước khi deploy, đảm bảo project chạy được trên máy local:

**Frontend Test:**
```bash
cd frontend
npm install
npm run dev
```
→ Mở http://localhost:3000 → Thấy trang chủ Cognitive Assessment

**Backend Test:**
```bash
cd backend
pip install -r requirements.txt
python run.py
```
→ Mở http://localhost:8000/api/health → Thấy JSON response

**Database Test:**
- Cần PostgreSQL local hoặc sử dụng Neon cloud database
- Import schema từ `frontend/db/schema.ts`

### 1.2. Công cụ cần cài đặt

- [ ] **Git:** `git --version` (tải từ git-scm.com nếu chưa có)
- [ ] **Node.js:** `node --version` (cần ≥18.17.0)
- [ ] **npm:** `npm --version` (có sẵn với Node.js)
- [ ] **Python:** `python --version` (nên 3.11+)
- [ ] **VS Code:** Editor để chỉnh sửa code

### 1.3. Tài khoản cần đăng ký (MIỄN PHÍ)

- [ ] **GitHub:** https://github.com/signup (để lưu code)
- [ ] **Vercel:** https://vercel.com/signup (deploy frontend, dùng GitHub login)
- [ ] **Railway:** https://railway.app/login (deploy backend, dùng GitHub login)
- [ ] **Neon:** https://neon.tech (database PostgreSQL, dùng GitHub login)
- [ ] **Clerk:** https://clerk.com (authentication, đăng ký qua Vercel)

### 1.4. Thông tin cần chuẩn bị

- [ ] Repository URL trên GitHub (sẽ tạo ở bước sau)
- [ ] Custom domain (tùy chọn, VD: cognitive-assessment.com)
- [ ] API keys: OpenAI, Gemini (cho AI features)

---

## 📁 PHẦN 2: CẤU TRÚC PROJECT

Project của bạn có cấu trúc đặc biệt:

```
cognitive-assessment-system/
├── frontend/                 ← Next.js 15 App (React/TypeScript)
│   ├── app/                  ← Next.js App Router
│   ├── components/           ← React components
│   ├── db/                   ← Drizzle ORM schema
│   ├── drizzle/              ← Database migrations
│   ├── lib/                  ← Utilities & configs
│   ├── package.json          ← Frontend dependencies
│   └── .env.example          ← Environment variables template
│
├── backend/                  ← Flask API (Python/ML)
│   ├── app.py                ← Main Flask app (49K+ lines)
│   ├── run.py                ← Entry point
│   ├── config/               ← Configuration files
│   ├── services/             ← ML pipeline services
│   ├── requirements.txt      ← Python dependencies
│   ├── gunicorn.conf.py      ← Production config
│   └── models/               ← ML models
│
├── .gitignore                ← Đã configure tốt
├── README.md                 ← Project documentation
└── [various config files]
```

**Quan trọng:** File `.env` KHÔNG được push lên GitHub!

---

## 🔼 PHẦN 3: PUSH CODE LÊN GITHUB

### 3.1. Kiểm tra .gitignore

File `.gitignore` đã được setup tốt, đảm bảo các file nhạy cảm không bị push:

```gitignore
# Environment variables (QUAN TRỌNG)
.env
.env.local
.env*.local

# Dependencies
node_modules/
__pycache__/

# Build outputs
.next/
dist/

# Logs
*.log
backend.log
transcriber.log

# Model files & data
*.pkl
*.wav
*.mp3
models/
data/
```

### 3.2. Khởi tạo Git (nếu chưa có)

```bash
# Kiểm tra Git
git status

# Nếu thấy "not a git repository", khởi tạo:
git init
```

### 3.3. Add & commit files

```bash
# Add tất cả files
git add .

# Kiểm tra không có file .env
git status

# Commit
git commit -m "Initial commit - Cognitive Assessment System"
```

### 3.4. Tạo repository trên GitHub

1. Vào https://github.com/new
2. **Repository name:** `cognitive-assessment-system` hoặc tên bạn thích
3. **Description:** `Vietnamese Cognitive Assessment platform using MMSE with AI`
4. **Chọn:** Private (nếu không muốn public)
5. **KHÔNG tick:** Add README, .gitignore, license
6. Click **"Create repository"**

### 3.5. Push lên GitHub

Copy commands từ GitHub:

```bash
git remote add origin https://github.com/your-username/cognitive-assessment-system.git
git branch -M main
git push -u origin main
```

**Nếu hỏi username/password:**
- Username: GitHub username
- Password: **Personal Access Token** (không phải password thường)

**Tạo Personal Access Token:**
1. GitHub → Settings → Developer settings → Personal access tokens
2. "Generate new token (classic)"
3. Chọn scopes: `repo` (full control of private repositories)
4. Copy token → paste vào Terminal

**✅ Kiểm tra:** Vào GitHub repository → thấy tất cả files đã lên!

---

## 🗄️ PHẦN 4: SETUP DATABASE (Neon PostgreSQL)

### 4.1. Tạo Neon project

1. Vào https://neon.tech → Login với GitHub
2. Click **"Create a project"**
3. **Project name:** `cognitive-assessment-db`
4. **Postgres version:** 15
5. **Region:** Chọn US East (Ohio) hoặc gần Railway server
6. Click **"Create project"**

### 4.2. Lấy connection string

Sau khi tạo, click **"Connection string"** → Copy toàn bộ string:

```
postgresql://username:password@ep-xyz.us-east-2.aws.neon.tech/dbname?sslmode=require
```

**⚠️ LƯU LẠI STRING NÀY!** Cần dùng cho backend.

### 4.3. Tạo database schema

Neon có UI để chạy SQL, nhưng dùng Drizzle từ frontend:

```bash
cd frontend

# Cài đặt dependencies nếu chưa có
npm install

# Tạo migration files
npm run drizzle:generate

# Chạy migration lên database
DATABASE_URL="your-neon-connection-string" npm run drizzle:migrate
```

**Hoặc chạy SQL trực tiếp trong Neon Console:**

```sql
-- Tạo tables theo schema trong frontend/db/schema.ts
-- Copy SQL từ drizzle/0000_*.sql files
```

### 4.4. Test connection

```bash
# Test với Python
python -c "import psycopg2; conn = psycopg2.connect('your-connection-string'); print('✅ Connected!')"
```

---

## 🚂 PHẦN 5: DEPLOY BACKEND (Railway)

### 5.1. Tạo Railway project

1. Vào https://railway.app → Login với GitHub
2. Click **"New Project"**
3. Chọn **"Deploy from GitHub repo"**
4. Authorize Railway truy cập GitHub repos
5. Chọn repository: `cognitive-assessment-system`
6. Click **"Deploy from GitHub"**

### 5.2. Configure backend service

Railway sẽ tự detect Python project và hỏi:

**Root Directory:** `/backend` (vì code backend ở folder backend/)

### 5.3. Set Environment Variables

Click vào service vừa tạo → Tab **"Variables"** → **"New Variable"**

**BẮT BUỘC (Critical):**

```bash
# Database (từ Neon)
DATABASE_URL=postgresql://username:password@ep-xyz.us-east-2.aws.neon.tech/dbname?sslmode=require

# Flask configuration
SECRET_KEY=your-32-character-random-secret-key-here
PORT=8000
HOST=0.0.0.0
FLASK_ENV=production
DEBUG=false

# AI Services (cần cho chức năng chính)
OPENAI_API_KEY=sk-proj-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here

# File storage (Vercel Blob - sẽ setup sau)
BLOB_READ_WRITE_TOKEN=vercel_blob_your_token_here
```

**Cách tạo SECRET_KEY:**
```bash
# Python command
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**TÙY CHỌN (Optional nhưng nên có):**

```bash
# Cache settings
CACHE_DIR=/tmp/cognitive_cache
CACHE_TTL=3600
AUDIO_CACHE_SIZE=100
TEXT_CACHE_SIZE=500

# Feature flags
ENABLE_PAID_TRANSCRIPTION=true
TRANSCRIPTION_BUDGET_LIMIT=5.00
ENABLE_ADVANCED_ANALYTICS=true
ENABLE_MODEL_EXPLAINABILITY=true

# Email (nếu có)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
EMAIL_USER=noreply@yourapp.com

# Monitoring
LOG_LEVEL=WARNING
```

### 5.4. Configure build settings

Railway tự động detect Python và requirements.txt.

**Build Command:** `pip install -r requirements.txt` (tự động)

**Start Command:** Railway tự detect là Flask app và dùng:
```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

Nếu cần tùy chỉnh, vào Settings → Build & Start Commands.

### 5.5. Deploy & test

Click **"Deploy"** → Chờ 5-10 phút...

**Xem logs:** Click **"View Logs"**

**Test backend:** Sau khi deploy thành công, copy Railway URL:
```
https://cognitive-assessment-production-xyz.up.railway.app
```

Test: `https://your-railway-url.up.railway.app/api/health`

**✅ Thành công nếu thấy JSON response!**

---

## 🎨 PHẦN 6: SETUP VERCEL BLOB (File Storage)

Trước khi deploy frontend, cần setup file storage cho audio recordings.

### 6.1. Tạo Vercel Blob store

1. Vào https://vercel.com → Dashboard
2. Chọn project hoặc tạo mới
3. Tab **"Storage"** → **"Create Database"** → **"Blob"**
4. **Name:** `cognitive-assessment-blob`
5. Click **"Create"**

### 6.2. Lấy token

Sau khi tạo, copy **BLOB_READ_WRITE_TOKEN**:
```
vercel_blob_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 6.3. Update backend environment

Vào Railway → Backend service → Variables → Thêm:
```bash
BLOB_READ_WRITE_TOKEN=vercel_blob_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Restart service để áp dụng.

---

## 🎨 PHẦN 7: DEPLOY FRONTEND (Vercel)

### 7.1. Import project

1. Vào https://vercel.com → Dashboard
2. Click **"Add New..."** → **"Project"**
3. Chọn repository: `cognitive-assessment-system`
4. Click **"Import"**

### 7.2. Configure project

**Root Directory:** `frontend` (vì code Next.js ở folder frontend/)

Vercel tự detect Next.js 15.

**Build Settings:** Tự động đúng:
- Build Command: `npm run build`
- Output Directory: `.next` (Next.js tự handle)

### 7.3. Environment Variables

Click **"Environment Variables"** → Add từng biến:

**BẮT BUỘC:**

```bash
# Backend API URL (từ Railway)
NEXT_PUBLIC_PYTHON_BACKEND_URL=https://cognitive-assessment-production-xyz.up.railway.app

# Clerk Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_clerk_key_here
CLERK_SECRET_KEY=sk_test_your_clerk_secret_key_here

# Database (cho Drizzle migrations)
DATABASE_URL=postgresql://username:password@ep-xyz.us-east-2.aws.neon.tech/dbname?sslmode=require
NEON_DATABASE_URL=postgresql://username:password@ep-xyz.us-east-2.aws.neon.tech/dbname?sslmode=require

# Vercel Blob (cho file uploads)
BLOB_READ_WRITE_TOKEN=vercel_blob_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**TÙY CHỌN:**

```bash
# AI Services (nếu frontend dùng trực tiếp)
OPENAI_API_KEY=sk-proj-your-key
GEMINI_API_KEY=your-key

# Analytics (tùy chọn)
NEXT_PUBLIC_GOOGLE_ANALYTICS_ID=G-XXXXXXXXXX
```

### 7.4. Deploy!

Click **"Deploy"** → Chờ 3-5 phút...

**Xem logs:** Real-time build logs sẽ hiện.

### 7.5. Lấy frontend URL

Sau khi deploy thành công, Vercel cho URL:
```
https://cognitive-assessment-xyz.vercel.app
```

**⚠️ LƯU LẠI URL NÀY!** Cần update CORS ở backend.

### 7.6. Update backend CORS

Vào Railway → Backend service → Variables → Update:

```bash
# Thay thế
ALLOWED_ORIGINS=https://cognitive-assessment-xyz.vercel.app
```

Hoặc trong code `backend/app.py` (dòng ~400):

```python
# Thay đổi từ
CORS(app)

# Thành
CORS(app, origins=["https://cognitive-assessment-xyz.vercel.app"])
```

Commit & push để Railway auto-redeploy.

### 7.7. Test frontend

Mở `https://cognitive-assessment-xyz.vercel.app`

**✅ Kiểm tra:**
- [ ] Trang load được
- [ ] Không có console errors (F12 → Console)
- [ ] API calls tới backend thành công (F12 → Network)
- [ ] Authentication (Clerk) hoạt động

---

## 🧪 PHẦN 8: TESTING & TROUBLESHOOTING

### 8.1. End-to-End Testing

**Test từng feature:**

1. **User Registration/Login:**
   - Đăng ký tài khoản mới
   - Login/logout

2. **Cognitive Assessment:**
   - Bắt đầu assessment
   - Record audio cho câu hỏi
   - Submit và xem kết quả

3. **MMSE Assessment:**
   - Start MMSE session
   - Answer questions
   - Complete assessment
   - View results

4. **File Upload:**
   - Upload audio files
   - Check Vercel Blob storage

### 8.2. Common Issues & Fixes

#### **Issue: "API calls failing with CORS error"**

**Symptoms:** Frontend shows network errors, console có:
```
Access to fetch at 'https://backend.railway.app' blocked by CORS policy
```

**Fix:**
1. Check Railway environment variables
2. Ensure `ALLOWED_ORIGINS` = Vercel URL
3. Restart Railway service
4. Check backend logs

#### **Issue: "Database connection failed"**

**Symptoms:** Backend logs show connection errors

**Fix:**
1. Verify DATABASE_URL in Railway
2. Test connection from local:
   ```bash
   psql "your-database-url"
   ```
3. Check Neon dashboard - database might be sleeping

#### **Issue: "Build failed on Vercel"**

**Symptoms:** Vercel deployment fails

**Common causes:**
- Missing environment variables
- Database migration failed
- Node.js version mismatch

**Fix:**
1. Check Vercel build logs
2. Test build locally: `cd frontend && npm run build`
3. Fix errors, commit, push

#### **Issue: "ML models not loading"**

**Symptoms:** Assessment fails with model errors

**Fix:**
1. Check backend logs for specific error
2. Ensure model files are in `backend/models/`
3. Verify PyTorch/TensorFlow installation
4. Check available RAM on Railway (upgrade if needed)

#### **Issue: "File uploads not working"**

**Symptoms:** Cannot upload audio files

**Fix:**
1. Verify BLOB_READ_WRITE_TOKEN in both Vercel and Railway
2. Check Vercel Blob dashboard
3. Ensure frontend has correct blob configuration

### 8.3. Performance Monitoring

**Check these metrics:**

- **Railway:** CPU, Memory usage
- **Vercel:** Response times, error rates
- **Neon:** Connection count, query performance
- **Frontend:** Lighthouse scores

---

## 🌐 PHẦN 9: CUSTOM DOMAIN (Optional)

### 9.1. Mua domain

**Nơi mua rẻ:**
- Namecheap (~$10/năm)
- Cloudflare Registrar (~$8/năm)
- Porkbun (~$9/năm)

### 9.2. Add domain vào Vercel

1. Vercel Dashboard → Project → Settings → Domains
2. Add domain: `cognitive-assessment.com`
3. Vercel show DNS records cần add

### 9.3. Configure DNS

Trong domain provider, add records:

```
Type: A
Name: @
Value: 76.76.21.21

Type: CNAME
Name: www
Value: cname.vercel-dns.com
```

### 9.4. Update backend CORS

Add custom domain:

```bash
ALLOWED_ORIGINS=https://cognitive-assessment-xyz.vercel.app,https://cognitive-assessment.com
```

---

## 🔄 PHẦN 10: CI/CD & MAINTENANCE

### 10.1. Auto-deployment

**Đã setup tự động:**
- Push code lên GitHub → Vercel rebuild frontend
- Push code → Railway rebuild backend

### 10.2. Database backups

**Neon tự động backup, nhưng nên:**

1. **Manual backup:**
   ```bash
   pg_dump "your-database-url" > backup_$(date +%Y%m%d).sql
   ```

2. **Schedule weekly backups**

### 10.3. Monitoring

**Setup free monitoring:**

1. **UptimeRobot:** Monitor cả frontend và backend URLs
2. **Railway Logs:** Check daily cho errors
3. **Vercel Analytics:** Monitor performance

### 10.4. Updates & Security

**Monthly maintenance:**

```bash
# Update dependencies
cd frontend && npm audit && npm audit fix
cd backend && pip list --outdated

# Security updates
npm audit --audit-level=high
pip install -U --dry-run  # Check updates
```

---

## 📊 PHẦN 11: COST OPTIMIZATION

### Current Costs
- **Vercel:** $0 (free tier: 100GB bandwidth)
- **Railway:** $5/month (starter plan)
- **Neon:** $0 (free tier: 0.5GB storage)
- **Clerk:** $0 (free tier: 10k users)
- **Vercel Blob:** $0 (free tier: 1GB storage)
**Total: $5/month**

### Scaling Considerations

**Nếu cần scale:**

1. **Railway:** Upgrade to $10/month plan (more RAM for ML)
2. **Neon:** $19/month for 2GB storage
3. **Vercel:** Remains free until very high traffic
4. **Clerk:** Free up to 10k users

---

## 🚨 EMERGENCY PROCEDURES

### Rollback Code
```bash
# Revert last commit
git revert HEAD
git push origin main
```

### Rollback on Platforms
- **Vercel:** Deployments → Select previous → "Promote to Production"
- **Railway:** Deployments → Select previous → "Rollback"

### Emergency Contacts
- Railway Discord: https://discord.gg/railway
- Vercel Discord: https://vercel.com/discord
- Clerk Support: https://clerk.com/support

---

## ✅ FINAL CHECKLIST

**Print và check từng item:**

### Pre-deployment ✅
- [x] Local development working
- [x] All dependencies installed
- [x] Environment variables documented
- [x] .gitignore configured
- [x] Code pushed to GitHub

### Infrastructure ✅
- [x] GitHub repository created
- [x] Neon database initialized
- [x] Railway backend deployed
- [x] Vercel frontend deployed
- [x] Clerk authentication setup

### Configuration ✅
- [x] Environment variables set
- [x] Database connected
- [x] CORS configured
- [x] File storage configured

### Testing ✅
- [x] Frontend loads without errors
- [x] API calls working
- [x] Authentication working
- [x] File uploads working
- [x] ML pipeline functional

### Production ✅
- [x] HTTPS enabled (auto)
- [x] Custom domain configured (optional)
- [x] Monitoring setup
- [x] Backup strategy defined

---

## 🎉 DEPLOYMENT COMPLETE!

**Your Cognitive Assessment System is live at:**
- **Frontend:** https://cognitive-assessment-xyz.vercel.app
- **Backend:** https://cognitive-assessment-production-abc.up.railway.app
- **Database:** [Neon dashboard]
- **File Storage:** [Vercel Blob dashboard]

**Next steps:**
1. Share với users
2. Collect feedback
3. Monitor performance
4. Add more features!

**Need help?**
- Check logs: Railway/Vercel dashboards
- Debug: Use browser DevTools (F12)
- Support: GitHub issues on your repo

---

## 📝 GHI CHÚ THÊM

### Project Specific Notes

1. **ML Pipeline:** Requires significant RAM for audio processing
2. **Audio Files:** All stored in Vercel Blob (not local)
3. **Database Schema:** Complex 12-table schema for assessments
4. **Authentication:** Clerk handles user management
5. **AI Features:** Requires OpenAI + Gemini API keys

### Performance Tips

1. **Caching:** Implement Redis for expensive operations
2. **CDN:** Vercel auto-serves static assets globally
3. **Database:** Use connection pooling for high traffic
4. **Monitoring:** Set up alerts for API failures

### Security Best Practices

1. **API Keys:** Rotate regularly
2. **Environment Variables:** Never commit secrets
3. **Rate Limiting:** Consider implementing
4. **Input Validation:** All user inputs validated
5. **HTTPS:** Always enforced

---

**🚀 HAPPY DEPLOYING! Your Cognitive Assessment System is ready to help users worldwide!**
