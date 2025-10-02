# 🚀 Hướng dẫn Deploy MMSE System lên Vercel

## ✅ Chuẩn bị hoàn tất

Hệ thống đã được chuẩn bị sẵn sàng cho deployment production với:

- ✅ **Vercel configuration** (`vercel.json`, `.vercelignore`)
- ✅ **Blob Storage integration** (API routes cho upload audio)
- ✅ **Production-ready endpoints** 
- ✅ **Migration scripts** cho audio files
- ✅ **Environment variables** template
- ✅ **Database ready** (Drizzle ORM + PostgreSQL)

## 🎯 Các bước deploy nhanh

### Bước 1: Install Vercel CLI

```bash
npm i -g vercel
```

### Bước 2: Login và link project

```bash
# Login to Vercel
vercel login

# Link project (trong thư mục frontend)
cd frontend
vercel
```

### Bước 3: Setup Vercel Postgres Database

1. Vào [Vercel Dashboard](https://vercel.com/dashboard)
2. Chọn project vừa tạo
3. Vào tab **Storage** 
4. Tạo **Postgres Database**
5. Copy connection strings

### Bước 4: Setup Vercel Blob Storage

1. Trong project dashboard
2. Vào tab **Storage**
3. Tạo **Blob Storage**
4. Copy `BLOB_READ_WRITE_TOKEN`

### Bước 5: Cấu hình Environment Variables

Có 2 cách:

#### Cách 1: Vercel CLI
```bash
vercel env add DATABASE_URL production
vercel env add NEON_DATABASE_URL production
vercel env add BLOB_READ_WRITE_TOKEN production
vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY production
vercel env add CLERK_SECRET_KEY production
vercel env add OPENAI_API_KEY production
vercel env add GOOGLE_API_KEY production
```

#### Cách 2: Vercel Dashboard
- Vào **Project Settings** → **Environment Variables**
- Add từng variable theo template trong `env.production.example`

### Bước 6: Deploy

```bash
# Option 1: Deploy với build tự động
vercel --prod

# Option 2: Deploy với preparation scripts
npm run deploy:full
```

### Bước 7: Migrate Database Schema

```bash
# Pull environment variables
vercel env pull .env.production

# Run migrations
npm run drizzle:migrate
```

### Bước 8: Migrate Audio Files (Optional)

Nếu có audio files local cần migrate:

```bash
# Make sure BLOB_READ_WRITE_TOKEN is set in .env.local
npm run migrate:audio
```

## 🔧 Environment Variables cần thiết

### Core Variables

```bash
# Database (from Vercel Postgres)
DATABASE_URL="postgresql://..."
NEON_DATABASE_URL="postgresql://..."

# Authentication (from Clerk Dashboard)
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY="pk_live_..."
CLERK_SECRET_KEY="sk_live_..."

# File Storage (from Vercel Blob)
BLOB_READ_WRITE_TOKEN="vercel_blob_rw_..."

# AI Services
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIzaSy..."

# Application
NODE_ENV="production"
NEXT_PUBLIC_APP_URL="https://your-app.vercel.app"
```

### Optional Variables

```bash
# Python backend integration
NEXT_PUBLIC_PYTHON_BACKEND_URL="https://your-python-backend.vercel.app"

# Email service
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="your-email@gmail.com"
SMTP_PASS="your-app-password"
```

## 🧪 Testing Deployment

### 1. Health Check

```bash
curl https://your-app.vercel.app/api/health
```

### 2. Database Connection

Test các API endpoints:
- `/api/database/user`
- `/api/profile/user`
- `/api/training-samples`

### 3. Authentication

Test Clerk authentication flow

### 4. File Upload

Test audio upload:
- `/api/save-recording-blob`
- `/api/upload-audio-blob`

## 🔄 Update và Redeploy

```bash
# Pull latest changes
git pull origin main

# Deploy updates
vercel --prod

# Or với preparation
npm run deploy:full
```

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check connection string format
# Make sure it includes ?pgbouncer=true for serverless

# Test connection
npm run drizzle:studio
```

### Blob Storage Issues

```bash
# Verify token
echo $BLOB_READ_WRITE_TOKEN

# Test upload locally
npm run migrate:audio
```

### Build Errors

```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Try build locally
npm run build
```

### API Timeout Issues

Check `vercel.json` maxDuration settings:
- Audio processing: 60s
- AI analysis: 300s
- Regular APIs: 30s

## 📊 Monitoring

### Vercel Dashboard

- **Functions** tab: Check API performance
- **Logs** tab: Debug errors
- **Analytics** tab: Usage metrics

### Custom Monitoring

Add health check endpoint:

```typescript
// app/api/health/route.ts
export async function GET() {
  return Response.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version
  })
}
```

## 🌐 Custom Domain (Optional)

```bash
# Add domain
vercel domains add yourdomain.com

# Update environment variable
vercel env add NEXT_PUBLIC_APP_URL "https://yourdomain.com" production
```

## 🚀 Production Checklist

- [ ] ✅ Vercel project created
- [ ] ✅ Database connection working
- [ ] ✅ Blob storage configured
- [ ] ✅ All environment variables set
- [ ] ✅ Authentication working
- [ ] ✅ Audio upload/download working
- [ ] ✅ MMSE assessment flow working
- [ ] ✅ Python backend integration (if used)
- [ ] ✅ Email service working (if used)
- [ ] ✅ SSL certificate active
- [ ] ✅ Custom domain configured (if used)

## 📞 Support

Nếu gặp vấn đề trong quá trình deploy:

1. Check **Vercel Logs**: `vercel logs your-app`
2. Check **Function Logs** trong Vercel Dashboard
3. Test locally: `npm run dev`
4. Verify environment variables: `vercel env ls`

---

**🎉 Deployment hoàn tất!** 

Hệ thống MMSE của bạn đã sẵn sàng phục vụ trên production với đầy đủ tính năng:
- 🧠 MMSE Assessment với AI analysis
- 🎵 Audio recording & processing
- 👥 User management với Clerk
- 📊 Data analytics & reporting
- 🔐 Production-grade security
