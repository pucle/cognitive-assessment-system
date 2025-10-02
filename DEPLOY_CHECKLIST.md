# ✅ MMSE System - Vercel Deployment Checklist

## 🚀 Quick Deploy (5 phút)

### 1. Pre-requisites
- [ ] Node.js 18+ installed
- [ ] Có account Vercel
- [ ] Có account Clerk (for authentication)

### 2. Install & Setup (1 phút)
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Go to frontend directory  
cd frontend
```

### 3. Create Vercel Services (2 phút)

**Database:**
1. Vào [Vercel Dashboard](https://vercel.com/dashboard)
2. Create new project → Import từ GitHub/GitLab hoặc deploy từ local
3. Vào Storage tab → Create **Postgres Database**
4. Copy: `DATABASE_URL` và `NEON_DATABASE_URL`

**Blob Storage:**
1. Cùng project → Storage tab → Create **Blob Storage**  
2. Copy: `BLOB_READ_WRITE_TOKEN`

### 4. Setup Environment Variables (1 phút)

**Required:**
```bash
vercel env add DATABASE_URL production
vercel env add NEON_DATABASE_URL production  
vercel env add BLOB_READ_WRITE_TOKEN production
vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY production
vercel env add CLERK_SECRET_KEY production
```

**Optional:**
```bash
vercel env add OPENAI_API_KEY production
vercel env add GOOGLE_API_KEY production
```

### 5. Deploy (1 phút)
```bash
# Auto deploy với script
.\scripts\deploy-to-vercel.ps1

# Hoặc manual
vercel --prod
```

## 🔧 Environment Variables Values

Copy từ các services:

| Variable | Source | Example |
|----------|--------|---------|
| `DATABASE_URL` | Vercel Postgres | `postgresql://user:pass@host/db?pgbouncer=true` |
| `NEON_DATABASE_URL` | Vercel Postgres | `postgresql://user:pass@host/db` |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blob | `vercel_blob_rw_xxxxx...` |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk Dashboard | `pk_live_xxxxx...` |
| `CLERK_SECRET_KEY` | Clerk Dashboard | `sk_live_xxxxx...` |
| `OPENAI_API_KEY` | OpenAI | `sk-xxxxx...` |
| `GOOGLE_API_KEY` | Google Cloud | `AIzaSyxxxxx...` |

## 🧪 Post-Deploy Testing

### 1. Basic Health Check
```bash
curl https://your-app.vercel.app/api/health
```

### 2. Authentication Test
- Visit your app URL
- Try sign up/login with Clerk

### 3. Core Features Test
- [ ] User registration/login works
- [ ] Audio recording works  
- [ ] MMSE assessment flow works
- [ ] Database connections work
- [ ] File upload/download works

## 🐛 Common Issues & Fixes

### Build Errors
```bash
# Clear cache
rm -rf .next node_modules
npm install
npm run build
```

### Database Connection
- Check connection strings include `?pgbouncer=true`
- Verify both `DATABASE_URL` and `NEON_DATABASE_URL` are set

### Authentication Issues  
- Verify Clerk keys are production keys (not test)
- Check Clerk domain settings match your Vercel URL

### File Upload Issues
- Verify `BLOB_READ_WRITE_TOKEN` is set
- Check API route timeouts in `vercel.json`

## 📞 Quick Help Commands

```bash
# Check deployment status
vercel ls

# View logs
vercel logs your-app

# Check environment variables
vercel env ls

# Redeploy
vercel --prod

# Rollback (if needed)
vercel rollback
```

## 🎯 One-Liner Deploy

Nếu đã setup environment variables:

```bash
cd frontend && npm install && vercel --prod
```

---

**Total time: ~5 phút** ⏱️

**Result: Production-ready MMSE system** 🎉
