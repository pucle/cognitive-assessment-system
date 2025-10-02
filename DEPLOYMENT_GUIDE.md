# 🚀 Cognitive Assessment System - Production Deployment Guide

## 🎯 Executive Summary

This guide provides complete instructions for deploying the Cognitive Assessment System to production with:

- **Frontend**: Next.js 15 + TypeScript on Vercel
- **Backend**: Flask + Python 3.11 on Railway
- **Database**: PostgreSQL on Neon
- **Storage**: Vercel Blob for audio files
- **Security**: Clerk authentication + comprehensive security measures
- **Cost**: $10-20/month for production-ready system

## 📋 Prerequisites Checklist

### ✅ Required Accounts & Services
- [ ] **GitHub Account**: For repository and CI/CD
- [ ] **Vercel Account**: For frontend hosting ($0 free tier)
- [ ] **Railway Account**: For backend hosting ($5/month starter)
- [ ] **Neon Account**: For PostgreSQL database ($0 free tier)
- [ ] **Clerk Account**: For authentication ($0 free tier)

### ✅ Required API Keys
- [ ] **OpenAI API Key**: For GPT analysis ($0.002/1K tokens)
- [ ] **Google Gemini API Key**: For AI processing (free tier available)
- [ ] **Vercel Blob Token**: For audio file storage (auto-generated)

### ✅ Development Environment
- [ ] **Node.js 18+**: `node --version`
- [ ] **Python 3.11+**: `python --version`
- [ ] **Railway CLI**: `npm install -g @railway/cli`
- [ ] **Vercel CLI**: `npm install -g vercel`

---

## 🔧 Step-by-Step Deployment

### **Phase 1: Infrastructure Setup**

#### **1.1 Setup Database (Neon PostgreSQL)**
```bash
# 1. Create Neon account at https://neon.tech
# 2. Create new project
# 3. Get connection string from dashboard

# Example DATABASE_URL:
# postgresql://username:password@hostname:5432/database_name?sslmode=require

# Test connection
psql "your_database_url" -c "SELECT version();"
```

#### **1.2 Setup Authentication (Clerk)**
```bash
# 1. Create Clerk account at https://clerk.com
# 2. Create new application
# 3. Get API keys from dashboard

# You need:
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_...
CLERK_SECRET_KEY=sk_live_...
```

#### **1.3 Generate Security Secrets**
```bash
# Generate cryptographically secure secrets
python scripts/generate_secrets.py

# This creates:
# SECRET_KEY, JWT_SECRET_KEY, API_TOKEN, AES_KEY, etc.
```

### **Phase 2: Backend Deployment**

#### **2.1 Prepare Backend for Production**
```bash
# 1. Ensure all dependencies are in requirements.txt
cd backend
pip install -r requirements.txt

# 2. Test locally
python app.py

# 3. Build Docker image (optional, Railway can build automatically)
docker build -t cognitive-backend:latest .
```

#### **2.2 Deploy Backend to Railway**
```bash
# 1. Login to Railway
railway login

# 2. Create/link project
railway link  # or railway init

# 3. Set environment variables
railway variables set FLASK_ENV=production
railway variables set DEBUG=false
railway variables set DATABASE_URL="your_neon_database_url"
railway variables set OPENAI_API_KEY="your_openai_key"
railway variables set GEMINI_API_KEY="your_gemini_key"
railway variables set SECRET_KEY="generated_secret_key"
railway variables set CLERK_SECRET_KEY="your_clerk_secret_key"

# 4. Deploy
railway deploy

# 5. Get backend URL
railway domain
# Example: https://cognitive-backend.railway.app
```

### **Phase 3: Frontend Deployment**

#### **3.1 Prepare Frontend for Production**
```bash
# 1. Install dependencies
cd frontend
npm install

# 2. Test build locally
npm run build

# 3. Test locally (optional)
npm run dev
```

#### **3.2 Deploy Frontend to Vercel**
```bash
# 1. Login to Vercel
vercel login

# 2. Link project (from frontend directory)
cd frontend
vercel link

# 3. Set environment variables
vercel env add NEXT_PUBLIC_APP_URL production
vercel env add NEXT_PUBLIC_API_URL production  # Backend URL from Railway
vercel env add DATABASE_URL production
vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY production
vercel env add CLERK_SECRET_KEY production
vercel env add OPENAI_API_KEY production
vercel env add GEMINI_API_KEY production

# 4. Deploy
vercel --prod

# 5. Get frontend URL
# Example: https://cognitive-assessment.vercel.app
```

### **Phase 4: Storage Setup**

#### **4.1 Setup Vercel Blob Storage**
```bash
# Vercel Blob is automatically configured when you deploy to Vercel
# The BLOB_READ_WRITE_TOKEN is auto-generated

# To get the token (if needed):
vercel env ls | grep BLOB
```

### **Phase 5: CI/CD Setup**

#### **5.1 GitHub Actions Setup**
```bash
# 1. Ensure workflows are in .github/workflows/
ls .github/workflows/

# 2. Add repository secrets to GitHub:
# Settings → Secrets and variables → Actions

# Required secrets:
# RAILWAY_TOKEN (from Railway dashboard)
# VERCEL_TOKEN (from Vercel dashboard)
# VERCEL_ORG_ID (from Vercel dashboard)
# VERCEL_PROJECT_ID (from Vercel dashboard)
# BACKEND_URL (Railway URL)
# FRONTEND_URL (Vercel URL)
```

#### **5.2 Push to GitHub**
```bash
# 1. Create GitHub repository
# 2. Push code
git add .
git commit -m "Production deployment setup"
git push origin main

# 3. GitHub Actions will automatically:
# - Run tests
# - Security scans
# - Deploy to Railway/Vercel on merge to main
```

---

## 🧪 Testing & Verification

### **Post-Deployment Testing**
```bash
# 1. Run comprehensive tests
python scripts/test_deployment.py https://your-backend.railway.app https://your-frontend.vercel.app

# 2. Manual verification
# - Visit frontend URL
# - Try authentication flow
# - Test MMSE assessment
# - Check audio recording/upload

# 3. API endpoint testing
curl https://your-backend.railway.app/api/health
curl https://your-backend.railway.app/api/mmse/questions
```

### **Performance Testing**
```bash
# Load testing (optional)
npm install -g artillery

# Create test script
# artillery quick --count 10 --num 5 https://your-backend.railway.app/api/health
```

---

## 📊 Monitoring & Maintenance

### **Setup Monitoring**
```bash
# 1. Sentry Error Tracking (recommended)
# - Sign up: https://sentry.io
# - Create Next.js project for frontend
# - Create Python project for backend
# - Add DSN keys to environment variables

# 2. Uptime Monitoring (optional)
# - UptimeRobot: https://uptimerobot.com (free tier)
# - Ping health endpoints every 5 minutes
```

### **Daily Monitoring**
- Check Railway/Vercel dashboards
- Review error logs in Sentry
- Monitor GitHub Actions status
- Verify backup completion

### **Weekly Maintenance**
```bash
# Update dependencies
cd frontend && npm audit && npm update
cd ../backend && pip install --upgrade -r requirements.txt

# Security scans
npm audit --audit-level high
```

### **Monthly Maintenance**
```bash
# Rotate secrets
python scripts/generate_secrets.py
# Update in Railway/Vercel

# Database maintenance
# Check Neon console for performance metrics

# Backup verification
python scripts/backup.sh --dry-run
```

---

## 🚨 Troubleshooting

### **Common Issues**

#### **Backend Deployment Fails**
```bash
# Check Railway logs
railway logs

# Common fixes:
# - Missing environment variables
# - Database connection issues
# - Python dependencies not installing
```

#### **Frontend Build Fails**
```bash
# Check Vercel build logs
vercel logs

# Common fixes:
# - Node.js version mismatch
# - Missing environment variables
# - Build script errors
```

#### **Database Connection Issues**
```bash
# Test connection
psql "your_database_url" -c "SELECT 1;"

# Check Neon dashboard
# - Connection limits
# - Database size
# - Performance metrics
```

#### **Authentication Problems**
```bash
# Check Clerk dashboard
# - API keys
# - Authorized domains
# - User management

# Verify environment variables
vercel env ls | grep CLERK
railway variables | grep CLERK
```

---

## 💰 Cost Optimization

### **Current Cost Breakdown**
```
Vercel (Frontend):     $0/month (Hobby plan)
Railway (Backend):     $5/month (Starter plan)
Neon (Database):       $0/month (Free tier)
Vercel Blob (Storage): $0-5/month (based on usage)
OpenAI API:           $0-10/month (based on usage)
TOTAL:                $5-20/month
```

### **Scaling Costs**
```
If you need more resources:
- Railway Professional: $25/month (2GB RAM, more CPU)
- Neon Pro: $19/month (more storage, better performance)
- Vercel Pro: $20/month (more bandwidth, analytics)
```

### **Cost Monitoring**
- Railway: Built-in usage dashboard
- Vercel: Analytics in dashboard
- Neon: Console metrics
- OpenAI: API usage dashboard

---

## 🔐 Security Checklist

### **Pre-Deployment Security**
- [ ] All secrets generated with `generate_secrets.py`
- [ ] Environment variables never committed to Git
- [ ] HTTPS enabled automatically (Vercel/Railway)
- [ ] CORS configured for correct domains
- [ ] Rate limiting active
- [ ] Input validation implemented

### **Post-Deployment Security**
- [ ] SSL certificate valid
- [ ] Security headers present
- [ ] Authentication working
- [ ] API keys protected
- [ ] Database access restricted
- [ ] Audit logging enabled

### **Ongoing Security**
- [ ] Regular dependency updates
- [ ] Security scans passing
- [ ] Secrets rotated quarterly
- [ ] Access logs monitored
- [ ] Backup security verified

---

## 📞 Support & Resources

### **Service Documentation**
- **Vercel**: https://vercel.com/docs
- **Railway**: https://docs.railway.app
- **Neon**: https://neon.tech/docs
- **Clerk**: https://clerk.com/docs

### **Troubleshooting Resources**
- **RUNBOOK.md**: Detailed operational procedures
- **ANALYSIS_REPORT.md**: System architecture details
- **scripts/**: Automation scripts for maintenance

### **Emergency Contacts**
- Railway Support: https://railway.app/support
- Vercel Support: https://vercel.com/support
- Neon Support: https://neon.tech/docs/introduction/support

---

## ✅ Final Verification Checklist

### **Deployment Complete**
- [ ] Frontend deployed and accessible
- [ ] Backend deployed and responding
- [ ] Database connected and populated
- [ ] Authentication working
- [ ] API endpoints functional
- [ ] Audio upload/download working
- [ ] CI/CD pipeline active
- [ ] Monitoring configured
- [ ] Backups scheduled
- [ ] Documentation updated

### **Go-Live Ready**
- [ ] All tests passing
- [ ] Performance acceptable
- [ ] Security verified
- [ ] Team notified
- [ ] Monitoring alerts active
- [ ] Rollback plan ready

---

## 🎉 Success!

Once all steps are completed, your Cognitive Assessment System will be:

- **Production-ready** with enterprise-grade security
- **Scalable** to handle increasing load
- **Monitored** with comprehensive error tracking
- **Maintainable** with automated CI/CD
- **Cost-effective** at $10-20/month

**Your URLs:**
- Frontend: https://your-app.vercel.app
- Backend: https://your-backend.railway.app
- Health Check: https://your-backend.railway.app/api/health

**Next Steps:**
1. Monitor system performance
2. Gather user feedback
3. Plan feature enhancements
4. Scale as needed

---

*Deployment Guide v2.0 - Updated January 2025*
