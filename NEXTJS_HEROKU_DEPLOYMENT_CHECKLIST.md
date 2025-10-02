# 🚀 Next.js TypeScript Heroku Deployment Checklist

## ✅ Chuẩn bị Files (Đã hoàn thành)

### 1. **Package.json Configuration**
- [x] `heroku-postbuild` script: `npm run build`
- [x] `postinstall` script: `npm run drizzle:generate`
- [x] Node.js engines specified: `>=18.17.0`
- [x] Production-ready build scripts

### 2. **Next.js Configuration**
- [x] `frontend/next.config.ts` - Production optimizations
- [x] Image optimization with Heroku domains
- [x] Security headers configured
- [x] CORS setup for API routes
- [x] Webpack audio file handling

### 3. **Environment Variables**
- [x] `frontend/env.example` - Complete template
- [x] All NEXT_PUBLIC_ variables documented
- [x] Backend API URL configuration
- [x] Database and auth variables

### 4. **Deployment Files**
- [x] `frontend/Procfile` - `web: npm start`
- [x] API configuration utility (`lib/api-config.ts`)
- [x] Static assets optimized (SVG files)

## 🔧 Heroku Setup Steps for Next.js Frontend

### Bước 1: Tạo Heroku App cho Frontend
```bash
# Navigate to frontend directory
cd frontend

# Login và tạo app
heroku login
heroku create your-cognitive-frontend-app

# Set buildpack
heroku buildpacks:set heroku/nodejs

# Add remote
git remote add heroku-frontend https://git.heroku.com/your-frontend-app.git
```

### Bước 2: Configure Environment Variables
```bash
# Next.js Configuration
heroku config:set NODE_ENV=production
heroku config:set NEXT_PUBLIC_APP_URL=https://your-frontend-app.herokuapp.com

# Backend API Configuration (CRITICAL!)
heroku config:set NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-backend-app.herokuapp.com

# Clerk Authentication
heroku config:set NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_key
heroku config:set CLERK_SECRET_KEY=sk_test_your_key

# Database Configuration
heroku config:set DATABASE_URL=postgresql://...
heroku config:set NEON_DATABASE_URL=postgresql://...

# API Keys
heroku config:set OPENAI_API_KEY=sk-proj-your_key
heroku config:set GEMINI_API_KEY=your_key

# NextAuth (if using)
heroku config:set NEXTAUTH_URL=https://your-frontend-app.herokuapp.com
heroku config:set NEXTAUTH_SECRET=your_secret

# Vercel Blob Storage (if using)
heroku config:set BLOB_READ_WRITE_TOKEN=your_token

# Email Configuration (if using)
heroku config:set EMAIL_USER=your_email@example.com
heroku config:set EMAIL_PASS=your_app_password
```

### Bước 3: Database Setup
```bash
# Add PostgreSQL addon
heroku addons:create heroku-postgresql:mini

# Check database URL
heroku config:get DATABASE_URL

# Run database migrations
heroku run npm run drizzle:migrate
```

### Bước 4: Deploy Frontend
```bash
# Prepare deployment (from frontend directory)
cd frontend

# Ensure all files are committed
git add .
git commit -m "Prepare Next.js for Heroku deployment"

# Deploy to Heroku
git push heroku-frontend main

# Scale dyno
heroku ps:scale web=1
```

### Bước 5: Verify Deployment
```bash
# Check logs
heroku logs --tail

# Check dyno status
heroku ps

# Open app
heroku open
```

## 📋 Critical Integration Steps

### 1. **Backend-Frontend Connection**
```bash
# BACKEND URL MUST BE SET CORRECTLY
heroku config:set NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-backend-app.herokuapp.com

# Verify backend is accessible
curl https://your-backend-app.herokuapp.com/api/health
```

### 2. **CORS Configuration**
Backend (Flask) should have CORS headers:
```python
from flask_cors import CORS
CORS(app, origins=["https://your-frontend-app.herokuapp.com"])
```

### 3. **Database Sync**
Both apps should use the same database:
```bash
# Frontend
heroku config:set DATABASE_URL=postgresql://...

# Backend
heroku config:set DATABASE_URL=postgresql://...
```

## 🔍 Testing Endpoints

### Frontend URLs to test:
```bash
# Main app
https://your-frontend-app.herokuapp.com

# Health check
https://your-frontend-app.herokuapp.com/api/health

# Cognitive assessment
https://your-frontend-app.herokuapp.com/cognitive-assessment
```

### API Integration Tests:
```javascript
// Test backend connection
const backendUrl = process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL;
fetch(`${backendUrl}/api/health`)
  .then(res => res.json())
  .then(data => console.log('Backend status:', data));
```

## ⚡ Performance Optimizations

### Build Optimizations:
```bash
# Enable build cache
heroku config:set NODE_OPTIONS="--max-old-space-size=4096"

# Optimize build time
heroku config:set NPM_CONFIG_PRODUCTION=false
heroku config:set YARN_PRODUCTION=false
```

### Memory Configuration:
```bash
# Scale to hobby dyno if needed
heroku ps:scale web=1:hobby

# For better performance (paid)
heroku ps:scale web=1:standard-1x
```

## 🚨 Common Issues & Solutions

### 1. **Build Failures**
```bash
# Check Node.js version
heroku config:set NODE_VERSION=18.17.0

# Clear build cache
heroku builds:cache:purge

# Check dependencies
npm audit fix
```

### 2. **API Connection Issues**
```bash
# Verify environment variables
heroku config

# Check CORS settings
# Ensure backend allows frontend domain
```

### 3. **Database Connection**
```bash
# Test database connection
heroku run npm run drizzle:studio

# Reset database if needed
heroku pg:reset DATABASE_URL
heroku run npm run drizzle:migrate
```

### 4. **Static Files Issues**
```bash
# Verify Next.js static export
npm run build
npm run start

# Check public folder
ls -la public/
```

## 📱 Multi-App Deployment Strategy

### Option 1: Separate Apps (Recommended)
```bash
# Backend App
your-cognitive-backend.herokuapp.com

# Frontend App  
your-cognitive-frontend.herokuapp.com
```

### Option 2: Monorepo Deployment
```bash
# Use buildpacks
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku/nodejs

# Configure build
heroku config:set NPM_CONFIG_PREFIX=/app/frontend
```

## 🔐 Security Checklist

### Environment Security:
- [ ] No sensitive data in code
- [ ] All API keys in environment variables
- [ ] CORS properly configured
- [ ] Security headers enabled

### Production Settings:
- [ ] `NODE_ENV=production`
- [ ] Debug mode disabled
- [ ] Error reporting configured
- [ ] Logging properly set up

## 📊 Monitoring & Maintenance

### Health Monitoring:
```bash
# Add monitoring addon
heroku addons:create newrelic:wayne

# Set up log management
heroku addons:create papertrail:choklad
```

### Performance Monitoring:
```bash
# Check app metrics
heroku logs --tail | grep "METRIC"

# Monitor dyno usage
heroku ps:type
```

## 🎯 Final Verification Checklist

### Pre-deployment:
- [ ] All environment variables set
- [ ] Database migrations ready
- [ ] Backend app deployed and working
- [ ] Frontend builds successfully locally

### Post-deployment:
- [ ] Frontend loads successfully
- [ ] API integration works
- [ ] Database operations function
- [ ] Authentication works (Clerk)
- [ ] Static assets load properly
- [ ] Mobile responsive

### Integration Testing:
- [ ] Cognitive assessment flow works
- [ ] Audio recording/upload functions
- [ ] Results saving to database
- [ ] User authentication flow
- [ ] Email notifications (if configured)

## 🚀 Deployment Commands Summary

```bash
# Quick deployment (from frontend directory)
cd frontend
heroku create your-frontend-app
heroku config:set NODE_ENV=production
heroku config:set NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-backend-app.herokuapp.com
heroku config:set NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_key
heroku config:set DATABASE_URL=postgresql://...
git add .
git commit -m "Deploy Next.js to Heroku"
git push heroku main
heroku ps:scale web=1
heroku open
```

---

**🎉 Next.js TypeScript app is ready for Heroku deployment!**

**Key Features:**
- ✅ Production-optimized Next.js configuration
- ✅ Environment variables properly configured
- ✅ Backend API integration ready
- ✅ Database setup with Drizzle ORM
- ✅ Clerk authentication ready
- ✅ Static assets optimized
- ✅ Security headers configured
- ✅ CORS properly handled

**🔗 Dependencies:**
- Backend Flask app should be deployed first
- Database should be accessible from both apps
- All API keys should be configured in both environments
