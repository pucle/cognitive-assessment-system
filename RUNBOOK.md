# 🚀 Cognitive Assessment System - Production Runbook

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Monitoring & Health Checks](#monitoring--health-checks)
- [Troubleshooting](#troubleshooting)
- [Maintenance Procedures](#maintenance-procedures)
- [Emergency Procedures](#emergency-procedures)
- [Performance Optimization](#performance-optimization)
- [Backup & Recovery](#backup--recovery)
- [Security Procedures](#security-procedures)

---

## 🚀 Quick Start

### **Pre-flight Checklist**
- [ ] All environment variables configured
- [ ] Database migrations completed
- [ ] Secrets rotated and secure
- [ ] Monitoring tools configured
- [ ] Backup procedures tested

### **Daily Operations**
1. **Check System Health**: Visit monitoring dashboard
2. **Review Error Logs**: Check Sentry/GitHub for issues
3. **Monitor Performance**: Watch response times and resource usage
4. **Review Backups**: Ensure automated backups are working

### **URLs & Access Points**
```
Frontend (Vercel): https://your-frontend.vercel.app
Backend API (Railway): https://your-backend.railway.app
Database (Neon): https://console.neon.tech
Monitoring (Sentry): https://sentry.io/your-org
GitHub Actions: https://github.com/your-org/repo/actions
```

---

## 📊 Monitoring & Health Checks

### **Health Check Endpoints**

#### **Backend Health Checks**
```bash
# Comprehensive health check
curl https://your-backend.railway.app/api/health

# Liveness probe (Kubernetes/Docker)
curl https://your-backend.railway.app/api/health/live

# Readiness probe
curl https://your-backend.railway.app/api/health/ready

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-01-01T10:00:00Z",
  "checks": {
    "database": {"status": "healthy", "message": "Connected"},
    "ml_models": {"status": "healthy", "message": "Models loaded"},
    "disk_space": {"status": "healthy", "message": "OK"}
  }
}
```

#### **Frontend Health Checks**
```bash
# Homepage accessibility
curl -I https://your-frontend.vercel.app

# API connectivity test
curl https://your-frontend.vercel.app/api/health
```

### **Monitoring Dashboards**

#### **Real-time Monitoring**
- **Railway Dashboard**: `https://railway.app/project/your-project`
  - CPU/Memory usage
  - Request logs
  - Deployment status

- **Vercel Dashboard**: `https://vercel.com/your-org/your-project`
  - Build status
  - Function logs
  - Analytics

- **Neon Console**: `https://console.neon.tech`
  - Database performance
  - Connection count
  - Query insights

#### **Error Tracking**
- **Sentry Dashboard**: `https://sentry.io/your-org`
  - Error rates
  - Performance issues
  - Release tracking

#### **CI/CD Monitoring**
- **GitHub Actions**: Check workflow status
- **Security Scans**: Review vulnerability reports

### **Alert Conditions**

#### **Critical Alerts (Immediate Action Required)**
- ❌ Backend health check fails
- ❌ Database connection lost
- ❌ ML model loading fails
- ❌ Memory usage > 90%
- ❌ API response time > 30s

#### **Warning Alerts (Monitor Closely)**
- ⚠️ CPU usage > 80%
- ⚠️ Disk space < 10%
- ⚠️ Error rate > 5%
- ⚠️ Response time > 5s

#### **Info Alerts (Track Trends)**
- 📊 User activity spikes
- 📊 New error patterns
- 📊 Performance degradation

---

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **Issue: Frontend Not Loading**
**Symptoms:** 404 errors, blank page, JavaScript errors
**Solutions:**
```bash
# 1. Check Vercel deployment status
vercel ls

# 2. Check build logs
vercel logs --follow

# 3. Verify environment variables
vercel env ls

# 4. Redeploy if needed
vercel redeploy
```

#### **Issue: Backend API Errors**
**Symptoms:** 500 errors, timeouts, API failures
**Solutions:**
```bash
# 1. Check Railway logs
railway logs

# 2. Check health endpoint
curl https://your-backend.railway.app/api/health

# 3. Restart service
railway restart

# 4. Check database connection
railway run python -c "import psycopg2; psycopg2.connect(os.environ['DATABASE_URL'])"
```

#### **Issue: Database Connection Issues**
**Symptoms:** Connection timeouts, query failures
**Solutions:**
```bash
# 1. Check Neon console
# Go to: https://console.neon.tech

# 2. Verify connection string
railway run python -c "import os; print(os.environ.get('DATABASE_URL', 'NOT SET'))"

# 3. Test connection
railway run python -c "
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
conn.close()
print('Connection successful')
"

# 4. Check connection limits
# In Neon: Project Settings → Connection Limits
```

#### **Issue: ML Model Errors**
**Symptoms:** Assessment failures, model loading errors
**Solutions:**
```bash
# 1. Check model files exist
railway run ls -la models/

# 2. Verify model integrity
railway run python -c "
import joblib
model = joblib.load('models/model.pkl')
print('Model loaded successfully')
print('Model type:', type(model))
"

# 3. Check memory usage
railway run python -c "
import psutil
print('Memory usage:', psutil.virtual_memory().percent, '%')
"
```

#### **Issue: Audio Processing Failures**
**Symptoms:** Upload fails, transcription errors
**Solutions:**
```bash
# 1. Check Vercel Blob storage
# Go to: Vercel Dashboard → Storage → Blob

# 2. Test blob operations
vercel env pull
npm run test-blob  # Create test script if needed

# 3. Check file size limits
# Vercel Blob: 5GB per file
# Railway: Check deployment limits
```

#### **Issue: Authentication Problems**
**Symptoms:** Login fails, API auth errors
**Solutions:**
```bash
# 1. Check Clerk dashboard
# Go to: https://dashboard.clerk.com

# 2. Verify environment variables
vercel env ls | grep CLERK
railway env | grep CLERK

# 3. Test Clerk API
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.clerk.com/v1/users
```

### **Performance Issues**

#### **High CPU Usage**
```bash
# Check Railway metrics
railway logs --tail 100

# Scale up if needed
railway up --plan professional
```

#### **High Memory Usage**
```bash
# Check for memory leaks
railway run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print('Memory usage:', process.memory_info().rss / 1024 / 1024, 'MB')
"

# Restart service
railway restart
```

#### **Slow API Responses**
```bash
# Check database query performance
railway run python -c "
import psycopg2
import time

conn = psycopg2.connect(os.environ['DATABASE_URL'])
cursor = conn.cursor()

start = time.time()
cursor.execute('SELECT COUNT(*) FROM questions')
result = cursor.fetchone()
end = time.time()

print(f'Query took: {end - start:.3f}s')
print(f'Result: {result}')
"
```

---

## 🛠️ Maintenance Procedures

### **Weekly Maintenance**

#### **1. Update Dependencies**
```bash
# Frontend
cd frontend
npm audit
npm update
npm run build
npm test

# Backend
cd backend
pip list --outdated
pip install --upgrade -r requirements.txt
python -m pytest
```

#### **2. Security Updates**
```bash
# Check for security vulnerabilities
npm audit --audit-level=high
pip install safety
safety check

# Update base images
docker pull python:3.11-slim
```

#### **3. Database Maintenance**
```bash
# Vacuum and analyze
railway run python -c "
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
conn.autocommit = True
cursor = conn.cursor()
cursor.execute('VACUUM ANALYZE')
print('Database maintenance completed')
"
```

### **Monthly Maintenance**

#### **1. Rotate Secrets**
```bash
# Generate new secrets
python scripts/generate_secrets.py

# Update in Railway/Vercel
railway env set SECRET_KEY=new_secret_key
vercel env add SECRET_KEY production
```

#### **2. Review Logs**
```bash
# Check error patterns
# Review Sentry issues
# Analyze performance metrics
```

#### **3. Backup Verification**
```bash
# Test backup restoration
python scripts/backup.sh --dry-run
# Restore to staging environment if available
```

### **Quarterly Maintenance**

#### **1. Major Updates**
```bash
# Update major versions
npm install next@latest
pip install flask==3.0.0

# Test compatibility
npm run build
python -m pytest
```

#### **2. Security Audit**
```bash
# Full security scan
npm audit --audit-level=moderate
# Manual code review
# Penetration testing if needed
```

---

## 🚨 Emergency Procedures

### **System Down - Critical Priority**

#### **Immediate Actions**
1. **Assess Impact**: Check which services are affected
2. **Notify Team**: Alert via Slack/phone
3. **Check Monitoring**: Review Sentry/Railway alerts
4. **Isolate Issue**: Determine root cause quickly

#### **Recovery Steps**
```bash
# 1. Check service status
railway status
vercel ls

# 2. Restart services
railway restart
vercel redeploy --prod

# 3. Check database
# Go to Neon console, check metrics

# 4. Restore from backup if needed
python scripts/rollback.sh database
```

#### **Communication Template**
```
🚨 INCIDENT: System Down

Status: Investigating
Impact: All users affected
ETA: 15 minutes
Updates: Monitoring closely

Will update when resolved.
```

### **Data Loss Incident**

#### **Assessment**
1. **Determine Scope**: What data was lost?
2. **Check Backups**: Verify backup integrity
3. **Impact Analysis**: Who is affected?

#### **Recovery**
```bash
# 1. Stop all writes
railway maintenance on

# 2. Restore from backup
python scripts/rollback.sh database

# 3. Verify data integrity
# Run comprehensive tests

# 4. Resume operations
railway maintenance off
```

### **Security Breach**

#### **Immediate Response**
1. **Isolate**: Disconnect compromised systems
2. **Assess**: Determine breach scope
3. **Contain**: Change all passwords/keys
4. **Notify**: Legal authorities if needed

#### **Post-Incident**
```bash
# Rotate all secrets
python scripts/generate_secrets.py

# Update all access keys
# Review security logs
# Implement additional security measures
```

---

## ⚡ Performance Optimization

### **Current Performance Targets**

| Metric | Target | Current | Action if Exceeded |
|--------|--------|---------|-------------------|
| API Response Time | <2s | ~1.5s | Optimize queries |
| Frontend Load Time | <3s | ~2.2s | Code splitting |
| Memory Usage | <80% | ~65% | Scale up |
| CPU Usage | <70% | ~55% | Optimize ML models |
| Error Rate | <1% | ~0.5% | Fix bugs |

### **Optimization Strategies**

#### **Database Optimization**
```sql
-- Add indexes for slow queries
CREATE INDEX CONCURRENTLY idx_questions_created_at ON questions(created_at);
CREATE INDEX CONCURRENTLY idx_sessions_user_id ON sessions(user_id);

-- Query optimization
EXPLAIN ANALYZE SELECT * FROM questions WHERE user_email = 'test@example.com';
```

#### **API Optimization**
```python
# Add caching
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@app.route('/api/questions')
@cache.cached(timeout=300)  # 5 minutes
def get_questions():
    # Cached API response
    pass
```

#### **ML Model Optimization**
```python
# Model quantization for smaller size
import torch
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Batch processing
def batch_predict(features_list):
    # Process multiple predictions at once
    pass
```

### **Scaling Strategies**

#### **Vertical Scaling**
```bash
# Upgrade Railway plan
railway up --plan professional  # $25/month

# Increase Vercel limits
# Contact Vercel support for higher limits
```

#### **Horizontal Scaling**
```bash
# Railway automatically scales
# Vercel serverless scales automatically

# Database read replicas (future)
# CDN for static assets
```

---

## 💾 Backup & Recovery

### **Automated Backups**

#### **Database Backups**
```bash
# Daily automated backup
0 2 * * * /path/to/scripts/backup.sh

# Backup verification
0 3 * * * /path/to/scripts/backup.sh --dry-run
```

#### **File Backups**
```bash
# Audio files backup (if needed)
# Vercel Blob has built-in redundancy
```

### **Backup Testing**

#### **Monthly Backup Verification**
```bash
# Test backup integrity
python scripts/backup.sh --dry-run

# Test restoration (on staging)
# Create staging environment
# Restore backup
# Verify data integrity
```

### **Disaster Recovery**

#### **Recovery Time Objectives (RTO)**
- **Critical Services**: 1 hour
- **Full System**: 4 hours
- **Data Recovery**: 2 hours

#### **Recovery Point Objectives (RPO)**
- **Database**: 1 hour (hourly backups)
- **Files**: Real-time (Vercel Blob replication)

---

## 🔐 Security Procedures

### **Access Control**

#### **Environment Variables**
- Never commit secrets to Git
- Rotate keys every 90 days
- Use different keys for each environment
- Monitor for secret leaks

#### **Database Security**
```sql
-- Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY users_policy ON users FOR ALL USING (clerk_id = current_setting('app.current_user_id'));

-- Audit logging
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name TEXT,
    operation TEXT,
    old_values JSONB,
    new_values JSONB,
    user_id TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### **Monitoring & Alerting**

#### **Security Alerts**
- Failed login attempts > 5
- Unusual API usage patterns
- New user registrations from suspicious IPs
- Secret leaks detected

#### **Compliance**
- GDPR compliance for EU users
- HIPAA compliance for medical data
- Regular security audits

---

## 📞 Support Contacts

### **Internal Team**
- **DevOps Lead**: [Name] - [Contact]
- **Backend Developer**: [Name] - [Contact]
- **Frontend Developer**: [Name] - [Contact]
- **Security Officer**: [Name] - [Contact]

### **External Services**
- **Railway Support**: https://railway.app/support
- **Vercel Support**: https://vercel.com/support
- **Neon Support**: https://neon.tech/docs/introduction/support
- **Clerk Support**: https://clerk.com/support
- **Sentry Support**: https://sentry.io/support

### **Emergency Contacts**
- **Primary**: [Phone] - [Name]
- **Secondary**: [Phone] - [Name]
- **On-call Schedule**: [Link to schedule]

---

## 📈 Metrics & KPIs

### **System Metrics**
- **Availability**: 99.9% uptime target
- **Performance**: P95 response time < 2s
- **Error Rate**: < 1% of requests
- **User Satisfaction**: > 95% based on feedback

### **Business Metrics**
- **Active Users**: Track daily/monthly active users
- **Assessment Completion**: Track successful assessments
- **Response Accuracy**: ML model performance metrics

### **Monitoring Dashboards**
- **Grafana**: Custom metrics dashboard
- **DataDog**: Infrastructure monitoring
- **Sentry**: Error tracking dashboard

---

*Last updated: January 2025*
*Version: 1.0*
