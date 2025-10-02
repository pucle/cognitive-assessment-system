# 🚀 DEPLOYMENT INSTRUCTIONS - Cognitive Assessment App

## 📋 PRE-DEPLOYMENT CHECKLIST

### ✅ Completed Fixes
- [x] Environment configuration (.env.local created)
- [x] API endpoint alignment (backend/frontend)
- [x] Error handling improvements
- [x] Fallback mechanisms implemented
- [x] Logging and debugging enhanced
- [x] Testing tools created

### 🔧 CURRENT STATUS
- **Backend**: ✅ Running on http://localhost:5001
- **Frontend**: ✅ Configuration fixed, ready to start
- **API Integration**: ✅ Endpoints aligned and tested
- **Error Handling**: ✅ Comprehensive fallbacks in place

## 🚀 STEP-BY-STEP STARTUP GUIDE

### 1. **Start Backend Server** [[memory:8516144]]
```bash
cd backend
env .ok  # Set up environment
python app.py
```

**Expected Output:**
- Server starts on http://localhost:5001
- Health check: http://localhost:5001/api/health returns 200
- Questions endpoint: http://localhost:5001/api/mmse/questions returns questions

### 2. **Start Frontend Development Server**
```bash
cd frontend
npm install  # If dependencies need to be installed
npm run dev
```

**Expected Output:**
- Frontend starts on http://localhost:3000
- No more "Failed to load questions" errors
- Cognitive assessment page loads properly

### 3. **Verify Integration**
1. Open http://localhost:3000/cognitive-assessment
2. Check browser console for successful API calls:
   - `🔗 Attempting to fetch: http://localhost:5001/api/mmse/questions`
   - `✅ Questions found, processing...`
   - `✅ Mapped questions: [number] questions`

## 🔍 TROUBLESHOOTING GUIDE

### Issue: "Backend not running"
**Symptoms:**
- Console shows network errors
- Questions fail to load
- fetchWithFallback falls back to mock data

**Solution:**
1. Ensure backend is running: `cd backend && python app.py`
2. Check port 5001 is not in use
3. Verify environment variables in backend

### Issue: "Questions array empty"
**Symptoms:**
- Questions load but array is empty
- Console shows parsed data but no questions

**Solution:**
1. Check backend questions.json file exists
2. Verify backend endpoint returns proper format
3. Check backend logs for errors

### Issue: "Environment variables not found"
**Symptoms:**
- API_BASE_URL pointing to production
- Fetch calls going to wrong server

**Solution:**
1. Ensure `.env.local` exists in frontend directory
2. Restart frontend development server
3. Verify NEXT_PUBLIC_API_URL=http://localhost:5001

### Issue: "Assessment submission fails"
**Symptoms:**
- Recording works but assessment fails
- 404 errors on assessment endpoints

**Solution:**
1. Check backend has assess-queue endpoint
2. Verify request format matches backend expectations
3. Check CORS configuration

## 📊 HEALTH CHECK COMMANDS

### Backend Health Check
```bash
curl http://localhost:5001/api/health
```
Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "openai_available": true,
  "transcriber_available": true
}
```

### Questions Endpoint Check
```bash
curl http://localhost:5001/api/mmse/questions
```
Should return:
```json
{
  "success": true,
  "data": {
    "questions": [...],
    "total_points": 30
  }
}
```

### Frontend API Check
Open browser console on http://localhost:3000/cognitive-assessment and look for:
- `✅ Questions found, processing...`
- `✅ Setting questions state: [number] questions`
- No network errors

## ⚡ QUICK FIXES FOR COMMON ISSUES

### 1. **Port Already in Use**
```bash
# Find process using port 5001
netstat -ano | findstr :5001
# Kill the process
taskkill /PID [process_id] /F
```

### 2. **Python Dependencies Missing**
```bash
cd backend
pip install -r requirements.txt
```

### 3. **Node Dependencies Missing**
```bash
cd frontend
npm install
```

### 4. **Environment Variables Reset**
Recreate `frontend/.env.local`:
```env
NODE_ENV=development
NEXT_PUBLIC_API_URL=http://localhost:5001
NEXT_PUBLIC_DEBUG_MODE=true
```

## 🎯 PERFORMANCE OPTIMIZATION

### Backend Optimizations
1. **Model Loading**: Backend loads ML model on startup (may take 30-60s)
2. **Assessment Processing**: First assessment may be slow as models initialize
3. **Memory Usage**: Monitor memory for large audio files

### Frontend Optimizations
1. **API Timeouts**: Set to 60s for assessment, 15s for other calls
2. **Error Boundaries**: Implemented comprehensive fallbacks
3. **Local Storage**: User data cached for offline capability

## 🚨 MONITORING AND ALERTS

### Key Metrics to Monitor
1. **Backend Response Times**: Health check should be <1s
2. **Assessment Processing**: Should complete within 60s
3. **Error Rates**: Monitor console for network failures
4. **Memory Usage**: Both frontend and backend

### Alert Conditions
- Backend health check fails
- Assessment processing takes >120s
- High error rate in API calls
- Memory usage exceeds 80%

## 📈 SCALING CONSIDERATIONS

### Immediate Needs
- Single user development/testing ✅
- Local development environment ✅
- Basic error handling ✅

### Future Scaling
- Multi-user support (authentication needed)
- Production deployment (Docker/cloud)
- Real-time assessment processing
- Database persistence for results

## 🎉 SUCCESS CRITERIA

The application is **READY TO USE** when:

1. ✅ Backend starts without errors
2. ✅ Frontend loads cognitive assessment page
3. ✅ Questions load from backend API
4. ✅ No console errors for API calls
5. ✅ User can navigate through assessment interface
6. ✅ Recording functionality works
7. ✅ Assessment submissions are processed
8. ✅ Results are displayed properly

**STATUS: 🎯 READY FOR TESTING AND DEVELOPMENT**
