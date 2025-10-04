# PROJECT ANALYSIS REPORT
Generated: December 2025

## 📦 Project Overview

**Project Name:** Cognitive Assessment System (cavang)
**Description:** A comprehensive Vietnamese cognitive assessment platform using MMSE (Mini-Mental State Examination) with AI-powered audio transcription, linguistic analysis, and machine learning evaluation for cognitive health assessment.
**Tech Stack:**
- Frontend: Next.js 15.4.5 with TypeScript
- Backend: Flask (Python) with ML pipeline
- Database: PostgreSQL (Neon)
- Authentication: Clerk
- File Storage: Vercel Blob
- AI Services: OpenAI GPT-4, Google Gemini, Hugging Face Transformers

---

## 🎨 FRONTEND ANALYSIS

### Structure
```
frontend/
├── app/                          # Next.js App Router
│   ├── (main)/                   # Main app routes
│   ├── (marketing)/              # Marketing pages
│   ├── api/                      # API routes (33 files)
│   ├── buttons/                  # Button components page
│   ├── globals.css               # Global styles
│   ├── hooks/                    # React hooks
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Home page
│   ├── providers/                # Context providers
│   ├── scripts/                  # Utility scripts
│   ├── types/                    # TypeScript types
│   └── utils/                    # Utility functions
├── components/                   # React components
│   ├── BilingualLayout.tsx       # Multi-language support
│   ├── CognitiveAssessmentRecorder.tsx
│   ├── CognitiveAssessmentResult.tsx
│   ├── ErrorBoundary.tsx
│   ├── LanguageSwitcher.tsx
│   ├── MemoryTest components
│   ├── MMSE components
│   ├── PersonalInfoForm.tsx
│   ├── ProfilePopup.tsx
│   ├── RealtimeAssessment.tsx
│   ├── ui/ (17 UI components)
│   └── Various result displays
├── db/                           # Database configuration
│   ├── drizzle.ts                # Drizzle ORM config
│   └── schema.ts                 # Database schema (12 tables)
├── drizzle/                      # Database migrations
├── lib/                          # Library configurations
│   ├── api-config.ts
│   ├── api-utils.ts
│   ├── clerk.ts                  # Authentication
│   ├── languages/                # Multi-language support
│   ├── research-papers.ts
│   ├── tts-service.ts            # Text-to-speech
│   └── utils.ts
├── public/                       # Static assets
└── Various config files
```

### Framework Details
- **Framework:** Next.js 15.4.5 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS v4
- **Build Tool:** Next.js built-in
- **Entry Point:** `app/layout.tsx` + `app/page.tsx`
- **Package Manager:** npm (Node.js ≥18.17.0)

### Build Configuration
- **Build Command:** `npm run build`
- **Output Directory:** `.next/` (auto-handled by Next.js)
- **Dev Command:** `npm run dev` (port 3000)
- **Production Start:** `npm run start`
- **Database Migrations:** `npm run drizzle:generate && npm run drizzle:migrate`

### Environment Variables
Frontend requires these env vars:

**Required for Production:**
```bash
# Backend API
NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-backend.railway.app

# Clerk Authentication (required)
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...

# Database (for Drizzle migrations)
DATABASE_URL=postgresql://user:pass@host:5432/db
NEON_DATABASE_URL=postgresql://user:pass@host:5432/db

# File Storage (Vercel Blob)
BLOB_READ_WRITE_TOKEN=vercel_blob_...

# AI Services (if used in frontend)
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...
```

**How to find:** Searched in code:
- File: `lib/api-config.ts` - Backend URL configuration
- File: `lib/clerk.ts` - Authentication setup
- File: `db/drizzle.ts` - Database connection
- File: Multiple components using `process.env.NEXT_PUBLIC_*`

### API Calls Made by Frontend
Frontend makes extensive API calls to backend:

**Authentication & User Management:**
- GET `/api/user/profile` - Get user profile
- POST `/api/database/user/save` - Save user data
- GET `/api/profile/user` - Get user profile (alternative)

**Assessment System:**
- POST `/api/assess` - Start cognitive assessment
- POST `/api/transcribe` - Audio transcription
- POST `/api/features` - Extract audio features
- POST `/api/evaluate` - AI evaluation
- POST `/api/mmse/assess` - MMSE assessment
- GET `/api/mmse/questions` - Get MMSE questions
- POST `/api/mmse/session/start` - Start MMSE session
- POST `/api/mmse/session/{id}/question` - Submit question answer
- GET `/api/mmse/session/{id}/progress` - Get session progress
- POST `/api/mmse/session/{id}/complete` - Complete session
- GET `/api/mmse/session/{id}/results` - Get results

**Data Management:**
- GET `/api/database/user` - Get user data
- POST `/api/generate-summary` - Generate assessment summary
- GET `/api/status` - System status
- GET `/api/config` - Configuration info

**File Operations:**
- POST `/auto-transcribe` - Automatic transcription
- POST `/api/auto-transcribe-raw` - Raw audio transcription

### Dependencies (Key Production Dependencies)
```json
{
  "next": "15.4.5",
  "react": "18.2.0",
  "react-dom": "18.2.0",
  "@clerk/nextjs": "6.31.10",
  "drizzle-orm": "0.44.5",
  "@neondatabase/serverless": "1.0.1",
  "@vercel/blob": "1.1.1",
  "openai": "5.15.0",
  "@google/generative-ai": "0.24.1",
  "framer-motion": "12.23.12",
  "tailwindcss": "4",
  "typescript": "5",
  "drizzle-kit": "0.31.4"
}
```

---

## ⚙️ BACKEND ANALYSIS

### Structure
```
backend/
├── app.py                        # Main Flask application (49K+ lines)
├── app_v2.py                     # Alternative Flask app
├── run.py                        # Entry point script
├── config/
│   └── production.py             # Production configuration
├── services/                     # Business logic services
├── routes/                       # API route handlers
├── models/                       # ML models directory
├── languages/                    # Language-specific code
├── middleware/                   # Middleware components
├── utils.py                      # Utility functions
├── gunicorn.conf.py              # Production WSGI config
├── requirements.txt              # Python dependencies
└── Various ML/data files
```

### Framework Details
- **Framework:** Flask (Python)
- **Entry Point:** `run.py` (imports from `app.py`)
- **WSGI App Object:** `app` (Flask application instance)
- **Default Port:** 5000 (configurable via PORT env var)
- **ML Pipeline:** Advanced audio processing + AI evaluation
- **Database:** PostgreSQL via direct SQL/connection strings

### Startup Flow
```python
# run.py
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
```

**For Production:**
- Use gunicorn: ✅ Configured in `gunicorn.conf.py`
- Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

### API Endpoints
Backend exposes 41+ API endpoints:

**Core Assessment Pipeline:**
- POST `/api/assess` - Full cognitive assessment
- POST `/api/transcribe` - Audio transcription (OpenAI/Gemini)
- POST `/api/features` - Audio feature extraction
- POST `/api/evaluate` - AI evaluation with GPT-4
- POST `/auto-transcribe` - Automatic transcription
- POST `/api/test-transcription` - Test transcription

**MMSE System:**
- POST `/api/mmse/assess` - MMSE assessment
- GET `/api/mmse/questions` - Get MMSE questions
- POST `/api/mmse/session/start` - Start assessment session
- POST `/api/mmse/session/{id}/question` - Submit question answer
- GET `/api/mmse/session/{id}/progress` - Session progress
- POST `/api/mmse/session/{id}/complete` - Complete session
- GET `/api/mmse/session/{id}/results` - Get results
- GET `/api/mmse/model-info` - Model information
- POST `/api/mmse/transcribe` - MMSE transcription
- GET `/api/mmse/performance` - Performance metrics

**User Management:**
- GET `/api/user/profile` - User profile
- GET `/api/database/user` - User data
- POST `/api/database/user/save` - Save user data
- GET `/api/profile/user` - Alternative profile endpoint

**System Management:**
- GET `/api/health` - Health check
- GET `/api/status` - System status
- GET `/api/config` - Configuration
- POST `/api/generate-summary` - Generate summary
- GET `/api/languages` - Supported languages
- GET `/api/translate/{key}` - Translation

**Queue System:**
- POST `/api/assess-queue` - Queue assessment
- GET `/api/assessment-status/{task_id}` - Check status
- GET `/api/assessment-results/{identifier}` - Get results
- POST `/api/test-queue-flow` - Test queue
- GET `/api/debug/queue-status` - Debug queue

### Database
- **Type:** PostgreSQL
- **Connection:** Via DATABASE_URL environment variable
- **Schema:** Complex multi-table schema (12+ tables)
- **Tables:** users, sessions, questions, stats, temp_questions, user_reports, training_samples, community_assessments, cognitive_assessment_results, contact_messages
- **Migration System:** Custom SQL scripts (no ORM in backend)

### Environment Variables
Backend requires extensive environment variables:

**Critical (Required):**
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
SECRET_KEY=[32+ char random string]
PORT=8000  # Railway auto-provides
HOST=0.0.0.0
FLASK_ENV=production
DEBUG=false
```

**AI Services (Required for full functionality):**
```bash
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
```

**Configuration:**
```bash
# File paths
UPLOAD_PATH=/tmp/uploads
STORAGE_PATH=./storage
RECORDINGS_PATH=./recordings
MODEL_PATH=./models

# Cache settings
CACHE_DIR=/tmp/cognitive_cache
CACHE_TTL=3600
AUDIO_CACHE_SIZE=100
TEXT_CACHE_SIZE=500

# Feature flags
ENABLE_PAID_TRANSCRIPTION=true
TRANSCRIPTION_BUDGET_LIMIT=5.00
ENABLE_ADVANCED_ANALYTICS=true
ENABLE_REAL_TIME_PROCESSING=false
ENABLE_MODEL_EXPLAINABILITY=true

# Vercel Blob (for file storage)
BLOB_READ_WRITE_TOKEN=vercel_blob_...

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASS=...
EMAIL_USER=noreply@yourapp.com

# Monitoring (optional)
SENTRY_DSN=https://...
LOG_LEVEL=WARNING
```

### Dependencies
```txt
# Web Framework
Flask>=2.0.0
flask-cors>=3.0.0

# ML & AI
torch>=1.9.0
transformers>=4.15.0
openai>=1.0.0
google-generativeai>=0.8.0
scikit-learn>=1.0.0
xgboost>=1.5.0
shap>=0.42.0
lime>=0.2.0

# Audio Processing
librosa>=0.9.0
soundfile>=0.10.0
pydub>=0.25.0
openai-whisper>=20230124

# Database
psycopg2-binary>=2.9.0

# Utilities
python-dotenv>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
joblib>=1.1.0
requests>=2.31.0

# Production Server
gunicorn>=21.2.0
```

### ML Pipeline
**Advanced cognitive assessment pipeline:**
1. **Audio Input** → Recording or upload
2. **Transcription** → OpenAI Whisper or Gemini ASR
3. **Audio Feature Extraction** → Librosa (MFCCs, spectral features)
4. **Linguistic Analysis** → GPT-4 evaluation
5. **ML Classification** → SVM/XGBoost models
6. **MMSE Scoring** → Standardized cognitive assessment
7. **Report Generation** → Comprehensive results

**Models Used:**
- Vietnamese ASR: nguyenvulebinh/wav2vec2-large-vietnamese-250h
- GPT-4: For linguistic analysis and evaluation
- Custom SVM models: For cognitive level classification
- XGBoost: Alternative ML model

---

## 🗄️ DATABASE SCHEMA

### Core Tables

**users**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| name | TEXT | User full name |
| age | TEXT | Age |
| gender | TEXT | Gender |
| email | TEXT | Email (unique) |
| phone | TEXT | Phone number |
| avatar | TEXT | Avatar URL |
| title | TEXT | Title/position |
| imageSrc | TEXT | Image source |
| mmseScore | TEXT | MMSE score |
| displayName | TEXT | Display name |
| createdAt | TIMESTAMP | Creation timestamp |
| updatedAt | TIMESTAMP | Update timestamp |

**sessions**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| userId | TEXT | User ID |
| mode | ENUM | personal/community |
| status | ENUM | in_progress/completed/error |
| startTime | TIMESTAMP | Session start |
| endTime | TIMESTAMP | Session end |
| totalScore | REAL | Total assessment score |
| mmseScore | INTEGER | MMSE score |
| cognitiveLevel | ENUM | mild/moderate/severe/normal |
| emailSent | INTEGER | Email sent flag |
| createdAt | TIMESTAMP | Creation timestamp |
| updatedAt | TIMESTAMP | Update timestamp |

**questions**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| sessionId | TEXT | Session ID |
| questionId | TEXT | Question identifier |
| questionContent | TEXT | Question text |
| audioFile | TEXT | Audio file path/URL |
| autoTranscript | TEXT | Automatic transcription |
| manualTranscript | TEXT | Manual transcription |
| linguisticAnalysis | JSONB | GPT-4 analysis results |
| audioFeatures | JSONB | Audio feature data |
| evaluation | TEXT | AI evaluation |
| feedback | TEXT | User feedback |
| score | REAL | Question score |
| processedAt | TIMESTAMP | Processing timestamp |
| createdAt | TIMESTAMP | Creation timestamp |
| userName | TEXT | User name |
| userAge | INTEGER | User age |
| userEducation | INTEGER | Years of education |
| userEmail | TEXT | User email |

**stats**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| sessionId | TEXT | Session ID |
| userId | TEXT | User ID |
| timestamp | TIMESTAMP | Record timestamp |
| mode | ENUM | personal/community |
| summary | JSONB | Assessment summary |
| detailedResults | JSONB | Full results |
| chartData | JSONB | Chart data |
| exerciseRecommendations | JSONB | Exercise recommendations |
| createdAt | TIMESTAMP | Creation timestamp |
| userName | TEXT | User name |
| userAge | INTEGER | User age |
| userEducation | INTEGER | Years of education |
| userEmail | TEXT | User email |
| audioFiles | JSONB | Audio file references |

### Additional Tables
- **temp_questions**: Temporary question data during assessment
- **user_reports**: Legacy user reports
- **training_samples**: ML training data
- **community_assessments**: Community assessment results
- **cognitive_assessment_results**: Comprehensive assessment results
- **contact_messages**: Contact form submissions

---

## 🔐 SECURITY ANALYSIS

### Current State
- [x] Environment variables properly configured
- [x] No hardcoded secrets in codebase
- [x] CORS configured (allow all origins in development)
- [x] Clerk authentication for frontend
- [x] Database connection via environment variables
- [x] File upload restrictions (configurable)
- [ ] Rate limiting not implemented
- [ ] Input validation partial
- [ ] HTTPS enforced in production (via hosting platforms)
- [ ] Security headers not implemented

### Recommendations
1. **Restrict CORS origins** in production to Vercel domain
2. **Implement rate limiting** (flask-limiter)
3. **Add input validation** (marshmallow or similar)
4. **Implement security headers** middleware
5. **Regular dependency updates** for security patches
6. **API key rotation** for production
7. **Audit logging** for sensitive operations

### File Upload Security
- **Storage:** Vercel Blob (secure, scalable)
- **Validation:** File type and size restrictions
- **Access:** Signed URLs for private files
- **Cleanup:** Automatic temporary file removal

---

## 📈 DEPLOYMENT REQUIREMENTS

### Services Needed

1. **Frontend Hosting:** Vercel (Free tier)
   - Root directory: `frontend`
   - Framework: Next.js 15
   - Build command: `npm run build`
   - Environment: Node.js 18+

2. **Backend Hosting:** Railway ($5/month)
   - Root directory: `backend`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Environment: Python 3.11+
   - Memory: 1GB+ recommended for ML models

3. **Database:** Neon Postgres (Free tier: 0.5GB)
   - Storage needed: ~1GB for production data
   - Connection pooling required
   - Backup automation needed

4. **File Storage:** Vercel Blob (Free: 1GB)
   - Audio recordings and assessment files
   - CDN delivery for global access

5. **Authentication:** Clerk (Free tier)
   - User management and authentication

### Environment Variables Setup

**Vercel (Frontend):**
```bash
NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-backend.railway.app
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
DATABASE_URL=postgresql://user:pass@host/db
BLOB_READ_WRITE_TOKEN=vercel_blob_...
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...
```

**Railway (Backend):**
```bash
DATABASE_URL=postgresql://user:pass@host/db
SECRET_KEY=your-32-char-secret-key
PORT=8000
HOST=0.0.0.0
FLASK_ENV=production
DEBUG=false
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...
BLOB_READ_WRITE_TOKEN=vercel_blob_...
CACHE_DIR=/tmp/cognitive_cache
ENABLE_PAID_TRANSCRIPTION=true
TRANSCRIPTION_BUDGET_LIMIT=5.00
```

### Estimated Monthly Cost
- Vercel: $0 (free tier)
- Railway: $5 (starter plan)
- Neon: $0 (free tier: 0.5GB)
- Clerk: $0 (free tier: 10k users)
- Vercel Blob: $0 (free tier: 1GB)
**Total: $5/month**

---

## ⚠️ BLOCKERS / ISSUES FOUND

### Critical Issues
1. ❌ **CORS allows all origins** (`CORS(app)` without restrictions)
   - Fix: Update to specific Vercel domain in production

2. ❌ **File uploads stored locally in Railway**
   - Problem: Railway ephemeral filesystem loses files on restart
   - Fix: Migrate all uploads to Vercel Blob storage

3. ❌ **No rate limiting implemented**
   - Risk: API abuse, resource exhaustion
   - Fix: Add flask-limiter dependency and middleware

4. ⚠️ **Heavy ML processing on single thread**
   - Problem: Large audio files may timeout
   - Fix: Implement background job queue (Celery + Redis)

### Configuration Issues
1. ⚠️ **Complex environment variable setup**
   - Many optional vars that affect functionality
   - Need clear documentation of which are required vs optional

2. ⚠️ **Database connection pooling not configured**
   - May hit connection limits under load
   - Need SQLAlchemy engine configuration

### Performance Considerations
1. ⚠️ **ML models loaded on every request**
   - Should cache models in memory
   - Current implementation reloads models frequently

2. ⚠️ **No caching layer for expensive operations**
   - Audio processing is compute-intensive
   - Need Redis for caching results

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-deployment
- [ ] Update CORS origins to production frontend URL
- [ ] Migrate file storage from local to Vercel Blob
- [ ] Implement rate limiting
- [ ] Test all API endpoints locally
- [ ] Verify database schema matches code
- [ ] Ensure all required environment variables documented

### Infrastructure Setup
- [ ] Create Neon PostgreSQL database
- [ ] Run database migrations
- [ ] Create Vercel project (frontend)
- [ ] Create Railway project (backend)
- [ ] Setup Clerk authentication
- [ ] Configure Vercel Blob storage

### Environment Configuration
- [ ] Set all Vercel environment variables
- [ ] Set all Railway environment variables
- [ ] Test database connectivity
- [ ] Verify API keys are valid

### Deployment
- [ ] Deploy backend to Railway
- [ ] Deploy frontend to Vercel
- [ ] Update frontend with backend URL
- [ ] Test end-to-end functionality

### Post-deployment
- [ ] Monitor application logs
- [ ] Test file upload functionality
- [ ] Verify ML pipeline works
- [ ] Setup monitoring and alerts
- [ ] Configure automated backups

---

## 📝 NEXT STEPS

Based on this analysis, here's what needs to be done:

1. **Immediate Fixes (Before Deployment):**
   - Fix CORS configuration for production
   - Implement Vercel Blob for file storage
   - Add rate limiting to prevent abuse

2. **Infrastructure Setup:**
   - Create accounts on Vercel, Railway, Neon, Clerk
   - Setup database and run migrations
   - Configure environment variables

3. **Deployment:**
   - Deploy backend first (Railway)
   - Deploy frontend second (Vercel)
   - Test integration between services

4. **Optimization:**
   - Implement caching for ML models
   - Add background job processing
   - Setup monitoring and logging

**Total estimated deployment time: 3-4 hours**
**Monthly cost: $5**

---

*End of Analysis Report - Generated for Cognitive Assessment System*
