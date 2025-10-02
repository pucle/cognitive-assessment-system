// Sentry Configuration for Error Tracking
// Add this to your frontend (Next.js) and backend (Flask) applications

// ================================
// Frontend Sentry Configuration
// ================================
// File: frontend/sentry.client.config.js
import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,

  // Adjust this value in production, or use tracesSampler for greater control
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  // Setting this option to true will print useful information to the console while you're setting up Sentry.
  debug: process.env.NODE_ENV === 'development',

  replaysOnErrorSampleRate: 1.0,

  // This sets the sample rate to be 10%. You may want this to be 100% while
  // in development and sample at a lower rate in production
  replaysSessionSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  // You can remove this option if you're not planning to use the Sentry Session Replay feature:
  integrations: [
    Sentry.replayIntegration({
      // Additional Replay configuration goes in here, for example:
      maskAllText: true,
      blockAllMedia: true,
    }),
    Sentry.browserTracingIntegration(),
    Sentry.httpIntegration(),
  ],

  // Performance monitoring
  enabled: process.env.NODE_ENV === 'production',

  // Filter out health check endpoints
  beforeSend(event) {
    // Don't send events for health check endpoints
    if (event.request?.url?.includes('/api/health')) {
      return null;
    }
    return event;
  },

  // Environment
  environment: process.env.NODE_ENV || 'development',

  // Release tracking
  release: process.env.VERCEL_GIT_COMMIT_SHA || process.env.npm_package_version,
});

// ================================
// Backend Sentry Configuration
// ================================
// File: backend/sentry_config.py
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.redis import RedisIntegration
import os

def init_sentry():
    """Initialize Sentry for error tracking"""

    sentry_dsn = os.getenv('SENTRY_DSN')
    if not sentry_dsn:
        print("⚠️  SENTRY_DSN not configured, skipping Sentry initialization")
        return

    sentry_sdk.init(
        dsn=sentry_dsn,

        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=0.1,

        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=0.1,

        integrations=[
            FlaskIntegration(),
            # RedisIntegration() if using Redis
        ],

        # Environment
        environment=os.getenv('FLASK_ENV', 'development'),

        # Release tracking
        release=os.getenv('GIT_COMMIT_SHA', 'unknown'),

        # Filter out health check endpoints
        before_send=lambda event, hint: (
            None if event.get('request', {}).get('url', '').endswith('/api/health')
            else event
        ),

        # Performance monitoring
        enable_tracing=True,

        # Debug mode
        debug=os.getenv('SENTRY_DEBUG', 'false').lower() == 'true',
    )

    print("✅ Sentry initialized for error tracking")

# ================================
# Environment Variables Required
# ================================
# Add these to your deployment environments:

# Frontend (.env.local):
# NEXT_PUBLIC_SENTRY_DSN=https://your-sentry-dsn@sentry.io/project_id

# Backend (.env):
# SENTRY_DSN=https://your-sentry-dsn@sentry.io/project_id
# SENTRY_DEBUG=false

# ================================
# Setup Instructions
# ================================

/*
1. Create Sentry Project:
   - Go to https://sentry.io and create account
   - Create new project for Frontend (Next.js)
   - Create new project for Backend (Python/Flask)
   - Get DSN keys for each project

2. Install Dependencies:
   Frontend: npm install @sentry/nextjs
   Backend: pip install sentry-sdk[flask]

3. Configure Environment Variables:
   - Add NEXT_PUBLIC_SENTRY_DSN to Vercel
   - Add SENTRY_DSN to Railway

4. Import Configuration:
   Frontend: Import in _app.tsx or layout.tsx
   Backend: Call init_sentry() in app.py after app creation

5. Test Error Reporting:
   - Frontend: Throw error in browser console
   - Backend: Add temporary error in API endpoint
   - Check Sentry dashboard for errors

6. Configure Alerts:
   - Set up email/Slack alerts for new issues
   - Configure issue owners and assignees
   - Set up release tracking
*/

export { init_sentry };
