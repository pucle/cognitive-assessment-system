#!/bin/bash
# Vercel Frontend Deployment Script
# Run this after logging in to Vercel

echo "⚡ Starting Vercel Frontend Deployment..."
echo "========================================"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Installing Vercel CLI..."
    npm install -g vercel
fi

# Login to Vercel (if not already logged in)
echo "🔑 Checking Vercel login..."
if ! vercel whoami &> /dev/null; then
    echo "Please login to Vercel first:"
    echo "vercel login"
    exit 1
fi

# Navigate to frontend directory
cd frontend

# Link project
echo "🔗 Linking Vercel project..."
vercel link

# Set environment variables
echo "⚙️ Setting environment variables..."

# Core app settings
vercel env add NEXT_PUBLIC_APP_URL production
vercel env add NODE_ENV production

# Backend connection
echo "Enter your Railway backend URL (from previous step):"
read BACKEND_URL
if [[ -z "$BACKEND_URL" ]]; then
    echo "❌ Backend URL is required"
    exit 1
fi
vercel env add NEXT_PUBLIC_API_URL production << EOF
$BACKEND_URL
EOF

# Database
vercel env add DATABASE_URL production
vercel env add NEON_DATABASE_URL production

# Authentication
echo "Enter your Clerk Publishable Key:"
read CLERK_PUBLISHABLE_KEY
if [[ -z "$CLERK_PUBLISHABLE_KEY" ]]; then
    echo "❌ Clerk Publishable Key is required"
    exit 1
fi
vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY production << EOF
$CLERK_PUBLISHABLE_KEY
EOF

echo "Enter your Clerk Secret Key:"
read CLERK_SECRET_KEY
if [[ -z "$CLERK_SECRET_KEY" ]]; then
    echo "❌ Clerk Secret Key is required"
    exit 1
fi
vercel env add CLERK_SECRET_KEY production << EOF
$CLERK_SECRET_KEY
EOF

# AI Services
echo "Enter your OpenAI API Key:"
read OPENAI_API_KEY
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "❌ OpenAI API Key is required"
    exit 1
fi
vercel env add OPENAI_API_KEY production << EOF
$OPENAI_API_KEY
EOF

echo "Enter your Gemini API Key:"
read GEMINI_API_KEY
if [[ -z "$GEMINI_API_KEY" ]]; then
    echo "❌ Gemini API Key is required"
    exit 1
fi
vercel env add GEMINI_API_KEY production << EOF
$GEMINI_API_KEY
EOF

# Audio settings
vercel env add ENABLE_PAID_TRANSCRIPTION production << EOF
true
EOF

vercel env add TRANSCRIPTION_BUDGET_LIMIT production << EOF
10.00
EOF

vercel env add VI_ASR_MODEL production << EOF
nguyenvulebinh/wav2vec2-large-vietnamese-250h
EOF

# Deploy to production
echo "🚀 Deploying to Vercel production..."
FRONTEND_URL=$(vercel --prod)

echo ""
echo "🎉 Vercel frontend deployment completed!"
echo "Frontend URL: $FRONTEND_URL"
echo ""
echo "Next step: Test your production deployment"
