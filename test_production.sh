#!/bin/bash
# Test Production Deployment
# Run this after both Railway and Vercel deployments are complete

echo "🧪 Testing Production Deployment..."
echo "==================================="

# Get URLs from user
echo "Enter your Railway backend URL:"
read BACKEND_URL

echo "Enter your Vercel frontend URL:"
read FRONTEND_URL

if [[ -z "$BACKEND_URL" || -z "$FRONTEND_URL" ]]; then
    echo "❌ Both URLs are required"
    exit 1
fi

echo ""
echo "Testing Backend: $BACKEND_URL"
echo "Testing Frontend: $FRONTEND_URL"
echo ""

# Test backend health
echo "🔍 Testing backend health..."
if curl -f "$BACKEND_URL/api/health" &> /dev/null; then
    echo "✅ Backend health check passed"
else
    echo "❌ Backend health check failed"
fi

# Test backend API
echo "🔍 Testing backend API endpoints..."
if curl -f "$BACKEND_URL/api/mmse/questions" &> /dev/null; then
    echo "✅ MMSE questions endpoint working"
else
    echo "❌ MMSE questions endpoint failed"
fi

# Test frontend
echo "🔍 Testing frontend accessibility..."
if curl -f "$FRONTEND_URL" &> /dev/null; then
    echo "✅ Frontend homepage accessible"
else
    echo "❌ Frontend homepage not accessible"
fi

# Test frontend-backend connection
echo "🔍 Testing frontend-backend connection..."
if curl -f "$FRONTEND_URL/cognitive-assessment" &> /dev/null; then
    echo "✅ Cognitive assessment page accessible"
else
    echo "❌ Cognitive assessment page not accessible"
fi

echo ""
echo "🎉 Production testing completed!"
echo ""
echo "📊 Your Cognitive Assessment System is now live:"
echo "🌐 Frontend: $FRONTEND_URL"
echo "🔧 Backend: $BACKEND_URL"
echo ""
echo "Next steps:"
echo "1. Test user registration and login"
echo "2. Test MMSE assessment flow"
echo "3. Test audio recording and transcription"
echo "4. Setup monitoring and alerts"
echo ""
echo "For automated monitoring, run:"
echo "python scripts/test_deployment.py $BACKEND_URL $FRONTEND_URL"
