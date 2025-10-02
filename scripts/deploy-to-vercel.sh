#!/bin/bash

# 🚀 Automatic deployment script for MMSE System to Vercel
# Usage: ./scripts/deploy-to-vercel.sh

set -e  # Exit on any error

echo "🚀 Starting MMSE System deployment to Vercel..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the frontend directory."
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    print_warning "Vercel CLI not found. Installing..."
    npm install -g vercel
    print_success "Vercel CLI installed successfully"
fi

# Step 1: Prepare for deployment
print_status "Step 1: Preparing for deployment..."

# Install dependencies
print_status "Installing dependencies..."
npm install

# Run build to check for errors
print_status "Running build check..."
npm run build

print_success "Build check completed successfully"

# Step 2: Setup Vercel project (if not already done)
print_status "Step 2: Setting up Vercel project..."

if [ ! -f ".vercel/project.json" ]; then
    print_status "Linking to Vercel project..."
    vercel
else
    print_success "Project already linked to Vercel"
fi

# Step 3: Check environment variables
print_status "Step 3: Checking environment variables..."

# List of required environment variables
required_vars=(
    "DATABASE_URL"
    "NEON_DATABASE_URL"
    "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY"
    "CLERK_SECRET_KEY"
    "BLOB_READ_WRITE_TOKEN"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if ! vercel env ls | grep -q "$var"; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    print_warning "The following environment variables are missing:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo ""
    print_warning "Please set these variables using:"
    print_warning "  vercel env add VARIABLE_NAME production"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Deployment cancelled due to missing environment variables"
        exit 1
    fi
else
    print_success "All required environment variables are set"
fi

# Step 4: Deploy to production
print_status "Step 4: Deploying to Vercel production..."

vercel --prod

if [ $? -eq 0 ]; then
    print_success "Deployment completed successfully!"
    
    # Get the deployment URL
    DEPLOYMENT_URL=$(vercel --prod --confirm 2>/dev/null | grep -o 'https://[^[:space:]]*')
    
    if [ ! -z "$DEPLOYMENT_URL" ]; then
        echo ""
        print_success "🌐 Your app is live at: $DEPLOYMENT_URL"
        echo ""
        
        # Test the deployment
        print_status "Testing deployment..."
        
        # Test health endpoint
        if curl -f -s "$DEPLOYMENT_URL/api/health" > /dev/null; then
            print_success "✅ Health check passed"
        else
            print_warning "⚠️  Health check failed - please check manually"
        fi
        
        echo ""
        print_status "📋 Next steps:"
        echo "  1. Test authentication: $DEPLOYMENT_URL"
        echo "  2. Test MMSE assessment flow"
        echo "  3. Test audio upload functionality"
        echo "  4. Verify database connections"
        echo "  5. Check Vercel dashboard for logs"
        echo ""
        print_status "📚 For detailed testing guide, see: VERCEL_DEPLOYMENT_GUIDE.md"
    fi
    
else
    print_error "Deployment failed. Please check the errors above."
    exit 1
fi

# Step 5: Post-deployment tasks
print_status "Step 5: Post-deployment recommendations..."

echo ""
print_status "🔧 Optional post-deployment tasks:"
echo "  • Run database migrations: npm run drizzle:migrate"
echo "  • Migrate audio files: npm run migrate:audio"
echo "  • Set up custom domain in Vercel dashboard"
echo "  • Configure monitoring and alerts"
echo "  • Test all API endpoints"
echo ""

print_success "🎉 MMSE System deployment to Vercel completed successfully!"
print_status "💡 For troubleshooting, check: VERCEL_DEPLOYMENT_GUIDE.md"
