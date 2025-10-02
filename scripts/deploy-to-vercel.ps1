# Automatic deployment script for MMSE System to Vercel (PowerShell)
# Usage: .\scripts\deploy-to-vercel.ps1

param(
    [switch]$SkipBuild,
    [switch]$Force
)

# Enable error handling
$ErrorActionPreference = "Stop"

Write-Host "Starting MMSE System deployment to Vercel..." -ForegroundColor Blue

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

try {
    # Check if we're in the frontend directory
    if (-not (Test-Path "package.json")) {
        Write-Error "package.json not found. Please run this script from the frontend directory."
        exit 1
    }

    # Check if Vercel CLI is installed
    try {
        vercel --version | Out-Null
        Write-Success "Vercel CLI found"
    }
    catch {
        Write-Warning "Vercel CLI not found. Installing..."
        npm install -g vercel
        Write-Success "Vercel CLI installed successfully"
    }

    # Step 1: Prepare for deployment
    Write-Status "Step 1: Preparing for deployment..."

    # Install dependencies
    Write-Status "Installing dependencies..."
    npm install

    if (-not $SkipBuild) {
        # Run build to check for errors
        Write-Status "Running build check..."
        npm run build
        Write-Success "Build check completed successfully"
    }

    # Step 2: Setup Vercel project (if not already done)
    Write-Status "Step 2: Setting up Vercel project..."

    if (-not (Test-Path ".vercel\project.json")) {
        Write-Status "Linking to Vercel project..."
        Write-Host "Please follow the prompts to link your project..." -ForegroundColor Yellow
        vercel
    }
    else {
        Write-Success "Project already linked to Vercel"
    }

    # Step 3: Check environment variables
    Write-Status "Step 3: Checking environment variables..."

    $requiredVars = @(
        "DATABASE_URL",
        "NEON_DATABASE_URL", 
        "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY",
        "CLERK_SECRET_KEY",
        "BLOB_READ_WRITE_TOKEN"
    )

    $envList = vercel env ls 2>&1
    $missingVars = @()

    foreach ($var in $requiredVars) {
        if ($envList -notmatch $var) {
            $missingVars += $var
        }
    }

    if ($missingVars.Count -gt 0) {
        Write-Warning "The following environment variables are missing:"
        foreach ($var in $missingVars) {
            Write-Host "  - $var" -ForegroundColor Yellow
        }
        Write-Host ""
        Write-Warning "Please set these variables using:"
        Write-Warning "  vercel env add VARIABLE_NAME production"
        Write-Host ""
        
        if (-not $Force) {
            $continue = Read-Host "Do you want to continue anyway? (y/N)"
            if ($continue -ne "y" -and $continue -ne "Y") {
                Write-Error "Deployment cancelled due to missing environment variables"
                exit 1
            }
        }
    }
    else {
        Write-Success "All required environment variables are set"
    }

    # Step 4: Deploy to production
    Write-Status "Step 4: Deploying to Vercel production..."

    $deployOutput = vercel --prod 2>&1
    $deployExitCode = $LASTEXITCODE

    if ($deployExitCode -eq 0) {
        Write-Success "Deployment completed successfully!"
        
        # Extract deployment URL from output
        $deploymentUrl = ($deployOutput | Select-String "https://.*\.vercel\.app" | Select-Object -First 1).Matches.Value
        
        if ($deploymentUrl) {
            Write-Host ""
            Write-Success "Your app is live at: $deploymentUrl"
            Write-Host ""
            
            # Test the deployment
            Write-Status "Testing deployment..."
            
            try {
                $healthCheck = Invoke-WebRequest -Uri "$deploymentUrl/api/health" -TimeoutSec 10 -UseBasicParsing
                if ($healthCheck.StatusCode -eq 200) {
                    Write-Success "Health check passed"
                }
                else {
                    Write-Warning "Health check returned status: $($healthCheck.StatusCode)"
                }
            }
            catch {
                Write-Warning "Health check failed - please check manually"
            }
            
            Write-Host ""
            Write-Status "Next steps:"
            Write-Host "  1. Test authentication: $deploymentUrl" -ForegroundColor White
            Write-Host "  2. Test MMSE assessment flow" -ForegroundColor White
            Write-Host "  3. Test audio upload functionality" -ForegroundColor White
            Write-Host "  4. Verify database connections" -ForegroundColor White
            Write-Host "  5. Check Vercel dashboard for logs" -ForegroundColor White
            Write-Host ""
            Write-Status "For detailed testing guide, see: VERCEL_DEPLOYMENT_GUIDE.md"
        }
    }
    else {
        Write-Error "Deployment failed. Please check the errors above."
        Write-Host $deployOutput -ForegroundColor Red
        exit 1
    }

    # Step 5: Post-deployment tasks
    Write-Status "Step 5: Post-deployment recommendations..."

    Write-Host ""
    Write-Status "Optional post-deployment tasks:"
    Write-Host "  - Run database migrations: npm run drizzle:migrate" -ForegroundColor White
    Write-Host "  - Migrate audio files: npm run migrate:audio" -ForegroundColor White
    Write-Host "  - Set up custom domain in Vercel dashboard" -ForegroundColor White
    Write-Host "  - Configure monitoring and alerts" -ForegroundColor White
    Write-Host "  - Test all API endpoints" -ForegroundColor White
    Write-Host ""

    Write-Success "MMSE System deployment to Vercel completed successfully!"
    Write-Status "For troubleshooting, check: VERCEL_DEPLOYMENT_GUIDE.md"

}
catch {
    Write-Error "Deployment script failed: $($_.Exception.Message)"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
