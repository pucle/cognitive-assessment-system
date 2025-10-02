# PowerShell script to start both frontend and backend
Write-Host "🚀 Starting Cognitive Assessment System" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Yellow

# Function to start backend
function Start-Backend {
    Write-Host "🔧 Starting Backend Server..." -ForegroundColor Cyan
    $backendPath = Join-Path $PSScriptRoot "backend"

    if (Test-Path $backendPath) {
        Set-Location $backendPath
        Start-Process -FilePath "python" -ArgumentList "app.py" -NoNewWindow
        Write-Host "✅ Backend started on http://localhost:5001" -ForegroundColor Green
    } else {
        Write-Host "❌ Backend directory not found!" -ForegroundColor Red
    }
}

# Function to start frontend
function Start-Frontend {
    Write-Host "🌐 Starting Frontend Server..." -ForegroundColor Cyan
    $frontendPath = Join-Path $PSScriptRoot "frontend"

    if (Test-Path $frontendPath) {
        Set-Location $frontendPath
        Start-Process -FilePath "npm" -ArgumentList "run dev" -NoNewWindow
        Write-Host "✅ Frontend started on http://localhost:3000" -ForegroundColor Green
    } else {
        Write-Host "❌ Frontend directory not found!" -ForegroundColor Red
    }
}

# Main execution
Write-Host "📋 Starting services in order..." -ForegroundColor Yellow
Write-Host ""

# Start backend first
Start-Backend

# Wait a bit for backend to initialize
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start frontend
Start-Frontend

# Final message
Write-Host ""
Write-Host "🎉 System Startup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "🔧 Backend:  http://localhost:5001" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 If you see backend connection errors, the backend may still be starting up." -ForegroundColor Yellow
Write-Host "   Just refresh the frontend page after a few seconds." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
