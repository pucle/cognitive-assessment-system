Write-Host "Starting Vietnamese Cognitive Assessment Backend..." -ForegroundColor Green
Write-Host ""
Write-Host "This script will start the backend server on port 5001" -ForegroundColor Yellow
Write-Host "Make sure you have Python and all dependencies installed" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host ""

Set-Location backend
python run.py

Write-Host ""
Write-Host "Backend server stopped. Press any key to exit..." -ForegroundColor Red
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
