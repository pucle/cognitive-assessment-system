@echo off
echo 🚀 Starting Cognitive Assessment System
echo ==================================================
echo.

echo 🔧 Starting Backend Server...
echo   Opening new terminal for backend...
start cmd /k "cd backend && python app.py"

echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo 🌐 Starting Frontend Server...
echo   Opening new terminal for frontend...
start cmd /k "cd frontend && npm run dev"

echo.
echo 🎉 System Startup Complete!
echo.
echo 🌐 Frontend: http://localhost:3000
echo 🔧 Backend:  http://localhost:5001
echo.
echo 💡 Tips:
echo   - If you see backend connection errors initially, just wait a few seconds and refresh
echo   - The backend needs time to load ML models on first start
echo   - Both servers will run in separate terminal windows
echo.
echo Press any key to exit...
pause > nul
