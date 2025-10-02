@echo off
echo ===================================
echo MMSE v2.0 System Startup
echo ===================================

echo.
echo Checking Python dependencies...
cd backend
pip install -q python-Levenshtein sentence-transformers fpdf2 cryptography

echo.
echo Starting Backend Server...
start "MMSE Backend" cmd /k "python app.py"

timeout /t 3 /nobreak > nul

echo.
echo Starting Frontend Server...
cd ../frontend
start "MMSE Frontend" cmd /k "npm run dev"

echo.
echo ===================================
echo MMSE v2.0 System Started!
echo ===================================
echo.
echo Backend:  http://localhost:5001
echo Frontend: http://localhost:3000
echo MMSE v2:  http://localhost:3000/mmse-v2
echo.
echo Press any key to run integration tests...
pause > nul

echo.
echo Running Integration Tests...
cd ..
python test_mmse_v2_integration.py

echo.
echo Press any key to exit...
pause > nul
