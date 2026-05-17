@echo off
echo.
echo ============================================
echo         DataForge AI -- Starting
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: Check Node
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)

:: Backend
echo [1/4] Setting up virtual environment...
cd backend
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [2/4] Installing Python dependencies...
pip install -r requirements.txt -q

echo [3/4] Starting backend on http://localhost:8000 ...
start "DataForge Backend" cmd /k "call venv\Scripts\activate.bat && python main.py"

cd ..

:: Frontend
echo [4/4] Setting up and starting frontend...
cd frontend

if not exist node_modules (
    echo Installing Node dependencies...
    npm install
)

echo VITE_API_URL= > .env.local

start "DataForge Frontend" cmd /k "npm run dev"

cd ..

echo.
echo ============================================
echo   DataForge AI is running!
echo.
echo   Frontend  ^>  http://localhost:5173
echo   Backend   ^>  http://localhost:8000
echo   API Docs  ^>  http://localhost:8000/docs
echo.
echo   Close the two terminal windows to stop.
echo ============================================
echo.
pause
