@echo off
echo ========================================
echo DeepSceneLoc Full Stack Launcher
echo ========================================
echo.

echo Starting Python Backend (Flask)...
start "DeepSceneLoc Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python webapp\backend_api.py"

timeout /t 3 /nobreak > nul

echo.
echo Starting Frontend (React + Express)...
start "DeepSceneLoc Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo Both servers starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to stop all servers...
pause > nul

taskkill /FI "WindowTitle eq DeepSceneLoc Backend*" /T /F
taskkill /FI "WindowTitle eq DeepSceneLoc Frontend*" /T /F

echo.
echo Servers stopped.
pause
