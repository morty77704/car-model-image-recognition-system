@echo off
chcp 65001 >nul
echo ================================================
echo   车型图像识别系统 - 启动
echo ================================================
echo.
cd /d "%~dp0backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8080
pause
