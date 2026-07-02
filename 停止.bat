@echo off
chcp 65001 >nul
echo 停止车型图像识别系统服务...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a 2>nul
)
echo 服务已停止。
pause
