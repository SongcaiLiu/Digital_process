@echo off
chcp 65001 >nul
call D:\App-Download\miniconda\Scripts\activate.bat digital_process
cd /d D:\digitalprocess
set "PYTHONNOUSERSITE=1"
python live_rgbd_system.py
if errorlevel 1 (
  echo.
  echo 启动失败，请查看上方错误信息。
  pause
)
