@echo off
setlocal

if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" -PythonVersion 3.11
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
)
if errorlevel 1 (
  echo.
  echo Windows build failed.
  exit /b 1
)

echo.
echo Windows build completed successfully.
exit /b 0
