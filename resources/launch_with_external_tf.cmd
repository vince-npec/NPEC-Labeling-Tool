@echo off
setlocal

set "TFPY=%~1"
if "%TFPY%"=="" set "TFPY=%NPEC_EXTERNAL_TF_PYTHON%"

if "%TFPY%"=="" (
  echo Usage:
  echo   launch_with_external_tf.cmd C:\path\to\python.exe
  echo.
  echo Or set NPEC_EXTERNAL_TF_PYTHON before launching.
  exit /b 1
)

if not exist "%TFPY%" (
  echo TensorFlow Python not found:
  echo   %TFPY%
  exit /b 1
)

set "APP_EXE=%~dp0dist-win\NPEC Labeling Tool\NPEC Labeling Tool.exe"
if not exist "%APP_EXE%" (
  echo Built app not found:
  echo   %APP_EXE%
  echo.
  echo Build first with:
  echo   resources\build_windows.cmd
  exit /b 1
)

set "NPEC_EXTERNAL_TF_PYTHON=%TFPY%"
echo Launching with NPEC_EXTERNAL_TF_PYTHON=%NPEC_EXTERNAL_TF_PYTHON%
start "" "%APP_EXE%"
