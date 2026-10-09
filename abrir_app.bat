@echo off
setlocal
cd /d "%~dp0"
title Cajeros Link en CABA

rem Abre la app del TPO. Doble clic en este archivo (Windows).

rem 1. Buscar Python 3.10 o superior (se prefiere 3.12)
set "PY="
for %%v in (3.12 3.13 3.11 3.10 3.14) do (
  if not defined PY (
    py -%%v -c "pass" >nul 2>nul && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY goto :sinpython

rem 2. Crear el entorno la primera vez
if not exist ".venv\Scripts\python.exe" (
  echo Creando el entorno de Python ^(solo la primera vez^)...
  %PY% -m venv .venv
  if errorlevel 1 goto :error
)

rem 3. Instalar o actualizar lo que necesita la app
echo Preparando la app. La primera vez tarda unos minutos...
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r app\requirements.txt
if errorlevel 1 goto :error

rem 4. Abrir el navegador y arrancar la app
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 8; Start-Process 'http://localhost:8501'"
echo.
echo La app se abre en http://localhost:8501
echo Para cerrarla: cerrar esta ventana.
echo.
".venv\Scripts\python.exe" -m streamlit run app\app.py
if errorlevel 1 pause
exit /b 0

:sinpython
echo.
echo Falta instalar Python 3.10 o superior (recomendado: 3.12).
echo.
echo   Forma simple: abrir una terminal y escribir   winget install -e --id Python.Python.3.12
echo   O bajarlo de https://www.python.org/downloads/windows/ y en el instalador marcar "Add python.exe to PATH".
echo.
echo Luego cerrar esta ventana y volver a abrir abrir_app.bat
echo.
pause
exit /b 1

:error
echo.
echo No se pudo preparar o abrir la app. Copiar el mensaje de arriba y pasarlo al grupo.
echo.
pause
exit /b 1
