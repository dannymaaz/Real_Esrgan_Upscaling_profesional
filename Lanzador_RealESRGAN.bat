@echo off
title Lanzador Real-ESRGAN Profesional - Danny Maaz
setlocal enabledelayedexpansion

:: Estética de la consola
echo ============================================================
echo   Real-ESRGAN Upscaling Profesional - Danny Maaz
echo ============================================================
echo.

:: Configurar rutas
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: Buscar el ejecutable de Python en el entorno virtual
set "PYTHON_EXE=python"

if exist "venv\Scripts\python.exe" (
    echo [*] Entorno virtual detectado: venv
    set "PYTHON_EXE=%ROOT_DIR%venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    echo [*] Entorno virtual detectado: .venv
    set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
) else (
    echo [!] ADVERTENCIA: No se encontro carpeta 'venv' o '.venv'.
    echo     Se usara el Python global del sistema.
    echo.
)

:: Verificar si el script existe
if not exist "run.py" (
    echo [X] ERROR: No se encuentra el archivo 'run.py' en: %ROOT_DIR%
    pause
    exit /b 1
)

:: Ejecutar la aplicacion
echo [*] Lanzando motor de IA y servidor...
echo.

"%PYTHON_EXE%" run.py

:: Manejo de errores
if %ERRORLEVEL% neq 0 (
    echo.
    echo ============================================================
    echo   [!] LA APLICACION SE HA DETENIDO (Codigo: %ERRORLEVEL%)
    echo ============================================================
    echo.
    echo [?] Posibles soluciones:
    echo  - Instala las dependencias: pip install -r requirements.txt
    echo  - Descarga los modelos: python download_models.py
    echo.
    pause
) else (
    echo.
    echo [*] Servidor cerrado correctamente.
    pause
)
