# Script de ejecución robusta para la aplicación Real-ESRGAN
# Creado por: Danny Maaz
# Este script asegura que el servidor se mantenga en ejecución y se reinicie automáticamente si falla.

$Host.UI.RawUI.WindowTitle = "Real-ESRGAN Stability Manager - Danny Maaz"

$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = $OutputEncoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Real-ESRGAN Upscaling Profesional - Stability Manager" -ForegroundColor Cyan
Write-Host "            Creado por: Danny Maaz (github.com/dannymaaz)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$LOG_FILE = "server_robust_log.txt"
$ERROR_LOG = "error_robust_log.txt"

# Detectar entorno virtual si existe
$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT_DIR

$PYTHON_EXE = "python"
if (Test-Path (Join-Path $ROOT_DIR "venv\Scripts\python.exe")) {
    Write-Host "[*] Entorno virtual detectado: venv" -ForegroundColor Yellow
    $PYTHON_EXE = Join-Path $ROOT_DIR "venv\Scripts\python.exe"
} elseif (Test-Path (Join-Path $ROOT_DIR ".venv\Scripts\python.exe")) {
    Write-Host "[*] Entorno virtual detectado: .venv" -ForegroundColor Yellow
    $PYTHON_EXE = Join-Path $ROOT_DIR ".venv\Scripts\python.exe"
} else {
    Write-Host "[!] ADVERTENCIA: No se encontro carpeta 'venv' o '.venv'." -ForegroundColor Yellow
    Write-Host "    Se usara el Python global del sistema." -ForegroundColor Yellow
}

# Bucle principal de ejecución
while ($true) {
    Write-Host "[*] Iniciando servidor: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    
    # Ejecutar la aplicación
    # Usamos python run.py que ya maneja la lógica de apertura de navegador y uvicorn
    & $PYTHON_EXE run.py 2>> $ERROR_LOG | Tee-Object -FilePath $LOG_FILE -Append -Encoding utf8
    
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "[!] El servidor se detuvo normalmente (Ctrl+C)." -ForegroundColor Yellow
        break
    } else {
        Write-Host "[!] El servidor falló con código de salida $exitCode. Reiniciando en 5 segundos..." -ForegroundColor Red
        Write-Output "[$(Get-Date)] Crash detectado. Exit Code: $exitCode" | Out-File -FilePath $ERROR_LOG -Append
        Start-Sleep -Seconds 5
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Stability Manager finalizado." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
