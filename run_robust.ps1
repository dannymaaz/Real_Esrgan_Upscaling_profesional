# Script de ejecución robusta para la aplicación Real-ESRGAN
# Creado por: Danny Maaz
# Este script asegura que el servidor se mantenga en ejecución y se reinicie automáticamente si falla.

$Host.UI.RawUI.WindowTitle = "Real-ESRGAN Stability Manager - Danny Maaz"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Real-ESRGAN Upscaling Profesional - Stability Manager" -ForegroundColor Cyan
Write-Host "            Creado por: Danny Maaz (github.com/dannymaaz)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$LOG_FILE = "server_robust_log.txt"
$ERROR_LOG = "error_robust_log.txt"

# Función para limpiar procesos en el puerto 8000
function Stop-PortProcess {
    $process = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "[!] Puerto 8000 ocupado. Cerrando proceso $((Get-Process -Id $process.OwningProcess[0]).Name)..." -ForegroundColor Yellow
        Stop-Process -Id $process.OwningProcess -Force
        Start-Sleep -Seconds 2
    }
}

# Bucle principal de ejecución
while ($true) {
    Stop-PortProcess
    
    Write-Host "[*] Iniciando servidor: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    
    # Ejecutar la aplicación
    # Usamos python run.py que ya maneja la lógica de apertura de navegador y uvicorn
    python run.py 2>> $ERROR_LOG | Tee-Object -FilePath $LOG_FILE -Append
    
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
