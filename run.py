"""
Script de ejecución simplificado para la aplicación
Autor: Danny Maaz (github.com/dannymaaz)
"""

import sys
import subprocess
import webbrowser
from pathlib import Path
from time import sleep

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        import fastapi
        import uvicorn
        import torch
        import cv2
        import realesrgan
        return True
    except ImportError as e:
        print("=" * 60)
        print("[!] DEPENDENCIAS FALTANTES")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nPor favor instala las dependencias:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return False

def check_models():
    """Verifica que los modelos estén descargados"""
    from app.config import MODELS
    
    missing_models = []
    for key, info in MODELS.items():
        if not info["path"].exists():
            missing_models.append(info["name"])
    
    if missing_models:
        print("=" * 60)
        print("[!] RESTANTES MODELOS")
        print("=" * 60)
        print("\nLos siguientes modelos no están descargados:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nDescarga los modelos ejecutando:")
        print("  python download_models.py")
        print("=" * 60)
        return False
    
    return True

def main():
    """Ejecuta la aplicación"""
    print("=" * 60)
    print("Real-ESRGAN Upscaling Profesional")
    print("    Creado por Danny Maaz")
    print("=" * 60)
    
    # Verificar dependencias
    print("\nVerificando dependencias...")
    if not check_dependencies():
        sys.exit(1)
    print("   [OK] Dependencias instaladas")
    
    # Verificar modelos
    print("\nVerificando modelos...")
    if not check_models():
        sys.exit(1)
    print("   [OK] Modelos disponibles")
    
    # Importar configuración
    from app.config import HOST, PORT
    
    # Iniciar servidor
    print("\n" + "=" * 60)
    print("Iniciando servidor...")
    print(f"   URL: http://{HOST}:{PORT}")
    print("=" * 60)
    print("\nPresiona Ctrl+C para detener el servidor\n")
    
    # Abrir navegador después de un momento
    def open_browser():
        sleep(2)
        webbrowser.open(f"http://{HOST}:{PORT}")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Ejecutar servidor
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=HOST,
            port=PORT,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Servidor detenido")
        print("=" * 60)

if __name__ == "__main__":
    main()
