"""
Script para descargar modelos Real-ESRGAN
Autor: Danny Maaz (github.com/dannymaaz)
"""

import os
import sys
import urllib.request
from pathlib import Path
from app.config import MODELS_DIR, MODEL_URLS

def download_file(url, destination):
    """
    Descarga un archivo con barra de progreso
    
    Args:
        url: URL del archivo
        destination: Ruta de destino
    """
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Descargando... {percent}%")
        sys.stdout.flush()
    
    print(f"\nDescargando desde: {url}")
    urllib.request.urlretrieve(url, destination, progress_hook)
    print("\n  ✓ Descarga completada")

def main():
    """Descarga todos los modelos necesarios"""
    print("=" * 60)
    print("Descargador de Modelos Real-ESRGAN")
    print("=" * 60)
    
    # Crear directorio de modelos si no existe
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Descargar cada modelo
    for model_name, url in MODEL_URLS.items():
        model_path = MODELS_DIR / f"{model_name}.pth"
        
        if model_path.exists():
            print(f"\n✓ {model_name} ya existe")
            continue
        
        print(f"\n📥 Descargando {model_name}...")
        try:
            download_file(url, model_path)
        except Exception as e:
            print(f"\n✗ Error al descargar {model_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✓ Descarga de modelos completada")
    print("=" * 60)
    print("\nPuedes ejecutar la aplicación con: python run.py")

if __name__ == "__main__":
    main()
