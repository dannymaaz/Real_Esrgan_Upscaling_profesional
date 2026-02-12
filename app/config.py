"""
Configuración centralizada de la aplicación Real-ESRGAN Upscaling
Autor: Danny Maaz (github.com/dannymaaz)
"""

import os
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de la aplicación
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Configuración de modelos Real-ESRGAN
MODELS = {
    "2x": {
        "name": "RealESRGAN_x2plus",
        "path": MODELS_DIR / "RealESRGAN_x2plus.pth",
        "scale": 2,
        "description": "Modelo 2x - Más rápido, ideal para imágenes con texto o detalles moderados"
    },
    "4x": {
        "name": "RealESRGAN_x4plus",
        "path": MODELS_DIR / "RealESRGAN_x4plus.pth",
        "scale": 4,
        "description": "Modelo 4x - Calidad superior para fotografías reales"
    },
    "4x_anime": {
        "name": "RealESRGAN_x4plus_anime_6B",
        "path": MODELS_DIR / "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "description": "Modelo 4x - Optimizado para anime e ilustraciones"
    }
}

# URLs de descarga de modelos
MODEL_URLS = {
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "GFPGANv1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
}

# Configuración de GFPGAN
GFPGAN_MODEL_PATH = MODELS_DIR / "GFPGANv1.3.pth"

# Configuración de uploads
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB (reducido para evitar problemas de memoria)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}

# Configuración del servidor
HOST = "127.0.0.1"
PORT = 8000
RELOAD = False  # Cambiar a True solo en desarrollo

# Configuración de procesamiento (OPTIMIZADO PARA EVITAR ERRORES DE MEMORIA)
USE_GPU = True  # Se detectará automáticamente si hay GPU disponible
TILE_SIZE = 400  # Procesar por bloques de 400x400 para evitar consumo excesivo de RAM
TILE_PAD = 10  # Padding para tiles
PRE_PAD = 0  # Pre-padding
HALF_PRECISION = False  # Usar FP16 para ahorrar memoria (requiere GPU)

# Configuración de limpieza automática
AUTO_CLEANUP = True  # Limpiar archivos antiguos automáticamente
CLEANUP_AFTER_HOURS = 24  # Eliminar archivos después de X horas

# Información del proyecto
PROJECT_NAME = "Real-ESRGAN Upscaling Profesional"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Danny Maaz"
PROJECT_GITHUB = "https://github.com/dannymaaz"
PROJECT_DESCRIPTION = "Aplicación profesional para escalar imágenes usando Real-ESRGAN"
