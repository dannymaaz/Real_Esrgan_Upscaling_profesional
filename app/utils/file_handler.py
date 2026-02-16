"""
Utilidades para manejo de archivos
Autor: Danny Maaz (github.com/dannymaaz)
"""

import os
import uuid
import aiofiles
import importlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException
from PIL import Image

try:
    pillow_heif = importlib.import_module("pillow_heif")
    pillow_heif.register_heif_opener()
    HAS_HEIF_SUPPORT = True
except Exception:
    HAS_HEIF_SUPPORT = False

from app.config import (
    UPLOADS_DIR,
    OUTPUTS_DIR,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_UPLOAD_SIZE,
    AUTO_CLEANUP,
    CLEANUP_AFTER_HOURS
)


def get_file_extension(filename: str) -> str:
    """
    Obtiene la extensión del archivo en minúsculas
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        Extensión del archivo (ej: '.png')
    """
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str) -> bool:
    """
    Verifica si el archivo tiene una extensión permitida
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        True si la extensión está permitida, False en caso contrario
    """
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def generate_unique_filename(original_filename: str) -> str:
    """
    Genera un nombre de archivo único manteniendo la extensión original
    
    Args:
        original_filename: Nombre original del archivo
        
    Returns:
        Nombre único del archivo
    """
    extension = get_file_extension(original_filename)
    unique_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{unique_id}{extension}"


def _convert_heic_to_png(file_path: Path) -> Tuple[str, Path]:
    """Convierte HEIC/HEIF a PNG para compatibilidad de pipeline."""
    extension = get_file_extension(file_path.name)
    if extension not in {".heic", ".heif"}:
        return file_path.name, file_path

    if not HAS_HEIF_SUPPORT:
        raise HTTPException(
            status_code=400,
            detail=(
                "El formato HEIC/HEIF requiere soporte adicional. "
                "Instala dependencias ejecutando: pip install pillow-heif"
            )
        )

    png_filename = f"{file_path.stem}.png"
    png_path = file_path.with_suffix(".png")

    try:
        with Image.open(file_path) as src:
            rgb = src.convert("RGB")
            rgb.save(png_path, format="PNG", optimize=True)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo convertir archivo HEIC/HEIF: {str(e)}"
        )

    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

    return png_filename, png_path


async def save_upload_file(upload_file: UploadFile) -> Tuple[str, Path]:
    """
    Guarda un archivo subido en el directorio de uploads
    
    Args:
        upload_file: Archivo subido por el usuario
        
    Returns:
        Tupla con (nombre_archivo, ruta_completa)
        
    Raises:
        HTTPException: Si el archivo no es válido o excede el tamaño máximo
    """
    supported_formats_text = ", ".join(ext.replace(".", "").upper() for ext in sorted(ALLOWED_EXTENSIONS))
    unsupported_message = (
        f"Este archivo no es compatible por ahora. "
        f"Formatos soportados: {supported_formats_text}."
    )

    original_filename = (upload_file.filename or "upload_image").strip() or "upload_image"

    # Validar extensión
    if not is_allowed_file(original_filename):
        raise HTTPException(
            status_code=400,
            detail=unsupported_message
        )

    # Validar MIME cuando está disponible
    content_type = (upload_file.content_type or "").lower().strip()
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=unsupported_message
        )
    
    # Generar nombre único
    filename = generate_unique_filename(original_filename)
    file_path = UPLOADS_DIR / filename
    
    # Guardar archivo con validación de tamaño
    total_size = 0
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await upload_file.read(8192):  # Leer en chunks de 8KB
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    # Eliminar archivo parcial
                    await f.close()
                    if file_path.exists():
                        file_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"Archivo demasiado grande. Máximo: {MAX_UPLOAD_SIZE / (1024*1024):.1f}MB"
                    )
                await f.write(chunk)
    except Exception as e:
        # Limpiar en caso de error
        if file_path.exists():
            file_path.unlink()
        raise
    
    # Convertir HEIC/HEIF a PNG para asegurar compatibilidad total con OpenCV/Real-ESRGAN.
    filename, file_path = _convert_heic_to_png(file_path)

    return filename, file_path


def get_output_filename(input_filename: str, scale: int, model_type: str = "", output_extension: Optional[str] = None) -> str:
    """
    Genera el nombre del archivo de salida basado en el archivo de entrada
    
    Args:
        input_filename: Nombre del archivo de entrada
        scale: Factor de escala (2, 4, etc.)
        model_type: Tipo de modelo usado (opcional)
        output_extension: Extensión de salida forzada (opcional, ej: '.png')
        
    Returns:
        Nombre del archivo de salida
    """
    stem = Path(input_filename).stem
    extension = (output_extension or get_file_extension(input_filename)).lower()
    if not extension.startswith('.'):
        extension = f".{extension}"
    
    # Agregar sufijo con escala y tipo de modelo
    suffix = f"_x{scale}"
    if model_type:
        suffix += f"_{model_type}"
    
    return f"{stem}{suffix}{extension}"


def cleanup_old_files(directory: Path, hours: int = CLEANUP_AFTER_HOURS) -> int:
    """
    Elimina archivos antiguos de un directorio
    
    Args:
        directory: Directorio a limpiar
        hours: Eliminar archivos más antiguos que X horas
        
    Returns:
        Número de archivos eliminados
    """
    if not AUTO_CLEANUP:
        return 0
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    deleted_count = 0
    
    for file_path in directory.glob("*"):
        if file_path.is_file():
            # Obtener tiempo de modificación
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception:
                    pass  # Ignorar errores al eliminar
    
    return deleted_count


def delete_file(file_path: Path) -> bool:
    """
    Elimina un archivo de forma segura
    
    Args:
        file_path: Ruta del archivo a eliminar
        
    Returns:
        True si se eliminó correctamente, False en caso contrario
    """
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return True
    except Exception:
        pass
    return False


def get_file_size_mb(file_path: Path) -> float:
    """
    Obtiene el tamaño de un archivo en MB
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Tamaño en MB
    """
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0
