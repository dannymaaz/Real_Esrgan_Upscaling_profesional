"""
Utilidades para manejo de archivos
Autor: Danny Maaz (github.com/dannymaaz)
"""

import os
import uuid
import aiofiles
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException

from app.config import (
    UPLOADS_DIR,
    OUTPUTS_DIR,
    ALLOWED_EXTENSIONS,
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
    # Validar extensión
    if not is_allowed_file(upload_file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Formato de archivo no permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generar nombre único
    filename = generate_unique_filename(upload_file.filename)
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
    
    return filename, file_path


def get_output_filename(input_filename: str, scale: int, model_type: str = "") -> str:
    """
    Genera el nombre del archivo de salida basado en el archivo de entrada
    
    Args:
        input_filename: Nombre del archivo de entrada
        scale: Factor de escala (2, 4, etc.)
        model_type: Tipo de modelo usado (opcional)
        
    Returns:
        Nombre del archivo de salida
    """
    stem = Path(input_filename).stem
    extension = get_file_extension(input_filename)
    
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
