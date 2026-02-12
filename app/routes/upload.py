"""
Rutas de API para upload y procesamiento de imágenes
Autor: Danny Maaz (github.com/dannymaaz)
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Optional

from app.config import UPLOADS_DIR, OUTPUTS_DIR, MODELS
from app.utils.file_handler import (
    save_upload_file,
    get_output_filename,
    cleanup_old_files,
    delete_file,
    get_file_size_mb
)
from app.services.image_analyzer import ImageAnalyzer
from app.services.upscaler import RealESRGANUpscaler

# Crear router
router = APIRouter(prefix="/api", tags=["upload"])

# Instancias de servicios (singleton)
analyzer = ImageAnalyzer()
upscaler = RealESRGANUpscaler()


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analiza una imagen y proporciona recomendaciones de upscaling
    
    Args:
        file: Archivo de imagen a analizar
        
    Returns:
        Análisis de la imagen con recomendaciones
    """
    try:
        # Guardar archivo temporalmente
        filename, file_path = await save_upload_file(file)
        
        # Analizar imagen
        analysis = analyzer.analyze_image(file_path)
        
        # Agregar información del archivo
        analysis["filename"] = filename
        analysis["file_size_mb"] = round(get_file_size_mb(file_path), 2)
        
        # Agregar información de modelos disponibles
        analysis["available_models"] = upscaler.get_available_models()
        
        return JSONResponse(content=analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar imagen: {str(e)}")


@router.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: str = Form(...),
    model: Optional[str] = Form(None),
    face_enhance: bool = Form(False)
):
    """
    Procesa upscaling de una imagen
    
    Args:
        file: Archivo de imagen a procesar
        scale: Escala deseada ('2x' o '4x')
        model: Modelo específico a usar
        face_enhance: Activar mejora de rostros (GFPGAN)
        
    Returns:
        Información del procesamiento y nombre del archivo de salida
    """
    try:
        # Guardar archivo de entrada
        input_filename, input_path = await save_upload_file(file)
        
        # Determinar modelo a usar
        if model and model in MODELS:
            model_key = model
        else:
            # Auto-seleccionar modelo basado en análisis
            # Nota: Esto se ejecuta solo si el frontend no envió un modelo específico
            # que coincida con la escala seleccionada
            analysis = analyzer.analyze_image(input_path)
            
            # Si se especificó escala, usarla; si no, usar recomendación
            if scale == "2x":
                model_key = "2x"
            elif scale == "4x":
                # Usar modelo recomendado del análisis
                model_key = analysis.get("recommended_model", "4x")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Escala no válida: {scale}. Use '2x' o '4x'"
                )
        
        # Generar nombre de archivo de salida
        # Incluir indicador de face_enhance en el nombre
        suffix = "_face_enhanced" if face_enhance else ""
        output_filename = get_output_filename(
            input_filename,
            MODELS[model_key]["scale"],
            MODELS[model_key]["name"] + suffix
        )
        output_path = OUTPUTS_DIR / output_filename
        
        # Procesar upscaling
        result = upscaler.upscale_image(
            input_path=input_path,
            output_path=output_path,
            model_key=model_key,
            face_enhance=face_enhance
        )
        
        # Agregar información adicional
        result["input_filename"] = input_filename
        result["output_filename"] = output_filename
        result["output_size_mb"] = round(get_file_size_mb(output_path), 2)
        
        # Limpiar archivos antiguos en segundo plano
        cleanup_old_files(UPLOADS_DIR)
        cleanup_old_files(OUTPUTS_DIR)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar imagen: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_result(filename: str):
    """
    Descarga una imagen procesada
    
    Args:
        filename: Nombre del archivo a descargar
        
    Returns:
        Archivo de imagen
    """
    file_path = OUTPUTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Archivo no encontrado"
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@router.get("/models")
async def get_models():
    """
    Obtiene información sobre los modelos disponibles
    
    Returns:
        Información de modelos
    """
    return JSONResponse(content=upscaler.get_available_models())


@router.delete("/cleanup")
async def cleanup_files():
    """
    Limpia archivos temporales antiguos
    
    Returns:
        Número de archivos eliminados
    """
    uploads_deleted = cleanup_old_files(UPLOADS_DIR, hours=1)
    outputs_deleted = cleanup_old_files(OUTPUTS_DIR, hours=1)
    
    return JSONResponse(content={
        "uploads_deleted": uploads_deleted,
        "outputs_deleted": outputs_deleted,
        "total_deleted": uploads_deleted + outputs_deleted
    })
