"""
Rutas de API para upload y procesamiento de imágenes
Autor: Danny Maaz (github.com/dannymaaz)
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Optional
from time import perf_counter
from threading import Lock
from starlette.concurrency import run_in_threadpool

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
upscale_lock = Lock()


def _run_upscale_locked(*, input_path: Path, output_path: Path, model_key: str, face_enhance: bool, resize_factor: float, processing_profile: dict, face_fidelity: str):
    """Ejecuta upscaling bajo lock para evitar sobrecarga concurrente en CPU/GPU."""
    with upscale_lock:
        return upscaler.upscale_image(
            input_path=input_path,
            output_path=output_path,
            model_key=model_key,
            face_enhance=face_enhance,
            resize_factor=resize_factor,
            processing_profile=processing_profile,
            face_fidelity=face_fidelity
        )

def _normalize_model_key(model: Optional[str]) -> Optional[str]:
    """Acepta clave interna (2x/4x/4x_anime) o nombre del modelo RealESRGAN."""
    if not model:
        return None

    if model in MODELS:
        return model

    for key, info in MODELS.items():
        if model == info.get("name"):
            return key

    return None


def _normalize_image_type_override(image_type: Optional[str]) -> Optional[str]:
    """Normaliza override manual de tipo de imagen enviado por UI."""
    if not image_type:
        return None

    normalized = image_type.strip().lower()
    if normalized in {"photo", "anime", "illustration", "filtered_photo"}:
        return normalized
    return None


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
        
        # Analizar imagen en threadpool para no bloquear event loop
        analysis = await run_in_threadpool(analyzer.analyze_image, file_path)
        
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
    face_enhance: bool = Form(False),
    forced_image_type: Optional[str] = Form(None)
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
        
        # Analizar imagen para obtener dimensiones y tipo
        analysis = await run_in_threadpool(analyzer.analyze_image, input_path)
        original_width = analysis.get("width", 0)
        original_height = analysis.get("height", 0)
        analyzed_image_type = analysis.get("image_type", "photo")
        normalized_forced_type = _normalize_image_type_override(forced_image_type)

        if forced_image_type and normalized_forced_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de imagen no válido para override: {forced_image_type}"
            )

        effective_image_type = normalized_forced_type or analyzed_image_type
        
        if scale not in {"2x", "4x"}:
            raise HTTPException(
                status_code=400,
                detail=f"Escala no válida: {scale}. Use '2x' o '4x'"
            )

        requested_scale_value = 2 if scale == "2x" else 4
        model_key_from_request = _normalize_model_key(model)
        auto_model_key = "4x_anime" if effective_image_type == "anime" else "4x"
        model_key = model_key_from_request or auto_model_key

        # En CPU, para solicitudes 2x con imágenes muy grandes, usar 2x nativo.
        limit_for_4x_side = 9000 if upscaler.device == "cpu" else 11800
        use_2x_native_for_2x = (
            scale == "2x"
            and MODELS[model_key]["scale"] == 4
            and (
                original_width * 4 > limit_for_4x_side
                or original_height * 4 > limit_for_4x_side
            )
        )
        if use_2x_native_for_2x:
            model_key = "2x"

        model_scale = MODELS[model_key]["scale"]
        resize_factor = requested_scale_value / model_scale
        effective_face_enhance = face_enhance
        cpu_fallback_note = None

        # Perfil de restauración para anti artefactos y sharpen por región
        processing_profile = {
            "apply_restoration": analysis.get("apply_restoration", False),
            "uniform_restore_mode": analysis.get("uniform_restore_mode", False),
            "noise_level": analysis.get("noise_level", "low"),
            "compression_score": analysis.get("compression_score", 0.0),
            "pixelation_score": analysis.get("pixelation_score", 0.0),
            "blur_severity": analysis.get("blur_severity", "low"),
            "lighting_condition": analysis.get("lighting_condition", "normal"),
            "image_type": effective_image_type
        }

        # Evitar modificación excesiva del rostro en fotos de buena calidad.
        severe_face_degradation = (
            analysis.get("blur_severity") == "strong"
            and (
                analysis.get("pixelation_score", 0.0) > 0.32
                or analysis.get("compression_score", 0.0) > 0.52
            )
        )
        face_fidelity_mode = "balanced" if severe_face_degradation else "ultra"

        # Plan de ejecución con fallback a 2x + resize cuando 4x falla por memoria/estabilidad.
        attempt_plan = [{
            "model_key": model_key,
            "resize_factor": resize_factor,
            "warning": None
        }]
        if requested_scale_value == 4 and MODELS[model_key]["scale"] == 4:
            attempt_plan.append({
                "model_key": "2x",
                "resize_factor": 2.0,
                "warning": "Fallback automático: se usó RealESRGAN_x2plus y redimensionado a 4x por límite de recursos."
            })

        # Evitar intentos duplicados
        unique_attempts = []
        seen = set()
        for attempt in attempt_plan:
            key = (attempt["model_key"], float(attempt["resize_factor"]))
            if key not in seen:
                seen.add(key)
                unique_attempts.append(attempt)

        # Procesar upscaling
        started_at = perf_counter()
        result = None
        output_filename = ""
        output_path = OUTPUTS_DIR / ""
        last_error = None

        for index, attempt in enumerate(unique_attempts):
            attempt_model_key = attempt["model_key"]
            attempt_resize = float(attempt["resize_factor"])
            attempt_warning = attempt["warning"]

            suffix = "_face_enhanced" if effective_face_enhance else ""
            if attempt_resize != 1.0:
                suffix += f"_resized_{scale}"
            if attempt_warning:
                suffix += "_fallback"

            output_filename = get_output_filename(
                input_filename,
                MODELS[attempt_model_key]["scale"] if attempt_resize == 1.0 else int(MODELS[attempt_model_key]["scale"] * attempt_resize),
                MODELS[attempt_model_key]["name"] + suffix
            )
            output_path = OUTPUTS_DIR / output_filename

            try:
                result = await run_in_threadpool(
                    _run_upscale_locked,
                    input_path=input_path,
                    output_path=output_path,
                    model_key=attempt_model_key,
                    face_enhance=effective_face_enhance,
                    resize_factor=attempt_resize,
                    processing_profile=processing_profile,
                    face_fidelity=face_fidelity_mode
                )
                if attempt_warning:
                    cpu_fallback_note = attempt_warning
                break
            except Exception as attempt_error:
                last_error = attempt_error
                if index == len(unique_attempts) - 1:
                    raise

        if result is None:
            raise HTTPException(
                status_code=500,
                detail=f"Error al procesar imagen: {str(last_error) if last_error else 'falló el procesamiento'}"
            )

        elapsed_seconds = perf_counter() - started_at
        
        # Agregar información adicional
        result["input_filename"] = input_filename
        result["output_filename"] = output_filename
        result["output_size_mb"] = round(get_file_size_mb(output_path), 2)
        result["processing_time_seconds"] = round(elapsed_seconds, 2)
        result["analysis_image_type"] = analyzed_image_type
        result["effective_image_type"] = effective_image_type
        result["type_overridden"] = bool(normalized_forced_type and normalized_forced_type != analyzed_image_type)
        if cpu_fallback_note:
            result["processing_warning"] = cpu_fallback_note
        
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
