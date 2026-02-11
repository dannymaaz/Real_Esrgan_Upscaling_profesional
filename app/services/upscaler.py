"""
Servicio de upscaling con Real-ESRGAN
Maneja la carga de modelos y el procesamiento de imágenes
Autor: Danny Maaz (github.com/dannymaaz)
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from app.config import (
    MODELS,
    MODELS_DIR,
    USE_GPU,
    TILE_SIZE,
    TILE_PAD,
    PRE_PAD,
    HALF_PRECISION
)


class RealESRGANUpscaler:
    """Servicio de upscaling usando Real-ESRGAN"""
    
    def __init__(self):
        """Inicializa el servicio de upscaling"""
        self.models: Dict[str, RealESRGANer] = {}
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """
        Detecta si hay GPU disponible
        
        Returns:
            'cuda' si hay GPU, 'cpu' en caso contrario
        """
        if USE_GPU and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def _get_model_arch(self, model_key: str):
        """
        Obtiene la arquitectura del modelo según la clave
        
        Args:
            model_key: Clave del modelo ('2x', '4x', '4x_anime')
            
        Returns:
            Arquitectura del modelo
        """
        if model_key == "2x":
            # RealESRGAN_x2plus usa RRDBNet
            return RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2
            )
        elif model_key == "4x_anime":
            # RealESRGAN_x4plus_anime_6B usa RRDBNet con 6 bloques
            return RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4
            )
        else:  # 4x
            # RealESRGAN_x4plus usa RRDBNet estándar
            return RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
    
    def load_model(self, model_key: str) -> RealESRGANer:
        """
        Carga un modelo Real-ESRGAN (lazy loading)
        
        Args:
            model_key: Clave del modelo a cargar ('2x', '4x', '4x_anime')
            
        Returns:
            Instancia de RealESRGANer
            
        Raises:
            FileNotFoundError: Si el modelo no existe
            ValueError: Si la clave del modelo no es válida
        """
        # Si el modelo ya está cargado, devolverlo
        if model_key in self.models:
            return self.models[model_key]
        
        # Validar clave del modelo
        if model_key not in MODELS:
            raise ValueError(f"Modelo '{model_key}' no válido. Use: {list(MODELS.keys())}")
        
        model_info = MODELS[model_key]
        model_path = model_info["path"]
        
        # Verificar que el modelo existe
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                f"Ejecute 'python download_models.py' para descargar los modelos."
            )
        
        # Obtener arquitectura del modelo
        model_arch = self._get_model_arch(model_key)
        
        # Configurar el upsampler
        upsampler = RealESRGANer(
            scale=model_info["scale"],
            model_path=str(model_path),
            model=model_arch,
            tile=TILE_SIZE,
            tile_pad=TILE_PAD,
            pre_pad=PRE_PAD,
            half=HALF_PRECISION and self.device == 'cuda',
            device=self.device
        )
        
        # Guardar en caché
        self.models[model_key] = upsampler
        
        return upsampler
    
    def upscale_image(
        self, 
        input_path: Path, 
        output_path: Path, 
        model_key: str = "4x",
        face_enhance: bool = False
    ) -> Dict:
        """
        Escala una imagen usando Real-ESRGAN
        
        Args:
            input_path: Ruta de la imagen de entrada
            output_path: Ruta donde guardar la imagen escalada
            model_key: Modelo a usar ('2x', '4x', '4x_anime')
            face_enhance: Si se debe mejorar rostros (requiere GFPGAN)
            
        Returns:
            Diccionario con información del procesamiento
            
        Raises:
            FileNotFoundError: Si la imagen de entrada no existe
            Exception: Si hay error en el procesamiento
        """
        # Verificar que la imagen existe
        if not input_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {input_path}")
        
        # Cargar modelo
        upsampler = self.load_model(model_key)
        
        # Leer imagen
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {input_path}")
        
        original_height, original_width = img.shape[:2]
        
        # Procesar upscaling
        try:
            output, _ = upsampler.enhance(img, outscale=MODELS[model_key]["scale"])
        except Exception as e:
            raise Exception(f"Error al procesar la imagen: {str(e)}")
        
        # Guardar resultado
        cv2.imwrite(str(output_path), output)
        
        # Obtener dimensiones finales
        final_height, final_width = output.shape[:2]
        
        return {
            "success": True,
            "model_used": MODELS[model_key]["name"],
            "scale": MODELS[model_key]["scale"],
            "original_size": {
                "width": original_width,
                "height": original_height
            },
            "output_size": {
                "width": final_width,
                "height": final_height
            },
            "device_used": self.device,
            "face_enhance": face_enhance
        }
    
    def get_available_models(self) -> Dict:
        """
        Obtiene información sobre los modelos disponibles
        
        Returns:
            Diccionario con información de modelos
        """
        available = {}
        for key, info in MODELS.items():
            available[key] = {
                "name": info["name"],
                "scale": info["scale"],
                "description": info["description"],
                "available": info["path"].exists(),
                "path": str(info["path"])
            }
        return available
    
    def clear_cache(self):
        """Limpia la caché de modelos cargados para liberar memoria"""
        self.models.clear()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
