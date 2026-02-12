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
    GFPGAN_MODEL_PATH,
    USE_GPU,
    TILE_SIZE,
    TILE_PAD,
    PRE_PAD,
    HALF_PRECISION
)

try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    HAS_GFPGAN = False
    print("Advertencia: gfpgan no instalado, mejora de rostros desactivada")


class RealESRGANUpscaler:
    """Servicio de upscaling usando Real-ESRGAN"""
    
    def __init__(self):
        """Inicializa el servicio de upscaling"""
        self.models: Dict[str, RealESRGANer] = {}
        self.face_enhancer = None
        self.device = self._detect_device()
        
    def _load_face_enhancer(self, scale: int, bg_upsampler=None):
        """
        Carga el modelo GFPGAN para mejora de rostros
        
        Args:
            scale: Escala de upscaling
            bg_upsampler: Upsampler de fondo (RealESRGAN)
        """
        if not HAS_GFPGAN:
            return None
            
        if not GFPGAN_MODEL_PATH.exists():
            return None
            
        # Si ya está cargado con la misma configuración, usarlo
        # Nota: GFPGANer es ligero, podemos recargarlo si es necesario o mantener una instancia
        # Por simplicidad y memoria, mantendremos una única instancia
        
        if self.face_enhancer is None:
            self.face_enhancer = GFPGANer(
                model_path=str(GFPGAN_MODEL_PATH),
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
                device=self.device
            )
        else:
            # Actualizar upsampler de fondo si cambió
            self.face_enhancer.bg_upsampler = bg_upsampler
            self.face_enhancer.upscale = scale
            
        return self.face_enhancer
        
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
        
        # Leer archivo como bytes primero para decodificar
        try:
            with open(input_path, "rb") as f:
                content = f.read()
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            raise ValueError(f"No se pudo leer la imagen: {input_path}. Error: {str(e)}")
            
        if img is None:
            raise ValueError(f"No se pudo decodificar la imagen: {input_path}")
        
        # === CORRECCIÓN DE CANALES PARA GFPGAN/REALESRGAN ===
        # Asegurar formato BGR (3 canales)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:  # Grayscale (H, W, 1)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            # Convertir a BGR, descartando alpha para evitar problemas
            # (Si se requiere transparencia, se necesita manejo más complejo)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        original_height, original_width = img.shape[:2]
        
        # VALIDACIÓN DE DIMENSIONES PARA EVITAR ERRORES DE MEMORIA
        max_dimension = 4000  # Límite máximo por dimensión
        scale = MODELS[model_key]["scale"]
        
        # Verificar dimensiones originales
        if original_width > max_dimension or original_height > max_dimension:
            raise ValueError(
                f"Imagen demasiado grande ({original_width}x{original_height}). "
                f"Dimensión máxima permitida: {max_dimension}px. "
                f"Por favor, reduce el tamaño de la imagen antes de procesarla."
            )
        
        # Verificar dimensiones después del escalado
        output_width = original_width * scale
        output_height = original_height * scale
        
        if output_width > max_dimension * 2 or output_height > max_dimension * 2:
            raise ValueError(
                f"La imagen escalada sería demasiado grande ({output_width}x{output_height}). "
                f"Considera usar una escala menor (2x en lugar de 4x)."
            )
        
        # Estimar uso de memoria (aproximado)
        estimated_memory_mb = (original_width * original_height * 3 * scale * scale) / (1024 * 1024)
        if estimated_memory_mb > 2000:  # Más de 2GB
            raise ValueError(
                f"Esta imagen requiere aproximadamente {estimated_memory_mb:.0f}MB de RAM. "
                f"Por favor, usa una imagen más pequeña o una escala menor."
            )
        
        # Procesar upscaling
        try:
            if face_enhance and HAS_GFPGAN and GFPGAN_MODEL_PATH.exists():
                # Cargar GFPGAN con el upsampler actual como background upsampler
                face_enhancer = self._load_face_enhancer(
                    scale=MODELS[model_key]["scale"],
                    bg_upsampler=upsampler
                )
                
                if face_enhancer is not None:
                    # Procesar con mejora de rostros
                    # enhance devuelve: cropped_faces, restored_faces, restored_img
                    _, _, output = face_enhancer.enhance(
                        img, 
                        has_aligned=False, 
                        only_center_face=False, 
                        paste_back=True
                    )
                else:
                    # Fallback si falla la carga de GFPGAN
                    output, _ = upsampler.enhance(img, outscale=MODELS[model_key]["scale"])
            else:
                # Procesamiento estándar sin GFPGAN
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
        self.face_enhancer = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()
