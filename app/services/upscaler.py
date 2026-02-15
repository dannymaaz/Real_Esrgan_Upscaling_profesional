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
        face_enhance: bool = False,
        resize_factor: float = 1.0,
        processing_profile: Optional[Dict] = None,
        face_fidelity: str = "balanced"
    ) -> Dict:
        """
        Escala una imagen usando Real-ESRGAN
        
        Args:
            input_path: Ruta de la imagen de entrada
            output_path: Ruta donde guardar la imagen escalada
            model_key: Modelo a usar ('2x', '4x', '4x_anime')
            face_enhance: Si se debe mejorar rostros (requiere GFPGAN)
            resize_factor: Factor para redimensionar la salida (ej: 0.5 para reducir a la mitad)
            processing_profile: Perfil de calidad detectado en análisis previo
            face_fidelity: 'high' para preservar rasgos, 'balanced' para más detalle
            
        Returns:
            Diccionario con información del procesamiento
            
        Raises:
            FileNotFoundError: Si la imagen de entrada no existe
            Exception: Si hay error en el procesamiento
        """
        # Limpiar memoria GPU si es posible
        if self.device == 'cuda':
            torch.cuda.empty_cache()

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
        total_pixels = original_width * original_height

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
        
        # Estimar uso de memoria (aproximado) con más margen
        estimated_memory_mb = (original_width * original_height * 3 * scale * scale * 4) / (1024 * 1024) # *4 bytes (float32)
        if estimated_memory_mb > 3000:  # ~3GB
            raise ValueError(
                f"Esta imagen requiere demasiada memoria (~{estimated_memory_mb:.0f}MB). "
                f"Por favor, usa una imagen más pequeña o una escala menor."
            )
        
        if processing_profile is None:
            processing_profile = {}

        # Procesar upscaling
        try:
            if face_enhance and HAS_GFPGAN and GFPGAN_MODEL_PATH.exists():
                # Cargar GFPGAN con el upsampler actual como background upsampler
                face_enhancer = self._load_face_enhancer(
                    scale=MODELS[model_key]["scale"],
                    bg_upsampler=upsampler
                )
                
                if face_enhancer is not None:
                    # weight: controla el balance entre fidelidad (preservar original) y mejora
                    _, _, output = face_enhancer.enhance(
                        img, 
                        has_aligned=False, 
                        only_center_face=False, 
                        paste_back=True,
                        weight=0.25 if face_fidelity == "high" else 0.5
                    )
                else:
                    # Fallback si falla la carga de GFPGAN
                    output, _ = upsampler.enhance(img, outscale=MODELS[model_key]["scale"])
            else:
                # Procesamiento estándar sin GFPGAN
                output, _ = upsampler.enhance(img, outscale=MODELS[model_key]["scale"])
                
        except MemoryError:
            raise Exception("Error de Memoria: La imagen es demasiado grande para procesarla. Intenta cerrando otros programas o usando una imagen más pequeña.")
        except Exception as e:
            raise Exception(f"Error al procesar la imagen: {str(e)}")
        
        # Redimensionar si es necesario (para mejorar calidad usando modelo x4 pero output x2)
        if resize_factor != 1.0:
            new_width = int(output.shape[1] * resize_factor)
            new_height = int(output.shape[0] * resize_factor)
            output = cv2.resize(output, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Post-proceso adaptativo anti artefactos
        # Desactivar si la imagen es muy grande (>8MP) para evitar OOM
        postprocess_applied = False
        if processing_profile.get("apply_restoration") and (output.shape[0] * output.shape[1]) < 8_000_000:
            try:
                output = self._apply_adaptive_artifact_reduction(output, processing_profile)
                output = self._apply_region_aware_sharpen(output, processing_profile)
                postprocess_applied = True
            except (MemoryError, cv2.error, ValueError, TypeError) as e:
                print(f"Advertencia: Saltando post-proceso por error: {str(e)}")

        # Guardar resultado
        try:
            cv2.imwrite(str(output_path), output)
        except Exception as e:
             raise Exception(f"No se pudo guardar la imagen procesada: {str(e)}")
        
        # Obtener dimensiones finales
        final_height, final_width = output.shape[:2]
        
        return {
            "success": True,
            "model_used": MODELS[model_key]["name"],
            "scale": MODELS[model_key]["scale"] * resize_factor,
            "original_size": {
                "width": original_width,
                "height": original_height
            },
            "output_size": {
                "width": final_width,
                "height": final_height
            },
            "device_used": self.device,
            "face_enhance": face_enhance,
            "postprocess_applied": postprocess_applied
        }

    def _apply_adaptive_artifact_reduction(self, img: np.ndarray, profile: Dict) -> np.ndarray:
        """
        Reduce ruido/artefactos de compresión.
        NOTA: Desactivado temporalmente para evitar suavizado excesivo y problemas de memoria.
        Real-ESRGAN ya realiza una restauración suficiente.
        """
        return img

    def _apply_region_aware_sharpen(self, img: np.ndarray, profile: Dict) -> np.ndarray:
        """Sharpen por regiones: menos en piel/contornos, más en ropa/fondo."""
        img = self._ensure_bgr_uint8(img)
        blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
        # addWeighted output depth matches src1 detph, usually uint8 here
        unsharp = cv2.addWeighted(img, 1.55, blurred, -0.55, 0)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 80, 180)
        # kernel for dilate
        kernel = np.ones((3, 3), np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        
        # Convertir a float32 solo lo necesario
        edge_mask = (edge_map.astype(np.float32) / 255.0)

        skin_mask = self._build_skin_mask(img)

        blur_severity = profile.get("blur_severity", "low")
        if blur_severity == "strong":
            base_amount = 0.65
        elif blur_severity == "medium":
            base_amount = 0.5
        else:
            base_amount = 0.35

        sharpen_strength = np.full(gray.shape, base_amount, dtype=np.float32)

        # Menos sharpen en piel (pies/manos/cara)
        sharpen_strength = np.where(skin_mask > 0.2, 0.16, sharpen_strength)
        # En contornos fuertes limitar para evitar halos/artefactos
        sharpen_strength = np.where(edge_mask > 0.2, np.minimum(sharpen_strength, 0.28), sharpen_strength)

        sharpen_strength_3c = np.repeat(sharpen_strength[:, :, None], 3, axis=2)
        
        # Cálculo final en float32
        blended = img.astype(np.float32) * (1.0 - sharpen_strength_3c) + unsharp.astype(np.float32) * sharpen_strength_3c

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _build_skin_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Genera máscara aproximada de piel en rango [0, 1] para evitar oversharpen en rostros/manos.
        Usa espacio YCrCb por estabilidad en iluminación.
        """
        img = self._ensure_bgr_uint8(img)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask.astype(np.float32) / 255.0

    def _ensure_bgr_uint8(self, img) -> np.ndarray:
        """
        Normaliza imágenes para OpenCV: ndarray contiguo uint8 en formato BGR (3 canales).
        """
        if img is None:
            raise ValueError("Imagen vacia (None) para procesamiento OpenCV")

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.size == 0:
            raise ValueError("Imagen vacia para procesamiento OpenCV")

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] != 3:
                raise ValueError(f"Numero de canales no soportado: {img.shape[2]}")
        else:
            raise ValueError(f"Dimensiones de imagen no soportadas: {img.ndim}")

        return np.ascontiguousarray(img)
    
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
