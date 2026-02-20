"""
Servicio de upscaling con Real-ESRGAN
Maneja la carga de modelos y el procesamiento de imágenes
Autor: Danny Maaz (github.com/dannymaaz)
"""

import cv2
import torch
import numpy as np
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
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

    def _compute_face_weight(self, face_fidelity: str, profile: Optional[Dict] = None) -> float:
        """Calcula peso de GFPGAN priorizando naturalidad e identidad."""
        profile = profile or {}
        blur_severity = profile.get("blur_severity", "low")
        pixelation_score = float(profile.get("pixelation_score", 0.0))
        compression_score = float(profile.get("compression_score", 0.0))

        # Base conservadora para evitar "cara nueva".
        if face_fidelity == "ultra":
            weight = 0.10
        elif face_fidelity == "high":
            weight = 0.12
        else:
            weight = 0.14

        severe_degradation = (
            blur_severity == "strong"
            and (pixelation_score > 0.3 or compression_score > 0.5)
        )
        if severe_degradation:
            weight = min(0.2, weight + 0.04)

        # Ajuste por condiciones de luz (en baja luz, GFPGAN tiende a verse muy artificial/pegado)
        lighting = profile.get("lighting_condition", "normal")
        if lighting == "low_light":
            weight *= 0.7  # Reducir peso en 30%

        if self._is_tone_lock_profile(profile):
            weight *= 0.72

        if bool(profile.get("story_overlay_detected", False)):
            weight *= 0.76
             
        return float(np.clip(weight, 0.04, 0.18))

    def _merge_face_enhancement(
        self,
        base_img: np.ndarray,
        enhanced_img: np.ndarray,
        face_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        """
        Mezcla conservadora de GFPGAN restringida a zonas de rostro.
        Evita alterar ropa/fondo para mantener resultado natural.
        """
        base_img = self._ensure_bgr_uint8(base_img)
        enhanced_img = self._ensure_bgr_uint8(enhanced_img)

        if base_img.shape != enhanced_img.shape:
            return enhanced_img

        # Si no logramos ubicar rostros, no aplicar mezcla global.
        if not face_boxes:
            return base_img

        # Usar solo detalle de luminancia de GFPGAN, manteniendo color base.
        base_lab = cv2.cvtColor(base_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        enh_lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        enh_l = enh_lab[:, :, 0]
        blur_l = cv2.GaussianBlur(enh_l, (0, 0), 1.2)
        detail_l = np.clip(enh_l - blur_l, -18.0, 18.0)

        diff = cv2.absdiff(base_img, enhanced_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, changed = cv2.threshold(diff_gray, 6, 255, cv2.THRESH_BINARY)
        changed = cv2.dilate(changed, np.ones((5, 5), np.uint8), iterations=1)

        h, w = diff_gray.shape
        face_mask = np.zeros((h, w), dtype=np.uint8)
        skin_mask = self._build_skin_mask(base_img)
        for x, y, fw, fh in face_boxes:
            if fw <= 0 or fh <= 0:
                continue
            aspect = fw / float(max(1, fh))
            if aspect < 0.6 or aspect > 1.7:
                continue
            box_area = (fw * fh) / float(max(1, w * h))
            if box_area < 0.002 or box_area > 0.45:
                continue

            shrink_x = int(fw * 0.08)
            shrink_y = int(fh * 0.12)
            x1 = max(0, x + shrink_x)
            y1 = max(0, y + shrink_y)
            x2 = min(w, x + fw - shrink_x)
            y2 = min(h, y + fh - shrink_y)
            if x2 <= x1 or y2 <= y1:
                continue

            skin_ratio = float(np.mean(skin_mask[y1:y2, x1:x2]))
            if skin_ratio < 0.05:
                continue

            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            axes_x = max(6, int((x2 - x1) * 0.5))
            axes_y = max(6, int((y2 - y1) * 0.58))
            cv2.ellipse(face_mask, (cx, cy), (axes_x, axes_y), 0, 0, 360, 255, -1)

        if not np.any(face_mask):
            return base_img

        face_mask = cv2.GaussianBlur(face_mask, (0, 0), 4.5).astype(np.float32) / 255.0
        changed_soft = cv2.GaussianBlur(changed, (0, 0), 2.0).astype(np.float32) / 255.0

        detail_mask = np.clip(changed_soft * 1.2, 0.0, 1.0)
        tone_mask = np.clip(skin_mask * 0.9 + detail_mask * 0.35, 0.0, 1.0)
        alpha = np.clip(face_mask * tone_mask, 0.0, 1.0) * 0.6

        merged_lab = base_lab.copy()
        merged_l = merged_lab[:, :, 0] + (detail_l * alpha)
        merged_lab[:, :, 0] = np.clip(merged_l, 0, 255)
        merged = cv2.cvtColor(merged_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return merged

    def _detect_face_boxes(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta cajas de rostro para limitar mezcla de GFPGAN."""
        img = self._ensure_bgr_uint8(img)

        if self.face_cascade.empty():
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_face = max(24, int(min(h, w) * 0.04))

        detections: List[Tuple[int, int, int, int]] = []

        primary = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_face, min_face)
        )
        detections.extend([(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in primary])

        relaxed = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(max(20, int(min_face * 0.85)), max(20, int(min_face * 0.85)))
        )
        detections.extend([(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in relaxed])

        if not self.profile_face_cascade.empty():
            profile = self.profile_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.06,
                minNeighbors=5,
                minSize=(max(20, int(min_face * 0.85)), max(20, int(min_face * 0.85)))
            )
            detections.extend([(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in profile])

        return self._deduplicate_boxes(detections)

    def _deduplicate_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Elimina detecciones casi duplicadas usando IoU."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        selected: List[Tuple[int, int, int, int]] = []

        for candidate in boxes:
            keep = True
            for chosen in selected:
                if self._box_iou(candidate, chosen) > 0.35:
                    keep = False
                    break
            if keep:
                selected.append(candidate)

        return selected

    def _box_iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        """Calcula IoU entre dos bounding boxes (x, y, w, h)."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        return inter_area / max(1e-6, (area_a + area_b - inter_area))
        
    def _detect_device(self) -> str:
        """
        Detecta si hay GPU disponible
        
        Returns:
            'cuda' si hay GPU, 'cpu' en caso contrario
        """
        if USE_GPU and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _is_portrait_sensitive_profile(self, profile: Dict) -> bool:
        """Perfiles donde priorizamos textura natural de piel frente a intervención agresiva."""
        key = str(profile.get("repair_profile", ""))
        return key in {
            "social_portrait_rescue",
            "lowlight_portrait_natural",
            "portrait_texture_guard",
            "social_story_natural",
            "clean_portrait_tone_lock",
            "clean_photo_soft"
        }

    def _is_heavy_artifact_profile(self, profile: Dict) -> bool:
        """Perfiles de recuperación fuerte por artefactos/degradación."""
        key = str(profile.get("repair_profile", ""))
        return key in {
            "artifact_rescue_general",
            "old_scan_repair"
        }

    def _is_tone_lock_profile(self, profile: Dict) -> bool:
        key = str(profile.get("repair_profile", ""))
        return key in {
            "clean_portrait_tone_lock",
            "social_story_natural",
            "social_color_balance",
            "clean_photo_soft"
        }
    
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

    def _configure_runtime_tiling(self, upsampler: RealESRGANer, total_pixels: int):
        """
        Ajusta tile dinámicamente para evitar OOM y permitir más fotos en 4x.
        """
        runtime_tile = TILE_SIZE

        if self.device == 'cpu':
            if total_pixels > 10_000_000:
                runtime_tile = min(runtime_tile, 160)
            elif total_pixels > 6_000_000:
                runtime_tile = min(runtime_tile, 192)
            elif total_pixels > 3_000_000:
                runtime_tile = min(runtime_tile, 256)
        else:
            if total_pixels > 18_000_000:
                runtime_tile = min(runtime_tile, 160)
            elif total_pixels > 10_000_000:
                runtime_tile = min(runtime_tile, 192)
            elif total_pixels > 5_000_000:
                runtime_tile = min(runtime_tile, 256)

        try:
            setattr(upsampler, "tile", runtime_tile)
        except Exception:
            pass

    def _compute_safe_preresize_factor(
        self,
        width: int,
        height: int,
        scale: int,
        max_input_dimension: int,
        memory_limit_mb: float
    ) -> float:
        """Calcula factor de pre-redimensionado para modo seguro (sin perder demasiado detalle)."""
        factor = 1.0

        max_side = max(width, height)
        if max_side > max_input_dimension:
            factor = min(factor, max_input_dimension / max_side)

        estimated_memory_mb = (width * height * 3 * scale * scale * 4) / (1024 * 1024)
        safe_target_mb = memory_limit_mb * 0.82
        if estimated_memory_mb > safe_target_mb and safe_target_mb > 0:
            memory_factor = float(np.sqrt(safe_target_mb / max(1e-6, estimated_memory_mb)))
            factor = min(factor, memory_factor)

        # Evitar reducción extrema para preservar calidad.
        return float(np.clip(factor, 0.58, 1.0))

    def _apply_filter_reduction(self, img: np.ndarray, profile: Dict) -> Tuple[np.ndarray, bool]:
        """
        Reduce dominantes de filtro (saturación/curva) para recuperar aspecto natural.
        """
        apply_color_filter_correction = bool(profile.get("remove_color_filter", profile.get("remove_filter", False)))
        apply_old_photo_restore = bool(profile.get("restore_old_photo", False))

        if not apply_color_filter_correction and not apply_old_photo_restore:
            return img, False

        img = self._ensure_bgr_uint8(img)
        filter_strength = profile.get("social_filter_strength", profile.get("filter_strength", "medium"))
        tone_lock_profile = self._is_tone_lock_profile(profile)

        working = img.copy()
        applied = False

        if apply_color_filter_correction:
            if filter_strength == "high":
                saturation_factor = 0.8
                wb_blend = 0.44
            elif filter_strength == "low":
                saturation_factor = 0.91
                wb_blend = 0.24
            else:
                saturation_factor = 0.86
                wb_blend = 0.33

            if tone_lock_profile:
                saturation_factor = min(1.0, saturation_factor + 0.08)
                wb_blend *= 0.82

            hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            filter_reduced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Balance de blancos suave (gray-world) para neutralizar cast.
            b, g, r = cv2.split(filter_reduced.astype(np.float32))
            mean_b = float(np.mean(b))
            mean_g = float(np.mean(g))
            mean_r = float(np.mean(r))
            gray_mean = max(1.0, (mean_b + mean_g + mean_r) / 3.0)
            b *= gray_mean / max(1.0, mean_b)
            g *= gray_mean / max(1.0, mean_g)
            r *= gray_mean / max(1.0, mean_r)
            balanced = cv2.merge([
                np.clip(b, 0, 255),
                np.clip(g, 0, 255),
                np.clip(r, 0, 255)
            ]).astype(np.uint8)

            # Mezcla para conservar esencia original.
            working = cv2.addWeighted(filter_reduced, 1.0 - wb_blend, balanced, wb_blend, 0)

            # Proteger tonos de piel para evitar desaturación artificial.
            skin_mask = self._build_skin_mask(img)
            if float(np.mean(skin_mask)) > 0.02:
                preserve_ratio = 0.48 if tone_lock_profile else 0.36
                skin_alpha = cv2.GaussianBlur(skin_mask.astype(np.float32), (0, 0), 1.4) * preserve_ratio
                skin_alpha_3c = np.repeat(skin_alpha[:, :, None], 3, axis=2)
                working = img.astype(np.float32) * skin_alpha_3c + working.astype(np.float32) * (1.0 - skin_alpha_3c)
                working = np.clip(working, 0, 255).astype(np.uint8)
            applied = True

        if apply_old_photo_restore:
            lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
            l, a, bch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.65, tileGridSize=(8, 8))
            l = clahe.apply(l)
            working = cv2.cvtColor(cv2.merge([l, a, bch]), cv2.COLOR_LAB2BGR)

            scratch_score = float(profile.get("scratch_score", 0.0))
            if scratch_score > 0.18:
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
                blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
                scratches = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
                _, scratch_mask = cv2.threshold(scratches, 18, 255, cv2.THRESH_BINARY)
                scratch_mask = cv2.dilate(scratch_mask, np.ones((2, 2), np.uint8), iterations=1)
                working = cv2.inpaint(working, scratch_mask, 2, cv2.INPAINT_TELEA)

            applied = True

        return working, applied

    def _restore_monochrome_photo(self, img: np.ndarray, profile: Dict) -> Tuple[np.ndarray, bool]:
        """
        Restauración experimental para imágenes B/N: recupera color de forma conservadora.
        """
        if not (profile.get("restore_monochrome") and profile.get("is_monochrome")):
            return img, False

        img = self._ensure_bgr_uint8(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        # Si existe mínima croma residual, amplificarla suavemente.
        residual_chroma = float(np.mean(np.abs(a_channel - 128)) + np.mean(np.abs(b_channel - 128)))
        if residual_chroma > 4.0:
            a_boost = 128 + (a_channel - 128) * 2.4
            b_boost = 128 + (b_channel - 128) * 2.4
            recolored_lab = np.stack([
                l_channel,
                np.clip(a_boost, 90, 170),
                np.clip(b_boost, 85, 180)
            ], axis=2).astype(np.uint8)
            recolored = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2BGR)
        else:
            # Fallback: colorización tonal muy suave para evitar resultado artificial.
            l_norm = np.clip(l_channel / 255.0, 0.0, 1.0)
            a_tint = 128 + (l_norm - 0.5) * 5 + 1.8
            b_tint = 128 + (l_norm - 0.5) * 9 + 4.5

            # Si detectamos rostro, calentar la zona para piel más natural.
            face_boxes = self._detect_face_boxes(img)
            if face_boxes:
                face_mask = np.zeros(l_channel.shape, dtype=np.float32)
                h, w = l_channel.shape
                for x, y, fw, fh in face_boxes:
                    ex = int(fw * 0.28)
                    ey = int(fh * 0.3)
                    x1 = max(0, x - ex)
                    y1 = max(0, y - ey)
                    x2 = min(w, x + fw + ex)
                    y2 = min(h, y + fh + ey)
                    cv2.rectangle(face_mask, (x1, y1), (x2, y2), 1.0, -1)

                face_mask = cv2.GaussianBlur(face_mask, (0, 0), 5.5)
                a_tint = a_tint + face_mask * 6.5
                b_tint = b_tint + face_mask * 11.0

            toned_lab = np.stack([
                l_channel,
                np.clip(a_tint, 112, 146),
                np.clip(b_tint, 112, 162)
            ], axis=2).astype(np.uint8)
            recolored = cv2.cvtColor(toned_lab, cv2.COLOR_LAB2BGR)

        recolored = cv2.addWeighted(img, 0.5, recolored, 0.5, 0)
        return recolored, True

    def _apply_preprocess_options(self, img: np.ndarray, profile: Dict) -> Tuple[np.ndarray, Dict]:
        """Aplica restauraciones opcionales previas al escalado."""
        metadata = {
            "filter_restoration_applied": False,
            "old_photo_restoration_applied": False,
            "bw_restoration_applied": False
        }

        working = self._ensure_bgr_uint8(img)

        requested_color_restore = bool(profile.get("remove_color_filter", profile.get("remove_filter", False)))
        requested_old_photo_restore = bool(profile.get("restore_old_photo", False))

        working, filter_applied = self._apply_filter_reduction(working, profile)
        metadata["filter_restoration_applied"] = bool(filter_applied and requested_color_restore)
        metadata["old_photo_restoration_applied"] = bool(filter_applied and requested_old_photo_restore)

        working, bw_applied = self._restore_monochrome_photo(working, profile)
        metadata["bw_restoration_applied"] = bool(bw_applied)

        return working, metadata
    
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
            
        if processing_profile is None:
            processing_profile = {}

        source_height, source_width = img.shape[:2]
        scale = MODELS[model_key]["scale"]

        target_final_width = max(1, int(round(source_width * scale * resize_factor)))
        target_final_height = max(1, int(round(source_height * scale * resize_factor)))

        # Restauraciones opcionales antes del modelo (filtro/BN).
        img, preprocess_metadata = self._apply_preprocess_options(img, processing_profile)

        # VALIDACIÓN DE DIMENSIONES PARA EVITAR ERRORES DE MEMORIA
        max_input_dimension = 6200 if self.device == 'cuda' else 4800
        max_output_dimension = 12400 if self.device == 'cuda' else 9600
        max_output_pixels = 140_000_000 if self.device == 'cuda' else 75_000_000
        memory_limit_mb = 7000 if self.device == 'cuda' else 4500

        pre_resize_applied = False
        pre_resize_factor = 1.0
        if processing_profile.get("safe_pre_resize", False):
            pre_resize_factor = self._compute_safe_preresize_factor(
                width=source_width,
                height=source_height,
                scale=scale,
                max_input_dimension=max_input_dimension,
                memory_limit_mb=memory_limit_mb
            )

            if pre_resize_factor < 0.995:
                safe_w = max(1, int(round(source_width * pre_resize_factor)))
                safe_h = max(1, int(round(source_height * pre_resize_factor)))
                img = cv2.resize(img, (safe_w, safe_h), interpolation=cv2.INTER_AREA)
                pre_resize_applied = True

        original_height, original_width = img.shape[:2]
        total_pixels = original_width * original_height

        # Verificar dimensiones de trabajo
        if original_width > max_input_dimension or original_height > max_input_dimension:
            raise ValueError(
                f"Imagen demasiado grande ({original_width}x{original_height}). "
                f"Dimensión máxima permitida: {max_input_dimension}px. "
                f"Por favor, reduce el tamaño de la imagen antes de procesarla."
            )

        output_width = original_width * scale
        output_height = original_height * scale
        if (
            output_width > max_output_dimension
            or output_height > max_output_dimension
            or (output_width * output_height) > max_output_pixels
        ):
            raise ValueError(
                f"La imagen escalada sería demasiado grande ({output_width}x{output_height}). "
                f"Límite actual aproximado: {max_output_dimension}px por lado. "
                f"Considera usar una escala menor (2x en lugar de 4x)."
            )

        estimated_memory_mb = (original_width * original_height * 3 * scale * scale * 4) / (1024 * 1024)
        if estimated_memory_mb > memory_limit_mb:
            raise ValueError(
                f"Esta imagen requiere demasiada memoria (~{estimated_memory_mb:.0f}MB). "
                f"Por favor, usa una imagen más pequeña o una escala menor."
            )

        face_enhance_effective = bool(face_enhance)
        face_enhance_skipped = False
        auto_face_enhance = False

        # Activar mejora facial automática cuando se detectan rostros relevantes
        if not face_enhance_effective:
            if processing_profile.get("has_faces", False) and processing_profile.get("face_importance") in {"medium", "high"}:
                face_enhance_effective = True
                auto_face_enhance = True

        if face_enhance_effective and self.device == 'cpu':
            projected_output_pixels = int(original_width * original_height * scale * scale)
            if projected_output_pixels > 20_000_000:
                face_enhance_effective = False
                face_enhance_skipped = True

        # Ajuste dinámico de tile para mayor estabilidad y permitir más casos en 4x.
        self._configure_runtime_tiling(upsampler, total_pixels)

        # Procesar upscaling
        try:
            if face_enhance_effective and HAS_GFPGAN and GFPGAN_MODEL_PATH.exists():
                # 1) Upscaling base con Real-ESRGAN.
                output, _ = upsampler.enhance(img, outscale=MODELS[model_key]["scale"])

                # 2) Mejora facial localizada sobre la imagen ya escalada.
                face_enhancer = self._load_face_enhancer(
                    scale=1,
                    bg_upsampler=None
                )
                if face_enhancer is not None:
                    face_weight = self._compute_face_weight(face_fidelity, processing_profile)
                    _, _, face_output = face_enhancer.enhance(
                        output,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                        weight=face_weight
                    )
                    if face_output is not None:
                        face_boxes = self._detect_face_boxes(output)
                        output = self._merge_face_enhancement(output, face_output, face_boxes)
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
            if resize_factor > 1.0:
                resize_interpolation = cv2.INTER_CUBIC
            else:
                resize_interpolation = cv2.INTER_AREA
            output = cv2.resize(output, (new_width, new_height), interpolation=resize_interpolation)

        # Si hubo pre-redimensionado de seguridad, devolver al tamaño final objetivo.
        safe_mode_output_downscaled = False
        if pre_resize_applied:
            if output.shape[1] != target_final_width or output.shape[0] != target_final_height:
                try:
                    output = cv2.resize(output, (target_final_width, target_final_height), interpolation=cv2.INTER_LANCZOS4)
                except (cv2.error, MemoryError):
                    safe_mode_output_downscaled = True

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

        detail_boost_applied = False
        try:
            output, detail_boost_applied = self._apply_photo_detail_boost(output, processing_profile)
        except (MemoryError, cv2.error, ValueError, TypeError) as e:
            print(f"Advertencia: Saltando detalle global por error: {str(e)}")

        skin_texture_applied = False
        try:
            output, skin_texture_applied = self._restore_skin_microtexture(output, processing_profile)
        except (MemoryError, cv2.error, ValueError, TypeError) as e:
            print(f"Advertencia: Saltando microtextura de piel por error: {str(e)}")

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
                "width": source_width,
                "height": source_height
            },
            "output_size": {
                "width": final_width,
                "height": final_height
            },
            "device_used": self.device,
            "face_enhance": face_enhance_effective,
            "auto_face_enhance_enabled": bool(auto_face_enhance and not face_enhance_skipped),
            "face_enhance_skipped": face_enhance_skipped,
            "postprocess_applied": postprocess_applied,
            "detail_boost_applied": detail_boost_applied,
            "skin_texture_applied": skin_texture_applied,
            "safe_pre_resize_applied": pre_resize_applied,
            "safe_pre_resize_factor": round(pre_resize_factor, 3),
            "safe_mode_output_downscaled": safe_mode_output_downscaled,
            "filter_restoration_applied": preprocess_metadata.get("filter_restoration_applied", False),
            "old_photo_restoration_applied": preprocess_metadata.get("old_photo_restoration_applied", False),
            "bw_restoration_applied": preprocess_metadata.get("bw_restoration_applied", False)
        }

    def _apply_photo_detail_boost(self, img: np.ndarray, profile: Dict) -> Tuple[np.ndarray, bool]:
        """
        Aumenta detalle global de forma natural en fotos reales, incluso sin GFPGAN.
        """
        image_type = profile.get("image_type", "photo")
        if image_type not in {"photo", "filtered_photo"}:
            return img, False

        img = self._ensure_bgr_uint8(img)

        noise_level = profile.get("noise_level", "low")
        compression_score = float(profile.get("compression_score", 0.0))
        blur_severity = profile.get("blur_severity", "low")
        lighting = profile.get("lighting_condition", "normal")
        has_relevant_faces = bool(profile.get("has_faces", False)) and profile.get("face_importance") in {"medium", "high"}
        profile_key = str(profile.get("repair_profile", "balanced_photo"))
        tone_lock_profile = self._is_tone_lock_profile(profile)

        if image_type == "filtered_photo":
            base_amount = 0.12
        elif blur_severity == "strong":
            base_amount = 0.14
        else:
            base_amount = 0.1

        if noise_level == "high":
            base_amount *= 0.65
        elif noise_level == "medium":
            base_amount *= 0.82

        if compression_score > 0.65:
            base_amount *= 0.7
        elif compression_score > 0.45:
            base_amount *= 0.85

        if lighting == "low_light":
            base_amount *= 0.75

        if bool(profile.get("degraded_social_portrait", False)):
            base_amount *= 0.72

        if profile_key in {"social_portrait_rescue", "portrait_texture_guard"}:
            base_amount *= 0.7

        if profile_key == "lowlight_portrait_natural":
            base_amount *= 0.74

        if profile_key in {"artifact_rescue_general", "old_scan_repair"}:
            base_amount *= 0.86

        if tone_lock_profile:
            base_amount *= 0.68

        if bool(profile.get("story_overlay_detected", False)):
            base_amount *= 0.72

        if has_relevant_faces and compression_score > 0.55:
            base_amount *= 0.74

        if compression_score > 0.72 and noise_level in {"medium", "high"}:
            base_amount *= 0.55

        detail_amount = float(np.clip(base_amount, 0.0, 0.2))
        if detail_amount < 0.035:
            return img, False

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=1.12 if tone_lock_profile else (1.28 if image_type == "filtered_photo" else 1.2),
            tileGridSize=(8, 8)
        )
        l_contrast = clahe.apply(l_channel)
        contrast_img = cv2.cvtColor(
            cv2.merge([l_contrast, a_channel, b_channel]),
            cv2.COLOR_LAB2BGR
        )

        blur = cv2.GaussianBlur(contrast_img, (0, 0), 1.05)
        if tone_lock_profile:
            sharpened = cv2.addWeighted(contrast_img, 1.05, blur, -0.05, 0)
        else:
            sharpened = cv2.addWeighted(contrast_img, 1.09, blur, -0.09, 0)

        skin_mask = self._build_skin_mask(img)
        detail_map = np.full(skin_mask.shape, detail_amount, dtype=np.float32)
        detail_map = np.where(skin_mask > 0.2, detail_map * 0.78, detail_map)
        detail_map = cv2.GaussianBlur(detail_map, (0, 0), 1.0)
        detail_map_3c = np.repeat(detail_map[:, :, None], 3, axis=2)

        boosted = img.astype(np.float32) * (1.0 - detail_map_3c) + sharpened.astype(np.float32) * detail_map_3c
        return np.clip(boosted, 0, 255).astype(np.uint8), True

    def _restore_skin_microtexture(self, img: np.ndarray, profile: Dict) -> Tuple[np.ndarray, bool]:
        """
        Recupera microtextura suave en piel para reducir apariencia plástica en retratos.
        """
        image_type = profile.get("image_type", "photo")
        if image_type not in {"photo", "filtered_photo"}:
            return img, False

        compression_score = float(profile.get("compression_score", 0.0))
        noise_level = profile.get("noise_level", "low")
        if compression_score > 0.78 and noise_level == "high":
            return img, False

        img = self._ensure_bgr_uint8(img)
        skin_mask = self._build_skin_mask(img)
        if float(np.mean(skin_mask)) < 0.03:
            return img, False
        skin_ratio = float(np.mean(skin_mask > 0.2))
        has_relevant_faces = bool(profile.get("has_faces", False)) and profile.get("face_importance") in {"medium", "high"}
        if not has_relevant_faces and skin_ratio < 0.1:
            return img, False

        high_pass = img.astype(np.float32) - cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.15)
        strength = 0.12 if image_type == "photo" else 0.09
        if compression_score > 0.55:
            strength *= 0.75
        if noise_level in {"medium", "high"}:
            strength *= 0.8

        if 0.45 <= compression_score <= 0.75 and skin_ratio > 0.1:
            strength *= 1.16

        if compression_score > 0.78:
            strength *= 0.85

        if self._is_portrait_sensitive_profile(profile):
            strength *= 1.12

        skin_strength = cv2.GaussianBlur(skin_mask.astype(np.float32), (0, 0), 1.2) * strength
        skin_strength_3c = np.repeat(skin_strength[:, :, None], 3, axis=2)
        textured = img.astype(np.float32) + high_pass * skin_strength_3c
        return np.clip(textured, 0, 255).astype(np.uint8), True

    def _apply_adaptive_artifact_reduction(self, img: np.ndarray, profile: Dict) -> np.ndarray:
        """
        Reduce ruido/artefactos de compresión de forma conservadora para evitar piel plástica.
        """
        img = self._ensure_bgr_uint8(img)

        noise_level = profile.get("noise_level", "low")
        compression_score = float(profile.get("compression_score", 0.0))
        blur_severity = profile.get("blur_severity", "low")
        uniform_restore_mode = bool(profile.get("uniform_restore_mode", False))
        portrait_sensitive_profile = self._is_portrait_sensitive_profile(profile)
        tone_lock_profile = self._is_tone_lock_profile(profile)
        heavy_artifact_profile = self._is_heavy_artifact_profile(profile)

        denoise_strength = 0
        if noise_level == "high":
            denoise_strength = 5
        elif noise_level == "medium":
            denoise_strength = 3

        if compression_score > 0.8:
            denoise_strength = max(denoise_strength, 5)
        elif compression_score > 0.45:
            denoise_strength = max(denoise_strength, 3)

        if uniform_restore_mode:
            denoise_strength = max(denoise_strength, 5)

        if heavy_artifact_profile:
            denoise_strength = max(denoise_strength, 3)

        # En fotos no severas, evitar suavizado global para preservar textura natural.
        has_relevant_faces = bool(profile.get("has_faces", False)) and profile.get("face_importance") in {"medium", "high"}
        skin_ratio_hint = 0.0
        try:
            skin_ratio_hint = float(np.mean(self._build_skin_mask(img) > 0.2))
        except Exception:
            skin_ratio_hint = 0.0
        has_skin_dominant = skin_ratio_hint > 0.11
        if (
            not uniform_restore_mode
            and blur_severity != "strong"
            and compression_score < (0.72 if has_relevant_faces else (0.66 if has_skin_dominant else 0.6))
            and float(profile.get("pixelation_score", 0.0)) < 0.35
        ):
            return img

        # En retratos relativamente limpios, denoise mínimo para evitar piel plástica.
        if has_relevant_faces and compression_score < 0.76 and blur_severity != "strong":
            denoise_strength = min(denoise_strength, 2)

        if has_skin_dominant and compression_score < 0.72 and blur_severity != "strong":
            denoise_strength = min(denoise_strength, 2)

        if has_relevant_faces and skin_ratio_hint > 0.12 and blur_severity != "strong":
            denoise_strength = min(denoise_strength, 2)

        if bool(profile.get("degraded_social_portrait", False)) and has_relevant_faces:
            denoise_strength = min(denoise_strength, 2)

        if portrait_sensitive_profile and has_relevant_faces:
            denoise_strength = min(denoise_strength, 2)

        if tone_lock_profile:
            denoise_strength = min(denoise_strength, 1)

        if compression_score < 0.6 and blur_severity != "strong":
            denoise_strength = min(denoise_strength, 2)

        lighting = profile.get("lighting_condition", "normal")
        if lighting == "low_light":
            denoise_strength *= 0.6  # Reducir denoise en noche para evitar "plástico"
            
        if denoise_strength < 1:
            return img

        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, denoise_strength, denoise_strength, 7, 21
        )

        if uniform_restore_mode:
            # Suaviza transiciones de bloques en fotos severamente degradadas.
            denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=28, sigmaSpace=28)

        # Mantener textura natural en piel al mezclar con la imagen original.
        if uniform_restore_mode and blur_severity == "strong":
            keep_original = 0.4
        elif uniform_restore_mode:
            keep_original = 0.48
        elif blur_severity == "strong":
            keep_original = 0.5
        elif blur_severity == "medium":
            keep_original = 0.58
        else:
            keep_original = 0.68

        if (has_relevant_faces or has_skin_dominant) and skin_ratio_hint > 0.1:
            keep_original = min(0.82, keep_original + 0.1)

        if bool(profile.get("degraded_social_portrait", False)) and has_relevant_faces:
            keep_original = min(0.84, max(keep_original, 0.74))

        if portrait_sensitive_profile and has_relevant_faces:
            keep_original = min(0.86, max(keep_original, 0.76))

        if tone_lock_profile:
            keep_original = min(0.9, max(keep_original, 0.8))

        if compression_score < 0.6 and blur_severity != "strong":
            keep_original = max(keep_original, 0.8)

        blended = cv2.addWeighted(img, keep_original, denoised, 1.0 - keep_original, 0)
        
        # Recuperar textura
        restored = self._reinject_texture(img, blended, profile)
        
        # Paso final anti-plástico: Inyección de micro-grano natural
        # Solo si se aplicó denoise fuerte o es imagen de alta calidad que quedó muy lisa
        if denoise_strength >= 3 or (lighting == "low_light" and denoise_strength >= 2):
            restored = self._add_micro_grain(restored, profile)
             
        return restored

    def _reinject_texture(self, original: np.ndarray, processed: np.ndarray, profile: Dict) -> np.ndarray:
        """Recupera microtextura (ropa/cabello) tras denoise para evitar acabado plástico."""
        original = self._ensure_bgr_uint8(original)
        processed = self._ensure_bgr_uint8(processed)

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = cv2.normalize(grad_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

        skin_mask = self._build_skin_mask(original)
        texture_mask = np.clip((grad_norm - 0.12) * 1.6, 0.0, 1.0)
        # Mantener una pequeña porción de microtextura en piel para evitar acabado plástico.
        texture_mask *= (1.0 - np.clip(skin_mask, 0.0, 1.0) * 0.78)
        texture_mask = cv2.GaussianBlur(texture_mask, (0, 0), 1.2)

        blur_severity = profile.get("blur_severity", "low")
        uniform_restore_mode = bool(profile.get("uniform_restore_mode", False))
        if uniform_restore_mode:
            texture_strength = 0.08
        elif blur_severity == "strong":
            texture_strength = 0.06
        elif blur_severity == "medium":
            texture_strength = 0.12
        else:
            texture_strength = 0.14

        high_pass = original.astype(np.float32) - cv2.GaussianBlur(original.astype(np.float32), (0, 0), 1.05)
        texture_gain = high_pass * (texture_strength * texture_mask[:, :, None])
        restored = processed.astype(np.float32) + texture_gain
        return np.clip(restored, 0, 255).astype(np.uint8)

    def _apply_region_aware_sharpen(self, img: np.ndarray, profile: Dict) -> np.ndarray:
        """Sharpen por regiones: menos en piel/contornos, más en ropa/fondo."""
        img = self._ensure_bgr_uint8(img)
        blurred = cv2.GaussianBlur(img, (0, 0), 0.8)
        # addWeighted output depth matches src1 detph, usually uint8 here
        unsharp = cv2.addWeighted(img, 1.25, blurred, -0.25, 0)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 80, 180)
        # kernel for dilate
        kernel = np.ones((3, 3), np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        
        # Convertir a float32 solo lo necesario
        edge_mask = (edge_map.astype(np.float32) / 255.0)

        skin_mask = self._build_skin_mask(img)

        blur_severity = profile.get("blur_severity", "low")
        uniform_restore_mode = bool(profile.get("uniform_restore_mode", False))
        portrait_sensitive_profile = self._is_portrait_sensitive_profile(profile)
        tone_lock_profile = self._is_tone_lock_profile(profile)
        if blur_severity == "strong":
            base_amount = 0.24
        elif blur_severity == "medium":
            base_amount = 0.18
        else:
            base_amount = 0.12

        if uniform_restore_mode:
            if blur_severity == "strong":
                base_amount = 0.14
            elif blur_severity == "medium":
                base_amount = 0.1
            else:
                base_amount = 0.08

            sharpen_strength = np.full(gray.shape, base_amount, dtype=np.float32)
            # Evitar halos en bordes duros manteniendo el acabado uniforme.
            sharpen_strength = np.where(edge_mask > 0.2, np.minimum(sharpen_strength, base_amount * 0.8), sharpen_strength)
        else:
            sharpen_strength = np.full(gray.shape, base_amount, dtype=np.float32)

            # Menos sharpen en piel (pies/manos/cara)
            sharpen_strength = np.where(skin_mask > 0.2, 0.06, sharpen_strength)
            # En contornos fuertes limitar para evitar halos/artefactos
            sharpen_strength = np.where(edge_mask > 0.2, np.minimum(sharpen_strength, 0.14), sharpen_strength)

        compression_score = float(profile.get("compression_score", 0.0))
        noise_level = profile.get("noise_level", "low")
        if compression_score > 0.7 or noise_level == "high":
            sharpen_strength *= 0.7

        if bool(profile.get("degraded_social_portrait", False)) and bool(profile.get("has_faces", False)):
            sharpen_strength *= 0.82
            sharpen_strength = np.where(skin_mask > 0.2, np.minimum(sharpen_strength, 0.05), sharpen_strength)

        if portrait_sensitive_profile:
            sharpen_strength *= 0.85
            sharpen_strength = np.where(skin_mask > 0.2, np.minimum(sharpen_strength, 0.05), sharpen_strength)

        if tone_lock_profile:
            sharpen_strength *= 0.78
            sharpen_strength = np.where(skin_mask > 0.2, np.minimum(sharpen_strength, 0.045), sharpen_strength)

        if bool(profile.get("story_overlay_detected", False)):
            sharpen_strength *= 0.86

        sharpen_strength_3c = np.repeat(sharpen_strength[:, :, None], 3, axis=2)
        
        # Cálculo final en float32
        blended = img.astype(np.float32) * (1.0 - sharpen_strength_3c) + unsharp.astype(np.float32) * sharpen_strength_3c

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _add_micro_grain(self, img: np.ndarray, profile: Dict) -> np.ndarray:
        """
        Añade ruido gaussiano sutil (grano de película) para eliminar el efecto plástico.
        """
        h, w, c = img.shape
        lighting = profile.get("lighting_condition", "normal")
        if self._is_tone_lock_profile(profile):
            return img
        
        # Base de fuerza de grano
        grain_strength = 2.0
        
        if lighting == "low_light":
            grain_strength = 3.5  # Más grano en noche es natural
        elif profile.get("source_info", {}).get("is_likely_social_media"):
            grain_strength = 2.5
            
        # Generar ruido
        noise = np.random.normal(0, grain_strength, (h, w, c)).astype(np.float32)
        
        # Añadir a la imagen
        noisy_img = img.astype(np.float32) + noise
        
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

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
        
        # Dilatar para cubrir bordes de manos/dedos que suelen sufrir artifacts
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
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
