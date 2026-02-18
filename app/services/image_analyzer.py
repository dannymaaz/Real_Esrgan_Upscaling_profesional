"""
Analizador inteligente de imágenes
Determina el tipo de imagen y recomienda la mejor configuración de upscaling
Autor: Danny Maaz (github.com/dannymaaz)
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path
from typing import Dict


class ImageAnalyzer:
    """Analizador de imágenes para determinar el mejor modelo y escala"""
    
    def __init__(self):
        """Inicializa el analizador de imágenes"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def analyze_image(self, image_path: Path) -> Dict:
        """
        Analiza una imagen y proporciona recomendaciones de upscaling
        
        Args:
            image_path: Ruta de la imagen a analizar
            
        Returns:
            Diccionario con análisis y recomendaciones
        """
        # Cargar imagen
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("No se pudo cargar la imagen")
        
        # Obtener información básica
        height, width = img.shape[:2]
        total_pixels = height * width
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Análisis de características
        blur_metrics = self._detect_blur(gray)
        sharpness = blur_metrics["laplacian_score"]
        tenengrad = blur_metrics["tenengrad_score"]
        noise_level = self._estimate_noise(img)
        compression_metrics = self._detect_compression_artifacts(gray)
        pixelation_metrics = self._detect_pixelation(gray)
        face_info = self._detect_faces_info(img)
        
        # Análisis profundo de origen y condiciones
        source_info = self._check_source_integrity(image_path)
        lighting_condition = self._analyze_lighting(img)
        restoration_signals = self._analyze_restoration_signals(
            img=img,
            gray=gray,
            compression_metrics=compression_metrics,
            pixelation_metrics=pixelation_metrics,
            noise_level=noise_level,
            source_info=source_info,
            blur_metrics=blur_metrics,
            face_info=face_info
        )
        
        image_type = self._detect_image_type(
            img=img,
            blur_metrics=blur_metrics,
            compression_metrics=compression_metrics,
            pixelation_metrics=pixelation_metrics,
            noise_level=noise_level,
            face_info=face_info
        )

        repair_profile = self._determine_repair_profile(
            image_type=image_type,
            blur_metrics=blur_metrics,
            noise_level=noise_level,
            compression_metrics=compression_metrics,
            pixelation_metrics=pixelation_metrics,
            face_info=face_info,
            lighting_condition=lighting_condition,
            restoration_signals=restoration_signals,
            source_info=source_info
        )
        # Para decisiones de UX/procesamiento, solo contar rostros relevantes.
        has_faces = face_info["has_faces"] and face_info["importance"] in {"medium", "high"}
        
        # Determinar escala recomendada
        recommended_scale = self._recommend_scale(
            width=width,
            height=height,
            image_type=image_type,
            sharpness=sharpness,
            noise_level=noise_level,
            compression_metrics=compression_metrics,
            pixelation_metrics=pixelation_metrics,
            blur_metrics=blur_metrics
        )
        
        # Determinar modelo recomendado
        recommended_model = self._recommend_model(image_type, has_faces)
        
        return {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "megapixels": round(total_pixels / 1_000_000, 2),
            "image_type": image_type,
            "sharpness_score": round(sharpness, 2),
            "tenengrad_score": round(tenengrad, 2),
            "is_blurry": blur_metrics["is_blurry"],
            "blur_severity": blur_metrics["blur_severity"],
            "noise_level": noise_level,
            "has_compression_artifacts": compression_metrics["has_compression_artifacts"],
            "compression_score": round(compression_metrics["compression_score"], 3),
            "is_pixelated": pixelation_metrics["is_pixelated"],
            "pixelation_score": round(pixelation_metrics["pixelation_score"], 3),
            "has_faces": has_faces,
            "face_count": face_info.get("count", 0),
            "face_importance": face_info["importance"],
            "source_info": source_info,
            "lighting_condition": lighting_condition,
            "filter_detected": restoration_signals["filter_detected"],
            "filter_strength": restoration_signals["filter_strength"],
            "social_color_filter_detected": restoration_signals["social_color_filter_detected"],
            "social_filter_strength": restoration_signals["social_filter_strength"],
            "degraded_social_portrait": restoration_signals["degraded_social_portrait"],
            "story_overlay_detected": restoration_signals["story_overlay_detected"],
            "is_monochrome": restoration_signals["is_monochrome"],
            "monochrome_confidence": round(restoration_signals["monochrome_confidence"], 3),
            "old_photo_detected": restoration_signals["old_photo_detected"],
            "scan_artifacts_detected": restoration_signals["scan_artifacts_detected"],
            "scratch_score": round(restoration_signals["scratch_score"], 3),
            "recommended_filter_restoration": restoration_signals["recommended_filter_restoration"],
            "recommended_color_filter_correction": restoration_signals["recommended_color_filter_correction"],
            "recommended_old_photo_restore": restoration_signals["recommended_old_photo_restore"],
            "recommended_bw_restore": restoration_signals["recommended_bw_restore"],
            "repair_profile": repair_profile["key"],
            "repair_profile_strength": repair_profile["strength"],
            "repair_profile_reason": repair_profile["reason"],
            "recommended_scale": recommended_scale,
            "recommended_model": recommended_model,
            "uniform_restore_mode": self._should_use_uniform_restore(
                blur_metrics,
                noise_level,
                compression_metrics,
                pixelation_metrics
            ),
            "apply_restoration": self._should_apply_restoration(
                blur_metrics,
                noise_level,
                compression_metrics,
                pixelation_metrics
            ),
            "analysis_notes": self._generate_detailed_notes(
                width,
                height,
                image_type,
                sharpness,
                noise_level,
                has_faces,
                blur_metrics,
                compression_metrics,
                pixelation_metrics,
                source_info,
                lighting_condition,
                restoration_signals,
                repair_profile
            )
        }

    def _check_source_integrity(self, image_path: Path) -> Dict:
        """
        Analiza metadatos y patrones para determinar origen
        """
        try:
            with Image.open(image_path) as pil_img:
                exif = pil_img.getexif()
                has_exif = bool(exif)

                exif_map = {
                    ExifTags.TAGS.get(tag, tag): value
                    for tag, value in (exif.items() if exif else [])
                }

                # Redes sociales suelen usar resoluciones/ratios estándar y software específico.
                w, h = pil_img.size
                social_widths = {720, 960, 1080, 1280, 1600, 1920, 2048}

                software_raw = str(exif_map.get("Software", "") or "").lower()
                make_raw = str(exif_map.get("Make", "") or "").strip()
                model_raw = str(exif_map.get("Model", "") or "").strip()
                date_original = exif_map.get("DateTimeOriginal")
                has_camera_tags = bool(make_raw or model_raw or date_original)

                social_software_tokens = (
                    "instagram", "facebook", "messenger", "whatsapp", "telegram",
                    "snapchat", "tiktok", "xiaohongshu", "weibo", "line"
                )
                software_social_hint = any(token in software_raw for token in social_software_tokens)

                portrait_ratio = h / max(1.0, float(w))
                story_shape_hint = portrait_ratio > 1.6 and (w in {720, 1080, 1440} or h in {1280, 1920, 2560})
                dimension_social_hint = (w in social_widths or h in social_widths)

                is_social = False

                if software_social_hint:
                    is_social = True
                elif dimension_social_hint and (not has_exif or not has_camera_tags):
                    is_social = True
                elif story_shape_hint and not has_camera_tags:
                    is_social = True

                return {
                    "has_exif": has_exif,
                    "is_likely_social_media": is_social,
                    "format": pil_img.format,
                    "software_hint": software_social_hint,
                    "has_camera_tags": has_camera_tags
                }
        except Exception:
            return {"has_exif": False, "is_likely_social_media": True, "format": "UNKNOWN"}

    def _analyze_lighting(self, img: np.ndarray) -> str:
        """Detecta condiciones de iluminación para ajustar denoise"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_luma = np.mean(gray)
        
        if mean_luma < 65:
            return "low_light"  # Poca luz/Noche: Ruido natural, no suavizar excesivamente
        elif mean_luma > 195:
            return "high_key"   # Muy iluminada/Quemada
        else:
            return "normal"

    def _generate_detailed_notes(
        self, width, height, image_type, sharpness, noise_level, has_faces, 
        blur, compression, pixelation, source_info, lighting, restoration_signals=None, repair_profile=None
    ) -> list:
        """Genera notas humanas incluyendo origen y luz"""
        # Llamar al legacy para la base
        notes = self._generate_notes(width, height, image_type, sharpness, noise_level, has_faces, blur, compression, pixelation)
        restoration_signals = restoration_signals or {}
        repair_profile = repair_profile or {"key": "balanced_photo", "strength": "medium", "reason": "Perfil balanceado"}
        
        if source_info["is_likely_social_media"]:
            notes.append("Probable origen de Redes Sociales (Compresión web)")
            
        if lighting == "low_light":
            notes.append("Condiciones de baja luz: Procesamiento conservador de ruido")

        if restoration_signals.get("filter_detected"):
            strength = restoration_signals.get("filter_strength", "medium")
            notes.append(f"Filtro detectado (intensidad {strength})")

        if restoration_signals.get("social_color_filter_detected"):
            strength = restoration_signals.get("social_filter_strength", "medium")
            notes.append(f"Posible filtro de color social/teléfono (intensidad {strength})")

        if restoration_signals.get("degraded_social_portrait"):
            notes.append("Foto social degradada detectada: se sugiere perfil de rescate")

        if restoration_signals.get("story_overlay_detected"):
            notes.append("Captura tipo historia/red social detectada")

        if restoration_signals.get("is_monochrome"):
            notes.append("Imagen monocromática detectada (B/N o desaturada)")

        if restoration_signals.get("old_photo_detected"):
            notes.append("Posible foto antigua: recomendada restauración tonal")

        if restoration_signals.get("scan_artifacts_detected"):
            notes.append("Posibles artefactos de escaneo/rayones detectados")

        notes.append(
            f"Perfil automático: {repair_profile.get('key', 'balanced_photo')} ({repair_profile.get('strength', 'medium')})"
        )
             
        return notes

    def _determine_repair_profile(
        self,
        image_type: str,
        blur_metrics: Dict,
        noise_level: str,
        compression_metrics: Dict,
        pixelation_metrics: Dict,
        face_info: Dict,
        lighting_condition: str,
        restoration_signals: Dict,
        source_info: Dict
    ) -> Dict:
        """
        Define un perfil de reparación especializado para guiar el pipeline.
        """
        blur_severity = blur_metrics.get("blur_severity", "low")
        compression_score = float(compression_metrics.get("compression_score", 0.0))
        pixelation_score = float(pixelation_metrics.get("pixelation_score", 0.0))
        has_relevant_faces = bool(face_info.get("has_faces", False) and face_info.get("importance") in {"medium", "high"})

        if image_type == "anime":
            return {
                "key": "anime_preserve_lines",
                "strength": "medium",
                "reason": "Ilustración/anime detectado"
            }

        if restoration_signals.get("is_monochrome"):
            return {
                "key": "bw_recovery",
                "strength": "high",
                "reason": "Imagen monocromática detectada"
            }

        if restoration_signals.get("old_photo_detected") or restoration_signals.get("scan_artifacts_detected"):
            return {
                "key": "old_scan_repair",
                "strength": "high",
                "reason": "Señales de foto antigua o escaneo"
            }

        if (
            restoration_signals.get("story_overlay_detected", False)
            and (
                compression_score > 0.2
                or pixelation_score > 0.12
                or noise_level in {"medium", "high"}
                or blur_severity in {"medium", "strong"}
            )
        ):
            if compression_score > 0.82 or (pixelation_score > 0.36 and blur_severity == "strong"):
                return {
                    "key": "artifact_rescue_general",
                    "strength": "high",
                    "reason": "Captura tipo historia con artefacto severo"
                }
            return {
                "key": "social_story_natural",
                "strength": "medium",
                "reason": "Captura tipo historia social con overlays"
            }

        if restoration_signals.get("degraded_social_portrait"):
            return {
                "key": "social_portrait_rescue",
                "strength": "high",
                "reason": "Foto social degradada (compresión + cast + blur)"
            }

        if source_info.get("is_likely_social_media", False) and (
            compression_score > 0.56
            or blur_severity == "strong"
            or (
                noise_level == "high"
                and compression_score > 0.45
            )
        ):
            return {
                "key": "artifact_rescue_general",
                "strength": "high",
                "reason": "Compresión/pixelado social severo"
            }

        if restoration_signals.get("social_color_filter_detected"):
            return {
                "key": "social_color_balance",
                "strength": "medium",
                "reason": "Filtro de color social/teléfono detectado"
            }

        if (
            has_relevant_faces
            and image_type in {"photo", "filtered_photo"}
            and blur_severity in {"low", "medium"}
            and noise_level in {"low", "medium"}
            and compression_score < 0.52
            and pixelation_score < 0.5
            and not restoration_signals.get("degraded_social_portrait", False)
            and not restoration_signals.get("old_photo_detected", False)
            and not restoration_signals.get("scan_artifacts_detected", False)
            and not restoration_signals.get("story_overlay_detected", False)
        ):
            return {
                "key": "clean_portrait_tone_lock",
                "strength": "low",
                "reason": "Retrato limpio: priorizar tono y naturalidad"
            }

        if has_relevant_faces and lighting_condition == "low_light" and (
            noise_level in {"medium", "high"}
            or compression_score > 0.3
            or blur_severity in {"medium", "strong"}
        ):
            return {
                "key": "lowlight_portrait_natural",
                "strength": "medium",
                "reason": "Retrato con poca luz y ruido"
            }

        if has_relevant_faces and (
            compression_score > 0.5
            or (pixelation_score > 0.38 and compression_score > 0.3)
            or blur_severity == "strong"
        ):
            return {
                "key": "portrait_texture_guard",
                "strength": "medium",
                "reason": "Retrato comprimido con riesgo de piel plástica"
            }

        if compression_score > 0.56 or pixelation_score > 0.3:
            return {
                "key": "artifact_rescue_general",
                "strength": "medium",
                "reason": "Artefactos de compresión/pixelado detectados"
            }

        if (
            image_type == "photo"
            and blur_severity == "low"
            and noise_level == "low"
            and compression_score < 0.22
            and pixelation_score < 0.14
            and not source_info.get("is_likely_social_media", False)
        ):
            return {
                "key": "clean_photo_detail",
                "strength": "low",
                "reason": "Foto relativamente limpia"
            }

        return {
            "key": "balanced_photo",
            "strength": "medium",
            "reason": "Perfil balanceado por defecto"
        }

    def _analyze_restoration_signals(
        self,
        img: np.ndarray,
        gray: np.ndarray,
        compression_metrics: Dict,
        pixelation_metrics: Dict,
        noise_level: str,
        source_info: Dict,
        blur_metrics: Dict,
        face_info: Dict
    ) -> Dict:
        """
        Detecta señales para restauración opcional (filtro, B/N, foto antigua/escaneo).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))

        b_channel, g_channel, r_channel = cv2.split(img)
        b_mean = float(np.mean(b_channel))
        g_mean = float(np.mean(g_channel))
        r_mean = float(np.mean(r_channel))
        global_mean = max(1.0, (r_mean + g_mean + b_mean) / 3.0)

        magenta_cast = (((r_mean + b_mean) * 0.5) - g_mean) / global_mean
        warm_cast = (r_mean - b_mean) / global_mean

        rg_diff = float(np.mean(np.abs(r_channel.astype(np.float32) - g_channel.astype(np.float32))))
        rb_diff = float(np.mean(np.abs(r_channel.astype(np.float32) - b_channel.astype(np.float32))))
        gb_diff = float(np.mean(np.abs(g_channel.astype(np.float32) - b_channel.astype(np.float32))))
        channel_diff = (rg_diff + rb_diff + gb_diff) / 3.0

        monochrome_confidence = 0.0
        if saturation < 11:
            monochrome_confidence += 0.65
        elif saturation < 18:
            monochrome_confidence += 0.45
        if channel_diff < 5.5:
            monochrome_confidence += 0.45
        elif channel_diff < 9.5:
            monochrome_confidence += 0.25
        if source_info.get("is_likely_social_media") and saturation < 14:
            monochrome_confidence += 0.12
        monochrome_confidence = float(np.clip(monochrome_confidence, 0.0, 1.0))
        is_monochrome = monochrome_confidence >= 0.6

        # Heurística de rayones/escaneo: líneas finas anómalas + bajo contraste global.
        edges = cv2.Canny(gray, 80, 180)
        edge_pixels = float(np.count_nonzero(edges))
        if edge_pixels > 0:
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13))
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
            thin_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
            thin_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
            scratch_ratio = float(np.count_nonzero(thin_v) + np.count_nonzero(thin_h)) / edge_pixels
        else:
            scratch_ratio = 0.0

        contrast_std = float(np.std(gray))
        compression_score = float(compression_metrics.get("compression_score", 0.0))
        pixelation_score = float(pixelation_metrics.get("pixelation_score", 0.0))
        highlight_clip_ratio = float(np.mean(gray > 245))
        blur_severity = blur_metrics.get("blur_severity", "low")
        has_relevant_faces = bool(face_info.get("has_faces", False) and face_info.get("importance") in {"medium", "high"})

        # Señal de captura tipo "story": barras/texto claros en franja superior.
        h, w = gray.shape
        top_h = max(24, int(h * 0.1))
        top_band = gray[:top_h, :]
        bright_top_ratio = float(np.mean(top_band > 220))
        top_edges = cv2.Canny(top_band, 80, 180)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, int(w * 0.12)), 1))
        top_horiz = cv2.morphologyEx(top_edges, cv2.MORPH_OPEN, horiz_kernel)
        top_horiz_ratio = float(np.count_nonzero(top_horiz)) / max(1.0, float(top_horiz.size))
        portrait_story_shape = h > int(w * 1.2)
        bright_threshold = 0.012 if portrait_story_shape else 0.018
        horiz_threshold = 0.0024 if portrait_story_shape else 0.0032
        story_overlay_detected = bool(
            (
                bright_top_ratio > bright_threshold
                and top_horiz_ratio > horiz_threshold
            )
            or (top_horiz_ratio > 0.008 and bright_top_ratio > 0.008)
        )
        if story_overlay_detected and not source_info.get("is_likely_social_media", False) and not portrait_story_shape:
            story_overlay_detected = bool(top_horiz_ratio > 0.0048 and bright_top_ratio > 0.012)

        format_hint = str(source_info.get("format", "") or "").upper()
        is_scan_friendly_format = format_hint in {"TIFF", "BMP", "PNG"}

        old_photo_detected = bool(
            (is_monochrome and contrast_std < 68)
            or (
                scratch_ratio > 0.27
                and (is_monochrome or is_scan_friendly_format or not has_relevant_faces)
            )
            or (
                contrast_std < 42
                and noise_level in {"medium", "high"}
                and not has_relevant_faces
            )
        )

        scan_artifacts_detected = bool(
            (
                scratch_ratio > 0.24
                and (is_scan_friendly_format or is_monochrome or not has_relevant_faces)
            )
            or (
                scratch_ratio > 0.33
                and (is_monochrome or not has_relevant_faces)
            )
            or (
                contrast_std < 46
                and compression_score > 0.45
                and (is_monochrome or not has_relevant_faces)
            )
            or (format_hint in {"TIFF", "BMP"} and contrast_std < 72)
        )

        social_filter_score = 0.0
        if source_info.get("is_likely_social_media"):
            social_filter_score += 0.24
        if magenta_cast > 0.07:
            social_filter_score += 0.42
        elif magenta_cast > 0.04:
            social_filter_score += 0.26
        if warm_cast > 0.06:
            social_filter_score += 0.18
        if highlight_clip_ratio > 0.04:
            social_filter_score += 0.2
        if saturation > 80:
            social_filter_score += 0.15

        social_degradation_core = (
            compression_score > 0.54
            or blur_severity in {"medium", "strong"}
            or (
                noise_level == "high"
                and (compression_score > 0.42 or blur_severity != "low")
            )
        )
        social_signature_score = 0
        if magenta_cast > 0.04:
            social_signature_score += 1
        if warm_cast > 0.05:
            social_signature_score += 1
        if highlight_clip_ratio > 0.03:
            social_signature_score += 1
        if saturation > 95:
            social_signature_score += 1
        if story_overlay_detected:
            social_signature_score += 1
        social_signature = social_signature_score >= 2

        degraded_social_portrait = bool(
            social_degradation_core
            and social_signature
            and (has_relevant_faces or source_info.get("is_likely_social_media", False))
        )
        if degraded_social_portrait:
            social_filter_score += 0.2

        color_cast_detected = (
            magenta_cast > 0.035
            or warm_cast > 0.055
            or highlight_clip_ratio > 0.055
        )
        if story_overlay_detected:
            social_filter_score += 0.1 if color_cast_detected else 0.03

        social_color_filter_detected = bool(
            not is_monochrome
            and (
                social_filter_score >= 0.5
                or (social_filter_score >= 0.42 and color_cast_detected)
            )
        )
        if social_filter_score >= 0.9:
            social_filter_strength = "high"
        elif social_filter_score >= 0.62:
            social_filter_strength = "medium"
        elif social_color_filter_detected:
            social_filter_strength = "low"
        else:
            social_filter_strength = "none"

        filter_score = 0.0
        if saturation > 120:
            filter_score += 0.55
        elif saturation > 95:
            filter_score += 0.35
        if compression_score > 0.35 or pixelation_score > 0.2:
            filter_score += 0.28
        if source_info.get("is_likely_social_media"):
            filter_score += 0.22
        if old_photo_detected or scan_artifacts_detected:
            filter_score += 0.2
        if social_color_filter_detected:
            filter_score += 0.32

        filter_detected = filter_score >= 0.45 and not is_monochrome
        if filter_score >= 0.95:
            filter_strength = "high"
        elif filter_score >= 0.65:
            filter_strength = "medium"
        elif filter_detected:
            filter_strength = "low"
        else:
            filter_strength = "none"

        return {
            "filter_detected": bool(filter_detected),
            "filter_strength": filter_strength,
            "social_color_filter_detected": bool(social_color_filter_detected),
            "social_filter_strength": social_filter_strength,
            "degraded_social_portrait": bool(degraded_social_portrait),
            "story_overlay_detected": bool(story_overlay_detected),
            "is_monochrome": bool(is_monochrome),
            "monochrome_confidence": monochrome_confidence,
            "old_photo_detected": bool(old_photo_detected),
            "scan_artifacts_detected": bool(scan_artifacts_detected),
            "scratch_score": float(np.clip(scratch_ratio, 0.0, 1.0)),
            "recommended_color_filter_correction": bool(social_color_filter_detected or degraded_social_portrait or filter_detected),
            "recommended_old_photo_restore": bool(old_photo_detected or scan_artifacts_detected),
            "recommended_filter_restoration": bool(filter_detected or old_photo_detected or scan_artifacts_detected or social_color_filter_detected),
            "recommended_bw_restore": bool(is_monochrome)
        }

    def _detect_blur(self, gray: np.ndarray) -> Dict:
        """Detecta blur usando Laplacian + Tenengrad."""
        # Usar float32 para ahorrar memoria (CV_32F) en lugar de CV_64F
        laplacian_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        tenengrad_score = float(np.mean(gradient_magnitude ** 2))

        if laplacian_score < 35 or tenengrad_score < 500:
            severity = "strong"
        elif laplacian_score < 80 or tenengrad_score < 1200:
            severity = "medium"
        else:
            severity = "low"

        return {
            "laplacian_score": laplacian_score,
            "tenengrad_score": tenengrad_score,
            "is_blurry": severity in {"strong", "medium"},
            "blur_severity": severity
        }
    
    def _detect_image_type(
        self,
        img: np.ndarray,
        blur_metrics: Dict = None,
        compression_metrics: Dict = None,
        pixelation_metrics: Dict = None,
        noise_level: str = "low",
        face_info: Dict = None
    ) -> str:
        """
        Detecta el tipo de imagen (foto real, anime, ilustración)
        
        Args:
            img: Imagen en formato numpy array
            
        Returns:
            Tipo de imagen: 'photo', 'anime', 'illustration'
        """
        # Convertir a HSV para análisis de color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calcular saturación promedio
        saturation = hsv[:, :, 1].mean()
        
        # Calcular variación de color
        color_variance = np.std(hsv[:, :, 0])
        
        # Detectar bordes para análisis de estilo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        blur_metrics = blur_metrics or {}
        compression_metrics = compression_metrics or {}
        pixelation_metrics = pixelation_metrics or {}
        face_info = face_info or {}

        compression_score = float(compression_metrics.get("compression_score", 0.0))
        pixelation_score = float(pixelation_metrics.get("pixelation_score", 0.0))
        blur_severity = blur_metrics.get("blur_severity", "low")
        has_relevant_faces = face_info.get("has_faces", False) and face_info.get("importance") in {"medium", "high"}

        # Score de anime conservador para evitar falsos positivos en fotos con filtros.
        anime_score = 0.0
        if saturation > 118:
            anime_score += 1.0
        if edge_density > 0.085:
            anime_score += 1.0
        if color_variance < 35:
            anime_score += 0.65
        if pixelation_score < 0.18 and compression_score < 0.2:
            anime_score += 0.35
        if noise_level == "low":
            anime_score += 0.15

        if blur_severity in {"medium", "strong"}:
            anime_score -= 0.4
        if compression_score > 0.32 or pixelation_score > 0.25:
            anime_score -= 0.45
        if has_relevant_faces:
            anime_score -= 0.9

        if anime_score >= 2.0:
            return "anime"

        # Foto real con filtros/compresión tipo redes sociales.
        if (
            saturation > 90
            and (
                compression_score > 0.35
                or pixelation_score > 0.22
                or blur_severity in {"medium", "strong"}
            )
        ):
            return "filtered_photo"

        # Ilustraciones (no anime) tienden a bordes más suaves y paleta uniforme.
        if color_variance < 26 and saturation > 70 and edge_density < 0.04:
            return "illustration"

        # Por defecto, asumir foto real.
        return "photo"
    
    def _calculate_sharpness(self, img: np.ndarray) -> float:
        """
        Calcula la nitidez de la imagen usando el operador Laplaciano
        
        Args:
            img: Imagen en formato numpy array
            
        Returns:
            Score de nitidez (mayor = más nítida)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._detect_blur(gray)["laplacian_score"]

    def _detect_compression_artifacts(self, gray: np.ndarray) -> Dict:
        """Detecta artefactos tipo JPEG/redes sociales usando blockiness 8x8."""
        gray_float = gray.astype(np.float32)
        h, w = gray.shape

        vertical_boundaries = [x for x in range(8, w, 8)]
        horizontal_boundaries = [y for y in range(8, h, 8)]

        boundary_diffs = []
        interior_diffs = []

        for x in vertical_boundaries:
            boundary_diffs.append(np.mean(np.abs(gray_float[:, x] - gray_float[:, x - 1])))
            if x + 1 < w:
                interior_diffs.append(np.mean(np.abs(gray_float[:, x + 1] - gray_float[:, x])))

        for y in horizontal_boundaries:
            boundary_diffs.append(np.mean(np.abs(gray_float[y, :] - gray_float[y - 1, :])))
            if y + 1 < h:
                interior_diffs.append(np.mean(np.abs(gray_float[y + 1, :] - gray_float[y, :])))

        if not boundary_diffs:
            return {"has_compression_artifacts": False, "compression_score": 0.0}

        b_avg = float(np.mean(boundary_diffs))
        i_avg = float(np.mean(interior_diffs)) if interior_diffs else 1.0
        compression_score = max(0.0, (b_avg - i_avg) / (i_avg + 1e-6))

        return {
            "has_compression_artifacts": compression_score > 0.25,
            "compression_score": compression_score
        }

    def _detect_pixelation(self, gray: np.ndarray) -> Dict:
        """Detecta pixelación por baja resolución/compresión agresiva."""
        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        quantized = (small // 16).astype(np.uint8)
        unique_bins_ratio = len(np.unique(quantized)) / 256.0

        edges = cv2.Canny(gray, 80, 180)
        straight_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        straight_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        straight_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, straight_kernel_h) + \
            cv2.morphologyEx(edges, cv2.MORPH_OPEN, straight_kernel_v)
        stair_ratio = float(np.count_nonzero(straight_edges)) / max(1, np.count_nonzero(edges))

        pixelation_score = max(0.0, (0.5 - unique_bins_ratio)) + max(0.0, stair_ratio - 0.45)

        return {
            "is_pixelated": pixelation_score > 0.22,
            "pixelation_score": float(pixelation_score)
        }
    
    def _estimate_noise(self, img: np.ndarray) -> str:
        """
        Estima el nivel de ruido en la imagen
        
        Args:
            img: Imagen en formato numpy array
            
        Returns:
            Nivel de ruido: 'low', 'medium', 'high'
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calcular desviación estándar en regiones pequeñas
        # como indicador de ruido
        h, w = gray.shape
        noise_scores = []

        # Muestreo determinista en rejilla para resultados reproducibles
        window = 50
        step_y = max(window, h // 4)
        step_x = max(window, w // 4)

        for y in range(0, max(1, h - window + 1), step_y):
            for x in range(0, max(1, w - window + 1), step_x):
                region = gray[y:y + window, x:x + window]
                if region.size > 0:
                    noise_scores.append(float(np.std(region)))
        
        avg_noise = np.mean(noise_scores) if noise_scores else 0
        
        if avg_noise < 15:
            return "low"
        elif avg_noise < 30:
            return "medium"
        else:
            return "high"
    
    def _detect_faces_info(self, img: np.ndarray) -> Dict:
        """
        Detecta si hay rostros y estima su relevancia visual.
        
        Args:
            img: Imagen en formato numpy array
            
        Returns:
            Dict con detección de rostro e importancia estimada
        """
        try:
            # Validar cascadas disponibles
            if self.face_cascade.empty():
                return {"has_faces": False, "importance": "none", "count": 0}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            min_face = max(32, int(min(h, w) * 0.045))

            detections = []
            faces_primary = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=6,
                minSize=(min_face, min_face)
            )
            detections.extend([(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in faces_primary])

            faces_relaxed = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(max(24, int(min_face * 0.85)), max(24, int(min_face * 0.85)))
            )
            detections.extend([(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in faces_relaxed])

            if not self.profile_face_cascade.empty():
                faces_profile = self.profile_face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.06,
                    minNeighbors=5,
                    minSize=(max(24, int(min_face * 0.85)), max(24, int(min_face * 0.85)))
                )
                detections.extend([(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in faces_profile])

            # Segundo pase en imagen reescalada para rostros pequeños/lejanos.
            upscale_factor = 1.5
            if min(h, w) >= 240:
                gray_large = cv2.resize(
                    gray,
                    (int(w * upscale_factor), int(h * upscale_factor)),
                    interpolation=cv2.INTER_CUBIC
                )
                faces_large = self.face_cascade.detectMultiScale(
                    gray_large,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(max(24, int(min_face * upscale_factor * 0.75)), max(24, int(min_face * upscale_factor * 0.75)))
                )
                for x, y, fw, fh in faces_large:
                    detections.append(
                        (
                            int(x / upscale_factor),
                            int(y / upscale_factor),
                            int(fw / upscale_factor),
                            int(fh / upscale_factor)
                        )
                    )

            if len(detections) == 0:
                return {"has_faces": False, "importance": "none", "count": 0}

            # Deduplicar rectángulos solapados.
            faces = self._deduplicate_face_boxes(detections)

            confirmed_faces = []
            for x, y, fw, fh in faces:
                roi_gray = gray[y:y + fh, x:x + fw]
                if roi_gray.size == 0:
                    continue

                # Buscar ojos en mitad superior para filtrar falsos positivos (pies/manos/objetos)
                upper_half = roi_gray[:max(1, int(fh * 0.7)), :]
                eyes_found = 0
                if not self.eye_cascade.empty() and upper_half.size > 0:
                    min_eye = max(8, int(min(fw, fh) * 0.12))
                    eyes = self.eye_cascade.detectMultiScale(
                        upper_half,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(min_eye, min_eye)
                    )
                    eyes_found = len(eyes)

                area_ratio = (fw * fh) / max(1, (h * w))
                center_y = (y + fh * 0.5) / max(1, h)

                confidence = 0.0
                if eyes_found >= 1:
                    confidence += 1.1
                if fw >= max(32, int(min(h, w) * 0.04)):
                    confidence += 0.45
                if area_ratio >= 0.0025:
                    confidence += 0.45
                if 0.08 <= center_y <= 0.95:
                    confidence += 0.2
                if area_ratio >= 0.015:
                    confidence += 0.35

                if confidence >= 1.2:
                    confirmed_faces.append((x, y, fw, fh))

            if len(confirmed_faces) == 0:
                return {"has_faces": False, "importance": "none", "count": 0}

            frame_area = h * w
            largest_face_area = max((fw * fh for _, _, fw, fh in confirmed_faces), default=0)
            ratio = largest_face_area / max(1, frame_area)

            if ratio > 0.12 or len(confirmed_faces) >= 3:
                importance = "high"
            elif ratio > 0.02 or len(confirmed_faces) >= 2:
                importance = "medium"
            else:
                importance = "low"

            return {"has_faces": True, "importance": importance, "count": len(confirmed_faces)}
        except Exception:
            # Si falla la detección, asumir que no hay rostros
            return {"has_faces": False, "importance": "none", "count": 0}

    def _deduplicate_face_boxes(self, boxes) -> list:
        """Elimina detecciones casi duplicadas usando IoU."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        selected = []

        for candidate in boxes:
            keep = True
            for chosen in selected:
                if self._box_iou(candidate, chosen) > 0.35:
                    keep = False
                    break
            if keep:
                selected.append(candidate)

        return selected

    def _box_iou(self, a, b) -> float:
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

    def _detect_faces(self, img: np.ndarray) -> bool:
        """Compatibilidad: devuelve True/False para detección de rostros."""
        return self._detect_faces_info(img)["has_faces"]

    def _should_apply_restoration(
        self,
        blur_metrics: Dict,
        noise_level: str,
        compression_metrics: Dict,
        pixelation_metrics: Dict
    ) -> bool:
        """Decide si conviene aplicar pre/post-proceso anti artefactos."""
        blur_severity = blur_metrics.get("blur_severity", "low")
        compression_score = float(compression_metrics.get("compression_score", 0.0))
        pixelation_score = float(pixelation_metrics.get("pixelation_score", 0.0))

        return any([
            blur_severity == "strong",
            noise_level == "high",
            compression_score > 0.55,
            pixelation_score > 0.3,
            (blur_severity == "medium" and compression_score > 0.48),
            (blur_severity == "medium" and pixelation_score > 0.28)
        ])

    def _should_use_uniform_restore(
        self,
        blur_metrics: Dict,
        noise_level: str,
        compression_metrics: Dict,
        pixelation_metrics: Dict
    ) -> bool:
        """Activa un perfil uniforme para escenas muy degradadas."""
        blur_severity = blur_metrics.get("blur_severity", "low")
        compression_score = float(compression_metrics.get("compression_score", 0.0))
        pixelation_score = float(pixelation_metrics.get("pixelation_score", 0.0))

        return (
            pixelation_score > 0.34
            or (blur_severity == "strong" and pixelation_score > 0.22)
            or (blur_severity == "strong" and compression_score > 0.5)
            or (blur_severity == "strong" and noise_level == "high")
        )
    
    def _recommend_scale(
        self,
        width: int,
        height: int,
        image_type: str,
        sharpness: float,
        noise_level: str = "low",
        compression_metrics: Dict = None,
        pixelation_metrics: Dict = None,
        blur_metrics: Dict = None
    ) -> int:
        """
        Recomienda la escala óptima basada en las características de la imagen
        
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            image_type: Tipo de imagen
            sharpness: Score de nitidez
            
        Returns:
            Escala recomendada (2 o 4)
        """
        total_pixels = width * height
        compression_metrics = compression_metrics or {}
        pixelation_metrics = pixelation_metrics or {}
        blur_metrics = blur_metrics or {}

        # Para imágenes de Instagram/altamente comprimidas, 2x produce resultados más naturales.
        if (
            noise_level in {"medium", "high"}
            and (
                compression_metrics.get("compression_score", 0.0) > 0.45
                or pixelation_metrics.get("pixelation_score", 0.0) > 0.25
                or blur_metrics.get("blur_severity") in {"medium", "strong"}
            )
        ):
            return 2
        
        # Si la imagen ya es grande (>2MP), recomendar 2x
        if total_pixels > 2_000_000:
            return 2
        
        # Si la imagen es muy pequeña (<0.5MP) y nítida, usar 4x
        if total_pixels < 500_000 and sharpness > 100:
            return 4
        
        # Para imágenes medianas, 4x es generalmente mejor
        if total_pixels < 1_500_000:
            return 4
        
        # Por defecto, 2x para imágenes grandes
        return 2
    
    def _recommend_model(self, image_type: str, has_faces: bool) -> str:
        """
        Recomienda el modelo óptimo basado en el tipo de imagen
        
        Args:
            image_type: Tipo de imagen
            has_faces: Si la imagen contiene rostros
            
        Returns:
            Clave del modelo recomendado
        """
        if image_type == "anime":
            return "4x_anime"
        elif image_type == "photo" or has_faces:
            return "4x"
        else:
            return "4x"
    
    def _generate_notes(
        self,
        width: int,
        height: int,
        image_type: str,
        sharpness: float,
        noise_level: str,
        has_faces: bool = False,
        blur_metrics: Dict = None,
        compression_metrics: Dict = None,
        pixelation_metrics: Dict = None
    ) -> list:
        """
        Genera notas descriptivas sobre el análisis
        
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            image_type: Tipo de imagen
            sharpness: Score de nitidez
            noise_level: Nivel de ruido
            has_faces: Si se detectaron rostros
            
        Returns:
            Lista de notas descriptivas
        """
        notes = []
        
        # Nota sobre resolución
        total_pixels = width * height
        if total_pixels < 500_000:
            notes.append("Imagen de baja resolución")
        elif total_pixels < 2_000_000:
            notes.append("Imagen de resolución media")
        else:
            notes.append("Imagen de alta resolución")
        
        # Nota sobre tipo
        type_names = {
            "photo": "Fotografía real",
            "filtered_photo": "Fotografía real con filtros/compresión",
            "anime": "Anime/Ilustración",
            "illustration": "Ilustración"
        }
        notes.append(f"Tipo: {type_names.get(image_type, 'Imagen')}")
        
        # Nota sobre rostros
        if has_faces:
            notes.append("Rostros detectados")
        
        # Nota sobre calidad
        if sharpness < 50:
            notes.append("Poca nitidez")
        elif sharpness > 200:
            notes.append("Buena nitidez")

        if blur_metrics and blur_metrics.get("blur_severity") == "strong":
            notes.append("Blur fuerte detectado")
        
        # Nota sobre ruido
        if noise_level == "high":
            notes.append("Nivel de ruido alto")

        if compression_metrics and compression_metrics.get("has_compression_artifacts"):
            notes.append("Artefactos de compresión")

        if pixelation_metrics and pixelation_metrics.get("is_pixelated"):
            notes.append("Pixelación detectada")
            if blur_metrics and blur_metrics.get("blur_severity") == "strong":
                notes.append("Se aplicará restauración uniforme por degradación severa")

        return notes
