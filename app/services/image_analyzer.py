"""
Analizador inteligente de imágenes
Determina el tipo de imagen y recomienda la mejor configuración de upscaling
Autor: Danny Maaz (github.com/dannymaaz)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict


class ImageAnalyzer:
    """Analizador de imágenes para determinar el mejor modelo y escala"""
    
    def __init__(self):
        """Inicializa el analizador de imágenes"""
        pass
    
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
        image_type = self._detect_image_type(img)
        blur_metrics = self._detect_blur(gray)
        sharpness = blur_metrics["laplacian_score"]
        tenengrad = blur_metrics["tenengrad_score"]
        noise_level = self._estimate_noise(img)
        compression_metrics = self._detect_compression_artifacts(gray)
        pixelation_metrics = self._detect_pixelation(gray)
        face_info = self._detect_faces_info(img)
        has_faces = face_info["has_faces"]
        
        # Determinar escala recomendada
        recommended_scale = self._recommend_scale(
            width, height, image_type, sharpness
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
            "face_importance": face_info["importance"],
            "recommended_scale": recommended_scale,
            "recommended_model": recommended_model,
            "apply_restoration": self._should_apply_restoration(
                blur_metrics,
                noise_level,
                compression_metrics,
                pixelation_metrics
            ),
            "analysis_notes": self._generate_notes(
                width,
                height,
                image_type,
                sharpness,
                noise_level,
                has_faces,
                blur_metrics,
                compression_metrics,
                pixelation_metrics
            )
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
    
    def _detect_image_type(self, img: np.ndarray) -> str:
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
        
        # Heurística simple para clasificación
        # Anime tiende a tener alta saturación y bordes definidos
        if saturation > 100 and edge_density > 0.05:
            return "anime"
        # Ilustraciones tienen colores más uniformes
        elif color_variance < 30 and saturation > 80:
            return "illustration"
        # Por defecto, asumir foto real
        else:
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
            # Usar detector de rostros de OpenCV (Haar Cascade)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                return {"has_faces": False, "importance": "none"}

            h, w = gray.shape
            frame_area = h * w
            largest_face_area = max((fw * fh for _, _, fw, fh in faces), default=0)
            ratio = largest_face_area / max(1, frame_area)

            if ratio > 0.15 or len(faces) >= 3:
                importance = "high"
            elif ratio > 0.05:
                importance = "medium"
            else:
                importance = "low"

            return {"has_faces": True, "importance": importance}
        except Exception:
            # Si falla la detección, asumir que no hay rostros
            return {"has_faces": False, "importance": "none"}

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
        return any([
            blur_metrics["blur_severity"] in {"medium", "strong"},
            noise_level in {"medium", "high"},
            compression_metrics["has_compression_artifacts"],
            pixelation_metrics["is_pixelated"]
        ])
    
    def _recommend_scale(
        self, width: int, height: int, image_type: str, sharpness: float
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

        return notes
