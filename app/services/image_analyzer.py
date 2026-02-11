"""
Analizador inteligente de imágenes
Determina el tipo de imagen y recomienda la mejor configuración de upscaling
Autor: Danny Maaz (github.com/dannymaaz)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image


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
        
        # Análisis de características
        image_type = self._detect_image_type(img)
        sharpness = self._calculate_sharpness(img)
        noise_level = self._estimate_noise(img)
        has_faces = self._detect_faces(img)
        
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
            "noise_level": noise_level,
            "has_faces": has_faces,
            "recommended_scale": recommended_scale,
            "recommended_model": recommended_model,
            "analysis_notes": self._generate_notes(
                width, height, image_type, sharpness, noise_level
            )
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
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness
    
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
        
        # Muestrear varias regiones
        for _ in range(10):
            y = np.random.randint(0, max(1, h - 50))
            x = np.random.randint(0, max(1, w - 50))
            region = gray[y:y+50, x:x+50]
            if region.size > 0:
                noise_scores.append(np.std(region))
        
        avg_noise = np.mean(noise_scores) if noise_scores else 0
        
        if avg_noise < 15:
            return "low"
        elif avg_noise < 30:
            return "medium"
        else:
            return "high"
    
    def _detect_faces(self, img: np.ndarray) -> bool:
        """
        Detecta si hay rostros en la imagen
        
        Args:
            img: Imagen en formato numpy array
            
        Returns:
            True si se detectan rostros, False en caso contrario
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
            return len(faces) > 0
        except Exception:
            # Si falla la detección, asumir que no hay rostros
            return False
    
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
        self, width: int, height: int, image_type: str, 
        sharpness: float, noise_level: str
    ) -> str:
        """
        Genera notas descriptivas sobre el análisis
        
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            image_type: Tipo de imagen
            sharpness: Score de nitidez
            noise_level: Nivel de ruido
            
        Returns:
            Notas descriptivas
        """
        notes = []
        
        # Nota sobre resolución
        total_pixels = width * height
        if total_pixels < 500_000:
            notes.append("Imagen de baja resolución - ideal para upscaling 4x")
        elif total_pixels < 2_000_000:
            notes.append("Imagen de resolución media - 4x recomendado")
        else:
            notes.append("Imagen de alta resolución - 2x recomendado para evitar archivos muy grandes")
        
        # Nota sobre tipo
        type_names = {
            "photo": "fotografía real",
            "anime": "anime/ilustración estilo anime",
            "illustration": "ilustración"
        }
        notes.append(f"Detectada como {type_names.get(image_type, 'imagen')}")
        
        # Nota sobre calidad
        if sharpness < 50:
            notes.append("Imagen con poca nitidez - el upscaling mejorará significativamente la calidad")
        elif sharpness > 200:
            notes.append("Imagen ya bastante nítida - el upscaling refinará los detalles")
        
        # Nota sobre ruido
        if noise_level == "high":
            notes.append("Alto nivel de ruido detectado - Real-ESRGAN ayudará a reducirlo")
        
        return " | ".join(notes)
