# 🚀 Real-ESRGAN Upscaling Profesional

<div align="center">
  <img src="https://img.shields.io/badge/AI-Upscaling-blue?style=for-the-badge&logo=ai" alt="AI Upscaling">
  <img src="https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Framework-green?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Real--ESRGAN-Powerful-orange?style=for-the-badge&logo=github" alt="Real-ESRGAN">
</div>

---

## 🌟 Visión General
Esta es una aplicación de **upscaling de imágenes de grado profesional** diseñada con una estética minimalista y tecnología de vanguardia. Utiliza los modelos **Real-ESRGAN** y **GFPGAN** potenciados por un motor de análisis inteligente que adapta el procesamiento según la calidad original de la fotografía.

### 🧠 ¿Qué nos hace diferentes?
A diferencia de otros upscalers genéricos, este proyecto implementa una capa de inteligencia artificial personalizada que resuelve los problemas comunes de la restauración digital:

*   **🛡️ Sistema Anti-Plástico (v2.0):** Inyección de micro-grano orgánico para evitar superficies lisas artificiales y mantener la textura real en la piel y telas.
*   **📱 Detector de Origen Digital:** Identifica automáticamente si una foto proviene de **WhatsApp, Instagram o Facebook**, aplicando técnicas de *deblocking* específicas para combatir la compresión agresiva de la web.
*   **🌙 Procesamiento Inteligente de Luz:** Analiza la luminancia de la imagen para detectar condiciones de baja iluminación (noche), ajustando el denoise para no borrar el detalle natural del grano fotográfico.
*   **💾 Optimización Dinámica de Memoria:** Implementa *tiling* adaptativo y pre-redimensionado seguro, permitiendo procesar imágenes de ultra alta resolución (4K/8K) incluso en hardware con recursos limitados.
*   **🎨 Restauración de Filtros y B/N:** Capacidad experimental para detectar fotos monocromáticas o escaneos antiguos, reduciendo dominantes cromáticas artificiales para recuperar la naturalidad.
*   **✋ Protección de Detalles Sensibles:** Máscaras inteligentes para manos, pies y rostros que evitan el exceso de nitidez (*oversharpening*) y halos extraños en los bordes.

---

## 🛠️ Tecnologías Utilizadas
- **Backend:** FastAPI (Python) para una comunicación ultrarrápida.
- **Modelos IA:** Real-ESRGAN (x2+, x4+, Anime) & GFPGAN (v1.3) para rostros.
- **Procesamiento:** OpenCV, PyTorch & NumPy.
- **Frontend:** Vanilla JS & Modern CSS con efectos de *Glassmorphism* y animaciones fluidas.

---

## 🚀 Formas de Ejecución

### 1. Lanzador Directo (Windows - Recomendado)
¡Ideal para abrir la app con un solo clic sin usar la terminal!
1. Navega a la carpeta del proyecto.
2. Haz doble clic en `Lanzador_RealESRGAN.bat`.
3. (Opcional) Crea un acceso directo de este archivo y colócalo en tu **Escritorio**.

### 2. Docker (Contenedorización Profesional)
Si prefieres usar Docker para evitar instalar dependencias localmente:
```bash
docker-compose up -d
```
*La app estará disponible en `http://localhost:8000`.*

### 3. Ejecución Manual
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la App
python run.py
```

### 4. Modo Robusto (Stability Manager)
Si deseas que el servidor se mantenga siempre activo y se reinicie automáticamente ante cualquier fallo:
```powershell
powershell -ExecutionPolicy Bypass -File run_robust.ps1
```

> **⚡ Nota sobre Puertos:** La aplicación ahora incluye **Detección Automática de Conflictos**. Si el puerto default (8000) está ocupado por otra app, el sistema buscará automáticamente el siguiente puerto libre (8001, 8002...) para no interferir con tus otros proyectos.


---

## 🛡️ Seguridad y Optimización
*   **Gestión de Errores:** Monitoreo activo de logs (`server_log.txt`) para detectar fallos de hardware o memoria en tiempo real.
*   **Escalabilidad:** Código modular estructurado para añadir nuevos modelos de IA con facilidad.
*   **Privacidad Efímera:** Sistema automático que elimina archivos procesados después de 24 horas para proteger la privacidad del usuario.

---

## ☕ Apoya el Proyecto
Si este proyecto te ha sido útil, considera apoyarme para seguir desarrollando herramientas de IA de alta fidelidad:

<div align="center">
  <a href="https://paypal.me/Creativegt" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Donar_vía_Paypal-00457C?style=for-the-badge&logo=paypal&logoColor=white" alt="PayPal Me">
  </a>
</div>

---

## 👨‍💻 Créditos y Autoría

<p align="left">
  <strong>Creado Por: Danny Maaz</strong>
  <br>
  <a href="https://github.com/dannymaaz" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/GitHub-dannymaaz-black?style=flat&logo=github" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/danny-maaz-a566251b5/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-Danny_Maaz-blue?style=flat&logo=linkedin" alt="LinkedIn">
  </a>
</p>

---
*© 2026 Real-ESRGAN Upscaling Profesional - Danny Maaz. Todos los derechos reservados.*
