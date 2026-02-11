# 🚀 Real-ESRGAN Upscaling Profesional

<div align="center">

![Real-ESRGAN](https://img.shields.io/badge/Real--ESRGAN-v0.3.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Aplicación profesional para escalar imágenes usando inteligencia artificial**

Interfaz minimalista y futurista | Análisis inteligente | Múltiples modelos | 100% Local

[Características](#características) • [Instalación](#instalación) • [Uso](#uso) • [Capturas](#capturas)

</div>

---

## 📋 Descripción

Real-ESRGAN Upscaling Profesional es una aplicación web de escritorio que permite escalar imágenes usando modelos de inteligencia artificial Real-ESRGAN. La aplicación analiza automáticamente tus imágenes y recomienda la mejor configuración para obtener resultados óptimos.

### ✨ Características

- 🎨 **Interfaz Futurista**: Diseño minimalista con efectos glassmorphism y paleta de azules
- 🤖 **Análisis Inteligente**: Detecta automáticamente el tipo de imagen y recomienda la mejor escala
- 🚀 **Múltiples Modelos**: Soporte para modelos 2x, 4x, y 4x anime
- 📤 **Drag & Drop**: Interfaz intuitiva con arrastrar y soltar
- ⚡ **GPU Acelerado**: Usa GPU si está disponible, funciona en CPU también
- 🔒 **100% Local**: Tus imágenes nunca salen de tu computadora
- 📱 **Responsive**: Funciona en cualquier dispositivo

### 🎯 Modelos Disponibles

| Modelo | Escala | Descripción | Uso Recomendado |
|--------|--------|-------------|-----------------|
| **RealESRGAN_x2plus** | 2x | Más rápido | Imágenes grandes, texto |
| **RealESRGAN_x4plus** | 4x | Mejor calidad | Fotografías reales |
| **RealESRGAN_x4plus_anime_6B** | 4x | Optimizado anime | Ilustraciones, anime |

---

## 🛠️ Instalación

### Requisitos Previos

- **Python 3.8 o superior**
- **pip** (gestor de paquetes de Python)
- **Git** (opcional, para clonar el repositorio)

### Paso 1: Clonar o Descargar

```bash
# Opción 1: Clonar con Git
git clone https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional.git
cd Real_Esrgan_Upscaling_profesional

# Opción 2: Descargar ZIP
# Descarga el ZIP desde GitHub y extráelo
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota para GPU**: Si tienes una GPU NVIDIA y quieres usarla:
```bash
# Desinstalar PyTorch CPU
pip uninstall torch torchvision

# Instalar PyTorch con CUDA (visita pytorch.org para tu versión específica)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Paso 4: Descargar Modelos

```bash
python download_models.py
```

Este script descargará automáticamente los modelos necesarios (~500MB total).

---

## 🚀 Uso

### Iniciar la Aplicación

```bash
python run.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://127.0.0.1:8000`

### Flujo de Trabajo

1. **Sube tu imagen**: Arrastra y suelta o haz clic para seleccionar
2. **Revisa el análisis**: La app analizará tu imagen y recomendará configuración
3. **Selecciona escala**: Elige 2x o 4x (o usa la recomendación)
4. **Procesa**: Haz clic en "Procesar Imagen"
5. **Descarga**: Descarga tu imagen mejorada

### Formatos Soportados

- **Entrada**: PNG, JPG, JPEG
- **Salida**: Mismo formato que la entrada
- **Tamaño máximo**: 20 MB

---

## 📸 Capturas

*Próximamente: Capturas de pantalla de la interfaz*

---

## 🏗️ Estructura del Proyecto

```
Real_Esrgan_Upscaling_profesional/
├── app/                      # Backend Python
│   ├── routes/              # Endpoints de API
│   ├── services/            # Lógica de negocio
│   ├── utils/               # Utilidades
│   ├── config.py            # Configuración
│   └── main.py              # Aplicación FastAPI
├── frontend/                # Frontend web
│   ├── css/                 # Estilos
│   ├── js/                  # JavaScript
│   ├── assets/              # Recursos
│   └── index.html           # Página principal
├── models/                  # Modelos Real-ESRGAN
├── uploads/                 # Imágenes subidas (temporal)
├── outputs/                 # Resultados (temporal)
├── requirements.txt         # Dependencias Python
├── run.py                   # Script de ejecución
├── download_models.py       # Descargador de modelos
└── README.md               # Este archivo
```

---

## ⚙️ Configuración Avanzada

Puedes modificar `app/config.py` para ajustar:

- **Puerto del servidor**: `PORT = 8000`
- **Tamaño máximo de archivo**: `MAX_UPLOAD_SIZE`
- **Uso de GPU**: `USE_GPU = True`
- **Limpieza automática**: `AUTO_CLEANUP = True`

---

## 🐛 Solución de Problemas

### Error: "Modelo no encontrado"
```bash
python download_models.py
```

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### La aplicación es muy lenta
- Asegúrate de tener una GPU compatible
- Reduce el tamaño de la imagen antes de procesarla
- Usa escala 2x en lugar de 4x

### Error de memoria
- Cierra otras aplicaciones
- Usa imágenes más pequeñas
- Modifica `TILE_SIZE` en `config.py` a 400

---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Si quieres mejorar este proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👨‍💻 Autor

**Danny Maaz**

- GitHub: [@dannymaaz](https://github.com/dannymaaz)
- Proyecto: [Real-ESRGAN Upscaling Profesional](https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional)

---

## 🙏 Agradecimientos

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) por los modelos de IA
- [FastAPI](https://fastapi.tiangolo.com/) por el framework web
- [PyTorch](https://pytorch.org/) por el framework de deep learning

---

## 📚 Recursos Adicionales

- [Documentación Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Guía de instalación detallada](INSTALL.md)
- [Reporte de bugs](https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional/issues)

---

<div align="center">

**⭐ Si te gusta este proyecto, dale una estrella en GitHub ⭐**

Hecho con ❤️ por [Danny Maaz](https://github.com/dannymaaz)

</div>
