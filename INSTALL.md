# 📦 Guía de Instalación Detallada

Esta guía te llevará paso a paso por el proceso de instalación de Real-ESRGAN Upscaling Profesional.

---

## 📋 Requisitos del Sistema

### Mínimos
- **Sistema Operativo**: Windows 10/11, Linux, macOS
- **Python**: 3.8 o superior
- **RAM**: 4 GB mínimo
- **Espacio en disco**: 2 GB (modelos + dependencias)
- **Procesador**: CPU de 64 bits

### Recomendados
- **RAM**: 8 GB o más
- **GPU**: NVIDIA con CUDA (para procesamiento acelerado)
- **Espacio en disco**: 5 GB

---

## 🪟 Instalación en Windows

### 1. Instalar Python

1. Descarga Python desde [python.org](https://www.python.org/downloads/)
2. Ejecuta el instalador
3. **IMPORTANTE**: Marca "Add Python to PATH"
4. Haz clic en "Install Now"

Verifica la instalación:
```cmd
python --version
```

### 2. Descargar el Proyecto

**Opción A: Con Git**
```cmd
git clone https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional.git
cd Real_Esrgan_Upscaling_profesional
```

**Opción B: Sin Git**
1. Ve a https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional
2. Haz clic en "Code" → "Download ZIP"
3. Extrae el ZIP en una carpeta
4. Abre CMD en esa carpeta

### 3. Crear Entorno Virtual

```cmd
python -m venv venv
venv\Scripts\activate
```

Verás `(venv)` al inicio de tu línea de comandos.

### 4. Instalar Dependencias

```cmd
pip install -r requirements.txt
```

Esto tomará varios minutos.

### 5. Descargar Modelos

```cmd
python download_models.py
```

Los modelos se descargarán automáticamente (~500MB).

### 6. Ejecutar la Aplicación

```cmd
python run.py
```

¡Listo! La aplicación se abrirá en tu navegador.

---

## 🐧 Instalación en Linux

### 1. Instalar Python y pip

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

**Fedora:**
```bash
sudo dnf install python3 python3-pip git
```

**Arch:**
```bash
sudo pacman -S python python-pip git
```

### 2. Clonar el Repositorio

```bash
git clone https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional.git
cd Real_Esrgan_Upscaling_profesional
```

### 3. Crear Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 5. Descargar Modelos

```bash
python download_models.py
```

### 6. Ejecutar

```bash
python run.py
```

---

## 🍎 Instalación en macOS

### 1. Instalar Homebrew (si no lo tienes)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Instalar Python

```bash
brew install python@3.11
```

### 3. Clonar el Repositorio

```bash
git clone https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional.git
cd Real_Esrgan_Upscaling_profesional
```

### 4. Crear Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 6. Descargar Modelos

```bash
python download_models.py
```

### 7. Ejecutar

```bash
python run.py
```

---

## 🎮 Instalación con GPU (NVIDIA)

Para usar aceleración GPU, necesitas:

1. **GPU NVIDIA compatible** con CUDA
2. **Drivers NVIDIA** actualizados
3. **CUDA Toolkit** (se instala con PyTorch)

### Instalar PyTorch con CUDA

```bash
# Desinstalar versión CPU
pip uninstall torch torchvision

# Instalar versión GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Para otras versiones de CUDA, visita [pytorch.org](https://pytorch.org/get-started/locally/)

### Verificar GPU

```python
python -c "import torch; print(torch.cuda.is_available())"
```

Si imprime `True`, ¡GPU está lista!

---

## 🔧 Solución de Problemas

### Error: "python no se reconoce como comando"

**Windows**: Reinstala Python y marca "Add Python to PATH"

**Linux/Mac**: Usa `python3` en lugar de `python`

### Error: "pip no se reconoce"

```bash
python -m pip install --upgrade pip
```

### Error: "No module named 'torch'"

```bash
pip install torch torchvision
```

### Error de permisos en Linux/Mac

```bash
sudo chown -R $USER:$USER .
```

### Error: "Address already in use"

Otro programa está usando el puerto 8000. Cambia el puerto en `app/config.py`:
```python
PORT = 8080  # O cualquier otro puerto
```

### Descarga de modelos falla

Descarga manualmente desde:
- [RealESRGAN_x2plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)
- [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
- [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

Colócalos en la carpeta `models/`

---

## 🚀 Próximos Pasos

Una vez instalado:

1. Lee el [README.md](README.md) para aprender a usar la aplicación
2. Prueba con diferentes tipos de imágenes
3. Experimenta con las diferentes escalas
4. ¡Comparte tus resultados!

---

## 💡 Consejos

- **Primera vez**: Usa imágenes pequeñas para probar
- **GPU**: Acelera el procesamiento 5-10x
- **Espacio**: Los archivos temporales se limpian automáticamente
- **Actualizaciones**: Haz `git pull` para obtener nuevas versiones

---

## 📞 Soporte

¿Problemas? Abre un issue en:
https://github.com/dannymaaz/Real_Esrgan_Upscaling_profesional/issues

---

**¡Disfruta escalando tus imágenes! 🎨**
