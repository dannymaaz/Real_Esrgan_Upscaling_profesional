# Usar una imagen de Python oficial con soporte para OpenCV
FROM python:3.10-slim

# Evitar que Python genere archivos .pyc y forzar salida de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para OpenCV y PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Crear directorios necesarios
RUN mkdir -p uploads outputs models

# Exponer el puerto configurado (por defecto 8000)
EXPOSE 8000

# Comando para iniciar la aplicación (usando el script de ejecución ya preparado)
CMD ["python", "run.py"]
