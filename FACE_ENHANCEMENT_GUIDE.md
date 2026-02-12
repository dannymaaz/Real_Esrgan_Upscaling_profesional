# Guía de Mejora Facial (Face Enhancement)

## 🎯 Resumen

Esta aplicación utiliza **GFPGAN (Generative Facial Prior GAN)** para mejorar rostros en imágenes escaladas. El sistema está configurado para **preservar los rasgos originales** de la persona, incluyendo características como ojos cerrados, expresiones faciales, y otros detalles únicos.

---

## 🔍 Detección Automática de Rostros

### Método de Detección
La aplicación utiliza **Haar Cascade Classifier** de OpenCV para detectar rostros:

```python
# Ubicación: app/services/image_analyzer.py
def _detect_faces(self, img: np.ndarray) -> bool:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces) > 0
```

### Características
- ✅ Detecta rostros frontales
- ✅ Funciona con múltiples rostros
- ✅ Tamaño mínimo de detección: 30x30 píxeles
- ✅ Auto-activa el checkbox "Mejorar Rostros" si detecta caras

---

## 🎨 Mejora Facial con GFPGAN

### Parámetro de Fidelidad (Weight)

El sistema usa `weight=0.5` para balancear entre **fidelidad** (preservar original) y **mejora** (calidad visual):

```python
# Ubicación: app/services/upscaler.py
_, _, output = face_enhancer.enhance(
    img, 
    has_aligned=False, 
    only_center_face=False, 
    paste_back=True,
    weight=0.5  # ← Balance óptimo
)
```

### Escala de Weight

| Valor | Comportamiento | Uso Recomendado |
|-------|---------------|-----------------|
| `0.0` | **Máxima fidelidad** - Preserva 100% los rasgos originales (ojos cerrados, expresiones, etc.) | Fotos artísticas, retratos con expresiones específicas |
| `0.5` | **Balance óptimo** - Mejora calidad manteniendo identidad y rasgos | **Uso general (configuración actual)** |
| `1.0` | **Máxima mejora** - Puede alterar rasgos para maximizar calidad visual | Fotos muy dañadas o de baja calidad |

---

## 🧪 Casos de Uso Específicos

### Ojos Cerrados
Con `weight=0.5`, GFPGAN:
- ✅ **Preserva** el estado de ojos cerrados
- ✅ Mejora la textura de los párpados
- ✅ Mantiene la expresión facial original
- ❌ **NO** abre los ojos artificialmente

### Expresiones Faciales
- ✅ Sonrisas, ceños fruncidos, etc. se mantienen
- ✅ Arrugas naturales se preservan (no se "suavizan" excesivamente)
- ✅ Identidad facial se mantiene intacta

### Rostros Parcialmente Ocultos
- ✅ Detecta y mejora rostros con accesorios (gafas, sombreros)
- ✅ Funciona con rostros en ángulos moderados
- ⚠️ Puede no detectar rostros de perfil completo

---

## ⚙️ Configuración Técnica

### Parámetros de GFPGAN

```python
face_enhancer = GFPGANer(
    model_path=str(GFPGAN_MODEL_PATH),
    upscale=scale,                    # 2 o 4
    arch='clean',                     # Arquitectura optimizada
    channel_multiplier=2,             # Capacidad del modelo
    bg_upsampler=upsampler,          # RealESRGAN para fondo
    device=self.device               # 'cuda' o 'cpu'
)
```

### Parámetros de Enhancement

```python
face_enhancer.enhance(
    img,                              # Imagen de entrada
    has_aligned=False,                # Rostros no están pre-alineados
    only_center_face=False,           # Procesar todos los rostros
    paste_back=True,                  # Pegar rostros mejorados en imagen original
    weight=0.5                        # Balance fidelidad/mejora
)
```

---

## 📊 Validación de Resultados

### Cómo Verificar que Funciona Correctamente

1. **Prueba con Ojos Cerrados**:
   - Sube una foto con ojos cerrados
   - Activa "Mejorar Rostros"
   - Verifica que los ojos permanezcan cerrados en el resultado

2. **Prueba con Expresiones**:
   - Usa fotos con sonrisas, ceños, etc.
   - Confirma que la expresión se mantiene

3. **Comparación Antes/Después**:
   - Usa el slider de comparación
   - Verifica que la identidad facial sea idéntica
   - Confirma que solo mejora la nitidez/textura

---

## 🔧 Ajustes Avanzados (Opcional)

Si necesitas más control sobre la fidelidad, puedes modificar el parámetro `weight` en `app/services/upscaler.py`:

```python
# Línea ~287
weight=0.5  # Cambia este valor según necesites
```

### Recomendaciones por Tipo de Foto

| Tipo de Foto | Weight Recomendado | Razón |
|--------------|-------------------|-------|
| Selfies/Retratos | `0.5` | Balance perfecto |
| Fotos artísticas | `0.3-0.4` | Preservar estilo original |
| Fotos antiguas/dañadas | `0.6-0.7` | Más restauración |
| Fotos profesionales | `0.4-0.5` | Mantener calidad original |

---

## 📝 Notas Técnicas

### Limitaciones Conocidas
- GFPGAN funciona mejor con rostros frontales (±45°)
- Rostros muy pequeños (<30px) pueden no detectarse
- Oclusiones extremas (>50% del rostro) pueden afectar resultados

### Modelo Utilizado
- **GFPGAN v1.3** (GFPGANv1.3.pth)
- Entrenado en dataset FFHQ
- Optimizado para rostros reales (no anime)

### Integración con Real-ESRGAN
- GFPGAN mejora **solo los rostros**
- Real-ESRGAN escala **el resto de la imagen**
- Ambos se combinan automáticamente para resultado uniforme

---

## ✅ Conclusión

La configuración actual (`weight=0.5`) está optimizada para:
- ✅ Preservar rasgos originales (ojos cerrados, expresiones)
- ✅ Mejorar calidad y nitidez facial
- ✅ Mantener identidad 100% intacta
- ✅ Evitar artefactos o alteraciones no deseadas

**No es necesario ajustar nada** para uso general. El sistema ya está configurado para respetar los rasgos originales de las personas.
