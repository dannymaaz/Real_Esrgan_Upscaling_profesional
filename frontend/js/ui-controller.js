/**
 * Controlador de interfaz de usuario
 * Autor: Danny Maaz (github.com/dannymaaz)
 */

class UIController {
    /**
     * Muestra el panel de control con información de la imagen
     * @param {Object} analysis - Análisis de la imagen
     */
    static showControlPanel(analysis) {
        const controlPanel = document.getElementById('controlPanel');

        // Actualizar información de la imagen
        document.getElementById('imageResolution').textContent =
            `${analysis.width} × ${analysis.height} px (${analysis.megapixels} MP)`;

        const typeNames = {
            'photo': 'Fotografía Real',
            'anime': 'Anime/Ilustración',
            'illustration': 'Ilustración'
        };
        document.getElementById('imageType').textContent =
            typeNames[analysis.image_type] || analysis.image_type;

        document.getElementById('imageSize').textContent =
            `${analysis.file_size_mb} MB`;

        // Mostrar recomendación
        const recommendation = document.getElementById('recommendation');
        const recommendationText = document.getElementById('recommendationText');

        // Limpiar contenido previo
        recommendationText.innerHTML = '';

        // Título de recomendación principal
        const mainRec = document.createElement('div');
        mainRec.className = 'rec-main';
        mainRec.textContent = `Recomendamos usar escala ${analysis.recommended_scale}x para mejores resultados.`;
        recommendationText.appendChild(mainRec);

        // Lista de detalles
        if (Array.isArray(analysis.analysis_notes) && analysis.analysis_notes.length > 0) {
            const list = document.createElement('ul');
            list.className = 'rec-list';

            analysis.analysis_notes.forEach(note => {
                const li = document.createElement('li');
                li.textContent = note;
                list.appendChild(li);
            });

            recommendationText.appendChild(list);
        } else if (typeof analysis.analysis_notes === 'string') {
            // Fallback para versiones anteriores
            const p = document.createElement('p');
            p.textContent = analysis.analysis_notes;
            recommendationText.appendChild(p);
        }

        recommendation.style.display = 'flex';

        // Seleccionar automáticamente la escala recomendada
        const scaleButtons = document.querySelectorAll('.scale-btn');
        scaleButtons.forEach(btn => {
            const scale = btn.dataset.scale;
            if (scale === `${analysis.recommended_scale}x`) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Mostrar panel
        controlPanel.style.display = 'block';
    }

    /**
     * Muestra el panel de procesamiento
     */
    static showProcessingPanel() {
        document.getElementById('controlPanel').style.display = 'none';
        document.getElementById('processingPanel').style.display = 'block';

        // Animar barra de progreso
        const progressFill = document.getElementById('progressFill');
        progressFill.style.width = '0%';

        setTimeout(() => {
            progressFill.style.width = '30%';
        }, 100);

        setTimeout(() => {
            progressFill.style.width = '60%';
        }, 1000);
    }

    /**
     * Completa la animación de progreso
     */
    static completeProgress() {
        const progressFill = document.getElementById('progressFill');
        progressFill.style.width = '100%';
    }

    /**
     * Muestra el panel de resultados (Alias para compatibilidad)
     */
    static showResultPanel(result) {
        this.showResult(result);
    }

    /**
     * Muestra el panel de resultados con comparación
     * @param {Object} data - Resultado del procesamiento
     */
    static showResult(data) {
        console.log('showResult called with data:', data);

        document.getElementById('processingPanel').style.display = 'none';
        const resultPanel = document.getElementById('resultPanel');
        resultPanel.style.display = 'block';

        // Configurar imágenes para comparación
        const originalImg = document.getElementById('originalImage');
        const processedImg = document.getElementById('processedImage');

        console.log('Image elements:', { originalImg, processedImg });

        // Función para cargar imagen con Promise
        const loadImage = (imgElement, src, label) => {
            return new Promise((resolve, reject) => {
                imgElement.onload = () => {
                    console.log(`${label} loaded successfully:`, src);
                    resolve();
                };
                imgElement.onerror = (e) => {
                    console.error(`${label} failed to load:`, src, e);
                    reject(new Error(`Failed to load ${label}`));
                };
                imgElement.src = src;
                console.log(`${label} src set to:`, src);
            });
        };

        // Cargar imagen original
        let originalSrc = null;
        if (window.currentFile) {
            originalSrc = URL.createObjectURL(window.currentFile);
            console.log('Using window.currentFile for original image');
        } else {
            const fileInput = document.getElementById('fileInput');
            if (fileInput && fileInput.files && fileInput.files[0]) {
                originalSrc = URL.createObjectURL(fileInput.files[0]);
                console.log('Using fileInput for original image');
            } else {
                console.error('No source found for original image!');
            }
        }

        // Cargar imagen procesada
        const processedSrc = `${APIClient.BASE_URL}/api/download/${data.output_filename}?t=${new Date().getTime()}`;
        console.log('Processed image URL:', processedSrc);

        // Cargar ambas imágenes en paralelo
        Promise.all([
            originalSrc ? loadImage(originalImg, originalSrc, 'Original') : Promise.reject('No original source'),
            loadImage(processedImg, processedSrc, 'Processed')
        ]).then(() => {
            console.log('Both images loaded successfully');
            // Configurar slider de comparación después de que las imágenes se carguen
            this.setupComparisonSlider();
            this.setupZoomControls();
        }).catch((error) => {
            console.error('Error loading images:', error);
            UIController.showError('Error al cargar las imágenes: ' + error.message);
        });

        // Actualizar información del resultado
        const resultInfo = document.getElementById('resultInfo');
        resultInfo.innerHTML = `
            <!--
            <div class="info-row">
                <span class="info-label">Modelo usado:</span>
                <span class="info-value">${data.model_used}</span>
            </div>
            -->
            <div class="info-row">
                <span class="info-label">Escala:</span>
                <span class="info-value">${data.scale}x</span>
            </div>
            <div class="info-row">
                <span class="info-label">Resolución final:</span>
                <span class="info-value">${data.output_size.width} × ${data.output_size.height}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Mejora Facial:</span>
                <span class="info-value">${data.face_enhance ? 'Sí' : 'No'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Tiempo:</span>
                <span class="info-value">${data.processing_time_seconds ?? '-'} s</span>
            </div>
        `;

        // Configurar botón de descarga
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.onclick = () => {
            // Crear enlace temporal para descarga directa
            const link = document.createElement('a');
            link.href = APIClient.getDownloadUrl(data.output_filename);
            link.download = data.output_filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };

        const newImageBtn = document.getElementById('newImageBtn');
        newImageBtn.onclick = () => {
            location.reload();
        };
    }

    static setupComparisonSlider() {
        const container = document.getElementById('comparisonContainer');
        const scroller = document.getElementById('scroller');
        const beforeWrapper = document.querySelector('.img-wrapper.before');

        if (!container || !scroller || !beforeWrapper) {
            console.error('Comparison slider elements not found:', { container, scroller, beforeWrapper });
            return;
        }

        console.log('Setting up comparison slider');

        let active = false;

        // Reset state
        beforeWrapper.style.width = '50%';
        scroller.style.left = '50%';

        // Mouse events
        container.addEventListener('mousedown', () => active = true);
        document.addEventListener('mouseup', () => active = false);
        document.addEventListener('mouseleave', () => active = false);

        container.addEventListener('mousemove', (e) => {
            if (!active) return;
            this.updateSliderPosition(e, container, scroller, beforeWrapper);
        });

        // Touch events
        container.addEventListener('touchstart', () => active = true);
        document.addEventListener('touchend', () => active = false);
        document.addEventListener('touchcancel', () => active = false);

        container.addEventListener('touchmove', (e) => {
            if (!active) return;
            this.updateSliderPosition(e.touches[0], container, scroller, beforeWrapper);
        });
    }

    static updateSliderPosition(e, container, scroller, beforeWrapper) {
        const rect = container.getBoundingClientRect();
        let x = e.pageX - rect.left;

        // Limitar movimiento dentro del contenedor
        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;

        const percentage = (x / rect.width) * 100;

        scroller.style.left = `${percentage}%`;
        beforeWrapper.style.width = `${percentage}%`;
    }
    static setupZoomControls() {
        const zoomInBtn = document.getElementById('zoomInBtn');
        const zoomOutBtn = document.getElementById('zoomOutBtn');
        const resetZoomBtn = document.getElementById('resetZoomBtn');
        const zoomLevelDisplay = document.getElementById('zoomLevel');

        const originalImg = document.getElementById('originalImage');
        const processedImg = document.getElementById('processedImage');
        const container = document.getElementById('comparisonContainer');

        if (!zoomInBtn || !originalImg || !processedImg || !container) return;

        let scale = 1;

        const updateZoom = () => {
            // Limitar zoom
            if (scale < 1) scale = 1;
            if (scale > 5) scale = 5;

            const transform = `scale(${scale})`;
            originalImg.style.transform = transform;
            processedImg.style.transform = transform;

            zoomLevelDisplay.textContent = `${Math.round(scale * 100)}%`;

            // Si hay zoom, cambiar cursor
            if (scale > 1) {
                container.style.cursor = 'grab';
            } else {
                container.style.cursor = 'col-resize';
            }
        };

        zoomInBtn.onclick = () => {
            scale += 0.5;
            updateZoom();
        };

        zoomOutBtn.onclick = () => {
            scale -= 0.5;
            updateZoom();
        };

        resetZoomBtn.onclick = () => {
            scale = 1;
            originalImg.style.transformOrigin = '0 0';
            processedImg.style.transformOrigin = '0 0';
            updateZoom();
        };

        // Pan logic simple para imágenes con zoom
        const setTransformOrigin = (e) => {
            if (scale <= 1) return;
            const rect = container.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * 100;
            const y = (e.clientY - rect.top) / rect.height * 100;

            originalImg.style.transformOrigin = `${x}% ${y}%`;
            processedImg.style.transformOrigin = `${x}% ${y}%`;
        };

        container.addEventListener('mousemove', setTransformOrigin);

        // Scroll wheel zoom
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY < 0) {
                scale += 0.1;
            } else {
                scale -= 0.1;
            }
            updateZoom();
        });
    }

    /**
     * Reinicia la aplicación para procesar una nueva imagen
     */
    static reset() {
        document.getElementById('controlPanel').style.display = 'none';
        document.getElementById('processingPanel').style.display = 'none';
        document.getElementById('resultPanel').style.display = 'none';

        // Limpiar selección de escala
        document.querySelectorAll('.scale-btn').forEach(btn => {
            btn.classList.remove('active');
        });

        // Resetear input de archivo
        document.getElementById('fileInput').value = '';
    }

    /**
     * Muestra un mensaje de error
     * @param {string} message - Mensaje de error
     */
    static showError(message) {
        const errorModal = document.getElementById('errorModal');
        const errorMessage = document.getElementById('errorMessage');

        const raw = (message ?? '').toString();
        const normalized = raw.includes('Failed to fetch')
            ? 'No se pudo conectar con el servidor. Verifica que este activo en http://127.0.0.1:8000'
            : raw;

        errorMessage.textContent = normalized || 'Ocurrio un error inesperado';
        errorModal.style.display = 'flex';
    }

    /**
     * Cierra el modal de error
     */
    static closeError() {
        document.getElementById('errorModal').style.display = 'none';
    }

    /**
     * Formatea bytes a una representación legible
     * @param {number} bytes - Tamaño en bytes
     * @returns {string} Tamaño formateado
     */
    static formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
    }

    // Helper shortcuts
    static hideElement(id) {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    }

    static showElement(id) {
        const el = document.getElementById(id);
        if (el) el.style.display = 'block';
    }
}
