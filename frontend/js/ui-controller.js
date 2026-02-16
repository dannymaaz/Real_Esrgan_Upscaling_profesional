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
            'filtered_photo': 'Fotografía con Filtro',
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

        // Opción de corrección manual del tipo detectado.
        const overrideContainer = document.getElementById('imageTypeOverrideContainer');
        const overrideBtn = document.getElementById('imageTypeOverrideBtn');
        const overrideLabel = document.getElementById('imageTypeOverrideLabel');
        const overrideDescription = document.getElementById('imageTypeOverrideDescription');
        if (overrideContainer && overrideBtn && overrideLabel && overrideDescription) {
            overrideBtn.checked = false;

            if (analysis.image_type === 'anime') {
                overrideContainer.style.display = 'block';
                overrideLabel.textContent = 'Tratar como Fotografía real';
                overrideDescription.textContent = 'Actívalo si la imagen detectada como anime es en realidad una foto de una persona o escena real.';
            } else if (analysis.image_type === 'photo' || analysis.image_type === 'filtered_photo') {
                overrideContainer.style.display = 'block';
                overrideLabel.textContent = 'Tratar como Anime/Ilustración';
                overrideDescription.textContent = 'Actívalo solo si la imagen es realmente anime o ilustración digital.';
            } else {
                overrideContainer.style.display = 'none';
            }
        }

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
            ${data.type_overridden ? `
            <div class="info-row">
                <span class="info-label">Tipo corregido:</span>
                <span class="info-value">${data.analysis_image_type} → ${data.effective_image_type}</span>
            </div>` : ''}
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
            if (window.AppController && typeof window.AppController.onNewImageRequested === 'function') {
                window.AppController.onNewImageRequested();
                return;
            }
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

        // Reset state
        beforeWrapper.style.width = '50%';
        scroller.style.left = '50%';

        if (container.dataset.sliderBound === '1') {
            return;
        }
        container.dataset.sliderBound = '1';

        let active = false;
        let pointerId = null;

        // Solo arrastrar desde el handle para evitar movimiento accidental.
        scroller.addEventListener('pointerdown', (e) => {
            active = true;
            pointerId = e.pointerId;
            scroller.setPointerCapture(pointerId);
            e.preventDefault();
        });

        scroller.addEventListener('pointermove', (e) => {
            if (!active) return;
            this.updateSliderPosition(e, container, scroller, beforeWrapper);
        });

        const stopDrag = () => {
            active = false;
            pointerId = null;
        };

        scroller.addEventListener('pointerup', stopDrag);
        scroller.addEventListener('pointercancel', stopDrag);
        scroller.addEventListener('lostpointercapture', stopDrag);
    }

    static updateSliderPosition(e, container, scroller, beforeWrapper) {
        const rect = container.getBoundingClientRect();
        let x = e.clientX - rect.left;

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

        if (!zoomInBtn || !originalImg || !processedImg || !container || !zoomLevelDisplay) return;

        if (!this._zoomState) {
            this._zoomState = { scale: 1, panX: 0, panY: 0 };
        } else {
            this._zoomState.scale = 1;
            this._zoomState.panX = 0;
            this._zoomState.panY = 0;
        }
        const state = this._zoomState;

        const getDisplaySize = (imgEl) => {
            const cw = container.clientWidth || 1;
            const ch = container.clientHeight || 1;
            const nw = imgEl.naturalWidth || cw;
            const nh = imgEl.naturalHeight || ch;
            const fit = Math.min(cw / nw, ch / nh);
            return {
                width: Math.max(1, nw * fit),
                height: Math.max(1, nh * fit)
            };
        };

        const clampPan = () => {
            const base = getDisplaySize(originalImg);
            const maxX = Math.max(0, ((base.width * state.scale) - base.width) * 0.5);
            const maxY = Math.max(0, ((base.height * state.scale) - base.height) * 0.5);
            state.panX = Math.max(-maxX, Math.min(maxX, state.panX));
            state.panY = Math.max(-maxY, Math.min(maxY, state.panY));
        };

        const applyTransform = () => {
            state.scale = Math.max(1, Math.min(5, state.scale));
            clampPan();

            const transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
            originalImg.style.transformOrigin = '50% 50%';
            processedImg.style.transformOrigin = '50% 50%';
            originalImg.style.transform = transform;
            processedImg.style.transform = transform;

            zoomLevelDisplay.textContent = `${Math.round(state.scale * 100)}%`;
            container.style.cursor = state.scale > 1 ? 'grab' : 'col-resize';
        };

        const followPointer = (e) => {
            if (state.scale <= 1) return;

            const rect = container.getBoundingClientRect();
            const xRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
            const yRatio = Math.max(0, Math.min(1, (e.clientY - rect.top) / Math.max(1, rect.height)));

            const base = getDisplaySize(originalImg);
            const maxX = Math.max(0, ((base.width * state.scale) - base.width) * 0.5);
            const maxY = Math.max(0, ((base.height * state.scale) - base.height) * 0.5);

            state.panX = (0.5 - xRatio) * (maxX * 2);
            state.panY = (0.5 - yRatio) * (maxY * 2);
            clampPan();
            applyTransform();
        };

        zoomInBtn.onclick = () => {
            state.scale += 0.5;
            applyTransform();
        };

        zoomOutBtn.onclick = () => {
            state.scale -= 0.5;
            if (state.scale <= 1) {
                state.scale = 1;
                state.panX = 0;
                state.panY = 0;
            }
            applyTransform();
        };

        resetZoomBtn.onclick = () => {
            state.scale = 1;
            state.panX = 0;
            state.panY = 0;
            applyTransform();
        };

        if (container.dataset.zoomBound !== '1') {
            container.dataset.zoomBound = '1';

            container.addEventListener('mousemove', (e) => {
                followPointer(e);
            });

            container.addEventListener('wheel', (e) => {
                e.preventDefault();
                if (e.deltaY < 0) {
                    state.scale += 0.1;
                } else {
                    state.scale -= 0.1;
                }
                if (state.scale <= 1) {
                    state.scale = 1;
                    state.panX = 0;
                    state.panY = 0;
                }
                applyTransform();
            }, { passive: false });
        }

        applyTransform();
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

        const faceEnhanceBtn = document.getElementById('faceEnhanceBtn');
        if (faceEnhanceBtn) faceEnhanceBtn.checked = false;

        const imageTypeOverrideBtn = document.getElementById('imageTypeOverrideBtn');
        if (imageTypeOverrideBtn) imageTypeOverrideBtn.checked = false;

        const imageTypeOverrideContainer = document.getElementById('imageTypeOverrideContainer');
        if (imageTypeOverrideContainer) imageTypeOverrideContainer.style.display = 'none';
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
