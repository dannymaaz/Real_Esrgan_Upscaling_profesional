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

        const filterState = document.getElementById('imageFilterState');
        if (filterState) {
            if (analysis.social_color_filter_detected || analysis.filter_detected) {
                const strength = analysis.social_filter_strength || analysis.filter_strength || 'medium';
                filterState.textContent = `Sí, color (${strength})`;
            } else if (analysis.old_photo_detected || analysis.scan_artifacts_detected) {
                filterState.textContent = 'Sí, antigua/escaneo';
            } else if (analysis.is_monochrome) {
                filterState.textContent = 'Blanco y negro detectado';
            } else {
                filterState.textContent = 'No';
            }
        }

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

            if (analysis.image_type === 'anime' || analysis.image_type === 'illustration') {
                overrideContainer.style.display = 'block';
                overrideLabel.textContent = 'Tratar como Fotografía real';
                overrideDescription.textContent = 'Actívalo si la imagen detectada como anime/ilustración es en realidad una foto de una persona o escena real.';
            } else if (analysis.image_type === 'photo' || analysis.image_type === 'filtered_photo') {
                overrideContainer.style.display = 'block';
                overrideLabel.textContent = 'Tratar como Anime/Ilustración';
                overrideDescription.textContent = 'Actívalo solo si la imagen es realmente anime o ilustración digital.';
            } else {
                overrideContainer.style.display = 'none';
            }
        }

        const filterRestoreContainer = document.getElementById('filterRestoreContainer');
        const removeFilterBtn = document.getElementById('removeFilterBtn');
        const filterRestoreDescription = document.getElementById('filterRestoreDescription');
        const oldPhotoRestoreContainer = document.getElementById('oldPhotoRestoreContainer');
        const restoreOldPhotoBtn = document.getElementById('restoreOldPhotoBtn');
        const oldPhotoRestoreDescription = document.getElementById('oldPhotoRestoreDescription');
        const dualOutputContainer = document.getElementById('dualOutputContainer');
        const dualOutputBtn = document.getElementById('dualOutputBtn');
        const bwRestoreContainer = document.getElementById('bwRestoreContainer');
        const bwRestoreBtn = document.getElementById('bwRestoreBtn');

        const canRestoreColorFilter = Boolean(
            analysis.recommended_color_filter_correction
            || analysis.social_color_filter_detected
            || analysis.filter_detected
            || analysis.degraded_social_portrait
        );
        const canRestoreOldPhoto = Boolean(
            analysis.recommended_old_photo_restore
            || analysis.old_photo_detected
            || analysis.scan_artifacts_detected
        );
        if (filterRestoreContainer && removeFilterBtn) {
            filterRestoreContainer.style.display = canRestoreColorFilter ? 'block' : 'none';
            removeFilterBtn.checked = false;

            if (filterRestoreDescription) {
                if (analysis.degraded_social_portrait) {
                    filterRestoreDescription.textContent = 'Retrato social degradado detectado: corrige dominante de color y contraste para un look más natural.';
                } else {
                    filterRestoreDescription.textContent = 'Corrige dominante de color, saturación y contraste de filtros modernos (Instagram/iPhone/Snapchat).';
                }
            }
        }

        if (oldPhotoRestoreContainer && restoreOldPhotoBtn) {
            oldPhotoRestoreContainer.style.display = canRestoreOldPhoto ? 'block' : 'none';
            restoreOldPhotoBtn.checked = false;
            if (oldPhotoRestoreDescription) {
                oldPhotoRestoreDescription.textContent = 'Ideal para fotos antiguas o escaneadas con desvanecimiento, rayones, polvo y bajo contraste.';
            }
        }

        if (dualOutputContainer && dualOutputBtn) {
            dualOutputContainer.style.display = 'none';
            dualOutputBtn.checked = false;
        }

        if (bwRestoreContainer && bwRestoreBtn) {
            bwRestoreContainer.style.display = analysis.is_monochrome ? 'block' : 'none';
            bwRestoreBtn.checked = false;
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
        controlPanel.classList.remove('is-hidden');
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
        document.getElementById('controlPanel').style.display = 'none';
        const resultPanel = document.getElementById('resultPanel');
        resultPanel.style.display = 'flex';
        document.body.classList.add('modal-open');

        // Configurar imágenes para comparación
        const originalImg = document.getElementById('originalImage');
        const processedImg = document.getElementById('processedImage');
        const variantSelector = document.getElementById('resultVariantSelector');
        const downloadBtn = document.getElementById('downloadBtn');

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
            if (this._currentOriginalObjectUrl) {
                try {
                    URL.revokeObjectURL(this._currentOriginalObjectUrl);
                } catch (_) {
                    // Ignorar
                }
            }
            originalSrc = URL.createObjectURL(window.currentFile);
            this._currentOriginalObjectUrl = originalSrc;
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

        const outputVariants = Array.isArray(data.output_variants) && data.output_variants.length > 0
            ? data.output_variants
            : [{
                label: 'Resultado',
                output_filename: data.output_filename,
                is_default: true
            }];

        let selectedVariant = outputVariants.find((v) => v.is_default) || outputVariants[0];

        const loadComparison = async (variant) => {
            const processedSrc = `${APIClient.BASE_URL}/api/download/${variant.output_filename}?t=${new Date().getTime()}`;
            console.log('Processed image URL:', processedSrc);

            await Promise.all([
                originalSrc ? loadImage(originalImg, originalSrc, 'Original') : Promise.reject(new Error('No original source')),
                loadImage(processedImg, processedSrc, 'Processed')
            ]);

            this.fitComparisonContainer(originalImg, processedImg);
            this.setupComparisonSlider();
            this.setupZoomControls();
        };

        const renderVariantSelector = () => {
            if (!variantSelector) return;

            if (outputVariants.length <= 1) {
                variantSelector.style.display = 'none';
                variantSelector.innerHTML = '';
                return;
            }

            variantSelector.style.display = 'flex';
            variantSelector.innerHTML = outputVariants
                .map((variant, index) => {
                    const isActive = variant.output_filename === selectedVariant.output_filename;
                    const label = variant.label || `Versión ${index + 1}`;
                    return `<button class="variant-btn ${isActive ? 'active' : ''}" data-filename="${variant.output_filename}">${label}</button>`;
                })
                .join('');

            variantSelector.querySelectorAll('button[data-filename]').forEach((btn) => {
                btn.onclick = async () => {
                    const targetFile = btn.dataset.filename;
                    const targetVariant = outputVariants.find((item) => item.output_filename === targetFile);
                    if (!targetVariant) return;
                    selectedVariant = targetVariant;
                    renderVariantSelector();
                    try {
                        await loadComparison(selectedVariant);
                    } catch (error) {
                        UIController.showError('Error al cargar la variante seleccionada: ' + error.message);
                    }
                };
            });
        };

        loadComparison(selectedVariant)
            .then(() => {
                console.log('Comparison loaded successfully');
            })
            .catch((error) => {
                console.error('Error loading images:', error);
                UIController.showError('Error al cargar las imágenes: ' + error.message);
            });

        renderVariantSelector();

        const formattedScaleTime = this.formatDuration(data.processing_time_seconds);
        const formattedAnalysisTime = this.formatDuration(data.analysis_time_seconds);
        const totalSeconds = (
            this.toFiniteNumber(data.total_pipeline_time_seconds)
            ?? (
                (this.toFiniteNumber(data.processing_time_seconds) ?? 0)
                + (this.toFiniteNumber(data.analysis_time_seconds) ?? 0)
            )
        );
        const formattedTotalTime = this.formatDuration(totalSeconds);

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
                <span class="info-label">Corrección color:</span>
                <span class="info-value">${data.filter_restoration_applied ? 'Aplicada' : 'No'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Restauración antigua:</span>
                <span class="info-value">${data.old_photo_restoration_applied ? 'Aplicada' : 'No'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Restauración B/N:</span>
                <span class="info-value">${data.bw_restoration_applied ? 'Aplicada' : 'No'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Análisis:</span>
                <span class="info-value time-value"><span class="time-chip">${formattedAnalysisTime}</span></span>
            </div>
            <div class="info-row">
                <span class="info-label">Escalado:</span>
                <span class="info-value time-value"><span class="time-chip">${formattedScaleTime}</span></span>
            </div>
            <div class="info-row">
                <span class="info-label">Tiempo total:</span>
                <span class="info-value time-value"><span class="time-chip time-chip-total">${formattedTotalTime}</span></span>
            </div>
            ${data.type_overridden ? `
            <div class="info-row">
                <span class="info-label">Tipo corregido:</span>
                <span class="info-value">${data.analysis_image_type} → ${data.effective_image_type}</span>
            </div>` : ''}
            ${data.processing_warning ? `
            <div class="info-row">
                <span class="info-label">Aviso:</span>
                <span class="info-value" style="max-width: 70%; text-align: right; color: hsl(42, 100%, 74%);">${data.processing_warning}</span>
            </div>` : ''}
        `;

        // Configurar botón de descarga
        downloadBtn.onclick = () => {
            // Crear enlace temporal para descarga directa
            const link = document.createElement('a');
            link.href = APIClient.getDownloadUrl(selectedVariant.output_filename);
            link.download = selectedVariant.output_filename;
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

        resultPanel.onclick = (event) => {
            if (event.target === resultPanel && window.AppController && typeof window.AppController.onNewImageRequested === 'function') {
                window.AppController.onNewImageRequested();
            }
        };
    }

    static setupComparisonSlider() {
        const container = document.getElementById('comparisonContainer');
        const scroller = document.getElementById('scroller');
        const beforeWrapper = document.querySelector('.img-wrapper.before');
        const afterWrapper = document.getElementById('afterWrapper');
        const comparison = document.querySelector('.image-comparison');
        const divider = document.getElementById('comparisonDivider');

        if (!container || !scroller || !beforeWrapper || !afterWrapper || !comparison) {
            console.error('Comparison slider elements not found:', {
                container,
                scroller,
                beforeWrapper,
                afterWrapper,
                comparison
            });
            return;
        }

        // Reset state
        beforeWrapper.style.width = '100%';
        afterWrapper.style.width = '100%';
        const defaultSplit = 50;
        scroller.style.left = `${defaultSplit}%`;
        comparison.style.setProperty('--split-pos', `${defaultSplit}%`);
        afterWrapper.style.clipPath = `inset(0 0 0 ${defaultSplit}%)`;
        if (divider) {
            divider.style.left = `${defaultSplit}%`;
        }

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
            this.updateSliderPosition(e, container, scroller, comparison, afterWrapper, divider);
        });

        const stopDrag = () => {
            active = false;
            pointerId = null;
        };

        scroller.addEventListener('pointerup', stopDrag);
        scroller.addEventListener('pointercancel', stopDrag);
        scroller.addEventListener('lostpointercapture', stopDrag);
    }

    static fitComparisonContainer(originalImg, processedImg) {
        const container = document.getElementById('comparisonContainer');
        if (!container || !originalImg || !processedImg) return;

        const ow = originalImg.naturalWidth || 0;
        const oh = originalImg.naturalHeight || 0;
        const pw = processedImg.naturalWidth || ow;
        const ph = processedImg.naturalHeight || oh;
        const refW = Math.max(ow, pw);
        const refH = Math.max(oh, ph);
        if (refW <= 0 || refH <= 0) return;

        const viewportMax = Math.max(360, Math.floor(window.innerHeight * 0.72));
        const viewportMin = 280;
        const availableWidth = Math.max(320, container.clientWidth || container.parentElement?.clientWidth || 320);
        const fittedHeight = Math.round((availableWidth * refH) / refW);
        const finalHeight = Math.max(viewportMin, Math.min(viewportMax, fittedHeight));
        container.style.height = `${finalHeight}px`;
    }

    static updateSliderPosition(e, container, scroller, comparison, afterWrapper, divider = null) {
        const rect = container.getBoundingClientRect();
        let x = e.clientX - rect.left;

        // Limitar movimiento dentro del contenedor
        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;

        const percentage = (x / rect.width) * 100;

        scroller.style.left = `${percentage}%`;
        comparison.style.setProperty('--split-pos', `${percentage}%`);
        afterWrapper.style.clipPath = `inset(0 0 0 ${percentage}%)`;
        if (divider) {
            divider.style.left = `${percentage}%`;
        }
    }
    static setupZoomControls() {
        const zoomInBtn = document.getElementById('zoomInBtn');
        const zoomOutBtn = document.getElementById('zoomOutBtn');
        const resetZoomBtn = document.getElementById('resetZoomBtn');
        const zoomLevelDisplay = document.getElementById('zoomLevel');

        const originalImg = document.getElementById('originalImage');
        const processedImg = document.getElementById('processedImage');
        const container = document.getElementById('comparisonContainer');
        const resultPanel = document.getElementById('resultPanel');

        if (!zoomInBtn || !originalImg || !processedImg || !container || !zoomLevelDisplay) return;

        if (!this._zoomState) {
            this._zoomState = {
                scale: 1,
                panX: 0,
                panY: 0,
                baseWidth: 0,
                baseHeight: 0,
                baseDirty: true,
                lastTransform: '',
                rafPending: false
            };
        } else {
            this._zoomState.scale = 1;
            this._zoomState.panX = 0;
            this._zoomState.panY = 0;
            this._zoomState.baseDirty = true;
            this._zoomState.lastTransform = '';
            this._zoomState.rafPending = false;
        }
        const state = this._zoomState;

        let dragActive = false;
        let dragPointerId = null;
        let dragStartX = 0;
        let dragStartY = 0;
        let dragStartPanX = 0;
        let dragStartPanY = 0;

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

        const updateBaseSize = () => {
            const base = getDisplaySize(originalImg);
            state.baseWidth = base.width;
            state.baseHeight = base.height;
            state.baseDirty = false;
        };

        const clampPan = () => {
            if (state.baseDirty) {
                updateBaseSize();
            }

            const baseWidth = state.baseWidth || 1;
            const baseHeight = state.baseHeight || 1;
            const maxX = Math.max(0, ((baseWidth * state.scale) - baseWidth) * 0.5);
            const maxY = Math.max(0, ((baseHeight * state.scale) - baseHeight) * 0.5);
            state.panX = Math.max(-maxX, Math.min(maxX, state.panX));
            state.panY = Math.max(-maxY, Math.min(maxY, state.panY));
        };

        const setZoomMode = (isZoomed) => {
            if (resultPanel) {
                resultPanel.classList.toggle('is-zooming', isZoomed);
            }
            container.classList.toggle('is-zooming', isZoomed);
        };

        const applyTransform = () => {
            if (state.rafPending) return;
            state.rafPending = true;

            requestAnimationFrame(() => {
                state.rafPending = false;

                if (state.scale <= 1) {
                    state.scale = 1;
                    state.panX = 0;
                    state.panY = 0;
                } else {
                    state.scale = Math.min(5, state.scale);
                }

                clampPan();

                const transform = `translate3d(${state.panX}px, ${state.panY}px, 0) scale(${state.scale})`;
                if (state.lastTransform !== transform) {
                    originalImg.style.transform = transform;
                    processedImg.style.transform = transform;
                    state.lastTransform = transform;
                }

                zoomLevelDisplay.textContent = `${Math.round(state.scale * 100)}%`;
                const zoomed = state.scale > 1;
                setZoomMode(zoomed);
                container.style.cursor = zoomed ? (dragActive ? 'grabbing' : 'grab') : 'col-resize';
            });
        };

        originalImg.style.transformOrigin = '50% 50%';
        processedImg.style.transformOrigin = '50% 50%';

        const stopPanDrag = () => {
            if (dragPointerId !== null) {
                try {
                    container.releasePointerCapture(dragPointerId);
                } catch (_) {
                    // Ignore
                }
            }
            dragActive = false;
            dragPointerId = null;
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

        const onWheel = (e) => {
            e.preventDefault();
            const previousScale = state.scale;

            if (e.deltaY < 0) {
                state.scale += 0.1;
            } else {
                state.scale -= 0.1;
            }

            if (state.scale <= 1) {
                state.scale = 1;
                state.panX = 0;
                state.panY = 0;
            } else if (previousScale === 1 && state.scale > 1) {
                state.panX = 0;
                state.panY = 0;
            }

            applyTransform();
        };

        const onPointerDown = (e) => {
            if (e.button !== 0) return;
            if (state.scale <= 1) return;
            if (e.target && e.target.closest('#scroller')) return;

            dragActive = true;
            dragPointerId = e.pointerId;
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            dragStartPanX = state.panX;
            dragStartPanY = state.panY;
            container.setPointerCapture(dragPointerId);
            applyTransform();
            e.preventDefault();
        };

        const onPointerMove = (e) => {
            if (!dragActive || dragPointerId !== e.pointerId) return;
            state.panX = dragStartPanX + (e.clientX - dragStartX);
            state.panY = dragStartPanY + (e.clientY - dragStartY);
            applyTransform();
        };

        const onPointerUp = () => {
            stopPanDrag();
        };

        const onPointerCancel = () => {
            stopPanDrag();
        };

        const onLostPointerCapture = () => {
            stopPanDrag();
        };

        if (this._zoomHandlers && this._zoomHandlers.container) {
            const old = this._zoomHandlers;
            old.container.removeEventListener('wheel', old.onWheel);
            old.container.removeEventListener('pointerdown', old.onPointerDown);
            old.container.removeEventListener('pointermove', old.onPointerMove);
            old.container.removeEventListener('pointerup', old.onPointerUp);
            old.container.removeEventListener('pointercancel', old.onPointerCancel);
            old.container.removeEventListener('lostpointercapture', old.onLostPointerCapture);
        }

        container.addEventListener('wheel', onWheel, { passive: false });
        container.addEventListener('pointerdown', onPointerDown);
        container.addEventListener('pointermove', onPointerMove);
        container.addEventListener('pointerup', onPointerUp);
        container.addEventListener('pointercancel', onPointerCancel);
        container.addEventListener('lostpointercapture', onLostPointerCapture);

        this._zoomHandlers = {
            container,
            onWheel,
            onPointerDown,
            onPointerMove,
            onPointerUp,
            onPointerCancel,
            onLostPointerCapture
        };

        if (this._zoomResizeHandler) {
            window.removeEventListener('resize', this._zoomResizeHandler);
        }
        this._zoomResizeHandler = () => {
            state.baseDirty = true;
            applyTransform();
        };
        window.addEventListener('resize', this._zoomResizeHandler, { passive: true });

        state.baseDirty = true;
        applyTransform();
    }

    /**
     * Reinicia la aplicación para procesar una nueva imagen
     */
    static reset() {
        document.getElementById('controlPanel').style.display = 'none';
        document.getElementById('processingPanel').style.display = 'none';
        document.getElementById('resultPanel').style.display = 'none';
        document.body.classList.remove('modal-open');

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

    static toFiniteNumber(value) {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : null;
    }

    static formatDuration(secondsValue) {
        const secondsRaw = this.toFiniteNumber(secondsValue);
        if (secondsRaw === null || secondsRaw < 0) {
            return '-';
        }

        const seconds = Math.max(0, secondsRaw);
        if (seconds < 60) {
            const oneDecimal = Math.round(seconds * 10) / 10;
            const compact = Number.isInteger(oneDecimal) ? oneDecimal.toFixed(0) : oneDecimal.toFixed(1);
            return `${compact} s`;
        }

        const roundedSeconds = Math.round(seconds);
        const minutes = Math.floor(roundedSeconds / 60);
        const secondsRemainder = roundedSeconds % 60;

        if (minutes < 60) {
            return secondsRemainder > 0 ? `${minutes} min ${secondsRemainder} s` : `${minutes} min`;
        }

        const hours = Math.floor(minutes / 60);
        const minutesRemainder = minutes % 60;
        return minutesRemainder > 0 ? `${hours} h ${minutesRemainder} min` : `${hours} h`;
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
        if (id === 'resultPanel') {
            document.body.classList.remove('modal-open');
        }
    }

    static showElement(id) {
        const el = document.getElementById(id);
        if (el) el.style.display = 'block';
    }
}
