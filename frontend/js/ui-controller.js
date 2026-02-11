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

        recommendationText.textContent =
            `Recomendamos usar escala ${analysis.recommended_scale}x para mejores resultados. ${analysis.analysis_notes}`;
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
     * Muestra el panel de resultados
     * @param {Object} result - Resultado del procesamiento
     */
    static showResultPanel(result) {
        document.getElementById('processingPanel').style.display = 'none';
        document.getElementById('resultPanel').style.display = 'block';

        // Actualizar información del resultado
        const resultInfo = document.getElementById('resultInfo');
        resultInfo.innerHTML = `
            <div class="info-row">
                <span class="info-label">Modelo usado:</span>
                <span class="info-value">${result.model_used}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Escala:</span>
                <span class="info-value">${result.scale}x</span>
            </div>
            <div class="info-row">
                <span class="info-label">Resolución original:</span>
                <span class="info-value">${result.original_size.width} × ${result.original_size.height}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Resolución final:</span>
                <span class="info-value">${result.output_size.width} × ${result.output_size.height}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Tamaño del archivo:</span>
                <span class="info-value">${result.output_size_mb} MB</span>
            </div>
            <div class="info-row">
                <span class="info-label">Dispositivo:</span>
                <span class="info-value">${result.device_used === 'cuda' ? 'GPU' : 'CPU'}</span>
            </div>
        `;

        // Configurar botón de descarga
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.onclick = () => {
            window.location.href = APIClient.getDownloadUrl(result.output_filename);
        };
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

        errorMessage.textContent = message;
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
}
