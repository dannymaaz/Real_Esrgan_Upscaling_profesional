/**
 * Lógica principal de la aplicación
 * Autor: Danny Maaz (github.com/dannymaaz)
 */

// Estado de la aplicación
let currentFile = null;
let currentAnalysis = null;
let selectedScale = null;
let isProcessing = false;

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

/**
 * Inicializa todos los event listeners
 */
function initializeEventListeners() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const newImageBtn = document.getElementById('newImageBtn');
    const closeErrorBtn = document.getElementById('closeErrorBtn');
    const scaleButtons = document.querySelectorAll('.scale-btn');

    // Dropzone click
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    });

    // Scale buttons
    scaleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            scaleButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedScale = btn.dataset.scale;
        });
    });

    // Process button
    processBtn.addEventListener('click', handleProcess);

    // New image button
    if (newImageBtn) {
        newImageBtn.addEventListener('click', () => {
            UIController.reset();
            currentFile = null;
            currentAnalysis = null;
            selectedScale = null;
        });
    }

    // Close error button
    closeErrorBtn.addEventListener('click', () => {
        UIController.closeError();
    });
}

/**
 * Maneja la selección de un archivo
 * @param {File} file - Archivo seleccionado
 */
async function handleFileSelect(file) {
    if (!file) return;

    // Validar tipo de archivo
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        UIController.showError('Por favor selecciona una imagen PNG o JPG');
        return;
    }

    // Validar tamaño (20MB máximo)
    const maxSize = 20 * 1024 * 1024;
    if (file.size > maxSize) {
        UIController.showError('La imagen es demasiado grande. Máximo 20MB');
        return;
    }

    currentFile = file;

    try {
        // Analizar imagen
        currentAnalysis = await APIClient.analyzeImage(file);

        // Mostrar panel de control con análisis
        UIController.showControlPanel(currentAnalysis);

        // Seleccionar escala recomendada por defecto
        selectedScale = `${currentAnalysis.recommended_scale}x`;

        // AUTO-DETECCIÓN DE ROSTROS
        const faceEnhanceBtn = document.getElementById('faceEnhanceBtn');
        if (faceEnhanceBtn) {
            // Activar automáticamente si tiene caras
            if (currentAnalysis.has_faces) {
                faceEnhanceBtn.checked = true;
                // Podríamos mostrar un mensaje también
            } else {
                faceEnhanceBtn.checked = false;
            }
        }

    } catch (error) {
        UIController.showError(error.message);
    }
}

/**
 * Maneja el procesamiento de la imagen
 */
async function handleProcess() {
    if (!currentFile) {
        UIController.showError('Por favor selecciona una imagen primero');
        return;
    }

    if (!selectedScale) {
        UIController.showError('Por favor selecciona una escala');
        return;
    }

    if (isProcessing) return;

    try {
        // Mostrar panel de procesamiento
        UIController.showProcessingPanel();
        isProcessing = true;

        // Obtener configuración de mejora de rostros
        const faceEnhanceBtn = document.getElementById('faceEnhanceBtn');
        const faceEnhance = faceEnhanceBtn ? faceEnhanceBtn.checked : false;

        // Determinar modelo a usar
        // Lógica: Si el usuario selecciona una escala, respetamos esa escala.
        // Solo sobrescribimos el modelo si es un caso especial (anime en 4x).
        // En 2x, dejamos que el backend use el default (que suele ser un modelo ligero o resize).

        let modelToSend = null;
        if (currentAnalysis?.image_type === 'anime' && selectedScale === '4x') {
            modelToSend = 'RealESRGAN_x4plus_anime_6B';
        }

        // Procesar imagen
        const result = await APIClient.upscaleImage(
            currentFile,
            selectedScale,
            modelToSend,
            faceEnhance
        );

        // Completar barra de progreso
        UIController.completeProgress();

        // Esperar un momento antes de mostrar resultados
        setTimeout(() => {
            UIController.showResultPanel(result);
        }, 500);

    } catch (error) {
        console.error("Processing error:", error);
        UIController.closeError();
        UIController.showError(error.message || "Error desconocido al procesar imagen");
        UIController.reset();
    } finally {
        isProcessing = false;
    }
}

/**
 * Prevenir comportamiento por defecto de drag and drop en toda la página
 */
window.addEventListener('dragover', (e) => {
    e.preventDefault();
}, false);

window.addEventListener('drop', (e) => {
    e.preventDefault();
}, false);
