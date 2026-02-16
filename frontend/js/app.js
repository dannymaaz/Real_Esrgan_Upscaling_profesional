/**
 * Lógica principal de la aplicación
 * Autor: Danny Maaz (github.com/dannymaaz)
 */

const MAX_QUEUE_ITEMS = 30;
const MAX_HISTORY_ITEMS = 40;
const MAX_VISIBLE_QUEUE_ITEMS = 10;
const MAX_VISIBLE_HISTORY_ITEMS = 10;

// Estado de sesión (en memoria, se limpia al reiniciar la app)
const jobs = new Map();
const jobOrder = [];
const processQueue = [];
const historyItems = [];

let selectedJobId = null;
let isQueueWorkerRunning = false;
let showAllQueueItems = false;
let showAllHistoryItems = false;
let notificationPermissionRequested = false;
let activeProcessingJobId = null;
let activeAbortController = null;
let pendingCancelJobId = null;

let currentFile = null;
let currentAnalysis = null;
let selectedScale = null;

window.currentFile = null;
window.AppController = {
    onNewImageRequested: handleReturnToEditor
};

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializeAnimatedPanels();
    renderQueuePanel();
    renderHistoryPanel();
});

function initializeAnimatedPanels() {
    ['controlPanel', 'queuePanel', 'historyPanel'].forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.add('panel-fade');
    });
}

function setPanelVisibility(panelId, visible) {
    const panel = document.getElementById(panelId);
    if (!panel) return;

    panel.classList.add('panel-fade');
    if (visible) {
        panel.style.display = 'block';
        requestAnimationFrame(() => {
            panel.classList.remove('is-hidden');
        });
        return;
    }

    panel.classList.add('is-hidden');
    setTimeout(() => {
        if (panel.classList.contains('is-hidden')) {
            panel.style.display = 'none';
        }
    }, 220);
}

function initializeEventListeners() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const closeErrorBtn = document.getElementById('closeErrorBtn');
    const scaleButtons = document.querySelectorAll('.scale-btn');
    const queueList = document.getElementById('queueList');
    const historyList = document.getElementById('historyList');
    const clearQueueBtn = document.getElementById('clearQueueBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const toggleQueueViewBtn = document.getElementById('toggleQueueViewBtn');
    const toggleHistoryViewBtn = document.getElementById('toggleHistoryViewBtn');
    const removeFilterBtn = document.getElementById('removeFilterBtn');
    const restoreOldPhotoBtn = document.getElementById('restoreOldPhotoBtn');
    const dualOutputContainer = document.getElementById('dualOutputContainer');
    const dualOutputBtn = document.getElementById('dualOutputBtn');
    const bwRestoreBtn = document.getElementById('bwRestoreBtn');
    const faceEnhanceBtn = document.getElementById('faceEnhanceBtn');
    const imageTypeOverrideBtn = document.getElementById('imageTypeOverrideBtn');
    const cancelConfirmModal = document.getElementById('cancelConfirmModal');
    const cancelConfirmYesBtn = document.getElementById('cancelConfirmYesBtn');
    const cancelConfirmNoBtn = document.getElementById('cancelConfirmNoBtn');

    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
        fileInput.value = '';
    });

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
        handleFileSelect(e.dataTransfer.files[0]);
    });

    scaleButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            scaleButtons.forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            selectedScale = btn.dataset.scale;

            const job = getSelectedJob();
            if (job) {
                job.options.scale = selectedScale;
                job.estimatedDurationSec = estimateDurationSeconds(job.analysis, job.options);
                jobs.set(job.id, job);
                renderQueuePanel();
            }
        });
    });

    processBtn.addEventListener('click', enqueueSelectedJob);

    closeErrorBtn.addEventListener('click', () => {
        UIController.closeError();
    });

    queueList.addEventListener('click', handleQueueActionClick);
    historyList.addEventListener('click', handleHistoryActionClick);

    clearQueueBtn.addEventListener('click', clearPendingQueue);
    clearHistoryBtn.addEventListener('click', clearHistory);

    if (toggleQueueViewBtn) {
        toggleQueueViewBtn.addEventListener('click', () => {
            showAllQueueItems = !showAllQueueItems;
            renderQueuePanel();
        });
    }

    if (toggleHistoryViewBtn) {
        toggleHistoryViewBtn.addEventListener('click', () => {
            showAllHistoryItems = !showAllHistoryItems;
            renderHistoryPanel();
        });
    }

    if (removeFilterBtn && dualOutputContainer) {
        removeFilterBtn.addEventListener('change', () => {
            const selected = getSelectedJob();
            if (selected) {
                selected.options.removeColorFilter = Boolean(removeFilterBtn.checked);
                if (!(selected.options.removeColorFilter || selected.options.restoreOldPhoto || selected.options.restoreMonochrome)) {
                    selected.options.dualOutput = false;
                }
                selected.estimatedDurationSec = estimateDurationSeconds(selected.analysis, selected.options);
                jobs.set(selected.id, selected);
                refreshDualOutputVisibilityForJob(selected);
                renderQueuePanel();
            } else {
                refreshDualOutputVisibilityForJob(null);
            }

            refreshOptionCardsState();
        });
    }

    if (restoreOldPhotoBtn) {
        restoreOldPhotoBtn.addEventListener('change', () => {
            const selected = getSelectedJob();
            if (!selected) {
                refreshDualOutputVisibilityForJob(null);
                return;
            }

            selected.options.restoreOldPhoto = Boolean(restoreOldPhotoBtn.checked);
            if (!(selected.options.removeColorFilter || selected.options.restoreOldPhoto || selected.options.restoreMonochrome)) {
                selected.options.dualOutput = false;
            }
            selected.estimatedDurationSec = estimateDurationSeconds(selected.analysis, selected.options);
            jobs.set(selected.id, selected);
            refreshDualOutputVisibilityForJob(selected);
            renderQueuePanel();
            refreshOptionCardsState();
        });
    }

    if (dualOutputBtn) {
        dualOutputBtn.addEventListener('change', () => {
            const selected = getSelectedJob();
            if (!selected) return;
            selected.options.dualOutput = Boolean(
                dualOutputBtn.checked
                && (selected.options.removeColorFilter || selected.options.restoreOldPhoto || selected.options.restoreMonochrome)
            );
            selected.estimatedDurationSec = estimateDurationSeconds(selected.analysis, selected.options);
            jobs.set(selected.id, selected);
            refreshDualOutputWarning(selected);
            refreshOptionCardsState();
            renderQueuePanel();
        });
    }

    if (bwRestoreBtn) {
        bwRestoreBtn.addEventListener('change', () => {
            const selected = getSelectedJob();
            if (!selected) return;
            selected.options.restoreMonochrome = Boolean(bwRestoreBtn.checked);
            if (!(selected.options.removeColorFilter || selected.options.restoreOldPhoto || selected.options.restoreMonochrome)) {
                selected.options.dualOutput = false;
            }
            selected.estimatedDurationSec = estimateDurationSeconds(selected.analysis, selected.options);
            jobs.set(selected.id, selected);
            refreshDualOutputVisibilityForJob(selected);
            renderQueuePanel();
            refreshOptionCardsState();
        });
    }

    if (faceEnhanceBtn) {
        faceEnhanceBtn.addEventListener('change', () => {
            const selected = getSelectedJob();
            if (!selected) return;
            selected.options.faceEnhance = Boolean(faceEnhanceBtn.checked);
            selected.estimatedDurationSec = estimateDurationSeconds(selected.analysis, selected.options);
            jobs.set(selected.id, selected);
            renderQueuePanel();
            refreshOptionCardsState();
        });
    }

    if (imageTypeOverrideBtn) {
        imageTypeOverrideBtn.addEventListener('change', () => {
            const selected = getSelectedJob();
            if (!selected) return;
            selected.options.forcedImageType = getForcedImageTypeFromUI(selected.analysis);
            jobs.set(selected.id, selected);
            renderQueuePanel();
            refreshOptionCardsState();
        });
    }

    if (cancelConfirmYesBtn) {
        cancelConfirmYesBtn.addEventListener('click', () => {
            const targetJobId = pendingCancelJobId;
            closeCancelConfirmModal();
            if (targetJobId) {
                cancelProcessingJob(targetJobId);
            }
        });
    }

    if (cancelConfirmNoBtn) {
        cancelConfirmNoBtn.addEventListener('click', () => {
            closeCancelConfirmModal();
        });
    }

    if (cancelConfirmModal) {
        cancelConfirmModal.addEventListener('click', (event) => {
            if (event.target === cancelConfirmModal) {
                closeCancelConfirmModal();
            }
        });
    }

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            const isCancelModalOpen = Boolean(cancelConfirmModal && cancelConfirmModal.style.display !== 'none');
            if (isCancelModalOpen) {
                closeCancelConfirmModal();
                return;
            }

            const resultPanel = document.getElementById('resultPanel');
            if (resultPanel && resultPanel.style.display !== 'none') {
                handleReturnToEditor();
            }
        }
    });

    refreshOptionCardsState();
}

function getSelectedJob() {
    if (!selectedJobId) return null;
    return jobs.get(selectedJobId) || null;
}

function generateJobId() {
    return `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function formatError(error, fallbackMessage) {
    const message = (error && error.message) ? String(error.message) : '';
    if (error && error.name === 'AbortError') {
        return 'Procesamiento cancelado por el usuario.';
    }
    if (message.includes('Failed to fetch')) {
        return 'No se pudo conectar con el servidor. Verifica que el backend siga activo en http://127.0.0.1:8000';
    }
    return message || fallbackMessage;
}

function isAbortError(error) {
    if (!error) return false;
    if (error.name === 'AbortError') return true;
    const msg = String(error.message || '').toLowerCase();
    return msg.includes('cancelad') || msg.includes('aborted');
}

function formatDurationShort(seconds) {
    const total = Math.max(0, Number(seconds) || 0);
    if (total < 60) {
        const rounded = Math.round(total);
        return `${Math.max(1, rounded)} s`;
    }

    const minutes = Math.floor(total / 60);
    const remainder = Math.round(total % 60);
    if (remainder === 0) {
        return `${minutes} min`;
    }
    return `${minutes} min ${remainder} s`;
}

function estimateDurationSeconds(analysis, options) {
    const megapixels = Number(analysis?.megapixels || 1);
    const requestedScale = options?.scale === '4x' ? 4 : 2;
    const base = requestedScale === 4 ? 16 : 7;
    let estimate = base + (megapixels * (requestedScale === 4 ? 13 : 7));

    if (options?.faceEnhance) {
        estimate *= 1.15;
    }
    if (options?.removeColorFilter) {
        estimate *= 1.18;
    }
    if (options?.restoreOldPhoto) {
        estimate *= 1.24;
    }
    if (options?.restoreMonochrome) {
        estimate *= 1.22;
    }
    if (options?.dualOutput && (options?.removeColorFilter || options?.restoreOldPhoto || options?.restoreMonochrome)) {
        estimate *= 1.95;
    }

    return Math.max(8, Math.round(estimate));
}

function updateJobProgress(job, elapsedSeconds) {
    const previousProgress = Number(job.progress || 0);
    const previousMessage = String(job.progressMessage || '');
    const estimate = Math.max(8, Number(job.estimatedDurationSec || 20));
    const ratio = Math.max(0, Math.min(0.94, elapsedSeconds / estimate));
    const eased = 1 - Math.pow(1 - ratio, 1.6);
    job.progress = Math.round(eased * 100);
    const remaining = Math.max(0, estimate - elapsedSeconds);
    job.progressMessage = `ETA aprox: ${formatDurationShort(remaining)}`;

    return previousProgress !== job.progress || previousMessage !== job.progressMessage;
}

function refreshDualOutputVisibilityForJob(job) {
    const dualOutputContainer = document.getElementById('dualOutputContainer');
    const dualOutputBtn = document.getElementById('dualOutputBtn');
    if (!dualOutputContainer || !dualOutputBtn) return;

    const canEnableDual = Boolean(job && (job.options?.removeColorFilter || job.options?.restoreOldPhoto || job.options?.restoreMonochrome));
    dualOutputContainer.style.display = canEnableDual ? 'block' : 'none';
    if (!canEnableDual) {
        dualOutputBtn.checked = false;
    }

    refreshDualOutputWarning(job);
    refreshOptionCardsState();
}

function refreshDualOutputWarning(job = null) {
    const dualOutputBtn = document.getElementById('dualOutputBtn');
    const dualOutputWarning = document.getElementById('dualOutputWarning');
    if (!dualOutputBtn || !dualOutputWarning) return;

    if (!dualOutputBtn.checked) {
        dualOutputWarning.style.display = 'none';
        return;
    }

    let warningText = 'Activado: se generarán 2 resultados, por eso tardará más.';
    if (job && job.analysis && job.options) {
        const singleEstimate = estimateDurationSeconds(job.analysis, {
            ...job.options,
            dualOutput: false
        });
        const dualEstimate = estimateDurationSeconds(job.analysis, {
            ...job.options,
            dualOutput: true
        });
        const extraSeconds = Math.max(0, dualEstimate - singleEstimate);
        if (extraSeconds > 0) {
            warningText = `Activado: se generarán 2 resultados (+${formatDurationShort(extraSeconds)} aprox).`;
        }
    }

    dualOutputWarning.textContent = warningText;
    dualOutputWarning.style.display = 'block';
}

function refreshOptionCardsState() {
    document.querySelectorAll('.processing-options').forEach((card) => {
        const checkbox = card.querySelector('input[type="checkbox"]');
        card.classList.toggle('is-checked', Boolean(checkbox && checkbox.checked));
    });
}

function validateFile(file) {
    if (!file) {
        return 'No se recibió ningún archivo.';
    }

    const supportedMimeTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/heic', 'image/heif', 'image/heic-sequence', 'image/heif-sequence'];
    const supportedExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif'];
    const extension = file.name ? `.${file.name.split('.').pop().toLowerCase()}` : '';
    const mimeType = (file.type || '').toLowerCase();

    const isSupportedMime = supportedMimeTypes.includes(mimeType);
    const isSupportedExtension = supportedExtensions.includes(extension);
    if (!isSupportedMime && !isSupportedExtension) {
        return 'Este archivo no es compatible por ahora. Formatos soportados: PNG, JPG, JPEG, WEBP, HEIC y HEIF.';
    }

    const maxSize = 20 * 1024 * 1024;
    if (file.size > maxSize) {
        return 'La imagen es demasiado grande. Máximo 20MB.';
    }

    if (jobs.size >= MAX_QUEUE_ITEMS) {
        return `Límite alcanzado: máximo ${MAX_QUEUE_ITEMS} imágenes en cola/historial activo de sesión.`;
    }

    return null;
}

async function handleFileSelect(file) {
    const validationError = validateFile(file);
    if (validationError) {
        UIController.showError(validationError);
        return;
    }

    try {
        const analysis = await APIClient.analyzeImage(file);

        const job = {
            id: generateJobId(),
            file,
            createdAt: Date.now(),
            updatedAt: Date.now(),
            status: 'ready',
            progress: 0,
            progressMessage: 'Lista para encolar',
            startedAt: null,
            estimatedDurationSec: estimateDurationSeconds(analysis, { scale: `${analysis.recommended_scale}x`, faceEnhance: false, removeColorFilter: false, restoreOldPhoto: false, dualOutput: false, restoreMonochrome: false }),
            analysis,
            result: null,
            error: null,
            options: {
                scale: `${analysis.recommended_scale}x`,
                faceEnhance: false,
                forcedImageType: null,
                removeColorFilter: false,
                restoreOldPhoto: false,
                dualOutput: false,
                restoreMonochrome: false
            }
        };

        jobs.set(job.id, job);
        jobOrder.push(job.id);
        selectJob(job.id);
        renderQueuePanel();
    } catch (error) {
        UIController.showError(formatError(error, 'Error al analizar la imagen'));
    }
}

function applyJobControls(job, showPanel = true) {
    currentFile = job.file;
    currentAnalysis = job.analysis;
    selectedScale = job.options.scale;
    window.currentFile = job.file;

    if (showPanel) {
        UIController.showControlPanel(job.analysis);
    }
    setActiveScaleButton(job.options.scale);

    const faceEnhanceBtn = document.getElementById('faceEnhanceBtn');
    if (faceEnhanceBtn) {
        faceEnhanceBtn.checked = Boolean(job.options.faceEnhance);
    }

    const removeFilterBtn = document.getElementById('removeFilterBtn');
    const filterRestoreContainer = document.getElementById('filterRestoreContainer');
    const restoreOldPhotoBtn = document.getElementById('restoreOldPhotoBtn');
    const oldPhotoRestoreContainer = document.getElementById('oldPhotoRestoreContainer');
    const dualOutputContainer = document.getElementById('dualOutputContainer');
    const dualOutputBtn = document.getElementById('dualOutputBtn');
    const bwRestoreContainer = document.getElementById('bwRestoreContainer');
    const bwRestoreBtn = document.getElementById('bwRestoreBtn');

    const canColorFilterCorrection = Boolean(
        job.analysis.recommended_color_filter_correction
        ||
        job.analysis.social_color_filter_detected
        || job.analysis.filter_detected
        || job.analysis.degraded_social_portrait
    );
    const canRestoreOldPhoto = Boolean(
        job.analysis.recommended_old_photo_restore
        || job.analysis.old_photo_detected
        || job.analysis.scan_artifacts_detected
    );
    const canRestoreBw = Boolean(job.analysis.is_monochrome);

    if (filterRestoreContainer && removeFilterBtn) {
        filterRestoreContainer.style.display = canColorFilterCorrection ? 'block' : 'none';
        removeFilterBtn.checked = canColorFilterCorrection && Boolean(job.options.removeColorFilter);
    }

    if (oldPhotoRestoreContainer && restoreOldPhotoBtn) {
        oldPhotoRestoreContainer.style.display = canRestoreOldPhoto ? 'block' : 'none';
        restoreOldPhotoBtn.checked = canRestoreOldPhoto && Boolean(job.options.restoreOldPhoto);
    }

    if (dualOutputContainer && dualOutputBtn) {
        const canEnableDual =
            (canColorFilterCorrection && Boolean(job.options.removeColorFilter))
            || (canRestoreOldPhoto && Boolean(job.options.restoreOldPhoto))
            || (canRestoreBw && Boolean(job.options.restoreMonochrome));
        dualOutputContainer.style.display = canEnableDual ? 'block' : 'none';
        dualOutputBtn.checked = canEnableDual && Boolean(job.options.dualOutput);
    }

    if (bwRestoreContainer && bwRestoreBtn) {
        bwRestoreContainer.style.display = canRestoreBw ? 'block' : 'none';
        bwRestoreBtn.checked = canRestoreBw && Boolean(job.options.restoreMonochrome);
    }

    const overrideContainer = document.getElementById('imageTypeOverrideContainer');
    const overrideBtn = document.getElementById('imageTypeOverrideBtn');
    if (overrideContainer && overrideBtn && overrideContainer.style.display !== 'none') {
        overrideBtn.checked = Boolean(job.options.forcedImageType);
    }

    refreshDualOutputWarning(job);
    refreshOptionCardsState();
}

function selectJob(jobId) {
    const job = jobs.get(jobId);
    if (!job) return;

    selectedJobId = job.id;
    const shouldShowPanel = job.status === 'ready' || job.status === 'failed';
    applyJobControls(job, shouldShowPanel);
    if (job.status === 'ready' || job.status === 'failed') {
        setPanelVisibility('controlPanel', true);
    } else {
        setPanelVisibility('controlPanel', false);
    }
    renderQueuePanel();
}

function setActiveScaleButton(scale) {
    const scaleButtons = document.querySelectorAll('.scale-btn');
    scaleButtons.forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.scale === scale);
    });
}

function getForcedImageTypeFromUI(analysis) {
    const overrideBtn = document.getElementById('imageTypeOverrideBtn');
    if (!overrideBtn || !overrideBtn.checked || !analysis?.image_type) {
        return null;
    }

    if (analysis.image_type === 'anime') {
        return 'photo';
    }

    if (analysis.image_type === 'photo' || analysis.image_type === 'filtered_photo') {
        return 'anime';
    }

    return null;
}

function syncSelectedJobOptionsFromUI() {
    const job = getSelectedJob();
    if (!job) return null;

    const activeScaleBtn = document.querySelector('.scale-btn.active');
    const faceEnhanceBtn = document.getElementById('faceEnhanceBtn');
    const removeFilterBtn = document.getElementById('removeFilterBtn');
    const restoreOldPhotoBtn = document.getElementById('restoreOldPhotoBtn');
    const dualOutputBtn = document.getElementById('dualOutputBtn');
    const bwRestoreBtn = document.getElementById('bwRestoreBtn');

    job.options.scale = activeScaleBtn ? activeScaleBtn.dataset.scale : job.options.scale;
    job.options.faceEnhance = Boolean(faceEnhanceBtn && faceEnhanceBtn.checked);
    job.options.forcedImageType = getForcedImageTypeFromUI(job.analysis);
    job.options.removeColorFilter = Boolean(removeFilterBtn && removeFilterBtn.checked);
    job.options.restoreOldPhoto = Boolean(restoreOldPhotoBtn && restoreOldPhotoBtn.checked);
    job.options.restoreMonochrome = Boolean(bwRestoreBtn && bwRestoreBtn.checked && job.analysis?.is_monochrome);
    job.options.dualOutput = Boolean(
        dualOutputBtn
        && dualOutputBtn.checked
        && (job.options.removeColorFilter || job.options.restoreOldPhoto || job.options.restoreMonochrome)
    );
    job.estimatedDurationSec = estimateDurationSeconds(job.analysis, job.options);
    job.updatedAt = Date.now();

    jobs.set(job.id, job);
    selectedScale = job.options.scale;
    refreshDualOutputWarning(job);
    refreshOptionCardsState();
    return job;
}

function enqueueSelectedJob() {
    const job = syncSelectedJobOptionsFromUI();
    if (!job) {
        UIController.showError('Primero analiza una imagen para poder agregarla a la cola.');
        return;
    }

    if (job.status === 'queued' || job.status === 'processing') {
        UIController.showError('Esta imagen ya está en cola o procesándose.');
        return;
    }

    job.status = 'queued';
    job.error = null;
    job.progress = 0;
    job.progressMessage = 'Esperando turno';
    job.estimatedDurationSec = estimateDurationSeconds(job.analysis, job.options);
    job.updatedAt = Date.now();
    jobs.set(job.id, job);

    if (!processQueue.includes(job.id)) {
        processQueue.push(job.id);
    }

    setPanelVisibility('controlPanel', false);
    ensureNotificationPermission();
    renderQueuePanel();
    runQueueWorker();
}

async function runQueueWorker() {
    if (isQueueWorkerRunning) return;
    isQueueWorkerRunning = true;

    while (processQueue.length > 0) {
        const jobId = processQueue.shift();
        const job = jobs.get(jobId);
        if (!job || job.status !== 'queued') {
            continue;
        }

        job.status = 'processing';
        job.error = null;
        job.progress = Math.max(2, job.progress || 0);
        job.progressMessage = 'Preparando procesamiento...';
        job.startedAt = Date.now();
        job.updatedAt = Date.now();
        jobs.set(job.id, job);
        renderQueuePanel();

        let progressTimer = window.setInterval(() => {
            const current = jobs.get(job.id);
            if (!current || current.status !== 'processing') {
                clearInterval(progressTimer);
                return;
            }

            const elapsedSec = (Date.now() - Number(current.startedAt || Date.now())) / 1000;
            const changed = updateJobProgress(current, elapsedSec);
            if (changed || Math.floor(elapsedSec) % 2 === 0) {
                current.updatedAt = Date.now();
                jobs.set(current.id, current);
                renderQueuePanel();
            }
        }, 350);

        activeProcessingJobId = job.id;
        activeAbortController = new AbortController();

        try {
            const effectiveImageType = job.options.forcedImageType || job.analysis.image_type;
            let modelToSend = null;
            if (effectiveImageType === 'anime' && job.options.scale === '4x') {
                modelToSend = 'RealESRGAN_x4plus_anime_6B';
            }

            const result = await APIClient.upscaleImage(
                job.file,
                job.options.scale,
                modelToSend,
                job.options.faceEnhance,
                job.options.forcedImageType,
                {
                    removeColorFilter: job.options.removeColorFilter,
                    restoreOldPhoto: job.options.restoreOldPhoto,
                    dualOutput: job.options.dualOutput,
                    restoreMonochrome: job.options.restoreMonochrome,
                    signal: activeAbortController.signal
                }
            );

            clearInterval(progressTimer);

            job.status = 'done';
            job.result = result;
            job.progress = 100;
            job.progressMessage = 'Completado';
            job.updatedAt = Date.now();
            jobs.set(job.id, job);

            pushHistory(job);
            notifyJobCompleted(job);

            if (selectedJobId === job.id) {
                showResultForJob(job);
            }
        } catch (error) {
            clearInterval(progressTimer);
            if (isAbortError(error)) {
                job.status = 'cancelled';
                job.error = null;
                job.progress = 100;
                job.progressMessage = 'Cancelada';
            } else {
                job.status = 'failed';
                job.error = formatError(error, 'Error desconocido al procesar imagen');
                job.progress = 100;
                job.progressMessage = 'Finalizado con error';
            }
            job.updatedAt = Date.now();
            jobs.set(job.id, job);

            if (selectedJobId === job.id && job.status === 'failed') {
                UIController.showError(job.error);
            }
        } finally {
            activeProcessingJobId = null;
            activeAbortController = null;
        }

        renderQueuePanel();
    }

    isQueueWorkerRunning = false;
}

function showResultForJob(job) {
    if (!job || !job.result) {
        UIController.showError('Este elemento aún no tiene un resultado disponible.');
        return;
    }

    window.currentFile = job.file;
    currentFile = job.file;
    UIController.showResultPanel(job.result);
}

function removeFromQueue(jobId) {
    const index = processQueue.indexOf(jobId);
    if (index >= 0) {
        processQueue.splice(index, 1);
    }
}

function removeCompletedFromQueue(jobId) {
    const job = jobs.get(jobId);
    if (!job || job.status !== 'done') {
        return;
    }

    const orderIndex = jobOrder.indexOf(jobId);
    if (orderIndex >= 0) {
        jobOrder.splice(orderIndex, 1);
    }

    renderQueuePanel();
}

function openCancelConfirmModal(jobId) {
    const modal = document.getElementById('cancelConfirmModal');
    if (!modal) {
        if (window.confirm('¿Seguro que deseas cancelar este procesamiento?')) {
            cancelProcessingJob(jobId);
        }
        return;
    }

    pendingCancelJobId = jobId;
    modal.style.display = 'flex';
}

function closeCancelConfirmModal() {
    pendingCancelJobId = null;
    const modal = document.getElementById('cancelConfirmModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function cancelProcessingJob(jobId) {
    const job = jobs.get(jobId);
    if (!job || job.status !== 'processing') {
        return;
    }

    if (activeProcessingJobId === jobId && activeAbortController) {
        try {
            activeAbortController.abort();
        } catch (_) {
            // Ignorar fallos de abort.
        }
    }

    job.progressMessage = 'Cancelando...';
    job.updatedAt = Date.now();
    jobs.set(job.id, job);
    renderQueuePanel();
}

function removeJob(jobId) {
    const job = jobs.get(jobId);
    if (!job) return;

    if (job.status === 'processing') {
        UIController.showError('No se puede eliminar una tarea que está en procesamiento.');
        return;
    }

    removeFromQueue(jobId);
    jobs.delete(jobId);

    const orderIndex = jobOrder.indexOf(jobId);
    if (orderIndex >= 0) {
        jobOrder.splice(orderIndex, 1);
    }

    for (let i = historyItems.length - 1; i >= 0; i -= 1) {
        if (historyItems[i].jobId === jobId) {
            historyItems.splice(i, 1);
        }
    }

    if (selectedJobId === jobId) {
        selectedJobId = null;
        const fallbackId = jobOrder[jobOrder.length - 1] || null;
        if (fallbackId) {
            selectJob(fallbackId);
        } else {
            UIController.reset();
        }
    }

    renderQueuePanel();
    renderHistoryPanel();
}

function retryJob(jobId) {
    const job = jobs.get(jobId);
    if (!job) return;
    if (job.status === 'processing') return;

    job.status = 'queued';
    job.error = null;
    job.progress = 0;
    job.progressMessage = 'Esperando turno';
    job.estimatedDurationSec = estimateDurationSeconds(job.analysis, job.options);
    job.updatedAt = Date.now();
    jobs.set(job.id, job);

    if (!processQueue.includes(job.id)) {
        processQueue.push(job.id);
    }

    renderQueuePanel();
    runQueueWorker();
}

function clearPendingQueue() {
    for (const jobId of [...processQueue]) {
        const job = jobs.get(jobId);
        if (!job) continue;
        if (job.status === 'queued') {
            job.status = 'ready';
            job.progress = 0;
            job.progressMessage = 'Lista para encolar';
            job.updatedAt = Date.now();
            jobs.set(job.id, job);
        }
    }
    processQueue.length = 0;
    showAllQueueItems = false;
    renderQueuePanel();
}

function pruneQueueBacklog() {
    const keepLimit = MAX_QUEUE_ITEMS;
    if (jobOrder.length <= keepLimit) {
        return;
    }

    const removableIds = jobOrder.filter((jobId) => {
        const job = jobs.get(jobId);
        if (!job) return false;
        if (job.id === selectedJobId) return false;
        return job.status === 'done' || job.status === 'failed' || job.status === 'cancelled';
    });

    while (jobOrder.length > keepLimit && removableIds.length > 0) {
        const targetId = removableIds.shift();
        const orderIndex = jobOrder.indexOf(targetId);
        if (orderIndex >= 0) {
            jobOrder.splice(orderIndex, 1);
        }
        jobs.delete(targetId);
    }
}

function pushHistory(job) {
    const existingIndex = historyItems.findIndex((item) => item.jobId === job.id);
    if (existingIndex >= 0) {
        historyItems.splice(existingIndex, 1);
    }

    historyItems.unshift({
        jobId: job.id,
        title: job.file?.name || 'Imagen',
        outputFilename: job.result?.output_filename,
        model: job.result?.model_used,
        scale: job.result?.scale,
        variantCount: Array.isArray(job.result?.output_variants) ? job.result.output_variants.length : 1,
        processedAt: Date.now()
    });

    if (historyItems.length > MAX_HISTORY_ITEMS) {
        historyItems.length = MAX_HISTORY_ITEMS;
    }

    pruneQueueBacklog();
    renderHistoryPanel();
}

function clearHistory() {
    historyItems.length = 0;
    showAllHistoryItems = false;
    renderHistoryPanel();
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
}

function statusLabel(status) {
    switch (status) {
        case 'ready':
            return 'Lista';
        case 'queued':
            return 'En cola';
        case 'processing':
            return 'Procesando';
        case 'done':
            return 'Completada';
        case 'cancelled':
            return 'Cancelada';
        case 'failed':
            return 'Con error';
        default:
            return status;
    }
}

function escapeHtml(text) {
    return String(text || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function renderQueuePanel() {
    const queuePanel = document.getElementById('queuePanel');
    const queueSummary = document.getElementById('queueSummary');
    const queueList = document.getElementById('queueList');
    const toggleQueueViewBtn = document.getElementById('toggleQueueViewBtn');

    if (jobOrder.length === 0) {
        setPanelVisibility('queuePanel', false);
        queueList.innerHTML = '';
        if (toggleQueueViewBtn) toggleQueueViewBtn.style.display = 'none';
        return;
    }

    setPanelVisibility('queuePanel', true);

    const activeQueueJobs = jobOrder
        .map((jobId) => jobs.get(jobId))
        .filter((job) => Boolean(job));

    const queuedCount = activeQueueJobs.filter((job) => job.status === 'queued').length;
    const processingCount = activeQueueJobs.filter((job) => job.status === 'processing').length;
    const doneCount = activeQueueJobs.filter((job) => job.status === 'done').length;
    const failedCount = activeQueueJobs.filter((job) => job.status === 'failed').length;
    const cancelledCount = activeQueueJobs.filter((job) => job.status === 'cancelled').length;

    const hiddenCount = Math.max(0, jobOrder.length - MAX_VISIBLE_QUEUE_ITEMS);
    queueSummary.textContent = `Total: ${jobOrder.length} · En cola: ${queuedCount} · Procesando: ${processingCount} · Completadas: ${doneCount} · Canceladas: ${cancelledCount} · Errores: ${failedCount}`;

    if (toggleQueueViewBtn) {
        toggleQueueViewBtn.style.display = hiddenCount > 0 ? 'inline-flex' : 'none';
        toggleQueueViewBtn.textContent = showAllQueueItems ? 'Ver menos' : `Ver más (${hiddenCount})`;
    }

    const orderedJobIds = jobOrder.slice().reverse();
    const visibleJobIds = showAllQueueItems ? orderedJobIds : orderedJobIds.slice(0, MAX_VISIBLE_QUEUE_ITEMS);

    queueList.innerHTML = visibleJobIds
        .map((jobId) => {
            const job = jobs.get(jobId);
            if (!job) return '';

            const isActive = selectedJobId === job.id;
            const dimensions = `${job.analysis.width}x${job.analysis.height}`;
            const optionMeta = `${job.options.scale} · Rostro ${job.options.faceEnhance ? 'ON' : 'OFF'}${job.options.removeColorFilter ? ' · Color correction' : ''}${job.options.restoreOldPhoto ? ' · Old-photo restore' : ''}${job.options.restoreMonochrome ? ' · B/N restore' : ''}${job.options.dualOutput ? ' · Doble salida' : ''}`;

            const actionButtons = [];
            actionButtons.push(`<button class="mini-btn" data-action="select" data-job-id="${job.id}">Abrir</button>`);

            if (job.status === 'done') {
                actionButtons.push(`<button class="mini-btn" data-action="download" data-job-id="${job.id}">Descargar</button>`);
                actionButtons.push(`<button class="mini-btn" data-action="retry" data-job-id="${job.id}">Reprocesar</button>`);
            } else if (job.status === 'processing') {
                actionButtons.push(`<button class="mini-btn danger" data-action="cancel" data-job-id="${job.id}">Cancelar</button>`);
            } else if (job.status === 'failed') {
                actionButtons.push(`<button class="mini-btn" data-action="retry" data-job-id="${job.id}">Reintentar</button>`);
            } else if (job.status === 'cancelled') {
                actionButtons.push(`<button class="mini-btn" data-action="retry" data-job-id="${job.id}">Reintentar</button>`);
            } else if (job.status === 'ready') {
                actionButtons.push(`<button class="mini-btn" data-action="enqueue" data-job-id="${job.id}">Encolar</button>`);
            }

            if (job.status !== 'processing') {
                actionButtons.push(`<button class="mini-btn danger" data-action="remove" data-job-id="${job.id}">Quitar</button>`);
            }

            const errorLine = job.error
                ? `<div class="queue-item-meta" style="color:hsl(355,100%,82%);margin-top:0.35rem;">${escapeHtml(job.error)}</div>`
                : '';

            const progressValue = Math.max(0, Math.min(100, Number(job.progress || 0)));
            const progressLine = `
                <div class="queue-progress">
                    <div class="queue-progress-meta">
                        <span>${escapeHtml(job.progressMessage || statusLabel(job.status))}</span>
                        <span>${progressValue}%</span>
                    </div>
                    <div class="queue-progress-bar">
                        <div class="queue-progress-fill" style="width:${progressValue}%"></div>
                    </div>
                </div>
            `;

            return `
                <article class="queue-item ${isActive ? 'active' : ''}">
                    <div class="queue-item-head">
                        <div class="queue-item-title" title="${escapeHtml(job.file?.name || 'Imagen')}">${escapeHtml(job.file?.name || 'Imagen')}</div>
                        <span class="status-badge status-${job.status}">${statusLabel(job.status)}</span>
                    </div>
                    <div class="queue-item-meta">${dimensions} · ${escapeHtml(optionMeta)}</div>
                    ${progressLine}
                    ${errorLine}
                    <div class="queue-actions">${actionButtons.join('')}</div>
                </article>
            `;
        })
        .join('');
}

function renderHistoryPanel() {
    const historyPanel = document.getElementById('historyPanel');
    const historySummary = document.getElementById('historySummary');
    const historyList = document.getElementById('historyList');
    const toggleHistoryViewBtn = document.getElementById('toggleHistoryViewBtn');

    if (historyItems.length === 0) {
        setPanelVisibility('historyPanel', false);
        historyList.innerHTML = '';
        if (toggleHistoryViewBtn) toggleHistoryViewBtn.style.display = 'none';
        return;
    }

    setPanelVisibility('historyPanel', true);

    const hiddenCount = Math.max(0, historyItems.length - MAX_VISIBLE_HISTORY_ITEMS);
    historySummary.textContent = `${historyItems.length} resultado(s) en esta sesión.`;

    if (toggleHistoryViewBtn) {
        toggleHistoryViewBtn.style.display = hiddenCount > 0 ? 'inline-flex' : 'none';
        toggleHistoryViewBtn.textContent = showAllHistoryItems ? 'Ver menos' : `Ver más (${hiddenCount})`;
    }

    const visibleHistory = showAllHistoryItems ? historyItems : historyItems.slice(0, MAX_VISIBLE_HISTORY_ITEMS);

    historyList.innerHTML = visibleHistory
        .map((item) => `
            <article class="history-item">
                <div class="history-item-head">
                    <div class="history-item-title" title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</div>
                    <span class="status-badge status-done">Listo</span>
                </div>
                <div class="history-item-meta">${escapeHtml(item.model || '-')} · ${escapeHtml(String(item.scale || '-'))}x · ${escapeHtml(formatTimestamp(item.processedAt))}${item.variantCount > 1 ? ` · ${item.variantCount} versiones` : ''}</div>
                <div class="history-actions">
                    <button class="mini-btn" data-action="history-view" data-job-id="${item.jobId}">Ver</button>
                    <button class="mini-btn" data-action="history-download" data-job-id="${item.jobId}">Descargar</button>
                    <button class="mini-btn danger" data-action="history-remove" data-job-id="${item.jobId}">Quitar</button>
                </div>
            </article>
        `)
        .join('');
}

function triggerDownload(filename) {
    if (!filename) {
        UIController.showError('No se encontró el archivo de salida para descargar.');
        return;
    }

    const link = document.createElement('a');
    link.href = APIClient.getDownloadUrl(filename);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function handleQueueActionClick(event) {
    const btn = event.target.closest('button[data-action]');
    if (!btn) return;

    const action = btn.dataset.action;
    const jobId = btn.dataset.jobId;
    const job = jobs.get(jobId);
    if (!job) return;

    switch (action) {
        case 'select':
            selectJob(jobId);
            if (job.status === 'done') {
                showResultForJob(job);
                removeCompletedFromQueue(jobId);
            }
            break;
        case 'enqueue':
            selectJob(jobId);
            enqueueSelectedJob();
            break;
        case 'view':
            selectJob(jobId);
            showResultForJob(job);
            break;
        case 'retry':
            retryJob(jobId);
            break;
        case 'cancel':
            openCancelConfirmModal(jobId);
            break;
        case 'remove':
            removeJob(jobId);
            break;
        case 'download':
            triggerDownload(job.result?.output_filename);
            if (job.status === 'done') {
                removeCompletedFromQueue(jobId);
            }
            break;
        default:
            break;
    }
}

function handleHistoryActionClick(event) {
    const btn = event.target.closest('button[data-action]');
    if (!btn) return;

    const action = btn.dataset.action;
    const jobId = btn.dataset.jobId;
    const historyItem = historyItems.find((item) => item.jobId === jobId);
    const job = jobs.get(jobId);

    if (!historyItem) return;

    switch (action) {
        case 'history-view':
            if (!job || !job.result) {
                UIController.showError('No se encontró el contexto original para vista comparativa.');
                return;
            }
            selectJob(jobId);
            showResultForJob(job);
            removeCompletedFromQueue(jobId);
            break;
        case 'history-download':
            triggerDownload(historyItem.outputFilename);
            removeCompletedFromQueue(jobId);
            break;
        case 'history-remove': {
            const index = historyItems.findIndex((item) => item.jobId === jobId);
            if (index >= 0) {
                historyItems.splice(index, 1);
                renderHistoryPanel();
            }
            break;
        }
        default:
            break;
    }
}

function handleReturnToEditor() {
    UIController.hideElement('resultPanel');
    document.body.classList.remove('modal-open');
    setPanelVisibility('controlPanel', false);
}

async function ensureNotificationPermission() {
    if (!('Notification' in window)) {
        return;
    }

    if (Notification.permission === 'granted' || Notification.permission === 'denied') {
        return;
    }

    if (notificationPermissionRequested) {
        return;
    }

    notificationPermissionRequested = true;
    try {
        await Notification.requestPermission();
    } catch (_) {
        // Ignorar si el navegador bloquea el prompt.
    }
}

function playCompletionTone() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;

    try {
        const ctx = new AudioCtx();
        const now = ctx.currentTime;

        const oscA = ctx.createOscillator();
        const gainA = ctx.createGain();
        oscA.type = 'sine';
        oscA.frequency.value = 660;
        gainA.gain.setValueAtTime(0.0001, now);
        gainA.gain.exponentialRampToValueAtTime(0.05, now + 0.02);
        gainA.gain.exponentialRampToValueAtTime(0.0001, now + 0.2);
        oscA.connect(gainA).connect(ctx.destination);
        oscA.start(now);
        oscA.stop(now + 0.22);

        const oscB = ctx.createOscillator();
        const gainB = ctx.createGain();
        oscB.type = 'sine';
        oscB.frequency.value = 880;
        gainB.gain.setValueAtTime(0.0001, now + 0.1);
        gainB.gain.exponentialRampToValueAtTime(0.04, now + 0.12);
        gainB.gain.exponentialRampToValueAtTime(0.0001, now + 0.3);
        oscB.connect(gainB).connect(ctx.destination);
        oscB.start(now + 0.1);
        oscB.stop(now + 0.32);

        setTimeout(() => {
            ctx.close().catch(() => {});
        }, 450);
    } catch (_) {
        // Ignorar fallos de audio en navegadores restringidos.
    }
}

function notifyJobCompleted(job) {
    playCompletionTone();

    if (!('Notification' in window) || Notification.permission !== 'granted') {
        return;
    }

    const title = 'Imagen procesada';
    const body = `${job.file?.name || 'Imagen'} lista para revisar`;

    try {
        const notification = new Notification(title, {
            body,
            tag: job.id,
            renotify: true
        });

        notification.onclick = () => {
            window.focus();
            selectJob(job.id);
            showResultForJob(job);
            removeCompletedFromQueue(job.id);
            notification.close();
        };
    } catch (_) {
        // Ignorar si el sistema bloquea notificaciones.
    }
}

window.addEventListener('dragover', (e) => {
    e.preventDefault();
}, false);

window.addEventListener('drop', (e) => {
    e.preventDefault();
}, false);
