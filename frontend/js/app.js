/**
 * Lógica principal de la aplicación
 * Autor: Danny Maaz (github.com/dannymaaz)
 */

const MAX_QUEUE_ITEMS = 30;
const MAX_HISTORY_ITEMS = 40;

// Estado de sesión (en memoria, se limpia al reiniciar la app)
const jobs = new Map();
const jobOrder = [];
const processQueue = [];
const historyItems = [];

let selectedJobId = null;
let isQueueWorkerRunning = false;

let currentFile = null;
let currentAnalysis = null;
let selectedScale = null;

window.currentFile = null;
window.AppController = {
    onNewImageRequested: handleReturnToEditor
};

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    renderQueuePanel();
    renderHistoryPanel();
});

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
    if (message.includes('Failed to fetch')) {
        return 'No se pudo conectar con el servidor. Verifica que el backend siga activo en http://127.0.0.1:8000';
    }
    return message || fallbackMessage;
}

function validateFile(file) {
    if (!file) {
        return 'No se recibió ningún archivo.';
    }

    const supportedMimeTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    const supportedExtensions = ['.png', '.jpg', '.jpeg', '.webp'];
    const extension = file.name ? `.${file.name.split('.').pop().toLowerCase()}` : '';
    const mimeType = (file.type || '').toLowerCase();

    const isSupportedMime = supportedMimeTypes.includes(mimeType);
    const isSupportedExtension = supportedExtensions.includes(extension);
    if (!isSupportedMime && !isSupportedExtension) {
        return 'Este archivo no es compatible por ahora. Formatos soportados: PNG, JPG, JPEG, WEBP.';
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
            analysis,
            result: null,
            error: null,
            options: {
                scale: `${analysis.recommended_scale}x`,
                faceEnhance: false,
                forcedImageType: null
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

function applyJobControls(job) {
    currentFile = job.file;
    currentAnalysis = job.analysis;
    selectedScale = job.options.scale;
    window.currentFile = job.file;

    UIController.showControlPanel(job.analysis);
    setActiveScaleButton(job.options.scale);

    const faceEnhanceBtn = document.getElementById('faceEnhanceBtn');
    if (faceEnhanceBtn) {
        faceEnhanceBtn.checked = Boolean(job.options.faceEnhance);
    }

    const overrideContainer = document.getElementById('imageTypeOverrideContainer');
    const overrideBtn = document.getElementById('imageTypeOverrideBtn');
    if (overrideContainer && overrideBtn && overrideContainer.style.display !== 'none') {
        overrideBtn.checked = Boolean(job.options.forcedImageType);
    }
}

function selectJob(jobId) {
    const job = jobs.get(jobId);
    if (!job) return;

    selectedJobId = job.id;
    applyJobControls(job);
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

    job.options.scale = activeScaleBtn ? activeScaleBtn.dataset.scale : job.options.scale;
    job.options.faceEnhance = Boolean(faceEnhanceBtn && faceEnhanceBtn.checked);
    job.options.forcedImageType = getForcedImageTypeFromUI(job.analysis);
    job.updatedAt = Date.now();

    jobs.set(job.id, job);
    selectedScale = job.options.scale;
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
    job.updatedAt = Date.now();
    jobs.set(job.id, job);

    if (!processQueue.includes(job.id)) {
        processQueue.push(job.id);
    }

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
        job.updatedAt = Date.now();
        jobs.set(job.id, job);
        renderQueuePanel();

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
                job.options.forcedImageType
            );

            job.status = 'done';
            job.result = result;
            job.updatedAt = Date.now();
            jobs.set(job.id, job);

            pushHistory(job);

            if (selectedJobId === job.id) {
                showResultForJob(job);
            }
        } catch (error) {
            job.status = 'failed';
            job.error = formatError(error, 'Error desconocido al procesar imagen');
            job.updatedAt = Date.now();
            jobs.set(job.id, job);

            if (selectedJobId === job.id) {
                UIController.showError(job.error);
            }
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
            job.updatedAt = Date.now();
            jobs.set(job.id, job);
        }
    }
    processQueue.length = 0;
    renderQueuePanel();
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
        processedAt: Date.now()
    });

    if (historyItems.length > MAX_HISTORY_ITEMS) {
        historyItems.length = MAX_HISTORY_ITEMS;
    }

    renderHistoryPanel();
}

function clearHistory() {
    historyItems.length = 0;
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

    if (jobOrder.length === 0) {
        queuePanel.style.display = 'none';
        queueList.innerHTML = '';
        return;
    }

    queuePanel.style.display = 'block';

    const queuedCount = [...jobs.values()].filter((job) => job.status === 'queued').length;
    const processingCount = [...jobs.values()].filter((job) => job.status === 'processing').length;
    const doneCount = [...jobs.values()].filter((job) => job.status === 'done').length;

    queueSummary.textContent = `Total: ${jobOrder.length} · En cola: ${queuedCount} · Procesando: ${processingCount} · Completadas: ${doneCount}`;

    queueList.innerHTML = jobOrder
        .slice()
        .reverse()
        .map((jobId) => {
            const job = jobs.get(jobId);
            if (!job) return '';

            const isActive = selectedJobId === job.id;
            const dimensions = `${job.analysis.width}x${job.analysis.height}`;
            const optionMeta = `${job.options.scale} · Rostro ${job.options.faceEnhance ? 'ON' : 'OFF'}`;

            const actionButtons = [];
            actionButtons.push(`<button class="mini-btn" data-action="select" data-job-id="${job.id}">Abrir</button>`);

            if (job.status === 'done') {
                actionButtons.push(`<button class="mini-btn" data-action="view" data-job-id="${job.id}">Ver</button>`);
                actionButtons.push(`<button class="mini-btn" data-action="download" data-job-id="${job.id}">Descargar</button>`);
                actionButtons.push(`<button class="mini-btn" data-action="retry" data-job-id="${job.id}">Reprocesar</button>`);
            } else if (job.status === 'failed') {
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

            return `
                <article class="queue-item ${isActive ? 'active' : ''}">
                    <div class="queue-item-head">
                        <div class="queue-item-title" title="${escapeHtml(job.file?.name || 'Imagen')}">${escapeHtml(job.file?.name || 'Imagen')}</div>
                        <span class="status-badge status-${job.status}">${statusLabel(job.status)}</span>
                    </div>
                    <div class="queue-item-meta">${dimensions} · ${escapeHtml(optionMeta)}</div>
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

    if (historyItems.length === 0) {
        historyPanel.style.display = 'none';
        historyList.innerHTML = '';
        return;
    }

    historyPanel.style.display = 'block';
    historySummary.textContent = `${historyItems.length} resultado(s) en esta sesión.`;

    historyList.innerHTML = historyItems
        .map((item) => `
            <article class="history-item">
                <div class="history-item-head">
                    <div class="history-item-title" title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</div>
                    <span class="status-badge status-done">Listo</span>
                </div>
                <div class="history-item-meta">${escapeHtml(item.model || '-')} · ${escapeHtml(String(item.scale || '-'))}x · ${escapeHtml(formatTimestamp(item.processedAt))}</div>
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
        case 'remove':
            removeJob(jobId);
            break;
        case 'download':
            triggerDownload(job.result?.output_filename);
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
            break;
        case 'history-download':
            triggerDownload(historyItem.outputFilename);
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
    const selected = getSelectedJob();
    if (selected) {
        applyJobControls(selected);
        return;
    }

    const fallbackId = jobOrder[jobOrder.length - 1] || null;
    if (fallbackId) {
        selectJob(fallbackId);
    }
}

window.addEventListener('dragover', (e) => {
    e.preventDefault();
}, false);

window.addEventListener('drop', (e) => {
    e.preventDefault();
}, false);
