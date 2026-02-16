/**
 * Cliente API para comunicación con el backend
 * Autor: Danny Maaz (github.com/dannymaaz)
 */

const API_BASE_URL = window.location.protocol === 'file:'
    ? 'http://127.0.0.1:8000'
    : '';

class APIClient {
    // Getter para la URL base (útil para referencias externas)
    static get BASE_URL() {
        return API_BASE_URL;
    }

    static async requestJson(path, options = {}) {
        let response;
        try {
            response = await fetch(`${API_BASE_URL}${path}`, options);
        } catch (error) {
            throw new Error(
                'No se pudo conectar con el servidor. Verifica que la app backend este activa en http://127.0.0.1:8000'
            );
        }

        if (!response.ok) {
            let detail = null;
            try {
                const errorData = await response.json();
                detail = errorData?.detail;
            } catch (_) {
                detail = null;
            }

            throw new Error(detail || `Error HTTP ${response.status} al procesar la solicitud`);
        }

        return await response.json();
    }

    /**
     * Analiza una imagen y obtiene recomendaciones
     * @param {File} file - Archivo de imagen
     * @returns {Promise<Object>} Análisis de la imagen
     */
    static async analyzeImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        return await this.requestJson('/api/analyze', {
            method: 'POST',
            body: formData
        });
    }

    /**
     * Procesa upscaling de una imagen
     * @param {File} file - Archivo de imagen
     * @param {string} scale - Escala ('2x' o '4x')
     * @param {string} model - Modelo específico (opcional)
     * @param {boolean} faceEnhance - Activar mejora de rostros (opcional)
     * @param {string|null} forcedImageType - Override manual del tipo (opcional)
     * @param {Object} extraOptions - Opciones avanzadas de restauración
     * @returns {Promise<Object>} Resultado del procesamiento
     */
    static async upscaleImage(file, scale, model = null, faceEnhance = false, forcedImageType = null, extraOptions = {}) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('scale', scale);
        
        // Solo enviar modelo si se especifica explícitamente y no es null/undefined
        if (model) {
            formData.append('model', model);
        }

        // Enviar flag de mejora de rostros
        if (faceEnhance) {
            formData.append('face_enhance', 'true');
        }
        if (forcedImageType) {
            formData.append('forced_image_type', forcedImageType);
        }

        if (extraOptions?.removeColorFilter) {
            formData.append('remove_filter', 'true');
            formData.append('remove_color_filter', 'true');
        }
        if (extraOptions?.restoreOldPhoto) {
            formData.append('restore_old_photo', 'true');
        }
        if (extraOptions?.dualOutput) {
            formData.append('dual_output', 'true');
        }
        if (extraOptions?.restoreMonochrome) {
            formData.append('restore_bw', 'true');
        }

        return await this.requestJson('/api/upscale', {
            method: 'POST',
            body: formData
        });
    }

    /**
     * Obtiene la URL de descarga para un archivo
     * @param {string} filename - Nombre del archivo
     * @returns {string} URL de descarga
     */
    static getDownloadUrl(filename) {
        return `${API_BASE_URL}/api/download/${filename}`;
    }
    
    /**
     * Helper para obtener URL de descarga (shim para compatibilidad con código nuevo)
     */
    static async deltDownloadUrl(filename) {
        return this.getDownloadUrl(filename);
    }

    /**
     * Obtiene información de modelos disponibles
     * @returns {Promise<Object>} Información de modelos
     */
    static async getModels() {
        return await this.requestJson('/api/models');
    }
}
