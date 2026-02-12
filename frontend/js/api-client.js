/**
 * Cliente API para comunicación con el backend
 * Autor: Danny Maaz (github.com/dannymaaz)
 */

const API_BASE_URL = '';

class APIClient {
    // Getter para la URL base (útil para referencias externas)
    static get BASE_URL() {
        return API_BASE_URL;
    }

    /**
     * Analiza una imagen y obtiene recomendaciones
     * @param {File} file - Archivo de imagen
     * @returns {Promise<Object>} Análisis de la imagen
     */
    static async analyzeImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error al analizar la imagen');
        }

        return await response.json();
    }

    /**
     * Procesa upscaling de una imagen
     * @param {File} file - Archivo de imagen
     * @param {string} scale - Escala ('2x' o '4x')
     * @param {string} model - Modelo específico (opcional)
     * @param {boolean} faceEnhance - Activar mejora de rostros (opcional)
     * @returns {Promise<Object>} Resultado del procesamiento
     */
    static async upscaleImage(file, scale, model = null, faceEnhance = false) {
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

        const response = await fetch(`${API_BASE_URL}/api/upscale`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error al procesar la imagen');
        }

        return await response.json();
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
        const response = await fetch(`${API_BASE_URL}/api/models`);

        if (!response.ok) {
            throw new Error('Error al obtener modelos');
        }

        return await response.json();
    }
}
