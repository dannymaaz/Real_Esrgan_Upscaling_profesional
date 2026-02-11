"""
Aplicación principal FastAPI
Real-ESRGAN Upscaling Profesional
Autor: Danny Maaz (github.com/dannymaaz)
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_DESCRIPTION,
    PROJECT_AUTHOR,
    PROJECT_GITHUB,
    FRONTEND_DIR
)
from app.routes import upload

# Crear aplicación FastAPI
app = FastAPI(
    title=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=PROJECT_DESCRIPTION,
    contact={
        "name": PROJECT_AUTHOR,
        "url": PROJECT_GITHUB
    }
)

# Configurar CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas de API
app.include_router(upload.router)

# Servir archivos estáticos del frontend
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/")
async def root():
    """Sirve la página principal de la aplicación"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar que la API está funcionando"""
    return {
        "status": "healthy",
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "author": PROJECT_AUTHOR
    }


if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT, RELOAD
    
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD
    )
