from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.routers import upload, remove, sam
from pathlib import Path

app = FastAPI(title="Object Remover API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(remove.router)
app.include_router(sam.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    """Serve the main index.html page"""
    return FileResponse("app/static/index.html")

# Mount static files (must be last)
app.mount("/uploads", StaticFiles(directory="app/static/uploads"), name="uploads")
