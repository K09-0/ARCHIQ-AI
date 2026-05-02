"""Main FastAPI application for site plan processing."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import logging

from .api.main import router as api_router
from .core.processor import SitePlanProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Site Plan Processor...")
    # Initialize global processor
    app.state.processor = SitePlanProcessor()
    yield
    # Shutdown
    logger.info("Shutting down Site Plan Processor...")


# Create FastAPI app
app = FastAPI(
    title="Site Plan Processor API",
    description="OCR and geometry extraction for site plans",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Site Plan Processor API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /api/v1/upload-site-plan",
            "parameters": "GET /api/v1/site-parameters",
            "geometry": "POST /api/v1/extract-geometry",
            "calibrate": "POST /api/v1/calibrate-scale",
            "export_dxf": "GET /api/v1/export-dxf",
            "export_svg": "GET /api/v1/export-svg",
            "export_geojson": "GET /api/v1/export-geojson"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )