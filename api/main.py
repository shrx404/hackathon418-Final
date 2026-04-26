"""
=============================================================================
 PYTHON VERSION REQUIREMENT
=============================================================================
 This project requires Python 3.11 or 3.12.

 DO NOT use Python 3.13 or 3.14.
 Reason: PyTorch (required by terratorch>=1.2.4) and geospatial libraries
 (GDAL, rasterio) rely on C extensions that do NOT have pre-built wheels
 for Python 3.13+. On those versions you will encounter:
   - "No matching distribution found for torch"
   - Compilation failures for numpy's C extensions

 Recommended: Create a venv with Python 3.11:
   py -3.11 -m venv .venv
   .venv\\Scripts\\activate   (Windows)
   source .venv/bin/activate  (Linux/macOS)
   pip install -r requirements.txt
=============================================================================
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
#  Path bootstrap — allow `from src.model import ...` regardless of CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.model import load_checkpoint, is_loaded, device_name
from routers import predict

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

API_HOST   = os.getenv("API_HOST", "0.0.0.0")
API_PORT   = int(os.getenv("API_PORT", "8000"))

# Default: look for the checkpoint relative to the repo root
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(_REPO_ROOT, "backend", "mIoU=0.78.ckpt"),
)


# ---------------------------------------------------------------------------
#  Lifespan — load the real TerraMind checkpoint on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the TerraMind checkpoint on startup; clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("FloodSense API starting …")
    logger.info("Checkpoint: %s", MODEL_PATH)

    if not os.path.isfile(MODEL_PATH):
        logger.error(
            "Checkpoint not found at '%s'. "
            "Set MODEL_PATH env var to the correct path.",
            MODEL_PATH,
        )
    else:
        try:
            load_checkpoint(MODEL_PATH)
            logger.info(
                "TerraMind checkpoint loaded on %s — ready for inference.",
                device_name(),
            )
        except Exception as exc:
            logger.exception(
                "Failed to load checkpoint — API will return 503 on /predict: %s", exc
            )

    logger.info("=" * 60)
    yield
    logger.info("FloodSense API shutting down.")


# ---------------------------------------------------------------------------
#  FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FloodSense — TerraMind SAR Flood Detection API",
    description=(
        "Runs real on-orbit-style inference using the fine-tuned TerraMind v1 Tiny "
        "model on Sentinel-1 SAR data. Accepts a demo tile ID or an uploaded .npy "
        "SAR array and returns a base64-encoded flood mask, flood area statistics, "
        "and edge inference telemetry.\n\n"
        "**Model**: TerraMind v1 Tiny + UperNet decoder  \n"
        "**Checkpoint mIoU**: 0.78 (Sen1Floods11 hand-labeled test split)"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
#  CORS — allow any frontend (restrict in production)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)},
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"{request.method} {request.url.path} does not exist.",
        },
    )


# ---------------------------------------------------------------------------
#  Routers
# ---------------------------------------------------------------------------
app.include_router(predict.router)


# ---------------------------------------------------------------------------
#  Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"], summary="Health check")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "service": "FloodSense TerraMind API",
        "version": "3.0.0",
        "model_loaded": is_loaded(),
        "inference_device": device_name(),
    }


# ---------------------------------------------------------------------------
#  Available demo tiles
# ---------------------------------------------------------------------------
@app.get("/tiles", tags=["Tiles"], summary="List available demo tiles")
async def list_tiles() -> dict:
    """Return available demo tile IDs and metadata for the UI selector."""
    return {"tiles": list(predict.DEMO_TILES.values())}


# ---------------------------------------------------------------------------
#  Entry point (dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
