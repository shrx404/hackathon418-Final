"""
=============================================================================
 Predict Router — POST /predict  (v3 — Real TerraMind Inference)
=============================================================================
 Accepts either:
   (a) a demo tile_id  → loads the pre-saved .npy from sample_input/
   (b) an uploaded .npy file  → 2-band SAR array, shape (2, H, W) float32

 Runs real inference via src.model.run_inference(), converts the binary
 mask to a base64 PNG, and returns telemetry + flood area statistics.
=============================================================================
"""

import base64
import io
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

# Path bootstrap — ensure src/ is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.model import run_inference, is_loaded, device_name
from src.preprocess import load_npy, prepare_sar
from schemas.prediction import OrbitalPredictionResponse, Telemetry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Demo tiles — reference the sample_input/ npy files
# ---------------------------------------------------------------------------
_SAMPLE_INPUT_DIR = os.path.join(_REPO_ROOT, "sample_input")

DEMO_TILES: dict[str, dict] = {
    "demo_sample": {
        "id":               "demo_sample",
        "name":             "Synthetic SAR Demo Tile",
        "region":           "Synthetic (river + flood patches)",
        "date":             "2024-01-01",
        "npy_path":         os.path.join(_SAMPLE_INPUT_DIR, "sample_sar.npy"),
        "raw_data_size_mb": 512.0,   # typical Sentinel-1 GRD product size
    },
    "sri_lanka": {
        "id":               "sri_lanka",
        "name":             "Sri Lanka Flood (Sen1Floods11 test split)",
        "region":           "Sri Lanka",
        "date":             "2017-05-01",
        "npy_path":         os.path.join(_SAMPLE_INPUT_DIR, "Sri-Lanka_534068.npy"),
        "tif_path":         os.path.join(_SAMPLE_INPUT_DIR, "Sri-Lanka_534068.tif"),
        "raw_data_size_mb": 840.0,
    },
}

router = APIRouter(tags=["Prediction"])


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _mask_to_png_b64(mask: np.ndarray) -> tuple[str, float]:
    """
    Convert a binary (0/1) uint8 mask to a base64-encoded grayscale PNG.

    Returns:
        (b64_str, mask_size_kb)
    """
    from PIL import Image
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()
    b64_str = base64.b64encode(raw).decode("utf-8")
    return b64_str, len(raw) / 1024.0


def _load_tile_array(tile: dict) -> np.ndarray:
    """
    Load the SAR array for a demo tile.

    Tries .npy first, then falls back to .tif (requires rasterio).
    """
    npy_path = tile.get("npy_path", "")
    tif_path = tile.get("tif_path", "")

    if os.path.isfile(npy_path):
        return load_npy(npy_path)

    if os.path.isfile(tif_path):
        from src.preprocess import load_tif
        return load_tif(tif_path)

    raise HTTPException(
        status_code=404,
        detail=(
            f"Demo tile data not found for '{tile['id']}'. "
            f"Run  python sample_input/generate_synthetic.py  or "
            f"python sample_input/download_sample.py  to create it."
        ),
    )


# ---------------------------------------------------------------------------
#  POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=OrbitalPredictionResponse,
    summary="Run TerraMind SAR flood detection inference",
    description=(
        "Accepts a demo `tile_id` **or** an uploaded `.npy` SAR file "
        "(shape `(2, H, W)`, float32, VV+VH bands).  \n\n"
        "Runs real inference with the fine-tuned **TerraMind v1 Tiny + UperNet** "
        "checkpoint (`mIoU = 0.78`) and returns a base64-encoded flood mask PNG "
        "along with telemetry and flood area statistics."
    ),
)
async def predict_flood(
    tile_id: str = Form(
        default="demo_sample",
        description="ID of a demo tile (e.g. 'demo_sample', 'sri_lanka'). "
                    "Ignored if a file is uploaded.",
    ),
    image: Optional[UploadFile] = File(
        default=None,
        description=(
            "Optional: upload a custom .npy SAR file.  "
            "Must be a numpy array saved with np.save(), shape (2, H, W), float32."
        ),
    ),
) -> OrbitalPredictionResponse:
    # -- Guard: model must be loaded ----------------------------------------
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Check that the checkpoint exists at the path configured by MODEL_PATH "
                "and that terratorch is installed."
            ),
        )

    try:
        # -- Step 1: Resolve input source -----------------------------------
        if image is not None:
            # Uploaded .npy file
            if not (image.filename or "").endswith(".npy"):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Only .npy uploads are supported. "
                        "Save your SAR array with np.save('file.npy', arr) "
                        "where arr has shape (2, H, W) and dtype float32."
                    ),
                )
            raw_bytes = await image.read()
            arr_raw   = np.load(io.BytesIO(raw_bytes))
            sar_array = prepare_sar(arr_raw)
            tile = {
                "id":               "custom_upload",
                "name":             f"Custom upload: {image.filename}",
                "region":           "User-defined",
                "date":             "N/A",
                "raw_data_size_mb": len(raw_bytes) / (1024 ** 2),
            }
            logger.info("Processing uploaded .npy: %s  shape=%s", image.filename, sar_array.shape)

        elif tile_id in DEMO_TILES:
            tile      = DEMO_TILES[tile_id]
            sar_array = _load_tile_array(tile)
            logger.info("Loaded demo tile '%s'  shape=%s", tile_id, sar_array.shape)

        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown tile_id '{tile_id}'. "
                    f"Available: {list(DEMO_TILES.keys())}. "
                    "Or upload a .npy SAR array directly."
                ),
            )

        # -- Step 2: Run real inference -------------------------------------
        t0 = time.perf_counter()
        mask = run_inference(sar_array)            # (H, W) uint8, 0=land, 1=flood
        inference_latency = round(time.perf_counter() - t0, 3)

        # -- Step 3: Encode mask as PNG ------------------------------------
        flood_mask_b64, mask_size_kb = _mask_to_png_b64(mask)

        # -- Step 4: Compute flood statistics ------------------------------
        flood_pixels      = int(np.sum(mask == 1))
        total_pixels      = mask.size
        flood_area_km2    = round(flood_pixels * 100 / 1_000_000, 4)  # 10 m/px → km²

        # -- Step 5: Bandwidth savings (mask vs. raw tile) -----------------
        raw_mb             = tile["raw_data_size_mb"]
        mask_mb            = mask_size_kb / 1024.0
        bandwidth_saved    = round(max(min((1.0 - mask_mb / raw_mb) * 100.0, 99.99), 0.0), 2)

        logger.info(
            "Inference done | tile=%s | latency=%.3fs | device=%s | "
            "flood_px=%d | area=%.4f km² | mask=%.2f KB | saved=%.2f%%",
            tile["id"], inference_latency, device_name(),
            flood_pixels, flood_area_km2, mask_size_kb, bandwidth_saved,
        )

        # -- Step 6: Build response ----------------------------------------
        return OrbitalPredictionResponse(
            tile_id=tile["id"],
            tile_name=tile["name"],
            tile_region=tile["region"],
            tile_date=tile["date"],
            flood_mask_b64=flood_mask_b64,
            image_format="image/png",
            raw_data_size_mb=raw_mb,
            mask_size_kb=round(mask_size_kb, 2),
            bandwidth_saved_pct=bandwidth_saved,
            flooded_area_sq_km=flood_area_km2,
            flood_pixel_count=flood_pixels,
            total_pixel_count=total_pixels,
            telemetry=Telemetry(
                inference_latency_s=inference_latency,
                inference_device=device_name(),
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unexpected error during inference: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "Inference failed", "message": str(exc)},
        )
