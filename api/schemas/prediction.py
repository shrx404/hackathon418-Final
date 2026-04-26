"""
=============================================================================
 Pydantic Schemas — FloodSense TerraMind Prediction API (v3)
=============================================================================
 Request/response models for the real TerraMind inference endpoint.
 Updated to include real flood pixel counts and actual device telemetry.
=============================================================================
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Telemetry — real edge inference performance statistics
# ---------------------------------------------------------------------------

class Telemetry(BaseModel):
    """
    Inference telemetry from the TerraMind v1 Tiny + UperNet model.
    """
    model_name: str = Field(
        default="TerraMind v1 Tiny + UperNet",
        description="Model architecture name",
    )
    checkpoint_miou: float = Field(
        default=0.78,
        description="Test-set mIoU achieved by this checkpoint on Sen1Floods11",
    )
    model_size_mb: float = Field(
        default=147.0,
        description="Approximate checkpoint size on disk in megabytes",
    )
    inference_latency_s: float = Field(
        ...,
        description="End-to-end inference time in seconds",
    )
    inference_device: str = Field(
        ...,
        description="Device used for inference (e.g. 'GPU (Tesla T4)' or 'CPU')",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name":           "TerraMind v1 Tiny + UperNet",
                "checkpoint_miou":      0.78,
                "model_size_mb":        147.0,
                "inference_latency_s":  2.3,
                "inference_device":     "GPU (Tesla T4)",
            }
        }
    }


# ---------------------------------------------------------------------------
#  Response — full inference result
# ---------------------------------------------------------------------------

class OrbitalPredictionResponse(BaseModel):
    """
    Complete response from the TerraMind SAR flood detection inference.

    Contains:
      - The generated flood mask as a base64 PNG
      - Real flood pixel counts and area estimation
      - Bandwidth savings (mask vs. raw Sentinel-1 product)
      - Actual inference telemetry (latency, device)
    """

    # -- Tile / source identification ----------------------------------------
    tile_id:     str = Field(..., description="Tile identifier")
    tile_name:   str = Field(..., description="Human-readable tile name")
    tile_region: str = Field(..., description="Geographic region")
    tile_date:   str = Field(..., description="SAR acquisition date")

    # -- Flood mask image ----------------------------------------------------
    flood_mask_b64: str = Field(
        ...,
        description=(
            "Base64-encoded PNG of the flood segmentation mask.  "
            "White pixels = flood (class 1), black = land (class 0)."
        ),
    )
    image_format: str = Field(
        default="image/png",
        description="MIME type of the encoded mask",
    )

    # -- Bandwidth savings ---------------------------------------------------
    raw_data_size_mb: float = Field(
        ...,
        description="Typical size of raw Sentinel-1 GRD tile in MB",
    )
    mask_size_kb: float = Field(
        ...,
        description="Size of the PNG mask downlink payload in KB",
    )
    bandwidth_saved_pct: float = Field(
        ...,
        description="Percentage bandwidth saved vs. raw downlink",
    )

    # -- Real flood statistics -----------------------------------------------
    flooded_area_sq_km: float = Field(
        ...,
        description=(
            "Estimated flooded area in km² "
            "(assumes 10 m × 10 m Sentinel-1 GRD pixel resolution)"
        ),
    )
    flood_pixel_count: int = Field(
        ...,
        description="Number of pixels classified as flood (class 1)",
    )
    total_pixel_count: int = Field(
        ...,
        description="Total number of pixels in the processed tile",
    )

    # -- Inference telemetry -------------------------------------------------
    telemetry: Telemetry

    model_config = {
        "json_schema_extra": {
            "example": {
                "tile_id":              "demo_sample",
                "tile_name":            "Synthetic SAR Demo Tile",
                "tile_region":          "Synthetic (river + flood patches)",
                "tile_date":            "2024-01-01",
                "flood_mask_b64":       "<base64 PNG string>",
                "image_format":         "image/png",
                "raw_data_size_mb":     512.0,
                "mask_size_kb":         5.8,
                "bandwidth_saved_pct":  98.9,
                "flooded_area_sq_km":   23.45,
                "flood_pixel_count":    234500,
                "total_pixel_count":    262144,
                "telemetry": {
                    "model_name":           "TerraMind v1 Tiny + UperNet",
                    "checkpoint_miou":      0.78,
                    "model_size_mb":        147.0,
                    "inference_latency_s":  2.3,
                    "inference_device":     "GPU (Tesla T4)",
                },
            }
        }
    }
