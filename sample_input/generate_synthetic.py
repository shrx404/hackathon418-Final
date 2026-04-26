#!/usr/bin/env python3
"""
generate_synthetic.py — Create a synthetic SAR demo tile
=========================================================
Generates a realistic-looking 2-band Sentinel-1 SAR array and saves it
as `sample_sar.npy` for use with infer.py and the FastAPI demo endpoint.

The synthetic data mimics real SAR backscatter statistics:
  - Water / flood areas: very low backscatter (≈ -20 to -25 dB in log scale)
  - Land / vegetation:   moderate backscatter (≈ -10 to -5 dB)
  - Urban / bright:      high backscatter    (≈ 0 to +5 dB)

Values are stored in linear scale (not dB) as float32, matching the format
used by rasterio.open().read() on real Sen1Floods11 tiles.

Usage:
    python sample_input/generate_synthetic.py
    # → writes sample_input/sample_sar.npy  (~2 MB, shape: (2, 512, 512))
"""

import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
SIZE          = 512     # tile size in pixels (matches Sen1Floods11 format)
SEED          = 42      # reproducible output
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "sample_sar.npy")

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
#  Helper: convert dB to linear scale (SAR backscatter convention)
# ---------------------------------------------------------------------------
def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


# ---------------------------------------------------------------------------
#  Build semantic map (determines per-pixel land cover type)
# ---------------------------------------------------------------------------
print("Generating synthetic SAR tile …")

# Land cover mask: 0=land, 1=water/flood, 2=urban
land_cover = np.zeros((SIZE, SIZE), dtype=np.uint8)

# ---- River channel (sinusoidal, left-center) --------------------------------
for row in range(SIZE):
    cx = int(SIZE * 0.38 + math.sin(row / 55.0) * SIZE * 0.10)
    width = int(SIZE * 0.05 + math.sin(row / 90.0 + 1.2) * SIZE * 0.02)
    x0 = max(0, cx - width)
    x1 = min(SIZE, cx + width)
    land_cover[row, x0:x1] = 1

# ---- Flooded fields (elliptical patches) ------------------------------------
flood_patches = [
    (180, 130, 70, 40),   # (cx, cy, rx, ry)
    (320, 200, 55, 30),
    (140, 320, 80, 45),
    (390, 380, 60, 35),
    (260, 440, 50, 25),
    (80,  430, 40, 20),
]
for cx, cy, rx, ry in flood_patches:
    y_idx, x_idx = np.ogrid[:SIZE, :SIZE]
    ellipse = ((x_idx - cx) / rx) ** 2 + ((y_idx - cy) / ry) ** 2 <= 1
    land_cover[ellipse] = 1

# ---- Urban cluster (top-right corner) ---------------------------------------
land_cover[20:120, 360:480] = 2

# ---- Smooth edges with a small Gaussian-style blur --------------------------
# (avoid rasterio / scipy dependency — manual 3x3 box filter is enough)
from PIL import Image as _PILImage
_lc_img = _PILImage.fromarray(land_cover).filter(
    __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=2)
)
land_cover_smooth = np.array(_lc_img)
# Re-threshold to clean classes
land_cover_final = np.where(land_cover_smooth > 128, land_cover, 0)
land_cover_final = land_cover.copy()   # keep hard edges for crisp label reference


# ---------------------------------------------------------------------------
#  Assign backscatter values per class (linear scale)
# ---------------------------------------------------------------------------
# VV and VH have correlated but independent noise

def make_band(land_cover_map: np.ndarray, vv: bool) -> np.ndarray:
    """Generate one SAR band in linear scale."""
    band = np.zeros((SIZE, SIZE), dtype=np.float32)

    # Land (class 0) — moderate backscatter with texture
    land_mean_db  = -9.0  if vv else -15.0
    land_std_db   =  2.0  if vv else   1.5
    land_linear   = db_to_linear(land_mean_db)
    noise         = rng.normal(loc=0.0, scale=db_to_linear(land_std_db), size=(SIZE, SIZE)).astype(np.float32)
    band         += land_linear + noise * 0.15

    # Water / flood (class 1) — very low backscatter
    water_mean_db = -22.0 if vv else -25.0
    water_noise   = rng.normal(0, 0.0005, (SIZE, SIZE)).astype(np.float32)
    water_mask    = land_cover_final == 1
    band[water_mask] = db_to_linear(water_mean_db) + water_noise[water_mask]

    # Urban (class 2) — high backscatter (double-bounce)
    urban_mean_db = 2.0  if vv else -8.0
    urban_noise   = rng.normal(0, 0.02, (SIZE, SIZE)).astype(np.float32)
    urban_mask    = land_cover_final == 2
    band[urban_mask] = db_to_linear(urban_mean_db) + urban_noise[urban_mask]

    # Clip to physical range [0, 1] (linear SAR backscatter ≤ 1 for natural surfaces)
    band = np.clip(band, 0.0, 1.0)
    return band


vv_band = make_band(land_cover_final, vv=True)
vh_band = make_band(land_cover_final, vv=False)

sar_array = np.stack([vv_band, vh_band], axis=0)   # shape: (2, 512, 512)

# ---------------------------------------------------------------------------
#  Save
# ---------------------------------------------------------------------------
np.save(OUTPUT_PATH, sar_array)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
print(f"✅  Saved: {OUTPUT_PATH}")
print(f"   Shape : {sar_array.shape}  dtype: {sar_array.dtype}")
print(f"   Size  : {size_mb:.2f} MB")
print(f"   VV range: [{vv_band.min():.4f}, {vv_band.max():.4f}]")
print(f"   VH range: [{vh_band.min():.4f}, {vh_band.max():.4f}]")
print()
print("Run inference with:")
print('  python infer.py "backend/mIoU=0.78.ckpt" "sample_input/sample_sar.npy"')
