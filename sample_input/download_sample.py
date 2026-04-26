#!/usr/bin/env python3
"""
download_sample.py — Download a real Sen1Floods11 test tile
=============================================================
Downloads a single Sentinel-1 SAR image and its ground truth label from
the public Sen1Floods11 Google Cloud Storage bucket for use with infer.py.

Requirements:
    pip install gsutil   # or have gcloud CLI installed
    OR: the script falls back to a direct HTTP download via the public GCS URL.

Tile selected: Sri-Lanka_534068 (heavily flooded, high water pixel count)
  - Path: gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/
  - File: Sri-Lanka_534068_S1Hand.tif  (≈ 2 MB per tile)

Usage:
    python sample_input/download_sample.py
"""

import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
TILE_NAME    = "Sri-Lanka_534068"
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))

# Public GCS URLs (no auth required for this public bucket)
BASE_URL     = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"
S1_URL       = f"{BASE_URL}/S1Hand/{TILE_NAME}_S1Hand.tif"
LABEL_URL    = f"{BASE_URL}/LabelHand/{TILE_NAME}_LabelHand.tif"

S1_OUT       = os.path.join(OUT_DIR, f"{TILE_NAME}.tif")
LABEL_OUT    = os.path.join(OUT_DIR, f"{TILE_NAME}_label.tif")


# ---------------------------------------------------------------------------
#  Download helper
# ---------------------------------------------------------------------------
def download(url: str, dest: str) -> None:
    if os.path.isfile(dest):
        print(f"  Already exists: {dest}  (skipping)")
        return

    print(f"  Downloading: {os.path.basename(dest)} …", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f" done  ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        print(f"  Try manually: gsutil cp '{url.replace('https://storage.googleapis.com/', 'gs://')}' '{dest}'")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
print("=" * 56)
print("  Sen1Floods11 Sample Tile Downloader")
print(f"  Tile: {TILE_NAME}")
print("=" * 56)

download(S1_URL, S1_OUT)
download(LABEL_URL, LABEL_OUT)

if os.path.isfile(S1_OUT):
    print()
    print("✅  Download complete!")
    print()
    print("Run inference on the real tile:")
    print(f'  python infer.py "backend/mIoU=0.78.ckpt" "sample_input/{TILE_NAME}.tif"')
    print()
    print("Note: .tif loading requires rasterio:")
    print("  pip install rasterio")
else:
    print()
    print("⚠️   Download failed. Fallback: use the synthetic sample instead.")
    print('  python sample_input/generate_synthetic.py')
    print('  python infer.py "backend/mIoU=0.78.ckpt" "sample_input/sample_sar.npy"')
