#!/usr/bin/env python3
"""
infer.py — Judge Entry Point
==============================
Runs flood segmentation inference using the fine-tuned TerraMind v1 Tiny
checkpoint and saves the result as a binary flood mask PNG.

Usage:
    python infer.py <checkpoint> <input> [output]

Arguments:
    checkpoint   Path to the .ckpt file
                   e.g.  backend/mIoU=0.78.ckpt
    input        Path to input SAR data:
                   .npy  — numpy array, shape (2, H, W), float32 (VV, VH)
                   .tif  — GeoTIFF with at least 2 bands (requires rasterio)
    output       (optional) Output PNG path.  Default: flood_mask.png

Examples:
    # Quick demo using the bundled synthetic SAR sample:
    python infer.py "backend/mIoU=0.78.ckpt" "sample_input/sample_sar.npy"

    # Real Sen1Floods11 GeoTIFF (after running sample_input/download_sample.py):
    python infer.py "backend/mIoU=0.78.ckpt" "sample_input/Sri-Lanka_534068.tif"

    # Custom output path:
    python infer.py "backend/mIoU=0.78.ckpt" "sample_input/sample_sar.npy" "out/mask.png"

Model:
    TerraMind v1 Tiny + UperNet decoder
    Fine-tuned on Sen1Floods11 (hand-labeled split)
    Test-set mIoU = 0.78
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
#  Bootstrap: add repo root to sys.path so src/ is importable when running
#  this script from any working directory.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model import load_checkpoint, run_inference, device_name
from src.preprocess import load_npy, load_tif

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("infer")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def save_mask_png(mask: np.ndarray, output_path: str) -> None:
    """Save a binary (0/1) mask as a grayscale PNG (black=land, white=flood)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to save the PNG mask.\n"
            "Install it with:  pip install Pillow"
        ) from exc

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    img.save(output_path)
    logger.info("Flood mask saved → %s", output_path)


def print_report(
    mask: np.ndarray,
    input_path: str,
    ckpt_path: str,
    load_time: float,
    infer_time: float,
) -> None:
    """Print a disaster assessment report to stdout."""
    h, w = mask.shape
    total_pixels = h * w
    flood_pixels = int(np.sum(mask == 1))
    land_pixels  = total_pixels - flood_pixels
    flood_pct    = flood_pixels / total_pixels * 100

    # Sentinel-1 GRD at 10 m resolution → 100 m² per pixel → km² conversion
    flood_area_km2 = flood_pixels * 100 / 1_000_000

    print()
    print("=" * 54)
    print("   FLOODSENSE DISASTER ASSESSMENT REPORT")
    print("=" * 54)
    print(f"  Checkpoint :  {os.path.basename(ckpt_path)}")
    print(f"  Model      :  TerraMind v1 Tiny + UperNet")
    print(f"  Test mIoU  :  0.78  (Sen1Floods11 hand-labeled)")
    print(f"  Device     :  {device_name()}")
    print(f"  Input      :  {os.path.basename(input_path)}")
    print(f"  Image size :  {w} × {h} px")
    print("-" * 54)
    print(f"  Load time  :  {load_time:.1f} s")
    print(f"  Infer time :  {infer_time:.2f} s")
    print("-" * 54)
    print(f"  Flood px   :  {flood_pixels:>10,}  ({flood_pct:.1f} %)")
    print(f"  Land px    :  {land_pixels:>10,}  ({100 - flood_pct:.1f} %)")
    print(f"  Total px   :  {total_pixels:>10,}")
    print(f"  Est. area  :  {flood_area_km2:.3f} km²  (10 m/px assumed)")
    print("=" * 54)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="FloodSense — TerraMind SAR flood detection inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the Lightning .ckpt checkpoint file",
    )
    parser.add_argument(
        "input",
        help="Path to input SAR data (.npy or .tif)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="flood_mask.png",
        help="Output PNG path (default: flood_mask.png)",
    )
    args = parser.parse_args()

    # -- Validate inputs ------------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        logger.info(
            "Tip: run  python sample_input/generate_synthetic.py  "
            "to create a demo SAR tile."
        )
        sys.exit(1)

    # -- Load model -----------------------------------------------------------
    logger.info("Loading TerraMind checkpoint …")
    t_load_start = time.perf_counter()
    load_checkpoint(args.checkpoint)
    load_time = time.perf_counter() - t_load_start
    logger.info("Model ready in %.1f s on %s", load_time, device_name())

    # -- Load SAR input -------------------------------------------------------
    ext = os.path.splitext(args.input)[1].lower()
    logger.info("Loading SAR input: %s", args.input)

    if ext == ".npy":
        sar_array = load_npy(args.input)
    elif ext in (".tif", ".tiff"):
        sar_array = load_tif(args.input)
    else:
        logger.error(
            "Unsupported input format '%s'. Use .npy or .tif.", ext
        )
        sys.exit(1)

    logger.info(
        "Input shape: %s  dtype: %s  range: [%.3f, %.3f]",
        sar_array.shape, sar_array.dtype,
        float(sar_array.min()), float(sar_array.max()),
    )

    # -- Run inference --------------------------------------------------------
    logger.info("Running inference …")
    t_infer_start = time.perf_counter()
    mask = run_inference(sar_array)
    infer_time = time.perf_counter() - t_infer_start
    logger.info("Inference complete in %.2f s", infer_time)

    # -- Save + report --------------------------------------------------------
    save_mask_png(mask, args.output)
    print_report(mask, args.input, args.checkpoint, load_time, infer_time)


if __name__ == "__main__":
    main()
