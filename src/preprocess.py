"""
=============================================================================
 src/preprocess.py — SAR data preprocessing helpers
=============================================================================
 Matches the preprocessing implicitly applied by terratorch's
 GenericNonGeoSegmentationDataModule during fine-tuning:
   - Raw float32 Sentinel-1 GRD bands (VV, VH)
   - NaN / Inf replaced with 0.0  (no_data_replace=0 in training config)
   - No explicit normalization — TerraMind's pre-trained weights handle this

 Supports loading from .npy and .tif files.
=============================================================================
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Core preprocessing
# ---------------------------------------------------------------------------

def prepare_sar(arr: np.ndarray, no_data_val: float = 0.0) -> np.ndarray:
    """
    Prepare a raw 2-band SAR array for inference.

    Steps:
        1. Cast to float32.
        2. Replace NaN / ±Inf with ``no_data_val`` (training used 0).

    Args:
        arr:         np.ndarray of shape ``(2, H, W)``.
        no_data_val: Replacement value for invalid pixels.

    Returns:
        float32 np.ndarray of the same shape.
    """
    arr = arr.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, no_data_val)
    return arr


# ---------------------------------------------------------------------------
#  File loaders
# ---------------------------------------------------------------------------

def load_npy(path: str) -> np.ndarray:
    """
    Load a (2, H, W) float32 SAR array from a .npy file.

    Args:
        path: Path to a .npy file saved with np.save().

    Returns:
        Preprocessed float32 np.ndarray of shape (2, H, W).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the array does not have shape (2, H, W).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SAR array not found: {path}")

    arr = np.load(path)

    if arr.ndim == 3 and arr.shape[0] == 2:
        logger.info("Loaded .npy SAR array: shape=%s dtype=%s", arr.shape, arr.dtype)
        return prepare_sar(arr)

    raise ValueError(
        f"Expected .npy array of shape (2, H, W), got {arr.shape}. "
        "The first dimension must be 2 (VV channel, VH channel)."
    )


def load_tif(path: str) -> np.ndarray:
    """
    Load bands 1 and 2 from a GeoTIFF Sentinel-1 file (requires rasterio).

    Args:
        path: Path to a .tif / .tiff file with at least 2 bands.

    Returns:
        Preprocessed float32 np.ndarray of shape (2, H, W).

    Raises:
        ImportError:       If rasterio is not installed.
        FileNotFoundError: If the file does not exist.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to load .tif files.\n"
            "Install it with:  pip install rasterio"
        ) from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        if src.count < 2:
            raise ValueError(
                f"Expected at least 2 bands in {path}, found {src.count}."
            )
        arr = src.read([1, 2]).astype(np.float32)

    logger.info("Loaded .tif SAR tile: shape=%s dtype=%s", arr.shape, arr.dtype)
    return prepare_sar(arr)
