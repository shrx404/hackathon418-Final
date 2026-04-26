"""
=============================================================================
 src/model.py — TerraMind v1 Tiny checkpoint loader & inference engine
=============================================================================
 Loads the fine-tuned SemanticSegmentationTask from a Lightning .ckpt file
 and exposes a single run_inference() function for SAR flood detection.

 The checkpoint was produced by terratorch >= 1.2.4 using:
   - Backbone : terramind_v1_tiny
   - Decoder  : UperNetDecoder
   - Input    : Sentinel-1 GRD, 2 bands (VV, VH), shape (B, 2, H, W)
   - Output   : logits (B, 2, H, W) → argmax → binary mask (0=land, 1=flood)
   - mIoU     : 0.78 on Sen1Floods11 hand-labeled test split

 Device selection:
   - CUDA GPU if available (preferred)
   - CPU fallback (slow, ~30 s per 512×512 tile)
=============================================================================
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Module-level singleton state
# ---------------------------------------------------------------------------
_task = None          # SemanticSegmentationTask instance
_device: Optional[torch.device] = None


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str) -> None:
    """
    Load the TerraMind SemanticSegmentationTask from a Lightning .ckpt file.

    Selects GPU automatically; falls back to CPU if CUDA is unavailable.
    Sets the module-level ``_task`` and ``_device`` singletons.

    Args:
        ckpt_path: Absolute or relative path to the .ckpt file.

    Raises:
        FileNotFoundError: If ckpt_path does not exist.
        ImportError:       If terratorch is not installed.
    """
    global _task, _device

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        from terratorch.tasks import SemanticSegmentationTask
    except ImportError as exc:
        raise ImportError(
            "terratorch is required to load the checkpoint.\n"
            "Install it with:  pip install 'terratorch>=1.2.4'"
        ) from exc

    # -- Device selection: GPU → CPU ----------------------------------------
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        logger.info("CUDA available — loading checkpoint on GPU.")
    else:
        _device = torch.device("cpu")
        logger.warning(
            "CUDA not available — loading checkpoint on CPU. "
            "Inference will be slow (~30 s per tile)."
        )

    logger.info("Loading checkpoint from: %s", ckpt_path)
    _task = SemanticSegmentationTask.load_from_checkpoint(
        ckpt_path,
        map_location=_device,
    )
    _task.eval()
    _task.to(_device)
    logger.info("TerraMind checkpoint loaded successfully on %s.", _device)


def run_inference(sar_array: np.ndarray) -> np.ndarray:
    """
    Run flood segmentation inference on a 2-band SAR array.

    Args:
        sar_array: np.ndarray of shape ``(2, H, W)``, dtype float32.
                   Band 0 = VV backscatter, Band 1 = VH backscatter.

    Returns:
        np.ndarray of shape ``(H, W)``, dtype uint8.
        Pixel values: **0 = land / no flood**, **1 = flood / water**.

    Raises:
        RuntimeError: If ``load_checkpoint()`` has not been called first.
        ValueError:   If sar_array has an unexpected shape.
    """
    if _task is None:
        raise RuntimeError(
            "No model loaded. Call src.model.load_checkpoint(ckpt_path) first."
        )

    if sar_array.ndim != 3 or sar_array.shape[0] != 2:
        raise ValueError(
            f"Expected SAR array of shape (2, H, W), got {sar_array.shape}."
        )

    # -- Prepare tensor [1, 2, H, W] ----------------------------------------
    tensor = (
        torch.tensor(sar_array, dtype=torch.float32)
        .unsqueeze(0)
        .to(_device)
    )

    with torch.no_grad():
        model_output = _task(tensor)

        # terratorch wraps outputs in ModelOutput / named tuples
        logits = _extract_logits(model_output)

        # Argmax over class dimension → (1, H, W) → squeeze → (H, W)
        mask = (
            torch.argmax(logits, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    return mask


def is_loaded() -> bool:
    """Return True if a checkpoint has been successfully loaded."""
    return _task is not None


def device_name() -> str:
    """Return a human-readable string for the current inference device."""
    if _device is None:
        return "not loaded"
    if _device.type == "cuda":
        name = torch.cuda.get_device_name(_device)
        return f"GPU ({name})"
    return "CPU"


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _extract_logits(model_output) -> torch.Tensor:
    """
    Extract the raw logit tensor from whatever wrapper terratorch returns.

    Handles: ModelOutput.output, ModelOutput.logits, dict["out" / "logits"],
    list/tuple[0], and bare tensors.
    """
    if hasattr(model_output, "output") and isinstance(
        model_output.output, torch.Tensor
    ):
        return model_output.output

    if hasattr(model_output, "logits") and isinstance(
        model_output.logits, torch.Tensor
    ):
        return model_output.logits

    if isinstance(model_output, dict):
        for key in ("out", "logits", "output"):
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                return model_output[key]

    if isinstance(model_output, (list, tuple)) and len(model_output) > 0:
        return model_output[0]

    if isinstance(model_output, torch.Tensor):
        return model_output

    raise TypeError(
        f"Cannot extract logit tensor from model output of type {type(model_output)}. "
        "Please file a bug with the output object repr."
    )
