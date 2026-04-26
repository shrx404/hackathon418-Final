"""
=============================================================================
 Flood Model — Orbital Edge Inference (v2)
=============================================================================
 Generates a binary flood segmentation mask from a satellite tile ID.
 Uses PIL/Pillow to create realistic-looking grayscale flood masks.

 In production, replace generate_mask() with actual IBM TerraMind inference:
   from terratorch.models import PrithviModelFactory
   model = PrithviModelFactory().build_model(...)
   mask = model(sentinel1_tensor)
=============================================================================
"""

import io
import base64
import random
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Attempt to import Pillow — required for proper PNG generation
try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
    logger.info("Pillow available — using PIL for mask generation.")
except ImportError:
    PIL_AVAILABLE = False
    logger.warning(
        "Pillow not installed. Run `pip install Pillow`. "
        "Falling back to minimal SVG mask."
    )


class FloodModel:
    """
    Encapsulates flood mask generation logic.

    Lifecycle:
        1. flood_model.load(path)   — load real model weights
        2. flood_model.generate_mask(tile_id) -> (b64_str, flooded_km2, size_kb)
    """

    def __init__(self) -> None:
        self._loaded = False

    def load(self, model_path: str = "") -> None:
        """
        Load model weights from disk.

        To integrate real IBM TerraMind-1.0-small inference:
            import torch
            from terratorch.models import build_prithvi_model
            self._model = build_prithvi_model(checkpoint=model_path)
            self._model.eval()
        """
        # STUB: no real weights loaded; using procedural mask generation
        logger.info("FloodModel stub active — using procedural mask generation.")
        self._loaded = True

    def generate_mask(self, tile_id: str) -> Tuple[str, float, float]:
        """
        Generate a binary flood segmentation mask for the given tile.

        Args:
            tile_id: Identifier string for the satellite tile.

        Returns:
            Tuple of:
                - b64_str: Base64-encoded PNG image string
                - flooded_area_sq_km: Estimated flooded area
                - mask_size_kb: Size of the encoded PNG in kilobytes
        """
        # Seed random with tile_id hash for reproducible results per tile
        seed = abs(hash(tile_id)) % 99999
        random.seed(seed)

        # Estimate flooded area based on tile ID (deterministic per tile)
        flooded_area = round(random.uniform(80, 520), 1)

        if PIL_AVAILABLE:
            return self._generate_pil_mask(flooded_area)
        else:
            return self._generate_svg_fallback(flooded_area)

    def _generate_pil_mask(self, flooded_area: float) -> Tuple[str, float, float]:
        """
        Generate a PNG flood mask using Pillow.
        Produces a 512x512 binary segmentation mask (white=flood, black=land).
        """
        SIZE = 512
        img = Image.new("L", (SIZE, SIZE), color=0)  # Black background = land
        draw = ImageDraw.Draw(img)

        # -- Draw river/water body simulation --------------------------------
        # Main river channel (sinusoidal-ish polygon)
        river_pts = []
        import math
        for i in range(0, SIZE + 1, 8):
            x = int(SIZE * 0.4 + math.sin(i / 60) * SIZE * 0.12)
            river_pts.append((x, i))
        for i in range(SIZE, -1, -8):
            x = int(SIZE * 0.4 + math.sin(i / 60) * SIZE * 0.12 + random.randint(18, 35))
            river_pts.append((x, i))
        if len(river_pts) >= 3:
            draw.polygon(river_pts, fill=255)

        # -- Draw flood extent polygons (inundated fields) -------------------
        num_patches = random.randint(4, 9)
        for _ in range(num_patches):
            cx = random.randint(60, SIZE - 60)
            cy = random.randint(40, SIZE - 40)
            rx = random.randint(25, 90)
            ry = random.randint(15, 55)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

        # -- Add some irregular edges via slight blur + re-threshold ----------
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
        img = img.point(lambda p: 255 if p > 100 else 0)

        # Encode to PNG
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        raw_bytes = buffer.getvalue()

        mask_size_kb = len(raw_bytes) / 1024.0
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")

        logger.debug(
            "PIL mask generated: %.1f KB, flooded area %.1f km²",
            mask_size_kb, flooded_area
        )
        return b64_str, flooded_area, mask_size_kb

    def _generate_svg_fallback(self, flooded_area: float) -> Tuple[str, float, float]:
        """
        Minimal SVG mask for environments without Pillow.
        Returns a simple SVG encoded as base64.
        """
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">'
            '<rect width="512" height="512" fill="black"/>'
            '<ellipse cx="180" cy="160" rx="120" ry="60" fill="white"/>'
            '<ellipse cx="300" cy="300" rx="100" ry="50" fill="white"/>'
            '<ellipse cx="220" cy="400" rx="80" ry="40" fill="white"/>'
            '<rect x="160" y="120" width="30" height="300" fill="white"/>'
            '</svg>'
        )
        svg_bytes = svg.encode("utf-8")
        b64_str = base64.b64encode(svg_bytes).decode("utf-8")
        mask_size_kb = len(svg_bytes) / 1024.0
        return b64_str, flooded_area, mask_size_kb


# Module-level singleton
flood_model = FloodModel()
