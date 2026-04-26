import io
import os
import re
import sys
import base64
import pathlib
import tempfile
import subprocess
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="FloodSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── absolute paths anchored to THIS file's directory ───────────────────────────
# This guarantees correct paths regardless of where uvicorn is launched from.
HERE            = pathlib.Path(__file__).parent.resolve()
INFER_SCRIPT    = HERE / "infer.py"
CKPT_PATH       = str(HERE / "backend" / "mIoU=0.78.ckpt")
FLOOD_MASK_PATH = HERE / "flood_mask.png"   # infer.py always writes here

# ── helpers ────────────────────────────────────────────────────────────────────

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def array_to_base64_png(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        img = Image.fromarray(normalize_to_uint8(arr), mode="L")
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            img = Image.fromarray(normalize_to_uint8(arr[:, :, 0]), mode="L")
        elif c == 3:
            img = Image.fromarray(normalize_to_uint8(arr), mode="RGB")
        elif c == 4:
            img = Image.fromarray(normalize_to_uint8(arr), mode="RGBA")
        else:
            img = Image.fromarray(normalize_to_uint8(arr[:, :, 0]), mode="L")
    else:
        raise ValueError(f"Cannot visualize {arr.ndim}-D array directly.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def pick_2d_slice(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def compute_stats(arr: np.ndarray) -> dict:
    flat = arr.flatten().astype(np.float64)
    return {
        "shape":          list(arr.shape),
        "dtype":          str(arr.dtype),
        "ndim":           int(arr.ndim),
        "total_elements": int(flat.size),
        "min":            round(float(flat.min()), 6),
        "max":            round(float(flat.max()), 6),
        "mean":           round(float(flat.mean()), 6),
        "std":            round(float(flat.std()), 6),
        "median":         round(float(np.median(flat)), 6),
        "non_zero":       int(np.count_nonzero(flat)),
        "nan_count":      int(np.isnan(flat).sum()),
    }


def custom_stats(arr: np.ndarray, filename: str) -> dict:
    flat = arr.flatten().astype(np.float64)
    p25, p75 = float(np.percentile(flat, 25)), float(np.percentile(flat, 75))
    return {
        "p25 (25th percentile)": round(p25, 6),
        "p75 (75th percentile)": round(p75, 6),
        "IQR":            round(p75 - p25, 6),
        "dynamic_range":  round(float(flat.max() - flat.min()), 6),
        "rms":            round(float(np.sqrt(np.mean(flat ** 2))), 6),
    }


# ── infer.py output parsers ────────────────────────────────────────────────────

def _parse_int(pattern: str, text: str, default: int = 0) -> int:
    m = re.search(pattern, text)
    if not m:
        return default
    return int(m.group(1).replace(",", ""))


def _parse_float(pattern: str, text: str, default: float = 0.0) -> float:
    m = re.search(pattern, text)
    if not m:
        return default
    return float(m.group(1))


def _parse_str(pattern: str, text: str, default: str = "—") -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


def parse_infer_stdout(stdout: str) -> dict:
    """Extract structured data from infer.py's printed report."""
    return {
        "flood_px":   _parse_int(r"Flood px\s*:\s*([\d,]+)", stdout),
        "land_px":    _parse_int(r"Land px\s*:\s*([\d,]+)", stdout),
        "total_px":   _parse_int(r"Total px\s*:\s*([\d,]+)", stdout),
        "flood_pct":  _parse_float(r"Flood px.*?\(([\d.]+)\s*%\)", stdout),
        "land_pct":   _parse_float(r"Land px.*?\(([\d.]+)\s*%\)", stdout),
        "area_km2":   _parse_float(r"Est\.\s*area\s*:\s*([\d.]+)", stdout),
        "load_time":  _parse_float(r"Load time\s*:\s*([\d.]+)", stdout),
        "infer_time": _parse_float(r"Infer time\s*:\s*([\d.]+)", stdout),
        "device":     _parse_str(r"Device\s*:\s*(.+)", stdout),
        "checkpoint": _parse_str(r"Checkpoint\s*:\s*(.+)", stdout),
        "model":      _parse_str(r"Model\s*:\s*(.+)", stdout),
        "miou":       _parse_str(r"Test mIoU\s*:\s*(.+)", stdout),
    }


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse(str(HERE / "index.html"))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in {".tif", ".tiff", ".npy"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .tif, .tiff, or .npy file.",
        )

    raw = await file.read()
    file_size_kb = round(len(raw) / 1024, 2)
    file_meta: dict = {
        "filename":     filename,
        "file_size_kb": file_size_kb,
        "file_size_mb": round(file_size_kb / 1024, 4),
        "extension":    ext,
    }

    if ext in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError:
            raise HTTPException(status_code=500, detail="tifffile is not installed.")

        with tifffile.TiffFile(io.BytesIO(raw)) as tif:
            arr = tif.asarray()
            page = tif.pages[0]
            file_meta["num_pages"]   = len(tif.pages)
            file_meta["compression"] = str(page.compression.name if hasattr(page.compression, "name") else page.compression)
            file_meta["tiff_format"] = (
                "ImageJ TIFF" if tif.is_imagej else
                "OME-TIFF"    if tif.is_ome    else
                "GeoTIFF"     if tif.is_geotiff else
                "Standard TIFF"
            )
            tag = page.tags.get("BitsPerSample")
            if tag:
                file_meta["bits_per_sample"] = tag.value
    else:
        arr = np.load(io.BytesIO(raw))
        file_meta["tiff_format"] = "NumPy Array (.npy)"

    stats   = compute_stats(arr)
    c_stats = custom_stats(arr, filename)
    vis     = pick_2d_slice(arr)

    try:
        img_b64 = array_to_base64_png(vis)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not render image: {exc}")

    return JSONResponse({
        "image":        img_b64,
        "stats":        stats,
        "metadata":     file_meta,
        "custom_stats": c_stats,
    })


@app.post("/infer")
async def run_inference(file: UploadFile = File(...)):
    """
    Run FloodSense inference on an uploaded .tif / .npy SAR file.
    Calls infer.py as a subprocess, parses its stdout report,
    and returns structured JSON including the flood mask as base64.
    """
    filename = file.filename or "input.npy"
    ext = pathlib.Path(filename).suffix.lower()

    if ext not in {".tif", ".tiff", ".npy"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    raw = await file.read()

    # write to temp file so infer.py can load it normally
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    # --- THE FIX: Clone environment and force CUDA ---
    custom_env = os.environ.copy()
    custom_env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Optional: Force high performance profile on some Windows setups
    custom_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 

    try:
        proc = subprocess.run(
            [sys.executable, str(INFER_SCRIPT), CKPT_PATH, tmp_path],
            capture_output=True,
            text=True,
            timeout=600,       
            cwd=str(HERE),     
            env=custom_env,    # <-- Inject the forced environment here
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference timed out.")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    # logging goes to stderr, the printed report goes to stdout
    combined_output = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"infer.py exited {proc.returncode}:\n{proc.stderr[-1200:]}",
        )

    result = parse_infer_stdout(combined_output)
    result["filename"] = filename

    # infer.py writes flood_mask.png relative to CWD (= HERE)
    if FLOOD_MASK_PATH.exists():
        result["flood_mask"] = base64.b64encode(FLOOD_MASK_PATH.read_bytes()).decode()
    else:
        result["flood_mask"] = None

    return JSONResponse(result)