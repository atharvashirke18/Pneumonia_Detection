"""FastAPI backend for the Pneumonia Detection Web UI."""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXP01_DIR = ROOT / "experiments/exp01_efficientnetb0_baseline"
EXP02A_DIR = ROOT / "experiments/exp02a_gradcam"
EXP02B_DIR = ROOT / "experiments/exp02b_gradcam_sensitivity_pilot"
WEB_DIR = ROOT / "web"

# Saved artifacts never change while the server runs: let the browser cache them so
# the inspector never re-downloads an image it has already shown.
CACHE = {"Cache-Control": "public, max-age=86400"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pneumonia_api")

_inference = None


def get_inference():
    global _inference
    if _inference is None:
        from src.ui.inference import PneumoniaInference
        log.info("Loading model checkpoint ...")
        _inference = PneumoniaInference.from_project(ROOT)
        log.info("Model loaded.")
    return _inference


app = FastAPI(title="Pneumonia Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _ndarray_to_b64(arr: np.ndarray) -> str:
    return _pil_to_b64(Image.fromarray(arr.astype(np.uint8)))


def _heatmap_to_rgb(h) -> np.ndarray:
    """Return a uint8 RGB Jet image for whatever Grad-CAM heatmap inference.explain() returns.
    A float map in [0, 1] cast straight to uint8 becomes all zeros (flat dark blue), so normalise first."""
    a = np.asarray(h, dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] in (3, 4):          # already a colour image
        a = a[..., :3]
        return (a * 255.0 if a.max() <= 1.0 else a).clip(0, 255).astype(np.uint8)
    a = np.squeeze(a)
    lo, hi = float(a.min()), float(a.max())
    v = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    c = lambda t: np.clip(t, 0.0, 1.0)
    rgb = np.stack([c(1.5 - np.abs(4 * v - 3)), c(1.5 - np.abs(4 * v - 2)), c(1.5 - np.abs(4 * v - 1))], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    from src.ui.inference import UploadValidationError, decode_uploaded_image
    payload = await file.read()
    try:
        image = decode_uploaded_image(payload, file.filename or "upload.jpg")
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    inference = get_inference()
    pred = inference.predict(image)

    hm_b64 = overlay_b64 = native_info = None
    native_h = native_w = 7
    try:
        native_hm, heatmap, overlay_arr = inference.explain(image)
        hm_b64 = _ndarray_to_b64(_heatmap_to_rgb(heatmap))
        overlay_b64 = _ndarray_to_b64(overlay_arr)
        native_h, native_w = native_hm.shape[:2]
    except Exception as exc:
        log.warning("Grad-CAM failed: %s", exc)

    return {
        "class_name": pred.class_name,
        "pneumonia_probability": pred.pneumonia_probability,
        "threshold": 0.5,
        "original_b64": _pil_to_b64(image),
        "heatmap_b64": hm_b64,
        "overlay_b64": overlay_b64,
        "gradcam_details": {
            "target_layer": "backbone.conv_head",
            "native_resolution": f"{native_h}x{native_w}",
            "upsampling": "Bicubic (224x224)",
            "normalization": "Min-max [0, 1]",
            "relu": "Applied (pre-upsampling)",
        },
    }


@app.get("/api/experiments/exp01/training-history")
def exp01_training_history():
    p = EXP01_DIR / "training_history.json"
    if not p.exists():
        raise HTTPException(404, "Not found")
    return json.loads(p.read_text())


@app.get("/api/experiments/exp01/metrics")
def exp01_metrics():
    p = EXP01_DIR / "training_history.json"
    if not p.exists():
        raise HTTPException(404, "Not found")
    history = json.loads(p.read_text())
    best = min(history, key=lambda r: r["val_loss"])
    return {
        "best_epoch": best["epoch"],
        "val_auc": round(best["val_auc"], 6),
        "val_sensitivity": round(best["val_sensitivity"], 6),
        "val_specificity": round(best["val_specificity"], 6),
        "val_f1": round(best["val_f1"], 6),
        "val_loss": round(best["val_loss"], 6),
    }


@app.get("/api/experiments/exp02a/metadata")
def exp02a_metadata():
    p = EXP02A_DIR / "gradcam_metadata.json"
    if not p.exists():
        raise HTTPException(404, "Not found")
    return json.loads(p.read_text())


@app.get("/api/experiments/exp02a/numerical-audit")
def exp02a_numerical_audit():
    p = EXP02A_DIR / "numerical_audit/gradcam_numerical_audit.csv"
    if not p.exists():
        raise HTTPException(404, "Not found")
    rows = []
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


@app.get("/api/experiments/exp02b/summary")
def exp02b_summary():
    p = EXP02B_DIR / "sensitivity_summary.json"
    if not p.exists():
        raise HTTPException(404, "Not found")
    return json.loads(p.read_text())


@app.get("/api/experiments/exp02b/results")
def exp02b_results():
    p = EXP02B_DIR / "sensitivity_results.csv"
    if not p.exists():
        raise HTTPException(404, "Not found")
    rows = []
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ds_idx"):
                rows.append(row)
    return rows


@app.get("/api/experiments/exp02a/audit-meta")
def exp02a_audit_meta():
    """Merge the JSON files in numerical_audit/ (confusion counts, border-dominance summary, ...).
    The large per-case 'examples' list is dropped; /numerical-audit serves those rows."""
    d = EXP02A_DIR / "numerical_audit"
    merged: dict = {}
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                merged.setdefault(k, v)
    merged.pop("examples", None)
    if not merged:
        raise HTTPException(404, "Not found")
    return merged


@app.get("/api/experiments/exp02a/image/{kind}/{filename}")
def exp02a_image(kind: str, filename: str):
    if kind not in {"originals", "heatmaps", "overlays"}:
        raise HTTPException(400, "Invalid kind")
    p = EXP02A_DIR / kind / filename
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), media_type="image/png", headers=CACHE)


@app.get("/api/experiments/exp02a/comparison/{filename}")
def exp02a_comparison(filename: str):
    p = EXP02A_DIR / "followup/comparisons" / filename
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), media_type="image/png", headers=CACHE)


@app.get("/api/experiments/exp02a/contact-sheet")
def exp02a_contact_sheet():
    p = EXP02A_DIR / "contact_sheet.png"
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), media_type="image/png", headers=CACHE)


@app.get("/api/experiments/exp02b/contact-sheet")
def exp02b_contact_sheet():
    p = EXP02B_DIR / "sensitivity_contact_sheet.png"
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), media_type="image/png", headers=CACHE)


# Mapping of ds_idx to stem for Exp 02B images
DS_IDX_TO_STEM = {
    0: "TN_0000_p0.000",
    2: "TN_0002_p0.267",
    40: "TN_0040_p0.000",
    43: "TN_0043_p0.000",
    51: "TN_0051_p0.000",
    119: "FP_0119_p0.742",
    120: "FP_0120_p0.985",
    123: "TN_0123_p0.007",
    134: "TP_0134_p1.000",
    138: "TP_0138_p1.000",
    211: "TP_0211_p0.777",
    241: "FN_0241_p0.027",
    260: "FN_0260_p0.212",
    315: "TP_0315_p0.791",
    369: "TP_0369_p1.000",
    402: "TP_0402_p1.000",
}


@app.get("/api/experiments/exp02b/image/{stem_or_idx}/{condition}")
def exp02b_perturbed_image(stem_or_idx: str, condition: str):
    valid_conds = {"original", "border_black", "central_black", "gamma_dark"}
    if condition not in valid_conds:
        raise HTTPException(400, f"Invalid condition: {condition}")

    # Check pre-generated image directory first
    images_dir = EXP02B_DIR / "images"
    if stem_or_idx.isdigit():
        direct_file = images_dir / f"{stem_or_idx}_{condition}.png"
        if direct_file.exists():
            return FileResponse(str(direct_file), media_type="image/png", headers=CACHE)

    # Resolve stem
    if stem_or_idx.isdigit() and int(stem_or_idx) in DS_IDX_TO_STEM:
        stem = DS_IDX_TO_STEM[int(stem_or_idx)]
    else:
        stem = stem_or_idx

    orig_path = EXP02A_DIR / "originals" / f"{stem}_orig.png"
    if not orig_path.exists():
        matches = list((EXP02A_DIR / "originals").glob(f"*{stem}*.png"))
        if matches:
            orig_path = matches[0]
        else:
            raise HTTPException(404, f"Original image for {stem_or_idx} not found")

    img = Image.open(orig_path).convert("RGB")
    if condition == "original":
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png", headers=CACHE)

    arr = np.array(img, dtype=np.uint8).copy()
    if condition == "border_black":
        m = 32
        arr[:m, :, :] = 0
        arr[-m:, :, :] = 0
        arr[:, :m, :] = 0
        arr[:, -m:, :] = 0
    elif condition == "central_black":
        H, W = arr.shape[:2]
        hs = 156 // 2
        cy, cx = H // 2, W // 2
        r0 = max(0, cy - hs)
        r1 = min(H, r0 + 156)
        c0 = max(0, cx - hs)
        c1 = min(W, c0 + 156)
        arr[r0:r1, c0:c1, :] = 0
    elif condition == "gamma_dark":
        arr_f = arr.astype(np.float32) / 255.0
        arr_f = np.power(np.clip(arr_f, 0, 1), 1.0 / 0.85)
        arr = (arr_f * 255.0).round().clip(0, 255).astype(np.uint8)

    out_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="image/png", headers=CACHE)


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
