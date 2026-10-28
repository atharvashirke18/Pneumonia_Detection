"""
run_exp02a.py — Experiment 2A: Grad-CAM Explainability and Visual Shortcut Analysis

This script:
1. Loads best_model.pt (frozen, eval mode, weights unchanged).
2. Runs inference on val_manifest.csv to reproduce predictions and probabilities.
3. Selects examples using a documented, reproducible rule (all errors + stratified
   confidence samples from correct predictions).
4. Generates Grad-CAM heatmaps and overlays for each selected example.
5. Saves originals, heatmaps, and overlays as PNGs and a contact sheet.
6. Writes a JSON metadata manifest for all selected examples.
7. Does NOT access the test set, retrain, fine-tune, calibrate, or change threshold.

Target layer: backbone.conv_head (Conv2d 320->1280 k=1x1, output 7x7)
Threshold:    0.5 (unchanged from Experiment 1)
"""
import json
import os
import sys
import types
import importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT    = PROJECT_ROOT / "experiments/exp01_efficientnetb0_baseline/best_model.pt"
OUTPUT_DIR    = PROJECT_ROOT / "experiments/exp02a_gradcam"
THRESHOLD     = 0.5

# ── Bypass src/__init__.py to avoid sklearn DLL block ────────────────────────
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(PROJECT_ROOT / "src")]
src_pkg.__package__ = "src"
sys.modules["src"] = src_pkg

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

config_mod   = _load("src/config.py",    "src.config")
dataset_mod  = _load("src/data/dataset.py", "src.data.dataset")
datamod_mod  = _load("src/data/datamodule.py", "src.data.datamodule")
model_mod    = _load("src/models/model.py", "src.models.model")
gcam_mod     = _load("src/explainability/gradcam.py", "src.explainability.gradcam")

load_config           = config_mod.load_config
PneumoniaDataModule   = datamod_mod.PneumoniaDataModule
EfficientNetB0Classifier = model_mod.EfficientNetB0Classifier
GradCAM               = gcam_mod.GradCAM
get_target_layer      = gcam_mod.get_target_layer
overlay_heatmap       = gcam_mod.overlay_heatmap
denormalize_image     = gcam_mod.denormalize_image


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.clip(-500, 500)))


def section(title):
    print(); print("=" * 70); print(f"  {title}"); print("=" * 70)


# ─── 1. Setup ─────────────────────────────────────────────────────────────────
section("1. SETUP")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "originals").mkdir(exist_ok=True)
(OUTPUT_DIR / "heatmaps").mkdir(exist_ok=True)
(OUTPUT_DIR / "overlays").mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
print(f"  Checkpoint: {CHECKPOINT}")
print(f"  Output dir: {OUTPUT_DIR}")

# ─── 2. Load model (frozen, eval mode) ────────────────────────────────────────
section("2. LOAD MODEL (frozen weights, eval mode)")
raw_ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = EfficientNetB0Classifier(pretrained=False)
model.load_state_dict(raw_ckpt["model_state_dict"])
model.eval()
model.to(device)

ckpt_epoch  = raw_ckpt.get("epoch", "?")
ckpt_arch   = raw_ckpt.get("architecture", "?")
ckpt_metrics = raw_ckpt.get("metrics", {})
print(f"  Architecture     : {ckpt_arch}")
print(f"  Checkpoint epoch : {ckpt_epoch} (training epoch {int(ckpt_epoch)+1})")
print(f"  Stored val_loss  : {ckpt_metrics.get('val_loss','?')}")

target_layer = get_target_layer(model)
print(f"  Grad-CAM target  : backbone.conv_head  "
      f"({target_layer.in_channels}->{target_layer.out_channels}, k={target_layer.kernel_size})")

# ─── 3. Load val dataset (eval transforms only) ────────────────────────────────
section("3. VAL DATASET")
cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
dm  = PneumoniaDataModule(cfg)
dm.setup("fit")
val_ds = dm.val_dataset
print(f"  Val manifest  : {cfg['dataset']['resolved_manifest_paths']['val']}")
print(f"  Val images    : {len(val_ds)}")
class_counts = val_ds.get_class_counts() if hasattr(val_ds, "get_class_counts") else {}
print(f"  Class counts  : {class_counts}")

# ─── 4. Full inference pass to get logits, probs, predictions ─────────────────
section("4. FULL INFERENCE PASS (eval mode, no_grad)")
loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

all_logits, all_targets, all_indices = [], [], []
idx = 0
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1).cpu().float().numpy()
        all_logits.extend(logits.tolist())
        all_targets.extend(labels.numpy().astype(int).tolist())
        all_indices.extend(range(idx, idx + len(labels)))
        idx += len(labels)

all_logits  = np.array(all_logits)
all_targets = np.array(all_targets)
all_probs   = sigmoid_np(all_logits)
all_preds   = (all_probs >= THRESHOLD).astype(int)

# Derive outcome labels
def outcome(gt, pred):
    if gt == 1 and pred == 1: return "TP"
    if gt == 0 and pred == 0: return "TN"
    if gt == 0 and pred == 1: return "FP"
    return "FN"

outcomes = [outcome(all_targets[i], all_preds[i]) for i in range(len(all_targets))]

tp_idx = [i for i, o in enumerate(outcomes) if o == "TP"]
tn_idx = [i for i, o in enumerate(outcomes) if o == "TN"]
fp_idx = [i for i, o in enumerate(outcomes) if o == "FP"]
fn_idx = [i for i, o in enumerate(outcomes) if o == "FN"]

print(f"  TP={len(tp_idx)}  TN={len(tn_idx)}  FP={len(fp_idx)}  FN={len(fn_idx)}")

# ─── 5. Systematic example selection ──────────────────────────────────────────
section("5. EXAMPLE SELECTION (documented reproducible rule)")
"""
Selection rule (documented):
  A. ALL errors: every FP and FN (n=4 total, feasible to include all).
  B. Correct TP stratified by PNEUMONIA probability:
       - 2 highest-confidence   (prob rank top-2 among TPs)
       - 2 lowest-confidence    (prob rank bottom-2 among TPs, i.e. borderline)
       - 2 near-median          (prob rank nearest to median of TPs)
  C. Correct TN stratified by NORMAL confidence (1 - prob):
       - 2 highest-confidence   (prob rank bottom-2 among TNs)
       - 2 lowest-confidence    (prob rank nearest to 0.5 among TNs, i.e. borderline)
       - 2 near-median          (prob rank nearest to median of TNs)
  Total target: ~16 examples, covering all error types and confidence range.
  Selection is deterministic (rank-based, no randomness).
"""

def stratified_select(indices, probs, ascending=False, n_top=2, n_bottom=2, n_mid=2):
    """Select n_top, n_mid, n_mid from sorted indices. Returns list of ds indices."""
    if not indices:
        return []
    vals = [(i, probs[i]) for i in indices]
    vals.sort(key=lambda x: x[1], reverse=not ascending)
    n = len(vals)
    selected = set()
    # Top
    for _, (i, _) in zip(range(n_top), vals):
        selected.add(i)
    # Bottom
    for _, (i, _) in zip(range(n_bottom), reversed(vals)):
        selected.add(i)
    # Near median
    mid_rank = n // 2
    for offset in range(n):
        for sign in [0, -1, 1]:
            r = mid_rank + sign * offset
            if 0 <= r < n and len(selected) < n_top + n_bottom + n_mid:
                selected.add(vals[r][0])
        if len(selected) >= n_top + n_bottom + n_mid:
            break
    return sorted(selected)

selected_tp = stratified_select(tp_idx, all_probs, ascending=False)
selected_tn = stratified_select(tn_idx, all_probs, ascending=True)   # lowest prob = most confident TN
selected_fp = fp_idx   # all errors
selected_fn = fn_idx   # all errors

all_selected = sorted(set(selected_tp + selected_tn + selected_fp + selected_fn))

print(f"  Selected TP    : {len(selected_tp)}  indices={selected_tp}")
print(f"  Selected TN    : {len(selected_tn)}  indices={selected_tn}")
print(f"  Selected FP    : {len(selected_fp)}  (all errors)")
print(f"  Selected FN    : {len(selected_fn)}  (all errors)")
print(f"  Total selected : {len(all_selected)}")

# ─── 6. Grad-CAM generation ───────────────────────────────────────────────────
section("6. GRAD-CAM GENERATION")

# We need single-image inference WITH gradients, so we iterate dataset items directly
# val_ds[i] returns (image_tensor, label)
# Use model in eval mode; Grad-CAM enables grad inside generate()

IMG_SIZE  = (224, 224)
CLASS_MAP = {0: "NORMAL", 1: "PNEUMONIA"}
meta_records = []

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def tensor_to_pil(tensor_chw):
    """Denormalise a (C,H,W) tensor and return PIL Image (RGB)."""
    img_np = denormalize_image(tensor_chw.unsqueeze(0))
    return Image.fromarray(img_np, mode="RGB")

for ds_idx in all_selected:
    image_tensor, label = val_ds[ds_idx]
    gt_label   = int(label)
    prob       = float(all_probs[ds_idx])
    pred_label = int(all_preds[ds_idx])
    oc         = outcomes[ds_idx]
    correct    = (gt_label == pred_label)

    # Safe filename stem
    stem = f"{oc}_{ds_idx:04d}_p{prob:.3f}"

    # Denormalised original image
    orig_np = denormalize_image(image_tensor.unsqueeze(0))
    orig_pil = Image.fromarray(orig_np, mode="RGB")
    orig_pil.save(OUTPUT_DIR / "originals" / f"{stem}_orig.png")

    # Grad-CAM heatmap (enable grad, model stays in eval mode, weights not modified)
    input_tensor = image_tensor.unsqueeze(0).to(device)
    model.eval()  # assert eval mode before each call

    with GradCAM(model, target_layer) as gcam:
        heatmap = gcam.generate(input_tensor, output_size=IMG_SIZE)

    # Sanity checks
    assert heatmap.shape == IMG_SIZE,   f"[{stem}] Unexpected heatmap shape {heatmap.shape}"
    assert np.all(np.isfinite(heatmap)), f"[{stem}] Heatmap has non-finite values"
    assert heatmap.dtype == np.float32,  f"[{stem}] Heatmap dtype {heatmap.dtype}"

    # Save raw heatmap as coloured PNG
    import matplotlib
    _jet = matplotlib.colormaps["jet"] if hasattr(matplotlib, "colormaps") else matplotlib.cm.jet
    heatmap_rgb = (_jet(heatmap)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(heatmap_rgb, mode="RGB").save(
        OUTPUT_DIR / "heatmaps" / f"{stem}_cam.png"
    )

    # Overlay (alpha=0.45 keeps X-ray readable)
    overlay_np  = overlay_heatmap(orig_np, heatmap, alpha=0.45)
    Image.fromarray(overlay_np, mode="RGB").save(
        OUTPUT_DIR / "overlays" / f"{stem}_overlay.png"
    )

    # Heatmap statistics for shortcut analysis
    h_mean = float(heatmap.mean())
    h_max  = float(heatmap.max())
    # Border activation: mean activation in outer 10% ring vs inner 80% core
    margin = int(0.1 * IMG_SIZE[0])
    border_mask = np.zeros(IMG_SIZE, dtype=bool)
    border_mask[:margin, :]  = True
    border_mask[-margin:, :] = True
    border_mask[:, :margin]  = True
    border_mask[:, -margin:] = True
    border_mean = float(heatmap[border_mask].mean())
    inner_mean  = float(heatmap[~border_mask].mean())

    record = {
        "ds_idx": ds_idx,
        "stem": stem,
        "gt_label": gt_label,
        "gt_class": CLASS_MAP[gt_label],
        "pred_label": pred_label,
        "pred_class": CLASS_MAP[pred_label],
        "pneumonia_prob": round(prob, 6),
        "outcome": oc,
        "correct": correct,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch_0indexed": int(ckpt_epoch),
        "target_layer": "backbone.conv_head",
        "heatmap_min": 0.0,
        "heatmap_max": round(h_max, 6),
        "heatmap_mean": round(h_mean, 6),
        "border_mean_activation": round(border_mean, 6),
        "inner_mean_activation": round(inner_mean, 6),
        "border_to_inner_ratio": round(border_mean / max(inner_mean, 1e-8), 4),
        "files": {
            "original": f"originals/{stem}_orig.png",
            "heatmap": f"heatmaps/{stem}_cam.png",
            "overlay": f"overlays/{stem}_overlay.png",
        }
    }
    meta_records.append(record)
    print(f"  [{oc:2s}] idx={ds_idx:4d}  prob={prob:.4f}  border/inner={border_mean/max(inner_mean,1e-8):.2f}  "
          f"correct={correct}  -> {stem}")

# ─── 7. Contact sheet ─────────────────────────────────────────────────────────
section("7. CONTACT SHEET")

n_examples = len(all_selected)
n_cols = 4  # orig | heatmap | overlay | (spacer for label)
n_rows = n_examples

fig_w = 3 * 3  # 3 panels × 3 inches each
fig_h = n_rows * 3
fig, axes = plt.subplots(n_rows, 3, figsize=(fig_w, fig_h))
if n_rows == 1:
    axes = axes[np.newaxis, :]

for row, record in enumerate(meta_records):
    base = OUTPUT_DIR
    orig_arr    = np.array(Image.open(base / record["files"]["original"]))
    heat_arr    = np.array(Image.open(base / record["files"]["heatmap"]))
    overlay_arr = np.array(Image.open(base / record["files"]["overlay"]))

    axes[row, 0].imshow(orig_arr, cmap="gray")
    axes[row, 0].axis("off")
    axes[row, 1].imshow(heat_arr)
    axes[row, 1].axis("off")
    axes[row, 2].imshow(overlay_arr)
    axes[row, 2].axis("off")

    oc   = record["outcome"]
    prob = record["pneumonia_prob"]
    gt   = record["gt_class"]
    axes[row, 0].set_title(
        f"{oc} | GT={gt}\np(pneum)={prob:.3f}", fontsize=7, pad=2
    )
    axes[row, 1].set_title("Grad-CAM", fontsize=7, pad=2)
    axes[row, 2].set_title("Overlay", fontsize=7, pad=2)

fig.suptitle(
    "Experiment 2A — Grad-CAM (backbone.conv_head) | Val split only\n"
    "VALIDATION RESULTS ONLY — not test-set or clinical performance",
    fontsize=8, y=1.002
)
plt.tight_layout(pad=0.5)
contact_path = OUTPUT_DIR / "contact_sheet.png"
plt.savefig(contact_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved contact sheet: {contact_path}")

# ─── 8. Quantitative shortcut summary ─────────────────────────────────────────
section("8. QUANTITATIVE SHORTCUT ANALYSIS (border vs inner activation ratio)")
print(f"  Metric: mean heatmap activation in outer-10% image border / inner 80% core")
print(f"  Threshold border/inner > 1.0 suggests possible edge/border activation.")
print()
print(f"  {'Outcome':5s}  {'idx':5s}  {'prob':6s}  {'border':7s}  {'inner':7s}  {'ratio':6s}  {'flag'}")
print(f"  {'-'*58}")
for r in meta_records:
    ratio = r["border_to_inner_ratio"]
    flag  = "BORDER-DOMINANT" if ratio > 1.0 else ""
    print(f"  {r['outcome']:5s}  {r['ds_idx']:5d}  {r['pneumonia_prob']:.4f}  "
          f"{r['border_mean_activation']:.4f}  {r['inner_mean_activation']:.4f}  "
          f"{ratio:.3f}  {flag}")

border_dominant = [r for r in meta_records if r["border_to_inner_ratio"] > 1.0]
print(f"\n  Border-dominant cases (ratio > 1.0): {len(border_dominant)} / {len(meta_records)}")

# ─── 9. Save metadata manifest ───────────────────────────────────────────────
meta_path = OUTPUT_DIR / "gradcam_metadata.json"
manifest  = {
    "experiment": "exp02a_gradcam",
    "checkpoint": str(CHECKPOINT),
    "checkpoint_epoch_0indexed": int(ckpt_epoch),
    "target_layer": "backbone.conv_head",
    "target_layer_description": "Conv2d(320->1280, k=1x1), output (1,1280,7,7) at input 224x224",
    "threshold": THRESHOLD,
    "val_manifest": cfg["dataset"]["resolved_manifest_paths"]["val"],
    "selection_rule": (
        "ALL FP and FN errors; TPs stratified by PNEUMONIA probability "
        "(top-2, bottom-2, near-median-2); TNs stratified by NORMAL confidence "
        "(lowest-prob-2, highest-prob-2, near-median-2). "
        "Deterministic rank-based selection, no randomness."
    ),
    "disclaimer": (
        "All metrics and heatmaps are based on the internal validation split. "
        "Not test-set results. No training, fine-tuning, or threshold changes performed."
    ),
    "examples": meta_records,
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"\n  Metadata manifest: {meta_path}")

# ─── Final artifact inventory ─────────────────────────────────────────────────
section("ARTIFACT INVENTORY")
for p in sorted(OUTPUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUTPUT_DIR)}  ({p.stat().st_size/1024:.1f} KB)")

section("COMPLETE")
print("  Experiment 2A artifacts saved to:", OUTPUT_DIR)
print("  Test set NOT accessed. No training. Threshold unchanged (0.5).")
print("  Awaiting approval before proceeding further.")
