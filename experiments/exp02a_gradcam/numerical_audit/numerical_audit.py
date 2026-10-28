"""
numerical_audit.py — Experiment 2A Grad-CAM Numerical Audit

Saves all outputs under experiments/exp02a_gradcam/numerical_audit/.
Does NOT modify existing artifacts, model weights, manifests, or checkpoint.
Does NOT access test/calibration data.
Does NOT train or fine-tune.

Audit objectives:
  1. Re-verify Grad-CAM implementation: target score, gradient capture, ReLU, normalization.
  2. Capture pre-normalization raw CAM values at native 7x7 grid for all 16 examples.
  3. Compute native 7x7 border/inner statistics using explicit 1-cell and 2-cell masks.
  4. Identify and report the silent epsilon replacement (max(inner,1e-8)) in the original
     run_exp02a.py and determine which cases it affected.
  5. Clarify the 22px vs 32px border mismatch (upsampled vs native grid).
  6. Re-verify checkpoint identity and that model logits are unchanged.
  7. Save machine-readable CSV and a concise Markdown report.
"""

import json
import sys
import types
import importlib.util
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("d:/Pneumonia_Detection")
CHECKPOINT   = PROJECT_ROOT / "experiments/exp01_efficientnetb0_baseline/best_model.pt"
ORIG_META    = PROJECT_ROOT / "experiments/exp02a_gradcam/gradcam_metadata.json"
AUDIT_DIR    = PROJECT_ROOT / "experiments/exp02a_gradcam/numerical_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD  = 0.5
IMG_SIZE   = (224, 224)   # original upsampled size
NATIVE_H   = 7
NATIVE_W   = 7

# ── Bypass src/__init__.py ────────────────────────────────────────────────────
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(PROJECT_ROOT / "src")]
src_pkg.__package__ = "src"
sys.modules["src"] = src_pkg

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

config_mod  = _load("src/config.py",    "src.config")
dataset_mod = _load("src/data/dataset.py", "src.data.dataset")
datamod_mod = _load("src/data/datamodule.py", "src.data.datamodule")
model_mod   = _load("src/models/model.py", "src.models.model")

load_config           = config_mod.load_config
PneumoniaDataModule   = datamod_mod.PneumoniaDataModule
EfficientNetB0Classifier = model_mod.EfficientNetB0Classifier


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def section(title):
    print(); print("=" * 70); print(f"  {title}"); print("=" * 70)


# ════════════════════════════════════════════════════════════════════════════
#  AuditGradCAM: captures raw intermediate values
# ════════════════════════════════════════════════════════════════════════════
class AuditGradCAM:
    """
    Extended Grad-CAM that preserves all intermediate arrays for audit.

    Captured at native 7×7 resolution (before any interpolation):
      raw_cam_pre_relu   : weighted activation sum, before ReLU (float32 numpy, (7,7))
      raw_cam_post_relu  : after ReLU, before normalization/upsampling (float32 numpy, (7,7))
      gradient_maps      : per-channel gradient maps, (1280, 7, 7) float32 numpy
      activation_maps    : per-channel activation maps, (1280, 7, 7) float32 numpy
      channel_weights    : spatially-averaged gradient weights, (1280,) float32 numpy

    generate() returns the standard normalized 224×224 heatmap, identical
    to the original GradCAM.generate() output.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self._acts  = None
        self._grads = None
        self._hooks = []
        # Audit captures
        self.raw_cam_pre_relu  = None
        self.raw_cam_post_relu = None
        self.gradient_maps     = None
        self.activation_maps   = None
        self.channel_weights   = None
        self._register()

    def _register(self):
        def fwd(m, inp, out):
            self._acts = out.detach().clone()
        def bwd(m, gi, go):
            self._grads = go[0].detach().clone()
        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def __enter__(self): return self
    def __exit__(self, *a): self.remove_hooks()

    def generate(self, image_tensor, output_size=IMG_SIZE):
        if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
            raise ValueError(f"Expected (1,C,H,W), got {image_tensor.shape}")

        self._acts  = None
        self._grads = None
        self.model.zero_grad()

        with torch.enable_grad():
            logit = self.model(image_tensor)
            score = logit[0, 0]
            score.backward()

        assert self._acts  is not None, "Forward hook did not fire"
        assert self._grads is not None, "Backward hook did not fire"

        # (1, C, 7, 7)
        acts  = self._acts   # (1, 1280, 7, 7)
        grads = self._grads  # (1, 1280, 7, 7)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, 1280, 1, 1)
        cam_raw = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)

        # ── Capture audit intermediates (squeeze to 7×7 or (C,7,7)) ──
        self.raw_cam_pre_relu  = cam_raw.squeeze().cpu().float().numpy()
        cam_relu = F.relu(cam_raw)
        self.raw_cam_post_relu = cam_relu.squeeze().cpu().float().numpy()
        self.gradient_maps     = grads.squeeze(0).cpu().float().numpy()   # (1280,7,7)
        self.activation_maps   = acts.squeeze(0).cpu().float().numpy()    # (1280,7,7)
        self.channel_weights   = weights.squeeze().cpu().float().numpy()  # (1280,)

        # ── Standard pipeline: upsample then normalize (matches original) ──
        cam_up = F.interpolate(cam_relu, size=output_size,
                               mode="bicubic", align_corners=False)
        cam_np = cam_up.squeeze().cpu().float().numpy()
        vmin, vmax = cam_np.min(), cam_np.max()
        if vmax - vmin > 1e-8:
            cam_np = (cam_np - vmin) / (vmax - vmin)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


# ════════════════════════════════════════════════════════════════════════════
#  Border mask helpers (native 7×7 grid)
# ════════════════════════════════════════════════════════════════════════════

def make_native_border_mask(h=7, w=7, n_cells=1):
    """
    Boolean mask of the outer n_cells ring on an (h,w) grid.
    1-cell border: the outermost ring  → 4*7 - 4 = 24 cells (of 49)
    2-cell border: the outer 2 rings   → 49 - 3×3 = 40 cells (of 49)
    """
    mask = np.ones((h, w), dtype=bool)
    if n_cells < min(h, w) // 2 + 1:
        mask[n_cells:h-n_cells, n_cells:w-n_cells] = False
    # If n_cells >= half-size, entire grid is border
    return mask   # True where border

def make_upsampled_border_mask(margin_px=22, size=224):
    """
    Boolean mask for the outer margin_px ring on a (size,size) image.
    This is what run_exp02a.py computed: int(0.1 * 224) = 22 px.
    """
    mask = np.zeros((size, size), dtype=bool)
    mask[:margin_px, :]   = True
    mask[-margin_px:, :]  = True
    mask[:, :margin_px]   = True
    mask[:, -margin_px:]  = True
    return mask


def border_inner_stats(arr2d, border_mask):
    """
    Compute border/inner activation stats.
    Zero denominator: ratio returned as float('nan'), NOT epsilon-patched.
    Returns dict with all requested metrics.
    """
    border_vals = arr2d[border_mask]
    inner_vals  = arr2d[~border_mask]

    b_mean = float(border_vals.mean()) if len(border_vals) else float("nan")
    i_mean = float(inner_vals.mean())  if len(inner_vals)  else float("nan")
    b_sum  = float(border_vals.sum())  if len(border_vals) else float("nan")
    i_sum  = float(inner_vals.sum())   if len(inner_vals)  else float("nan")
    total  = b_sum + i_sum

    # Energy share: fraction of total activation in border (NaN if all-zero)
    b_energy_share = (b_sum / total) if (np.isfinite(total) and total > 0) else float("nan")

    # Ratio: NaN if inner_mean is exactly 0
    if np.isfinite(i_mean) and i_mean != 0.0:
        ratio = b_mean / i_mean
    else:
        ratio = float("nan")   # <— explicit, no silent epsilon

    # Peak location (argmax)
    flat_idx = int(np.argmax(arr2d))
    peak_row = flat_idx // arr2d.shape[1]
    peak_col = flat_idx % arr2d.shape[1]

    return {
        "border_n": int(border_vals.shape[0]),
        "inner_n":  int(inner_vals.shape[0]),
        "border_mean": b_mean,
        "inner_mean":  i_mean,
        "border_sum":  b_sum,
        "inner_sum":   i_sum,
        "border_energy_share": b_energy_share,
        "border_to_inner_ratio": ratio,
        "peak_row": peak_row,
        "peak_col": peak_col,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Load model, config, val dataset
# ════════════════════════════════════════════════════════════════════════════
section("1. MODEL AND DATASET LOADING")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

model = EfficientNetB0Classifier(pretrained=False)
model.load_state_dict(raw_ckpt["model_state_dict"])
model.eval()
model.to(device)

# Verify checkpoint identity
ckpt_epoch = raw_ckpt.get("epoch", "?")
ckpt_hash  = {k: v.sum().item() for k, v in raw_ckpt["model_state_dict"].items()
              if v.dtype == torch.float32}
param_sum  = sum(ckpt_hash.values())
print(f"  Checkpoint: epoch={ckpt_epoch}, param_sum={param_sum:.6f}")

target_layer = model.backbone.conv_head
print(f"  Target layer: backbone.conv_head  {target_layer}")

cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
dm  = PneumoniaDataModule(cfg)
dm.setup("fit")
val_ds = dm.val_dataset
print(f"  Val images: {len(val_ds)}")

# ════════════════════════════════════════════════════════════════════════════
#  Re-run full inference to verify logits unchanged
# ════════════════════════════════════════════════════════════════════════════
section("2. VERIFY LOGITS (full val pass)")
loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
all_logits, all_targets = [], []
with torch.no_grad():
    for imgs, labs in loader:
        logits = model(imgs.to(device)).squeeze(1).cpu().float().numpy()
        all_logits.extend(logits.tolist())
        all_targets.extend(labs.numpy().astype(int).tolist())
all_logits  = np.array(all_logits)
all_targets = np.array(all_targets)
all_probs   = sigmoid_np(all_logits)
all_preds   = (all_probs >= THRESHOLD).astype(int)

def outcome(gt, pred):
    if gt==1 and pred==1: return "TP"
    if gt==0 and pred==0: return "TN"
    if gt==0 and pred==1: return "FP"
    return "FN"
outcomes = [outcome(all_targets[i], all_preds[i]) for i in range(len(all_targets))]

tp_n = sum(1 for o in outcomes if o=="TP")
tn_n = sum(1 for o in outcomes if o=="TN")
fp_n = sum(1 for o in outcomes if o=="FP")
fn_n = sum(1 for o in outcomes if o=="FN")
print(f"  TP={tp_n}  TN={tn_n}  FP={fp_n}  FN={fn_n}  (expected: TP=382, TN=132, FP=2, FN=2)")

# ════════════════════════════════════════════════════════════════════════════
#  Load original example selection from metadata
# ════════════════════════════════════════════════════════════════════════════
section("3. LOAD ORIGINAL SELECTION")
with open(ORIG_META, encoding="utf-8") as f:
    orig_manifest = json.load(f)

selected_indices = [ex["ds_idx"] for ex in orig_manifest["examples"]]
print(f"  Total selected: {len(selected_indices)}")
for ex in orig_manifest["examples"]:
    # Verify original logit matches recomputed
    recomp_prob = float(all_probs[ex["ds_idx"]])
    stored_prob = ex["pneumonia_prob"]
    match = abs(recomp_prob - stored_prob) < 1e-4
    print(f"  ds_idx={ex['ds_idx']:4d}  orig_prob={stored_prob:.6f}  "
          f"recomp_prob={recomp_prob:.6f}  match={'OK' if match else 'MISMATCH'}")

# ════════════════════════════════════════════════════════════════════════════
#  Define masks
# ════════════════════════════════════════════════════════════════════════════
section("4. BORDER MASKS")

native_1cell = make_native_border_mask(7, 7, n_cells=1)
native_2cell = make_native_border_mask(7, 7, n_cells=2)
ups_22px     = make_upsampled_border_mask(margin_px=22, size=224)

print(f"  Native 7x7, 1-cell border mask : {native_1cell.sum()} / 49 cells")
print(f"  Native 7x7, 2-cell border mask : {native_2cell.sum()} / 49 cells")
print(f"  Upsampled 224px, 22px margin   : {ups_22px.sum()} / {224*224} pixels")

# Key mismatch explanation:
# 224 / 7 = 32.0 px per native grid cell.
# int(0.1 * 224) = 22 px margin used in original run.
# 22 < 32 => the 22px margin is SMALLER than one native grid cell width.
# It does NOT align with any native grid boundary.
# Equivalent native coverage: 22/32 = 0.6875 of one cell.
# So the 22px border mask captures a partial outer cell, not a full cell.
native_cell_px = 224 / 7
frac = 22 / native_cell_px
print(f"\n  Native cell width in px: {native_cell_px:.4f}")
print(f"  22px border = {frac:.4f} of one native cell (NOT a full cell)")
print(f"  Nearest full-cell border in px: 1 cell = {native_cell_px:.1f}px, "
      f"2 cells = {2*native_cell_px:.1f}px")

# ════════════════════════════════════════════════════════════════════════════
#  Per-example audit: raw CAM, native grid stats, upsampled stats
# ════════════════════════════════════════════════════════════════════════════
section("5. PER-EXAMPLE AUDIT (raw CAM + native 7x7 + upsampled 22px stats)")

rows = []  # for CSV

for ex in orig_manifest["examples"]:
    ds_idx  = ex["ds_idx"]
    oc      = ex["outcome"]
    gt      = ex["gt_class"]
    prob    = float(all_probs[ds_idx])
    stem    = ex["stem"]

    image_tensor, label = val_ds[ds_idx]
    input_t = image_tensor.unsqueeze(0).to(device)
    model.eval()

    with AuditGradCAM(model, target_layer) as agcam:
        norm_heatmap_224 = agcam.generate(input_t, output_size=IMG_SIZE)

    # ── Raw CAM at native 7×7 ─────────────────────────────────────
    pre  = agcam.raw_cam_pre_relu   # (7,7), may have negatives
    post = agcam.raw_cam_post_relu  # (7,7), >= 0 everywhere (post-ReLU)
    grads = agcam.gradient_maps     # (1280,7,7)
    acts  = agcam.activation_maps   # (1280,7,7)
    wts   = agcam.channel_weights   # (1280,)

    # ── Raw CAM statistics ─────────────────────────────────────────
    pre_min   = float(pre.min())
    pre_max   = float(pre.max())
    pre_mean  = float(pre.mean())
    pre_sum   = float(pre.sum())

    post_min  = float(post.min())
    post_max  = float(post.max())
    post_mean = float(post.mean())
    post_sum  = float(post.sum())
    post_l1   = float(np.abs(post).mean())

    # Gradient magnitude stats (across all 1280 channels and 7x7)
    grad_abs  = np.abs(grads)
    grad_mean = float(grad_abs.mean())
    grad_max  = float(grad_abs.max())

    # Check if all-zero post-ReLU (degenerate CAM)
    is_zero_cam = (post_max < 1e-10)

    # ── Native 7×7 border stats ────────────────────────────────────
    nat1 = border_inner_stats(post, native_1cell)
    nat2 = border_inner_stats(post, native_2cell)

    # ── Upsampled 224px stats (replication of original method) ─────
    # The original computed stats on norm_heatmap_224 (normalized, bicubic)
    ups_border_vals = norm_heatmap_224[ups_22px]
    ups_inner_vals  = norm_heatmap_224[~ups_22px]
    ups_b_mean = float(ups_border_vals.mean())
    ups_i_mean = float(ups_inner_vals.mean())
    # Original code: border_to_inner_ratio = border_mean / max(inner_mean, 1e-8)
    # We replicate but ALSO report the epsilon-patched vs NaN-correct versions
    if ups_i_mean != 0.0:
        ups_ratio_correct = ups_b_mean / ups_i_mean
    else:
        ups_ratio_correct = float("nan")
    ups_ratio_epsilon_patch = ups_b_mean / max(ups_i_mean, 1e-8)

    # Was the epsilon patch triggered?
    eps_triggered = (ups_i_mean == 0.0)

    # ── Stored values from original report ─────────────────────────
    stored_border  = ex.get("border_mean_activation", None)
    stored_inner   = ex.get("inner_mean_activation", None)
    stored_ratio   = ex.get("border_to_inner_ratio", None)

    # ── Reconciliation ─────────────────────────────────────────────
    border_match = (stored_border is None or abs(ups_b_mean - stored_border) < 1e-4)
    inner_match  = (stored_inner  is None or abs(ups_i_mean - stored_inner)  < 1e-4)

    # ── Print row ──────────────────────────────────────────────────
    print(f"\n  [{oc}] idx={ds_idx:4d}  p={prob:.4f}  stem={stem}")
    print(f"    Pre-ReLU  7x7: min={pre_min:+.4f}  max={pre_max:+.4f}  "
          f"mean={pre_mean:+.4f}  sum={pre_sum:+.4f}")
    print(f"    Post-ReLU 7x7: min={post_min:.4f}   max={post_max:.4f}  "
          f"mean={post_mean:.4f}   sum={post_sum:.4f}  L1={post_l1:.4f}  "
          f"all_zero={is_zero_cam}")
    print(f"    Grad magn 7x7: mean_abs={grad_mean:.6f}  max_abs={grad_max:.6f}")
    _r1 = "NaN" if np.isnan(nat1["border_to_inner_ratio"]) else ("%.4f" % nat1["border_to_inner_ratio"])
    _r2 = "NaN" if np.isnan(nat2["border_to_inner_ratio"]) else ("%.4f" % nat2["border_to_inner_ratio"])
    print("    Native 1-cell border: b_mean=%.4f  i_mean=%.4f  ratio=%s"
          % (nat1["border_mean"], nat1["inner_mean"], _r1))
    print("    Native 2-cell border: b_mean=%.4f  i_mean=%.4f  ratio=%s"
          % (nat2["border_mean"], nat2["inner_mean"], _r2))
    _ru = "NaN" if np.isnan(ups_ratio_correct) else f"{ups_ratio_correct:.4f}"
    print(f"    Upsamp 22px: b_mean={ups_b_mean:.4f}  i_mean={ups_i_mean:.4f}  "
          f"ratio_correct={_ru}  ratio_eps={ups_ratio_epsilon_patch:.4f}  eps_triggered={eps_triggered}")
    print(f"    Stored ratio: {stored_ratio}  match_b={border_match}  match_i={inner_match}")
    print(f"    Peak (post-ReLU 7x7): row={nat1['peak_row']}  col={nat1['peak_col']}")

    rows.append({
        "ds_idx":             ds_idx,
        "outcome":            oc,
        "gt_class":           gt,
        "pneumonia_prob":     round(prob, 6),
        # Raw CAM
        "pre_relu_min":       round(pre_min, 6),
        "pre_relu_max":       round(pre_max, 6),
        "pre_relu_mean":      round(pre_mean, 6),
        "pre_relu_sum":       round(pre_sum, 6),
        "post_relu_min":      round(post_min, 6),
        "post_relu_max":      round(post_max, 6),
        "post_relu_mean":     round(post_mean, 6),
        "post_relu_sum":      round(post_sum, 6),
        "post_relu_l1":       round(post_l1, 6),
        "is_zero_cam":        is_zero_cam,
        "grad_abs_mean":      round(grad_mean, 8),
        "grad_abs_max":       round(grad_max, 8),
        # Native 7x7 1-cell border
        "nat1_border_n":      nat1["border_n"],
        "nat1_inner_n":       nat1["inner_n"],
        "nat1_border_mean":   round(nat1["border_mean"], 6),
        "nat1_inner_mean":    round(nat1["inner_mean"], 6),
        "nat1_border_sum":    round(nat1["border_sum"], 6),
        "nat1_inner_sum":     round(nat1["inner_sum"], 6),
        "nat1_energy_share":  (round(nat1["border_energy_share"], 6)
                               if np.isfinite(nat1["border_energy_share"]) else "NaN"),
        "nat1_ratio":         (round(nat1["border_to_inner_ratio"], 6)
                               if np.isfinite(nat1["border_to_inner_ratio"]) else "NaN"),
        "nat1_peak_row":      nat1["peak_row"],
        "nat1_peak_col":      nat1["peak_col"],
        # Native 7x7 2-cell border
        "nat2_border_n":      nat2["border_n"],
        "nat2_inner_n":       nat2["inner_n"],
        "nat2_border_mean":   round(nat2["border_mean"], 6),
        "nat2_inner_mean":    round(nat2["inner_mean"], 6),
        "nat2_border_sum":    round(nat2["border_sum"], 6),
        "nat2_inner_sum":     round(nat2["inner_sum"], 6),
        "nat2_energy_share":  (round(nat2["border_energy_share"], 6)
                               if np.isfinite(nat2["border_energy_share"]) else "NaN"),
        "nat2_ratio":         (round(nat2["border_to_inner_ratio"], 6)
                               if np.isfinite(nat2["border_to_inner_ratio"]) else "NaN"),
        # Upsampled 22px (original report replication + corrected)
        "ups22_border_mean":  round(ups_b_mean, 6),
        "ups22_inner_mean":   round(ups_i_mean, 6),
        "ups22_ratio_correct": ("NaN" if np.isnan(ups_ratio_correct)
                                else round(ups_ratio_correct, 6)),
        "ups22_ratio_eps_patch": round(ups_ratio_epsilon_patch, 6),
        "eps_patch_triggered": eps_triggered,
        # Reconciliation
        "stored_border_mean": stored_border,
        "stored_inner_mean":  stored_inner,
        "stored_ratio":       stored_ratio,
        "border_recompute_match": border_match,
        "inner_recompute_match":  inner_match,
    })

# ════════════════════════════════════════════════════════════════════════════
#  Summary statistics
# ════════════════════════════════════════════════════════════════════════════
section("6. SUMMARY — BORDER DOMINANCE RECONCILIATION")

n_border_ups22_correct   = sum(1 for r in rows
    if r["ups22_ratio_correct"] != "NaN" and r["ups22_ratio_correct"] > 1.0)
n_border_ups22_eps       = sum(1 for r in rows if r["ups22_ratio_eps_patch"] > 1.0)
n_border_nat1            = sum(1 for r in rows
    if r["nat1_ratio"] != "NaN" and r["nat1_ratio"] > 1.0)
n_border_nat2            = sum(1 for r in rows
    if r["nat2_ratio"] != "NaN" and r["nat2_ratio"] > 1.0)
n_zero_cams              = sum(1 for r in rows if r["is_zero_cam"])
n_eps_triggered          = sum(1 for r in rows if r["eps_patch_triggered"])
n_nan_ratio_ups22        = sum(1 for r in rows if r["ups22_ratio_correct"] == "NaN")
n_nan_ratio_nat1         = sum(1 for r in rows if r["nat1_ratio"] == "NaN")

print(f"  Total examples             : {len(rows)}")
print(f"  All-zero post-ReLU CAMs    : {n_zero_cams}")
print(f"  Epsilon patch triggered    : {n_eps_triggered}")
print(f"  Upsampled 22px:")
print(f"    ratio > 1.0 (NaN-correct): {n_border_ups22_correct} / {len(rows) - n_nan_ratio_ups22} finite")
print(f"    ratio > 1.0 (eps-patch)  : {n_border_ups22_eps} / {len(rows)}")
print(f"    undefined (NaN)          : {n_nan_ratio_ups22}")
print(f"  Native 7x7 1-cell border:")
print(f"    ratio > 1.0              : {n_border_nat1} / {len(rows) - n_nan_ratio_nat1} finite")
print(f"    undefined (NaN)          : {n_nan_ratio_nat1}")
print(f"  Native 7x7 2-cell border:")
print(f"    ratio > 1.0              : {n_border_nat2} / {len(rows)}")

print()
print("  Cases where epsilon patch affected result interpretation:")
for r in rows:
    if r["eps_patch_triggered"]:
        print(f"    ds_idx={r['ds_idx']:4d}  outcome={r['outcome']}  "
              f"inner_mean={r['ups22_inner_mean']:.8f}  "
              f"ratio_correct=NaN  ratio_eps_patch={r['ups22_ratio_eps_patch']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
#  Save CSV
# ════════════════════════════════════════════════════════════════════════════
csv_path = AUDIT_DIR / "gradcam_numerical_audit.csv"
if rows:
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
print(f"\n  CSV saved: {csv_path}")

# ════════════════════════════════════════════════════════════════════════════
#  Save JSON (full data)
# ════════════════════════════════════════════════════════════════════════════
json_out = {
    "audit_description": "Experiment 2A Grad-CAM Numerical Audit",
    "checkpoint_epoch_0indexed": int(ckpt_epoch),
    "param_sum_for_identity": param_sum,
    "val_size": len(val_ds),
    "confusion": {"TP": tp_n, "TN": tn_n, "FP": fp_n, "FN": fn_n},
    "native_cell_px": float(native_cell_px),
    "22px_as_fraction_of_cell": float(frac),
    "border_dominance_summary": {
        "upsampled_22px_ratio_gt1_finite_only":   n_border_ups22_correct,
        "upsampled_22px_ratio_gt1_eps_patch":     n_border_ups22_eps,
        "upsampled_22px_undefined_NaN":           n_nan_ratio_ups22,
        "native_7x7_1cell_ratio_gt1":             n_border_nat1,
        "native_7x7_1cell_undefined_NaN":         n_nan_ratio_nat1,
        "native_7x7_2cell_ratio_gt1":             n_border_nat2,
        "zero_cam_cases":                         n_zero_cams,
        "eps_patch_triggered_cases":              n_eps_triggered,
    },
    "examples": rows,
}
json_path = AUDIT_DIR / "gradcam_numerical_audit.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_out, f, indent=2, default=str)
print(f"  JSON saved: {json_path}")

section("AUDIT COMPLETE")
print("  Existing artifacts in experiments/exp02a_gradcam/ were NOT modified.")
print("  New outputs written to experiments/exp02a_gradcam/numerical_audit/")
print("  Test set NOT accessed. No training or fine-tuning performed.")
