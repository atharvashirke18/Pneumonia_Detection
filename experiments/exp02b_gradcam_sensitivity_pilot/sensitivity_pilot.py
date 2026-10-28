"""
sensitivity_pilot.py — Experiment 2B: Grad-CAM Border Shortcut-Sensitivity Pilot

Frozen EfficientNetB0 baseline. Read-only. No training, fine-tuning, threshold
changes, or test-set access.

Perturbation design (pre-committed, not tuned to results):
  BORDER_MARGIN_PX = 32  : one native conv_head cell (224/7 = 32 px exactly)
  FILL_VALUE        = 0  : black fill in [0,255] RGB space (dark background, like X-ray)
  CENTRAL_SIDE_PX   = 156: central square matching ≈ border-mask area (see below)
  GAMMA             = 0.85: mild global darkening (preprocessing control)

Area accounting (224×224 input):
  Border mask (32px, full perimeter): 224² − (224−64)² = 50176 − 25600 = 24576 px
  Central square (156×156 = 24336 px): within 1% of border area
  Gamma: applied to all 50176 pixels

Perturbation conditions:
  1. original     : unchanged image (uses exact dataset tensor — no round-trip error)
  2. border_black : outer 32px ring replaced with pixel value 0 (black)
  3. central_black: central 156×156 region replaced with pixel value 0
  4. gamma_dark   : all pixels raised to power 1/gamma (darkening)

For conditions 2–4: the dataset tensor is denormalized back to uint8 PIL (224×224),
the perturbation is applied, then the image is re-normalized with the same ImageNet
parameters. The round-trip introduces ≤0.5/255 ≈ 0.002 per-channel quantization noise
(documented limitation). The "original" condition uses the exact dataset tensor so its
probability matches stored values to floating-point precision.
"""

import csv
import json
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path("d:/Pneumonia_Detection")
CHECKPOINT   = PROJECT_ROOT / "experiments/exp01_efficientnetb0_baseline/best_model.pt"
ORIG_META    = PROJECT_ROOT / "experiments/exp02a_gradcam/gradcam_metadata.json"
OUTPUT_DIR   = PROJECT_ROOT / "experiments/exp02b_gradcam_sensitivity_pilot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PRE-COMMITTED PERTURBATION PARAMETERS ────────────────────────────────────
IMG_SIZE         = (224, 224)
THRESHOLD        = 0.5
BORDER_MARGIN_PX = 32             # = 224/7; one native conv_head grid cell
FILL_VALUE_U8    = 0              # Black fill
CENTRAL_SIDE_PX  = 156            # 156×156 = 24336 px ≈ border area 24576 px (Δ0.98%)
GAMMA_PARAM      = 0.85           # darkening: output = input^(1/0.85)
IMAGENET_MEAN    = (0.485, 0.456, 0.406)
IMAGENET_STD     = (0.229, 0.224, 0.225)

CONDITIONS = ["original", "border_black", "central_black", "gamma_dark"]

# ── Module loading (bypass sklearn DLL) ──────────────────────────────────────
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
datamod_mod = _load("src/data/datamodule.py", "src.data.datamodule")
model_mod   = _load("src/models/model.py", "src.models.model")

load_config              = config_mod.load_config
PneumoniaDataModule      = datamod_mod.PneumoniaDataModule
EfficientNetB0Classifier = model_mod.EfficientNetB0Classifier


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def section(title):
    print(); print("=" * 70); print(f"  {title}"); print("=" * 70)


# ── Image conversion helpers ──────────────────────────────────────────────────

_to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def tensor_to_uint8_pil(tensor_chw: torch.Tensor) -> Image.Image:
    """
    Denormalize a normalized (C, H, W) float tensor back to uint8 RGB PIL.
    Precision: ≤0.5/255 ≈ 0.002 per-channel quantization noise per pixel.
    This is the inverse of ToTensor + Normalize for the documented parameters.
    """
    t = tensor_chw.cpu().clone().float()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def apply_perturbation(pil_224: Image.Image, condition: str) -> Image.Image:
    """
    Apply named perturbation to a 224×224 uint8 PIL image.
    Returns a new image; input is not modified.
    """
    assert pil_224.size == (IMG_SIZE[1], IMG_SIZE[0]), \
        f"Expected 224×224 PIL, got {pil_224.size}"

    if condition == "original":
        return pil_224.copy()

    arr = np.array(pil_224, dtype=np.uint8).copy()

    if condition == "border_black":
        m = BORDER_MARGIN_PX
        arr[:m,  :, :] = FILL_VALUE_U8
        arr[-m:, :, :] = FILL_VALUE_U8
        arr[:, :m,  :] = FILL_VALUE_U8
        arr[:, -m:, :] = FILL_VALUE_U8

    elif condition == "central_black":
        H, W = arr.shape[:2]
        hs = CENTRAL_SIDE_PX // 2
        cy, cx = H // 2, W // 2
        r0 = max(0, cy - hs)
        r1 = min(H, r0 + CENTRAL_SIDE_PX)
        c0 = max(0, cx - hs)
        c1 = min(W, c0 + CENTRAL_SIDE_PX)
        arr[r0:r1, c0:c1, :] = FILL_VALUE_U8

    elif condition == "gamma_dark":
        arr_f = arr.astype(np.float32) / 255.0
        arr_f = np.power(np.clip(arr_f, 0, 1), 1.0 / GAMMA_PARAM)
        arr   = (arr_f * 255.0).round().clip(0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    return Image.fromarray(arr, mode="RGB")


def pil_to_batch_tensor(pil_img: Image.Image, device) -> torch.Tensor:
    """ToTensor + Normalize → (1,3,224,224) on device."""
    return _to_tensor_normalize(pil_img).unsqueeze(0).to(device)


def compute_pixel_area(condition: str) -> dict:
    H, W = IMG_SIZE
    total = H * W
    if condition == "original":
        return {"perturbed_px": 0, "total_px": total}
    elif condition == "border_black":
        inner = (H - 2*BORDER_MARGIN_PX) * (W - 2*BORDER_MARGIN_PX)
        return {"perturbed_px": total - inner, "total_px": total}
    elif condition == "central_black":
        return {"perturbed_px": CENTRAL_SIDE_PX**2, "total_px": total}
    elif condition == "gamma_dark":
        return {"perturbed_px": total, "total_px": total}
    return {}


# ════════════════════════════════════════════════════════════════════════════
section("1. PARAMETER MANIFEST")
# ════════════════════════════════════════════════════════════════════════════
params = {
    "img_size":          list(IMG_SIZE),
    "threshold":         THRESHOLD,
    "border_margin_px":  BORDER_MARGIN_PX,
    "fill_value_u8":     FILL_VALUE_U8,
    "central_side_px":   CENTRAL_SIDE_PX,
    "gamma_param":       GAMMA_PARAM,
    "imagenet_mean":     list(IMAGENET_MEAN),
    "imagenet_std":      list(IMAGENET_STD),
    "conditions":        CONDITIONS,
    "border_area_px":    compute_pixel_area("border_black")["perturbed_px"],
    "central_area_px":   CENTRAL_SIDE_PX**2,
    "area_diff_pct":     round(
        abs(compute_pixel_area("border_black")["perturbed_px"] - CENTRAL_SIDE_PX**2)
        / compute_pixel_area("border_black")["perturbed_px"] * 100, 2),
    "round_trip_note": (
        "Conditions 2-4 denormalize the dataset tensor to uint8 PIL, apply "
        "perturbation, then renormalize. Max per-channel quantization noise "
        "~0.5/255 ≈ 0.002. The 'original' condition uses the exact dataset "
        "tensor to avoid this."
    ),
}
print(json.dumps(params, indent=2))

# ════════════════════════════════════════════════════════════════════════════
section("2. LOAD MODEL AND ORIGINAL METADATA")
# ════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

raw_ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = EfficientNetB0Classifier(pretrained=False)
model.load_state_dict(raw_ckpt["model_state_dict"])
model.eval()
model.to(device)

ckpt_epoch = int(raw_ckpt.get("epoch", -1))
param_sum_before = sum(v.sum().item() for v in model.state_dict().values()
                       if v.dtype == torch.float32)
print(f"  Checkpoint epoch: {ckpt_epoch}")
print(f"  Param sum (before experiment): {param_sum_before:.6f}")

with open(ORIG_META, encoding="utf-8") as f:
    orig_manifest = json.load(f)
examples = orig_manifest["examples"]
print(f"  Examples loaded: {len(examples)}")

cfg    = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
dm     = PneumoniaDataModule(cfg)
dm.setup("fit")
val_ds = dm.val_dataset
print(f"  Val dataset size: {len(val_ds)}")

# ════════════════════════════════════════════════════════════════════════════
section("3. PER-IMAGE PERTURBATION INFERENCE")
# ════════════════════════════════════════════════════════════════════════════
rows = []

for ex in examples:
    ds_idx      = ex["ds_idx"]
    gt          = ex["gt_label"]
    oc          = ex["outcome"]
    stored_prob = ex["pneumonia_prob"]

    # Get exact dataset tensor (same path as training/audit)
    image_tensor, _label = val_ds[ds_idx]   # (3,224,224) normalized
    # Denormalize once to uint8 PIL for perturbation
    orig_pil_224 = tensor_to_uint8_pil(image_tensor)

    row = {
        "ds_idx":      ds_idx,
        "outcome":     oc,
        "gt_label":    gt,
        "gt_class":    ex["gt_class"],
        "stored_prob": stored_prob,
    }

    for cond in CONDITIONS:
        if cond == "original":
            # Use exact dataset tensor — avoids any round-trip quantization
            inp = image_tensor.unsqueeze(0).to(device)
        else:
            perturbed_pil = apply_perturbation(orig_pil_224, cond)
            inp = pil_to_batch_tensor(perturbed_pil, device)

        model.eval()
        with torch.no_grad():
            logit = model(inp)[0, 0].item()
        prob = float(sigmoid_np(np.array(logit)))
        pred = int(prob >= THRESHOLD)
        orig_pred = int(stored_prob >= THRESHOLD)

        row[f"{cond}_prob"] = round(prob, 6)
        row[f"{cond}_pred"] = pred
        row[f"{cond}_flip"] = int(pred != orig_pred)

    # Derived delta columns using consistent naming: <condition>_delta
    for cond in CONDITIONS:
        if cond != "original":
            delta = row[f"{cond}_prob"] - row["original_prob"]
            row[f"{cond}_delta"]     = round(delta, 6)
            row[f"{cond}_abs_delta"] = round(abs(delta), 6)

    row["border_vs_central_abs_delta"] = round(
        row["border_black_abs_delta"] - row["central_black_abs_delta"], 6)

    # Verify original prob matches stored (should be exact since same tensor)
    diff = abs(row["original_prob"] - stored_prob)
    match_ok = diff < 1e-4
    row["recomputed_prob_matches_stored"] = match_ok
    if not match_ok:
        print(f"  WARNING: ds_idx={ds_idx}: orig_prob={row['original_prob']:.6f}  "
              f"stored={stored_prob:.6f}  diff={diff:.2e}")

    print(f"  [{oc}] ds_idx={ds_idx:4d}  "
          f"orig={row['original_prob']:.4f}  "
          f"border={row['border_black_prob']:.4f}(Δ{row['border_black_delta']:+.4f})  "
          f"central={row['central_black_prob']:.4f}(Δ{row['central_black_delta']:+.4f})  "
          f"gamma={row['gamma_dark_prob']:.4f}(Δ{row['gamma_dark_delta']:+.4f})  "
          f"match={'OK' if match_ok else 'MISMATCH'}")
    rows.append(row)

# ════════════════════════════════════════════════════════════════════════════
section("4. AGGREGATE SUMMARY")
# ════════════════════════════════════════════════════════════════════════════

def summarise(rows_list, cond):
    deltas     = [r[f"{cond}_delta"]     for r in rows_list]
    abs_deltas = [r[f"{cond}_abs_delta"] for r in rows_list]
    flips      = [r[f"{cond}_flip"]      for r in rows_list]
    return {
        "n":                len(rows_list),
        "n_flip":           sum(flips),
        "mean_abs_delta":   round(float(np.mean(abs_deltas)), 6),
        "max_abs_delta":    round(float(np.max(abs_deltas)), 6),
        "median_abs_delta": round(float(np.median(abs_deltas)), 6),
        "n_prob_increase":  sum(1 for d in deltas if d > 0.01),
        "n_prob_decrease":  sum(1 for d in deltas if d < -0.01),
    }

summary = {
    "experiment":                    "exp02b_gradcam_sensitivity_pilot",
    "checkpoint_epoch":              ckpt_epoch,
    "param_sum_before_experiment":   param_sum_before,
    "n_examples":                    len(rows),
    "parameters":                    params,
    "by_condition": {
        "border_black":  summarise(rows, "border_black"),
        "central_black": summarise(rows, "central_black"),
        "gamma_dark":    summarise(rows, "gamma_dark"),
    },
    "by_outcome_border_black": {},
}

for oc in ["TP", "TN", "FP", "FN"]:
    oc_rows = [r for r in rows if r["outcome"] == oc]
    if oc_rows:
        summary["by_outcome_border_black"][oc] = {
            "n":              len(oc_rows),
            "n_flip":         sum(r["border_black_flip"] for r in oc_rows),
            "mean_abs_delta": round(float(np.mean(
                [r["border_black_abs_delta"] for r in oc_rows])), 6),
            "examples": [
                {"ds_idx": r["ds_idx"],
                 "orig_prob": r["original_prob"],
                 "border_prob": r["border_black_prob"],
                 "border_delta": r["border_black_delta"],
                 "border_flip": r["border_black_flip"]}
                for r in oc_rows
            ],
        }

for cond_key, s in summary["by_condition"].items():
    print(f"\n  Condition: {cond_key}")
    print(f"    n_flip={s['n_flip']}  mean_abs_Δ={s['mean_abs_delta']:.4f}  "
          f"max_abs_Δ={s['max_abs_delta']:.4f}  "
          f"increase={s['n_prob_increase']}  decrease={s['n_prob_decrease']}")

print("\n  Border condition by outcome:")
for oc, s in summary["by_outcome_border_black"].items():
    print(f"    {oc}: n={s['n']}  n_flip={s['n_flip']}  "
          f"mean_abs_Δ={s['mean_abs_delta']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
section("5. VERIFY CHECKPOINT IMMUTABILITY")
# ════════════════════════════════════════════════════════════════════════════
param_sum_after = sum(v.sum().item() for v in model.state_dict().values()
                      if v.dtype == torch.float32)
params_unchanged = abs(param_sum_after - param_sum_before) < 1e-3
print(f"  Param sum before: {param_sum_before:.6f}")
print(f"  Param sum after : {param_sum_after:.6f}")
print(f"  Parameters unchanged: {params_unchanged}")
assert params_unchanged, "FATAL: Model parameters changed during experiment!"
summary["param_sum_after_experiment"] = param_sum_after
summary["params_unchanged"] = params_unchanged

# ════════════════════════════════════════════════════════════════════════════
section("6. CONTACT SHEET")
# ════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_LABELS = {
    "original":     "Original",
    "border_black": "Border black\n(32px ring)",
    "central_black":"Central black\n(156×156 sq)",
    "gamma_dark":   "Gamma dark\n(γ=0.85)",
}

n_examples = len(rows)
n_cols     = len(CONDITIONS)
fig_w, fig_h = n_cols * 2.8, n_examples * 2.8
fig, axes = plt.subplots(n_examples, n_cols, figsize=(fig_w, fig_h))
if n_examples == 1:
    axes = axes[np.newaxis, :]

for row_idx, r in enumerate(rows):
    ds_idx       = r["ds_idx"]
    image_tensor, _ = val_ds[ds_idx]
    orig_pil_224 = tensor_to_uint8_pil(image_tensor)

    for col_idx, cond in enumerate(CONDITIONS):
        ax  = axes[row_idx, col_idx]
        img = apply_perturbation(orig_pil_224, cond)
        prob = r[f"{cond}_prob"]
        flip = r[f"{cond}_flip"] if cond != "original" else False

        ax.imshow(np.array(img), cmap="gray")
        ax.axis("off")

        if row_idx == 0:
            ax.set_title(COND_LABELS[cond], fontsize=7, fontweight="bold")

        if cond == "original":
            txt = f"{r['outcome']}  p={prob:.3f}"
            color = "white"
        else:
            delta = r[f"{cond}_delta"]
            txt = f"p={prob:.3f} (Δ{delta:+.3f})"
            if flip:
                txt += "\n★FLIP"
            color = "yellow" if flip else "white"

        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=5.5,
                color=color, va="bottom",
                bbox=dict(facecolor="black", alpha=0.55, pad=1.2))

plt.suptitle(
    f"Exp2B Sensitivity Pilot  |  {n_examples} examples  |  "
    f"border=32px  central=156px  γ=0.85",
    fontsize=8, y=1.001)
plt.tight_layout(h_pad=0.3, w_pad=0.3)

contact_path = OUTPUT_DIR / "sensitivity_contact_sheet.png"
plt.savefig(contact_path, dpi=110, bbox_inches="tight")
plt.close()
print(f"  Contact sheet: {contact_path}")

# ════════════════════════════════════════════════════════════════════════════
section("7. SAVE OUTPUTS")
# ════════════════════════════════════════════════════════════════════════════

csv_path = OUTPUT_DIR / "sensitivity_results.csv"
fieldnames = list(rows[0].keys())
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"  CSV : {csv_path}")

json_path = OUTPUT_DIR / "sensitivity_summary.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  JSON: {json_path}")

section("DONE — All outputs saved. Test set not accessed. No training performed.")
print(f"  Existing artifacts unchanged.")
print(f"  Results directory: {OUTPUT_DIR}")
