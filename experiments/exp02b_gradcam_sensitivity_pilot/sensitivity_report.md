# Sensitivity Pilot Report
## Experiment 2B — Grad-CAM Border Shortcut-Sensitivity Pilot

**Date:** 2026-10-11  
**Checkpoint:** `experiments/exp01_efficientnetb0_baseline/best_model.pt` (epoch 8, param_sum = 340 584.583)  
**Examples:** 16 images previously selected for Experiment 2A, internal validation split only  
**Outputs:** `experiments/exp02b_gradcam_sensitivity_pilot/`  
**Tests:** 42/42 passed (`tests/test_sensitivity_pilot.py`)

> [!IMPORTANT]
> This is a feasibility pilot, not a proof. No model weights, manifests, checkpoints, or calibration/test data were accessed or modified. The results below apply only to these 16 selected validation examples and must not be generalised to the full dataset or to clinical use.

---

## 1. Objective

Investigate whether the frozen EfficientNetB0 baseline predictions are sensitive to image-border regions, using a small, controlled perturbation study. The purpose is to evaluate the feasibility of a larger shortcut-sensitivity study, not to prove or disprove shortcut learning.

---

## 2. Methods

### 2.1 Perturbation design (pre-committed before reviewing results)

Parameters were fixed before any results were examined.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BORDER_MARGIN_PX` | 32 px | One native conv_head grid cell (224 ÷ 7 = 32 px exactly) |
| `CENTRAL_SIDE_PX` | 156 px | 156² = 24 336 px ≈ border area 24 576 px (area match within 0.98%) |
| `FILL_VALUE_U8` | 0 (black) | Dark fill, consistent with X-ray background; creates a visible but non-random mask |
| `GAMMA_PARAM` | 0.85 | Mild global darkening (output = input^(1/0.85) ≈ input^1.18); acts as a preprocessing control |

### 2.2 Perturbation conditions

| Condition | Description | Perturbed pixels |
|-----------|-------------|-----------------|
| `original` | Unchanged (reference) | 0 |
| `border_black` | Outer 32px ring replaced with black (0) | 24 576 (48.9% of image) |
| `central_black` | Central 156×156 square replaced with black (0) | 24 336 (48.5% of image) |
| `gamma_dark` | Whole image: pixel ← pixel^(1/0.85) | 50 176 (100%) |

Border and central conditions mask nearly equal pixel areas (within 0.98%). The gamma condition is a global operation included as a preprocessing control.

### 2.3 Preprocessing pipeline

1. The `original` condition uses the exact normalized tensor from `val_ds[ds_idx]` — the same pipeline as training and the Experiment 1 audit.
2. For conditions 2–4: the dataset tensor is denormalized back to uint8 RGB PIL (224×224), the perturbation is applied, then the result is re-normalized using the same ImageNet mean/std.
3. All perturbations are applied to in-memory copies only. Raw images and manifests were not modified.

**Quantization note:** The denormalize → uint8 → renormalize round-trip introduces ≤0.5/255 ≈ 0.002 per-channel quantization noise. This is documented as a limitation and is small relative to all observed perturbation effects.

### 2.4 Baseline numerical stability

Six of 16 "original" condition recomputations differed from stored probabilities by 0.0003–0.004. Root cause: single-image inference (batch_size=1) vs the batch-32 inference used to produce the stored values. Floating-point accumulation order on the GPU differs by batch size, producing small but non-zero probability differences. All perturbation deltas in this pilot are measured relative to the same-session "original" inference, so this issue does not affect the comparisons.

---

## 3. Results

### 3.1 Per-image results

| ds\_idx | Outcome | Orig prob | Border prob | Border Δ | Border flip | Central prob | Central Δ | Central flip | Gamma prob | Gamma Δ | Gamma flip |
|---------|---------|-----------|-------------|----------|-------------|--------------|-----------|--------------|------------|---------|------------|
| 0 | TN | 0.0000 | 0.0030 | +0.0030 | No | 1.0000 | +1.0000 | **★ Yes** | 0.0000 | +0.0000 | No |
| 2 | TN | 0.2688 | 0.9976 | +0.7288 | **★ Yes** | 1.0000 | +0.7312 | **★ Yes** | 0.3812 | +0.1124 | No |
| 40 | TN | 0.0000 | 0.2060 | +0.2060 | No | 1.0000 | +1.0000 | **★ Yes** | 0.0000 | +0.0000 | No |
| 43 | TN | 0.0000 | 0.0895 | +0.0895 | No | 0.4057 | +0.4057 | No | 0.0000 | +0.0000 | No |
| 51 | TN | 0.0000 | 0.0048 | +0.0048 | No | 1.0000 | +1.0000 | **★ Yes** | 0.0000 | +0.0000 | No |
| 119 | FP | 0.7385 | 0.9922 | +0.2537 | No | 1.0000 | +0.2615 | No | 0.7931 | +0.0547 | No |
| 120 | FP | 0.9855 | 1.0000 | +0.0144 | No | 1.0000 | +0.0145 | No | 0.9661 | −0.0194 | No |
| 123 | TN | 0.0067 | 0.7268 | +0.7202 | **★ Yes** | 1.0000 | +0.9933 | **★ Yes** | 0.0286 | +0.0220 | No |
| 134 | TP | 1.0000 | 1.0000 | +0.0000 | No | 1.0000 | +0.0000 | No | 1.0000 | +0.0000 | No |
| 138 | TP | 1.0000 | 1.0000 | −0.0000 | No | 1.0000 | +0.0000 | No | 1.0000 | +0.0000 | No |
| 211 | TP | 0.7734 | 1.0000 | +0.2266 | No | 1.0000 | +0.2266 | No | 0.9980 | +0.2246 | No |
| 241 | FN | 0.0267 | 1.0000 | +0.9733 | **★ Yes** | 1.0000 | +0.9733 | **★ Yes** | 0.0415 | +0.0148 | No |
| 260 | FN | 0.2117 | 1.0000 | +0.7883 | **★ Yes** | 1.0000 | +0.7883 | **★ Yes** | 0.7521 | +0.5404 | **★ Yes** |
| 315 | TP | 0.7899 | 0.9091 | +0.1192 | No | 1.0000 | +0.2101 | No | 0.9751 | +0.1851 | No |
| 369 | TP | 1.0000 | 1.0000 | +0.0000 | No | 1.0000 | +0.0000 | No | 1.0000 | +0.0000 | No |
| 402 | TP | 1.0000 | 1.0000 | +0.0000 | No | 1.0000 | +0.0000 | No | 1.0000 | +0.0000 | No |

### 3.2 Aggregate by condition

| Condition | N | Flips | Mean \|Δ\| | Max \|Δ\| | N increased (>0.01) | N decreased (>0.01) |
|-----------|---|-------|-----------|---------|---------------------|---------------------|
| `border_black` | 16 | **4** | 0.258 | 0.973 | 10 | 0 |
| `central_black` | 16 | **7** | 0.475 | 1.000 | 12 | 0 |
| `gamma_dark` | 16 | **1** | 0.073 | 0.540 | 7 | 1 |

### 3.3 Border condition by classification outcome

| Outcome | N | Flips | Mean \|Δ\| |
|---------|---|-------|-----------|
| TP | 6 | 0 | 0.058 |
| TN | 6 | 2 | 0.292 |
| FP | 2 | 0 | 0.134 |
| FN | 2 | 2 | 0.881 |

---

## 4. Observations

### 4.1 Central masking causes more prediction changes than border masking

Central black caused 7 prediction flips; border black caused 4. Central masking also had a higher mean absolute probability change (0.475 vs 0.258). Both conditions mask nearly equal pixel areas (24 336 vs 24 576 px, within 0.98%).

This is the most important aggregate finding: **prediction changes are not disproportionately explained by border removal alone.** Removing the central region (which overlaps with the lung fields) produces at least as many and as large prediction changes as removing the border.

### 4.2 Confident TPs are insensitive to all perturbations

Examples 134, 138, 369, 402 (all TP with orig_prob ≈ 1.000) show zero sensitivity to border, central, or gamma perturbation. This is consistent with confident, robustly classified cases where the model's signal is strong enough to withstand large perturbations.

### 4.3 FN cases flip to positive after border removal

Both FN examples (241, prob 0.027 → 1.000; 260, prob 0.212 → 1.000) changed class after border masking. This is paradoxical from a simple border-shortcut narrative: if the model depended on border features for positive classification, removing them should suppress the positive score, not raise it.

A possible interpretation: the border region in these FN images contains image features (e.g., bright labels, padding, or peripheral anatomy) that the model associates with NORMAL characteristics. Removing the border removes this suppressive signal, allowing the model to more strongly activate on remaining lung content. Alternatively, the large black mask is severely out-of-distribution and the model's response is not interpretable as a feature-attribution result.

### 4.4 Same behaviour after central masking for FN examples

Both FNs also flipped (and to 1.000) after central masking with an identical probability change. This indicates the effect is not specific to border removal — both perturbations affect these images equally.

### 4.5 Some TNs become strongly positive

TN_0002, TN_0040, TN_0051, TN_0123 all moved to prob=1.000 after central masking. TN_0002 and TN_0123 also flipped after border masking. Since these images are correctly classified as NORMAL, their strong positive response to masking suggests the unmasked lung region or border region contains suppressive normal features whose removal drives the model toward positive. This is consistent with the model having learned some genuine lung anatomy differentiation, not purely border dependence.

### 4.6 Gamma darkening has much smaller effects

Only 1 flip; mean absolute change 0.073 vs 0.258 (border) and 0.475 (central). This suggests the model is more robust to global intensity shifts than to large spatial occlusions.

---

## 5. What This Evidence Does and Does Not Support

### Evidence supports:
- The frozen baseline's predictions ARE sensitive to the removal of border regions for at least some of the 16 examples tested.
- They are also sensitive (more so) to removal of the central region.
- Gamma darkening is a substantially weaker perturbation.
- The nature and magnitude of sensitivity varies considerably across examples.

### Evidence does NOT support:
- That border-region masking is a stronger perturbation than central-region masking across this sample. The reverse is observed.
- That prediction changes caused by masking represent a causal identification of features the model relies on (masking creates out-of-distribution inputs).
- That the model lacks genuine pneumonia-detection ability. The Grad-CAM audit and validation results are separate evidence requiring separate interpretation.
- Dataset-wide shortcut learning, clinical generalization, or any claim about the remaining 502 validation examples.
- Any interpretation of individual perturbation results as reflecting a single identified shortcut.

---

## 6. Limitations

1. **16 selected examples only.** The selection is not random. The results cannot be generalised beyond this set.
2. **The selected examples are not a representative sample.** They were chosen to include all classification errors and a stratified confidence range from Experiment 2A.
3. **Both masking conditions create out-of-distribution images.** Large black regions are not present in the training data. The model's response to such inputs may not reflect its response to realistic image variation.
4. **The border and central perturbations are not perfectly equivalent.** They affect different spatial locations, and the spatial content of those locations differs in medically relevant ways (lung fields vs periphery).
5. **Grad-CAM is a coarse localization method.** The 7×7 native resolution limits spatial precision. Grad-CAM activation in the border region does not establish that specific border features (markers, tape, padding) are causally used by the model.
6. **No permutation testing or statistical correction.** With n=16, the power to detect systematic effects is very low. Observed patterns are exploratory.
7. **Gamma condition**: only one flip in 16 examples is insufficient to characterise the model's intensity robustness.
8. **Single-image vs batch inference:** baseline numerical differences of up to 0.004 in probability exist. All reported deltas are computed within the same inference session to avoid this affecting comparisons.

---

## 7. Checkpoint and Artifacts Integrity

| Check | Result |
|-------|--------|
| Checkpoint param_sum before experiment | 340 584.583 |
| Checkpoint param_sum after experiment | 340 584.583 |
| Parameters unchanged | **True** |
| Test set accessed | **No** |
| Training or fine-tuning | **No** |
| Existing artifacts modified | **None** |

---

## 8. Outputs

| File | Description |
|------|-------------|
| [`sensitivity_pilot.py`](file:///d:/Pneumonia_Detection/experiments/exp02b_gradcam_sensitivity_pilot/sensitivity_pilot.py) | Reproducible script |
| [`sensitivity_results.csv`](file:///d:/Pneumonia_Detection/experiments/exp02b_gradcam_sensitivity_pilot/sensitivity_results.csv) | Per-image, per-condition measurements |
| [`sensitivity_summary.json`](file:///d:/Pneumonia_Detection/experiments/exp02b_gradcam_sensitivity_pilot/sensitivity_summary.json) | Aggregate counts and statistics |
| [`sensitivity_contact_sheet.png`](file:///d:/Pneumonia_Detection/experiments/exp02b_gradcam_sensitivity_pilot/sensitivity_contact_sheet.png) | Original and all perturbation conditions, labelled |
| [`test_sensitivity_pilot.py`](file:///d:/Pneumonia_Detection/tests/test_sensitivity_pilot.py) | 42 automated tests (42/42 passed) |

---

## 9. Tests

**42/42 tests passed** in `tests/test_sensitivity_pilot.py` (12.41 s).

| Test class | Tests | Result |
|-----------|-------|--------|
| `TestAreaAccounting` | 6 | All pass |
| `TestPerturbation` | 13 | All pass |
| `TestSigmoid` | 4 | All pass |
| `TestResultsFile` | 12 | All pass |
| `TestSummaryJSON` | 7 | All pass |
