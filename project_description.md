# Explainable Pneumonia Detection from Chest X-Rays: A Quantitative Grad-CAM Audit and Spatial Sensitivity Analysis

## 1. Research Overview

This project investigates **whether the activation patterns of a high-performing chest X-ray classifier (EfficientNetB0) are spatially consistent with clinically expected localization**, and whether those patterns are genuinely anchored in lung parenchyma or confounded by peripheral image features.

The work is intentionally narrow and verifiable: rather than proposing a new model or claiming clinical deployment, it performs a rigorous **post-hoc audit** of a frozen, fully-trained classifier using two complementary analytical approaches — a native-resolution Grad-CAM spatial decomposition (Exp 2A) and a controlled spatial perturbation pilot (Exp 2B).

The central scientific question is:

> *Does the Grad-CAM activation of a near-perfect pneumonia classifier concentrate in border/peripheral regions or in central lung fields, and does this spatial bias correspond to meaningful predictive dependence?*

---

## 2. Motivation and Research Gap

### 2.1 The Shortcut Learning Problem in Medical Imaging

Deep learning classifiers trained on benchmark medical datasets often achieve near-perfect accuracy, yet post-hoc analysis frequently reveals that they rely on spurious correlations rather than clinically meaningful features. In chest X-ray datasets, known confounds include:

- **Radiographic labels and text markers** burned into image borders
- **Positioning artifacts** and film edges in the peripheral region
- **Acquisition differences** between NORMAL and PNEUMONIA patients (e.g., portable vs. standing views)
- **Dataset composition biases** where patient demographics, scanner types, or institutional protocols correlate with class labels

Prior work (Zech et al., 2018; Pooch et al., 2020; DeGrave et al., 2021) has demonstrated that classifiers can learn these peripheral shortcuts while achieving high AUC. The challenge is that standard saliency maps such as Grad-CAM may themselves highlight these peripheral regions, creating a misleading impression of anatomical localization.

### 2.2 The Gap This Work Fills

Existing studies typically address shortcut learning by:
1. Training on multiple institutions and reporting performance drops (dataset shift)
2. Visually inspecting heatmaps without quantification
3. Using proxy metrics that conflate upsampling artifacts with native model behavior

This work addresses these gaps through three specific methodological contributions:

1. **Native-resolution audit at the correct grid granularity** — all Grad-CAM statistics are computed on the raw 7x7 activation grid, avoiding artifacts introduced by bicubic upsampling. Border/inner decomposition uses exactly 1 native cell = 32 px, corresponding to the exact receptive field unit.

2. **Mathematically verified spatial statistics** — border and inner activation means are recomputed from scratch in a separate audit script and verified against stored values to machine-precision tolerance. Undefined ratios (zero inner activation) are explicitly flagged rather than replaced with epsilon-inflated values.

3. **Controlled area-matched perturbation study** — the sensitivity pilot uses perturbation regions with areas matched within 0.98% (border: 24,576 px; central: 24,336 px), enabling a direct apples-to-apples comparison of spatial sensitivity without confounding area differences.

---

## 3. Research Design and Methodology

### 3.1 Dataset and Model Architecture

**Dataset:** Kaggle Chest X-Ray Images (Pneumonia) — Guangzhou Women and Children's Medical Center
**Split used:** Internal validation split (n=518), stratified to exclude patient-token overlap between training and validation partitions

**Architecture:** EfficientNetB0 (ImageNet pretrained) with a custom binary classification head:
- Backbone: EfficientNetB0 feature extractor
- Classifier: Dropout(0.2) then Linear(1280, 1)
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=14)
- Input: 224x224 RGB, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Training: Early stopping on validation loss, batch size 32

**Model immutability:** Parameter checksum verification (sum of absolute values of all parameters) is run before and after every experiment. No weight mutation occurred across any experiment.

### 3.2 Grad-CAM Implementation

The Grad-CAM implementation taps `backbone.conv_head` (the final 1x1 convolutional layer before global average pooling), producing a native **7x7 activation map**:

1. Forward pass computes logit = f(x) with gradient hooks registered on conv_head
2. Backward pass propagates gradient of logit w.r.t. conv_head activations
3. Channel weights: alpha_k = (1/49) * sum over (i,j) of d(logit)/d(A^k_{i,j})
4. Raw CAM: L = ReLU( sum over k of alpha_k * A^k ) — ReLU applied **before** upsampling
5. Min-max normalization to [0, 1] at native 7x7 resolution
6. Bicubic upsampling to 224x224 for visualization

The critical design choice is ReLU application **prior** to upsampling. Applying ReLU post-upsampling would allow negative activations to partially cancel positive regions during interpolation, producing misleading spatial distributions. This choice is verified by a dedicated unit test.

### 3.3 Exp 2A: Native-Resolution Spatial Audit

**Selection of 16 reference cases:** From the 518 validation images, 16 were selected to cover:
- All classification errors: both False Positives (n=2) and False Negatives (n=2)
- Stratified True Positives (n=6): 4 high-confidence (P>=0.99) and 2 moderate-confidence (P~0.78-0.79)
- Stratified True Negatives (n=6): covering a range of pneumonia probabilities (0.000-0.267)

**Spatial decomposition on the native 7x7 grid:**
- **Width-1 border mask (W1):** 24 cells (the outermost ring)
- **Inner mask (W1 inner):** 25 cells (the 5x5 interior)
- **Width-2 border mask (W2):** 40 cells (two outermost rings)
- **Inner mask (W2):** 9 cells (the 3x3 centre)

Border-to-inner ratio is defined as border_mean / inner_mean. For cases with zero inner activation, the ratio is reported as **undefined (NaN)** — not epsilon-patched, not set to infinity, not excluded from analysis.

**Upsampled comparison (22 px, approximately 2/3 of one native cell):** A 22px upsampled border region is also measured as a supplementary analysis. This corresponds to a commonly used margin that does NOT align with native cell boundaries, and is deliberately included to show how margin choice affects the ratio.

### 3.4 Exp 2B: Controlled Spatial Perturbation Pilot

Parameters were **pre-committed before any results were examined** (registered pre-analysis, recorded in sensitivity_pilot.py header). Perturbation design:

| Condition | Description | Pixels affected | Spatial region |
|-----------|-------------|-----------------|----------------|
| original | Unchanged reference | 0 | — |
| border_black | Outer 32px ring set to black (0) | 24,576 (48.9%) | Peripheral |
| central_black | Central 156x156 set to black (0) | 24,336 (48.5%) | Central (lung field) |
| gamma_dark | Global gamma=0.85 darkening | 50,176 (100%) | Global |

The 32px border width corresponds exactly to **one native conv_head grid cell** (224 / 7 = 32). The 156px central side was chosen such that 156 squared = 24,336, approximately equal to 24,576 (within 0.98%), enabling area-controlled spatial comparison.

**Preprocessing pipeline integrity:** Conditions 2-4 denormalize from ImageNet-normalized tensor to uint8 PIL, apply perturbation, then re-normalize. Round-trip quantization noise is bounded at <=0.5/255 = 0.002. All delta-P values are computed within the same inference session to avoid batch-size floating-point order effects (documented separately: 0.0003-0.004 inter-session variance for single-image vs. batch-32 inference).

---

## 4. Results

### 4.1 Experiment 1 — Baseline Classification Performance

**Best checkpoint:** Epoch 8 (selected by minimum validation loss)

| Metric | Training | Validation |
|--------|----------|------------|
| ROC-AUC | 0.9996 | **0.9997** |
| Sensitivity | 99.45% | **99.48%** |
| Specificity | 97.76% | **98.51%** |
| F1 Score | 0.9934 | **0.9948** |
| Loss (BCE) | 0.0310 | **0.0220** |

**Full confusion matrix on validation split (n=518):**

|  | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **Actual NORMAL** (n=134) | 132 (TN) | 2 (FP) |
| **Actual PNEUMONIA** (n=384) | 2 (FN) | 382 (TP) |

**Training trajectory:** The model converged rapidly, achieving ROC-AUC >0.997 by Epoch 1 and maintaining validation AUC >0.999 from Epoch 3 onward. Cosine annealing with a 14-epoch period produced smooth, monotonically improving validation loss from Epoch 0 (0.6417) to Epoch 8 (0.0220).

### 4.2 Experiment 2A — Grad-CAM Native-Resolution Spatial Audit

#### 4.2.1 Border-to-Inner Activation Ratios (16 Cases, Width-1 Native Mask)

*All values computed on the raw (pre-normalized) native 7x7 Grad-CAM grid. Border: 24 cells, Inner: 25 cells.*

| Case | Outcome | P(PNEUMONIA) | Border Mean | Inner Mean | Ratio (B/I) |
|------|---------|-------------|-------------|------------|-------------|
| TN_0000 | TN | 0.000 | 0.2861 | 0.0000 | **Undefined** |
| TN_0002 | TN | 0.267 | 0.4859 | 0.0017 | 286.3 |
| TN_0040 | TN | 0.000 | 0.1162 | 0.0000 | **Undefined** |
| TN_0043 | TN | 0.000 | 0.1002 | 0.0000 | **Undefined** |
| TN_0051 | TN | 0.000 | 0.2138 | 0.0000 | **Undefined** |
| FP_0119 | FP | 0.742 | 0.4995 | 0.0169 | 29.5 |
| FP_0120 | FP | 0.985 | 0.2368 | 0.0146 | 16.2 |
| TN_0123 | TN | 0.007 | 0.2227 | 0.0037 | 60.2 |
| TP_0134 | TP | 1.000 | 0.8964 | 0.3189 | 2.81 |
| TP_0138 | TP | 1.000 | 0.7942 | 0.1848 | 4.30 |
| TP_0211 | TP | 0.777 | 0.3174 | 0.0398 | 7.98 |
| FN_0241 | FN | 0.027 | 0.1051 | 0.0022 | 47.9 |
| FN_0260 | FN | 0.212 | 0.2243 | 0.0434 | 5.18 |
| TP_0315 | TP | 0.791 | 0.1026 | 0.0129 | 7.93 |
| TP_0369 | TP | 1.000 | 0.4239 | 0.3835 | **1.11** |
| TP_0402 | TP | 1.000 | 1.0778 | 0.1834 | 5.88 |

#### 4.2.2 Key Findings

**Finding 1 — Universal border dominance at upsampled resolution:**
At the 22px upsampled margin (the reported margin in the stored metadata), **all 16 cases** (16/16) show a border-to-inner ratio > 1.0. This persists across all four outcome categories (TP, TN, FP, FN).

**Finding 2 — Strong native border dominance in normal-pattern images:**
At native 7x7 resolution with the W1 mask, **12/16** cases show finite border-to-inner ratio > 1, with **4 cases showing zero inner activation** (ratio undefined). The 4 undefined cases are all True Negatives with pneumonia probability = 0.000. Their entire Grad-CAM energy is confined to border cells.

**Finding 3 — True positives show more distributed activation:**
Confident True Positives (TP_0134, TP_0138, TP_0369, TP_0402) have substantially higher inner activation than other cases. TP_0369 has the most balanced distribution (ratio = 1.11), with its activation peak located at the interior cell (row 5, col 5) rather than a corner. This is the only case where the native peak cell is not on the border.

**Finding 4 — False negatives show strong border dominance despite PNEUMONIA ground truth:**
FN_0241 (P=0.027, true PNEUMONIA) has 100% of its energy in border cells at W2 resolution. FN_0260 (P=0.212) shows moderate but still dominant border activation. The model's failure to classify these as PNEUMONIA may be partially attributable to weak or absent central activation.

**Finding 5 — Upsampling method substantially affects ratio estimates:**
Nearest-neighbour upsampling produces systematically higher border-to-inner ratios than bicubic (e.g., TN_0000: NaN vs. 2.45; TN_0002: NaN vs. 2.96). This is because bicubic interpolation smears border activation into adjacent interior pixels, inflating the measured inner mean. Native-resolution analysis avoids this artifact entirely.

**Finding 6 — Audit replication is exact to machine precision:**
For all 16 cases, recomputed border and inner means match stored values to float32 precision (border_recompute_match: true, inner_recompute_match: true for all 16 entries). Parameter immutability is confirmed: max_parameter_abs_diff = 0.0, max_logit_abs_diff = 0.0.

#### 4.2.3 Pre-ReLU vs. Post-ReLU Statistics

| Statistic | Pre-ReLU (Linear CAM) | Post-ReLU (Grad-CAM) |
|-----------|----------------------|---------------------|
| Mean (ds_0 / TN) | -0.3065 | +0.1401 |
| Sum (ds_0 / TN) | -15.02 | +6.87 |
| Mean (ds_134 / TP) | +0.5799 | +0.6017 |
| Sum (ds_134 / TP) | +28.42 | +29.49 |

ReLU removes negative activations that would otherwise suppress positive regions. The mean pre-ReLU value is negative for NORMAL cases (model is suppressing PNEUMONIA class features), while True Positives show a positive mean even before ReLU. This validates the design choice of pre-upsampling ReLU.

### 4.3 Experiment 2B — Spatial Perturbation Sensitivity Pilot

#### 4.3.1 Aggregate Results (n=16)

| Condition | Prediction Flips | Mean |delta-P| | Max |delta-P| | Median |delta-P| |
|-----------|-----------------|----------|-----------|-----------| 
| border_black | **4 / 16** | 0.258 | 0.973 | 0.104 |
| central_black | **7 / 16** | 0.475 | 1.000 | 0.334 |
| gamma_dark | **1 / 16** | 0.073 | 0.540 | 0.007 |

#### 4.3.2 Border Condition by Outcome Category

| Outcome | N | Flips | Mean |delta-P| | Key observation |
|---------|---|-------|----------|-----------------| 
| TP | 6 | 0 | 0.058 | High-confidence TPs are insensitive to border removal |
| TN | 6 | 2 | 0.292 | TN cases show more variable border sensitivity |
| FP | 2 | 0 | 0.134 | FPs become more confident, not corrected |
| FN | 2 | 2 | 0.881 | FNs flip to positive — paradoxical finding |

#### 4.3.3 Per-Image Summary

| ds_idx | Outcome | Orig P | Border P | Border delta | Flip | Central P | Central delta | Flip | Gamma P | Gamma delta | Flip |
|--------|---------|--------|----------|----------|------|-----------|-----------|------|---------|---------|------|
| 0 | TN | 0.000 | 0.003 | +0.003 | — | 1.000 | +1.000 | FLIP | 0.000 | +0.000 | — |
| 2 | TN | 0.269 | 0.998 | +0.729 | FLIP | 1.000 | +0.731 | FLIP | 0.381 | +0.112 | — |
| 40 | TN | 0.000 | 0.206 | +0.206 | — | 1.000 | +1.000 | FLIP | 0.000 | +0.000 | — |
| 43 | TN | 0.000 | 0.090 | +0.090 | — | 0.406 | +0.406 | — | 0.000 | +0.000 | — |
| 51 | TN | 0.000 | 0.005 | +0.005 | — | 1.000 | +1.000 | FLIP | 0.000 | +0.000 | — |
| 119 | FP | 0.739 | 0.992 | +0.254 | — | 1.000 | +0.262 | — | 0.793 | +0.055 | — |
| 120 | FP | 0.986 | 1.000 | +0.014 | — | 1.000 | +0.015 | — | 0.966 | -0.019 | — |
| 123 | TN | 0.007 | 0.727 | +0.720 | FLIP | 1.000 | +0.993 | FLIP | 0.029 | +0.022 | — |
| 134 | TP | 1.000 | 1.000 | 0.000 | — | 1.000 | 0.000 | — | 1.000 | 0.000 | — |
| 138 | TP | 1.000 | 1.000 | 0.000 | — | 1.000 | 0.000 | — | 1.000 | 0.000 | — |
| 211 | TP | 0.773 | 1.000 | +0.227 | — | 1.000 | +0.227 | — | 0.998 | +0.225 | — |
| 241 | FN | 0.027 | 1.000 | +0.973 | FLIP | 1.000 | +0.973 | FLIP | 0.042 | +0.015 | — |
| 260 | FN | 0.212 | 1.000 | +0.788 | FLIP | 1.000 | +0.788 | FLIP | 0.752 | +0.540 | FLIP |
| 315 | TP | 0.791 | 0.909 | +0.119 | — | 1.000 | +0.210 | — | 0.975 | +0.185 | — |
| 369 | TP | 1.000 | 1.000 | 0.000 | — | 1.000 | 0.000 | — | 1.000 | 0.000 | — |
| 402 | TP | 1.000 | 1.000 | 0.000 | — | 1.000 | 0.000 | — | 1.000 | 0.000 | — |

#### 4.3.4 Key Findings

**Central removal causes more disruption than border removal:**
With equal pixel areas masked (within 0.98%), central blackout causes 7 flips vs. 4 for border blackout, and has 1.84x higher mean |delta-P| (0.475 vs. 0.258). This is the principal finding of Exp 2B and is **inconsistent with a simple border-shortcut hypothesis** — if the model relied primarily on border features for prediction, border removal should produce greater disruption than central removal.

**Paradoxical FN behaviour:**
Both False Negative cases (241, 260) flip from NORMAL prediction to PNEUMONIA after border removal. This is counter-intuitive under a shortcut narrative: removing border regions *increases* the PNEUMONIA score for images the model initially classified as NORMAL. A plausible interpretation is that the border region in these images contains suppressive features (peripheral anatomy, labels, or film edges associated with normal acquisitions) that when removed, allow underlying lung content to dominate.

**Confident TPs are perturbation-resistant:**
Cases 134, 138, 369, 402 (all P=1.000) show zero change under any perturbation including complete central blackout. This is consistent with strong, distributed activation across the network's representational space.

**Gamma darkening is a weak perturbation:**
Only 1 flip out of 16, with median |delta-P| = 0.007. The model appears substantially more sensitive to spatial occlusion than to global intensity scaling. This suggests the model does not rely heavily on absolute pixel intensity for classification.

**Directionality:** No case showed *decreased* probability after border removal; 10/16 cases increased. After central removal, 12/16 increased. This systematic upward shift after occlusion is itself informative and requires further investigation in future work.

---

## 5. What Makes This Work Distinctive

### 5.1 vs. Standard Grad-CAM Papers

Most papers using Grad-CAM for medical imaging either:
- Display visualizations without quantification
- Use upsampled heatmaps with arbitrary pixel margins for analysis
- Conflate visualization quality with explanation faithfulness

This work departs by auditing at **native 7x7 resolution** — the actual scale at which the model operates — rather than at an arbitrary upsampled scale. The 32px margin is derived from the exact receptive field cell size (224/7=32), not chosen post-hoc.

### 5.2 vs. Shortcut Learning Studies

Most shortcut learning studies use domain shift (multiple hospital datasets) to reveal shortcuts. This requires data from multiple institutions. This work uses a single-site dataset and instead probes sensitivity via controlled spatial perturbation — a more accessible methodology with traceable experimental design.

### 5.3 vs. Standard Perturbation Studies

Unlike random pixel perturbation (occlusion sensitivity) or gradient-based deletion curves (faithfulness audits), this study uses **anatomically motivated perturbation regions**:
- Border region = peripheral anatomy outside the lung fields (where known artifacts concentrate)
- Central region = lung parenchyma (clinically relevant region)
- Area-matched design enables direct comparison without confounding pixel count

### 5.4 Methodological Rigour

| Property | This Work |
|----------|-----------|
| Pre-committed perturbation parameters | Yes — before results examined |
| Parameter immutability verification | Yes — checksum checked before/after every experiment |
| Undefined-ratio handling | NaN reported, not epsilon-patched |
| Test set isolation | Never accessed |
| Quantization noise documentation | <=0.002, bounded analytically |
| Automated test suite | 42/42 tests pass |
| Audit replication to machine precision | All 16 cases: exact match |
| Open reproducible codebase | Full Python stack, deterministic |

---

## 6. Limitations

1. **Pilot scale — n=16.** The 16 cases are a selected, non-random subset. Results must not be generalised to the full 518-image validation set or beyond.

2. **Out-of-distribution perturbations.** Large black masks (border or central) do not correspond to any image type in the training distribution. The model's response cannot be interpreted as a principled feature attribution.

3. **Grad-CAM coarseness.** The 7x7 native resolution is approximately 32x32 pixels per cell. Spatial localization claims are limited to this granularity.

4. **Single-site dataset.** All images are from a single institution (Guangzhou Women and Children's Medical Center). Findings may not generalise across imaging protocols, equipment, or patient demographics.

5. **No permutation testing.** With n=16, statistical power is insufficient to characterise effect significance. All findings are descriptive/exploratory.

6. **Binary threshold sensitivity.** Results are reported at fixed threshold 0.5. Using a calibrated threshold may change flip counts.

7. **Single model.** One EfficientNetB0 checkpoint is analysed. Findings may not transfer to other architectures (ResNet, ViT, DenseNet).

---

## 7. Research Contributions Summary

**Contribution 1 — A rigorous native-resolution Grad-CAM audit framework**
The audit decomposes Grad-CAM activation at the correct native scale (7x7), with exact border/inner masks derived from receptive field geometry, mathematically verified replication, and proper handling of degenerate cases (zero inner activation).

**Contribution 2 — Empirical evidence against a simple border-shortcut hypothesis**
Across the 16 audited cases, central occlusion (lung field) causes more prediction disruption than border occlusion of equal area. This does not eliminate shortcut learning concerns but provides quantitative evidence that the classification signal is not exclusively border-dependent.

**Contribution 3 — Paradoxical FN-to-positive flip pattern**
Both false negatives in the pilot increase to certainty after border removal — including after central removal. This anomalous result, reproducible and numerically verified, suggests suppressive peripheral features in misclassified cases and motivates deeper investigation of failure mode anatomy.

**Contribution 4 — Methodological transparency framework**
Pre-committed experimental parameters, parameter-sum immutability checks, exact audit replication, quantization noise characterization, and an automated 42-test suite collectively establish a template for reproducible explainability audits in medical AI.

---
