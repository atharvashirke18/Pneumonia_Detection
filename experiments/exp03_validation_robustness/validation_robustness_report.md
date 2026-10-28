# Experiment 3 — Full validation robustness analysis

**Sample:** 518 existing internal-validation images; every row in `val_manifest.csv`.
**Threshold:** 0.5 (unchanged).
**Checkpoint:** frozen `best_model.pt`; no training or tuning.
**Official test/calibration manifest contents and images:** not read.

## Results

| Condition | N | Prediction flips | Mean absolute probability change | Median | 95th percentile |
|---|---:|---:|---:|---:|---:|

### Single-factor perturbations

| brightness_0.9 | 518 | 2 (0.4%) | 0.005370 | 0.000000 | 0.000546 |
| brightness_1.1 | 518 | 3 (0.6%) | 0.007310 | 0.000000 | 0.000300 |
| contrast_0.9 | 518 | 1 (0.2%) | 0.003373 | 0.000000 | 0.000126 |
| contrast_1.1 | 518 | 3 (0.6%) | 0.004309 | 0.000000 | 0.000088 |
| jpeg_quality_85 | 518 | 2 (0.4%) | 0.006203 | 0.000000 | 0.001388 |

### Photometric control

| gamma_dark | 518 | 1 (0.2%) | 0.002946 | 0.000000 | 0.000163 |

### OOD mask stress tests

| border_black | 518 | 51 (9.8%) | 0.105124 | 0.000001 | 0.978463 |
| central_black | 518 | 132 (25.5%) | 0.253343 | 0.000000 | 0.999999 |

### Baseline internal-validation confusion matrix

TN=132, FP=2, FN=2, TP=382.

## Interpretation and limits

Descriptive results for this internal-validation split only. No confidence intervals or population-level generalization claims are made. Black-mask results are OOD stress responses, not evidence of causal feature reliance. Brightness/contrast magnitudes match the existing training augmentation bounds; JPEG quality is a fixed compression stress condition.

Border and central masking are deliberately separated as large black-mask OOD stress tests. The complete validation set is still from the training dataset source, so this does not establish external or clinical generalization. NORMAL filename-derived patient identity remains unavailable.

Per-image results are in `validation_robustness_results.csv`; machine-readable aggregate data are in `validation_robustness_summary.json`.
