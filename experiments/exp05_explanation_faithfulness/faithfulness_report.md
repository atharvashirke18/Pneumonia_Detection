# Experiment 5 — Grad-CAM faithfulness audit

**Sample:** all 518 internal-validation images.
**Checkpoint:** frozen baseline; threshold 0.5.
**Native map:** 7 × 7 cells.
**Replacement:** Gaussian-blurred resized RGB image, PIL blur radius 12, then the unchanged evaluation transform.
**Random controls:** 5 fixed-seed random cell rankings per image.
**Official test and calibration manifests/images:** not read.

## Results

| Curve / ground-truth group | Grad-CAM AUC | Random AUC | Paired advantage | Image-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Deletion | 0.8272 | 0.8603 | 0.0331 | [0.0265, 0.0401] |
| Deletion — NORMAL truth (n=134) | 0.3787 | 0.4843 | 0.1056 | [0.0861, 0.1257] |
| Deletion — PNEUMONIA truth (n=384) | 0.9836 | 0.9915 | 0.0079 | [0.0045, 0.0118] |
| Insertion | 0.9006 | 0.8607 | 0.0399 | [0.0315, 0.0486] |
| Insertion — NORMAL truth (n=134) | 0.6330 | 0.4864 | 0.1466 | [0.1221, 0.1686] |
| Insertion — PNEUMONIA truth (n=384) | 0.9940 | 0.9914 | 0.0026 | [0.0005, 0.0051] |

Positive deletion advantage means deleting top Grad-CAM cells lowered the target-class score more than the random control. Positive insertion advantage means restoring those cells raised it more. These are model-faithfulness tests under the stated blur replacement; they do not show that a highlighted region is pneumonia or medically causal.

Class subgroups are grouped by ground-truth label. The native map is only 7 × 7. Image-bootstrap intervals do not model patient clustering, and blurred replacement can itself shift inputs out of distribution. Expert lesion masks are still required for region-localization claims.

Per-image results are in `faithfulness_results.csv`; machine-readable summary is in `faithfulness_summary.json`.
