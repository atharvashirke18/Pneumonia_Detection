# Experiment 4 — Calibration and MC-dropout uncertainty audit

**Calibration split:** 489 images.
**Fit subset:** 342 images; **audit subset:** 147 images.
**Official test manifest contents and images:** not read.
**Temperature:** 2.860098, fit only on the calibration-fit subset.

## Held-out internal audit subset

| Metric | Raw baseline | Temperature-scaled |
|---|---:|---:|
| NLL | 0.085371 | 0.038282 |
| Brier score | 0.013467 | 0.011120 |
| 10-bin ECE | 0.013537 | 0.011831 |
| Accuracy at 0.5 | 0.9864 | 0.9864 |

## MC-dropout error ranking

50 stochastic dropout passes; calibrated pass probabilities use the fit-subset temperature. Audit errors: 2 / 147.

| Uncertainty score | AUROC for ranking errors |
|---|---:|
| predictive entropy | 0.9862 |
| probability variance | 0.9931 |
| mutual information | 0.9931 |

## Interpretation limits

This is a method-development audit on a single internal calibration partition. MC-dropout scores are approximate model uncertainty, not clinical certainty. The audit partition is not an external cohort; NORMAL patient identities are unavailable. ECE depends on binning and sample size. These results do not establish clinical calibration, generalization, or distribution-free coverage guarantees.

No model weights were changed. No images were used for training. Per-example probabilities and uncertainty values are in `calibration_uncertainty_audit.csv`.
