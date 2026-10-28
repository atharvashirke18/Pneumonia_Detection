# Limitations follow-up — status and evidence

This follow-up records the approved incremental work on the five limitation
areas. It does not replace the original baseline or pilot reports. The frozen
checkpoint, original manifests, and prior experiment outputs remain
unchanged.

## Priority status

| Priority | Area | Status |
|---|---|---|
| 1 | Patient-independent evaluation | Partially addressed by metadata audit; a fully patient-independent evaluation is not supportable for all classes using available IDs. |
| 2 | Sensitivity pilot size and masking OOD effects | Expanded to every one of 518 internal-validation images; masking remains a separately labeled OOD stress test. |
| 3 | Grad-CAM resolution and interpretation | UI displays native and upsampled maps; a full-validation deletion/insertion faithfulness audit now compares Grad-CAM ranking with random controls. Expert-mask localization remains unavailable. |
| 4 | External/clinical generalization | Not addressable with the available data; no external cohort was supplied, and the official test split remains untouched. |
| 5 | Calibration and uncertainty | Completed a bounded, calibration-split-only temperature-scaling and MC-dropout method audit. This does not establish clinical calibration or reliable clinical uncertainty. |

## 1. Patient identity and split feasibility

The audit read metadata from `train_manifest.csv`, `val_manifest.csv`,
`calibration_manifest.csv`, and the already-existing `split_summary.json`.
It did not open images or read the official test manifest contents.

- Train has 4,183 images: 1,072 NORMAL images with no filename-derived
  patient token, and 3,111 PNEUMONIA images from 1,308 distinct tokens.
- Validation has 518 images: 134 NORMAL images without tokens and 384
  PNEUMONIA images from 163 tokens.
- Calibration has 489 images: 134 NORMAL images without tokens and 355
  PNEUMONIA images from 164 tokens.
- The existing split summary reports zero pneumonia-token overlap among
  train/validation/calibration and 135 pneumonia tokens shared between
  train and the official test partition.

The current internal pneumonia validation/calibration comparisons are
token-disjoint from training, but NORMAL patient identity is unavailable. A
fully patient-independent evaluation across both classes cannot be claimed.
The official test split was not accessed in this follow-up; the previously
recorded overlap means it cannot substantiate a patient-independent claim
without a separately designed evaluation. Obtaining an independent cohort
with reliable patient IDs is required to address this limitation.

## 2. Full-validation robustness analysis

Experiment 3 is in `experiments/exp03_validation_robustness/`. It evaluated
the complete 518-image internal-validation manifest with the frozen
checkpoint and the unchanged 0.5 threshold. Conditions were pre-specified;
no model tuning or threshold selection occurred.

| Condition | Flips / 518 | Mean absolute probability change |
|---|---:|---:|
| Brightness factor 0.9 | 2 | 0.005370 |
| Brightness factor 1.1 | 3 | 0.007310 |
| Contrast factor 0.9 | 1 | 0.003373 |
| Contrast factor 1.1 | 3 | 0.004309 |
| JPEG quality 85 | 2 | 0.006203 |
| Gamma darkening (existing control) | 1 | 0.002946 |
| Border black mask | 51 | 0.105124 |
| Central black mask | 132 | 0.253343 |

Brightness/contrast factors match the magnitude of existing training
augmentations. JPEG quality is a fixed compression stress condition.
Border/central masks are reported separately because large black masks can
create out-of-distribution inputs. These are descriptive measurements on one
same-source internal-validation split; no population-level confidence
interval or external robustness claim is made.

The recorded baseline confusion counts remained TN=132, FP=2, FN=2, TP=382.
The model state hash was identical before and after this analysis.

## 3. Grad-CAM spatial-resolution display

The UI and `src.explainability.gradcam.GradCAM.generate_with_native()` now provide the native
map and upsampled map from the same gradient pass. The UI shows a nearest-
neighbor enlarged rendering of the native 7 × 7 grid next to the bicubically
upsampled heatmap and overlay. Existing `generate()` behavior remains
unchanged and is tested for exact parity with the upsampled output from the
new method.

This improves transparency; it does not increase model spatial resolution,
prove causal feature use, establish localization, or turn Grad-CAM into a
medical segmentation.

Experiment 5 adds a faithfulness audit on all 518 internal-validation images.
Using each image's baseline-predicted class, the evaluation progressively
replaced/restored native 7 × 7 cells with a Gaussian-blurred input and compared
the resulting target-class probability curves with five fixed-seed random
cell rankings. Overall, Grad-CAM's deletion AUC was 0.8272 versus 0.8603 for
random ranking (paired advantage 0.0331; image-bootstrap 95% CI 0.0265–0.0401).
Insertion AUC was 0.9006 versus 0.8607 (advantage 0.0399; CI 0.0315–0.0486).
The signal was notably smaller for PNEUMONIA ground-truth examples: deletion
advantage 0.0079 and insertion advantage 0.0026, compared with 0.1056 and
0.1466 for NORMAL examples. These are image-level intervals that do not
account for patient clustering, and blur replacement may create distribution
shift. The results measure sensitivity of model scores under this protocol;
they do not show that highlighted regions are pneumonia or medically causal.

An annotation-ready evaluator is provided under
`experiments/exp06_expert_localization/`. It accepts only masks mapped to
validation-manifest images and scores native-grid activation/point overlap.
No expert masks were present, so it has produced no localization metrics.

## 4. External and clinical generalization

No external cohort was available or provided. The official test split was not
opened or evaluated. Therefore no external or clinical generalization result
is claimed. This limitation cannot be resolved by reanalyzing the current
validation/calibration splits; an independently collected or otherwise
independent cohort with reliable patient identifiers and appropriate
evaluation governance is needed.

## 5. Post-hoc calibration and uncertainty method audit

Experiment 4 is in `experiments/exp04_calibration_uncertainty/`. It used only
the 489-image calibration split. A fixed seed created a 342-image temperature-
fit subset and a 147-image audit subset. PNEUMONIA examples were grouped by
available patient token between the subsets (zero token overlap); NORMAL
examples were split at image level because patient identifiers are absent.
One positive scalar temperature was fit by minimizing binary negative
log-likelihood on the fit subset. The held-out audit subset was not used in
fitting.

| Audit metric (n=147) | Raw | Temperature-scaled |
|---|---:|---:|
| Temperature | — | 2.860098 |
| Negative log-likelihood | 0.085371 | 0.038282 |
| Brier score | 0.013467 | 0.011120 |
| 10-bin expected calibration error | 0.013537 | 0.011831 |
| Accuracy at threshold 0.5 | 0.9864 | 0.9864 |

MC dropout used the model's existing functional EfficientNet head dropout
(p=0.2), with batch-normalization and stochastic-depth behavior held in
evaluation mode, for 50 passes. On the 147-image audit subset there were only
2 prediction errors. AUROC for ranking those errors was 0.9862 for predictive
entropy, 0.9931 for probability variance, and 0.9931 for mutual information.
With only two errors, these ranking estimates are highly sample-sensitive and
must not be interpreted as proof of useful uncertainty estimation.

The app continues to display the original uncalibrated model probability and
does not apply the fitted temperature. The study is internal method evidence
only; it is not external calibration, clinical certainty, or a
distribution-free coverage guarantee. The checkpoint state hash was
identical before and after.

## Integrity and reproducibility

- No training or fine-tuning was performed.
- No official test manifest contents or test image was read.
- Experiment 3 read the validation manifest/images only.
- Experiment 4 read the calibration manifest/images only.
- New outputs were written only under `experiments/exp03_validation_robustness/`
  and `experiments/exp04_calibration_uncertainty/`, plus the new explanation
  evaluation outputs under `experiments/exp05_explanation_faithfulness/`.
- The expert-mask localization tool and blank annotation template do not
  contain or fabricate expert annotations or localization results.
- The baseline checkpoint and pre-existing manifest/experiment files were
  preserved. See `ui_testing_report.md` for original UI artifact hashes and
  test environment limitations.

## Remaining work

The highest-value next step is to obtain an independent, appropriately
governed external cohort with patient identifiers for both classes. Until
then, patient-independent and clinical generalization remain unresolved.
Calibration and uncertainty work would also need independent data and a
larger, pre-specified audit before being considered established.
