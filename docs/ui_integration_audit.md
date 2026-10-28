# UI integration audit

## Scope and inspected project areas

The project contains `src/`, `configs/`, `tests/`, `experiments/`,
`data/manifests/`, `audit_output/`, and `docx/`. The repository root has
training and manifest-generation scripts plus `requirements.txt`, but no
README or existing application. The project virtual environment is Python
3.12.10 with PyTorch 2.11.0+cu126, torchvision, timm, Pillow, PyYAML, and
the scientific Python dependencies installed. Streamlit is declared in
`requirements.txt` but is not installed in that environment.

Inspected configuration, package exports, model, manifest dataset, data
transforms, Grad-CAM, trainer checkpoint I/O, all existing test modules,
baseline training history/log and checkpoint metadata, split/leakage
summaries, Grad-CAM metadata and numerical-audit outputs, and sensitivity
pilot report/summary/results. Relevant tree entries include:

- `configs/config.yaml`
- `src/config.py`, `src/data/`, `src/models/`, `src/training/`,
  `src/explainability/`
- `tests/test_data_pipeline.py`, `tests/test_training_loop.py`,
  `tests/test_gradcam.py`, `tests/test_gradcam_numerical_audit.py`,
  `tests/test_sensitivity_pilot.py`
- `experiments/exp01_efficientnetb0_baseline/`
- `experiments/exp02a_gradcam/`, including `numerical_audit/` and `followup/`
- `experiments/exp02b_gradcam_sensitivity_pilot/` and the additional
  `experiments/exp02b_perturbation/` output directory
- `data/manifests/` and `audit_output/`

## Authoritative interfaces and artifacts

- **Checkpoint:** `experiments/exp01_efficientnetb0_baseline/best_model.pt`
  is the selected frozen checkpoint (zero-based epoch 8, 0.5 decision
  threshold). `last_model.pt` is also present but is not the selected model.
  Both are state-dictionary checkpoints; the trainer's
  `load_checkpoint()` loads them with `map_location`.
- **Model:** `src.models.model.EfficientNetB0Classifier`, constructed with
  `pretrained=False` before loading the checkpoint. This avoids network
  weight downloads and prevents replacing trained weights with newly fetched
  weights.
- **Class mapping:** `NORMAL: 0`, `PNEUMONIA: 1`.
- **Inference preprocessing:** `src.data.datamodule.get_transforms()` in
  evaluation mode: bicubic resize to 224 x 224, RGB tensor conversion, then
  ImageNet normalization (means 0.485/0.456/0.406 and standard deviations
  0.229/0.224/0.225). `ManifestDataset` also explicitly converts inputs to
  RGB. The app should call the shared evaluation transform and must not
  instantiate `PneumoniaDataModule.setup()`, which can load calibration and
  test datasets.
- **Prediction:** there is no shared single-image inference function. The
  model returns a single logit; apply sigmoid and the documented 0.5
  threshold. A new reusable UI inference helper is warranted.
- **Explanation:** `src.explainability.gradcam.GradCAM`, `get_target_layer()`, and
  `overlay_heatmap()` are the established entry points. The target is
  `backbone.conv_head`; its 224-pixel input produces a native 7 x 7 map that
  the implementation upsamples bicubically. Grad-CAM uses gradients; ordinary
  prediction must run without gradients.
- **Baseline results:** `training_history.json` records validation metrics at
  the best checkpoint: loss 0.022016, AUC 0.999747, sensitivity 0.994792,
  specificity 0.985075, and F1 0.994792. The Experiment 2A numerical-audit
  JSON records internal-validation confusion counts TP=382, TN=132, FP=2,
  FN=2 on 518 examples. These are validation results, not test results.
- **Experiment 2A:** machine-readable files include
  `experiments/exp02a_gradcam/numerical_audit/gradcam_numerical_audit.json`
  and `numerical_audit_metadata.json`; metadata identifies the target layer,
  epoch, threshold, and validation-only sample selection. The numerical audit
  reports a 7 x 7 grid cell size of 32 pixels. A 22-pixel border occupies
  only 0.6875 of a native cell; among 16 selected audit examples, the upsampled
  22-pixel ratio exceeds 1 in 16, while the aligned one-cell native-grid
  comparison does so in 12 and is undefined in 4. This demonstrates that
  interpolation and mask/grid
  alignment affect the spatial statistic; it is not evidence of precise
  localization. The follow-up metadata records unchanged parameters/logits.
- **Experiment 2B:** the project report and machine-readable summary are in
  `experiments/exp02b_gradcam_sensitivity_pilot/`. The summary records
  border masking: 4/16 flips, mean absolute probability change 0.257993;
  central masking: 7/16, 0.475278; gamma darkening: 1/16, 0.073331. These
  are exploratory results on 16 selected internal-validation examples.
  `experiments/exp02b_perturbation/` contains additional metadata/results;
  the UI should use the report-linked Grad-CAM sensitivity-pilot summary to
  avoid mixing outputs.
- **Data split:** `data/manifests/split_summary.json` reports 4,183 train,
  518 validation, 489 calibration, 624 official test, and 16 reference
  examples. Pneumonia filename patient tokens were grouped for the train,
  validation, and calibration split; NORMAL filenames do not provide reliable
  patient tokens. The report notes 135 pneumonia patient tokens overlap
  between train and the unchanged official test set. The test manifest and
  images must not be read by the UI or its tests.

There is no standalone Experiment 1 or Experiment 2A Markdown report, but
their training history/log and experiment JSON artifacts are available.
Optional/missing result files must produce a visible unavailable notice, not
hard-coded replacement values. The app will read only the required baseline
history and Grad-CAM audit summary and the sensitivity-pilot summary; no
calibration or test data are required.

## Risks and integration decisions

- `src.__init__` eagerly imports `src.training.trainer`, which originally imported
  scikit-learn at module load. In the existing virtual environment Windows
  Application Control blocks one compiled scikit-learn extension. The
  implementation moves the scikit-learn import inside
  `compute_binary_metrics()`, preserving the public package API while
  allowing unrelated model and checkpoint imports; metric computation still
  requires a functioning scikit-learn installation.
- `src.config.load_config()` resolves data and manifest paths and validates
  the complete training configuration. Keep app artifact paths rooted at the
  project directory and use only the config's model/preprocessing sections;
  do not use paths copied as absolute paths in experiment metadata.
- `PneumoniaDataModule.setup()` defaults to constructing train, validation,
  calibration, and test datasets. Avoid this entry point entirely in app
  startup/inference.
- The 2B black masks may be out-of-distribution; findings are not causal
  evidence. Filename-derived patient tokens do not prove patient-independent
  generalization. Grad-CAM is coarse and not a segmentation.
- Validate uploads by byte size, supported extension, successful full image
  decode, and RGB conversion. Keep upload bytes in memory. Do not save,
  transmit, train on, or evaluate user uploads. Render generic errors and log
  internal diagnostics without exposing paths or traces.
- Streamlit is the preferred UI. Add UI/integration tests with Streamlit's
  testing API and synthetic images; no browser-based test is assumed.

## Proposed architecture

- Root `app.py`: overview, upload/prediction, Grad-CAM visualization,
  artifact-backed experiment results, limitations, and explicit research-only
  disclaimer.
- `src/ui/inference.py`: one-time frozen checkpoint load, safe in-memory
  upload validation, the shared evaluation transform, sigmoid prediction,
  Grad-CAM generation, and image overlay. Keep UI rendering out of this module.
- `tests/test_ui_integration.py`: synthetic RGB/grayscale images, validation
  failures, inference/Grad-CAM integrity, artifact-value parity, and Streamlit
  UI smoke/flow checks. Tests must not read calibration/test manifests or
  images.
- `docs/ui_testing_report.md`: commands, actual outcomes, environment
  limitations, and artifact-integrity evidence.
- Update the README with virtual-environment setup and launch instructions;
  retain the existing baseline, manifests, and experiment outputs unchanged.
