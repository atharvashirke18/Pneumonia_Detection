# Project File Guide

This guide provides a comprehensive inventory and functional description of every core file, module, asset, test, and research artifact in the repository.

---

## 1. Project Entry Points & Server

| File | Purpose |
|---|---|
| [`app.py`](file:///d:/Pneumonia_Detection/app.py) | Local application entry point launching the FastAPI server on `http://127.0.0.1:8000` via Uvicorn (`python app.py`). |
| [`train.py`](file:///d:/Pneumonia_Detection/train.py) | Command-line training pipeline for the baseline EfficientNetB0 model. Reads `configs/config.yaml`, runs epoch orchestration with early stopping, and serializes model checkpoints. |
| [`build_manifests.py`](file:///d:/Pneumonia_Detection/build_manifests.py) | Dataset indexing and auditing utility. Scans `data/chest_xray/`, detects exact duplicates, extracts patient tokens, and writes stratified CSV manifests under `data/manifests/`. |
| [`verify_counts.py`](file:///d:/Pneumonia_Detection/verify_counts.py) | Dataset verification script confirming image counts across splits, duplicate exclusion, and checking for patient-token leakage. |
| [`run_exp02a.py`](file:///d:/Pneumonia_Detection/run_exp02a.py) | Standalone script reproducing validation predictions, Grad-CAM heatmaps, overlays, and metadata for Experiment 2A. |
| [`requirements.txt`](file:///d:/Pneumonia_Detection/requirements.txt) | Python package dependencies (PyTorch, torchvision, FastAPI, Uvicorn, Pillow, NumPy, PyYAML, httpx, etc.). |
| [`README.md`](file:///d:/Pneumonia_Detection/README.md) | Primary setup guide, quickstart instructions, API descriptions, and scope boundaries. |
| [`project_summary.md`](file:///d:/Pneumonia_Detection/project_summary.md) | High-level architectural overview, implementation breakdown, end-to-end workflow, and clinical limitations. |
| [`file_content.md`](file:///d:/Pneumonia_Detection/file_content.md) | This document; complete inventory and guide to all project files. |
| [`docx/project_specs.md`](file:///d:/Pneumonia_Detection/docx/project_specs.md) | Original research specifications and planned roadmap. |

---

## 2. Web Application & API Backend

### 2.1 Backend (`api/`)

| File | Purpose |
|---|---|
| [`api/main.py`](file:///d:/Pneumonia_Detection/api/main.py) | FastAPI application. Mounts static directories (`/`, `/css`, `/js`, `/images`), implements `/api/predict` (in-memory validation, model inference, Grad-CAM generation), and exposes experiment endpoints (`/api/experiments/exp01/...`, `/api/experiments/exp02a/...`, `/api/experiments/exp02b/...`). Serves static perturbation images directly via `FileResponse`. |

### 2.2 Frontend Pages & Assets (`web/`)

| File | Purpose |
|---|---|
| [`web/index.html`](file:///d:/Pneumonia_Detection/web/index.html) | Main prediction interface ("PneumoLens"). Features a segmented viewer card (`#uploadCard`) with view switcher buttons (`Original`, `Heatmap`, `Overlay`), interactive drag-and-drop zone (`#uploadZone`), anatomical marker (`R`), Jet colormap scale, right sidebar with live diagnosis card (`#resultContent`), calibrated probability bar, fixed 0.5 threshold line, Grad-CAM technical table (`#gradcamTable`), and clinical disclaimer. |
| [`web/experiments.html`](file:///d:/Pneumonia_Detection/web/experiments.html) | Research results dashboard featuring a sidebar layout (`.exp-layout`, `.side`). Includes Exp 01 Baseline metrics and loss/AUC curves, Exp 02A Grad-CAM Audit with the **Image Grid ⇄ Bar Charts toggle** (16 visual cards with ratio color badges and triple-image modal), and Exp 02B Sensitivity Pilot with perturbation summary cards, charts, and 16-case × 4-condition image gallery. |
| [`web/css/theme.css`](file:///d:/Pneumonia_Detection/web/css/theme.css) | Modern unified dark theme stylesheet defining color tokens (`--bg: #0B1220`, `--card: #111B2E`, `--accent: #2DD4BF`), segmented view switcher, stage viewer, sidebar research layout, and visual grid cards. |
| [`web/css/style.css`](file:///d:/Pneumonia_Detection/web/css/style.css) | Core stylesheet providing base layout containers, cards, tables, alerts, and modal dialog utilities. |
| [`web/css/detection.css`](file:///d:/Pneumonia_Detection/web/css/detection.css) | Specialized clinical light theme stylesheet (retained for backward compatibility and alternative styling). |
| [`web/js/main.js`](file:///d:/Pneumonia_Detection/web/js/main.js) | Client-side controller for `web/index.html`: manages file dropzone and picker, loading overlay, `/api/predict` multipart communication, prediction badge rendering, and segmented image view switching (`orig`, `heat`, `over`). |
| [`web/js/experiments.js`](file:///d:/Pneumonia_Detection/web/js/experiments.js) | Client-side controller for `web/experiments.html`: fetches experiment artifacts, manages tab navigation, coordinates the Image Grid ⇄ Bar Charts toggle, handles modal image expansion, and renders the Exp 02B perturbation gallery. |
| [`web/js/charts.js`](file:///d:/Pneumonia_Detection/web/js/charts.js) | Chart.js helper wrappers for training loss/AUC curves, border vs. inner activation comparisons, ratio plots, and horizontal per-image sensitivity plots. |

---

## 3. Configuration & Core Source Modules (`src/`)

### 3.1 Configuration (`configs/` and `src/config.py`)

| File | Purpose |
|---|---|
| [`configs/config.yaml`](file:///d:/Pneumonia_Detection/configs/config.yaml) | Authoritative YAML configuration declaring paths, dataset manifests, input dimensions (224×224), ImageNet normalization statistics, model hyperparameters, optimizer/scheduler settings, and device. |
| [`src/config.py`](file:///d:/Pneumonia_Detection/src/config.py) | Configuration parser loading `config.yaml`, resolving absolute project paths, and validating required parameters. |

### 3.2 Data Modules (`src/data/`)

| File | Purpose |
|---|---|
| [`src/data/__init__.py`](file:///d:/Pneumonia_Detection/src/data/__init__.py) | Public exports for data-loading classes and transforms. |
| [`src/data/dataset.py`](file:///d:/Pneumonia_Detection/src/data/dataset.py) | `PneumoniaDataset` implementation; reads manifest CSVs, resolves image paths, and returns transformed `(image_tensor, label)` pairs. |
| [`src/data/datamodule.py`](file:///d:/Pneumonia_Detection/src/data/datamodule.py) | `PneumoniaDataModule` managing PyTorch DataLoaders and shared evaluation/training torchvision transform pipelines. |

### 3.3 Model Architecture (`src/models/`)

| File | Purpose |
|---|---|
| [`src/models/__init__.py`](file:///d:/Pneumonia_Detection/src/models/__init__.py) | Public exports for model definitions. |
| [`src/models/model.py`](file:///d:/Pneumonia_Detection/src/models/model.py) | `EfficientNetB0Classifier` neural network architecture. Replaces default classifier head with a dropout-linear binary classification layer and provides the `build_model()` factory. |

### 3.4 Training & Checkpoints (`src/training/`)

| File | Purpose |
|---|---|
| [`src/training/__init__.py`](file:///d:/Pneumonia_Detection/src/training/__init__.py) | Public exports for training utilities. |
| [`src/training/trainer.py`](file:///d:/Pneumonia_Detection/src/training/trainer.py) | High-level `Trainer` class managing training/validation epochs, learning rate scheduling, early stopping, and history persistence. |
| [`src/training/metrics.py`](file:///d:/Pneumonia_Detection/src/training/metrics.py) | Binary classification metrics (loss, accuracy, sensitivity, specificity, F1-score, ROC-AUC) and early stopping state tracker. |
| [`src/training/components.py`](file:///d:/Pneumonia_Detection/src/training/components.py) | Factory functions constructing the BCEWithLogitsLoss loss function, AdamW optimizer, and CosineAnnealingLR scheduler. |
| [`src/training/checkpoints.py`](file:///d:/Pneumonia_Detection/src/training/checkpoints.py) | Serialization and restoration utilities for model weights, optimizer state, scheduler state, and AMP scaler state. |
| [`src/training/runtime.py`](file:///d:/Pneumonia_Detection/src/training/runtime.py) | Hardware/environment provenance collector and reusable single-epoch train/evaluation loops. |

### 3.5 Explainability & Grad-CAM (`src/explainability/`)

| File | Purpose |
|---|---|
| [`src/explainability/__init__.py`](file:///d:/Pneumonia_Detection/src/explainability/__init__.py) | Public exports for explainability services. |
| [`src/explainability/gradcam.py`](file:///d:/Pneumonia_Detection/src/explainability/gradcam.py) | `GradCAM` implementation. Taps `backbone.conv_head`, captures gradients via hooks, applies channel pooling and pre-upsampling ReLU, and outputs native $7 \times 7$ matrices, 224×224 heatmaps, and colormap overlays. |
| [`src/explainability/evaluation.py`](file:///d:/Pneumonia_Detection/src/explainability/evaluation.py) | Localization evaluation metrics (Intersection over Union, pointing game, energy hit rate) for comparing Grad-CAM against bounding boxes or segmentation masks. |

### 3.6 UI Inference Service (`src/ui/`)

| File | Purpose |
|---|---|
| [`src/ui/__init__.py`](file:///d:/Pneumonia_Detection/src/ui/__init__.py) | Public exports for UI inference utilities. |
| [`src/ui/inference.py`](file:///d:/Pneumonia_Detection/src/ui/inference.py) | `PneumoniaInference` service and upload validation helper `decode_uploaded_image()`. Handles in-memory payload verification, frozen checkpoint loading, forward prediction, and explanation generation. |

---

## 4. Automated Test Suite (`tests/`)

| Test File | Verification Scope |
|---|---|
| [`tests/test_data_pipeline.py`](file:///d:/Pneumonia_Detection/tests/test_data_pipeline.py) | Config parsing, manifest integrity, dataset indexing, transform normalization, and DataLoader batch shapes. |
| [`tests/test_training_loop.py`](file:///d:/Pneumonia_Detection/tests/test_training_loop.py) | Metric calculations (AUC, sensitivity, specificity), early stopping behavior, checkpoint saving/loading, and optimizer steps. |
| [`tests/test_gradcam.py`](file:///d:/Pneumonia_Detection/tests/test_gradcam.py) | Target layer resolution, gradient hook registration, output shape, deterministic reproducibility, pre-upsampling ReLU, and parameter immutability. |
| [`tests/test_gradcam_numerical_audit.py`](file:///d:/Pneumonia_Detection/tests/test_gradcam_numerical_audit.py) | Exact spatial mask math (1-cell border $n=24$, inner $n=25$), numerical audit recomputations, and undefined ratio handling for zero inner activation. |
| [`tests/test_sensitivity_pilot.py`](file:///d:/Pneumonia_Detection/tests/test_sensitivity_pilot.py) | Exp 2B perturbation implementations (32 px border, 156 px central, gamma darkening), pixel area parity, and flip counters. |
| [`tests/test_validation_robustness.py`](file:///d:/Pneumonia_Detection/tests/test_validation_robustness.py) | Full-validation stress-testing routines across synthetic perturbation conditions. |
| [`tests/test_calibration_uncertainty.py`](file:///d:/Pneumonia_Detection/tests/test_calibration_uncertainty.py) | Calibration split isolation, temperature scaling optimization, and Monte Carlo dropout variance checks. |
| [`tests/test_ui_integration.py`](file:///d:/Pneumonia_Detection/tests/test_ui_integration.py) | In-memory upload validation, invalid image rejection, inference preprocessing parity, and frozen model predictions. |
| [`tests/test_faithfulness_audit.py`](file:///d:/Pneumonia_Detection/tests/test_faithfulness_audit.py) | Grad-CAM explanation faithfulness conventions, pixel deletion/insertion curves, and area under the curve calculations. |
| [`tests/test_explanation_evaluation.py`](file:///d:/Pneumonia_Detection/tests/test_explanation_evaluation.py) | Expert mask comparison functions (IoU, Dice, pointing game) and annotation manifest format validation. |

---

## 5. Dataset & Manifest Audits (`data/` and `audit_output/`)

| Path | Contents & Role |
|---|---|
| `data/chest_xray/` | Source dataset containing labeled chest radiographs organized by split and class (`train/`, `val/`, `test/`). |
| [`data/manifests/train_manifest.csv`](file:///d:/Pneumonia_Detection/data/manifests/train_manifest.csv) | Audited training split images with labels and metadata. |
| [`data/manifests/val_manifest.csv`](file:///d:/Pneumonia_Detection/data/manifests/val_manifest.csv) | Audited validation split images used for Grad-CAM, sensitivity, and robustness evaluations. |
| [`data/manifests/calibration_manifest.csv`](file:///d:/Pneumonia_Detection/data/manifests/calibration_manifest.csv) | Isolated partition reserved for post-hoc calibration and temperature scaling. |
| [`data/manifests/test_manifest.csv`](file:///d:/Pneumonia_Detection/data/manifests/test_manifest.csv) | Official test partition, preserved in complete isolation from tuning. |
| [`data/manifests/val_reference_16.csv`](file:///d:/Pneumonia_Detection/data/manifests/val_reference_16.csv) | Stratified 16-case subset used across Exp 2A and Exp 2B (all FP & FN, stratified TP & TN). |
| [`data/manifests/duplicates_manifest.csv`](file:///d:/Pneumonia_Detection/data/manifests/duplicates_manifest.csv) | Record of exact duplicate images identified during dataset ingestion. |
| [`data/manifests/split_summary.json`](file:///d:/Pneumonia_Detection/data/manifests/split_summary.json) | Numerical summary of partition sizes and class distributions. |
| [`data/manifests/leakage_report.json`](file:///d:/Pneumonia_Detection/data/manifests/leakage_report.json) | Patient token overlap audit verifying zero subject leakage across splits. |
| `audit_output/exact_duplicates.json` | Hash-based duplicate image detection output. |
| `audit_output/filename_patient_overlap.json` | Patient token cross-split overlap audit output. |

---

## 6. Experiment Artifacts & Reports (`experiments/`)

### 6.1 Experiment 1 — Baseline EfficientNetB0 (`experiments/exp01_efficientnetb0_baseline/`)
- `best_model.pt`: Frozen model checkpoint from epoch 8 (lowest validation loss: 0.0384).
- `last_model.pt`: Final epoch model checkpoint.
- `training_history.json`: Epoch-by-epoch loss, AUC, sensitivity, specificity, and F1 metrics.
- `config_snapshot.json`: Exact snapshot of training configuration.
- `environment_metadata.json`: Hardware, PyTorch, CUDA, and OS environment provenance.
- `training.log`: Console logging output from the training run.

### 6.2 Experiment 2A — Grad-CAM Audit (`experiments/exp02a_gradcam/`)
- `gradcam_metadata.json`: Machine-readable records for the 16 selected validation cases.
- `contact_sheet.png`: High-resolution overview sheet displaying all 16 cases.
- `originals/`: Original 224×224 normalized input images for the 16 cases.
- `heatmaps/`: Grad-CAM heatmap PNGs for the 16 cases.
- `overlays/`: Grad-CAM overlay PNGs for the 16 cases.
- `followup/`: Native $7 \times 7$ vs. upsampled bicubic/nearest-neighbour comparison images.
- `numerical_audit/`:
  - `numerical_audit.py`: Audit script computing exact native 7×7 border/inner statistics.
  - `gradcam_numerical_audit.csv` & `gradcam_numerical_audit.json`: Audited values (native 1-cell border mean $n=24$, inner mean $n=25$, exact ratios, and epsilon patch analysis).
  - `numerical_audit_metadata.json` & `numerical_audit_summary.csv`: Protocol metadata and summary tables.

### 6.3 Experiment 2B — Sensitivity Pilot (`experiments/exp02b_gradcam_sensitivity_pilot/`)
- `sensitivity_pilot.py`: Reproducible pilot script evaluating 16 cases under 4 conditions.
- `sensitivity_results.csv` & `sensitivity_summary.json`: Pre- and post-perturbation probabilities, $\Delta P$, and flip counts.
- `sensitivity_report.md`: Scientific findings and interpretation of the sensitivity pilot.
- `sensitivity_contact_sheet.png`: 16-row × 4-column visual contact sheet.
- `images/`: **64 pre-rendered PNG artifacts** (`{ds_idx}_original.png`, `{ds_idx}_border_black.png`, `{ds_idx}_central_black.png`, `{ds_idx}_gamma_dark.png`), served directly by the web backend for instant loading.

### 6.4 Experiment 2B Peripheral Perturbation (`experiments/exp02b_perturbation/`)
- `exp02b_metadata.json` & `exp02b_sensitivity_results.csv`: Synthetic marker and peripheral occlusion protocol records.
- `figures/` & `masks/`: Visualizations and explicit binary masks used for spatial sensitivity testing.

### 6.5 Experiment 3 — Full-Validation Robustness (`experiments/exp03_validation_robustness/`)
- `run_validation_robustness.py`: Runner testing entire validation split under synthetic perturbations.
- `validation_robustness_report.md`: Markdown summary of robustness findings.
- `validation_robustness_summary.json` & `validation_robustness_results.csv`: Aggregate metrics and per-image predictions under perturbation.

### 6.6 Experiment 4 — Calibration & Uncertainty (`experiments/exp04_calibration_uncertainty/`)
- `run_calibration_uncertainty.py`: Post-hoc temperature scaling and Monte Carlo dropout evaluator.
- `calibration_uncertainty_report.md`: Calibration report analyzing Expected Calibration Error (ECE).
- `calibration_uncertainty_summary.json` & `calibration_uncertainty_audit.csv`: Calibration parameters, optimal temperature $T$, and predictive variance estimates.

### 6.7 Experiment 5 — Explanation Faithfulness (`experiments/exp05_explanation_faithfulness/`)
- `run_faithfulness_audit.py`: Insertion and deletion pixel-flipping audit script.
- `faithfulness_report.md`: Explanation faithfulness analysis comparing Grad-CAM rankings against random baselines.
- `faithfulness_summary.json` & `faithfulness_results.csv`: Insertion/deletion curve metrics and Area Under the Curve (AUC) scores.

### 6.8 Experiment 6 — Expert Localization Framework (`experiments/exp06_expert_localization/`)
- `README.md`: Protocol specification, required annotation format, and evaluation commands.
- `evaluate_localization.py`: Standalone localization evaluator computing IoU and pointing game accuracy against supplied clinical masks.
- `annotations_manifest_template.csv`: Schema template for expert clinical annotations.

---

## 7. Documentation (`docs/`)

| File | Purpose |
|---|---|
| [`docs/ui_integration_audit.md`](file:///d:/Pneumonia_Detection/docs/ui_integration_audit.md) | Initial architecture analysis and UI integration planning. |
| [`docs/ui_testing_report.md`](file:///d:/Pneumonia_Detection/docs/ui_testing_report.md) | Verification commands, test execution logs, and validation summaries. |
| [`docs/limitations_followup.md`](file:///d:/Pneumonia_Detection/docs/limitations_followup.md) | Detailed analysis of clinical limitations, calibration boundaries, and interpretability constraints. |
