# Explainable Pneumonia Detection from Chest X-Rays

A rigorous, open, and reproducible research platform for chest X-ray classification and
post-hoc explainability auditing. The system classifies radiographs as **NORMAL** or
**PNEUMONIA** using a frozen EfficientNetB0 checkpoint and pairs every prediction with a
mathematically audited **Grad-CAM** visual explanation. It is not intended for clinical
diagnosis or treatment.

## Research context

The project investigates whether Grad-CAM activation patterns of a high-performing
classifier are spatially consistent with clinically expected localization, or whether they
are confounded by peripheral image features. Two complementary analyses are performed:

- **Exp 2A -- Native-resolution Grad-CAM audit**: Activation patterns for 16 stratified
  validation cases are decomposed on the raw 7x7 conv_head grid. Border-to-inner ratios
  are computed without epsilon patching; undefined ratios (zero inner activation) are
  reported as NaN.
- **Exp 2B -- Controlled spatial perturbation pilot**: Area-matched border and central
  blackout conditions (24,576 px vs. 24,336 px, within 0.98%) are applied to the same 16
  cases. Perturbation parameters were pre-committed before any results were examined.

Full results, methodology, and limitations are described in `project_description.md`.

## Run the local web UI

From the project root, activate the virtual environment and start the server:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Alternatively run `python app.py` from the project root to start the same server.

| URL | Contents |
|-----|----------|
| `http://127.0.0.1:8000/` | Live upload, prediction, and Grad-CAM explanation |
| `http://127.0.0.1:8000/experiments.html` | Saved Exp 1, 2A, 2B results and audit galleries |

PyTorch and torchvision are intentionally excluded from `requirements.txt`. Install them
separately using the CUDA or CPU index URL shown at the top of that file.

Uploaded images are limited to 15 MiB, processed entirely in memory, and never saved or
transmitted externally. The experiments page reads only pre-saved artifacts from Exp 01,
02A, and 02B; no test or calibration split is loaded at runtime.

## Run the test suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The suite covers 11 test modules (175 collected functions) exercising:
- EfficientNetB0 model construction and forward pass
- Grad-CAM correctness (ReLU order, native 7x7 output, normalization)
- Numerical audit replication to float32 precision (all 16 cases)
- Sensitivity pilot perturbation pipeline and result verification
- Data pipeline, manifest loading, and leakage checks
- In-memory upload validation and FastAPI prediction/Grad-CAM endpoints
- Calibration, MC-dropout uncertainty, validation robustness, and faithfulness modules
- Artifact-backed experiment visualization API endpoints

Note: `test_ui_integration.py` has a known import issue (`read_json_artifact` from
`app`) that causes a collection error; the remaining 175 tests collect and run normally.
Tests do not perform training or load the calibration or official test splits.

## Scope and limitations

**Classifier**: Binary NORMAL/PNEUMONIA predictions at a fixed 0.5 threshold using
`experiments/exp01_efficientnetb0_baseline/best_model.pt` (Epoch 8, val AUC=0.9997,
sensitivity=99.48%, specificity=98.51%, n=518).

**Grad-CAM**: Native 7x7 grid produced by hooking `backbone.conv_head`. ReLU is applied
before bicubic upsampling to 224x224. Activations reflect model correlations; they do not
constitute causal evidence or anatomical segmentation.

**Pilot scale (n=16)**: Exp 2A and 2B results are based on a stratified but non-random
subset and must not be generalised to the full validation set or beyond.

**OOD perturbations**: Large black masks (border or central) used in Exp 2B and Exp 3 do
not correspond to any image type in the training distribution. Model responses cannot be
interpreted as principled feature attributions.

**Calibration and uncertainty (Exp 04)**: Temperature scaling (T=2.860) and MC-dropout
(50 passes) are audited on an internal calibration partition only. These results do not
establish clinical-grade calibration or distribution-free coverage guarantees.

**Faithfulness audit (Exp 05)**: Deletion and insertion curves use a Gaussian-blurred
replacement; they confirm model-level faithfulness under the stated protocol and do not
establish lesion localization.

**Expert localization (Exp 06)**: The evaluation framework and `annotations_manifest_template.csv`
are implemented, but no expert masks are currently available and no localization results
are claimed.

**Single site and single model**: All images originate from Guangzhou Women and Children's
Medical Center. Findings may not transfer to other institutions, patient demographics,
acquisition protocols, or model architectures.

New architectures, retraining, external/clinical validation, and additional robustness
experiments are outside the current scope.
