# Explainable Pneumonia Detection

A local research and educational demonstration of the existing EfficientNetB0
chest X-ray baseline, its Grad-CAM visual explanations, and completed
validation-only experiments. The application is not for clinical diagnosis or
treatment.

## Run the local web UI

From the project root, use the project's virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for X-ray analysis and
`http://127.0.0.1:8000/experiments.html` for the saved experiment results.
Alternatively, run `python app.py` from the project root to start the same
local server. The browser UI is plain HTML, CSS, and JavaScript served by
FastAPI. The API uses the existing inference and Grad-CAM implementation and
frozen `experiments/exp01_efficientnetb0_baseline/best_model.pt` checkpoint.
Uploaded images are limited to 15 MiB, processed in memory, and never saved
or transmitted externally. The web results read only the saved Experiment 1,
2A, and 2B artifacts; no test or calibration split is loaded.

PyTorch and torchvision are intentionally installed separately; follow the
CUDA/CPU instructions at the top of `requirements.txt` if they are not already
available in the environment.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The web/API tests exercise image validation, prediction and Grad-CAM, static
navigation, and artifact-backed result payloads using synthetic images where
practical. They do not run training or load the calibration or official test
splits.

## Scope and limitations

The application displays binary NORMAL/PNEUMONIA predictions at the existing
0.5 threshold, the native 7 × 7 Grad-CAM grid, its upsampled display, and an
overlay. The displayed probability is not necessarily calibrated. Grad-CAM
does not establish causal evidence or segment anatomy. The bounded
16-example sensitivity pilot is exploratory; a separate full-validation
robustness analysis keeps black masking distinct as an out-of-distribution
stress test. A small post-hoc temperature-scaling and MC-dropout audit is
available, but it does not establish clinical calibration or reliable
uncertainty. Filename-derived patient-token limitations mean
patient-independent and clinical generalization are not established.

The Grad-CAM deletion/insertion faithfulness audit is available in
`experiments/exp05_explanation_faithfulness/`. It compares attribution-ranked
regions with random controls under a blurred-image replacement protocol; it
does not establish lesion localization. An evaluator for expert-annotated
validation masks is prepared in `experiments/exp06_expert_localization/`, but
no expert masks are currently supplied and no localization results are
claimed.

New architectures, retraining, external/clinical validation, and additional
robustness experiments are outside the current scope.
