# UI integration testing report

## Commands and results

Commands were run from the project root with the project's `.venv`:

| Command / check | Result |
|---|---|
| `.\.venv\Scripts\python.exe -m pip install "streamlit>=1.35"` | Installed Streamlit 1.65.0 and its runtime dependencies in `.venv`. Streamlit was already declared in `requirements.txt`; no dependency manifest was changed. |
| `.\.venv\Scripts\python.exe -m compileall -q app.py src\ui_inference.py src\trainer.py tests\test_ui_integration.py` | Passed. |
| `.\.venv\Scripts\python.exe -c "import src.model, src.trainer, app; print('imports ok')"` | Passed; printed `imports ok`. |
| `.\.venv\Scripts\python.exe -m streamlit --version` | Passed; Streamlit 1.65.0. |
| `.\.venv\Scripts\python.exe -m pytest tests\test_data_pipeline.py tests\test_gradcam.py tests\test_gradcam_numerical_audit.py tests\test_sensitivity_pilot.py tests\test_training_loop.py -q` | 141 passed, 6 failed in 37.10 s. |
| `.\.venv\Scripts\python.exe -m pytest tests\test_ui_integration.py -q` | 21 passed in 15.55 s. |
| `.\.venv\Scripts\python.exe -m pytest tests\test_calibration_uncertainty.py tests\test_validation_robustness.py tests\test_gradcam.py tests\test_ui_integration.py -q` | 60 passed in 17.96 s. |
| `.\.venv\Scripts\python.exe -m pytest tests\test_explanation_evaluation.py tests\test_faithfulness_audit.py tests\test_gradcam.py tests\test_ui_integration.py tests\test_validation_robustness.py tests\test_calibration_uncertainty.py -q` | 71 passed in 29.45 s. |
| `.\.venv\Scripts\python.exe experiments\exp05_explanation_faithfulness\run_faithfulness_audit.py` | Completed for all 518 internal-validation images; five random ranking controls per image; checkpoint/model state unchanged. |
| `.\.venv\Scripts\python.exe -m py_compile src\gradcam.py src\explanation_evaluation.py experiments\exp05_explanation_faithfulness\run_faithfulness_audit.py experiments\exp06_expert_localization\evaluate_localization.py tests\test_explanation_evaluation.py tests\test_faithfulness_audit.py` | Passed. |
| `.\.venv\Scripts\python.exe -m pytest tests -q` (latest full run after explanation-evaluation work) | 190 passed, 6 failed in 56.01 s. All UI, Grad-CAM, Experiment 3/4/5, and expert-mask helper tests passed; the six failures remain scikit-learn binary-load failures. |
| `.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false` | App started locally; the health request below returned HTTP 200. The test server was stopped after the check. |
| `.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8502 --browser.gatherUsageStats false` | Final integrated app health endpoint returned HTTP 200. Browser smoke check verified overview and Experiment 3/4 results tabs rendered without Streamlit exceptions; server stopped afterward. |
| `.\.venv\Scripts\python.exe experiments\exp03_validation_robustness\run_validation_robustness.py` | Completed on all 518 validation images; baseline confusion matched TN=132, FP=2, FN=2, TP=382; model state unchanged. |
| `.\.venv\Scripts\python.exe experiments\exp04_calibration_uncertainty\run_calibration_uncertainty.py` | Completed with calibration-only fit/audit subsets (342/147), temperature 2.860098, zero pneumonia-token overlap, and unchanged model state. |
| `.\.venv\Scripts\python.exe experiments\exp05_explanation_faithfulness\run_faithfulness_audit.py` | Completed on all 518 validation images; deletion advantage 0.0331 (image-bootstrap 95% CI 0.0265–0.0401); insertion advantage 0.0399 (0.0315–0.0486); frozen state unchanged. |
| `.\.venv\Scripts\python.exe experiments\exp06_expert_localization\evaluate_localization.py --annotations experiments\exp06_expert_localization\annotations_manifest_template.csv` | Correctly refused the empty template and reported that no expert annotations were supplied; no localization metrics were produced. |

Health-check command:

```powershell
curl.exe --silent --show-error --max-time 10 --write-out "`nHTTP %{http_code}`n" http://127.0.0.1:8501/_stcore/health
```

No training was run. UI integration tests used synthetic RGB, grayscale, and
JPEG data; they did not load validation, calibration, or official test images.
As separately approved follow-up analyses, Experiment 3 read all 518
internal-validation images and Experiment 4 read only the calibration split.
Experiment 5 read all 518 internal-validation images. Experiments 3–5 did not
parse the official test manifest or access test images.
As a later integrity check, only the official test manifest's SHA-256 was
recomputed; its records were not parsed or used for model development or
evaluation.
Configuration validation checks required paths for existence, but does not
read the calibration/test manifest contents during Experiment 3 or the test
manifest contents during Experiment 4. Streamlit's
`AppTest` exercised initial rendering, artifact-backed result display, a valid
synthetic upload through prediction and Grad-CAM, and a corrupted upload with
a safe error. A browser smoke check verified the updated title, overview, and
results section (including Experiments 3 and 4); no browser upload flow was
tested.

## Legacy-suite failures

All six final-suite failures are in scikit-learn-dependent metrics/training
tests:

- `TestMetricComputation.test_completely_wrong_predictions`
- `TestMetricComputation.test_metric_keys_present`
- `TestMetricComputation.test_perfect_predictions`
- `TestMetricComputation.test_random_chance_predictions`
- `TestTrainerCheckpointPolicy.test_default_checkpoint_policy_retains_best_and_last_only`
- `TestTrainerCheckpointPolicy.test_opt_in_saves_per_epoch_checkpoints`

The installed scikit-learn package cannot load its compiled
`_middle_term_computer` extension: Windows Application Control blocks the
binary, and subsequent imports also reported `KeyError: '__reduce_cython__'`.
This is an environment restriction, not an assertion regression from the UI.
Moving the scikit-learn import into `compute_binary_metrics()` prevents this
unrelated optional import from blocking UI/model loading; calling the existing
metrics function still requires a working scikit-learn installation.

The initial pre-implementation full-suite attempt also failed at collection
because importing `src` eagerly imported scikit-learn. No pytest warning
summary was emitted in the final run.

## Coverage and safety checks

The 21 new tests check checkpoint loading, CPU loading, RGB/grayscale/JPEG
handling, preprocessing parity with `get_transforms(is_training=False)`,
corrupt/empty/unsupported/mismatched/oversized uploads, the 0.5 class
threshold, bounded repeatability (absolute probability tolerance `1e-6`),
Grad-CAM heatmap and overlay dimensions/range, and unchanged model parameters.
They also compare displayed experiment source values to the stored JSON
artifacts, check a missing optional artifact returns an unavailable state,
guard against opening calibration/test manifests during inference, and test
that upload code does not save or transmit image content.

The app uses the project's shared deterministic evaluation transform,
`load_checkpoint()`, `GradCAM`, `get_target_layer()`, and
`overlay_heatmap()`. It does not instantiate the data module or load
calibration/test datasets. Upload size is capped at 15 MiB by Streamlit
configuration and validated again by the in-memory decoder.

## Artifact integrity

The selected and last checkpoint SHA-256 values match the values recorded
before UI inference:

| Artifact | SHA-256 |
|---|---|
| `best_model.pt` | `44fd6e9f60c64ab729144139361f059526c520c1c81779fbe2e4cca903b726ba` |
| `last_model.pt` | `52fc0693d02d58e4faec723726b12e280075439519364e8f31769fb99a0186d6` |

The original checkpoint, manifests, and baseline/Grad-CAM/sensitivity-pilot
outputs remain read-only. No training, official-test evaluation, or
manifest-generation command was run. The follow-up experiments read only the
authorized validation or calibration images and wrote outputs only in their
new experiment directories. The official test manifest was not parsed; its
SHA-256 was recomputed as an integrity check only. The following current
SHA-256 values were recorded during final verification:

| Artifact | SHA-256 |
|---|---|
| `train_manifest.csv` | `4baf79fa5fcc0fd761337f9365cb2b590cadd3a292b14ecb9942d7421178eab7` |
| `val_manifest.csv` | `b6642b30c5fd84255be51eb16e42c3078414781e74c2e7b69027047f4ca780ed` |
| `calibration_manifest.csv` | `094b33143a44f60f801e8ac232de9fb53c05ea66b8812eaa21430ce5dc516b8d` |
| `test_manifest.csv` | `7bc24f0edbc93df6f672a87ec76594d9b2a318595140ebf712117d4c61e14c52` |
| `training_history.json` | `8a8ec290f461e7a7a40f99d85bbb4b5270f65af9488b5a427a1790d4db2bcf2a` |
| `gradcam_numerical_audit.json` | `70d30733a764792618179ad1945ae81a93aacf1ea3e49cd05d45bb4412fc7e6e` |
| `sensitivity_summary.json` | `5da15921a47cd8c7aaec39439057efaa59f1f4feb697b44973a0172a60db13ec` |
| `validation_robustness_summary.json` | `9645d2f0314dc618200e55db2c84b4e82555e2aec6f05ebf37080b0276d0856c` |
| `calibration_uncertainty_summary.json` | `c63fe7b40c4b985cd1241bc587e9d7d8155ba5e142ea4205125b0b7b5dd48c63` |

## Deviations and unresolved issues

- The six scikit-learn-dependent legacy tests remain blocked by the local
  Windows Application Control policy. The UI/inference-specific tests and all
  other existing tests pass.
- The UI was validated using Streamlit's testing API, a local health endpoint,
  and a browser smoke check of the overview/results tabs. Synthetic inputs
  were used for upload/inference UI tests; the separately documented
  experiments used the authorized validation and calibration splits. A
  browser-based image upload was not tested.
- Accuracy was not added as a newly derived displayed metric. The UI shows
  only the validation metrics stored in training history and confusion counts
  stored in the numerical-audit artifact.
- Experiment 5 evaluates score faithfulness under a specific blurred-image
  replacement protocol; it does not establish causal explanations or lesion
  localization. Experiment 6 produced no localization result because no
  expert-annotated masks were available.

## Source package organization verification

The flat source modules were grouped by responsibility without changing the
model or experiment algorithms: data, models, training, explainability, and
UI inference. Import sites and explicit file-path loaders in the tests and
experiment scripts were updated to the new locations.

| Command / check | Result |
|---|---|
| `.\.venv\Scripts\python.exe -m compileall -q src app.py ui train.py run_exp02a.py tests experiments\exp02a_gradcam\numerical_audit\numerical_audit.py experiments\exp02b_gradcam_sensitivity_pilot\sensitivity_pilot.py` | Passed. |
| `.\.venv\Scripts\python.exe -c "import app, src.config, src.data.dataset, src.data.datamodule, src.models.model, src.training.trainer, src.explainability.gradcam, src.explainability.evaluation, src.ui.inference; print('all reorganized imports ok')"` | Passed; printed `all reorganized imports ok`. |
| `.\.venv\Scripts\python.exe -m pytest tests -q` | 190 passed, 6 failed in 57.39 s. All six failures are the pre-existing scikit-learn DLL/Application Control failures listed above. |

No training, experiment, or artifact-generation script was run during this
source-only reorganization.

## Training module organization

The baseline training implementation was separated into focused modules:
metrics and early stopping, checkpoint I/O, loss/optimizer/scheduler
factories, runtime metadata and epoch execution, and the high-level trainer.
`src.training.trainer` continues to expose the existing helper names for
compatibility with the training entry point and tests.

| Command / check | Result |
|---|---|
| `.\.venv\Scripts\python.exe -m compileall -q src\training train.py tests\test_training_loop.py` | Passed. |
| `.\.venv\Scripts\python.exe -c "from src.training.trainer import Trainer, EarlyStopping, compute_binary_metrics, get_environment_metadata, save_checkpoint, load_checkpoint, build_optimizer, build_scheduler, build_loss_fn, _run_epoch; from src.training import Trainer as PackageTrainer; print(Trainer is PackageTrainer)"` | Passed; printed `True`. |
| `.\.venv\Scripts\python.exe -m pytest tests\test_training_loop.py tests\test_ui_integration.py -q` | 61 passed in 74.82 s. |
| `.\.venv\Scripts\python.exe -m pytest tests -q` | 196 passed in 55.16 s. |
| `.\.venv\Scripts\python.exe -m pytest tests\test_ui_integration.py -q` | 21 passed in 16.38 s after clarifying the separate 16-image pilot and 518-image full-validation results and removing the limitations tab. |
