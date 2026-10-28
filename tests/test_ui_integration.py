"""UI integration tests using synthetic inputs and the existing frozen model."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import streamlit.testing.v1 as st_testing
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import read_json_artifact
from src.ui.inference import (
    MAX_UPLOAD_BYTES,
    PneumoniaInference,
    UploadValidationError,
    decode_uploaded_image,
)


def encoded_image(mode: str = "RGB", image_format: str = "PNG") -> bytes:
    if mode == "L":
        image = Image.new(mode, (224, 224), color=120)
    elif mode == "RGB":
        pixels = np.zeros((224, 224, 3), dtype=np.uint8)
        pixels[:, :112, 0] = 180
        pixels[:, 112:, 1] = 125
        pixels[80:145, 65:160, 2] = 220
        image = Image.fromarray(pixels)
    else:
        image = Image.new(mode, (224, 224), color=(60, 120, 180))
    result = io.BytesIO()
    image.save(result, format=image_format)
    return result.getvalue()


@pytest.fixture(scope="module")
def inference():
    return PneumoniaInference.from_project(PROJECT_ROOT)


def test_checkpoint_loads_and_uses_eval_mode(inference):
    assert not inference.model.training
    assert inference.device.type in {"cpu", "cuda"}
    assert (PROJECT_ROOT / "experiments/exp01_efficientnetb0_baseline/best_model.pt").is_file()


@pytest.mark.parametrize(
    ("mode", "filename"),
    [("RGB", "sample.png"), ("L", "sample.png")],
)
def test_rgb_and_grayscale_images_are_converted_and_preprocessed(
    inference, mode, filename
):
    image = decode_uploaded_image(encoded_image(mode), filename)
    assert image.mode == "RGB"
    tensor = inference._tensor_for_image(image)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_supported_three_channel_non_png_image_is_handled(inference):
    image = decode_uploaded_image(encoded_image("RGB", "JPEG"), "sample.jpg")
    prediction = inference.predict(image)
    assert 0.0 <= prediction.pneumonia_probability <= 1.0


@pytest.mark.parametrize(
    ("payload", "filename", "message"),
    [
        (b"not an image", "broken.png", "valid, decodable"),
        (encoded_image(), "sample.gif", "Unsupported image type"),
        (encoded_image("RGB", "JPEG"), "sample.png", "extension"),
        (b"", "empty.png", "empty"),
    ],
)
def test_invalid_uploads_are_rejected(payload, filename, message):
    with pytest.raises(UploadValidationError, match=message):
        decode_uploaded_image(payload, filename)


def test_oversized_upload_is_rejected():
    payload = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(UploadValidationError, match="15 MiB"):
        decode_uploaded_image(payload, "large.png")


def test_oversized_pixel_dimensions_are_rejected(monkeypatch):
    import sys

    image = Image.new("RGB", (2, 2))
    monkeypatch.setattr(sys.modules["src.ui.inference"], "MAX_IMAGE_PIXELS", 1)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with pytest.raises(UploadValidationError, match="pixel limit"):
        decode_uploaded_image(buffer.getvalue(), "sample.png")


def test_prediction_class_uses_the_project_threshold(inference):
    image = decode_uploaded_image(encoded_image(), "sample.png")
    prediction = inference.predict(image)
    assert prediction.class_name == (
        "PNEUMONIA"
        if prediction.pneumonia_probability >= 0.5
        else "NORMAL"
    )


def test_prediction_and_gradcam_leave_parameters_unchanged(inference):
    image = decode_uploaded_image(encoded_image(), "sample.png")
    before = {name: parameter.detach().clone() for name, parameter in inference.model.named_parameters()}
    inference.predict(image)
    native_heatmap, heatmap, overlay = inference.explain(image)
    assert native_heatmap.shape == (7, 7)
    assert heatmap.shape == (image.height, image.width)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert overlay.shape == (image.height, image.width, 3)
    for name, parameter in inference.model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_repeated_predictions_are_stable(inference):
    image = decode_uploaded_image(encoded_image(), "sample.png")
    first = inference.predict(image).pneumonia_probability
    second = inference.predict(image).pneumonia_probability
    assert first == pytest.approx(second, abs=1e-6)


def test_evaluation_transform_matches_training_pipeline(inference):
    from src.data.datamodule import get_transforms

    image = decode_uploaded_image(encoded_image("L"), "gray.png")
    established_transform = get_transforms(is_training=False)
    expected = established_transform(image).unsqueeze(0).to(inference.device)
    actual = inference._tensor_for_image(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_displayed_experiment_values_match_authoritative_artifacts():
    history_path = (
        PROJECT_ROOT
        / "experiments/exp01_efficientnetb0_baseline/training_history.json"
    )
    audit_path = (
        PROJECT_ROOT
        / "experiments/exp02a_gradcam/numerical_audit/gradcam_numerical_audit.json"
    )
    sensitivity_path = (
        PROJECT_ROOT
        / "experiments/exp02b_gradcam_sensitivity_pilot/sensitivity_summary.json"
    )
    robustness_path = (
        PROJECT_ROOT
        / "experiments/exp03_validation_robustness/"
        "validation_robustness_summary.json"
    )
    calibration_audit_path = (
        PROJECT_ROOT
        / "experiments/exp04_calibration_uncertainty/"
        "calibration_uncertainty_summary.json"
    )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    calibration_audit = json.loads(
        calibration_audit_path.read_text(encoding="utf-8")
    )
    best_epoch = audit["checkpoint_epoch_0indexed"]
    baseline = next(row for row in history if row["epoch"] == best_epoch)
    assert baseline["val_auc"] == 0.999747
    assert audit["confusion"] == {"TP": 382, "TN": 132, "FP": 2, "FN": 2}
    assert audit["val_size"] == 518
    assert len(audit["examples"]) == 16
    assert audit["border_dominance_summary"]["native_7x7_1cell_undefined_NaN"] == 4
    assert sensitivity["n_examples"] == 16
    assert sensitivity["by_condition"]["border_black"]["n_flip"] == 4
    assert sensitivity["by_condition"]["border_black"]["mean_abs_delta"] == 0.257993
    assert sensitivity["by_condition"]["central_black"]["n_flip"] == 7
    assert sensitivity["by_condition"]["central_black"]["mean_abs_delta"] == 0.475278
    assert sensitivity["by_condition"]["gamma_dark"]["n_flip"] == 1
    assert sensitivity["by_condition"]["gamma_dark"]["mean_abs_delta"] == 0.073331
    assert robustness["n_validation_images"] == 518
    assert robustness["threshold"] == 0.5
    assert robustness["baseline_confusion"] == {
        "TN": 132,
        "FP": 2,
        "FN": 2,
        "TP": 382,
    }
    assert robustness["model_state_unchanged"] is True
    assert robustness["results"]["brightness_0.9"]["n_flip"] == 2
    assert robustness["results"]["border_black"]["n_flip"] == 51
    assert robustness["results"]["central_black"]["n_flip"] == 132
    assert calibration_audit["n_calibration_images"] == 489
    assert calibration_audit["n_fit_images"] == 342
    assert calibration_audit["n_audit_images"] == 147
    assert calibration_audit["temperature"] == pytest.approx(2.860098, abs=1e-6)
    assert calibration_audit["model_state_unchanged"] is True


def test_missing_optional_artifact_is_informative_without_fabricating_data(tmp_path):
    assert read_json_artifact(tmp_path / "missing-summary.json") is None


def test_inference_does_not_read_calibration_or_test_manifests(monkeypatch):
    import builtins

    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        path = str(file).lower()
        assert "calibration_manifest.csv" not in path
        assert "test_manifest.csv" not in path
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    PneumoniaInference.from_project(PROJECT_ROOT, device=torch.device("cpu"))


def test_streamlit_app_smoke():
    app = st_testing.AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(
        timeout=30
    )
    assert not app.exception
    assert any("Explainable Pneumonia Detection" in title.value for title in app.title)
    assert len(app.tabs) == 3
    assert any("16 audit examples" in item.value for item in app.markdown)
    assert any(
        "full-split analysis is separate from the selected 16-image pilot"
        in item.value
        for item in app.caption
    )
    assert not any("Limitations and future work" in item.value for item in app.header)
    assert any(
        "Experiment 3 — full-validation robustness analysis" in item.value
        for item in app.subheader
    )
    assert any(
        "Experiment 4 — calibration and uncertainty audit" in item.value
        for item in app.subheader
    )
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "[data-testid=\"stMetricLabel\"]" in source
    assert "#334155" in source
    assert "[data-testid=\"stMetricValue\"]" in source
    assert "#0f172a" in source


def test_streamlit_valid_upload_prediction_and_explanation_flow():
    app = st_testing.AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(
        timeout=30
    )
    app.file_uploader[0].upload(
        "synthetic.png", encoded_image(), "image/png"
    ).run(timeout=60)
    assert not app.exception
    assert any(
        "Model prediction — not a diagnosis" in item.value
        for item in app.subheader
    )
    assert any("Predicted class" in metric.label for metric in app.metric)
    assert any("Grad-CAM explanation" in subheader.value for subheader in app.subheader)
    assert any("Native 7 × 7 grid" in item.value for item in app.caption)


def test_streamlit_corrupted_upload_flow_shows_safe_error():
    app = st_testing.AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(
        timeout=30
    )
    app.file_uploader[0].upload(
        "corrupted.png", b"not an image", "image/png"
    ).run(timeout=30)
    assert not app.exception
    assert any("valid, decodable image" in error.value for error in app.error)


def test_app_and_inference_do_not_save_or_transmit_uploads():
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    inference_source = (
        PROJECT_ROOT / "src" / "ui" / "inference.py"
    ).read_text(encoding="utf-8")
    assert ".save(" not in source
    assert ".save(" not in inference_source
    assert "requests." not in source + inference_source
    assert "urlopen(" not in source + inference_source
