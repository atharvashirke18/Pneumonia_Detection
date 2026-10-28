"""Unit tests for the pre-specified full-validation robustness protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    PROJECT_ROOT
    / "experiments"
    / "exp03_validation_robustness"
    / "run_validation_robustness.py"
)
SPEC = importlib.util.spec_from_file_location("validation_robustness", SCRIPT)
ROBUSTNESS = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(ROBUSTNESS)


def uniform_image(value: int = 128) -> Image.Image:
    return Image.fromarray(
        np.full((224, 224, 3), value, dtype=np.uint8)
    )


def test_protocol_covers_approved_conditions_and_groups():
    assert set(ROBUSTNESS.CONDITIONS) == {
        "brightness_0.9",
        "brightness_1.1",
        "contrast_0.9",
        "contrast_1.1",
        "jpeg_quality_85",
        "gamma_dark",
        "border_black",
        "central_black",
    }
    assert ROBUSTNESS.CONDITION_GROUPS["ood_mask_stress"] == (
        "border_black",
        "central_black",
    )
    assert ROBUSTNESS.THRESHOLD == 0.5


def test_brightness_conditions_are_deterministic_and_do_not_mutate_source():
    source = uniform_image(100)
    before = np.asarray(source).copy()
    darker = ROBUSTNESS.apply_condition(source, "brightness_0.9")
    brighter = ROBUSTNESS.apply_condition(source, "brightness_1.1")
    assert np.asarray(darker).mean() == 90
    assert np.asarray(brighter).mean() == 110
    assert np.array_equal(np.asarray(source), before)
    assert np.array_equal(
        np.asarray(darker),
        np.asarray(ROBUSTNESS.apply_condition(source, "brightness_0.9")),
    )


def test_contrast_conditions_are_identity_on_uniform_input():
    source = uniform_image(100)
    for condition in ("contrast_0.9", "contrast_1.1"):
        assert np.array_equal(
            np.asarray(ROBUSTNESS.apply_condition(source, condition)),
            np.asarray(source),
        )


def test_jpeg_condition_round_trips_to_rgb_in_memory():
    source = uniform_image()
    result = ROBUSTNESS.apply_condition(source, "jpeg_quality_85")
    assert result.mode == "RGB"
    assert result.size == source.size
    np.testing.assert_allclose(
        np.asarray(result).mean(), np.asarray(source).mean(), atol=1
    )


def test_gamma_control_darkens_midrange_pixels():
    result = ROBUSTNESS.apply_condition(uniform_image(128), "gamma_dark")
    assert np.asarray(result).mean() < 128


def test_mask_conditions_cover_only_the_precommitted_regions():
    source = uniform_image(128)
    border = np.asarray(ROBUSTNESS.apply_condition(source, "border_black"))
    central = np.asarray(ROBUSTNESS.apply_condition(source, "central_black"))
    assert np.all(border[:32] == 0)
    assert np.all(border[-32:] == 0)
    assert np.all(border[:, :32] == 0)
    assert np.all(border[:, -32:] == 0)
    assert np.all(border[32:-32, 32:-32] == 128)
    assert np.all(central[34:190, 34:190] == 0)
    assert np.all(central[:34] == 128)
    assert np.all(central[:, :34] == 128)
    assert np.all(np.asarray(source) == 128)


def test_summary_computes_descriptive_metrics_and_class_counts():
    rows = [
        {
            "condition": "brightness_0.9",
            "ground_truth": "NORMAL",
            "absolute_probability_change": 0.2,
            "prediction_flip": 1,
        },
        {
            "condition": "brightness_0.9",
            "ground_truth": "PNEUMONIA",
            "absolute_probability_change": 0.4,
            "prediction_flip": 0,
        },
    ]
    result = ROBUSTNESS.summarize_condition(rows, "brightness_0.9")
    assert result["n"] == 2
    assert result["n_flip"] == 1
    assert result["flip_rate"] == 0.5
    assert result["mean_abs_probability_change"] == pytest.approx(0.3)
    assert result["median_abs_probability_change"] == pytest.approx(0.3)
    assert result["by_ground_truth"]["NORMAL"]["n_flip"] == 1
    assert result["by_ground_truth"]["PNEUMONIA"]["n_flip"] == 0


def test_full_validation_outputs_are_complete_and_frozen():
    output = SCRIPT.parent
    summary_path = output / "validation_robustness_summary.json"
    results_path = output / "validation_robustness_results.csv"
    report_path = output / "validation_robustness_report.md"
    assert summary_path.is_file()
    assert results_path.is_file()
    assert report_path.is_file()
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_validation_images"] == 518
    assert summary["baseline_confusion"] == {
        "TN": 132,
        "FP": 2,
        "FN": 2,
        "TP": 382,
    }
    assert summary["model_state_unchanged"] is True
    assert summary["official_test_manifest_contents_or_images_read"] is False
    assert summary["calibration_manifest_contents_or_images_read"] is False
    assert all(
        result["n"] == 518
        for result in summary["results"].values()
    )
    assert "OOD mask stress tests" in report_path.read_text(encoding="utf-8")
