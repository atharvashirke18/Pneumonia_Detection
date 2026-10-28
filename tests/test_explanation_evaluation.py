from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.explainability.evaluation import (
    evaluate_native_cam_against_mask,
    mask_occupancy_on_cam_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "experiments"
    / "exp06_expert_localization"
    / "evaluate_localization.py"
)
SPEC = importlib.util.spec_from_file_location("expert_localization", SCRIPT)
LOCALIZATION = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(LOCALIZATION)


def test_native_grid_occupancy_preserves_partial_cells():
    mask = np.zeros((14, 14), dtype=np.uint8)
    mask[:7, :7] = 255
    occupancy = mask_occupancy_on_cam_grid(mask, (2, 2))
    np.testing.assert_array_equal(
        occupancy,
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_evaluation_rewards_peak_and_activation_within_mask():
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[2, 3] = 1
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[64:96, 96:128] = 255
    result = evaluate_native_cam_against_mask(cam, Image.fromarray(mask))
    assert result["pointing_occupancy_at_peak"] == pytest.approx(1.0)
    assert result["activation_fraction_in_mask"] == pytest.approx(1.0)
    assert result["mask_area_fraction"] == pytest.approx(1 / 49)


def test_empty_or_invalid_masks_are_rejected():
    with pytest.raises(ValueError, match="no positive pixels"):
        mask_occupancy_on_cam_grid(np.zeros((14, 14)), (7, 7))
    with pytest.raises(ValueError, match="finite"):
        evaluate_native_cam_against_mask(
            np.full((7, 7), np.nan), np.ones((224, 224))
        )


def test_all_zero_cam_does_not_report_a_fabricated_peak_hit():
    mask = np.ones((224, 224), dtype=np.uint8)
    result = evaluate_native_cam_against_mask(np.zeros((7, 7)), mask)
    assert result["pointing_occupancy_at_peak"] is None
    assert result["activation_fraction_in_mask"] == 0.0


def test_annotation_manifest_refuses_empty_template():
    template = SCRIPT.parent / "annotations_manifest_template.csv"
    with pytest.raises(ValueError, match="No expert annotations"):
        LOCALIZATION.load_annotations(template)


def test_annotation_manifest_requires_correct_schema(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("image_id,mask_path\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        LOCALIZATION.load_annotations(path)
