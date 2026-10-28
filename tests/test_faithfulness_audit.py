from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "experiments"
    / "exp05_explanation_faithfulness"
    / "run_faithfulness_audit.py"
)
SPEC = importlib.util.spec_from_file_location("faithfulness_audit", SCRIPT)
FAITHFULNESS = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(FAITHFULNESS)


def test_grid_mask_has_expected_coverage_and_determinism():
    order = np.arange(49)
    mask = FAITHFULNESS.grid_mask(order, 0.2)
    assert mask.shape == (7, 7)
    assert mask.sum() == 10
    np.testing.assert_array_equal(mask, FAITHFULNESS.grid_mask(order, 0.2))


def test_curve_auc_is_normalized_over_fraction_interval():
    curves = np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0]])
    original_fractions = FAITHFULNESS.CURVE_FRACTIONS
    FAITHFULNESS.CURVE_FRACTIONS = (0.0, 0.5, 1.0)
    try:
        np.testing.assert_allclose(
            FAITHFULNESS.normalized_auc(curves), [0.5, 1.0]
        )
    finally:
        FAITHFULNESS.CURVE_FRACTIONS = original_fractions


def test_deletion_advantage_direction_is_random_minus_gradcam():
    rng = np.random.default_rng(7)
    result = FAITHFULNESS.summarize(
        cam_auc=np.array([0.2, 0.3]),
        random_auc=np.array([0.6, 0.5]),
        class_names=["NORMAL", "PNEUMONIA"],
        rng=rng,
        advantage_direction="random_minus_cam",
    )
    assert result["paired_mean_advantage"] == pytest.approx(0.3)
    assert result["NORMAL"]["paired_mean_advantage"] == pytest.approx(0.4)


def test_insertion_advantage_direction_is_gradcam_minus_random():
    result = FAITHFULNESS.summarize(
        cam_auc=np.array([0.8, 0.7]),
        random_auc=np.array([0.6, 0.5]),
        class_names=["NORMAL", "PNEUMONIA"],
        rng=np.random.default_rng(9),
        advantage_direction="cam_minus_random",
    )
    assert result["paired_mean_advantage"] == pytest.approx(0.2)
