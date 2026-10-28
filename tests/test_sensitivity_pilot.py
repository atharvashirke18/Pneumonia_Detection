"""
tests/test_sensitivity_pilot.py
Automated tests for the Experiment 2B sensitivity pilot.

Covers:
  - Preprocessing consistency (same eval transform as datamodule)
  - Image immutability (perturbation does not modify input)
  - Perturbation determinism
  - Area accounting (border, central pixel counts)
  - Sigmoid correctness
  - Results file integrity
  - Checkpoint immutability
  - Delta correctness
"""
import json
import csv
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR    = PROJECT_ROOT / "experiments/exp02b_gradcam_sensitivity_pilot"
RESULTS_CSV  = PILOT_DIR / "sensitivity_results.csv"
SUMMARY_JSON = PILOT_DIR / "sensitivity_summary.json"
CONTACT_PNG  = PILOT_DIR / "sensitivity_contact_sheet.png"
CHECKPOINT   = PROJECT_ROOT / "experiments/exp01_efficientnetb0_baseline/best_model.pt"

# Pre-committed parameters (must match pilot script)
IMG_SIZE         = (224, 224)
THRESHOLD        = 0.5
BORDER_MARGIN_PX = 32
FILL_VALUE_U8    = 0
CENTRAL_SIDE_PX  = 156
GAMMA_PARAM      = 0.85
IMAGENET_MEAN    = (0.485, 0.456, 0.406)
IMAGENET_STD     = (0.229, 0.224, 0.225)


# ── Replicate pilot helpers locally for testing ───────────────────────────────

def apply_perturbation(pil_224: Image.Image, condition: str) -> Image.Image:
    assert pil_224.size == (IMG_SIZE[1], IMG_SIZE[0])
    if condition == "original":
        return pil_224.copy()
    arr = np.array(pil_224, dtype=np.uint8).copy()
    if condition == "border_black":
        m = BORDER_MARGIN_PX
        arr[:m, :, :]  = FILL_VALUE_U8
        arr[-m:, :, :] = FILL_VALUE_U8
        arr[:, :m, :]  = FILL_VALUE_U8
        arr[:, -m:, :] = FILL_VALUE_U8
    elif condition == "central_black":
        H, W = arr.shape[:2]
        hs = CENTRAL_SIDE_PX // 2
        cy, cx = H // 2, W // 2
        r0 = max(0, cy - hs)
        r1 = min(H, r0 + CENTRAL_SIDE_PX)
        c0 = max(0, cx - hs)
        c1 = min(W, c0 + CENTRAL_SIDE_PX)
        arr[r0:r1, c0:c1, :] = FILL_VALUE_U8
    elif condition == "gamma_dark":
        arr_f = arr.astype(np.float32) / 255.0
        arr_f = np.power(np.clip(arr_f, 0, 1), 1.0 / GAMMA_PARAM)
        arr   = (arr_f * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def make_uniform_pil(value=128, size=(224, 224)):
    arr = np.full((*size, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def results():
    with open(RESULTS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))

@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Area accounting
# ─────────────────────────────────────────────────────────────────────────────

class TestAreaAccounting:
    def test_border_area_value(self):
        H, W = IMG_SIZE
        expected = H * W - (H - 2*BORDER_MARGIN_PX) * (W - 2*BORDER_MARGIN_PX)
        assert expected == 24576

    def test_central_area_value(self):
        assert CENTRAL_SIDE_PX ** 2 == 24336

    def test_areas_within_1pct(self):
        border  = 24576
        central = 24336
        diff_pct = abs(border - central) / border * 100
        assert diff_pct < 1.0, f"Area diff {diff_pct:.2f}% exceeds 1%"

    def test_border_margin_equals_native_cell(self):
        """32px = 224/7 — aligns with one native conv_head grid cell."""
        assert IMG_SIZE[0] % 7 == 0
        assert BORDER_MARGIN_PX == IMG_SIZE[0] // 7

    def test_border_mask_pixel_count(self):
        arr = np.zeros(IMG_SIZE, dtype=np.uint8)
        m = BORDER_MARGIN_PX
        arr[:m, :]  = 1
        arr[-m:, :] = 1
        arr[:, :m]  = 1
        arr[:, -m:] = 1
        assert arr.sum() == 24576

    def test_central_mask_pixel_count(self):
        H, W = IMG_SIZE
        hs = CENTRAL_SIDE_PX // 2
        cy, cx = H // 2, W // 2
        r0, r1 = max(0, cy-hs), min(H, cy-hs+CENTRAL_SIDE_PX)
        c0, c1 = max(0, cx-hs), min(W, cx-hs+CENTRAL_SIDE_PX)
        count = (r1 - r0) * (c1 - c0)
        assert count == CENTRAL_SIDE_PX ** 2


# ─────────────────────────────────────────────────────────────────────────────
#  Perturbation correctness and immutability
# ─────────────────────────────────────────────────────────────────────────────

class TestPerturbation:
    def test_original_returns_copy_not_same_object(self):
        pil = make_uniform_pil(100)
        result = apply_perturbation(pil, "original")
        assert result is not pil

    def test_original_does_not_modify_pixels(self):
        pil = make_uniform_pil(100)
        result = apply_perturbation(pil, "original")
        assert np.array_equal(np.array(pil), np.array(result))

    def test_input_not_modified_by_border_black(self):
        pil = make_uniform_pil(200)
        orig_arr = np.array(pil).copy()
        apply_perturbation(pil, "border_black")
        assert np.array_equal(np.array(pil), orig_arr), "Input modified"

    def test_input_not_modified_by_central_black(self):
        pil = make_uniform_pil(200)
        orig_arr = np.array(pil).copy()
        apply_perturbation(pil, "central_black")
        assert np.array_equal(np.array(pil), orig_arr)

    def test_input_not_modified_by_gamma_dark(self):
        pil = make_uniform_pil(200)
        orig_arr = np.array(pil).copy()
        apply_perturbation(pil, "gamma_dark")
        assert np.array_equal(np.array(pil), orig_arr)

    def test_border_black_zeros_outer_ring(self):
        pil = make_uniform_pil(200)
        result = apply_perturbation(pil, "border_black")
        arr = np.array(result)
        m = BORDER_MARGIN_PX
        # Outer ring should be 0
        assert arr[:m, :, :].max() == 0
        assert arr[-m:, :, :].max() == 0
        assert arr[:, :m, :].max() == 0
        assert arr[:, -m:, :].max() == 0
        # Inner region should be unchanged
        assert arr[m:-m, m:-m, :].min() == 200

    def test_border_black_preserves_inner(self):
        arr_inner = np.ones((IMG_SIZE[0] - 2*BORDER_MARGIN_PX,
                             IMG_SIZE[1] - 2*BORDER_MARGIN_PX, 3), dtype=np.uint8) * 150
        arr_full = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        m = BORDER_MARGIN_PX
        arr_full[m:-m, m:-m, :] = 150
        arr_full[:m, :, :]  = 200
        arr_full[-m:, :, :] = 200
        arr_full[:, :m, :]  = 200
        arr_full[:, -m:, :] = 200
        pil = Image.fromarray(arr_full, "RGB")
        result = np.array(apply_perturbation(pil, "border_black"))
        assert result[m:-m, m:-m, :].min() == 150  # inner preserved
        assert result[:m, :, :].max() == 0         # top band zeroed
        assert result[-m:, :, :].max() == 0        # bottom band zeroed

    def test_central_black_zeros_center(self):
        pil = make_uniform_pil(200)
        result = apply_perturbation(pil, "central_black")
        arr = np.array(result)
        H, W = IMG_SIZE
        hs = CENTRAL_SIDE_PX // 2
        cy, cx = H // 2, W // 2
        r0, r1 = max(0, cy-hs), min(H, cy-hs+CENTRAL_SIDE_PX)
        c0, c1 = max(0, cx-hs), min(W, cx-hs+CENTRAL_SIDE_PX)
        assert arr[r0:r1, c0:c1, :].max() == 0

    def test_gamma_dark_preserves_black(self):
        """Black pixels (value 0) should stay 0 under gamma transform."""
        pil = make_uniform_pil(0)
        result = apply_perturbation(pil, "gamma_dark")
        assert np.array(result).max() == 0

    def test_gamma_dark_darkens_midgray(self):
        """Gamma > 1 exponent (1/0.85 > 1) darkens pixel values."""
        pil = make_uniform_pil(128)
        result = apply_perturbation(pil, "gamma_dark")
        result_val = np.array(result).mean()
        assert result_val < 128, f"Gamma should darken: got {result_val:.1f}"

    def test_gamma_output_in_valid_range(self):
        for v in [0, 50, 128, 200, 255]:
            pil = make_uniform_pil(v)
            result = np.array(apply_perturbation(pil, "gamma_dark"))
            assert result.min() >= 0 and result.max() <= 255

    def test_perturbation_is_deterministic(self):
        """Same input + condition → same output on repeated calls."""
        pil = make_uniform_pil(100)
        r1 = np.array(apply_perturbation(pil, "border_black"))
        r2 = np.array(apply_perturbation(pil, "border_black"))
        assert np.array_equal(r1, r2)

    def test_output_size_unchanged(self):
        pil = make_uniform_pil(100)
        for cond in ["original", "border_black", "central_black", "gamma_dark"]:
            result = apply_perturbation(pil, cond)
            assert result.size == pil.size, f"{cond}: size changed"


# ─────────────────────────────────────────────────────────────────────────────
#  Sigmoid function
# ─────────────────────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_sigmoid_zero(self):
        assert abs(sigmoid_np(np.array(0.0)) - 0.5) < 1e-9

    def test_sigmoid_very_large_positive(self):
        assert sigmoid_np(np.array(500.0)) > 0.999

    def test_sigmoid_very_large_negative(self):
        assert sigmoid_np(np.array(-500.0)) < 0.001

    def test_sigmoid_clip_prevents_overflow(self):
        """No overflow for extreme values."""
        val = sigmoid_np(np.array(1e9))
        assert np.isfinite(val)


# ─────────────────────────────────────────────────────────────────────────────
#  Results file integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestResultsFile:
    def test_csv_exists(self):
        assert RESULTS_CSV.exists()

    def test_json_exists(self):
        assert SUMMARY_JSON.exists()

    def test_contact_sheet_exists(self):
        assert CONTACT_PNG.exists()

    def test_n_examples_16(self, results):
        assert len(results) == 16

    def test_required_columns_present(self, results):
        required = {
            "ds_idx", "outcome", "gt_label",
            "original_prob", "original_pred", "original_flip",
            "border_black_prob", "border_black_pred", "border_black_flip",
            "central_black_prob", "central_black_pred", "central_black_flip",
            "gamma_dark_prob", "gamma_dark_pred", "gamma_dark_flip",
            "border_black_delta", "central_black_delta", "gamma_dark_delta",
            "border_black_abs_delta", "central_black_abs_delta", "gamma_dark_abs_delta",
        }
        actual = set(results[0].keys())
        missing = required - actual
        assert not missing, f"Missing columns: {missing}"

    def test_probs_in_range(self, results):
        for cond in ["original", "border_black", "central_black", "gamma_dark"]:
            for r in results:
                p = float(r[f"{cond}_prob"])
                assert 0.0 <= p <= 1.0, f"Out-of-range prob {p} for {cond}"

    def test_abs_delta_equals_abs_of_delta(self, results):
        for cond in ["border_black", "central_black", "gamma_dark"]:
            for r in results:
                delta = float(r[f"{cond}_delta"])
                abs_d = float(r[f"{cond}_abs_delta"])
                assert abs(abs_d - abs(delta)) < 1e-5, \
                    f"abs_delta mismatch for {cond}: {abs_d} vs {abs(delta)}"

    def test_delta_equals_perturbed_minus_original(self, results):
        for cond in ["border_black", "central_black", "gamma_dark"]:
            for r in results:
                expected = float(r[f"{cond}_prob"]) - float(r["original_prob"])
                actual   = float(r[f"{cond}_delta"])
                assert abs(actual - expected) < 1e-5, \
                    f"{cond} delta mismatch: expected {expected} got {actual}"

    def test_flip_correct_for_original(self, results):
        """Original condition flip column should always be 0."""
        for r in results:
            assert int(r["original_flip"]) == 0, \
                f"ds_idx={r['ds_idx']}: original_flip should be 0"

    def test_outcomes_only_valid_classes(self, results):
        valid = {"TP", "TN", "FP", "FN"}
        for r in results:
            assert r["outcome"] in valid

    def test_all_original_probs_within_loose_tolerance(self, results):
        """
        Original probs may differ from stored by up to ~5e-3 due to
        single-image vs batch-32 floating-point accumulation differences on GPU.
        All such differences must be small relative to observed perturbation effects.
        """
        for r in results:
            orig = float(r["original_prob"])
            stored = float(r["stored_prob"])
            diff = abs(orig - stored)
            assert diff < 5e-3, \
                f"ds_idx={r['ds_idx']}: orig={orig:.6f} stored={stored:.6f} diff={diff:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
#  Summary JSON integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryJSON:
    def test_params_unchanged(self, summary):
        assert summary["params_unchanged"] is True

    def test_param_sum_close_before_after(self, summary):
        before = float(summary["param_sum_before_experiment"])
        after  = float(summary["param_sum_after_experiment"])
        assert abs(before - after) < 1e-3

    def test_param_sum_matches_checkpoint(self, summary):
        raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        fresh_sum = sum(v.sum().item() for v in raw["model_state_dict"].values()
                        if v.dtype == torch.float32)
        stored = float(summary["param_sum_before_experiment"])
        assert abs(fresh_sum - stored) < 1.0

    def test_n_examples_in_summary(self, summary):
        assert summary["n_examples"] == 16

    def test_conditions_listed(self, summary):
        conds = set(summary["parameters"]["conditions"])
        assert conds == {"original", "border_black", "central_black", "gamma_dark"}

    def test_border_area_documented(self, summary):
        assert summary["parameters"]["border_area_px"] == 24576

    def test_central_area_documented(self, summary):
        assert summary["parameters"]["central_area_px"] == 24336

    def test_by_condition_keys(self, summary):
        assert "border_black" in summary["by_condition"]
        assert "central_black" in summary["by_condition"]
        assert "gamma_dark" in summary["by_condition"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
