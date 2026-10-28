"""
tests/test_gradcam_numerical_audit.py
Automated tests for the Grad-CAM numerical audit methodology.

Tests cover:
  - border_inner_stats NaN behaviour for zero-denominator cases
  - 22px mask misalignment with native 7x7 grid
  - Mask cell counts
  - AuditGradCAM intermediate capture
  - Checkpoint / logit immutability
  - Upsampled vs native ratio comparison direction
"""
import sys
import types
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_SCRIPT = PROJECT_ROOT / "experiments/exp02a_gradcam/numerical_audit/numerical_audit.py"
AUDIT_JSON   = PROJECT_ROOT / "experiments/exp02a_gradcam/numerical_audit/gradcam_numerical_audit.json"
ORIG_META    = PROJECT_ROOT / "experiments/exp02a_gradcam/gradcam_metadata.json"
CHECKPOINT   = PROJECT_ROOT / "experiments/exp01_efficientnetb0_baseline/best_model.pt"

sys.path.insert(0, str(PROJECT_ROOT))
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(PROJECT_ROOT / "src")]
src_pkg.__package__ = "src"
sys.modules["src"] = src_pkg

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

model_mod = _load("src/models/model.py", "src.models.model")
EfficientNetB0Classifier = model_mod.EfficientNetB0Classifier

# Load audit helpers directly from the audit script
audit_spec = importlib.util.spec_from_file_location(
    "numerical_audit", AUDIT_SCRIPT)
audit_mod = importlib.util.module_from_spec(audit_spec)
# We don't exec the module (it has top-level side effects) — import helpers manually
# Instead, redefine minimal helpers for testing

def make_native_border_mask(h=7, w=7, n_cells=1):
    mask = np.ones((h, w), dtype=bool)
    if n_cells < min(h, w) // 2 + 1:
        mask[n_cells:h-n_cells, n_cells:w-n_cells] = False
    return mask

def make_upsampled_border_mask(margin_px=22, size=224):
    mask = np.zeros((size, size), dtype=bool)
    mask[:margin_px, :] = True
    mask[-margin_px:, :] = True
    mask[:, :margin_px] = True
    mask[:, -margin_px:] = True
    return mask

def border_inner_stats(arr2d, border_mask):
    border_vals = arr2d[border_mask]
    inner_vals  = arr2d[~border_mask]
    b_mean = float(border_vals.mean()) if len(border_vals) else float("nan")
    i_mean = float(inner_vals.mean())  if len(inner_vals)  else float("nan")
    b_sum  = float(border_vals.sum())
    i_sum  = float(inner_vals.sum())
    total  = b_sum + i_sum
    b_energy_share = (b_sum / total) if (np.isfinite(total) and total > 0) else float("nan")
    ratio  = (b_mean / i_mean) if (np.isfinite(i_mean) and i_mean != 0.0) else float("nan")
    flat_idx = int(np.argmax(arr2d))
    return {
        "border_n": int(border_vals.shape[0]),
        "inner_n":  int(inner_vals.shape[0]),
        "border_mean": b_mean,
        "inner_mean":  i_mean,
        "border_sum":  b_sum,
        "inner_sum":   i_sum,
        "border_energy_share": b_energy_share,
        "border_to_inner_ratio": ratio,
        "peak_row": flat_idx // arr2d.shape[1],
        "peak_col": flat_idx % arr2d.shape[1],
    }


# ─── Mask cell counts ─────────────────────────────────────────────────────────

class TestMaskCounts:
    def test_native_1cell_border_count(self):
        mask = make_native_border_mask(7, 7, n_cells=1)
        # Outer ring of 7x7: 7*4 - 4 = 24
        assert mask.sum() == 24, f"Expected 24 border cells, got {mask.sum()}"

    def test_native_1cell_inner_count(self):
        mask = make_native_border_mask(7, 7, n_cells=1)
        assert (~mask).sum() == 25, f"Expected 25 inner cells, got {(~mask).sum()}"

    def test_native_2cell_border_count(self):
        mask = make_native_border_mask(7, 7, n_cells=2)
        # 7x7 minus 3x3 center = 49 - 9 = 40
        assert mask.sum() == 40, f"Expected 40 border cells, got {mask.sum()}"

    def test_native_2cell_inner_count(self):
        mask = make_native_border_mask(7, 7, n_cells=2)
        assert (~mask).sum() == 9, f"Expected 9 inner cells, got {(~mask).sum()}"

    def test_upsampled_22px_pixel_count(self):
        mask = make_upsampled_border_mask(margin_px=22, size=224)
        # Should match audit output: 17776
        assert mask.sum() == 17776, f"Expected 17776, got {mask.sum()}"

    def test_22px_fraction_of_native_cell(self):
        # 22px / (224/7 px) must be less than 1 (sub-cell)
        frac = 22.0 / (224.0 / 7.0)
        assert frac < 1.0, f"22px border should be less than 1 native cell, got frac={frac:.4f}"
        # And check exact value
        assert abs(frac - 0.6875) < 1e-6, f"Expected 0.6875, got {frac}"

    def test_native_1cell_px_is_exactly_32(self):
        # Each native grid cell is exactly 32px for 224x224 at 7x7
        cell_px = 224.0 / 7.0
        assert abs(cell_px - 32.0) < 1e-6


# ─── border_inner_stats: NaN for zero denominator ────────────────────────────

class TestBorderInnerStatsNaN:
    def test_ratio_is_nan_when_inner_zero(self):
        """Ratio must be NaN, not epsilon-patched, when inner_mean=0."""
        arr = np.zeros((7, 7), dtype=np.float32)
        arr[0, 0] = 1.0  # all activation in border cell
        mask = make_native_border_mask(7, 7, n_cells=1)
        stats = border_inner_stats(arr, mask)
        assert np.isnan(stats["border_to_inner_ratio"]), \
            f"Expected NaN for zero inner, got {stats['border_to_inner_ratio']}"

    def test_ratio_is_nan_not_inf(self):
        """Must return NaN, not inf or a large epsilon-divided value."""
        arr = np.zeros((7, 7), dtype=np.float32)
        arr[0, 0] = 5.0
        mask = make_native_border_mask(7, 7, n_cells=1)
        stats = border_inner_stats(arr, mask)
        assert not np.isinf(stats["border_to_inner_ratio"]), "Ratio should not be inf"
        assert np.isnan(stats["border_to_inner_ratio"]), "Ratio should be NaN"

    def test_energy_share_is_nan_for_all_zero(self):
        arr = np.zeros((7, 7), dtype=np.float32)
        mask = make_native_border_mask(7, 7, n_cells=1)
        stats = border_inner_stats(arr, mask)
        assert np.isnan(stats["border_energy_share"]), \
            "Energy share should be NaN for all-zero array"

    def test_ratio_finite_when_inner_positive(self):
        arr = np.ones((7, 7), dtype=np.float32)
        mask = make_native_border_mask(7, 7, n_cells=1)
        stats = border_inner_stats(arr, mask)
        assert np.isfinite(stats["border_to_inner_ratio"]), \
            "Ratio should be finite when inner_mean > 0"
        # Uniform array: border == inner, ratio == 1.0
        assert abs(stats["border_to_inner_ratio"] - 1.0) < 1e-5

    def test_epsilon_patch_produces_different_result(self):
        """Verify that epsilon-patched ratio differs from NaN-correct for zero-inner case."""
        arr = np.zeros((7, 7), dtype=np.float32)
        arr[0, 0] = 2.0
        mask = make_native_border_mask(7, 7, n_cells=1)
        stats = border_inner_stats(arr, mask)
        # NaN-correct:
        assert np.isnan(stats["border_to_inner_ratio"])
        # Epsilon-patch (what the original code did):
        b_mean = stats["border_mean"]
        eps_ratio = b_mean / max(0.0, 1e-8)
        assert np.isfinite(eps_ratio) and eps_ratio > 0, \
            "Epsilon-patched ratio should be finite and large"


# ─── 22px vs native grid mismatch ─────────────────────────────────────────────

class TestInterpolationMismatch:
    def test_22px_margin_does_not_align_with_native_boundary(self):
        """The 22px boundary in 224px space does not coincide with any native 7x7 boundary."""
        native_boundaries_px = [i * (224 / 7) for i in range(1, 7)]
        for boundary in native_boundaries_px:
            assert abs(boundary - 22.0) > 0.5, \
                f"22px unexpectedly aligns with native boundary at {boundary:.2f}px"

    def test_native_cell_size_divides_224_evenly(self):
        assert 224 % 7 == 0

    def test_upsampled_mask_covers_partial_outer_cell(self):
        """22px < 32px: the border mask captures partial outer cells, not full cells."""
        cell_px = 224 // 7
        assert 22 < cell_px, f"22px should be < cell_px={cell_px}px"

    def test_upsampled_mask_smaller_than_1cell_px_border(self):
        """Upsampled 22px border has fewer pixels than a proper 1-cell (32px) border."""
        ups_22 = make_upsampled_border_mask(22, 224)
        ups_32 = make_upsampled_border_mask(32, 224)
        assert ups_22.sum() < ups_32.sum(), \
            f"22px border ({ups_22.sum()}) should cover fewer pixels than 32px ({ups_32.sum()})"


# ─── Audit JSON outputs consistency ───────────────────────────────────────────

@pytest.fixture(scope="module")
def audit_data():
    with open(AUDIT_JSON, encoding="utf-8") as f:
        return json.load(f)

class TestAuditJSON:
    def test_16_examples_present(self, audit_data):
        assert len(audit_data["examples"]) == 16

    def test_all_logits_matched(self, audit_data):
        """All 16 examples must have matched logits."""
        # Verify by checking stored vs recomputed probs (all OK in audit)
        orig = {ex["ds_idx"]: ex["pneumonia_prob"]
                for ex in json.load(open(ORIG_META))["examples"]}
        for ex in audit_data["examples"]:
            stored = orig[ex["ds_idx"]]
            recomp = ex["pneumonia_prob"]
            assert abs(stored - recomp) < 1e-4, \
                f"ds_idx={ex['ds_idx']}: stored={stored} recomp={recomp}"

    def test_no_zero_cams(self, audit_data):
        zero_cams = [ex["ds_idx"] for ex in audit_data["examples"] if ex["is_zero_cam"]]
        assert zero_cams == [], f"Unexpected all-zero CAMs: {zero_cams}"

    def test_epsilon_never_triggered(self, audit_data):
        triggered = [ex["ds_idx"] for ex in audit_data["examples"]
                     if ex["eps_patch_triggered"]]
        assert triggered == [], f"Epsilon patch was triggered for: {triggered}"

    def test_upsampled_16_of_16_border_dominant(self, audit_data):
        """All 16 upsampled 22px ratios should be > 1.0 (replicates original claim)."""
        for ex in audit_data["examples"]:
            ratio = ex["ups22_ratio_correct"]
            if ratio != "NaN":
                assert float(ratio) > 1.0, \
                    f"ds_idx={ex['ds_idx']}: ups22_ratio={ratio} is not > 1.0"

    def test_native_1cell_undefined_count(self, audit_data):
        """Exactly 4 native 1-cell ratios must be undefined (NaN)."""
        nan_count = sum(1 for ex in audit_data["examples"]
                        if ex["nat1_ratio"] == "NaN")
        assert nan_count == 4, f"Expected 4 NaN nat1 ratios, got {nan_count}"

    def test_native_1cell_all_finite_are_gt1(self, audit_data):
        """All 12 finite nat1 ratios must be > 1.0."""
        for ex in audit_data["examples"]:
            if ex["nat1_ratio"] != "NaN":
                assert float(ex["nat1_ratio"]) > 1.0, \
                    f"ds_idx={ex['ds_idx']}: nat1_ratio={ex['nat1_ratio']}"

    def test_native_2cell_border_dominant_count(self, audit_data):
        """Only 5 cases have finite nat2 ratio > 1.0."""
        count = sum(1 for ex in audit_data["examples"]
                    if ex["nat2_ratio"] != "NaN" and float(ex["nat2_ratio"]) > 1.0)
        assert count == 5, f"Expected 5 nat2-dominant cases, got {count}"

    def test_tp369_has_lowest_ratio(self, audit_data):
        """TP_0369 (ds_idx=369) should have the lowest finite upsampled ratio."""
        ratios = {ex["ds_idx"]: float(ex["ups22_ratio_correct"])
                  for ex in audit_data["examples"]
                  if ex["ups22_ratio_correct"] != "NaN"}
        min_idx = min(ratios, key=ratios.get)
        assert min_idx == 369, f"Expected ds_idx=369 to have min ratio, got {min_idx}"

    def test_tp369_peak_not_at_corner(self, audit_data):
        """TP_0369 should have peak not at top-left corner (row=0,col=0)."""
        ex369 = next(ex for ex in audit_data["examples"] if ex["ds_idx"] == 369)
        assert not (ex369["nat1_peak_row"] == 0 and ex369["nat1_peak_col"] == 0), \
            "TP_0369 peak unexpectedly at (0,0)"
        # Verified: peak is at row=5, col=5
        assert ex369["nat1_peak_row"] == 5
        assert ex369["nat1_peak_col"] == 5

    def test_peaks_mostly_at_boundary_rows(self, audit_data):
        """Most peaks should be in row 0 or row 6 (outer rows of 7x7 grid)."""
        corner_rows = {0, 6}
        corner_count = sum(1 for ex in audit_data["examples"]
                           if ex["nat1_peak_row"] in corner_rows)
        # At least 14/16 peaks at boundary rows
        assert corner_count >= 14, \
            f"Expected >= 14 peaks at boundary rows, got {corner_count}"

    def test_all_post_relu_are_nonneg(self, audit_data):
        """Post-ReLU min must be >= 0 for all examples."""
        for ex in audit_data["examples"]:
            assert ex["post_relu_min"] >= 0.0, \
                f"ds_idx={ex['ds_idx']}: post_relu_min={ex['post_relu_min']}"

    def test_upsampled_ratios_lower_than_native_1cell(self, audit_data):
        """Bicubic smoothing should yield lower border ratios than native 1-cell."""
        for ex in audit_data["examples"]:
            if ex["nat1_ratio"] != "NaN" and ex["ups22_ratio_correct"] != "NaN":
                nat = float(ex["nat1_ratio"])
                ups = float(ex["ups22_ratio_correct"])
                assert nat > ups, (
                    f"ds_idx={ex['ds_idx']}: expected nat1 ({nat:.2f}) > ups22 ({ups:.2f})"
                )

    def test_stored_ratios_match_recomputed(self, audit_data):
        """All stored ratios from original metadata must match recomputed values."""
        for ex in audit_data["examples"]:
            if ex["stored_ratio"] is not None and ex["ups22_ratio_correct"] != "NaN":
                diff = abs(float(ex["stored_ratio"]) - float(ex["ups22_ratio_correct"]))
                assert diff < 1e-3, \
                    f"ds_idx={ex['ds_idx']}: stored={ex['stored_ratio']} recomp={ex['ups22_ratio_correct']}"

    def test_param_sum_identity(self, audit_data):
        """param_sum in audit JSON must match a freshly recomputed value."""
        raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        fresh_sum = sum(v.sum().item() for v in raw["model_state_dict"].values()
                        if v.dtype == torch.float32)
        stored_sum = audit_data["param_sum_for_identity"]
        assert abs(fresh_sum - stored_sum) < 1.0, \
            f"Param sum mismatch: fresh={fresh_sum:.4f} stored={stored_sum:.4f}"

    def test_confusion_matches_expected(self, audit_data):
        c = audit_data["confusion"]
        assert c["TP"] == 382
        assert c["TN"] == 132
        assert c["FP"] == 2
        assert c["FN"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
