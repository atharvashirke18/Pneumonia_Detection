"""
tests/test_gradcam.py
Automated tests for Grad-CAM implementation and target layer selection.
"""
import sys
import types
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bypass src/__init__.py to avoid sklearn DLL block
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

model_mod  = _load("src/models/model.py", "src.models.model")
gcam_mod   = _load("src/explainability/gradcam.py", "src.explainability.gradcam")

EfficientNetB0Classifier = model_mod.EfficientNetB0Classifier
GradCAM         = gcam_mod.GradCAM
get_target_layer = gcam_mod.get_target_layer
overlay_heatmap  = gcam_mod.overlay_heatmap
denormalize_image = gcam_mod.denormalize_image
TARGET_LAYER_ACCESSOR = gcam_mod.TARGET_LAYER_ACCESSOR

import pytest
import torch
import numpy as np


@pytest.fixture(scope="module")
def model():
    m = EfficientNetB0Classifier(pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def target_layer(model):
    return get_target_layer(model)


@pytest.fixture(scope="module")
def dummy_input():
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


# ─── Target layer tests ───────────────────────────────────────────────────────

class TestTargetLayer:
    def test_get_target_layer_returns_conv2d(self, model):
        import torch.nn as nn
        layer = get_target_layer(model)
        assert isinstance(layer, nn.Conv2d), f"Expected Conv2d, got {type(layer)}"

    def test_target_layer_is_conv_head(self, model):
        layer = get_target_layer(model)
        assert layer is model.backbone.conv_head

    def test_target_layer_dimensions(self, model):
        layer = get_target_layer(model)
        assert layer.in_channels == 320
        assert layer.out_channels == 1280
        assert layer.kernel_size == (1, 1)

    def test_target_layer_output_shape(self, model, dummy_input):
        shapes = {}
        hook = model.backbone.conv_head.register_forward_hook(
            lambda m, i, o: shapes.__setitem__("conv_head", tuple(o.shape))
        )
        with torch.no_grad():
            model(dummy_input)
        hook.remove()
        assert shapes["conv_head"] == (1, 1280, 7, 7), \
            f"Unexpected conv_head output shape: {shapes['conv_head']}"

    def test_accessor_string_matches_actual(self, model):
        """TARGET_LAYER_ACCESSOR string must resolve to backbone.conv_head."""
        parts = TARGET_LAYER_ACCESSOR.split(".")
        layer = model
        for part in parts:
            layer = getattr(layer, part)
        assert layer is model.backbone.conv_head


# ─── GradCAM output tests ─────────────────────────────────────────────────────

class TestGradCAMOutput:
    def test_heatmap_shape(self, model, target_layer, dummy_input):
        with GradCAM(model, target_layer) as gcam:
            heatmap = gcam.generate(dummy_input, output_size=(224, 224))
        assert heatmap.shape == (224, 224), f"Unexpected shape: {heatmap.shape}"

    def test_native_and_upsampled_maps_are_returned_from_one_pass(
        self, model, target_layer, dummy_input
    ):
        with GradCAM(model, target_layer) as gcam:
            native, upsampled = gcam.generate_with_native(
                dummy_input, output_size=(224, 224)
            )
        assert native.shape == (7, 7)
        assert upsampled.shape == (224, 224)
        assert native.dtype == np.float32
        assert upsampled.dtype == np.float32
        assert np.isfinite(native).all()
        assert np.isfinite(upsampled).all()
        assert 0.0 <= native.min() <= native.max() <= 1.0
        assert 0.0 <= upsampled.min() <= upsampled.max() <= 1.0
        with GradCAM(model, target_layer) as gcam:
            existing_api = gcam.generate(dummy_input, output_size=(224, 224))
        np.testing.assert_array_equal(upsampled, existing_api)

    def test_heatmap_dtype(self, model, target_layer, dummy_input):
        with GradCAM(model, target_layer) as gcam:
            heatmap = gcam.generate(dummy_input)
        assert heatmap.dtype == np.float32

    def test_heatmap_range(self, model, target_layer, dummy_input):
        with GradCAM(model, target_layer) as gcam:
            heatmap = gcam.generate(dummy_input)
        assert heatmap.min() >= 0.0 - 1e-6, f"min={heatmap.min()}"
        assert heatmap.max() <= 1.0 + 1e-6, f"max={heatmap.max()}"

    def test_heatmap_is_finite(self, model, target_layer, dummy_input):
        with GradCAM(model, target_layer) as gcam:
            heatmap = gcam.generate(dummy_input)
        assert np.all(np.isfinite(heatmap)), "Heatmap contains NaN or Inf"

    def test_heatmap_not_all_zeros_real_input(self, model, target_layer):
        """Non-trivial input should produce a non-zero heatmap."""
        torch.manual_seed(0)
        x = torch.randn(1, 3, 224, 224)
        with GradCAM(model, target_layer) as gcam:
            heatmap = gcam.generate(x)
        assert heatmap.max() > 0, "Heatmap is all-zero for non-trivial input"

    def test_custom_output_size(self, model, target_layer, dummy_input):
        for size in [(128, 128), (256, 256), (224, 224)]:
            with GradCAM(model, target_layer) as gcam:
                heatmap = gcam.generate(dummy_input, output_size=size)
            assert heatmap.shape == size, f"Expected {size}, got {heatmap.shape}"

    def test_negative_class_target_and_invalid_class(self, model, target_layer, dummy_input):
        with GradCAM(model, target_layer) as gcam:
            native, _ = gcam.generate_with_native(
                dummy_input, target_class=0
            )
        assert native.shape == (7, 7)
        with GradCAM(model, target_layer) as gcam:
            with pytest.raises(ValueError, match="target_class"):
                gcam.generate(dummy_input, target_class=2)

    def test_hooks_removed_after_context(self, model, target_layer, dummy_input):
        gcam = GradCAM(model, target_layer)
        with gcam:
            _ = gcam.generate(dummy_input)
        assert len(gcam._hooks) == 0, "Hooks not removed after context exit"

    def test_deterministic_output(self, model, target_layer):
        """Same input must produce identical heatmaps (model.eval, no stochasticity)."""
        x = torch.ones(1, 3, 224, 224) * 0.5
        with GradCAM(model, target_layer) as gcam:
            h1 = gcam.generate(x)
        with GradCAM(model, target_layer) as gcam:
            h2 = gcam.generate(x)
        np.testing.assert_array_equal(h1, h2, err_msg="Grad-CAM is not deterministic")

    def test_wrong_batch_size_raises(self, model, target_layer):
        with pytest.raises(ValueError, match="shape"):
            x = torch.randn(2, 3, 224, 224)
            with GradCAM(model, target_layer) as gcam:
                gcam.generate(x)

    def test_model_weights_unchanged_after_gradcam(self, model, target_layer, dummy_input):
        """Grad-CAM must not modify model weights."""
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        with GradCAM(model, target_layer) as gcam:
            _ = gcam.generate(dummy_input)
        for name, param in model.named_parameters():
            torch.testing.assert_close(
                param, params_before[name],
                msg=f"Parameter '{name}' was modified by Grad-CAM"
            )


# ─── Overlay and denormalise utilities ────────────────────────────────────────

class TestUtilities:
    def test_overlay_shape(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = overlay_heatmap(img, heatmap)
        assert result.shape == (224, 224, 3)

    def test_overlay_dtype(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap = np.ones((224, 224), dtype=np.float32)
        result = overlay_heatmap(img, heatmap)
        assert result.dtype == np.uint8

    def test_overlay_range(self):
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = overlay_heatmap(img, heatmap)
        assert result.min() >= 0 and result.max() <= 255

    def test_denormalize_shape(self):
        t = torch.zeros(1, 3, 224, 224)
        result = denormalize_image(t)
        assert result.shape == (224, 224, 3)

    def test_denormalize_dtype(self):
        t = torch.zeros(1, 3, 224, 224)
        result = denormalize_image(t)
        assert result.dtype == np.uint8

    def test_denormalize_range(self):
        t = torch.randn(1, 3, 224, 224)
        result = denormalize_image(t)
        assert result.min() >= 0 and result.max() <= 255

    def test_denormalize_reconstructs_black_image(self):
        """A tensor filled with -mean/std should reconstruct to all-zero (black)."""
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        # Normalised value of 0 → (0 - mean) / std
        t = torch.zeros(1, 3, 224, 224)
        for c, (m, s) in enumerate(zip(mean, std)):
            t[0, c] = (0.0 - m) / s
        result = denormalize_image(t, mean=mean, std=std)
        assert result.max() <= 1, f"Expected near-zero reconstruction, got max={result.max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
