"""Grad-CAM generation and explanation-evaluation utilities."""

from src.explainability.evaluation import evaluate_native_cam_against_mask
from src.explainability.gradcam import GradCAM, get_target_layer, overlay_heatmap

__all__ = [
    "GradCAM",
    "evaluate_native_cam_against_mask",
    "get_target_layer",
    "overlay_heatmap",
]
