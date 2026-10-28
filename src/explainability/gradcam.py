"""
Hook-based Grad-CAM implementation for EfficientNetB0 binary classification.

Design decisions:
- Implemented from scratch using PyTorch forward/backward hooks.
  No reliance on pytorch_grad_cam or any external Grad-CAM library.
- Target layer: backbone.conv_head (Conv2d 320->1280, 1x1, output 7x7).
  This is the last spatial convolutional layer before global average pooling
  in timm EfficientNetB0, giving 7x7 activation maps at 1280 channels.
- Grad-CAM direction: gradient of the raw PNEUMONIA logit (index 0 of the
  single-head output, positive direction) with respect to conv_head activations.
- Normalisation: each heatmap is individually min-max normalised to [0,1].
  Zero-variance maps (all-zeros after ReLU) are kept as zeros rather than NaN.
- Upsample: bicubic interpolation back to the source image size.
- Thread-safety: a new GradCAM instance per call; hooks are removed in __exit__
  or manually via remove_hooks() to avoid stale hook state across batches.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_LAYER_ACCESSOR = "backbone.conv_head"
"""
Canonical name for the Grad-CAM target layer in EfficientNetB0Classifier.
Retrieve via: model.backbone.conv_head
"""


class GradCAM:
    """
    Grad-CAM implementation for a binary single-logit classifier.

    Usage (context manager, recommended):
        with GradCAM(model, model.backbone.conv_head) as gcam:
            heatmap = gcam.generate(image_tensor)

    Or manual:
        gcam = GradCAM(model, model.backbone.conv_head)
        heatmap = gcam.generate(image_tensor)
        gcam.remove_hooks()

    Args:
        model: EfficientNetB0Classifier in eval mode. Caller must ensure
               model.eval() is set before calling generate().
        target_layer: nn.Module to attach hooks to (backbone.conv_head).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _forward_hook(module, inp, out):
            self._activations = out.detach().clone()

        def _backward_hook(module, grad_in, grad_out):
            # grad_out[0]: gradient flowing back into the target layer output
            self._gradients = grad_out[0].detach().clone()

        self._hooks.append(
            self.target_layer.register_forward_hook(_forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_backward_hook)
        )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def generate(
        self,
        image_tensor: torch.Tensor,
        output_size: Tuple[int, int] = (224, 224),
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for the PNEUMONIA logit.

        Args:
            image_tensor: Float tensor of shape (1, 3, H, W), already
                          normalised with ImageNet mean/std, on model device.
            output_size: (H, W) to upsample the heatmap to. Should match the
                         original source image dimensions.

        Returns:
            heatmap: Float32 numpy array of shape (H, W), values in [0, 1].
                     Higher values indicate regions most relevant to the
                     requested class logit.

        Raises:
            RuntimeError: If hooks failed to capture activations or gradients.
        """
        cam = self._compute_native_cam(image_tensor, target_class)
        cam = self._resize_cam(cam, output_size)
        return self._normalize_cam(cam)

    def generate_with_native(
        self,
        image_tensor: torch.Tensor,
        output_size: Tuple[int, int] = (224, 224),
        target_class: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the native-grid CAM and the existing upsampled CAM in one pass.

        Both maps are independently min-max normalised. The native map retains
        the target layer's spatial dimensions; interpolation is only applied to
        the second map for display.
        """
        native_cam = self._compute_native_cam(image_tensor, target_class)
        native = self._normalize_cam(native_cam)
        upsampled = self._normalize_cam(self._resize_cam(native_cam, output_size))
        return native, upsampled

    def _compute_native_cam(
        self,
        image_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> torch.Tensor:
        if target_class not in (0, 1):
            raise ValueError(f"target_class must be 0 or 1, got {target_class}")
        if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
            raise ValueError(
                f"image_tensor must be shape (1, C, H, W), got {image_tensor.shape}"
            )

        # Reset state
        self._activations = None
        self._gradients = None

        # Ensure grad computation is enabled (even though model is in eval mode)
        self.model.zero_grad()

        # Forward pass — grad enabled so backward can propagate
        with torch.enable_grad():
            logit = self.model(image_tensor)          # (1, 1)
            score = logit[0, 0] if target_class == 1 else -logit[0, 0]
            score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Check that target_layer is in the forward graph."
            )

        # Grad-CAM: global average pool gradients over spatial dims
        # activations: (1, C, h, w), gradients: (1, C, h, w)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        return F.relu(cam)

    @staticmethod
    def _resize_cam(cam: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            cam,
            size=output_size,
            mode="bicubic",
            align_corners=False,
        )

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
        cam_np = cam.squeeze().cpu().numpy().astype(np.float32)
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def get_target_layer(model) -> nn.Module:
    """
    Return the documented Grad-CAM target layer from an EfficientNetB0Classifier.

    Returns:
        model.backbone.conv_head (Conv2d 320->1280 k=1, output 7x7 at input 224).

    Raises:
        AttributeError: If the model does not have the expected layer structure.
    """
    try:
        layer = model.backbone.conv_head
    except AttributeError as e:
        raise AttributeError(
            f"Could not access '{TARGET_LAYER_ACCESSOR}'. "
            f"Ensure model is EfficientNetB0Classifier from timm. Error: {e}"
        ) from e

    if not isinstance(layer, nn.Conv2d):
        raise TypeError(
            f"Expected backbone.conv_head to be nn.Conv2d, got {type(layer).__name__}"
        )
    return layer


def overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blend a normalised heatmap over a uint8 RGB image.

    Args:
        image_np: uint8 numpy array (H, W, 3), original image in RGB.
        heatmap: float32 numpy array (H, W) in [0, 1].
        alpha: Heatmap opacity (0=invisible, 1=fully opaque).
               Default 0.45 keeps underlying X-ray readable.
        colormap: matplotlib colormap name. 'jet' is conventional for Grad-CAM.

    Returns:
        overlay: uint8 numpy array (H, W, 3) — blended image.
    """
    import matplotlib.cm as cm

    import matplotlib
    try:
        cmap = matplotlib.colormaps[colormap]
    except AttributeError:
        cmap = matplotlib.cm.get_cmap(colormap)  # noqa: deprecated pre-3.9
    heatmap_rgb = cmap(heatmap)[..., :3]                      # (H, W, 3), float [0,1]
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
    img_float  = image_np.astype(np.float32)
    heat_float = heatmap_rgb.astype(np.float32)
    blended    = (1 - alpha) * img_float + alpha * heat_float
    return blended.clip(0, 255).astype(np.uint8)


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Convert a normalised image tensor back to uint8 RGB for display/saving.

    Args:
        tensor: Float tensor (1, 3, H, W) or (3, H, W), ImageNet-normalised.
        mean: Per-channel mean used during normalisation.
        std: Per-channel std used during normalisation.

    Returns:
        uint8 numpy array (H, W, 3) in RGB.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    t = tensor.cpu().clone().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.clamp(0.0, 1.0)
    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img
