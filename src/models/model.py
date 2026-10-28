"""
EfficientNetB0 binary classification model for Pneumonia Detection.

Design decisions:
- Wraps timm EfficientNetB0 with ImageNet pretrained weights.
- Outputs a single raw logit (num_classes=1) for BCEWithLogitsLoss.
  Using a single logit rather than two softmax classes is standard for
  binary classification and enables threshold calibration more naturally.
- Dropout is applied via timm's built-in drop_rate argument to avoid
  double-stacking dropout over the already-present timm classifier head.
- The classifier head is fully replaced (not only partially frozen) so
  that fine-tuning of the head is always explicit and auditable.
- Backbone parameters can be optionally frozen for feature-extraction-only
  experiments, but are unfrozen (all params updated) by default.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required. Install with: pip install timm"
    ) from e


class EfficientNetB0Classifier(nn.Module):
    """
    Binary EfficientNetB0 classifier with a single logit output.

    Output contract:
        forward(x) -> Tensor of shape (batch, 1), dtype float32.
        Intended for use with nn.BCEWithLogitsLoss.
        Apply torch.sigmoid() at inference to obtain probabilities.

    Args:
        pretrained: Whether to load ImageNet pretrained backbone weights.
        dropout_rate: Dropout probability applied inside the timm classifier head.
        freeze_backbone: If True, freezes all backbone parameters except
            the final classifier, useful for feature-extraction experiments.
        in_chans: Number of input image channels (must be 3 for pretrained weights).
    """

    ARCHITECTURE = "efficientnet_b0"
    NUM_CLASSES = 1  # Single logit for BCEWithLogitsLoss

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        in_chans: int = 3,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            self.ARCHITECTURE,
            pretrained=pretrained,
            num_classes=self.NUM_CLASSES,
            drop_rate=dropout_rate,
            in_chans=in_chans,
        )

        if freeze_backbone:
            self._freeze_backbone_layers()

        # Record construction parameters for reproducibility logging
        self._pretrained = pretrained
        self._dropout_rate = dropout_rate
        self._freeze_backbone = freeze_backbone

    def _freeze_backbone_layers(self) -> None:
        """Freeze all parameters except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (undo freeze_backbone)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (batch, 3, H, W), already normalized.
        Returns:
            Tensor of shape (batch, 1) containing raw logits.
        """
        return self.backbone(x)

    def get_classifier(self) -> nn.Module:
        """Return the timm classifier head module."""
        return self.backbone.classifier

    def get_feature_info(self) -> Any:
        """Return timm feature_info for Grad-CAM target layer selection."""
        return self.backbone.feature_info

    def parameter_counts(self) -> Dict[str, int]:
        """Return trainable vs. frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def __repr__(self) -> str:
        counts = self.parameter_counts()
        return (
            f"EfficientNetB0Classifier("
            f"pretrained={self._pretrained}, "
            f"dropout_rate={self._dropout_rate}, "
            f"freeze_backbone={self._freeze_backbone}, "
            f"params_total={counts['total']:,}, "
            f"params_trainable={counts['trainable']:,})"
        )


def build_model(model_cfg: Dict[str, Any]) -> EfficientNetB0Classifier:
    """
    Construct an EfficientNetB0Classifier from the 'model' section of config.yaml.

    Args:
        model_cfg: Dict with keys: pretrained, dropout_rate, in_chans.
            Unrecognised keys are ignored with a warning to support forward
            compatibility as config.yaml evolves.

    Returns:
        EfficientNetB0Classifier ready for training or inference.

    Raises:
        ValueError: If architecture key is present and is not 'efficientnet_b0'.
    """
    arch = model_cfg.get("architecture", EfficientNetB0Classifier.ARCHITECTURE)
    if arch != EfficientNetB0Classifier.ARCHITECTURE:
        raise ValueError(
            f"build_model only supports architecture='efficientnet_b0'; got '{arch}'. "
            "Implement a separate model builder for other architectures."
        )

    return EfficientNetB0Classifier(
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.2)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        in_chans=int(model_cfg.get("in_chans", 3)),
    )
