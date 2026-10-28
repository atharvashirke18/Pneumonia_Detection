"""Model checkpoint serialization and restoration."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import GradScaler

from src.models.model import EfficientNetB0Classifier

log = logging.getLogger(__name__)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: EfficientNetB0Classifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    metrics: Dict[str, float],
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """Save state dictionaries and run metadata to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "architecture": EfficientNetB0Classifier.ARCHITECTURE,
        "num_classes": EfficientNetB0Classifier.NUM_CLASSES,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if config_snapshot is not None:
        checkpoint["config_snapshot"] = config_snapshot

    torch.save(checkpoint, path)
    log.info("Checkpoint saved: %s (epoch=%d)", path, epoch)


def load_checkpoint(
    path: Path,
    model: EfficientNetB0Classifier,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model state and any supplied optimizer/scheduler/scaler state."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    log.info("Checkpoint loaded: %s (epoch=%d)", path, checkpoint.get("epoch", -1))
    return checkpoint
