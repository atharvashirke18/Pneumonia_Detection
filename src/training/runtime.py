"""Runtime provenance and one-epoch train/evaluation execution."""

import platform
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


def get_environment_metadata(model: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Collect runtime environment and model/weight provenance metadata."""
    cuda_avail = torch.cuda.is_available()
    meta = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda if cuda_avail else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_avail else None,
        "timm_version": getattr(timm, "__version__", "unknown"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    if cuda_avail:
        meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        meta["cuda_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    else:
        meta["cuda_device_name"] = None
        meta["cuda_total_memory_gb"] = None

    if model is not None:
        meta["model_architecture"] = getattr(model, "ARCHITECTURE", model.__class__.__name__)
        meta["num_classes"] = getattr(model, "NUM_CLASSES", 1)
        if hasattr(model, "parameter_counts"):
            meta["parameters"] = model.parameter_counts()
        if hasattr(model, "backbone"):
            backbone = model.backbone
            cfg = getattr(backbone, "pretrained_cfg", getattr(backbone, "default_cfg", {}))
            meta["pretrained_cfg"] = {
                key: value if isinstance(value, (str, int, float, bool, list, tuple, dict))
                or value is None else str(value)
                for key, value in cfg.items()
            }
    return meta


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    is_training: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run one forward pass over a loader, with optional training updates."""
    model.train(is_training)
    total_loss = 0.0
    all_logits: list = []
    all_targets: list = []
    autocast_dtype = torch.float16 if scaler is not None else None

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)

            if autocast_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(images).squeeze(1)
                    loss = criterion(logits, labels)
            else:
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu().float().numpy())
            all_targets.append(labels.detach().cpu().numpy().astype(int))

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, np.concatenate(all_logits), np.concatenate(all_targets)
