"""Factories for baseline loss, optimizer, and learning-rate scheduler."""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

log = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build the baseline AdamW optimizer using trainable parameters only."""
    opt_name = training_cfg.get("optimizer", "adamw").lower()
    if opt_name != "adamw":
        raise ValueError(f"Only 'adamw' is supported in this baseline. Got '{opt_name}'.")

    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: Dict[str, Any],
    steps_per_epoch: Optional[int] = None,
) -> Any:
    """Build the epoch-level cosine scheduler with optional linear warmup."""
    sched_name = training_cfg.get("lr_scheduler", "cosine").lower()
    if sched_name != "cosine":
        raise ValueError(f"Only 'cosine' scheduler is supported. Got '{sched_name}'.")

    total_epochs = int(training_cfg.get("epochs", 15))
    min_lr = float(training_cfg.get("min_lr", 1e-6))
    warmup_epochs = int(training_cfg.get("warmup_epochs", 1))
    if warmup_epochs > 0 and warmup_epochs < total_epochs:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)


def build_loss_fn(
    training_cfg: Dict[str, Any],
    train_dataset: Optional[Any] = None,
) -> nn.BCEWithLogitsLoss:
    """Build BCEWithLogitsLoss, optionally deriving positive weight from data."""
    loss_cfg = training_cfg.get("loss", {})
    pw_value = loss_cfg.get("pos_weight", 1.0)
    pos_weight_tensor: Optional[torch.Tensor] = None

    if isinstance(pw_value, str) and pw_value.lower() == "auto":
        if train_dataset is not None and hasattr(train_dataset, "get_class_counts"):
            counts = train_dataset.get_class_counts()
            n_neg = counts.get("NORMAL", 1)
            n_pos = counts.get("PNEUMONIA", 1)
            auto_pw = n_neg / max(n_pos, 1)
            log.info(
                "Auto pos_weight computed: %.4f (NORMAL=%d, PNEUMONIA=%d)",
                auto_pw,
                n_neg,
                n_pos,
            )
            pos_weight_tensor = torch.tensor([auto_pw], dtype=torch.float32)
        else:
            log.warning(
                "pos_weight='auto' requested but no train_dataset provided; "
                "defaulting to 1.0"
            )
    elif isinstance(pw_value, (int, float)):
        if float(pw_value) != 1.0:
            pos_weight_tensor = torch.tensor([float(pw_value)], dtype=torch.float32)
            log.info("Using configured pos_weight=%.4f", float(pw_value))
    else:
        log.warning(
            "Unrecognized pos_weight value '%s'; defaulting to 1.0 (no weighting)",
            pw_value,
        )

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
