"""High-level orchestration for the baseline training run."""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler

from src.models.model import EfficientNetB0Classifier
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.components import build_loss_fn, build_optimizer, build_scheduler
from src.training.metrics import EarlyStopping, compute_binary_metrics
from src.training.runtime import get_environment_metadata, run_epoch

log = logging.getLogger(__name__)

# Compatibility for callers that imported this private helper from trainer.py.
_run_epoch = run_epoch


class Trainer:
    """Run configured training/validation epochs and persist their artifacts."""

    def __init__(
        self,
        model: EfficientNetB0Classifier,
        data_module: Any,
        config: Dict[str, Any],
        checkpoint_dir: Optional[str | Path] = None,
        history_path: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.dm = data_module
        self.config = config
        self.training_cfg = config.get("training", {})
        self.system_cfg = config.get("system", {})

        if checkpoint_dir is None or checkpoint_dir == "checkpoints":
            checkpoint_dir = self.training_cfg.get(
                "output_dir", "experiments/exp01_efficientnetb0_baseline"
            )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_epoch_checkpoints = bool(
            self.training_cfg.get("save_epoch_checkpoints", False)
        )

        device_str = self.system_cfg.get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to CPU.")
            device_str = "cpu"
        self.device = torch.device(device_str)

        use_amp = (
            self.system_cfg.get("mixed_precision", "no") == "fp16"
            and self.device.type == "cuda"
        )
        self.scaler: Optional[GradScaler] = GradScaler() if use_amp else None

        self._set_seeds(config.get("project", {}).get("random_seed", 42))
        if history_path is None:
            history_path = self.checkpoint_dir / "training_history.json"
        self.history_path = Path(history_path)
        self.history: list = []

    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = self.system_cfg.get(
            "cudnn_deterministic", True
        )
        torch.backends.cudnn.benchmark = self.system_cfg.get("cudnn_benchmark", False)

    def save_metadata(self) -> None:
        """Save configuration snapshot and runtime environment provenance."""
        config_snapshot_path = self.checkpoint_dir / "config_snapshot.json"
        with open(config_snapshot_path, "w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=2)
        log.info("Saved configuration snapshot to %s", config_snapshot_path)

        env_meta_path = self.checkpoint_dir / "environment_metadata.json"
        with open(env_meta_path, "w", encoding="utf-8") as file:
            json.dump(get_environment_metadata(self.model), file, indent=2)
        log.info("Saved environment metadata to %s", env_meta_path)

    def fit(self) -> list:
        """Execute the training loop and return per-epoch metrics history."""
        self.save_metadata()
        self.dm.setup("fit")
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()

        criterion = build_loss_fn(self.training_cfg, self.dm.train_dataset).to(self.device)
        optimizer = build_optimizer(self.model, self.training_cfg)
        scheduler = build_scheduler(optimizer, self.training_cfg)
        es_cfg = self.training_cfg.get("early_stopping", {})
        early_stopper = EarlyStopping(
            patience=int(es_cfg.get("patience", 5)),
            mode=es_cfg.get("mode", "min"),
        )
        self.model.to(self.device)

        total_epochs = int(self.training_cfg.get("epochs", 15))
        best_val_loss = float("inf")
        best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        last_checkpoint_path = self.checkpoint_dir / "last_model.pt"
        log.info(
            "Training start | device=%s | AMP=%s | epochs=%d | patience=%d | out=%s",
            self.device,
            self.scaler is not None,
            total_epochs,
            es_cfg.get("patience", 5),
            self.checkpoint_dir,
        )

        last_epoch_record: Optional[Dict[str, Any]] = None
        last_epoch = 0
        for epoch in range(total_epochs):
            last_epoch = epoch
            epoch_start = time.time()
            train_loss, train_logits, train_targets = run_epoch(
                self.model,
                train_loader,
                criterion,
                self.device,
                optimizer,
                self.scaler,
                is_training=True,
            )
            train_metrics = compute_binary_metrics(train_logits, train_targets)

            val_loss, val_logits, val_targets = run_epoch(
                self.model,
                val_loader,
                criterion,
                self.device,
                None,
                None,
                is_training=False,
            )
            val_metrics = compute_binary_metrics(val_logits, val_targets)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_auc": round(train_metrics["auc"], 6),
                "train_sensitivity": round(train_metrics["sensitivity"], 6),
                "train_specificity": round(train_metrics["specificity"], 6),
                "train_f1": round(train_metrics["f1"], 6),
                "val_loss": round(val_loss, 6),
                "val_auc": round(val_metrics["auc"], 6),
                "val_sensitivity": round(val_metrics["sensitivity"], 6),
                "val_specificity": round(val_metrics["specificity"], 6),
                "val_f1": round(val_metrics["f1"], 6),
                "lr": round(current_lr, 8),
                "epoch_time_s": round(epoch_time, 2),
            }
            self.history.append(epoch_record)
            last_epoch_record = epoch_record

            log.info(
                "Epoch %02d/%02d | train_loss=%.4f auc=%.4f | "
                "val_loss=%.4f auc=%.4f sens=%.4f spec=%.4f f1=%.4f | "
                "lr=%.2e | %.1fs",
                epoch + 1,
                total_epochs,
                train_loss,
                train_metrics["auc"],
                val_loss,
                val_metrics["auc"],
                val_metrics["sensitivity"],
                val_metrics["specificity"],
                val_metrics["f1"],
                current_lr,
                epoch_time,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_checkpoint_path,
                    epoch,
                    self.model,
                    optimizer,
                    scheduler,
                    self.scaler,
                    epoch_record,
                )

            if self.save_epoch_checkpoints:
                save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    epoch,
                    self.model,
                    optimizer,
                    scheduler,
                    self.scaler,
                    epoch_record,
                )

            with open(self.history_path, "w", encoding="utf-8") as file:
                json.dump(self.history, file, indent=2)

            if early_stopper.step(val_loss):
                log.info(
                    "Early stopping triggered at epoch %d "
                    "(no val_loss improvement for %d epochs).",
                    epoch + 1,
                    es_cfg.get("patience", 5),
                )
                break

        if last_epoch_record is not None:
            save_checkpoint(
                last_checkpoint_path,
                last_epoch,
                self.model,
                optimizer,
                scheduler,
                self.scaler,
                last_epoch_record,
            )

        log.info(
            "Training complete. Best val_loss=%.4f saved to %s | Last model saved to %s",
            best_val_loss,
            best_checkpoint_path,
            last_checkpoint_path,
        )
        return self.history
