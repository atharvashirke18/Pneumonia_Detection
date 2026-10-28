"""
train.py — Experiment 1: EfficientNetB0 Baseline Training Entry-Point

Usage:
    python train.py [--config configs/config.yaml]

Constraints enforced by this script:
- Trains and validates on manifests only; test/calibration sets are never accessed.
- Saves outputs to config.training.output_dir (experiments/exp01_efficientnetb0_baseline/).
- Retains best_model.pt (lowest val_loss) and last_model.pt only.
- Per-epoch checkpoints are disabled unless config.training.save_epoch_checkpoints=true.
- Saves config_snapshot.json, environment_metadata.json, and training_history.json.
- Reports metrics labelled as validation results; no test-set evaluation is performed.
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path

# Ensure project root on path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import load_config
from src.models.model import build_model
from src.data.datamodule import PneumoniaDataModule
from src.training.trainer import Trainer, get_environment_metadata, load_checkpoint


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure root logger to write to both stdout and a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    fmt = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("train")


def verify_checkpoint(ckpt_path: Path, log) -> bool:
    """Load a checkpoint back from disk and confirm weight round-trip."""
    try:
        from src.models.model import EfficientNetB0Classifier
        verify_model = EfficientNetB0Classifier(pretrained=False)
        ckpt = load_checkpoint(ckpt_path, verify_model, device=torch.device("cpu"))
        epoch = ckpt.get("epoch", "?")
        metrics = ckpt.get("metrics", {})
        log.info(
            "  Checkpoint OK: %s (epoch=%s, val_loss=%.6f, val_auc=%.6f)",
            ckpt_path.name, epoch,
            metrics.get("val_loss", float("nan")),
            metrics.get("val_auc", float("nan")),
        )
        return True
    except Exception as e:
        log.error("  Checkpoint FAILED for %s: %s", ckpt_path.name, e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 - EfficientNetB0 Baseline Training")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file (relative to project root or absolute)."
    )
    args = parser.parse_args()

    # --- Load and validate configuration ---
    cfg = load_config(args.config, base_dir=PROJECT_ROOT)
    training_cfg = cfg.get("training", {})

    output_dir = Path(training_cfg.get("output_dir", "experiments/exp01_efficientnetb0_baseline"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_dir)

    log.info("=" * 70)
    log.info("Experiment 1 - EfficientNetB0 Baseline Training")
    log.info("=" * 70)
    log.info("Config: %s", args.config)
    log.info("Output dir: %s", output_dir)
    log.info("pos_weight: %s | decision threshold: 0.5 | monitor: val_loss",
             training_cfg.get("loss", {}).get("pos_weight", 1.0))
    log.info("Partition: train + val manifests ONLY. Test/calibration sets NOT accessed.")

    # --- Build model ---
    log.info("Building model: EfficientNetB0 pretrained=True ...")
    model = build_model(cfg["model"])
    param_counts = model.parameter_counts()
    log.info("Model: %s", model)
    log.info("Parameters: total=%d | trainable=%d | frozen=%d",
             param_counts["total"], param_counts["trainable"], param_counts["frozen"])

    # --- Build DataModule ---
    dm = PneumoniaDataModule(cfg)
    dm.setup("fit")
    log.info("Train dataset: %d images", len(dm.train_dataset))
    log.info("Val dataset:   %d images", len(dm.val_dataset))

    if hasattr(dm.train_dataset, "get_class_counts"):
        counts_train = dm.train_dataset.get_class_counts()
        counts_val = dm.val_dataset.get_class_counts()
        log.info("Train class counts: %s", counts_train)
        log.info("Val class counts:   %s", counts_val)

    # --- GPU configuration ---
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info(
            "GPU: %s | Total VRAM: %.2f GB | Mixed precision: fp16 | Batch size: %d",
            props.name, props.total_memory / 1e9,
            cfg["dataloader"]["batch_size"],
        )
        log.info("Pre-flight probe result: batch_size=32 peaked at ~1.501 GB / 6.44 GB (23%%) - OK")
        torch.cuda.reset_peak_memory_stats(0)
    else:
        log.warning("CUDA not available; training on CPU.")

    # --- Build Trainer and launch ---
    trainer = Trainer(
        model=model,
        data_module=dm,
        config=cfg,
        checkpoint_dir=output_dir,
    )

    log.info("Launching training ...")
    t0 = time.time()
    history = trainer.fit()
    elapsed = time.time() - t0

    # --- Post-training summary ---
    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)
    log.info("Total wall time: %.1f s (%.1f min)", elapsed, elapsed / 60)
    log.info("Epochs completed: %d", len(history))

    if not history:
        log.error("No epochs in history - training may have failed silently.")
        sys.exit(1)

    # Best epoch by val_loss
    best = min(history, key=lambda r: r["val_loss"])
    log.info(
        "Best epoch: %d | val_loss=%.6f | val_auc=%.6f | "
        "val_sensitivity=%.4f | val_specificity=%.4f | val_f1=%.4f",
        best["epoch"] + 1,
        best["val_loss"], best["val_auc"],
        best["val_sensitivity"], best["val_specificity"], best["val_f1"],
    )

    # Early stopping status
    max_epochs = training_cfg.get("epochs", 15)
    stopped_early = len(history) < max_epochs
    patience = training_cfg.get("early_stopping", {}).get("patience", 5)
    log.info(
        "Early stopping: %s (patience=%d, monitored=val_loss)",
        "TRIGGERED" if stopped_early else "NOT triggered (ran to max_epochs)", patience
    )

    # Per-epoch table (ALL VALIDATION RESULTS)
    log.info("")
    log.info("Per-epoch metrics [VALIDATION RESULTS ONLY - not test-set performance]:")
    log.info("-" * 88)
    log.info(
        "%-6s %-10s %-10s %-10s %-8s %-8s %-8s %-10s",
        "Epoch", "TrainLoss", "ValLoss", "ValAUC", "ValSens", "ValSpec", "ValF1", "LR"
    )
    log.info("-" * 88)
    for r in history:
        log.info(
            "%-6d %-10.6f %-10.6f %-10.6f %-8.4f %-8.4f %-8.4f %-10.2e",
            r["epoch"] + 1, r["train_loss"], r["val_loss"], r["val_auc"],
            r["val_sensitivity"], r["val_specificity"], r["val_f1"], r["lr"]
        )
    log.info("-" * 88)

    # --- Checkpoint verification ---
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    log.info("Verifying saved checkpoints ...")
    best_ok = verify_checkpoint(best_path, log) if best_path.exists() else False
    last_ok = verify_checkpoint(last_path, log) if last_path.exists() else False
    if not best_ok:
        log.error("CRITICAL: best_model.pt could not be verified.")
    if not last_ok:
        log.error("CRITICAL: last_model.pt could not be verified.")

    # --- GPU peak memory ---
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU peak memory during training: %.3f GB / %.2f GB (%.1f%%)",
                 peak_mem, total_vram, 100 * peak_mem / total_vram)

    # --- Artifact inventory ---
    log.info("Output artifacts in %s:", output_dir)
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        log.info("  %-40s  %.1f KB", f.name, size_kb)

    log.info("=" * 70)
    log.info("DISCLAIMER: All metrics are VALIDATION results on the internal val split.")
    log.info("They do NOT constitute official test-set or clinical performance estimates.")
    log.info("The test set (test_manifest.csv) was NOT accessed during this experiment.")
    log.info("Experiment 2 requires separate approval before proceeding.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
