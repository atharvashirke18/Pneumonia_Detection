"""Post-hoc calibration and MC-dropout audit on the calibration split only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import ManifestDataset
from src.ui.inference import PneumoniaInference


OUTPUT_DIR = Path(__file__).resolve().parent
SPLIT_SEED = 42
FIT_FRACTION = 0.70
THRESHOLD = 0.5
MC_DROPOUT_PASSES = 50
ECE_BINS = 10
BATCH_SIZE = 32


def partition_calibration(
    samples: list[dict[str, Any]],
    fit_fraction: float = FIT_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Partition deterministically; keep available pneumonia patient groups intact."""
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    fit_indices: list[int] = []
    audit_indices: list[int] = []

    normal_indices = [
        index
        for index, sample in enumerate(samples)
        if sample["label"] == 0
    ]
    rng.shuffle(normal_indices)
    normal_fit_count = round(len(normal_indices) * fit_fraction)
    fit_indices.extend(normal_indices[:normal_fit_count])
    audit_indices.extend(normal_indices[normal_fit_count:])

    pneumonia_groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample["label"] != 1:
            continue
        token = sample.get("patient_token", "")
        if not token:
            raise ValueError(
                "A pneumonia calibration record is missing its patient token; "
                "patient-grouped partition cannot be guaranteed."
            )
        pneumonia_groups[token].append(index)

    groups = list(pneumonia_groups.values())
    rng.shuffle(groups)
    target_fit_images = round(
        sum(len(group) for group in groups) * fit_fraction
    )
    current_fit_images = 0
    for group in groups:
        if current_fit_images < target_fit_images:
            fit_indices.extend(group)
            current_fit_images += len(group)
        else:
            audit_indices.extend(group)

    fit_indices.sort()
    audit_indices.sort()
    if set(fit_indices) & set(audit_indices):
        raise RuntimeError("Calibration fit and audit rows overlap.")

    fit_tokens = {
        samples[index]["patient_token"]
        for index in fit_indices
        if samples[index]["label"] == 1
    }
    audit_tokens = {
        samples[index]["patient_token"]
        for index in audit_indices
        if samples[index]["label"] == 1
    }
    if fit_tokens & audit_tokens:
        raise RuntimeError("Pneumonia patient tokens cross calibration partitions.")
    if not fit_indices or not audit_indices:
        raise RuntimeError("Calibration partition produced an empty subset.")
    return fit_indices, audit_indices


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive scalar temperature by binary log-loss minimization."""
    logits = logits.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    labels = labels.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.binary_cross_entropy_with_logits(
            logits / temperature, labels
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature_value = float(log_temperature.detach().exp().item())
    if not math.isfinite(temperature_value) or temperature_value <= 0:
        raise RuntimeError("Temperature fitting produced an invalid value.")
    return temperature_value


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = ECE_BINS,
) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[dict[str, Any]] = []
    total = len(probabilities)
    weighted_error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        count = int(selected.sum())
        if count:
            mean_confidence = float(probabilities[selected].mean())
            fraction_positive = float(labels[selected].mean())
            weighted_error += (
                count / total
            ) * abs(mean_confidence - fraction_positive)
        else:
            mean_confidence = None
            fraction_positive = None
        result.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n": count,
                "mean_probability": mean_confidence,
                "observed_positive_fraction": fraction_positive,
            }
        )
    return weighted_error, result


def binary_probability_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    eps = np.finfo(np.float64).eps
    probabilities = np.clip(probabilities.astype(np.float64), eps, 1.0 - eps)
    labels = labels.astype(np.float64)
    predictions = probabilities >= THRESHOLD
    positives = labels == 1
    negatives = labels == 0
    ece, bins = expected_calibration_error(probabilities, labels)
    return {
        "n": int(len(labels)),
        "accuracy_at_0.5": float((predictions == positives).mean()),
        "sensitivity_at_0.5": (
            float(predictions[positives].mean()) if positives.any() else float("nan")
        ),
        "specificity_at_0.5": (
            float((~predictions[negatives]).mean()) if negatives.any() else float("nan")
        ),
        "negative_log_likelihood": float(
            -np.mean(
                labels * np.log(probabilities)
                + (1.0 - labels) * np.log(1.0 - probabilities)
            )
        ),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "expected_calibration_error_10_bins": float(ece),
        "reliability_bins": bins,
    }


def ranking_auc(error_labels: np.ndarray, scores: np.ndarray) -> float | None:
    error_labels = error_labels.astype(bool)
    positive_count = int(error_labels.sum())
    negative_count = int((~error_labels).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(ranks[error_labels].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def model_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _probabilities_for_indices(
    model: torch.nn.Module,
    tensors: torch.Tensor,
    indices: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    logit_batches = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start : start + BATCH_SIZE]
        batch = tensors[batch_indices].to(device)
        with torch.inference_mode():
            logit_batches.append(model(batch).reshape(-1).cpu())
    logits = torch.cat(logit_batches)
    return logits.numpy().astype(np.float64), torch.sigmoid(logits).numpy().astype(
        np.float64
    )


def mc_dropout_logits(
    model: torch.nn.Module,
    tensors: torch.Tensor,
    indices: list[int],
    device: torch.device,
    passes: int = MC_DROPOUT_PASSES,
    seed: int = SPLIT_SEED,
) -> np.ndarray:
    """Run stochastic head dropout while keeping the rest of the model in eval."""
    dropout_types = (
        nn.Dropout,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    dropouts = [
        module for module in model.modules() if isinstance(module, dropout_types)
    ]
    backbone = getattr(model, "backbone", None)
    functional_dropout_rate = float(getattr(backbone, "drop_rate", 0.0))
    if not dropouts and functional_dropout_rate <= 0.0:
        raise RuntimeError(
            "The loaded model contains no dropout layers; MC-dropout "
            "uncertainty cannot be estimated."
        )

    output = np.empty((passes, len(indices)), dtype=np.float64)
    model.eval()
    try:
        with torch.random.fork_rng(
            devices=[device] if device.type == "cuda" else []
        ):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model.train()
            for module in model.modules():
                if isinstance(
                    module,
                    (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.SyncBatchNorm,
                    ),
                ):
                    module.eval()
                elif type(module).__name__ in {"DropPath", "DropBlock2d"}:
                    module.eval()
            for module in dropouts:
                module.train()
            for pass_index in range(passes):
                logits, _ = _probabilities_for_indices(
                    model, tensors, indices, device
                )
                output[pass_index] = logits
    finally:
        model.eval()
    if not np.isfinite(output).all():
        raise RuntimeError("MC-dropout produced non-finite logits.")
    if float(output.var(axis=0).mean()) <= 0.0:
        raise RuntimeError(
            "MC-dropout samples are identical; stochastic dropout was not active."
        )
    return output


def _make_report(summary: dict[str, Any]) -> str:
    raw = summary["audit_metrics"]["raw"]
    scaled = summary["audit_metrics"]["temperature_scaled"]
    aucs = summary["uncertainty_error_ranking_auc"]
    lines = [
        "# Experiment 4 — Calibration and MC-dropout uncertainty audit",
        "",
        f"**Calibration split:** {summary['n_calibration_images']} images.",
        f"**Fit subset:** {summary['n_fit_images']} images; "
        f"**audit subset:** {summary['n_audit_images']} images.",
        "**Official test split:** not accessed.",
        f"**Temperature:** {summary['temperature']:.6f}, fit only on the "
        "calibration-fit subset.",
        "",
        "## Held-out internal audit subset",
        "",
        "| Metric | Raw baseline | Temperature-scaled |",
        "|---|---:|---:|",
        f"| NLL | {raw['negative_log_likelihood']:.6f} | {scaled['negative_log_likelihood']:.6f} |",
        f"| Brier score | {raw['brier_score']:.6f} | {scaled['brier_score']:.6f} |",
        f"| 10-bin ECE | {raw['expected_calibration_error_10_bins']:.6f} | {scaled['expected_calibration_error_10_bins']:.6f} |",
        f"| Accuracy at 0.5 | {raw['accuracy_at_0.5']:.4f} | {scaled['accuracy_at_0.5']:.4f} |",
        "",
        "## MC-dropout error ranking",
        "",
        f"50 stochastic dropout passes; calibrated pass probabilities use the "
        f"fit-subset temperature. Audit errors: {summary['n_audit_errors']} / "
        f"{summary['n_audit_images']}.",
        "",
        "| Uncertainty score | AUROC for ranking errors |",
        "|---|---:|",
    ]
    for name in ("predictive_entropy", "probability_variance", "mutual_information"):
        value = aucs[name]
        lines.append(
            f"| {name.replace('_', ' ')} | "
            f"{'Not defined (audit contains one outcome class)' if value is None else f'{value:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is a method-development audit on a single internal calibration "
            "partition. MC-dropout scores are approximate model uncertainty, "
            "not clinical certainty. The audit partition is not an external "
            "cohort; NORMAL patient identities are unavailable. ECE depends "
            "on binning and sample size. These results do not establish "
            "clinical calibration, generalization, or distribution-free "
            "coverage guarantees.",
            "",
            "No model weights were changed. No images were used for training. "
            "Per-example probabilities and uncertainty values are in "
            "`calibration_uncertainty_audit.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    config = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
    inference = PneumoniaInference.from_project(PROJECT_ROOT)
    model = inference.model
    state_before = model_state_digest(model)
    checkpoint_path = (
        PROJECT_ROOT / config["training"]["output_dir"] / "best_model.pt"
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    dataset = ManifestDataset(
        manifest_path=config["dataset"]["resolved_manifest_paths"]["calibration"],
        data_root=config["dataset"]["data_root_resolved"],
        expected_split="calibration",
        class_to_idx=config["dataset"]["classes"],
        transform=inference.transform,
    )
    fit_indices, audit_indices = partition_calibration(dataset.samples)
    tensors = torch.stack(
        [dataset[index][0] for index in range(len(dataset))]
    )
    labels = np.asarray(
        [dataset.samples[index]["label"] for index in range(len(dataset))],
        dtype=np.int64,
    )
    device = inference.device

    fit_logits_np, _ = _probabilities_for_indices(
        model, tensors, fit_indices, device
    )
    temperature = fit_temperature(
        torch.from_numpy(fit_logits_np),
        torch.from_numpy(labels[fit_indices]),
    )

    audit_logits_np, raw_probabilities = _probabilities_for_indices(
        model, tensors, audit_indices, device
    )
    audit_labels = labels[audit_indices]
    scaled_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(audit_logits_np / temperature, -500, 500))
    )
    mc_logits = mc_dropout_logits(
        model, tensors, audit_indices, device
    )
    calibrated_mc = 1.0 / (
        1.0 + np.exp(-np.clip(mc_logits / temperature, -500, 500))
    )
    mean_mc_probability = calibrated_mc.mean(axis=0)
    probability_variance = calibrated_mc.var(axis=0, ddof=1)
    clipped_mean = np.clip(
        mean_mc_probability,
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    predictive_entropy = -(
        clipped_mean * np.log(clipped_mean)
        + (1.0 - clipped_mean) * np.log(1.0 - clipped_mean)
    )
    clipped_passes = np.clip(
        calibrated_mc,
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    pass_entropy = -(
        clipped_passes * np.log(clipped_passes)
        + (1.0 - clipped_passes) * np.log(1.0 - clipped_passes)
    )
    mutual_information = predictive_entropy - pass_entropy.mean(axis=0)

    error_labels = (
        (mean_mc_probability >= THRESHOLD).astype(np.int64) != audit_labels
    )
    fit_tokens = {
        dataset.samples[index]["patient_token"]
        for index in fit_indices
        if dataset.samples[index]["label"] == 1
    }
    audit_tokens = {
        dataset.samples[index]["patient_token"]
        for index in audit_indices
        if dataset.samples[index]["label"] == 1
    }
    if fit_tokens & audit_tokens:
        raise RuntimeError("Patient tokens overlap between calibration partitions.")
    if model_state_digest(model) != state_before:
        raise RuntimeError("Model state changed during calibration/uncertainty audit.")

    summary: dict[str, Any] = {
        "experiment": "exp04_calibration_uncertainty_audit",
        "description": (
            "Post-hoc scalar temperature scaling and MC-dropout error-ranking "
            "audit using a grouped fit/audit partition of the calibration split."
        ),
        "n_calibration_images": len(dataset),
        "n_fit_images": len(fit_indices),
        "n_audit_images": len(audit_indices),
        "class_counts": {
            class_name: int(sum(label == class_index for label in labels))
            for class_name, class_index in config["dataset"]["classes"].items()
        },
        "fit_class_counts": {
            class_name: int(sum(labels[index] == class_index for index in fit_indices))
            for class_name, class_index in config["dataset"]["classes"].items()
        },
        "audit_class_counts": {
            class_name: int(sum(labels[index] == class_index for index in audit_indices))
            for class_name, class_index in config["dataset"]["classes"].items()
        },
        "split": {
            "seed": SPLIT_SEED,
            "fit_fraction_target": FIT_FRACTION,
            "method": (
                "NORMAL images split at image level because reliable patient "
                "identifiers are unavailable; PNEUMONIA images split by "
                "patient_token, with tokens kept within one partition."
            ),
            "pneumonia_patient_token_overlap": len(fit_tokens & audit_tokens),
            "fit_manifest_indices": fit_indices,
            "audit_manifest_indices": audit_indices,
        },
        "threshold": THRESHOLD,
        "temperature": temperature,
        "temperature_fit_objective": "Binary negative log-likelihood",
        "ece_bins": ECE_BINS,
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "mc_dropout_seed": SPLIT_SEED,
        "mc_dropout_layers": [
            {
                "name": name,
                "type": type(module).__name__,
                "probability": float(module.p),
            }
            for name, module in model.named_modules()
            if isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout2d,
                    nn.Dropout3d,
                    nn.AlphaDropout,
                    nn.FeatureAlphaDropout,
                ),
            )
        ],
        "functional_head_dropout": {
            "name": "backbone.forward_head",
            "probability": float(
                getattr(getattr(model, "backbone", None), "drop_rate", 0.0)
            ),
            "enabled_during_training": True,
            "batch_normalization_frozen_during_mc_sampling": True,
        },
        "audit_metrics": {
            "raw": binary_probability_metrics(raw_probabilities, audit_labels),
            "temperature_scaled": binary_probability_metrics(
                scaled_probabilities, audit_labels
            ),
        },
        "n_audit_errors": int(error_labels.sum()),
        "uncertainty_error_ranking_auc": {
            "predictive_entropy": ranking_auc(
                error_labels, predictive_entropy
            ),
            "probability_variance": ranking_auc(
                error_labels, probability_variance
            ),
            "mutual_information": ranking_auc(
                error_labels, mutual_information
            ),
        },
        "official_test_manifest_contents_or_images_read": False,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": model_state_digest(model),
        "model_state_unchanged": True,
        "checkpoint_sha256": checkpoint_sha256,
        "interpretation": (
            "Internal method-development evidence only. MC-dropout values are "
            "approximate uncertainty scores, not clinical certainty. The "
            "audit subset is not external data; NORMAL patient identity is "
            "unknown; ECE depends on binning; no coverage guarantee is claimed."
        ),
    }

    audit_rows = []
    for local_index, manifest_index in enumerate(audit_indices):
        audit_rows.append(
            {
                "calibration_manifest_index": manifest_index,
                "ground_truth": (
                    "PNEUMONIA" if audit_labels[local_index] else "NORMAL"
                ),
                "raw_probability": float(raw_probabilities[local_index]),
                "temperature_scaled_probability": float(
                    scaled_probabilities[local_index]
                ),
                "mc_dropout_mean_probability": float(
                    mean_mc_probability[local_index]
                ),
                "predictive_entropy": float(predictive_entropy[local_index]),
                "probability_variance": float(probability_variance[local_index]),
                "mutual_information": float(mutual_information[local_index]),
                "mc_dropout_prediction_error": int(error_labels[local_index]),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "calibration_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "calibration_uncertainty_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    (OUTPUT_DIR / "calibration_uncertainty_report.md").write_text(
        _make_report(summary),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = run()
    print(
        f"Completed {result['experiment']}: "
        f"fit={result['n_fit_images']}, audit={result['n_audit_images']}, "
        f"temperature={result['temperature']:.6f}, "
        f"state_unchanged={result['model_state_unchanged']}"
    )
