"""Pre-specified, frozen-checkpoint robustness analysis on all validation images."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import ManifestDataset
from src.ui.inference import PneumoniaInference


OUTPUT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 32
THRESHOLD = 0.5
GAMMA_PARAM = 0.85
BRIGHTNESS_FACTORS = (0.9, 1.1)
CONTRAST_FACTORS = (0.9, 1.1)
JPEG_QUALITY = 85
BORDER_MARGIN_PX = 32
CENTRAL_SIDE_PX = 156

CONDITION_GROUPS = {
    "realistic_single_factor": (
        "brightness_0.9",
        "brightness_1.1",
        "contrast_0.9",
        "contrast_1.1",
        "jpeg_quality_85",
    ),
    "photometric_control": ("gamma_dark",),
    "ood_mask_stress": ("border_black", "central_black"),
}
CONDITIONS = tuple(
    condition
    for group in CONDITION_GROUPS.values()
    for condition in group
)


def apply_condition(image: Image.Image, condition: str) -> Image.Image:
    """Apply a fixed single-factor change to an already-resized RGB image."""
    rgb = image.convert("RGB")
    if condition == "brightness_0.9":
        return ImageEnhance.Brightness(rgb).enhance(0.9)
    if condition == "brightness_1.1":
        return ImageEnhance.Brightness(rgb).enhance(1.1)
    if condition == "contrast_0.9":
        return ImageEnhance.Contrast(rgb).enhance(0.9)
    if condition == "contrast_1.1":
        return ImageEnhance.Contrast(rgb).enhance(1.1)
    if condition == "jpeg_quality_85":
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            compressed.load()
            return compressed.convert("RGB").copy()

    pixels = np.asarray(rgb, dtype=np.uint8).copy()
    if condition == "gamma_dark":
        adjusted = np.power(
            pixels.astype(np.float32) / 255.0,
            1.0 / GAMMA_PARAM,
        )
        return Image.fromarray(
            (adjusted * 255.0).round().clip(0, 255).astype(np.uint8)
        )

    if condition == "border_black":
        margin = BORDER_MARGIN_PX
        pixels[:margin, :, :] = 0
        pixels[-margin:, :, :] = 0
        pixels[:, :margin, :] = 0
        pixels[:, -margin:, :] = 0
        return Image.fromarray(pixels)

    if condition == "central_black":
        height, width = pixels.shape[:2]
        top = (height - CENTRAL_SIDE_PX) // 2
        left = (width - CENTRAL_SIDE_PX) // 2
        pixels[
            top : top + CENTRAL_SIDE_PX,
            left : left + CENTRAL_SIDE_PX,
            :,
        ] = 0
        return Image.fromarray(pixels)

    raise ValueError(f"Unknown validation perturbation: {condition}")


def model_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize_condition(
    records: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    selected = [row for row in records if row["condition"] == condition]
    deltas = np.asarray(
        [row["absolute_probability_change"] for row in selected],
        dtype=np.float64,
    )
    flips = sum(row["prediction_flip"] for row in selected)
    by_class: dict[str, dict[str, Any]] = {}
    for class_name in ("NORMAL", "PNEUMONIA"):
        class_rows = [row for row in selected if row["ground_truth"] == class_name]
        class_deltas = np.asarray(
            [row["absolute_probability_change"] for row in class_rows],
            dtype=np.float64,
        )
        by_class[class_name] = {
            "n": len(class_rows),
            "n_flip": sum(row["prediction_flip"] for row in class_rows),
            "mean_abs_probability_change": (
                float(class_deltas.mean()) if len(class_deltas) else None
            ),
        }
    return {
        "n": len(selected),
        "n_flip": flips,
        "flip_rate": flips / len(selected) if selected else None,
        "mean_abs_probability_change": (
            float(deltas.mean()) if len(deltas) else None
        ),
        "median_abs_probability_change": (
            float(np.median(deltas)) if len(deltas) else None
        ),
        "p95_abs_probability_change": (
            float(np.quantile(deltas, 0.95)) if len(deltas) else None
        ),
        "max_abs_probability_change": (
            float(deltas.max()) if len(deltas) else None
        ),
        "by_ground_truth": by_class,
    }


def _resize_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    height, width = size
    return image.convert("RGB").resize(
        (width, height), resample=Image.Resampling.BICUBIC
    )


def _batch_probabilities(
    model: torch.nn.Module,
    tensors: list[torch.Tensor],
    device: torch.device,
) -> list[float]:
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        logits = model(batch).reshape(-1)
        probabilities = torch.sigmoid(logits)
    return [float(probability) for probability in probabilities.cpu().tolist()]


def run() -> dict[str, Any]:
    config = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
    inference = PneumoniaInference.from_project(PROJECT_ROOT)
    model = inference.model
    state_before = model_state_digest(model)
    checkpoint_path = (
        PROJECT_ROOT / config["training"]["output_dir"] / "best_model.pt"
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    prep = config["preprocessing"]
    image_size = tuple(prep["image_size"])
    eval_transform = inference.transform
    dataset = ManifestDataset(
        manifest_path=config["dataset"]["resolved_manifest_paths"]["val"],
        data_root=config["dataset"]["data_root_resolved"],
        expected_split="val",
        class_to_idx=config["dataset"]["classes"],
        transform=eval_transform,
    )
    index_to_class = {
        class_index: class_name
        for class_name, class_index in config["dataset"]["classes"].items()
    }

    records: list[dict[str, Any]] = []
    original_probabilities: list[float] = []
    ground_truths: list[str] = []
    device = inference.device

    for batch_start in range(0, len(dataset), BATCH_SIZE):
        indices = range(batch_start, min(batch_start + BATCH_SIZE, len(dataset)))
        batch_indices = list(indices)
        originals: list[torch.Tensor] = []
        resized_images: list[Image.Image] = []
        labels: list[str] = []

        for index in batch_indices:
            tensor, target = dataset[index]
            sample = dataset.samples[index]
            with Image.open(sample["path"]) as source:
                resized = _resize_rgb(source, image_size)
            originals.append(tensor)
            resized_images.append(resized)
            labels.append(index_to_class[int(target.item())])

        original_probs = _batch_probabilities(model, originals, device)
        original_probabilities.extend(original_probs)
        ground_truths.extend(labels)

        for condition in CONDITIONS:
            transformed = [
                eval_transform(apply_condition(image, condition))
                for image in resized_images
            ]
            perturbed_probs = _batch_probabilities(model, transformed, device)
            for index, ground_truth, original_prob, perturbed_prob in zip(
                batch_indices, labels, original_probs, perturbed_probs
            ):
                original_class = (
                    "PNEUMONIA"
                    if original_prob >= THRESHOLD
                    else "NORMAL"
                )
                perturbed_class = (
                    "PNEUMONIA"
                    if perturbed_prob >= THRESHOLD
                    else "NORMAL"
                )
                records.append(
                    {
                        "validation_index": index,
                        "ground_truth": ground_truth,
                        "original_probability": original_prob,
                        "condition": condition,
                        "perturbed_probability": perturbed_prob,
                        "absolute_probability_change": abs(
                            perturbed_prob - original_prob
                        ),
                        "prediction_flip": int(
                            original_class != perturbed_class
                        ),
                    }
                )

    if len(dataset) != 518:
        raise RuntimeError(
            f"Expected the audited 518-image validation set, found {len(dataset)}."
        )
    if model_state_digest(model) != state_before:
        raise RuntimeError("Model parameters changed during the frozen evaluation.")

    baseline_predictions = [
        int(probability >= THRESHOLD)
        for probability in original_probabilities
    ]
    confusion = Counter(
        (truth, prediction)
        for truth, prediction in zip(
            [int(name == "PNEUMONIA") for name in ground_truths],
            baseline_predictions,
        )
    )
    summary: dict[str, Any] = {
        "experiment": "exp03_full_validation_robustness",
        "description": (
            "Pre-specified single-factor sensitivity evaluation on the complete "
            "internal-validation split using the frozen EfficientNetB0 checkpoint."
        ),
        "n_validation_images": len(dataset),
        "threshold": THRESHOLD,
        "checkpoint": "experiments/exp01_efficientnetb0_baseline/best_model.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": model_state_digest(model),
        "model_state_unchanged": True,
        "official_test_manifest_contents_or_images_read": False,
        "calibration_manifest_contents_or_images_read": False,
        "selection": "Every row in the existing val_manifest.csv, in manifest order.",
        "conditions": {
            "brightness_0.9": "Brightness factor 0.9, applied alone after resize.",
            "brightness_1.1": "Brightness factor 1.1, applied alone after resize.",
            "contrast_0.9": "Contrast factor 0.9, applied alone after resize.",
            "contrast_1.1": "Contrast factor 1.1, applied alone after resize.",
            "jpeg_quality_85": (
                "RGB JPEG re-encoding at quality 85 after resize; decoded before normalization."
            ),
            "gamma_dark": (
                "Existing pilot control: pixel^(1/0.85), applied after resize."
            ),
            "border_black": (
                "OOD stress test only: outer 32-pixel border set to black after resize."
            ),
            "central_black": (
                "OOD stress test only: central 156×156 square set to black after resize."
            ),
        },
        "condition_groups": {
            group: list(names) for group, names in CONDITION_GROUPS.items()
        },
        "preprocessing": (
            "Conditions are applied to RGB after bicubic 224×224 resize, then "
            "the established evaluation transform applies ImageNet normalization."
        ),
        "baseline_confusion": {
            "TN": confusion[(0, 0)],
            "FP": confusion[(0, 1)],
            "FN": confusion[(1, 0)],
            "TP": confusion[(1, 1)],
        },
        "results": {
            condition: summarize_condition(records, condition)
            for condition in CONDITIONS
        },
        "interpretation": (
            "Descriptive results for this internal-validation split only. "
            "No confidence intervals or population-level generalization claims "
            "are made. Black-mask results are OOD stress responses, not evidence "
            "of causal feature reliance. Brightness/contrast magnitudes match "
            "the existing training augmentation bounds; JPEG quality is a fixed "
            "compression stress condition."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "validation_robustness_results.csv"
    json_path = OUTPUT_DIR / "validation_robustness_summary.json"
    report_path = OUTPUT_DIR / "validation_robustness_report.md"
    fieldnames = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment 3 — Full validation robustness analysis",
        "",
        f"**Sample:** {summary['n_validation_images']} existing internal-validation "
        "images; every row in `val_manifest.csv`.",
        f"**Threshold:** {summary['threshold']} (unchanged).",
        "**Checkpoint:** frozen `best_model.pt`; no training or tuning.",
        "**Official test/calibration data:** not accessed.",
        "",
        "## Results",
        "",
        "| Condition | N | Prediction flips | Mean absolute probability change | Median | 95th percentile |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group_name, condition_names in summary["condition_groups"].items():
        if group_name == "ood_mask_stress":
            lines.extend(["", "### OOD mask stress tests", ""])
        elif group_name == "realistic_single_factor":
            lines.extend(["", "### Single-factor perturbations", ""])
        else:
            lines.extend(["", "### Photometric control", ""])
        for name in condition_names:
            result = summary["results"][name]
            lines.append(
                f"| {name} | {result['n']} | {result['n_flip']} "
                f"({result['flip_rate']:.1%}) | "
                f"{result['mean_abs_probability_change']:.6f} | "
                f"{result['median_abs_probability_change']:.6f} | "
                f"{result['p95_abs_probability_change']:.6f} |"
            )
    lines.extend(
        [
            "",
            "### Baseline internal-validation confusion matrix",
            "",
            f"TN={summary['baseline_confusion']['TN']}, "
            f"FP={summary['baseline_confusion']['FP']}, "
            f"FN={summary['baseline_confusion']['FN']}, "
            f"TP={summary['baseline_confusion']['TP']}.",
            "",
            "## Interpretation and limits",
            "",
            summary["interpretation"],
            "",
            "Border and central masking are deliberately separated as large "
            "black-mask OOD stress tests. The complete validation set is still "
            "from the training dataset source, so this does not establish "
            "external or clinical generalization. NORMAL filename-derived "
            "patient identity remains unavailable.",
            "",
            "Per-image results are in `validation_robustness_results.csv`; "
            "machine-readable aggregate data are in "
            "`validation_robustness_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = run()
    print(
        f"Completed {result['experiment']} on "
        f"{result['n_validation_images']} validation images."
    )
    for condition, metrics in result["results"].items():
        print(
            f"{condition}: flips={metrics['n_flip']}/{metrics['n']}, "
            f"mean_abs_delta={metrics['mean_abs_probability_change']:.6f}"
        )
