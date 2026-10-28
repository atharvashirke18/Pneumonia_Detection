"""Deletion/insertion faithfulness audit on the full internal validation split."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import ManifestDataset
from src.explainability.gradcam import GradCAM, get_target_layer
from src.ui.inference import PneumoniaInference


OUTPUT_DIR = Path(__file__).resolve().parent
CURVE_FRACTIONS = tuple(np.linspace(0.0, 1.0, 11))
RANDOM_REPEATS = 5
RANDOM_SEED = 20261011
BOOTSTRAP_REPEATS = 2000
INFERENCE_BATCH_SIZE = 32
BLUR_RADIUS = 12


def model_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def grid_mask(order: np.ndarray, fraction: float) -> np.ndarray:
    selected = int(round(len(order) * fraction))
    mask = np.zeros(len(order), dtype=np.float32)
    mask[order[:selected]] = 1.0
    return mask.reshape(7, 7)


def target_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_classes: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(images), INFERENCE_BATCH_SIZE):
            end = min(start + INFERENCE_BATCH_SIZE, len(images))
            logits = model(images[start:end].to(device)).reshape(-1)
            pneumonia = torch.sigmoid(logits)
            targets = target_classes[start:end].to(device)
            score = torch.where(targets == 1, pneumonia, 1.0 - pneumonia)
            values.extend(float(value) for value in score.cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def evaluate_curve(
    model: torch.nn.Module,
    originals: torch.Tensor,
    blurred: torch.Tensor,
    target_classes: torch.Tensor,
    native_cams: np.ndarray,
    device: torch.device,
    insertion: bool,
    random_orders: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate a deletion or insertion curve for every validation image."""
    n_images = len(originals)
    result = np.zeros((n_images, len(CURVE_FRACTIONS)), dtype=np.float64)
    for step, fraction in enumerate(CURVE_FRACTIONS):
        masks: list[np.ndarray] = []
        for image_index in range(n_images):
            if random_orders is None:
                order = np.argsort(native_cams[image_index].ravel(), kind="stable")[::-1]
            else:
                order = random_orders[image_index]
            masks.append(grid_mask(order, fraction))
        mask_tensor = torch.from_numpy(np.stack(masks)).unsqueeze(1)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=originals.shape[-2:],
            mode="nearest",
        )
        if insertion:
            perturbed = blurred * (1.0 - mask_tensor) + originals * mask_tensor
        else:
            perturbed = originals * (1.0 - mask_tensor) + blurred * mask_tensor
        result[:, step] = target_probabilities(
            model, perturbed, target_classes, device
        )
    return result


def normalized_auc(curves: np.ndarray) -> np.ndarray:
    return np.trapezoid(curves, x=np.asarray(CURVE_FRACTIONS), axis=1)


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
) -> list[float]:
    samples = np.asarray(values, dtype=np.float64)
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        estimates[index] = rng.choice(samples, size=len(samples), replace=True).mean()
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def summarize(
    cam_auc: np.ndarray,
    random_auc: np.ndarray,
    class_names: list[str],
    rng: np.random.Generator,
    advantage_direction: str,
) -> dict[str, Any]:
    if advantage_direction == "random_minus_cam":
        difference = random_auc - cam_auc
    elif advantage_direction == "cam_minus_random":
        difference = cam_auc - random_auc
    else:
        raise ValueError(f"Unknown advantage direction: {advantage_direction}")
    output: dict[str, Any] = {
        "cam_mean_auc": float(cam_auc.mean()),
        "random_mean_auc": float(random_auc.mean()),
        "paired_mean_advantage": float(difference.mean()),
        "paired_mean_advantage_image_bootstrap_95pct_ci": bootstrap_mean_ci(
            difference, rng
        ),
        "n": int(len(cam_auc)),
    }
    for class_name in ("NORMAL", "PNEUMONIA"):
        selected = np.asarray(class_names) == class_name
        if selected.any():
            class_difference = difference[selected]
            output[class_name] = {
                "n": int(selected.sum()),
                "cam_mean_auc": float(cam_auc[selected].mean()),
                "random_mean_auc": float(random_auc[selected].mean()),
                "paired_mean_advantage": float(class_difference.mean()),
                "paired_mean_advantage_image_bootstrap_95pct_ci": (
                    bootstrap_mean_ci(class_difference, rng)
                ),
            }
    return output


def run() -> dict[str, Any]:
    config = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
    inference = PneumoniaInference.from_project(PROJECT_ROOT)
    model = inference.model
    device = inference.device
    state_before = model_state_digest(model)
    checkpoint_path = (
        PROJECT_ROOT / config["training"]["output_dir"] / "best_model.pt"
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    image_size = tuple(config["preprocessing"]["image_size"])
    dataset = ManifestDataset(
        manifest_path=config["dataset"]["resolved_manifest_paths"]["val"],
        data_root=config["dataset"]["data_root_resolved"],
        expected_split="val",
        class_to_idx=config["dataset"]["classes"],
        transform=inference.transform,
    )
    if len(dataset) != 518:
        raise RuntimeError(
            f"Expected 518 validation images, found {len(dataset)}."
        )

    originals: list[torch.Tensor] = []
    blurred: list[torch.Tensor] = []
    labels: list[str] = []
    for index, sample in enumerate(dataset.samples):
        tensor, target = dataset[index]
        with Image.open(sample["path"]) as source:
            resized = source.convert("RGB").resize(
                (image_size[1], image_size[0]),
                resample=Image.Resampling.BICUBIC,
            )
            blur_image = resized.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
        originals.append(tensor)
        blurred.append(inference.transform(blur_image))
        labels.append("PNEUMONIA" if int(target.item()) == 1 else "NORMAL")

    original_tensor = torch.stack(originals)
    blurred_tensor = torch.stack(blurred)
    with torch.inference_mode():
        baseline_logits = torch.cat(
            [
                model(original_tensor[start : start + INFERENCE_BATCH_SIZE].to(device))
                .reshape(-1)
                .cpu()
                for start in range(0, len(dataset), INFERENCE_BATCH_SIZE)
            ]
        )
        baseline_probability = torch.sigmoid(baseline_logits).numpy()
    target_classes = torch.from_numpy(
        (baseline_probability >= 0.5).astype(np.int64)
    )

    native_cams: list[np.ndarray] = []
    for tensor, predicted_class in zip(originals, target_classes.tolist()):
        with GradCAM(model, get_target_layer(model)) as gradcam:
            native, _ = gradcam.generate_with_native(
                tensor.unsqueeze(0).to(device),
                output_size=(image_size[0], image_size[1]),
                target_class=predicted_class,
            )
        native_cams.append(native)
    cams = np.stack(native_cams)

    cam_deletion = evaluate_curve(
        model, original_tensor, blurred_tensor, target_classes,
        cams, device, insertion=False,
    )
    cam_insertion = evaluate_curve(
        model, original_tensor, blurred_tensor, target_classes,
        cams, device, insertion=True,
    )

    random = np.random.default_rng(RANDOM_SEED)
    random_deletion_auc: list[np.ndarray] = []
    random_insertion_auc: list[np.ndarray] = []
    for _ in range(RANDOM_REPEATS):
        orders = [random.permutation(49) for _ in range(len(dataset))]
        deletion = evaluate_curve(
            model, original_tensor, blurred_tensor, target_classes,
            cams, device, insertion=False, random_orders=orders,
        )
        insertion = evaluate_curve(
            model, original_tensor, blurred_tensor, target_classes,
            cams, device, insertion=True, random_orders=orders,
        )
        random_deletion_auc.append(normalized_auc(deletion))
        random_insertion_auc.append(normalized_auc(insertion))

    deletion_auc = normalized_auc(cam_deletion)
    insertion_auc = normalized_auc(cam_insertion)
    mean_random_deletion_auc = np.mean(random_deletion_auc, axis=0)
    mean_random_insertion_auc = np.mean(random_insertion_auc, axis=0)
    if model_state_digest(model) != state_before:
        raise RuntimeError("Model parameters changed during explanation audit.")

    confidence_rng = np.random.default_rng(RANDOM_SEED + 1)
    summary = {
        "experiment": "exp05_validation_gradcam_faithfulness",
        "n_validation_images": len(dataset),
        "checkpoint": "experiments/exp01_efficientnetb0_baseline/best_model.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "threshold": 0.5,
        "target_class": (
            "Each image's baseline predicted class; Grad-CAM is computed for "
            "the positive class logit or its negation, as appropriate."
        ),
        "replacement_baseline": (
            f"Gaussian-blurred resized RGB image, PIL blur radius {BLUR_RADIUS}, "
            "then the unchanged evaluation transform."
        ),
        "native_cam_grid": list(cams.shape[1:]),
        "curve_fractions": list(CURVE_FRACTIONS),
        "random_control_repeats": RANDOM_REPEATS,
        "random_seed": RANDOM_SEED,
        "deletion": summarize(
            deletion_auc,
            mean_random_deletion_auc,
            labels,
            confidence_rng,
            advantage_direction="random_minus_cam",
        ),
        "insertion": summarize(
            insertion_auc,
            mean_random_insertion_auc,
            labels,
            confidence_rng,
            advantage_direction="cam_minus_random",
        ),
        "interpretation": (
            "Deletion advantage is random-control AUC minus Grad-CAM AUC; "
            "insertion advantage is Grad-CAM AUC minus random-control AUC. "
            "Positive values indicate better ranking than the sampled random "
            "controls under this blur-replacement protocol, not causal proof "
            "or lesion localization. The mask replacement can still create "
            "distribution shift. Confidence intervals resample images and do "
            "not account for patient clustering."
        ),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": model_state_digest(model),
        "model_state_unchanged": True,
        "official_test_manifest_contents_or_images_read": False,
        "calibration_manifest_contents_or_images_read": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(dataset)):
        rows.append(
            {
                "validation_index": index,
                "ground_truth": labels[index],
                "predicted_class": "PNEUMONIA" if target_classes[index] else "NORMAL",
                "pneumonia_probability": float(baseline_probability[index]),
                "cam_deletion_auc": float(deletion_auc[index]),
                "random_deletion_auc_mean": float(mean_random_deletion_auc[index]),
                "deletion_advantage": float(
                    mean_random_deletion_auc[index] - deletion_auc[index]
                ),
                "cam_insertion_auc": float(insertion_auc[index]),
                "random_insertion_auc_mean": float(mean_random_insertion_auc[index]),
                "insertion_advantage": float(
                    insertion_auc[index] - mean_random_insertion_auc[index]
                ),
            }
        )
    with (OUTPUT_DIR / "faithfulness_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "faithfulness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "faithfulness_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment 5 — Grad-CAM faithfulness audit",
        "",
        f"**Sample:** all {summary['n_validation_images']} internal-validation images.",
        "**Checkpoint:** frozen baseline; threshold 0.5.",
        f"**Native map:** {summary['native_cam_grid'][0]} × "
        f"{summary['native_cam_grid'][1]} cells.",
        f"**Replacement:** {summary['replacement_baseline']}",
        f"**Random controls:** {summary['random_control_repeats']} fixed-seed "
        "random cell rankings per image.",
        "**Official test and calibration manifests/images:** not read.",
        "",
        "## Results",
        "",
        "| Curve / ground-truth group | Grad-CAM AUC | Random AUC | Paired advantage | Image-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, display in (("deletion", "Deletion"), ("insertion", "Insertion")):
        result = summary[name]
        lo, hi = result["paired_mean_advantage_image_bootstrap_95pct_ci"]
        lines.append(
            f"| {display} | {result['cam_mean_auc']:.4f} | "
            f"{result['random_mean_auc']:.4f} | "
            f"{result['paired_mean_advantage']:.4f} | [{lo:.4f}, {hi:.4f}] |"
        )
        for class_name in ("NORMAL", "PNEUMONIA"):
            class_result = result[class_name]
            class_lo, class_hi = (
                class_result["paired_mean_advantage_image_bootstrap_95pct_ci"]
            )
            lines.append(
                f"| {display} — {class_name} truth (n={class_result['n']}) | "
                f"{class_result['cam_mean_auc']:.4f} | "
                f"{class_result['random_mean_auc']:.4f} | "
                f"{class_result['paired_mean_advantage']:.4f} | "
                f"[{class_lo:.4f}, {class_hi:.4f}] |"
            )
    lines.extend(
        [
            "",
            "Positive deletion advantage means deleting top Grad-CAM cells lowered "
            "the target-class score more than the random control. Positive insertion "
            "advantage means restoring those cells raised it more. These are "
            "model-faithfulness tests under the stated blur replacement; they do "
            "not show that a highlighted region is pneumonia or medically causal.",
            "",
            "Class subgroups are grouped by ground-truth label. The native map is "
            "only 7 × 7. Image-bootstrap intervals do not model "
            "patient clustering, and blurred replacement can itself shift inputs "
            "out of distribution. Expert lesion masks are still required for "
            "region-localization claims.",
            "",
            "Per-image results are in `faithfulness_results.csv`; machine-readable "
            "summary is in `faithfulness_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
