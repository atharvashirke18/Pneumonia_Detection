"""Evaluate native Grad-CAM maps against supplied expert validation masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import ManifestDataset
from src.explainability.evaluation import evaluate_native_cam_against_mask
from src.explainability.gradcam import GradCAM, get_target_layer
from src.ui.inference import PneumoniaInference


OUTPUT_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {"image_id", "annotation_type", "mask_path", "annotator_id"}


def load_annotations(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Annotation manifest is missing columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError(
            "No expert annotations were supplied; localization metrics were not computed."
        )
    seen: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        row = {key: (value or "").strip() for key, value in row.items()}
        rows[row_number - 2] = row
        if not all(row.get(key) for key in REQUIRED_COLUMNS):
            raise ValueError(f"Annotation manifest row {row_number} has an empty field.")
        if row["annotation_type"] not in {"lung", "lesion"}:
            raise ValueError(
                f"Row {row_number}: annotation_type must be 'lung' or 'lesion'."
            )
        key = (row["image_id"], row["annotation_type"], row["annotator_id"])
        if key in seen:
            raise ValueError(f"Duplicate image/type/annotator record on row {row_number}.")
        seen.add(key)
    return rows


def run(annotation_path: Path) -> dict[str, Any]:
    annotation_path = annotation_path.resolve()
    annotations = load_annotations(annotation_path)
    config = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
    inference = PneumoniaInference.from_project(PROJECT_ROOT)
    model = inference.model
    device = inference.device
    checkpoint = (
        PROJECT_ROOT / config["training"]["output_dir"] / "best_model.pt"
    )
    state_before = hashlib.sha256(
        b"".join(
            tensor.detach().cpu().contiguous().numpy().tobytes()
            for _, tensor in sorted(model.state_dict().items())
        )
    ).hexdigest()

    dataset = ManifestDataset(
        manifest_path=config["dataset"]["resolved_manifest_paths"]["val"],
        data_root=config["dataset"]["data_root_resolved"],
        expected_split="val",
        class_to_idx=config["dataset"]["classes"],
        transform=inference.transform,
    )
    data_root = Path(config["dataset"]["data_root_resolved"]).resolve()
    by_id = {
        Path(sample["path"]).resolve().relative_to(data_root).as_posix(): index
        for index, sample in enumerate(dataset.samples)
    }
    if len(by_id) != len(dataset):
        raise RuntimeError("Validation manifest rel_path values are not unique.")

    result_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        image_id = annotation["image_id"].replace("\\", "/")
        if image_id not in by_id:
            raise ValueError(
                f"Image '{image_id}' is not in the internal validation manifest."
            )
        mask_path = (annotation_path.parent / annotation["mask_path"]).resolve()
        if not mask_path.is_relative_to(annotation_path.parent):
            raise ValueError("Mask paths must remain inside the annotation directory.")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Annotation mask not found: {mask_path.name}")

        index = by_id[image_id]
        tensor, target = dataset[index]
        source_path = Path(dataset.samples[index]["path"])
        with Image.open(source_path) as source:
            source_size = source.size
        with Image.open(mask_path) as mask_image:
            mask = mask_image.copy()
        if mask.size != source_size:
            raise ValueError(
                f"Mask dimensions for '{image_id}' do not match its source image."
            )

        with torch.inference_mode():
            logit = model(tensor.unsqueeze(0).to(device)).reshape(-1)[0]
            predicted_class = int(torch.sigmoid(logit).item() >= 0.5)
        with GradCAM(model, get_target_layer(model)) as gradcam:
            native_cam, _ = gradcam.generate_with_native(
                tensor.unsqueeze(0).to(device),
                output_size=(224, 224),
                target_class=predicted_class,
            )
        metrics = evaluate_native_cam_against_mask(native_cam, mask)
        result_rows.append(
            {
                "image_id": image_id,
                "annotation_type": annotation["annotation_type"],
                "annotator_id": annotation["annotator_id"],
                "ground_truth_class": "PNEUMONIA" if int(target.item()) else "NORMAL",
                "predicted_class": "PNEUMONIA" if predicted_class else "NORMAL",
                **metrics,
            }
        )

    if state_before != hashlib.sha256(
        b"".join(
            tensor.detach().cpu().contiguous().numpy().tobytes()
            for _, tensor in sorted(model.state_dict().items())
        )
    ).hexdigest():
        raise RuntimeError("Model parameters changed during localization evaluation.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[row["annotation_type"]].append(row)
    aggregate = {}
    for annotation_type, rows in grouped.items():
        aggregate[annotation_type] = {"n_annotations": len(rows)}
        for metric in (
            "pointing_occupancy_at_peak",
            "activation_fraction_in_mask",
            "top_20pct_cells_inside_mask",
            "top_20pct_cell_recall",
            "mask_area_fraction",
        ):
            valid_values = [
                row[metric]
                for row in rows
                if row[metric] is not None
            ]
            if not valid_values:
                aggregate[annotation_type][metric] = None
                continue
            aggregate[annotation_type][metric] = float(
                np.mean(valid_values)
            )
    summary = {
        "experiment": "exp06_expert_mask_localization",
        "n_annotations": len(result_rows),
        "source_split": "internal validation only",
        "native_cam_grid": [7, 7],
        "threshold": 0.5,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "results_by_annotation_type": aggregate,
        "official_test_manifest_contents_or_images_read": False,
        "model_parameters_unchanged": True,
        "interpretation": (
            "Localization metrics describe overlap with supplied annotations "
            "only. Coarse 7x7 Grad-CAM resolution limits small-region assessment; "
            "these metrics are not clinical localization performance."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "localization_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    (OUTPUT_DIR / "localization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="CSV manifest of expert masks; see annotations_manifest_template.csv",
    )
    arguments = parser.parse_args()
    try:
        print(json.dumps(run(arguments.annotations), indent=2))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
