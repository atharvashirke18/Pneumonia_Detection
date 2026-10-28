"""In-memory upload validation and frozen-checkpoint inference for the UI."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.config import load_config
from src.data.datamodule import get_transforms
from src.explainability.gradcam import (
    GradCAM,
    get_target_layer,
    overlay_heatmap,
)
from src.models.model import build_model
from src.training.trainer import load_checkpoint


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
DECISION_THRESHOLD = 0.5
SUPPORTED_FORMATS = {
    ".bmp": "BMP",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


class UploadValidationError(ValueError):
    """An upload is unsupported, too large, or cannot be safely decoded."""


@dataclass(frozen=True)
class Prediction:
    class_name: str
    pneumonia_probability: float


def decode_uploaded_image(payload: bytes, filename: str) -> Image.Image:
    """Validate upload size/format and return a decoded in-memory RGB copy."""
    if not payload:
        raise UploadValidationError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("The image exceeds the 15 MiB upload limit.")

    suffix = Path(filename).suffix.lower()
    expected_format = SUPPORTED_FORMATS.get(suffix)
    if expected_format is None:
        raise UploadValidationError(
            "Unsupported image type. Upload a PNG, JPEG, BMP, or TIFF image."
        )

    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != expected_format:
                raise UploadValidationError(
                    "The file extension does not match the decoded image format."
                )
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise UploadValidationError(
                    "The image dimensions exceed the supported pixel limit."
                )
            image.load()
            return image.convert("RGB").copy()
    except UploadValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        EOFError,
        Image.DecompressionBombError,
    ) as exc:
        raise UploadValidationError(
            "The uploaded file is not a valid, decodable image."
        ) from exc


class PneumoniaInference:
    """Frozen EfficientNetB0 checkpoint with the project's evaluation pipeline."""

    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.transform = transform
        self.device = device
        self.threshold = DECISION_THRESHOLD
        self._model_lock = Lock()

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        device: Optional[torch.device] = None,
    ) -> "PneumoniaInference":
        root = Path(project_root).resolve()
        config = load_config("configs/config.yaml", base_dir=root)
        model_config = dict(config["model"])
        model_config["pretrained"] = False
        model = build_model(model_config)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        checkpoint_path = (
            root / config["training"]["output_dir"] / "best_model.pt"
        )
        load_checkpoint(checkpoint_path, model, device=device)

        prep = config["preprocessing"]
        transform = get_transforms(
            image_size=tuple(prep["image_size"]),
            mean=tuple(prep["normalization"]["mean"]),
            std=tuple(prep["normalization"]["std"]),
            is_training=False,
        )
        return cls(model, transform, device)

    def _tensor_for_image(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> Prediction:
        with self._model_lock:
            tensor = self._tensor_for_image(image)
            with torch.inference_mode():
                logit = self.model(tensor)
                probability = torch.sigmoid(logit).item()

        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("The model returned an invalid probability.")

        return Prediction(
            class_name=(
                "PNEUMONIA"
                if probability >= self.threshold
                else "NORMAL"
            ),
            pneumonia_probability=float(probability),
        )

    def explain(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._model_lock:
            tensor = self._tensor_for_image(image)
            rgb = np.asarray(image.convert("RGB"))
            try:
                with GradCAM(self.model, get_target_layer(self.model)) as gradcam:
                    native_heatmap, heatmap = gradcam.generate_with_native(
                        tensor, output_size=(image.height, image.width)
                    )
                overlay = overlay_heatmap(rgb, heatmap)
                return native_heatmap, heatmap, overlay
            finally:
                self.model.zero_grad(set_to_none=True)
