"""Quantitative evaluation helpers for coarse Grad-CAM maps and expert masks."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def mask_occupancy_on_cam_grid(
    mask: Image.Image | np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Return fractional annotation occupancy per native Grad-CAM cell."""
    if isinstance(mask, Image.Image):
        values = np.asarray(mask.convert("L"), dtype=np.float32)
    else:
        values = np.asarray(mask, dtype=np.float32)
        if values.ndim == 3:
            values = values.max(axis=2)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Annotation mask must be a finite 2-D image.")
    if values.size == 0:
        raise ValueError("Annotation mask cannot be empty.")
    binary = values > 0
    if not binary.any():
        raise ValueError("Annotation mask contains no positive pixels.")

    height, width = values.shape
    grid_height, grid_width = grid_shape
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("Grad-CAM grid dimensions must be positive.")
    row_edges = np.linspace(0, height, grid_height + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_width + 1, dtype=int)
    occupancy = np.zeros((grid_height, grid_width), dtype=np.float32)
    for row in range(grid_height):
        for col in range(grid_width):
            cell = binary[row_edges[row] : row_edges[row + 1],
                          col_edges[col] : col_edges[col + 1]]
            if cell.size:
                occupancy[row, col] = float(cell.mean())
    return occupancy


def evaluate_native_cam_against_mask(
    native_cam: np.ndarray,
    mask: Image.Image | np.ndarray,
) -> dict[str, float | None]:
    """Measure coarse-grid pointing and CAM activation within an annotation."""
    cam = np.asarray(native_cam, dtype=np.float64)
    if cam.ndim != 2 or cam.size == 0 or not np.isfinite(cam).all():
        raise ValueError("Native Grad-CAM must be a finite non-empty 2-D array.")
    if np.any(cam < 0):
        raise ValueError("Native Grad-CAM values must be non-negative.")
    occupancy = mask_occupancy_on_cam_grid(mask, cam.shape).astype(np.float64)
    total_activation = float(cam.sum())
    activation_in_mask = float((cam * occupancy).sum())
    peak_occupancy = (
        float(occupancy.flat[int(np.argmax(cam))])
        if total_activation > 0
        else None
    )
    selected_count = max(1, math.ceil(cam.size * 0.2))
    top_indices = np.argpartition(cam.ravel(), -selected_count)[-selected_count:]
    top_occupancy = occupancy.ravel()[top_indices]
    positive_cells = int(np.count_nonzero(occupancy >= 0.5))
    top_positive_cells = int(np.count_nonzero(top_occupancy >= 0.5))
    return {
        "pointing_occupancy_at_peak": peak_occupancy,
        "activation_fraction_in_mask": (
            activation_in_mask / total_activation if total_activation > 0 else 0.0
        ),
        "top_20pct_cells_inside_mask": float(top_occupancy.mean()),
        "top_20pct_cell_recall": (
            top_positive_cells / positive_cells if positive_cells else 0.0
        ),
        "mask_area_fraction": float(occupancy.mean()),
    }
