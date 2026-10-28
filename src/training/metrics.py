"""Binary classification metrics and early-stopping policy."""

from typing import Dict, Optional

import numpy as np


def compute_binary_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard binary classification metrics from logits and targets."""
    try:
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from e

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    metrics: Dict[str, float] = {"threshold": threshold}

    try:
        metrics["auc"] = float(roc_auc_score(targets, probs))
    except ValueError:
        metrics["auc"] = float("nan")

    try:
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    except ValueError:
        metrics["sensitivity"] = float("nan")
        metrics["specificity"] = float("nan")

    try:
        metrics["f1"] = float(f1_score(targets, preds, zero_division=0))
    except ValueError:
        metrics["f1"] = float("nan")

    return metrics


class EarlyStopping:
    """Stop when a monitored scalar has not improved for ``patience`` steps."""

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 1e-5,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Record the current value and return whether training should stop."""
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
