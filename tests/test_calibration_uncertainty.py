"""Unit tests for grouped calibration partitioning and uncertainty statistics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    PROJECT_ROOT
    / "experiments"
    / "exp04_calibration_uncertainty"
    / "run_calibration_uncertainty.py"
)
SPEC = importlib.util.spec_from_file_location("calibration_uncertainty", SCRIPT)
CALIBRATION = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(CALIBRATION)


def synthetic_samples():
    return [
        {"label": 0, "patient_token": ""}
        for _ in range(10)
    ] + [
        {"label": 1, "patient_token": token}
        for token, count in (("p1", 3), ("p2", 2), ("p3", 4), ("p4", 2))
        for _ in range(count)
    ]


def test_partition_is_deterministic_complete_and_grouped():
    samples = synthetic_samples()
    fit_a, audit_a = CALIBRATION.partition_calibration(samples, seed=12)
    fit_b, audit_b = CALIBRATION.partition_calibration(samples, seed=12)
    assert fit_a == fit_b
    assert audit_a == audit_b
    assert sorted(fit_a + audit_a) == list(range(len(samples)))
    assert set(fit_a).isdisjoint(audit_a)
    fit_tokens = {
        samples[index]["patient_token"]
        for index in fit_a
        if samples[index]["label"] == 1
    }
    audit_tokens = {
        samples[index]["patient_token"]
        for index in audit_a
        if samples[index]["label"] == 1
    }
    assert fit_tokens.isdisjoint(audit_tokens)


def test_partition_rejects_missing_positive_patient_identity():
    with pytest.raises(ValueError, match="missing its patient token"):
        CALIBRATION.partition_calibration(
            [{"label": 1, "patient_token": ""}]
        )


def test_temperature_is_positive_finite_and_reduces_fit_nll():
    logits = torch.tensor([-5.0, -3.0, 2.0, 4.0, -2.0, 3.0])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    temperature = CALIBRATION.fit_temperature(logits, labels)
    assert np.isfinite(temperature)
    assert temperature > 0
    raw_loss = nn.functional.binary_cross_entropy_with_logits(
        logits.double(), labels.double()
    ).item()
    scaled_loss = nn.functional.binary_cross_entropy_with_logits(
        logits.double() / temperature, labels.double()
    ).item()
    assert scaled_loss <= raw_loss


def test_calibration_metrics_include_brier_nll_and_bins():
    probabilities = np.asarray([0.1, 0.3, 0.7, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    result = CALIBRATION.binary_probability_metrics(probabilities, labels)
    assert result["n"] == 4
    assert result["accuracy_at_0.5"] == 1.0
    assert result["brier_score"] == pytest.approx(0.05)
    assert result["negative_log_likelihood"] > 0
    assert result["expected_calibration_error_10_bins"] == pytest.approx(0.2)
    assert sum(item["n"] for item in result["reliability_bins"]) == 4


def test_error_ranking_auc_handles_ties_and_missing_error_class():
    assert CALIBRATION.ranking_auc(
        np.asarray([0, 1, 0, 1]), np.asarray([0.1, 0.8, 0.4, 0.8])
    ) == pytest.approx(1.0)
    assert CALIBRATION.ranking_auc(
        np.asarray([0, 0]), np.asarray([0.1, 0.2])
    ) is None


def test_mc_dropout_produces_stochastic_logits_and_restores_eval_mode():
    torch.manual_seed(8)
    model = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(8, 1),
    ).eval()
    inputs = torch.ones(6, 8)
    samples = CALIBRATION.mc_dropout_logits(
        model,
        inputs,
        list(range(len(inputs))),
        torch.device("cpu"),
        passes=8,
        seed=25,
    )
    assert samples.shape == (8, 6)
    assert np.isfinite(samples).all()
    assert np.any(samples.var(axis=0) > 0)
    assert not model.training


def test_functional_head_dropout_keeps_batchnorm_in_eval_mode():
    class EfficientLikeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop_rate = 0.5
            self.batch_norm = nn.BatchNorm1d(8)
            self.linear = nn.Linear(8, 1)

        def forward(self, values):
            values = self.batch_norm(values)
            values = torch.nn.functional.dropout(
                values, p=self.drop_rate, training=self.training
            )
            return self.linear(values)

    class EfficientLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = EfficientLikeBackbone()

        def forward(self, values):
            return self.backbone(values)

    model = EfficientLikeModel().eval()
    inputs = torch.ones(6, 8)
    samples = CALIBRATION.mc_dropout_logits(
        model,
        inputs,
        list(range(len(inputs))),
        torch.device("cpu"),
        passes=8,
        seed=25,
    )
    assert np.any(samples.var(axis=0) > 0)
    assert not model.training
    assert not model.backbone.batch_norm.training


def test_completed_calibration_audit_is_frozen_and_test_free():
    output = SCRIPT.parent
    summary_path = output / "calibration_uncertainty_summary.json"
    csv_path = output / "calibration_uncertainty_audit.csv"
    report_path = output / "calibration_uncertainty_report.md"
    assert summary_path.is_file()
    assert csv_path.is_file()
    assert report_path.is_file()
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_calibration_images"] == 489
    assert summary["n_fit_images"] + summary["n_audit_images"] == 489
    assert summary["split"]["pneumonia_patient_token_overlap"] == 0
    assert summary["model_state_unchanged"] is True
    assert summary["official_test_manifest_contents_or_images_read"] is False
    assert summary["mc_dropout_passes"] == 50
    assert summary["temperature"] > 0
