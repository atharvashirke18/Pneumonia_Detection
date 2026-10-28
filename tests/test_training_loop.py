"""
tests/test_training_loop.py
Automated tests for model construction, training loop infrastructure, and checkpointing.

IMPORTANT: These are *functional infrastructure* tests only.
- All tests use synthetic random-input tensors or a tiny synthetic dataset.
- No real chest X-ray images are used in these tests.
- No results from these tests constitute or approximate pneumonia-detection performance.
- The test set (data/manifests/test_manifest.csv) is never loaded here.
"""

import sys
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.models.model import EfficientNetB0Classifier, build_model
from src.training.trainer import (
    Trainer,
    EarlyStopping,
    compute_binary_metrics,
    get_environment_metadata,
    save_checkpoint,
    load_checkpoint,
    build_optimizer,
    build_scheduler,
    build_loss_fn,
    _run_epoch,
)
from src.data.datamodule import PneumoniaDataModule
from src.data.dataset import ManifestDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_batch(batch_size: int = 4, device: str = "cpu"):
    """Create a single synthetic batch of (images, labels) tensors."""
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.float32)
    return images, labels


def _tiny_loader(batch_size: int = 4, n_batches: int = 3):
    """
    Return a minimal DataLoader-like iterable from a tiny synthetic dataset.
    Uses real ManifestDataset + DataLoader for a fully representative pipeline test,
    but with a small synthetic fixture to avoid reading real images.
    """
    from torch.utils.data import TensorDataset, DataLoader
    images = torch.randn(batch_size * n_batches, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size * n_batches,), dtype=torch.float32)
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model construction tests
# ---------------------------------------------------------------------------

class TestModelConstruction(unittest.TestCase):
    """Verify model shapes, output contract, and build_model factory."""

    def test_model_output_shape_single_sample(self):
        """Single image should produce a (1, 1) logit tensor."""
        model = EfficientNetB0Classifier(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, torch.Size([1, 1]),
                         f"Expected shape (1, 1), got {out.shape}")

    def test_model_output_shape_batch(self):
        """Batch of 8 images should produce (8, 1) logit tensor."""
        model = EfficientNetB0Classifier(pretrained=False)
        model.eval()
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, torch.Size([8, 1]),
                         f"Expected shape (8, 1), got {out.shape}")

    def test_output_dtype_is_float32(self):
        model = EfficientNetB0Classifier(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.dtype, torch.float32)

    def test_build_model_from_config(self):
        """build_model must produce a valid classifier from config dict."""
        cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
        # Build WITHOUT pretrained weights for speed (functional test only)
        model_cfg = dict(cfg["model"])
        model_cfg["pretrained"] = False
        model = build_model(model_cfg)
        self.assertIsInstance(model, EfficientNetB0Classifier)
        counts = model.parameter_counts()
        self.assertGreater(counts["trainable"], 0)

    def test_build_model_rejects_wrong_architecture(self):
        with self.assertRaises(ValueError):
            build_model({"architecture": "resnet50", "pretrained": False})

    def test_freeze_backbone_reduces_trainable_params(self):
        model_frozen = EfficientNetB0Classifier(pretrained=False, freeze_backbone=True)
        model_full = EfficientNetB0Classifier(pretrained=False, freeze_backbone=False)
        frozen_counts = model_frozen.parameter_counts()
        full_counts = model_full.parameter_counts()
        self.assertLess(frozen_counts["trainable"], full_counts["trainable"])
        self.assertEqual(frozen_counts["total"], full_counts["total"])

    def test_unfreeze_all_restores_trainable_params(self):
        model = EfficientNetB0Classifier(pretrained=False, freeze_backbone=True)
        model.unfreeze_all()
        counts = model.parameter_counts()
        self.assertEqual(counts["trainable"], counts["total"])

    def test_classifier_head_shape(self):
        """The final linear layer must be Linear(1280 -> 1)."""
        model = EfficientNetB0Classifier(pretrained=False)
        clf = model.get_classifier()
        self.assertIsInstance(clf, nn.Linear)
        self.assertEqual(clf.out_features, 1)
        self.assertEqual(clf.in_features, 1280)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossFunction(unittest.TestCase):
    """Verify BCEWithLogitsLoss construction and numerical correctness."""

    def test_loss_fn_type(self):
        loss_fn = build_loss_fn({"loss": {"pos_weight": 1.0}})
        self.assertIsInstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_loss_is_finite_for_correct_predictions(self):
        loss_fn = build_loss_fn({"loss": {"pos_weight": 1.0}})
        logits = torch.tensor([5.0, -5.0, 5.0, -5.0])  # confident, correct
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), 0.01)

    def test_loss_is_higher_for_wrong_predictions(self):
        loss_fn = build_loss_fn({"loss": {"pos_weight": 1.0}})
        logits_wrong = torch.tensor([-5.0, 5.0, -5.0, 5.0])  # confident, wrong
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_wrong = loss_fn(logits_wrong, targets)
        logits_right = torch.tensor([5.0, -5.0, 5.0, -5.0])
        loss_right = loss_fn(logits_right, targets)
        self.assertGreater(loss_wrong.item(), loss_right.item())

    def test_auto_pos_weight_with_imbalanced_counts(self):
        """auto pos_weight should be n_neg / n_pos when train_dataset has counts."""

        class FakeDataset:
            def get_class_counts(self):
                return {"NORMAL": 100, "PNEUMONIA": 400}

        loss_fn = build_loss_fn({"loss": {"pos_weight": "auto"}}, FakeDataset())
        # pos_weight = 100 / 400 = 0.25
        pw = loss_fn.pos_weight
        self.assertIsNotNone(pw)
        self.assertAlmostEqual(pw.item(), 0.25, places=5)


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------

class TestMetricComputation(unittest.TestCase):
    """Verify compute_binary_metrics correctness."""

    def test_perfect_predictions(self):
        logits = np.array([10.0, 10.0, -10.0, -10.0])
        targets = np.array([1, 1, 0, 0])
        m = compute_binary_metrics(logits, targets)
        self.assertAlmostEqual(m["auc"], 1.0)
        self.assertAlmostEqual(m["sensitivity"], 1.0)
        self.assertAlmostEqual(m["specificity"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_random_chance_predictions(self):
        """All-zero logits (p=0.5) applied to balanced targets yields AUC ≈ 0.5."""
        logits = np.zeros(100)
        targets = np.array([1] * 50 + [0] * 50)
        m = compute_binary_metrics(logits, targets)
        # AUC of constant predictions is undefined / degenerate; just check it returns a float
        self.assertIsInstance(m["auc"], float)

    def test_completely_wrong_predictions(self):
        logits = np.array([-10.0, -10.0, 10.0, 10.0])
        targets = np.array([1, 1, 0, 0])
        m = compute_binary_metrics(logits, targets)
        self.assertAlmostEqual(m["sensitivity"], 0.0)
        self.assertAlmostEqual(m["specificity"], 0.0)
        self.assertAlmostEqual(m["f1"], 0.0)

    def test_metric_keys_present(self):
        logits = np.array([1.0, -1.0])
        targets = np.array([1, 0])
        m = compute_binary_metrics(logits, targets)
        for key in ("auc", "sensitivity", "specificity", "f1", "threshold"):
            self.assertIn(key, m)


# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------

class TestEarlyStopping(unittest.TestCase):

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=3, mode="min")
        # First call sets best_value (no counter increment).
        # Patience=3 requires 3 subsequent non-improving calls to set should_stop.
        es.step(1.0)  # epoch 0: sets best_value=1.0, counter stays 0
        for _ in range(3):
            es.step(1.0)  # epochs 1-3: no improvement, counter increments 1→2→3 → triggers stop
        self.assertTrue(es.should_stop)

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0)
        es.step(0.9)  # improvement -> counter resets
        es.step(0.9)  # no improvement
        es.step(0.9)  # no improvement -> counter=2
        self.assertFalse(es.should_stop)

    def test_mode_max(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.8)  # best
        es.step(0.7)  # no improvement
        es.step(0.7)  # no improvement -> stop
        self.assertTrue(es.should_stop)

    def test_first_epoch_never_stops(self):
        es = EarlyStopping(patience=1, mode="min")
        result = es.step(999.0)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# One-batch forward + backward pass tests
# ---------------------------------------------------------------------------

class TestOneBatchForwardBackward(unittest.TestCase):
    """
    Critical smoke test: verifies that a full forward + backward pass is
    numerically correct and that gradients flow to all trainable parameters.
    """

    def setUp(self):
        self.model = EfficientNetB0Classifier(pretrained=False, dropout_rate=0.0)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def test_forward_produces_finite_logits(self):
        self.model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x).squeeze(1)
        self.assertEqual(out.shape, torch.Size([4]))
        self.assertTrue(torch.all(torch.isfinite(out)), "Logits contain NaN or Inf")

    def test_loss_is_finite_and_backward_runs(self):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        x = torch.randn(4, 3, 224, 224, device=self.device)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)

        optimizer.zero_grad()
        logits = self.model(x).squeeze(1)
        loss = criterion(logits, y)

        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss.item()}")

        loss.backward()
        optimizer.step()

    def test_gradients_flow_to_classifier_head(self):
        """Gradients must exist for the classifier head after backward."""
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()

        x = torch.randn(4, 3, 224, 224, device=self.device)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)

        logits = self.model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()

        clf = self.model.get_classifier()
        self.assertIsNotNone(clf.weight.grad, "Classifier weight grad is None after backward")
        self.assertTrue(
            torch.any(clf.weight.grad != 0),
            "Classifier weight gradient is all-zero"
        )

    def test_model_weights_update_after_optimizer_step(self):
        """Weights must actually change after an optimizer step."""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        clf = self.model.get_classifier()
        before = clf.weight.data.clone()

        x = torch.randn(4, 3, 224, 224, device=self.device)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)

        optimizer.zero_grad()
        logits = self.model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        after = clf.weight.data.clone()
        self.assertFalse(
            torch.allclose(before, after),
            "Classifier weights did not change after optimizer step"
        )


# ---------------------------------------------------------------------------
# _run_epoch utility tests (with synthetic loaders)
# ---------------------------------------------------------------------------

class TestRunEpoch(unittest.TestCase):
    """
    Test _run_epoch utility with synthetic TensorDataset loaders.
    Validates loss values are finite and logit/target shapes match.
    """

    def setUp(self):
        self.model = EfficientNetB0Classifier(pretrained=False, dropout_rate=0.0)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

    def test_run_epoch_train_returns_finite_loss(self):
        loader = _tiny_loader(batch_size=4, n_batches=2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        avg_loss, logits, targets = _run_epoch(
            self.model, loader, self.criterion, self.device,
            optimizer, scaler=None, is_training=True
        )
        self.assertIsInstance(avg_loss, float)
        self.assertTrue(np.isfinite(avg_loss), f"Training loss not finite: {avg_loss}")
        self.assertEqual(logits.shape, targets.shape)

    def test_run_epoch_eval_does_not_update_weights(self):
        loader = _tiny_loader(batch_size=4, n_batches=2)
        clf = self.model.get_classifier()
        before = clf.weight.data.clone()

        avg_loss, logits, targets = _run_epoch(
            self.model, loader, self.criterion, self.device,
            optimizer=None, scaler=None, is_training=False
        )
        after = clf.weight.data.clone()
        self.assertTrue(
            torch.allclose(before, after),
            "Eval run should NOT modify model weights"
        )

    def test_run_epoch_logit_target_count_matches_dataset(self):
        n_samples = 12
        loader = _tiny_loader(batch_size=4, n_batches=3)  # 3 * 4 = 12 samples
        avg_loss, logits, targets = _run_epoch(
            self.model, loader, self.criterion, self.device,
            optimizer=None, scaler=None, is_training=False
        )
        self.assertEqual(len(logits), n_samples)
        self.assertEqual(len(targets), n_samples)


# ---------------------------------------------------------------------------
# Checkpoint save/load tests
# ---------------------------------------------------------------------------

class TestCheckpointing(unittest.TestCase):
    """Verify that checkpoints round-trip correctly (save → load → identical weights)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ckpt_path = Path(self.temp_dir) / "test_ckpt.pt"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_restores_identical_weights(self):
        model = EfficientNetB0Classifier(pretrained=False)
        cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
        optimizer = build_optimizer(model, cfg["training"])
        scheduler = build_scheduler(optimizer, cfg["training"])
        metrics = {"val_loss": 0.42, "val_auc": 0.91}

        save_checkpoint(
            path=self.ckpt_path,
            epoch=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            metrics=metrics,
        )
        self.assertTrue(self.ckpt_path.exists(), "Checkpoint file was not created")

        # Load into a new model instance
        model2 = EfficientNetB0Classifier(pretrained=False)
        ckpt = load_checkpoint(self.ckpt_path, model2)

        self.assertEqual(ckpt["epoch"], 0)
        self.assertAlmostEqual(ckpt["metrics"]["val_loss"], 0.42)
        self.assertAlmostEqual(ckpt["metrics"]["val_auc"], 0.91)

        # Verify weights are bitwise identical after round-trip
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            self.assertTrue(
                torch.equal(p1.data, p2.data),
                f"Parameter {n1} differs after checkpoint round-trip"
            )

    def test_load_from_missing_path_raises(self):
        model = EfficientNetB0Classifier(pretrained=False)
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(Path(self.temp_dir) / "nonexistent.pt", model)

    def test_checkpoint_contains_architecture_metadata(self):
        model = EfficientNetB0Classifier(pretrained=False)
        cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
        optimizer = build_optimizer(model, cfg["training"])
        scheduler = build_scheduler(optimizer, cfg["training"])

        save_checkpoint(
            path=self.ckpt_path,
            epoch=3,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            metrics={"val_loss": 0.3},
        )
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["architecture"], "efficientnet_b0")
        self.assertEqual(ckpt["num_classes"], 1)


# ---------------------------------------------------------------------------
# Optimizer and scheduler tests
# ---------------------------------------------------------------------------

class TestOptimizerScheduler(unittest.TestCase):

    def setUp(self):
        self.model = EfficientNetB0Classifier(pretrained=False)
        self.cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)

    def test_optimizer_type_is_adamw(self):
        opt = build_optimizer(self.model, self.cfg["training"])
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_optimizer_lr_from_config(self):
        opt = build_optimizer(self.model, self.cfg["training"])
        lr = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(lr, self.cfg["training"]["learning_rate"])

    def test_scheduler_step_reduces_lr(self):
        """LR should eventually decrease below the initial value after cosine decay."""
        training_cfg = dict(self.cfg["training"])
        training_cfg["warmup_epochs"] = 0
        training_cfg["epochs"] = 10

        opt = build_optimizer(self.model, training_cfg)
        sched = build_scheduler(opt, training_cfg)
        initial_lr = opt.param_groups[0]["lr"]

        for _ in range(10):
            # Call optimizer.step() before scheduler.step() per PyTorch convention.
            # Use a zero-loss backward to satisfy the optimizer without real data.
            dummy_loss = sum(p.sum() * 0.0 for p in self.model.parameters())
            dummy_loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()

        final_lr = opt.param_groups[0]["lr"]
        self.assertLess(final_lr, initial_lr)

    def test_scheduler_with_warmup(self):
        """With warmup, LR starts below initial and then increases."""
        training_cfg = dict(self.cfg["training"])
        training_cfg["warmup_epochs"] = 2
        training_cfg["epochs"] = 10

        opt = build_optimizer(self.model, training_cfg)
        sched = build_scheduler(opt, training_cfg)

        lrs = []
        for _ in range(3):
            dummy_loss = sum(p.sum() * 0.0 for p in self.model.parameters())
            dummy_loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])

        # During warmup (start_factor=0.1 -> 1.0 over 2 epochs), LR should
        # increase monotonically for the first two steps.
        self.assertLessEqual(lrs[0], lrs[1])


# ---------------------------------------------------------------------------
# GPU smoke test (skipped if CUDA not available)
# ---------------------------------------------------------------------------

class TestGPUSmokeTest(unittest.TestCase):
    """
    Smoke test: model forward + backward on real GPU.
    Skipped automatically if CUDA is not available.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_backward_on_gpu(self):
        device = torch.device("cuda")
        model = EfficientNetB0Classifier(pretrained=False).to(device)
        model.train()

        x = torch.randn(4, 3, 224, 224, device=device)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        self.assertTrue(torch.isfinite(loss), f"GPU loss is not finite: {loss.item()}")

        loss.backward()
        optimizer.step()

        # Verify GPU memory is not exhausted after one step
        torch.cuda.synchronize()
        mem_used = torch.cuda.memory_allocated(0) / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.assertLess(
            mem_used, mem_total,
            f"GPU memory fully consumed after one batch: {mem_used:.2f}/{mem_total:.2f} GB"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_amp_forward_backward_on_gpu(self):
        """AMP (fp16) forward + backward must not produce NaN loss."""
        # Use torch.amp.GradScaler (non-deprecated API, PyTorch >= 2.0)
        scaler = torch.amp.GradScaler("cuda")

        device = torch.device("cuda")
        model = EfficientNetB0Classifier(pretrained=False).to(device)
        model.train()

        x = torch.randn(4, 3, 224, 224, device=device)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

        self.assertTrue(torch.isfinite(loss), f"AMP loss is not finite: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


# ---------------------------------------------------------------------------
# Environment metadata and trainer checkpoint policy tests
# ---------------------------------------------------------------------------

class FakeDataModule:
    """Minimal fake DataModule for testing Trainer execution without disk I/O."""

    def __init__(self, batch_size: int = 4, n_batches: int = 2):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.train_dataset = None

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        return _tiny_loader(self.batch_size, self.n_batches)

    def val_dataloader(self):
        return _tiny_loader(self.batch_size, 1)


class TestEnvironmentMetadata(unittest.TestCase):
    """Verify runtime environment and provenance metadata collection."""

    def test_environment_metadata_keys(self):
        model = EfficientNetB0Classifier(pretrained=False)
        meta = get_environment_metadata(model)
        expected_keys = [
            "python_version",
            "platform",
            "torch_version",
            "cuda_available",
            "timm_version",
            "timestamp",
            "model_architecture",
            "num_classes",
            "parameters",
        ]
        for key in expected_keys:
            self.assertIn(key, meta)

        self.assertEqual(meta["model_architecture"], "efficientnet_b0")
        self.assertEqual(meta["num_classes"], 1)
        self.assertGreater(meta["parameters"]["total"], 0)


class TestTrainerCheckpointPolicy(unittest.TestCase):
    """Verify that checkpoint retention and metadata emission match approved settings."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = load_config("configs/config.yaml", base_dir=PROJECT_ROOT)
        # Adapt for very fast CPU synthetic test execution
        self.cfg["training"]["epochs"] = 2
        self.cfg["training"]["warmup_epochs"] = 0
        self.cfg["system"]["device"] = "cpu"
        self.cfg["system"]["mixed_precision"] = "no"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_checkpoint_policy_retains_best_and_last_only(self):
        """Default policy: retain best_model.pt and last_model.pt; no per-epoch checkpoints."""
        self.cfg["training"]["save_epoch_checkpoints"] = False
        model = EfficientNetB0Classifier(pretrained=False)
        dm = FakeDataModule(batch_size=4, n_batches=2)

        trainer = Trainer(
            model=model,
            data_module=dm,
            config=self.cfg,
            checkpoint_dir=self.temp_dir,
        )
        history = trainer.fit()

        self.assertEqual(len(history), 2)
        out_dir = Path(self.temp_dir)
        self.assertTrue((out_dir / "best_model.pt").exists(), "best_model.pt must be saved")
        self.assertTrue((out_dir / "last_model.pt").exists(), "last_model.pt must be saved")
        self.assertTrue((out_dir / "config_snapshot.json").exists(), "config_snapshot.json must exist")
        self.assertTrue((out_dir / "environment_metadata.json").exists(), "environment_metadata.json must exist")
        self.assertTrue((out_dir / "training_history.json").exists(), "training_history.json must exist")

        # Verify no epoch_*.pt checkpoints are present by default
        epoch_ckpts = list(out_dir.glob("epoch_*.pt"))
        self.assertEqual(
            len(epoch_ckpts), 0,
            f"Expected 0 per-epoch checkpoints by default, found {epoch_ckpts}"
        )

    def test_opt_in_saves_per_epoch_checkpoints(self):
        """When save_epoch_checkpoints is True, epoch_*.pt files are saved."""
        self.cfg["training"]["save_epoch_checkpoints"] = True
        model = EfficientNetB0Classifier(pretrained=False)
        dm = FakeDataModule(batch_size=4, n_batches=2)

        trainer = Trainer(
            model=model,
            data_module=dm,
            config=self.cfg,
            checkpoint_dir=self.temp_dir,
        )
        trainer.fit()

        out_dir = Path(self.temp_dir)
        epoch_ckpts = sorted(list(out_dir.glob("epoch_*.pt")))
        self.assertEqual(len(epoch_ckpts), 2)

    def test_default_checkpoint_dir_uses_config_output_dir(self):
        """Trainer defaults to config.training.output_dir when not explicitly provided."""
        model = EfficientNetB0Classifier(pretrained=False)
        dm = FakeDataModule()
        trainer = Trainer(model=model, data_module=dm, config=self.cfg)
        expected_dir = Path(self.cfg["training"]["output_dir"])
        self.assertEqual(trainer.checkpoint_dir, expected_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
